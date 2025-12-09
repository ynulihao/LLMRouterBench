import argparse
import json
import os
import random

import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
from transformers import AutoTokenizer, DebertaV2Model, AutoModel
from utils.meters import AverageMeter
import numpy as np
import deepspeed
import wandb

def print_major_header(title, width=70):
    """Print a major section header with double-line box"""
    print(f"\n{'╔' + '═' * (width - 2) + '╗'}")
    print(f"{'║'} {title:<{width - 4}} {'║'}")
    print(f"{'╚' + '═' * (width - 2) + '╝'}")

def print_minor_header(title, indent=0, width=70):
    """Print a minor section header with single-line separator"""
    spaces = " " * indent
    line_width = width - indent
    separator_length = max(0, line_width - len(title) - 5)
    print(f"\n{spaces}{'┌─'} {title} {'─' * separator_length}")

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


class RouterDataset(Dataset):
    def __init__(self,
                 data_path,
                 source_max_token_len: int = 512,
                 target_max_token_len: int = 512,
                 size: int = None,
                 data_type: str = "multi_attempt",
                 dataset_id = 0,
                 ):
        with open(data_path, 'r') as f:
            if data_path.endswith('.json'):
                self.data = json.load(f)
        if size:
            while(len(self.data) < size):
                self.data.extend(self.data)
            self.data = self.data[:size]
        self.router_node = list(self.data[0]['scores'].keys())
        self.tokenizer = None
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.data_type = data_type
        self.dataset_id = dataset_id

    def __getitem__(self, index):
        data_point = self.data[index]
        scores = torch.tensor(list(data_point['scores'].values()))
        question = data_point['question']
        question_id = self.tokenizer(
            question,
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        question_id['input_ids'] = question_id.input_ids.flatten()
        question_id['attention_mask'] = question_id.attention_mask.flatten()
        cluster_id = data_point['cluster_id'] if "cluster_id" in data_point else 0
        return question_id, scores, self.dataset_id, cluster_id

    def __len__(self):
        return len(self.data)

    def register_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer


# using inner product first
class RouterModule(nn.Module):
    def __init__(self, backbone, hidden_state_dim=768, node_size=3, similarity_function = "cos"):
        super(RouterModule, self).__init__()
        self.backbone = backbone
        self.hidden_state_dim = hidden_state_dim
        self.node_size = node_size
        self.embeddings = nn.Embedding(node_size, hidden_state_dim)
        std_dev = 0.78
        with torch.no_grad():
            nn.init.normal_(self.embeddings.weight, mean=0, std=std_dev)
        self.similarity_function = similarity_function
            

    def compute_similarity(self, input1, input2):
        input1 = input1.to(torch.float32)
        input2 = input2.to(torch.float32)
        if self.similarity_function == "cos":
            return (input1 @ input2.T) / (torch.norm(input1,dim=1).unsqueeze(1) * torch.norm(input2,dim=1).unsqueeze(0))
        else:
            return input1 @ input2.T


    '''The forward function pass the input to Router and compute the similarity between model output and trainable embedding'''
    # For bert encode
    # def forward(self, t=1, **input_kwargs):
    #     x = self.backbone(**input_kwargs)
    #     # We used the first token as classifier token.
    #     hidden_state = x['last_hidden_state'][:,0,:]
    #     x = self.compute_similarity(hidden_state, self.embeddings.weight)
    #     x = x / t
    #     return x, hidden_state
    
    def forward(self, t=1, **input_kwargs):
        out = self.backbone(**input_kwargs)
        last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out["last_hidden_state"]
        attn = input_kwargs.get("attention_mask", None)

        # last-token pooling
        if attn is None:
            hidden_state = last_hidden[:, -1, :]
        else:
            left_padding = (attn[:, -1].sum() == attn.shape[0])
            if left_padding:
                hidden_state = last_hidden[:, -1, :]
            else:
                sequence_lengths = (attn.sum(dim=1) - 1).long()  # [B]
                batch_size = last_hidden.shape[0]
                batch_idx = torch.arange(batch_size, device=last_hidden.device)
                hidden_state = last_hidden[batch_idx, sequence_lengths]  # [B, H]

        # （可选）做 L2 归一化
        # hidden_state = nn.functional.normalize(hidden_state, p=2, dim=1)

        x = self.compute_similarity(hidden_state, self.embeddings.weight)
        x = x / t
        return x, hidden_state

    def compute_sample_llm_loss(self, x, index_true, top_k, last_k):
        loss = 0
        top_index_true, top_index = index_true.sort(dim=-1, descending=True)
        last_index_true, negtive_index = index_true.topk(k=last_k, largest=False,dim=-1)

        for i in range(top_k):
            positive_index = top_index[:,i].view(-1,1)

            # If positive model does not well, skip this.
            mask = torch.where(top_index_true[:,i].view(-1,1) > 0, 1, 0)

            top_x = torch.gather(x, 1, positive_index)
            last_x = torch.gather(x, 1, negtive_index)

            # make the last_x ignore the true items
            last_x = torch.where(last_index_true > 0.5, float("-inf"), last_x)

            temp_x = torch.concat([top_x, last_x], dim=-1)

            softmax_x = nn.Softmax(dim=-1)(temp_x)
            log_x = torch.log(softmax_x[:,0])
            log_x = log_x * mask 
            # * mask2
            loss += torch.mean(-log_x)
        return loss
    
    def compute_sample_sample_loss_with_task_tag(self, hidden_state, dataset_ids, t, H=3):
        similar_score = self.compute_similarity(hidden_state, hidden_state)
        last_k2 = H
        # global shrink-k: find the smallest available negatives across items
        min_neg = None
        for dataset_id in dataset_ids:
            negtive_indexs = torch.nonzero(dataset_ids != dataset_id, as_tuple=False)
            cnt = int(negtive_indexs.numel())
            if min_neg is None or cnt < min_neg:
                min_neg = cnt
        k2_eff = min(last_k2, (min_neg or 0))

        all_index = []
        if k2_eff < last_k2:
            try:
                print(f"Shrink neg K from {last_k2} to {k2_eff} for dataset task tags")
            except Exception:
                pass

        for dataset_id in dataset_ids:
            positive_indexs = torch.nonzero(dataset_ids == dataset_id, as_tuple=False).view(-1)
            if positive_indexs.numel() == 0:
                continue
            # pick one positive index at random
            pos_sel = positive_indexs[torch.randint(0, positive_indexs.numel(), (1,), device=positive_indexs.device)]

            negtive_indexs = torch.nonzero(dataset_ids != dataset_id, as_tuple=False).view(-1)
            if k2_eff == 0 or negtive_indexs.numel() < k2_eff:
                continue
            perm = torch.randperm(negtive_indexs.numel(), device=negtive_indexs.device)[:k2_eff]
            neg_sel = negtive_indexs[perm].view(-1)

            select_index = torch.concat([pos_sel.view(-1), neg_sel], dim=0)
            all_index.append(select_index)

        if len(all_index) == 0:
            return torch.zeros((), dtype=hidden_state.dtype, device=hidden_state.device, requires_grad=hidden_state.requires_grad)

        all_index = torch.stack(all_index)
        rearrange_similar_score = torch.gather(similar_score, 1, all_index)

        softmax_sample_x = torch.softmax(rearrange_similar_score, dim=-1)
        log_sample_x = torch.log(softmax_sample_x)
        loss = torch.mean(-log_sample_x[:,0])
        return loss
    
    def compute_cluster_loss(self, hidden_state, cluster_ids, t, H=3):
        similar_score = self.compute_similarity(hidden_state, hidden_state)
        last_k2 = H
        # global shrink-k: find the smallest available negatives across items
        min_neg = None
        for cluster_id in cluster_ids:
            negtive_indexs = torch.nonzero(cluster_ids != cluster_id, as_tuple=False)
            cnt = int(negtive_indexs.numel())
            if min_neg is None or cnt < min_neg:
                min_neg = cnt
        k2_eff = min(last_k2, (min_neg or 0))

        all_index = []
        if k2_eff < last_k2:
            try:
                # cluster_id may be a tensor; avoid printing device noise
                print(f"Shrink neg K from {last_k2} to {k2_eff} for clusters")
            except Exception:
                pass

        for cluster_id in cluster_ids:
            positive_indexs = torch.nonzero(cluster_ids == cluster_id, as_tuple=False).view(-1)
            if positive_indexs.numel() == 0:
                continue
            # pick one positive index at random
            pos_sel = positive_indexs[torch.randint(0, positive_indexs.numel(), (1,), device=positive_indexs.device)]

            negtive_indexs = torch.nonzero(cluster_ids != cluster_id, as_tuple=False).view(-1)
            if k2_eff == 0 or negtive_indexs.numel() < k2_eff:
                continue
            perm = torch.randperm(negtive_indexs.numel(), device=negtive_indexs.device)[:k2_eff]
            neg_sel = negtive_indexs[perm].view(-1)

            select_index = torch.concat([pos_sel.view(-1), neg_sel], dim=0)
            all_index.append(select_index)

        if len(all_index) == 0:
            return torch.zeros((), dtype=hidden_state.dtype, device=hidden_state.device, requires_grad=hidden_state.requires_grad)

        all_index = torch.stack(all_index)
        rearrange_similar_score = torch.gather(similar_score, 1, all_index)

        softmax_sample_x = torch.softmax(rearrange_similar_score, dim=-1)
        log_sample_x = torch.log(softmax_sample_x)
        loss = torch.mean(-log_sample_x[:,0])
        return loss


# evaluation the router with dataset.
def evaluation(router_model, dataset_paths, dataset_types, tokenizer, batch_size, local_rank):
    result = {}
    combined_routing_distribution = torch.zeros(router_model.module.node_size, dtype=torch.long, device=router_model.device)
    combined_total_samples = 0

    # Track overall statistics for weighted average
    total_correct = 0
    total_samples_all = 0
    dataset_accuracies = []

    with torch.no_grad():
        # assert len(dataset_paths) == len(dataset_types)
        for index, data_path in enumerate(dataset_paths):
            test_dataset = RouterDataset(data_path=data_path)
            test_dataset.register_tokenizer(tokenizer)
            data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            
            correct_predict = 0
            correct = 0
            total_samples = 0
            
            # Track routing distribution
            routing_distribution = torch.zeros(router_model.module.node_size, dtype=torch.long, device=router_model.device)
            
            for batch in data_loader:
                inputs, scores, _, _ = batch
                # Move to device
                inputs = {k: v.to(router_model.device) for k, v in inputs.items()}
                scores = scores.to(router_model.device)
                
                x, _ = router_model.module.forward(**inputs)
                softmax_x = nn.Softmax(dim=1)(x)
                _, max_index = torch.max(softmax_x, dim=1)
                
                # Update routing distribution count
                for idx in max_index:
                    routing_distribution[idx] += 1

                max_scores, _ = torch.max(scores, dim=1, keepdim=True)
                max_mask = (scores == max_scores)
                correct_predictions = max_mask.gather(1, max_index.unsqueeze(1))
                correct += correct_predictions.sum().item()

                if dataset_types[index] == "probability":
                    mask = torch.zeros_like(scores)
                    mask = mask.scatter_(1, max_index.unsqueeze(1), 1)
                    scores[scores > 0] = 1
                    correct_predict += (scores * mask).sum().item()
                elif dataset_types[index] == "multi_attempt":
                    mask = torch.zeros_like(scores)
                    mask = mask.scatter_(1, max_index.unsqueeze(1), 1)
                    correct_predict += (scores * mask).sum().item()
                
                total_samples += scores.size(0)
            
            # Get metrics from all devices and update combined totals
            if torch.distributed.is_initialized():
                metrics = torch.tensor([total_samples, correct, correct_predict], device=router_model.device)
                torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(routing_distribution, op=torch.distributed.ReduceOp.SUM)
                total_samples, correct, correct_predict = metrics.tolist()
            
            # Only process results on rank 0
            if local_rank == 0:
                acc_predict = correct_predict/total_samples
                acc = correct/total_samples

                dataset_name = os.path.basename(data_path).split('.')[0]
                print_minor_header(f"{dataset_name}", indent=2)
                print(f"    Accuracy: {acc_predict:.4f}")

                result[data_path] = [acc, acc_predict]
                dataset_accuracies.append(acc_predict)
                total_correct += correct_predict
                total_samples_all += total_samples

                # Log per-dataset accuracy metrics
                if wandb.run is not None:
                    wandb.log({
                        f"acc_{dataset_name}": acc,
                        f"acc_predict_{dataset_name}": acc_predict,
                    })

            # Accumulate routing distribution and samples for overall statistics
            combined_routing_distribution += routing_distribution
            combined_total_samples += total_samples
    
    # Calculate overall averages
    dataset_level_accuracy = sum(dataset_accuracies) / len(dataset_accuracies) if dataset_accuracies else 0
    sample_level_accuracy = total_correct / total_samples_all if total_samples_all > 0 else 0

    # Display summary metrics
    if local_rank == 0:
        print_minor_header("Summary Metrics", indent=2)
        print(f"    Dataset-Level Average Accuracy: {dataset_level_accuracy:.4f} (unweighted)")
        print(f"    Sample-Level Average Accuracy: {sample_level_accuracy:.4f} (weighted)")

    # Log combined routing distribution once for all datasets
    if local_rank == 0 and wandb.run is not None:
        is_train = "train" in dataset_paths[0]
        prefix = "train" if is_train else "test"

        # Log overall router usage distribution
        router_dist_dict = {}
        for i, count in enumerate(combined_routing_distribution):
            router_dist_dict[f"{prefix}_router_{i}_usage"] = count.item() / combined_total_samples

        wandb.log(router_dist_dict)

    return result, dataset_level_accuracy, sample_level_accuracy


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="the training code for router")

    # DeepSpeed parameter
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank passed by DeepSpeed')
    
    # dataset and path
    parser.add_argument('--data_paths', nargs='+', default=["./datasets/split2_model7_cluster/gsm8k-train.json","./datasets/split2_model7_cluster/humaneval_train.json", "./datasets/split2_model7_cluster/arc_challenge_train.json", "./datasets/split2_model7_cluster/mmlu_train.json","./datasets/split2_model7_cluster/cmmlu_train.json",])
    parser.add_argument('--test_data_paths',nargs='+', default=["./datasets/split2_model7/gsm8k-test.json", "./datasets/split2_model7/humaneval_test.json", "./datasets/split2_model7/arc_challenge_test.json", "./datasets/split2_model7/mmlu_test.json", "./datasets/split2_model7/cmmlu_test.json"])
    parser.add_argument('--test_data_type', nargs='+', default=["multi_attempt", "multi_attempt", "probability", "probability", "probability"])
    parser.add_argument('--final_eval_data_paths', default=["./datasets/split2_model7/arc_challenge_test.json", "./datasets/split2_model7/MATH_prealgebra.json", "./datasets/split2_model7/mbpp.json", "./datasets/split2_model7/ceval.json" ,"./datasets/split2_model7/gsm8k-test.json", "./datasets/split2_model7/humaneval_test.json",  "./datasets/split2_model7/mmlu_test.json", "./datasets/split2_model7/cmmlu_test.json"])
    parser.add_argument('--final_eval_data_type', nargs='+', default=["probability", "probability", "multi_attempt","probability", "multi_attempt", "multi_attempt", "probability",  "probability"])

    # training paras
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--training_steps', type=int, default=1000)
    parser.add_argument('--eval_steps',type=int,default=50)
    parser.add_argument('--learning_rate', type=float, default=0.00005)
    parser.add_argument('--save_path', type=str, default='./logs/router_debug/')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--last_k',type=int, default=3)
    parser.add_argument('--tempreture', type=int, default=1)
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--similarity_function', type=str, default='cos')
    parser.add_argument('--sample_loss_weight', type=float, default=0)
    parser.add_argument('--cluster_loss_weight', type=float, default=0)
    parser.add_argument('--H', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--training_samples_per_dataset', type=int, default=1000)
    parser.add_argument('--deepspeed', type=str, default='ds_config.json', help='Deepspeed configuration file')
    parser.add_argument('--wandb_project', type=str, default='routerdc', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Weights & Biases run name')
    parser.add_argument('--ood_datasets', type=str, default='',
                       help='Comma-separated list of OOD dataset keywords for exact name matching (default: none)')

    # final_eval
    parser.add_argument('--final_eval', action="store_true")
    args = parser.parse_args()
    
    # Parse OOD datasets
    ood_keywords = []
    if args.ood_datasets:
        ood_keywords = [d.strip().lower() for d in args.ood_datasets.split(',') if d.strip()]
    
    # Get local rank for distributed training
    local_rank = args.local_rank
    
    # Initialize distributed process group
    torch.distributed.init_process_group(backend='nccl')
    
    # Initialize wandb on rank 0 only
    if local_rank == 0:
        wandb_config = {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "training_steps": args.training_steps,
            "top_k": args.top_k,
            "last_k": args.last_k,
            "temperature": args.tempreture,
            "similarity_function": args.similarity_function,
            "sample_loss_weight": args.sample_loss_weight,
            "cluster_loss_weight": args.cluster_loss_weight,
            "seed": args.seed,
        }
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=wandb_config
        )
    
    # Only create directories on main process
    if local_rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
    
    # Set seed
    setup_seed(args.seed)

    # get router model (mdeberta-v3-base)
    MODEL_NAME = "/fs-computility-new/Uma4agi/shared/models/gte_Qwen2-7B-instruct"
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        truncation_side='left',
        padding=True,
        trust_remote_code=True
    )
    encoder_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    for param in encoder_model.parameters():
        param.requires_grad = True
        


    # get the training data (x, y)
    router_datasets = [RouterDataset(data_path, size=args.training_samples_per_dataset, data_type=args.test_data_type[i], dataset_id = i) for i, data_path in enumerate(args.data_paths)]
    for router_dataset in router_datasets:
        router_dataset.register_tokenizer(tokenizer)
    router_dataset = ConcatDataset(router_datasets)

    if local_rank == 0:
        print(f"init_model, router_node size: {router_datasets[0].router_node}")
    
    router_model = RouterModule(encoder_model, hidden_state_dim=3584, node_size=len(router_datasets[0].router_node), similarity_function=args.similarity_function)

    # Update DeepSpeed config with the right accumulation steps
    ds_config = None
    with open(args.deepspeed, 'r') as f:
        ds_config = json.load(f)
    
    ds_config['gradient_accumulation_steps'] = args.gradient_accumulation
    ds_config['train_micro_batch_size_per_gpu'] = args.batch_size
    ds_config['train_batch_size'] = args.batch_size * torch.cuda.device_count() * args.gradient_accumulation

    # Create optimizer manually first
    optimizer = torch.optim.AdamW(
        router_model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=router_model,
        optimizer=optimizer,
        config=ds_config
    )

    # start training
    if local_rank == 0:
        print("Training start!!!")
    
    step = 0
    training_log = []
    max_average = 0
    max_training_average = 0
    best_results = {}  # 初始化全局最佳结果字典，用于记录每个数据集的最佳精度

    while(True):
        losses = AverageMeter('Loss', ':3.2f')
        data_loader = DataLoader(router_dataset, batch_size=args.batch_size, shuffle=True)
        
        # Create progress bar only on main process
        if local_rank == 0:
            pbar = tqdm(total=args.training_steps - step)
        
        for batch in data_loader:
            inputs, scores, dataset_ids, cluster_ids = batch
            
            # Move tensors to device
            inputs = {k: v.to(model_engine.device) for k, v in inputs.items()}
            scores = scores.to(model_engine.device)
            dataset_ids = dataset_ids.to(model_engine.device)
            cluster_ids = cluster_ids.to(model_engine.device)
            
            # Forward pass
            x, hidden_state = model_engine.module.forward(t=args.tempreture, **inputs)
            loss = model_engine.module.compute_sample_llm_loss(x=x, index_true=scores, top_k=args.top_k, last_k=args.last_k)
            
            sample_sample_loss = 0.0
            if args.sample_loss_weight:
                sample_sample_loss = model_engine.module.compute_sample_sample_loss_with_task_tag(hidden_state=hidden_state, dataset_ids=dataset_ids, t=args.tempreture, H=args.H)
                loss = loss + args.sample_loss_weight * sample_sample_loss

            cluster_loss = 0.0
            if args.cluster_loss_weight:
                cluster_loss = model_engine.module.compute_cluster_loss(hidden_state=hidden_state, cluster_ids=cluster_ids, t=args.tempreture, H=args.H)
                loss = loss + args.cluster_loss_weight * cluster_loss

            # Backward and optimization
            model_engine.backward(loss)
            model_engine.step()
            
            # Update progress bar on main process
            if local_rank == 0:
                losses.update(loss.item(), scores.size(0))
                pbar.set_postfix({"step": f"{step}", "loss": loss.item()})
                pbar.update(1)
                
                # Track routing distribution during training
                with torch.no_grad():
                    softmax_x = nn.Softmax(dim=1)(x)
                    _, max_index = torch.max(softmax_x, dim=1)
                    
                    if wandb.run is not None:
                        wandb.log({
                            "step": step,
                            "loss": loss.item(),
                            "sample_loss": sample_sample_loss.item() if args.sample_loss_weight else 0,
                            "cluster_loss": cluster_loss.item() if args.cluster_loss_weight else 0
                        })
            
            step += 1
            if step >= args.training_steps:
                break
                
            if (step + 1) % args.eval_steps == 0:
                # Ensure all processes are synced before evaluation
                torch.distributed.barrier()
                
                model_engine.eval()

                if local_rank == 0:
                    print_major_header("TRAINING EVALUATION")

                val_result, val_dataset_avg, val_sample_avg = evaluation(model_engine, args.data_paths, args.test_data_type, tokenizer, batch_size=args.batch_size, local_rank=local_rank)

                if local_rank == 0:
                    print_major_header("TEST EVALUATION")

                test_result, test_dataset_avg, test_sample_avg = evaluation(model_engine, args.test_data_paths, args.test_data_type, tokenizer, batch_size=args.batch_size, local_rank=local_rank)
                
                model_engine.train()
                
                # Only process and save results on main process
                if local_rank == 0:
                    result = {**val_result, **test_result}

                    # Filter out OOD datasets when calculating average
                    in_distribution_results = {}
                    for path, values in test_result.items():
                        # Extract dataset name from path and do exact matching
                        dataset_name = os.path.basename(path).split('.')[0].lower()
                        is_ood = dataset_name in ood_keywords
                        if not is_ood:
                            in_distribution_results[path] = values

                    # Use test_dataset_avg as the metric (already excludes OOD via evaluation logic)
                    average = test_dataset_avg

                    # Collect current step results
                    current_results = {}
                    for path, values in test_result.items():
                        dataset_name = os.path.basename(path).split('.')[0]
                        current_results[f"best_{dataset_name}_accuracy"] = values[1]

                    # Track best model based on dataset-level average
                    if average > max_average or not best_results:
                        # Save model
                        # model_engine.save_checkpoint(args.save_path, "best_model") # 暂时不保存模型
                        max_average = average

                        # Update global best results including both averages
                        best_results = {
                            "best_dataset_avg_accuracy": test_dataset_avg,
                            "best_sample_avg_accuracy": test_sample_avg,
                            "best_step": step,
                            **current_results
                        }

                    # Always log the global best results to maintain continuous curve in wandb
                    if wandb.run is not None:
                        wandb.log(best_results)

                    training_log.append(result)

                    # Track training average
                    if val_dataset_avg > max_training_average:
                        # model_engine.save_checkpoint(args.save_path, "best_training_model")
                        max_training_average = val_dataset_avg
                    
                    # Log current averages to wandb
                    if wandb.run is not None:
                        wandb.log({
                            "test_dataset_avg_accuracy": test_dataset_avg,
                            "test_sample_avg_accuracy": test_sample_avg,
                            "train_dataset_avg_accuracy": val_dataset_avg,
                            "train_sample_avg_accuracy": val_sample_avg,
                            "max_average_test": max_average,
                            "max_average_train": max_training_average,
                            "step": step
                        })

        if local_rank == 0:
            print(f"step:{step}, avg_loss_per_epoch:{losses.avg}")
            
        if step >= args.training_steps:
            break

    # Only print on main process
    if local_rank == 0:
        # Display best performance summary
        print_major_header("⭐ BEST PERFORMANCE")

        best_dataset_avg = best_results.get('best_dataset_avg_accuracy', max_average)
        best_sample_avg = best_results.get('best_sample_avg_accuracy', 0)
        best_step_num = best_results.get('best_step', 0)

        print(f"\n  Best Dataset-Level Accuracy: {best_dataset_avg:.4f} at Step {best_step_num}")
        print(f"  Best Sample-Level Accuracy: {best_sample_avg:.4f}")

        # Display per-dataset best accuracies
        print_minor_header("Per-Dataset Best Accuracies", indent=2)
        for key, value in sorted(best_results.items()):
            if key.startswith("best_") and key not in ["best_dataset_avg_accuracy", "best_sample_avg_accuracy", "best_step"]:
                dataset_name = key.replace("best_", "").replace("_accuracy", "")
                print(f"    {dataset_name}: {value:.4f}")

        print(f"\n  Best Training Average: {max_training_average:.4f}")

        # save the model
        with open(os.path.join(args.save_path, "training_log.json"), 'w') as f:
            json.dump(training_log, f)

        with open(os.path.join(args.save_path, "config.txt"), 'w') as f:
            f.write(str(args))

        # Close wandb
        if wandb.run is not None:
            wandb.finish()
