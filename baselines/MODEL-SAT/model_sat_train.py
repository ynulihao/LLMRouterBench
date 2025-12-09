import sys
import numpy as np
import transformers
import torch
import json
from time import sleep
from torch import nn
from safetensors.torch import save_file as safe_save_file
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoConfig
from functools import partial
from datetime import datetime
import datasets
import copy
import matplotlib.pyplot as plt
import re
import os
import string
import shutil
import random
from modeling_projector import ProjectorConfig, LinearProjector, MLPProjector
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import tqdm, argparse
import deepspeed
import time
import torch.distributed as dist

def parse_args():
    parser = argparse.ArgumentParser(description="Train a router model with DeepSpeed")
    parser.add_argument("--base_model_name_or_path", type=str, default="/path/to/Qwen2.5-7B-Instruct", help="The model name or path to the model")
    parser.add_argument("--embed_model_name_or_path", type=str, default="/path/to/gte_Qwen2-7B-instruct", help="The model name or path to the Embedding model")
    parser.add_argument("--experts_information_file", type=str, default='./data/seed42/model_description.json', help="The file containing the experts information")
    parser.add_argument("--dataset_name", type=str, default='{"train": "./data/seed42/train.json", "validation": "./data/seed42/test.json", "ood": "./data/seed42/ood.json" }', help="The dataset to use")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="The output directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4, help="The batch size")
    parser.add_argument("--max_length", type=int, default=1024, help="The maximum length of the input")
    parser.add_argument("--proj_lr", type=float, default=1e-4, help="The learning rate for projector")
    parser.add_argument("--llm_unfreeze_step", type=int, default=3000,
                    help="Global step after which to unfreeze LLM & embed model")
    parser.add_argument("--llm_lr", type=float, default=2e-6, help="Learning rate for LLM parameters")
    parser.add_argument("--embed_lr", type=float, default=2e-6, help="Learning rate for embedding-model parameters")
    parser.add_argument("--projector_path", type=str, default=None, help="The path of projector to load from")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="The number of training epochs")
    parser.add_argument("--projector_type", type=str, default="nonlinear", help="The type of projector to use")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--enable_wandb", action="store_true", help="Enable Weights & Biases logging")
    # DeepSpeed will read some arguments such as local_rank from the command line automatically
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def pad_batch(batch, pad_token_id, label_pad_token_id=-100):
    max_len = max(tensor.size(1) for tensor in batch["input_ids"])
    padded_input_ids, padded_labels, padded_attention_mask = [], [], []

    for input_ids, labels, attention_mask in zip(
        batch["input_ids"], batch["labels"], batch["attention_mask"]
    ):
        current_len = input_ids.size(1)
        pad_length = max_len - current_len

        if pad_length > 0:
            padded_ids = torch.cat(
                [input_ids, torch.full((input_ids.size(0), pad_length), pad_token_id, dtype=torch.long)],
                dim=1
            )
        else:
            padded_ids = input_ids

        if pad_length > 0:
            padded_lbl = torch.cat(
                [labels, torch.full((labels.size(0), pad_length), label_pad_token_id, dtype=torch.long)],
                dim=1
            )
        else:
            padded_lbl = labels

        if pad_length > 0:
            padded_mask = torch.cat(
                [attention_mask, torch.zeros((attention_mask.size(0), pad_length), dtype=torch.long)],
                dim=1
            )
        else:
            padded_mask = attention_mask

        padded_input_ids.append(padded_ids)
        padded_labels.append(padded_lbl)
        padded_attention_mask.append(padded_mask)

    batch["input_ids"] = torch.cat(padded_input_ids, dim=0)
    batch["labels"] = torch.cat(padded_labels, dim=0)
    batch["attention_mask"] = torch.cat(padded_attention_mask, dim=0)
    return batch


def process_batched(queries, models, is_correct_sc_list, tasks, indices, tokenizer, template_type, expert_name_to_idx=None):        
    def encode_text(text: str):
        return tokenizer.encode(
            text, 
            return_tensors="pt", 
            truncation=False, 
            padding=False, 
            add_special_tokens=False
        )
    
    # Build batch dict; don't store full strings but store indices instead
    batch = {"input_ids": [], "labels": [], "attention_mask": [], "text_indices": [], "is_correct_sc": [], "task": [], "index": [], "model": []}
    icl_vector_pad_tensor = torch.tensor([[151662]], dtype=torch.long)
    sep2_tensor = encode_text("\n\n")
    
    instruction = f"Based on the model description and the test sample, predict whether the model can handle test sample by indicating 'Yes' or 'No'.{tokenizer.eos_token}\n"
    
    instruction_tokens = encode_text(instruction)
    
    for query, model, is_correct_sc, task, index in zip(queries, models, is_correct_sc_list, tasks, indices):
        text_indices = []  # index of expert text in global list
        
        # template
        if template_type == 'qwen':
            template = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
            )
            template_tokens = encode_text(template)
            
        query_tokens = encode_text(f"{query}")
        
        idx = expert_name_to_idx[model]
        text_indices.append(idx)
            
        template_assistant_tokens = encode_text("<|im_start|>assistant\n")
        output = "Yes" if is_correct_sc else "No"
        tokenizer_output = encode_text(f"{output}")

        input_ids = torch.cat([
            template_tokens,
            icl_vector_pad_tensor,
            sep2_tensor,
            query_tokens,
            sep2_tensor,
            instruction_tokens,
            template_assistant_tokens,
            tokenizer_output
        ], dim=1)
        prefix_length = (template_tokens.size(1) +
                        icl_vector_pad_tensor.size(1) +
                        sep2_tensor.size(1) +
                        query_tokens.size(1) +
                        sep2_tensor.size(1) +
                        instruction_tokens.size(1) +   
                        template_assistant_tokens.size(1))
        loss_start = prefix_length - 1
        output_length = tokenizer_output.size(1)
        
        labels = input_ids.clone()
        labels_shifted = torch.full_like(labels, -100)
        labels_shifted[:, :-1] = labels[:, 1:]
        labels_shifted[:, -1] = -100

        labels_shifted[:, :loss_start] = -100
        labels_shifted[:, loss_start + output_length:] = -100

        batch["input_ids"].append(input_ids)
        batch["labels"].append(labels_shifted)
        batch["text_indices"].append(text_indices)
        batch["attention_mask"].append(torch.ones_like(input_ids))
        batch["is_correct_sc"].append(is_correct_sc)
        batch["task"].append(task)
        batch["index"].append(index)
        batch["model"].append(model)

    return batch


def custom_collate_fn(batch, tokenizer):
    collated_batch = {}
    for key in batch[0].keys():
        if key in ["input_ids", "labels", "attention_mask", "text_indices"]:
            collated_batch[key] = [torch.tensor(item[key]) if not isinstance(item[key], torch.Tensor) else item[key] for item in batch]
        else:
            collated_batch[key] = [item[key] for item in batch]
    collated_batch = pad_batch(
        collated_batch, 
        pad_token_id=tokenizer.pad_token_id, 
        label_pad_token_id=-100
    )
    return collated_batch


def get_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def save_model_and_configs(args, model_engine, model_saving_key, state):
    """
    Save trained projector, embed model and LLM, plus configs & tokenizer
    """
    if get_rank() == 0:
        ckpt_dir = os.path.join(args.output_dir, model_saving_key, state)
        os.makedirs(ckpt_dir, exist_ok=True)

        # 1. save projector
        projector_save_dir = os.path.join(ckpt_dir, "projector")
        os.makedirs(projector_save_dir, exist_ok=True)
        safe_save_file(model_engine.module.projector.state_dict(), os.path.join(projector_save_dir, "model.safetensors"), metadata={"format": "pt"})
        print(f"Projector saved to {projector_save_dir}/model.safetensors")

        # 1.5 save embed model
        embed_save_dir = os.path.join(ckpt_dir, "embed")
        os.makedirs(embed_save_dir, exist_ok=True)
        safe_save_file(model_engine.module.embed_model.state_dict(), os.path.join(embed_save_dir, "model.safetensors"), metadata={"format": "pt"})
        print(f"Embed model saved to {embed_save_dir}/model.safetensors")

        # 2. save LLM
        llm_save_dir = os.path.join(ckpt_dir, "llm")
        os.makedirs(llm_save_dir, exist_ok=True) 
        model_engine.module.big_model.save_pretrained(
            llm_save_dir,
            safe_serialization=True,
            max_shard_size="2GB"
        )
        print(f"LLM saved to {llm_save_dir}/model.safetensors")
        
        try:
            embed_cfg = AutoConfig.from_pretrained(args.embed_model_name_or_path,
                                                   trust_remote_code=True)
            embed_cfg.save_pretrained(embed_save_dir)
        except Exception as e:
            print(f"[warn] embed config failed to save: {e}")

        try:
            embed_tok = AutoTokenizer.from_pretrained(args.embed_model_name_or_path,
                                                      trust_remote_code=True)
            embed_tok.save_pretrained(embed_save_dir)
        except Exception as e:
            print(f"[warn] embed tokenizer failed to save: {e}")

        # 3. config for LLM
        config = AutoConfig.from_pretrained(args.base_model_name_or_path, trust_remote_code=True)
        config.save_pretrained(llm_save_dir)
        # 4. tokenizer for LLM
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, trust_remote_code=True)
        tokenizer.save_pretrained(llm_save_dir)
        # 5. generation config
        src_generation_config = os.path.join(args.base_model_name_or_path, "generation_config.json")
        if os.path.exists(src_generation_config):
            shutil.copy(src_generation_config, os.path.join(llm_save_dir, "generation_config.json"))


class CompositeModel(nn.Module):
    def __init__(
        self,
        embed_model,
        projector,
        big_model,
        expert_desc,
        is_sentence_transformer,
        embed_max_length=512,
    ):
        super().__init__()
        self.embed_model = embed_model
        self.projector = projector
        self.big_model = big_model
        self.expert_desc = expert_desc
        self.is_sentence_transformer = is_sentence_transformer
        self.embed_max_length = embed_max_length
        # tokenizer for embed model when not using SentenceTransformer
        if not self.is_sentence_transformer:
            try:
                self.embed_tokenizer = AutoTokenizer.from_pretrained(
                    getattr(self.embed_model, 'config', None).name_or_path,
                    trust_remote_code=True
                )
            except Exception:
                # Fallback to using big model tokenizer settings if needed later
                self.embed_tokenizer = None

        with torch.no_grad():
            self.cached_embed = self._encode_experts().to(torch.bfloat16)   # [N, D]
        self.register_buffer("cached_embed_buf", self.cached_embed, persistent=False)

        self.embedding_layer = big_model.get_input_embeddings()

    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Mean pool over valid tokens
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def _embed_texts(self, texts) -> torch.Tensor:
        device = next(self.big_model.parameters()).device
        if self.is_sentence_transformer:
            # Use SentenceTransformer forward for grad-enabled embeddings
            features = self.embed_model.tokenize(texts)
            features = {k: v.to(device) for k, v in features.items()}
            outputs = self.embed_model(features)
            if isinstance(outputs, dict) and 'sentence_embedding' in outputs:
                embs = outputs['sentence_embedding']
            else:
                # Fallback to CLS/mean pooling if module returns hidden states
                if hasattr(outputs, 'last_hidden_state'):
                    embs = self._mean_pooling(outputs.last_hidden_state, features.get('attention_mask'))
                elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                    embs = outputs[0]
                else:
                    embs = outputs
            return embs
        else:
            # Hugging Face model path (e.g., NV-Embed-v2)
            if self.embed_tokenizer is None:
                # Lazily initialize if creation failed in __init__
                try:
                    self.embed_tokenizer = AutoTokenizer.from_pretrained(
                        getattr(self.embed_model, 'config', None).name_or_path,
                        trust_remote_code=True
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to initialize embed tokenizer: {e}")
            enc = self.embed_tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=self.embed_max_length,
                return_tensors='pt'
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = self.embed_model(**enc)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embs = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                embs = self._mean_pooling(outputs.last_hidden_state, enc.get('attention_mask'))
            elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                # Assume [last_hidden_state, ...]
                hs = outputs[0]
                if isinstance(hs, torch.Tensor) and hs.dim() == 3:
                    embs = self._mean_pooling(hs, enc.get('attention_mask'))
                else:
                    embs = hs
            else:
                embs = outputs
            return embs

    def _encode_experts(self) -> torch.Tensor:
        # Use forward pass (no_grad for caching) to support consistent behavior
        embs = self._embed_texts(self.expert_desc)
        return embs.detach().cpu()

    @torch.no_grad()
    def refresh_cached_embed(self):
        self.cached_embed = self._encode_experts().to(torch.bfloat16)
        self.cached_embed_buf = self.cached_embed      

    def forward(
        self,
        text_indices: torch.LongTensor,   # [B, M]
        input_ids:   torch.LongTensor,    # [B, L]
        attention_mask: torch.LongTensor,
    ):
        B, M = text_indices.size()
        device = input_ids.device
        flat_idx = text_indices.view(-1)
        
        need_dynamic = (
            self.training                                   
            and any(p.requires_grad for p in self.embed_model.parameters())  
        )

        if need_dynamic:
            texts = [self.expert_desc[i] for i in flat_idx.tolist()]
            embed_out = self._embed_texts(texts)
        else:
            embed_out = self.cached_embed_buf[flat_idx.cpu()].to(device)

        embed_out = embed_out.view(B, M, -1)                    # [B, M, D]
        projected = self.projector(embed_out)                   # [B, M, d_model]
        model_input = self.embedding_layer(input_ids)

        # Replace the <ICL_PAD> positions with model description vectors.
        mask = (input_ids == 151662)
        order = torch.cumsum(mask, dim=1) - 1
        row_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(input_ids)
        replacement = torch.zeros_like(model_input)
        replacement[mask] = projected[row_idx[mask], order[mask]]
        model_input = torch.where(mask.unsqueeze(-1), replacement, model_input)

        outputs = self.big_model(
            inputs_embeds=model_input,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return outputs.logits


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    time_tag = datetime.now().strftime("%Y%m%d%H%M")
    
    model_saving_key = "{}_sft_router_{}_{}_{}_{}_{}_{}_train_proj_llm_embed_{}".format(
        args.projector_type,
        args.proj_lr,
        args.llm_lr,
        args.embed_lr,
        args.num_train_epochs,
        args.embed_model_name_or_path.split('/')[-1],
        "Qwen2.5-7B-Instruct",
        time_tag
    )
    
    # Optionally initialize Weights & Biases
    wandb = None
    if args.enable_wandb and get_rank() == 0:
        try:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(
                project="model_sat",
                name=model_saving_key,
                config=vars(args)
            )
        except Exception as e:
            print(f"[warn] Failed to initialize wandb, proceeding without it: {e}")
    # distributed setup
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        if not torch.distributed.is_initialized():
            dist.init_process_group(backend="nccl")
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attn_implementation = "flash_attention_2"
    big_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        # attn_implementation=attn_implementation,
        trust_remote_code=True
    )
    big_model.config.use_cache = False
    big_model.gradient_checkpointing_enable()
    big_model = big_model.to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model_name_or_path, trust_remote_code=True)
    
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id  = tokenizer.encode("No",  add_special_tokens=False)[0]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_sentence_transformer = False
    if "jina-embeddings-v3" in args.embed_model_name_or_path:
        embed_model = SentenceTransformer(args.embed_model_name_or_path, trust_remote_code=True, model_kwargs={'lora_main_params_trainable': True}, local_files_only=True)
        embed_model.max_seq_length = args.max_length
        is_sentence_transformer = True
        in_features = 768
    elif "gte_Qwen2" in args.embed_model_name_or_path:
        embed_model = SentenceTransformer(args.embed_model_name_or_path, trust_remote_code=True)
        embed_model.max_seq_length = args.max_length
        is_sentence_transformer = True
        if "gte_Qwen2-7B-instruct" in args.embed_model_name_or_path:
            in_features = 3584
        elif "gte_Qwen2-1.5B-instruct" in args.embed_model_name_or_path:
            in_features = 1536
    elif "NV-Embed-v2" in args.embed_model_name_or_path:
        embed_config = AutoConfig.from_pretrained(args.embed_model_name_or_path, trust_remote_code=True)
        embed_model = AutoModel.from_pretrained(
            args.embed_model_name_or_path,
            config=embed_config,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        in_features = 4096
    else:
        raise ValueError(f"Unsupported embed model: {args.embed_model_name_or_path}")
    embed_model = embed_model.to(device)
    
    # embed_model.gradient_checkpointing_enable()

    # determine in/out features
    embedding_layer = big_model.get_input_embeddings()
    out_features = embedding_layer.weight.shape[1]

    expansion_ratio = 1

    linear_config = ProjectorConfig(in_features=in_features, out_features=out_features, expansion_ratio=expansion_ratio)
    Projector = LinearProjector if args.projector_type == "linear" else MLPProjector

    if args.projector_path is not None:
        projector = Projector.from_pretrained(pretrained_model_name_or_path=args.projector_path, config=linear_config, dtype=torch.float32)
    else:
        projector = Projector(config=linear_config, dtype=torch.float32)

    def evaluate_metrics(model_engine, dataloader):
        """
        Return task_acc_sc, overall_sc, avg_loss
        """
        rank        = get_rank()
        world_size  = dist.get_world_size() if dist.is_initialized() else 1
        device      = next(model_engine.parameters()).device

        model_engine.eval()
        ce_loss_fn  = nn.CrossEntropyLoss(ignore_index=-100)

        total_loss  = 0.0
        score_dict  = defaultdict(list)          # (task, idx) → [(yes_prob, is_sc, is_dir), …]
        
        with torch.no_grad():
            for batch in dataloader:
                

                indices = torch.stack(batch["text_indices"]).to(device)
                input_ids      = batch['input_ids'     ].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels         = batch['labels'        ].to(device)

                logits = model_engine(indices, input_ids, attention_mask)

                loss = ce_loss_fn(logits.view(-1, logits.size(-1)),
                                labels.view(-1))
                total_loss += loss.item()

                yes_probs = []
                for i in range(logits.size(0)):
                    tgt_pos = (labels[i] != -100).nonzero(as_tuple=False).item()
                    prob = F.softmax(logits[i, tgt_pos], dim=-1)[yes_token_id]
                    yes_probs.append(prob.cpu())

                for i, task in enumerate(batch['task']):
                    key = (task, batch['index'][i])
                    score_dict[key].append(
                        (yes_probs[i],
                        bool(batch['is_correct_sc'][i]))      
                    )
                    
                del logits
                torch.cuda.empty_cache()

         # ---- calculate loss ----
        total_loss_tensor = torch.tensor(total_loss, device=device)
        if dist.is_initialized():
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = total_loss_tensor.item() / world_size / (len(dataloader) or 1)

        # ---- gather score_dict ----
        if dist.is_initialized():
            gathered = [None] * world_size
            dist.all_gather_object(gathered, score_dict)
            if rank == 0:
                merged = defaultdict(list)
                for d in gathered:
                    for k, v in d.items():
                        merged[k].extend(v)
                score_dict = merged
            else:
                return {}, 0.0, avg_loss

        per_task_sc = defaultdict(list)
        for (task, _), lst in score_dict.items():
            best = max(lst, key=lambda x: x[0])
            per_task_sc[task].append(int(best[1]))
        task_acc_sc = {t: sum(v)/len(v) for t, v in per_task_sc.items()}
        overall_sc  = sum(task_acc_sc.values())/len(task_acc_sc) if task_acc_sc else 0.0

        return task_acc_sc, overall_sc, avg_loss
    # dataset loading
    data_files = json.loads(args.dataset_name)
    raw_dataset = datasets.load_dataset("json", data_files=data_files)
    my_collate_fn = partial(custom_collate_fn, tokenizer=tokenizer)

    # expert descriptions
    with open(args.experts_information_file, 'r') as f:
        experts_information = json.load(f)

    expert_name_to_idx = {}
    unique_expert_desc = []
    for model_name, desc in experts_information.items():
        if model_name not in expert_name_to_idx:
            expert_name_to_idx[model_name] = len(expert_name_to_idx)
            unique_expert_desc.append(desc)

    has_ood = "ood" in raw_dataset

    train_dataset = raw_dataset['train'].map(
        process_batched,
        fn_kwargs={"tokenizer": tokenizer, "template_type": "qwen", "expert_name_to_idx": expert_name_to_idx},
        batched=True,
        batch_size=args.batch_size,
        num_proc=32,
        input_columns=['query', 'model', 'is_correct_sc', 'task', 'index'],
        remove_columns=raw_dataset['train'].column_names
    )
    val_dataset = raw_dataset['validation'].map(
        process_batched,
        fn_kwargs={"tokenizer": tokenizer, "template_type": "qwen", "expert_name_to_idx": expert_name_to_idx},
        batched=True,
        batch_size=args.batch_size,
        num_proc=32,
        input_columns=['query', 'model', 'is_correct_sc', 'task', 'index'],
        remove_columns=raw_dataset['validation'].column_names
    )
    
    ood_dataset = None
    if has_ood:
        ood_dataset = raw_dataset['ood'].map(
            process_batched,
            fn_kwargs={"tokenizer": tokenizer, "template_type": "qwen", "expert_name_to_idx": expert_name_to_idx},
            batched=True,
            batch_size=args.batch_size,
            num_proc=32,
            input_columns=['query', 'model', 'is_correct_sc', 'task', 'index'],
            remove_columns=raw_dataset['ood'].column_names
        )

    # dataloaders
    if args.local_rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, collate_fn=my_collate_fn, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler, collate_fn=my_collate_fn)
        if ood_dataset is not None:
            ood_sampler = torch.utils.data.distributed.DistributedSampler(ood_dataset, shuffle=False)
            ood_dataloader = torch.utils.data.DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, sampler=ood_sampler, collate_fn=my_collate_fn)
        else:
            ood_dataloader = None
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn, drop_last=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
        ood_dataloader = torch.utils.data.DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn) if ood_dataset is not None else None

    num_training_steps = len(train_dataloader) * args.num_train_epochs
    warmup_steps = int(0.1 * num_training_steps)

    deepspeed_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {"weight_decay": 0.1},
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {"warmup_num_steps": warmup_steps, "total_num_steps": num_training_steps},
            "param_schedulers": {
                "projector": {
                "scheduler": "WarmupCosineLR",
                "warmup_num_steps": warmup_steps,
                "total_num_steps": num_training_steps
                }
            }
        },
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 1e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 1e8,
            "contiguous_gradients": True,
            "round_robin_gradients": True,
        }
    }

    composite_model = CompositeModel(embed_model, projector, big_model, unique_expert_desc, is_sentence_transformer, args.max_length)

    for p in composite_model.big_model.parameters():
        p.requires_grad = False
        
    for p in composite_model.embed_model.parameters():
        p.requires_grad = False
        
    optimizer_grouped_parameters = [
        {"params": composite_model.projector.parameters(), "lr": args.proj_lr, "weight_decay": 0.1, "name": "projector"},
    ]

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(args=args, model=composite_model, model_parameters=optimizer_grouped_parameters, config=deepspeed_config)

    CELoss = nn.CrossEntropyLoss()
    pbar = tqdm.tqdm(total=num_training_steps)
    
    global_step = 0    
    model_engine.train()

    for epoch in range(args.num_train_epochs):
            
        if args.local_rank != -1:
            train_dataloader.sampler.set_epoch(epoch)

        for batch in train_dataloader:
            if global_step == args.llm_unfreeze_step:
                for p in composite_model.big_model.parameters():
                    p.requires_grad = True
                for p in composite_model.embed_model.parameters():
                    p.requires_grad = True
                    
                full_ds_cfg = copy.deepcopy(deepspeed_config)
                full_ds_cfg["scheduler"]["param_schedulers"]["big_model"] = {
                        "scheduler": "WarmupCosineLR",
                        "warmup_num_steps": int(0.1 * (num_training_steps - global_step)),
                        "total_num_steps": num_training_steps - global_step,
                }
                full_ds_cfg["scheduler"]["param_schedulers"]["embed_model"] = {
                        "scheduler": "WarmupCosineLR",
                        "warmup_num_steps": int(0.1 * (num_training_steps - global_step)),
                        "total_num_steps": num_training_steps - global_step,
                }
                full_ds_cfg["scheduler"]["param_schedulers"]["projector"]["start_step"] = global_step
                del model_engine, optimizer, lr_scheduler
                torch.cuda.empty_cache()
                
                optimizer_grouped_parameters = [
                {"params": composite_model.projector.parameters(), "lr": args.proj_lr, "weight_decay": 0.1, "name": "projector"},
                {"params": composite_model.big_model.parameters(), "lr": args.llm_lr, "weight_decay": 0.1, "name": "big_model"},
                {"params": composite_model.embed_model.parameters(), "lr": args.embed_lr, "weight_decay": 0.1, "name": "embed_model"},
                ]
                
                model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
                    args=args,
                    model=composite_model,
                    model_parameters=optimizer_grouped_parameters,
                    config=full_ds_cfg
                )
            
            
            global_step += 1

            indices = torch.stack(batch["text_indices"]).to(device)           # [B, M]
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model_engine(indices, input_ids, attention_mask)
            loss = CELoss(logits.view(-1, logits.size(-1)), labels.view(-1))

            model_engine.backward(loss)
            
            
            if (not torch.isfinite(loss)
                or any(p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model_engine.module.parameters())
            ):
                if get_rank() == 0:
                    print(f"[step {global_step}] loss/grad 有异常，跳过该 batch")
                model_engine.zero_grad()
                continue
            
            model_engine.step()
            model_engine.zero_grad()

            # --- logging ---
            with torch.no_grad():
                red_loss = loss.detach().clone()
                if dist.is_initialized():
                    dist.all_reduce(red_loss)
                    red_loss /= dist.get_world_size()
                if get_rank() == 0 and global_step % 20 == 0:
                    
                    tqdm.tqdm.write(f"Epoch {epoch} Step {global_step} | train_loss={red_loss.item():.4f}")

            # --- evaluation every 500 optimisation steps ---
            if global_step % 3000 == 0 or global_step == num_training_steps:
                if get_rank() == 0:
                    save_model_and_configs(args, model_engine, model_saving_key, f"step_{global_step}")
                    
                dist.barrier()   
                composite_model.refresh_cached_embed()
                
                torch.cuda.empty_cache()
                va_task_acc_sc, va_overall_sc, va_loss = evaluate_metrics(model_engine, val_dataloader)
                
                torch.cuda.empty_cache() 
                
                if ood_dataloader is not None:
                    ood_task_acc_sc, ood_overall_sc, ood_loss = evaluate_metrics(model_engine, ood_dataloader)
                    torch.cuda.empty_cache()
                else:
                    ood_task_acc_sc, ood_overall_sc, ood_loss = {}, None, None
                
                if get_rank() == 0:
                    print(f"★ Global step {global_step}")
                    
                    print(f" Validation loss: {va_loss:.4f} | SC acc: {va_overall_sc:.4f}")
                    for t in sorted(va_task_acc_sc):
                        sc = va_task_acc_sc[t]
                        print(f"  {t}: SC {sc:.4f}")
                        
                    if ood_dataloader is not None:
                        print(f"    OOD     loss: {ood_loss:.4f} | SC acc: {ood_overall_sc:.4f}")
                        for t in sorted(ood_task_acc_sc):
                            sc = ood_task_acc_sc[t]
                            print(f"  {t}: SC {sc:.4f}")
                    else:
                        print("    OOD evaluation skipped (no OOD dataset)")
                        
                    if args.enable_wandb and wandb is not None:
                        wandb.log({"valid_acc": va_overall_sc}, step=global_step)
                        if ood_dataloader is not None:
                            wandb.log({"ood_acc": ood_overall_sc}, step=global_step)
                        
                model_engine.train()
                    
            pbar.update(1)

        

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    main()
