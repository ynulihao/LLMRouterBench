import torch
import torch.nn.functional as F
from torch_geometric.nn import GeneralConv
from torch_geometric.data import Data
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score

class FeatureAlign(nn.Module):

    def __init__(self, query_feature_dim, llm_feature_dim, common_dim):
        super(FeatureAlign, self).__init__()
        self.query_transform = nn.Linear(query_feature_dim, common_dim)
        self.llm_transform = nn.Linear(llm_feature_dim, common_dim*2)
        self.task_transform = nn.Linear(llm_feature_dim, common_dim)

    def forward(self,task_id, query_features, llm_features):
        aligned_task_features = self.task_transform(task_id)
        aligned_query_features = self.query_transform(query_features)
        aligned_two_features=torch.cat([aligned_task_features,aligned_query_features], 1)
        aligned_llm_features = self.llm_transform(llm_features)
        aligned_features = torch.cat([aligned_two_features, aligned_llm_features], 0)
        return aligned_features


class EncoderDecoderNet(torch.nn.Module):

    def __init__(self, query_feature_dim, llm_feature_dim, hidden_features, in_edges):
        super(EncoderDecoderNet, self).__init__()
        self.in_edges = in_edges
        self.model_align = FeatureAlign(query_feature_dim, llm_feature_dim, hidden_features)
        self.encoder_conv_1 = GeneralConv(in_channels=hidden_features* 2, out_channels=hidden_features* 2, in_edge_channels=in_edges)
        self.encoder_conv_2 = GeneralConv(in_channels=hidden_features* 2, out_channels=hidden_features* 2, in_edge_channels=in_edges)
        self.edge_mlp = nn.Linear(in_edges, in_edges)
        self.bn1 = nn.BatchNorm1d(hidden_features * 2)
        self.bn2 = nn.BatchNorm1d(hidden_features * 2)

    def forward(self, task_id, query_features, llm_features, edge_index, edge_mask=None,
                edge_can_see=None, edge_weight=None):
        if edge_mask is not None:
            edge_index_mask = edge_index[:, edge_can_see]
            edge_index_predict = edge_index[:, edge_mask]
            if edge_weight is not None:
                edge_weight_mask = edge_weight[edge_can_see]
        edge_weight_mask=F.relu(self.edge_mlp(edge_weight_mask.reshape(-1, self.in_edges)))
        edge_weight_mask = edge_weight_mask.reshape(-1,self.in_edges)
        x_ini = (self.model_align(task_id, query_features, llm_features))
        x = F.relu(self.bn1(self.encoder_conv_1(x_ini, edge_index_mask, edge_attr=edge_weight_mask)))
        x = self.bn2(self.encoder_conv_2(x, edge_index_mask, edge_attr=edge_weight_mask))
        edge_predict = F.sigmoid(
            (x_ini[edge_index_predict[0]] * x[edge_index_predict[1]]).mean(dim=-1))
        return edge_predict

class form_data:

    def __init__(self,device):
        self.device = device

    def formulation(self,task_id,query_feature,llm_feature,org_node,des_node,edge_feature,label,edge_mask,combined_edge,train_mask,valide_mask,test_mask,cost_usd=None):

        query_features = torch.tensor(query_feature, dtype=torch.float).to(self.device)
        llm_features = torch.tensor(llm_feature, dtype=torch.float).to(self.device)
        task_id=torch.tensor(task_id, dtype=torch.float).to(self.device)
        query_indices = list(range(len(query_features)))
        llm_indices = [i + len(query_indices) for i in range(len(llm_features))]
        des_node=[(i+1 + org_node[-1]) for i in des_node]
        edge_index = torch.tensor([org_node, des_node], dtype=torch.long).to(self.device)
        edge_weight = torch.tensor(edge_feature, dtype=torch.float).reshape(-1,1).to(self.device)
        combined_edge=torch.tensor(combined_edge, dtype=torch.float).reshape(-1,2).to(self.device)
        combined_edge=torch.cat((edge_weight, combined_edge), dim=-1)
        # Optional USD cost tensor for evaluation-only cost accounting
        if cost_usd is not None:
            cost_usd_t = torch.tensor(cost_usd, dtype=torch.float).reshape(-1).to(self.device)
        else:
            cost_usd_t = None

        data = Data(task_id=task_id,query_features=query_features, llm_features=llm_features, edge_index=edge_index,
                        edge_attr=edge_weight,query_indices=query_indices, llm_indices=llm_indices,label=torch.tensor(label, dtype=torch.float).to(self.device),
                        edge_mask=edge_mask,combined_edge=combined_edge,
                    train_mask=train_mask,valide_mask=valide_mask,test_mask=test_mask)

        # Attach USD costs if provided (used only at evaluation time)
        if cost_usd_t is not None:
            data.cost_usd = cost_usd_t

        return data


class GNN_prediction:
    def __init__(self, query_feature_dim, llm_feature_dim,hidden_features_size,in_edges_size,wandb,config,device):

        self.model = EncoderDecoderNet(query_feature_dim=query_feature_dim, llm_feature_dim=llm_feature_dim,
                                        hidden_features=hidden_features_size,in_edges=in_edges_size).to(device)
        self.wandb = wandb
        self.config = config
        self.optimizer =AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        self.criterion = torch.nn.BCELoss()
        self.best_test_per_task_acc = {}  # Store per-task accuracy at best validation per-task avg accuracy
        # Store comprehensive metrics at best test performance
        self.best_test_metrics = {
            'per_task_acc': {},           # {task_id: accuracy}
            'per_task_total_cost': {},    # {task_id: total_cost}
            'dataset_avg_acc': 0.0,       # Dataset-level average accuracy
            'sample_avg_acc': 0.0,        # Sample-level average accuracy
            'total_cost': 0.0             # Total cost across all samples
        }
        # Track last-epoch metrics (for logging/printing after training)
        self.last_epoch_metrics = {
            'per_task_acc': {},           # {task_id: accuracy}
            'per_task_total_cost': {},    # {task_id: total_cost}
            'dataset_avg_acc': 0.0,
            'sample_avg_acc': 0.0,
            'total_cost': 0.0,
            'cost_source': 'normalized'
        }

    def train_validate(self,data,data_validate,data_for_test,query_task_ids=None):
        """
        Train the model and use test set for model selection.

        NOTE: data_validate is kept for compatibility but will be empty (validation set removed).
        Model selection is now based on test set performance.
        """
        best_test_per_task_avg_acc = -1
        best_test_result = float('-inf')
        self.save_path= self.config['model_path']
        self.num_edges = len(data.edge_attr)
        self.train_mask = torch.tensor(data.train_mask, dtype=torch.bool)
        self.valide_mask = torch.tensor(data.valide_mask, dtype=torch.bool)
        self.test_mask = torch.tensor(data.test_mask, dtype=torch.bool)
        for epoch in range(self.config['train_epoch']):
            self.model.train()
            loss_mean=0
            mask_train = data.edge_mask
            for inter in range(self.config['batch_size']):
                mask = mask_train.clone()
                mask = mask.bool()
                random_mask = torch.rand(mask.size()) < self.config['train_mask_rate']
                random_mask = random_mask.to(torch.bool)
                mask = torch.where(mask & random_mask, torch.tensor(False, dtype=torch.bool), mask)
                mask = mask.bool()
                edge_can_see = torch.logical_and(~mask, self.train_mask)
                self.optimizer.zero_grad()
                predicted_edges= self.model(task_id=data.task_id,query_features=data.query_features, llm_features=data.llm_features, edge_index=data.edge_index,
                                            edge_mask=mask,edge_can_see=edge_can_see,edge_weight=data.combined_edge)
                loss = self.criterion(predicted_edges.reshape(-1), data.label[mask].reshape(-1))
                loss_mean+=loss
            loss_mean=loss_mean/self.config['batch_size']
            loss_mean.backward()
            self.optimizer.step()

            # Evaluate on test set for model selection
            self.model.eval()
            mask_test = torch.tensor(data_for_test.edge_mask, dtype=torch.bool)
            edge_can_see = self.train_mask  # Test can only see training edges
            with torch.no_grad():
                predicted_edges_test = self.model(task_id=data_for_test.task_id,query_features=data_for_test.query_features,
                                                                            llm_features=data_for_test.llm_features,
                                                                            edge_index=data_for_test.edge_index,
                                                                            edge_mask=mask_test,edge_can_see=edge_can_see, edge_weight=data_for_test.combined_edge)
                observe_edge= predicted_edges_test.reshape(-1, self.config['llm_num'])
                observe_idx = torch.argmax(observe_edge, 1)

                # Accuracy based on correctness (score > 0), not utility
                combined_edge_masked = data_for_test.combined_edge[mask_test]
                # combined_edge columns: [0]=utility, [1]=cost, [2]=raw_effect (score)
                score_matrix = combined_edge_masked[:, 2].reshape(-1, self.config['llm_num'])
                is_correct_matrix = (score_matrix > 0).to(dtype=torch.float32)
                correct = is_correct_matrix[torch.arange(len(observe_idx)), observe_idx].sum().item()
                total = len(observe_idx)
                test_accuracy = correct / total if total > 0 else 0.0
                # For F1 score calculation (still using argmax for compatibility)
                value_test=data_for_test.edge_attr[mask_test].reshape(-1, self.config['llm_num'])
                row_indices_for_ckpt = torch.arange(len(value_test))
                test_result_for_ckpt = value_test[row_indices_for_ckpt, observe_idx].mean().item()
                label_idx = torch.argmax(value_test, 1)
                observe_idx_ = observe_idx.cpu().numpy()
                label_idx_ = label_idx.cpu().numpy()
                # calculate macro F1 score
                test_f1 = f1_score(label_idx_, observe_idx_, average='macro')
                loss_test = self.criterion(predicted_edges_test.reshape(-1), data_for_test.label[mask_test].reshape(-1))

                # Compute test per-task accuracy
                test_per_task_acc_dict = {}
                if query_task_ids is not None:
                    test_per_task_acc_dict = self.compute_per_task_accuracy(
                        data_for_test, query_task_ids
                    )
                    test_per_task_avg_acc = sum(test_per_task_acc_dict.values()) / len(test_per_task_acc_dict)
                else:
                    test_per_task_avg_acc = test_accuracy  # Fallback to overall accuracy

                # Compute test per-task cost
                test_per_task_total_cost = {}
                test_total_cost = 0.0
                cost_source = 'normalized'
                if query_task_ids is not None:
                    test_per_task_total_cost, test_total_cost, cost_source = self.compute_per_task_cost(
                        data_for_test, query_task_ids
                    )

                # Save model if test effect improves (selection based on test_result_for_ckpt)
                if test_result_for_ckpt > best_test_result:
                    best_test_result = test_result_for_ckpt
                    best_test_per_task_avg_acc = test_per_task_avg_acc
                    torch.save(self.model.state_dict(), self.save_path)
                    # Store the best test per-task accuracy (for backward compatibility)
                    if query_task_ids is not None:
                        self.best_test_per_task_acc = test_per_task_acc_dict
                    # Store comprehensive metrics at best checkpoint
                    self.best_test_metrics = {
                        'per_task_acc': test_per_task_acc_dict.copy() if query_task_ids is not None else {},
                        'per_task_total_cost': test_per_task_total_cost.copy() if query_task_ids is not None else {},
                        'dataset_avg_acc': test_per_task_avg_acc,
                        'sample_avg_acc': test_accuracy,
                        'total_cost': test_total_cost,
                        'cost_source': cost_source
                    }

                # Always track last-epoch metrics (for summary printing)
                self.last_epoch_metrics = {
                    'per_task_acc': test_per_task_acc_dict.copy() if query_task_ids is not None else {},
                    'per_task_total_cost': test_per_task_total_cost.copy() if query_task_ids is not None else {},
                    'dataset_avg_acc': test_per_task_avg_acc,
                    'sample_avg_acc': test_accuracy,
                    'total_cost': test_total_cost,
                    'cost_source': cost_source
                }

                # Compute test result (average effect)
                row_indices = torch.arange(len(value_test))
                test_result = value_test[row_indices, observe_idx].mean()

                # Calculate average cost per sample
                test_avg_cost_per_sample = test_total_cost / total if total > 0 else 0.0

                log_payload = {
                    "train_loss": loss_mean,
                    "test_loss": loss_test,
                    "test_accuracy": test_accuracy,
                    "test_f1": test_f1,
                    "test_per_task_avg_acc": test_per_task_avg_acc,
                    "test_result": test_result,
                    "test_total_cost": test_total_cost,
                    "test_avg_cost_per_sample": test_avg_cost_per_sample,
                    # Dataset-level summary metrics (mirror BEST TEST CHECKPOINT METRICS)
                    "dataset_avg_acc": test_per_task_avg_acc,
                    "sample_avg_acc": test_accuracy,
                    "total_cost": test_total_cost,
                }

                # Log per-task accuracy and cost (if query_task_ids provided)
                if query_task_ids is not None and test_per_task_acc_dict:
                    for task_id, acc in test_per_task_acc_dict.items():
                        # Use namespaced keys so wandb aggregates nicely
                        log_payload[f"per_task/acc/{task_id}"] = acc
                        cost_total = test_per_task_total_cost.get(task_id, 0.0)
                        log_payload[f"per_task/cost/{task_id}"] = cost_total
                # Optionally log cost_source tag if wandb is available
                if self.wandb is not None:
                    try:
                        log_payload["cost_source"] = 1 if cost_source == 'usd' else 0
                        log_payload["cost_source/text"] = cost_source
                    except Exception:
                        pass
                self.wandb.log(log_payload)

        # Print comprehensive metrics at best test checkpoint
        if self.best_test_metrics['per_task_acc']:
            print("\n" + "="*70)
            print("BEST TEST CHECKPOINT METRICS (used for model selection)")
            print("="*70)

            # Summary metrics
            print("\n[Summary Metrics]")
            print(f"  Dataset-Level Average Accuracy: {self.best_test_metrics['dataset_avg_acc']:.4f}")
            print(f"  Sample-Level Average Accuracy:  {self.best_test_metrics['sample_avg_acc']:.4f}")
            print(f"  Total Cost:                     {self.best_test_metrics['total_cost']:.4f}")
            if 'cost_source' in self.best_test_metrics:
                print(f"  Cost Source:                    {self.best_test_metrics['cost_source']}")

            # Per-task metrics
            print("\n[Per-Task Metrics]")
            for task_id, acc in sorted(self.best_test_metrics['per_task_acc'].items()):
                cost_total = self.best_test_metrics['per_task_total_cost'].get(task_id, 0.0)
                print(f"  {task_id:30} Acc: {acc:.4f}  |  Total Cost: {cost_total:.4f}")

            print("="*70 + "\n")

        # Print comprehensive metrics for the last training epoch
        if self.last_epoch_metrics['per_task_acc'] or self.last_epoch_metrics['dataset_avg_acc'] > 0.0:
            print("\n" + "="*70)
            print("LAST EPOCH METRICS")
            print("="*70)

            # Summary metrics
            print("\n[Summary Metrics]")
            print(f"  Dataset-Level Average Accuracy: {self.last_epoch_metrics['dataset_avg_acc']:.4f}")
            print(f"  Sample-Level Average Accuracy:  {self.last_epoch_metrics['sample_avg_acc']:.4f}")
            print(f"  Total Cost:                     {self.last_epoch_metrics['total_cost']:.4f}")
            if 'cost_source' in self.last_epoch_metrics:
                print(f"  Cost Source:                    {self.last_epoch_metrics['cost_source']}")

            # Per-task metrics
            if self.last_epoch_metrics['per_task_acc']:
                print("\n[Per-Task Metrics]")
                for task_id, acc in sorted(self.last_epoch_metrics['per_task_acc'].items()):
                    cost_total = self.last_epoch_metrics['per_task_total_cost'].get(task_id, 0.0)
                    print(f"  {task_id:30} Acc: {acc:.4f}  |  Total Cost: {cost_total:.4f}")

            print("="*70 + "\n")

    def test(self,data,model_path):
        """
        Evaluate the model on test set.

        NOTE: Since validation set is removed, test can only see training edges.
        """
        # self.model.load_state_dict(model_path)
        self.model.eval()
        mask = torch.tensor(data.edge_mask, dtype=torch.bool)
        edge_can_see = self.train_mask  # Test can only see training edges (no validation set)
        with torch.no_grad():
            edge_predict = self.model(task_id=data.task_id,query_features=data.query_features, llm_features=data.llm_features, edge_index=data.edge_index,
                             edge_mask=mask,edge_can_see=edge_can_see,edge_weight=data.combined_edge)
        label = data.label[mask].reshape(-1)
        loss_test = self.criterion(edge_predict, label)
        edge_predict = edge_predict.reshape(-1, self.config['llm_num'])
        max_idx = torch.argmax(edge_predict, 1)
        value_test = data.edge_attr[mask].reshape(-1, self.config['llm_num'])
        label_idx = torch.argmax(value_test, 1)
        row_indices = torch.arange(len(value_test))
        result = value_test[row_indices, max_idx].mean()
        result_golden = value_test[row_indices, label_idx].mean()
        print("result_predict:", result, "result_golden:",result_golden)

        return result,loss_test

    def compute_per_task_accuracy(self, data, query_task_ids):
        """
        Compute per-task accuracy on test set.

        Args:
            data: Test set data
            query_task_ids: List of task_id for each query

        Returns:
            dict: {task_id: accuracy} mapping
        """
        self.model.eval()
        mask = torch.tensor(data.edge_mask, dtype=torch.bool)
        edge_can_see = self.train_mask  # Can only see training edges (no validation set)

        with torch.no_grad():
            edge_predict = self.model(
                task_id=data.task_id,
                query_features=data.query_features,
                llm_features=data.llm_features,
                edge_index=data.edge_index,
                edge_mask=mask,
                edge_can_see=edge_can_see,
                edge_weight=data.combined_edge
            )

        # Get predicted LLM indices
        edge_predict = edge_predict.reshape(-1, self.config['llm_num'])
        predicted_llm_idx = torch.argmax(edge_predict, 1)

        # Correctness based on raw score (effect > 0), not utility
        combined_edge_masked = data.combined_edge[mask]  # [num_test_edges, 3]
        score_matrix = combined_edge_masked[:, 2].reshape(-1, self.config['llm_num'])
        is_correct_matrix = (score_matrix > 0).to(dtype=torch.float32)
        is_correct = is_correct_matrix[torch.arange(len(predicted_llm_idx)), predicted_llm_idx].cpu().numpy()

        # Compute per-task statistics
        from collections import defaultdict
        task_correct = defaultdict(int)
        task_total = defaultdict(int)

        # Find test query indices
        test_query_indices = []
        for i in range(len(mask)):
            if mask[i] and i % self.config['llm_num'] == 0:
                test_query_indices.append(i // self.config['llm_num'])

        # Count correct predictions per task
        for idx, query_idx in enumerate(test_query_indices):
            task_id = query_task_ids[query_idx]
            task_total[task_id] += 1
            if is_correct[idx] == 1:  # Check if prediction is in optimal set
                task_correct[task_id] += 1

        # Compute accuracy per task
        task_accuracy = {}
        for task_id in task_total:
            task_accuracy[task_id] = task_correct[task_id] / task_total[task_id]

        return task_accuracy

    def compute_per_task_cost(self, data, query_task_ids):
        """
        Compute per-task total cost on test set.

        Args:
            data: Test set data
            query_task_ids: List of task_id for each query

        Returns:
            tuple: (total_cost_dict, overall_total_cost)
                - total_cost_dict: {task_id: total_cost}
                - overall_total_cost: sum of all costs
        """
        self.model.eval()
        mask = torch.tensor(data.edge_mask, dtype=torch.bool)
        edge_can_see = self.train_mask  # Can only see training edges (no validation set)

        with torch.no_grad():
            edge_predict = self.model(
                task_id=data.task_id,
                query_features=data.query_features,
                llm_features=data.llm_features,
                edge_index=data.edge_index,
                edge_mask=mask,
                edge_can_see=edge_can_see,
                edge_weight=data.combined_edge
            )

        # Get predicted LLM indices
        edge_predict = edge_predict.reshape(-1, self.config['llm_num'])
        predicted_llm_idx = torch.argmax(edge_predict, 1).cpu().numpy()

        # Prefer USD costs if available, otherwise fall back to normalized cost
        cost_source = 'normalized'
        if hasattr(data, 'cost_usd') and data.cost_usd is not None:
            cost_vec = data.cost_usd[mask]
            cost_source = 'usd'
        else:
            # combined_edge structure: [:, 0]=utility, [:, 1]=cost(norm), [:, 2]=raw_score
            combined_edge_masked = data.combined_edge[mask]
            cost_vec = combined_edge_masked[:, 1]

        cost_matrix = cost_vec.reshape(-1, self.config['llm_num']).cpu().numpy()

        # Compute per-task total cost
        from collections import defaultdict
        task_total_cost = defaultdict(float)

        # Find test query indices
        test_query_indices = []
        for i in range(len(mask)):
            if mask[i] and i % self.config['llm_num'] == 0:
                test_query_indices.append(i // self.config['llm_num'])

        # Accumulate cost per task
        for idx, query_idx in enumerate(test_query_indices):
            task_id = query_task_ids[query_idx]
            predicted_cost = cost_matrix[idx, predicted_llm_idx[idx]]
            task_total_cost[task_id] += predicted_cost

        # Overall total cost
        overall_total_cost = sum(task_total_cost.values())

        return dict(task_total_cost), overall_total_cost, cost_source
