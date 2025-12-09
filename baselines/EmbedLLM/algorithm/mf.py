import argparse
import random
import torch
import pandas as pd
import numpy as np
import time
import wandb

from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import Adam
from tqdm import tqdm

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Formatting helper functions for better output readability
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

class TextMF(nn.Module):
    def __init__(self, question_embeddings, model_embedding_dim, alpha, num_models, num_prompts, text_dim=3584, num_classes=2):
        super(TextMF, self).__init__()
        # Model embedding network
        self.P = nn.Embedding(num_models, model_embedding_dim)

        # Question embedding network
        self.Q = nn.Embedding(num_prompts, text_dim).requires_grad_(False)
        self.Q.weight.data.copy_(question_embeddings)
        self.text_proj = nn.Linear(text_dim, model_embedding_dim)

        # Noise/Regularization level
        self.alpha = alpha
        self.classifier = nn.Linear(model_embedding_dim, num_classes)

    def forward(self, model, prompt, test_mode=False):
        p = self.P(model)
        q = self.Q(prompt)
        if not test_mode:
            # Adding a small amount of noise in question embedding to reduce overfitting
            q += torch.randn_like(q) * self.alpha
        q = self.text_proj(q)
        return self.classifier(p * q)
    
    @torch.no_grad()
    def predict(self, model, prompt):
        logits = self.forward(model, prompt, test_mode=True) # During inference no noise is applied
        return torch.argmax(logits, dim=1)
    
# Helper functions to load and process the data into desired format needed for MF
# For MF we need a "model ID" either in the form of name or index and so we use the tabular data instead of tensors
def load_and_process_data(train_data, test_data, batch_size=64):
    # NOTE: Due to the nature of the embedding layer we need to take max prompt ID from both train and test data
    # But during training we won't be using test question
    num_prompts = int(max(max(train_data["prompt_id"]), max(test_data["prompt_id"]))) + 1
    class CustomDataset(Dataset):
        def __init__(self, data):
            model_ids = torch.tensor(data["model_id"], dtype=torch.int64)
            unique_ids, inverse_indices = torch.unique(model_ids, sorted=True, return_inverse=True)
            id_to_rank = {id.item(): rank for rank, id in enumerate(unique_ids)}
            ranked_model_ids = torch.tensor([id_to_rank[id.item()] for id in model_ids])
            self.models = ranked_model_ids
            self.prompts = torch.tensor(data["prompt_id"], dtype=torch.int64)
            self.labels = torch.tensor(data["label"], dtype=torch.int64)
            self.num_models = len(data["model_id"].unique())
            self.num_prompts = num_prompts
            self.num_classes = len(data["label"].unique())

        def get_num_models(self):
            return self.num_models

        def get_num_prompts(self):
            return self.num_prompts

        def get_num_classes(self):
            return self.num_classes

        def __len__(self):
            return len(self.models)

        def __getitem__(self, index):
            return self.models[index], self.prompts[index], self.labels[index]

        def get_dataloaders(self, batch_size):
            return DataLoader(self, batch_size, shuffle=False)

    train_dataset = CustomDataset(train_data)
    test_dataset = CustomDataset(test_data)

    train_loader = train_dataset.get_dataloaders(batch_size)
    test_loader = test_dataset.get_dataloaders(batch_size)

    return train_loader, test_loader
    
def correctness_prediction_evaluator(net, test_loader, device):
    """Standard evaluator for correctness prediction"""
    net.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0
    correct = 0
    num_samples = 0

    with torch.no_grad():
        for models, prompts, labels in test_loader:
            models, prompts, labels = models.to(device), prompts.to(device), labels.to(device)
            logits = net(models, prompts)
            loss = loss_fn(logits, labels)
            pred_labels = net.predict(models, prompts)
            correct += (pred_labels == labels).sum().item()
            total_loss += loss.item()
            num_samples += labels.shape[0]

    mean_loss = total_loss / num_samples
    accuracy = correct / num_samples
    net.train()
    return mean_loss, accuracy

def evaluate(net, test_loader, device, eval_mode="correctness", acc_dict=None, model_num=112, category_names=None, model_names=None, ood_keywords=None):
    """Unified evaluation function that routes to specific evaluator based on mode"""
    if eval_mode == "correctness":
        return correctness_prediction_evaluator(net, test_loader, device)
    elif eval_mode == "router":
        return evaluator_router(net, test_loader, [device], acc_dict, model_num, category_names, model_names, ood_keywords)
    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")

def evaluator_router(net, test_iter, devices, acc_dict, model_num=112, category_names=None, model_names=None, ood_keywords=None):
    start_time = time.time()
    net.eval()

    # 正常的路由评估
    category_successes = {}  # {category: successful_routes}
    category_counts = {}     # {category: total_prompts}
    model_counts = [0] * model_num
    correctness_result = {}

    with torch.no_grad():
        for prompts, models, labels, categories in test_iter:
            # 移动到设备
            prompts = prompts.to(devices[0])
            models = models.to(devices[0])
            labels = labels.to(devices[0])
            categories = categories.to(devices[0])
            
            # 获取当前batch的prompt_id和category
            prompt_id = prompts[0].item()
            category = categories[0].item()
            
            # 对整个batch进行一次性预测
            logits = net(models, prompts)
            logit_diff = logits[:, 1] - logits[:, 0]
            max_index = torch.argmax(logit_diff)
            selected_model = max_index.item()
            
            # 记录选择的模型
            model_counts[selected_model] += 1
            
            # 检查路由是否成功
            is_successful = int(labels[max_index] == 1)
            
            # 更新category统计
            if category not in category_successes:
                category_successes[category] = 0
                category_counts[category] = 0
            
            category_counts[category] += 1
            if is_successful:
                category_successes[category] += 1
            
            # 记录结果
            correctness_result[prompt_id] = is_successful

    # Calculate per-category accuracies
    category_accuracies = {}
    category_metrics = {}

    print_minor_header("Dataset Accuracies")

    for category in sorted(category_counts.keys()):
        acc = float(category_successes[category] / category_counts[category])
        category_accuracies[category] = acc

        # Use category name if available
        if category_names and category in category_names:
            category_name = category_names[category]
        else:
            category_name = f"Category {category}"

        # Display per-dataset accuracy
        print(f"{category_name:20} {acc:.4f} ({category_successes[category]}/{category_counts[category]})")

        # Prepare for wandb logging
        category_metrics[f"category/{category_name}"] = acc

    # Calculate the three core metrics
    total_successes = sum(category_successes.values())
    total_prompts = sum(category_counts.values())

    # 1. Sample-level average accuracy (weighted by sample count)
    sample_level_accuracy = float(total_successes / total_prompts)

    # 2. Dataset-level average accuracy (unweighted - each dataset counts equally)
    dataset_level_accuracy = sum(category_accuracies.values()) / len(category_accuracies) if category_accuracies else 0

    print_minor_header("Summary Metrics")

    # Display the core metrics clearly
    print(f"  Dataset-Level Average Accuracy: {dataset_level_accuracy:.4f} (unweighted)")
    print(f"  Sample-Level Average Accuracy:  {sample_level_accuracy:.4f} (weighted)")
    print(f"  Total Samples: {total_prompts} | Correct: {total_successes}")

    # Calculate model routing distribution
    print_minor_header("Model Routing Distribution")
    
    total_prompts = sum(model_counts)
    model_percentages = {}
    
    # Sort models by routing frequency
    sorted_models = sorted([(model_id, count) for model_id, count in enumerate(model_counts) if count > 0], 
                           key=lambda x: x[1], reverse=True)
    
    # Display and collect the top routed models
    for model_id, count in sorted_models[:10]:  # Show top 10 most routed models
        percentage = (count / total_prompts) * 100
        # Use model name instead of ID
        model_name = model_names.get(model_id, f"Model {model_id}") if model_names else f"Model {model_id}"
        # For wandb, use consistent model_id for tracking, but more descriptive name in display
        model_percentages[f"routing/model_{model_id}"] = percentage
        print(f"{model_name}: {count} prompts ({percentage:.2f}%)")
    
    # Prepare metrics for wandb logging
    summary_metrics = {
        "performance/dataset_avg_accuracy": dataset_level_accuracy,
        "performance/sample_avg_accuracy": sample_level_accuracy,
    }

    # Add routing distribution to metrics
    routing_metrics = {
        "routing/total_models_used": len([c for c in model_counts if c > 0]),
        "routing/top_model_percentage": model_percentages.get(f"routing/model_{sorted_models[0][0]}", 0) if sorted_models else 0,
    }

    # Combine all metrics
    all_metrics = {**summary_metrics, **category_metrics, **model_percentages, **routing_metrics}

    net.train()
    end_time = time.time()
    print(f"Time used to route {total_prompts} questions: {end_time - start_time:.2f}s")
    
    # Return dataset-level accuracy as the main metric for best model selection
    return dataset_level_accuracy, sample_level_accuracy, correctness_result, model_counts, category_accuracies, all_metrics

def create_router_dataloader(original_dataloader, test_data_path=None):
    """
    Transform a standard dataloader into a router-compatible dataloader.
    Returns:
    - router_dataloader: DataLoader with (prompt, models, labels, categories) batches
    - label_dict: Dictionary mapping {prompt_id: {model_id: label}}
    - acc_dict: Dictionary mapping {model_id: accuracy}
    - category_names: Dictionary mapping {category_id: category_name}
    - model_names: Dictionary mapping {model_id: model_name}
    """
    # Concatenate all batches
    all_models, all_prompts, all_labels = [], [], []
    for models, prompts, labels in original_dataloader:
        all_models.append(models)
        all_prompts.append(prompts)
        all_labels.append(labels)
    
    all_models = torch.cat(all_models)
    all_prompts = torch.cat(all_prompts)
    all_labels = torch.cat(all_labels)

    # Create label dictionary
    label_dict = {}
    for i in range(len(all_prompts)):
        prompt_id = int(all_prompts[i])
        model_id = int(all_models[i])
        label = int(all_labels[i])
        
        if prompt_id not in label_dict:
            label_dict[prompt_id] = {}
        label_dict[prompt_id][model_id] = label

    # Get unique prompts and models
    unique_prompts = sorted(set(all_prompts.tolist()))
    unique_models = sorted(set(all_models.tolist()))
    model_num = len(unique_models)
    
    # Get category information and model names from test data file if provided
    category_dict = {}
    category_names = {}
    model_names = {}  # New dict for model names
    
    if test_data_path:
        try:
            test_df = pd.read_csv(test_data_path)
            
            # First, handle the category mapping
            if 'category' in test_df.columns:
                # Get unique category names
                unique_categories = sorted(test_df['category'].unique())
                print(f"Raw unique categories: {unique_categories}")
                
                # Create a new ID for each unique category name
                name_to_new_id = {name: idx for idx, name in enumerate(unique_categories)}
                
                # Create the reverse mapping for output
                category_names = {idx: name for idx, name in enumerate(unique_categories)}
                
                # Now map prompt_id to the new category_id
                if 'prompt_id' in test_df.columns:
                    for _, row in test_df.iterrows():
                        prompt_id = int(row['prompt_id'])
                        # Use the category name to get the new consistent ID
                        category_id = name_to_new_id[row['category']]
                        category_dict[prompt_id] = category_id
            
            # If no 'category' column but has 'category_id'
            elif 'prompt_id' in test_df.columns and 'category_id' in test_df.columns:
                for _, row in test_df.iterrows():
                    prompt_id = int(row['prompt_id'])
                    category_id = int(row['category_id'])
                    category_dict[prompt_id] = category_id
                    # Without category names, just use the IDs
                    category_names[category_id] = f"Category {category_id}"
            
            # Get model_names if available
            if 'model_id' in test_df.columns and 'model_name' in test_df.columns:
                for _, row in test_df.iterrows():
                    model_id = int(row['model_id'])
                    model_name = row['model_name']
                    model_names[model_id] = model_name
                    
            print(f"Loaded categories for {len(category_dict)} prompts from {test_data_path}")
            print(f"Found {len(category_names)} unique categories: {', '.join(category_names.values())}")
            if model_names:
                print(f"Loaded {len(model_names)} model names")
        except Exception as e:
            print(f"Warning: Could not load categories from {test_data_path}: {e}")
            print("Will use zero categories as fallback")

    # Build router dataloader content
    new_models, new_prompts, new_labels, new_categories = [], [], [], []
    for prompt_id in unique_prompts:
        prompt_tensor = torch.tensor([prompt_id] * model_num)
        model_tensor = torch.tensor(unique_models)
        label_tensor = torch.tensor([label_dict[prompt_id].get(model_id, 0) for model_id in unique_models])
        
        # Use category from dictionary or 0 as default
        category_id = category_dict.get(prompt_id, 0)
        category_tensor = torch.tensor([category_id] * model_num)
        
        new_prompts.append(prompt_tensor)
        new_models.append(model_tensor)
        new_labels.append(label_tensor)
        new_categories.append(category_tensor)

    # Concatenate tensors
    new_prompts = torch.cat(new_prompts)
    new_models = torch.cat(new_models)
    new_labels = torch.cat(new_labels)
    new_categories = torch.cat(new_categories)

    # Create router dataloader
    router_dataset = TensorDataset(new_prompts, new_models, new_labels, new_categories)
    router_dataloader = DataLoader(router_dataset, batch_size=model_num, shuffle=False)

    # Compute accuracy for each model
    acc_dict = {}
    for model_id in range(model_num):
        correct = sum(label_dict[p].get(model_id, 0) for p in label_dict)
        total = sum(1 for p in label_dict if model_id in label_dict[p])
        acc_dict[model_id] = correct / total if total > 0 else 0.0

    return router_dataloader, label_dict, acc_dict, category_names, model_names

# Main training loop
def train(net, train_loader, test_loader, num_epochs, lr, device, eval_mode="correctness", 
          acc_dict=None, model_num=112, weight_decay=1e-5, save_path=None, category_names=None, model_names=None, ood_keywords=None):
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    progress_bar = tqdm(total=num_epochs)
    
    # Track best test accuracy
    best_test_accuracy = 0.0
    best_metrics = {}
    
    # Create router train loader if needed for evaluation
    train_router_loader = None
    if eval_mode == "router":
        # Create router dataloader for train set evaluation with proper category mapping
        train_router_loader, train_label_dict, train_acc_dict, train_category_names, train_model_names = create_router_dataloader(train_loader, args.train_data_path)

    for epoch in range(num_epochs):
        net.train()
        total_loss = 0
        for models, prompts, labels in train_loader:
            models, prompts, labels = models.to(device), prompts.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = net(models, prompts)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}")
        
        # Initialize metrics dict for wandb
        metrics = {"train/loss": train_loss, "epoch": epoch + 1}

        # Evaluate on training set first
        print_major_header("TRAINING SET EVALUATION")

        if eval_mode == "correctness":
            train_eval_results = correctness_prediction_evaluator(net, train_loader, device)
            train_loss, train_accuracy = train_eval_results
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            
            # Add train correctness metrics
            metrics.update({
                "train/eval_loss": train_loss,
                "train/accuracy": train_accuracy
            })
        else:  # router mode
            train_eval_results = evaluator_router(net, train_router_loader, [device], train_acc_dict, model_num, category_names, model_names, ood_keywords)
            train_dataset_avg_acc, train_sample_avg_acc, _, _, train_category_accuracies, train_wandb_metrics = train_eval_results
            print(f"Train Dataset-Level Accuracy: {train_dataset_avg_acc:.4f}")
            print(f"Train Sample-Level Accuracy: {train_sample_avg_acc:.4f}")

            # Add train router metrics with prefix
            train_metrics = {f"train/{k.split('/', 1)[1] if '/' in k else k}": v for k, v in train_wandb_metrics.items()}
            metrics.update(train_metrics)

        # Now evaluate on test set
        print_major_header("TEST SET EVALUATION")
        
        if eval_mode == "correctness":
            eval_results = correctness_prediction_evaluator(net, test_loader, device)
            test_loss, test_accuracy = eval_results
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            progress_bar.set_postfix(train_loss=train_loss, test_loss=test_loss, test_acc=test_accuracy)
            
            # Add correctness metrics
            metrics.update({
                "test/loss": test_loss,
                "test/accuracy": test_accuracy
            })
            
            # Track best test accuracy
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_metrics = {
                    "best/epoch": epoch + 1,
                    "best/test_accuracy": test_accuracy,
                    "best/train_accuracy": train_accuracy
                }
                
                # Save best model
                if save_path:
                    best_model_path = save_path.replace(".pth", "_best.pth")
                    torch.save(net.state_dict(), best_model_path)
                    print(f"Best model saved to {best_model_path}")
                
                # Add best metrics to the metrics dictionary
                metrics.update(best_metrics)
            else:
                # 即使不是最佳模型，也使用上一次记录的best_metrics，确保曲线连续
                # 只更新epoch编号
                temp_best_metrics = best_metrics.copy()
                temp_best_metrics["best/epoch"] = epoch + 1
                # 添加到metrics中，但不更新best_metrics本身（保留最佳值）
                metrics.update(temp_best_metrics)
        else:  # router mode
            eval_results = evaluator_router(net, test_loader, [device], acc_dict, model_num, category_names, model_names, ood_keywords)
            dataset_avg_acc, sample_avg_acc, correctness_result, model_counts, category_accuracies, wandb_metrics = eval_results
            print(f"Dataset-Level Average Accuracy: {dataset_avg_acc:.4f}")
            print(f"Sample-Level Average Accuracy: {sample_avg_acc:.4f}")
            progress_bar.set_postfix(train_loss=train_loss, dataset_acc=dataset_avg_acc)

            # Add all wandb_metrics to metrics
            metrics.update(wandb_metrics)
            
            # Track best test accuracy using dataset-level average accuracy
            if dataset_avg_acc > best_test_accuracy:
                best_test_accuracy = dataset_avg_acc

                # Create a copy of the best metrics
                best_metrics = {"best/epoch": epoch + 1}

                # Add all metrics from wandb_metrics
                for key, value in wandb_metrics.items():
                    if key.startswith("category/") or key.startswith("performance/"):
                        # Keep the original metric namespace (e.g., "category/aime")
                        # so we can reliably parse category metrics later.
                        best_metrics[f"best/{key}"] = value

                # Add the core metrics
                best_metrics.update({
                    "best/dataset_avg_accuracy": dataset_avg_acc,
                    "best/sample_avg_accuracy": sample_avg_acc
                })
                
                # Save best model
                if save_path:
                    best_model_path = save_path.replace(".pth", "_best.pth")
                    torch.save(net.state_dict(), best_model_path)
                    print(f"Best model saved to {best_model_path}")
                
                # Add best metrics to main metrics dictionary
                metrics.update(best_metrics)
            else:
                # 即使不是最佳模型，也使用上一次记录的best_metrics，确保曲线连续
                # 只更新epoch编号
                temp_best_metrics = best_metrics.copy()
                temp_best_metrics["best/epoch"] = epoch + 1
                # 添加到metrics中，但不更新best_metrics本身（保留最佳值）
                metrics.update(temp_best_metrics)
                
        # Log best metrics on each epoch to track progress
        metrics["best_accuracy"] = best_test_accuracy
        
        # Log metrics to wandb - using the epoch as the step to ensure only one step per epoch
        wandb.log(metrics, step=epoch)
            
        progress_bar.update(1)
    
    # Log best metrics at the end of training
    print_major_header("⭐ BEST PERFORMANCE")
    print(f"\n  Best Dataset-Level Accuracy: {best_test_accuracy:.4f} at Epoch {best_metrics.get('best/epoch', 0)}")

    # Print detailed breakdown of best model performance in router mode
    if eval_mode == "router" and best_metrics:
        # Core performance metrics
        print_minor_header("Overall Performance", indent=2)

        dataset_avg_acc = best_metrics.get('best/dataset_avg_accuracy', None)
        sample_avg_acc = best_metrics.get('best/sample_avg_accuracy', None)

        if dataset_avg_acc is not None:
            print(f"    Dataset-Level Average Accuracy:    {dataset_avg_acc:.4f} (unweighted)")
        if sample_avg_acc is not None:
            print(f"    Sample-Level Average Accuracy:     {sample_avg_acc:.4f} (weighted)")

        # Per-dataset accuracies
        print_minor_header("Per-Dataset Accuracies", indent=2)

        # Collect category accuracies
        category_accs = []

        # Keywords to exclude from category list
        exclude_keywords = [
            "epoch", "dataset_avg_accuracy", "sample_avg_accuracy"
        ]

        for key, value in best_metrics.items():
            if not key.startswith("best/"):
                continue

            # Remove "best/" prefix
            metric_name = key[5:]

            # Skip non-category or core metrics
            if metric_name in exclude_keywords or not metric_name.startswith("category/"):
                continue

            # Remove "category/" prefix for display
            display_name = metric_name.replace("category/", "")
            category_accs.append((display_name, value))

        # Sort by dataset name (alphabetical order)
        category_accs.sort(key=lambda x: x[0])

        # Print categories
        for category_name, acc in category_accs:
            print(f"    {category_name:<35} {acc:.4f}")


    # Make sure best metrics are in the wandb summary
    for key, value in best_metrics.items():
        wandb.run.summary[key] = value
    
    # Record category accuracies in wandb summary
    if eval_mode == "router":
        print_major_header("FINAL TEST SET RESULTS")

        # Re-evaluate on the test set with the final model to get
        # a complete report (dataset/sample metrics + routing distribution).
        final_test_results = evaluator_router(
            net, test_loader, [device], acc_dict, model_num, category_names, model_names, ood_keywords
        )
        final_test_dataset_avg_acc, final_test_sample_avg_acc, _, _, final_test_category_accuracies, _ = final_test_results

        # Record final test set metrics to wandb summary
        for category, accuracy in final_test_category_accuracies.items():
            category_name = category_names[category] if category_names and category in category_names else f"Category {category}"
            wandb.run.summary[f"test_category/{category_name}"] = accuracy

        wandb.run.summary["test_dataset_avg_accuracy"] = final_test_dataset_avg_acc
        wandb.run.summary["test_sample_avg_accuracy"] = final_test_sample_avg_acc
        
        # Now record training set category accuracies
        print_major_header("FINAL TRAINING SET RESULTS")

        # Run the evaluator again to get final training set results
        train_eval_results = evaluator_router(net, train_router_loader, [device], train_acc_dict, model_num, category_names, model_names, ood_keywords)
        final_train_dataset_avg_acc, final_train_sample_avg_acc, _, _, final_train_category_accuracies, _ = train_eval_results

        print_minor_header("Dataset Accuracies")
        for category, accuracy in final_train_category_accuracies.items():
            category_name = category_names[category] if category_names and category in category_names else f"Category {category}"
            print(f"  {category_name}: {accuracy:.4f}")
            # Add to wandb summary
            wandb.run.summary[f"train_category/{category_name}"] = accuracy

        print_minor_header("Summary Metrics")
        print(f"  Dataset-Level Average Accuracy: {final_train_dataset_avg_acc:.4f} (unweighted)")
        print(f"  Sample-Level Average Accuracy:  {final_train_sample_avg_acc:.4f} (weighted)")

        # Add training set accuracies to wandb summary
        wandb.run.summary["train_dataset_avg_accuracy"] = final_train_dataset_avg_acc
        wandb.run.summary["train_sample_avg_accuracy"] = final_train_sample_avg_acc
    
    if save_path:
        torch.save(net.state_dict(), save_path)
        print(f"Final model saved to {save_path}")

def load_model(net, path, device):
    net.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-epochs", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-data-path", type=str, default="../data/train.csv")
    parser.add_argument("--test-data-path", type=str, default="../data/test.csv")
    parser.add_argument("--question-embedding-path", type=str, default="../data/question_embeddings.pth")
    parser.add_argument("--embedding-save-path", type=str, default="../data/model_embeddings.pth")
    parser.add_argument("--model-save-path", type=str, default="../data/saved_model.pth")
    parser.add_argument("--model-load-path", type=str, default=None)
    parser.add_argument("--eval-mode", type=str, default="correctness", 
                       choices=["correctness", "router"],
                       help="Evaluation mode: correctness or router")
    parser.add_argument("--model-num", type=int, default=112,
                       help="Number of models for router evaluation")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                       help="Weights & Biases run name")
    parser.add_argument("--ood-datasets", type=str, default="",
                       help="Comma-separated list of OOD dataset keywords (default: "")")
    args = parser.parse_args()
    
    # Parse OOD datasets
    ood_keywords = []
    if args.ood_datasets:
        ood_keywords = [d.strip().lower() for d in args.ood_datasets.split(',') if d.strip()]

    # Initialize wandb
    run_name = args.wandb_run_name or f"embedLLM-{time.strftime('%Y%m%d-%H%M%S')}"
    wandb.init(
        project="embedLLM",
        name=run_name,
        config={
            "embedding_dim": args.embedding_dim,
            "alpha": args.alpha,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": 1e-5,
            "eval_mode": args.eval_mode,
            "model_num": args.model_num,
        }
    )

    print("Loading dataset...")
    train_data = pd.read_csv(args.train_data_path)
    test_data = pd.read_csv(args.test_data_path)
    question_embeddings = torch.load(args.question_embedding_path)
    num_prompts = question_embeddings.shape[0]
    num_models = len(test_data["model_id"].unique())
    model_names = list(np.unique(list(test_data["model_name"])))

    train_loader, test_loader = load_and_process_data(train_data, test_data, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Initializing model...")
    model = TextMF(question_embeddings=question_embeddings,
                   model_embedding_dim=args.embedding_dim, alpha=args.alpha,
                   num_models=num_models, num_prompts=num_prompts)
    model = model.to(device)

    if args.model_load_path:
        model.load_state_dict(torch.load(args.model_load_path, map_location=device))
        print(f"Model loaded from {args.model_load_path}")

    print("Training model...")
    # Transform test_loader for router mode if needed
    category_names = None
    model_names_dict = None
    if args.eval_mode == "router":
        test_loader, label_dict, acc_dict, category_names, model_names_dict = create_router_dataloader(test_loader, args.test_data_path)
        
        # If model_names_dict is empty, create a mapping from model names list
        if not model_names_dict and model_names:
            print("Creating model name dictionary from model_names list")
            model_names_dict = {i: name for i, name in enumerate(model_names)}
    else:
        acc_dict = None

    train(model, train_loader, test_loader, 
          num_epochs=args.num_epochs, 
          lr=args.learning_rate,
          device=device, 
          eval_mode=args.eval_mode,
          acc_dict=acc_dict,
          model_num=args.model_num,
          save_path=args.model_save_path,
          category_names=category_names,
          model_names=model_names_dict,
          ood_keywords=ood_keywords)
    if args.embedding_save_path:
        torch.save(model.P.weight.detach().to("cpu"), args.embedding_save_path) # Save model embeddings if needed
    if args.model_save_path:
        torch.save(model.state_dict(), args.model_save_path)
        print(f"Model saved to {args.model_save_path}")
    
    # Finish wandb run
    wandb.finish()
