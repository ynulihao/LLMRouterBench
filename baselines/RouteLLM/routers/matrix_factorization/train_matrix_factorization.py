import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import MODEL_IDS


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DEFAULT_CONFIG = {
    "json_path": "",
    "npy_path": "",
    "dim": 128,
    "batch_size": 64,
    "num_epochs": 100,
    "alpha": 0.1,
    "use_proj": True,
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "device": "cuda",
    "save_path": "mf_model.pt",
}


def load_config(config_path: Optional[str]) -> dict:
    config = DEFAULT_CONFIG.copy()
    if not config_path:
        return config

    with open(config_path, "r") as fp:
        user_config = json.load(fp)
    # Only override keys that appear in the default config
    for key, value in user_config.items():
        if key in config and value is not None:
            config[key] = value
    return config

# NOTE: This training script is for the Matrix Factorization approach only.
# Other routing methods are intentionally ignored in this iteration.
# TODO: Polish data pipeline and add checkpoints/save_pretrained if needed.
class PairwiseDataset(Dataset):
    def __init__(self, data):
        self.models_a = torch.tensor(
            [MODEL_IDS[sample["model_a"]] for sample in data], dtype=torch.int64
        )
        self.models_b = torch.tensor(
            [MODEL_IDS[sample["model_b"]] for sample in data], dtype=torch.int64
        )
        self.prompt_id = [sample["idx"] for sample in data]
        self.winners = [sample["winner"] for sample in data]

    def __len__(self):
        return len(self.models_a)

    def __getitem__(self, index):
        assert self.winners[index] in ["model_a", "model_b"], self.winners[index]
        if self.winners[index] == "model_a":
            return self.models_a[index], self.models_b[index], self.prompt_id[index]
        else:
            return self.models_b[index], self.models_a[index], self.prompt_id[index]

    def get_dataloaders(self, batch_size, shuffle=True):
        return DataLoader(self, batch_size, shuffle=shuffle)


class MFModel_Train(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_models: int,
        num_prompts: int,
        text_dim=1536,
        num_classes=1,
        use_proj=True,
        npy_path=None,
    ):
        super().__init__()
        self.use_proj = use_proj
        self.P = torch.nn.Embedding(num_models, dim)
        self.Q = torch.nn.Embedding(num_prompts, dim).requires_grad_(False)
        embeddings = np.load(npy_path)
        self.Q.weight.data.copy_(torch.tensor(embeddings))
        
        if self.use_proj:
            self.text_proj = torch.nn.Linear(text_dim, dim, bias=False)
        else:
            assert (
                text_dim == dim
            ), f"text_dim {text_dim} must be equal to dim {dim} if not using projection"
        
        # bias should be False!
        self.classifier = torch.nn.Linear(dim, num_classes, bias=False)
        
    def get_device(self):
        return self.P.weight.device

    def forward(self, model_win, model_loss, prompt, test=False, alpha=0.05):
        model_win = model_win.to(self.get_device())
        model_loss = model_loss.to(self.get_device())
        prompt = prompt.to(self.get_device())
        
        model_win_embed = self.P(model_win)
        model_win_embed = torch.nn.functional.normalize(model_win_embed, p=2, dim=1)
        model_loss_embed = self.P(model_loss)
        model_loss_embed = torch.nn.functional.normalize(model_loss_embed, p=2, dim=1)
        prompt_embed = self.Q(prompt)
        # Project prompt embedding to model-embedding space if enabled.
        if self.use_proj:
            prompt_embed = self.text_proj(prompt_embed)
        
        return self.classifier((model_win_embed - model_loss_embed) * prompt_embed).squeeze()
    
    @torch.no_grad()
    def predict(self, model_win, model_loss, prompt):
        logits = self.forward(model_win, model_loss, prompt, test=True)
        return logits > 0
    
def evaluator(net, test_iter, device):
    net.eval()
    ls_fn = nn.BCEWithLogitsLoss(reduction="sum")
    ls_list = []
    correct = 0
    num_samples = 0
    with torch.no_grad():
        for models_a, models_b, prompts in test_iter:
            # Assuming devices refer to potential GPU usage
            models_a = models_a.to(device)
            models_b = models_b.to(device)
            prompts = prompts.to(device)

            logits = net(models_a, models_b, prompts)
            labels = torch.ones_like(logits)
            loss = ls_fn(logits, labels)  # Calculate the loss
            pred_labels = net.predict(models_a, models_b, prompts)

            # update eval stats
            correct += (pred_labels == labels).sum().item()
            ls_list.append(loss.item())
            num_samples += labels.shape[0]

    net.train()
    return float(sum(ls_list) / num_samples), correct / num_samples

def train_loops(
    net,
    train_iter,
    test_iter,
    lr,
    weight_decay,
    alpha,
    num_epochs,
    device="cuda",
    evaluator=evaluator,
    **kwargs,
):
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss = nn.BCEWithLogitsLoss(reduction="mean")
    
    def train_epoch():  # Inner function for one epoch of training
        net.train()  # Set the model to training mode
        train_loss_sum, n = 0.0, 0
        for models_a, models_b, prompts in train_iter:
            # Assuming devices refer to potential GPU usage
            models_a = models_a.to(device)
            models_b = models_b.to(device)
            prompts = prompts.to(device)

            output = net(models_a, models_b, prompts, alpha=alpha)
            ls = loss(output, torch.ones_like(output))

            optimizer.zero_grad()
            ls.backward()
            optimizer.step()

            train_loss_sum += ls.item() * len(models_a)
            n += len(models_a)
        return train_loss_sum / n
    
    train_losses = []
    test_losses = []
    test_acces = []
    best_test_acc = -1
    progress_bar = tqdm(total=num_epochs)
    
    best_state_dict = None

    for epoch in range(num_epochs):
        train_ls = train_epoch()
        train_losses.append(train_ls)
        info = {"train_loss": train_ls, "epoch": epoch}

        if evaluator:
            test_ls, test_acc = evaluator(net, test_iter, device)
            test_losses.append(test_ls)
            test_acces.append(test_acc)
            info.update(
                {
                    "test_loss": test_ls,
                    "test_acc": test_acc,
                    "epoch": epoch,
                    "best_test_acc": best_test_acc,
                    "best_test_loss": min(test_losses),
                }
            )
        else:
            test_ls = None  # No evaluation

        if evaluator and test_acc > best_test_acc:
            best_test_acc = test_acc
            # Keep a CPU copy of the best-performing weights.
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in net.state_dict().items()
            }

        progress_bar.set_postfix(**info)
        progress_bar.update(1)

    progress_bar.close()
    return best_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Matrix Factorization router from pairwise data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config file overriding training defaults",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if not config["json_path"] or not config["npy_path"]:
        raise ValueError(
            "json_path and npy_path must be specified via the config file or defaults."
        )

    json_path = Path(config["json_path"]).expanduser()
    npy_path = Path(config["npy_path"]).expanduser()

    with open(json_path, "r") as fp:
        data = json.load(fp)

    embeddings = np.load(npy_path)
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embeddings array from {npy_path}, got shape {embeddings.shape}"
        )
    num_prompts_from_embeddings, embedding_dim = embeddings.shape

    filtered_data = [
        sample
        for sample in data
        if sample["winner"] in ["model_a", "model_b"]
        and sample["model_a"] != sample["model_b"]
    ]

    data_shuffled = filtered_data.copy()
    random.shuffle(data_shuffled)
    train_data = data_shuffled[: int(len(data_shuffled) * 0.95)]
    test_data = data_shuffled[int(len(data_shuffled) * 0.95) :]

    train_dataset = PairwiseDataset(train_data)
    test_dataset = PairwiseDataset(test_data)

    train_data_loader = train_dataset.get_dataloaders(
        batch_size=config["batch_size"], shuffle=True
    )
    test_data_loader = test_dataset.get_dataloaders(1024, shuffle=False)

    # Determine number of distinct prompts from dataset indices.
    all_prompt_ids = [sample["idx"] for sample in data]
    if not all_prompt_ids:
        raise ValueError("No prompts found in data; idx field must be present.")
    max_idx = max(all_prompt_ids)
    if max_idx >= num_prompts_from_embeddings:
        raise ValueError(
            f"Prompt idx {max_idx} exceeds embedding table size {num_prompts_from_embeddings}."
        )
    num_prompts = num_prompts_from_embeddings

    model = MFModel_Train(
        dim=config["dim"],
        num_models=len(MODEL_IDS),
        num_prompts=num_prompts,
        text_dim=embedding_dim,
        num_classes=1,
        use_proj=config["use_proj"],
        npy_path=npy_path,
    ).to(config["device"])

    best_state_dict = train_loops(
        model,
        train_data_loader,
        test_data_loader,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        alpha=config["alpha"],
        num_epochs=config["num_epochs"],
        device=config["device"],
    )

    save_path = Path(config["save_path"]).expanduser()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state_to_save = best_state_dict or {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
    }
    torch.save(state_to_save, save_path)
    print(f"Saved MF model weights to {save_path}")
