#!/usr/bin/env python3
"""
Fine-tune a local encoder model (HuggingFace format) as a FrugalGPT-compatible scorer
using LLMRouterBench result JSON files (prompt, cost, score, prediction/raw_output).

What it does
- Recursively loads JSON files from --inputs (dirs or files)
- Builds training texts matching FrugalGPT scorer_text (final 'Q:' segment + answer)
- Labels a sample as 1 if JSON 'score' >= --score-threshold else 0
- Loads a local base encoder from --local-base. Supports both BERT-style sequence classification models and pooled-embedding LLMs (e.g., gte_Qwen2-7B-Instruct), prefers provided sentence embeddings when available, and can run with DeepSpeed for multi-GPU training. Supports fp16/bf16 mixed precision control via CLI.
- Fine-tunes a sequence classification head (num_labels=2) on your data
- Evaluates on a test split and prints accuracy/ROC-AUC and per-dataset accuracy
- Optionally saves the fine-tuned model + tokenizer to --output-dir

Dependencies
- pandas, numpy, scikit-learn, torch, transformers
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from scipy import optimize

# Make original FrugalGPT optimizer-style logic available without importing missing "service" deps
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# --------------------------
# Data loading and labeling
# --------------------------

def iter_result_files(inputs: Iterable[str]) -> Iterator[Path]:
    """Yield every JSON file found under the given paths."""
    for target in inputs:
        path = Path(target).expanduser()
        if path.is_dir():
            yield from sorted(path.rglob("*.json"))
        elif path.is_file() and path.suffix.lower() == ".json":
            yield path
        else:
            raise FileNotFoundError(f"Input path does not exist or is not JSON: {path}")


def infer_metadata(file_path: Path, payload: Dict) -> Dict[str, Optional[str]]:
    """Infer dataset/model metadata from JSON payload or directory layout."""
    model_name = payload.get("model_name") or file_path.parent.name
    dataset_name = payload.get("dataset_name")
    split = payload.get("split")

    parts = file_path.parts
    if dataset_name is None and "bench" in parts:
        i = parts.index("bench")
        if i + 1 < len(parts):
            dataset_name = parts[i + 1]
    if split is None and "bench" in parts:
        i = parts.index("bench")
        if i + 2 < len(parts):
            split = parts[i + 2]

    return {"model_name": model_name, "dataset_name": dataset_name, "split": split}


def scorer_text_like(text: str) -> str:
    """Replicate FrugalGPT's scorer_text preprocessing."""
    if not text:
        return ""
    parts = text.split("Q:")
    if len(parts) <= 1:
        return text.strip()
    return ("Q:" + parts[-1]).strip()


def build_text(record: Dict) -> str:
    """Build the text input following FrugalGPT's scorer preprocessing."""
    prompt = record.get("prompt") or record.get("origin_query") or record.get("query") or ""
    answer = record.get("prediction")
    if not answer:
        answer = record.get("raw_output", "")
    combined = f"{prompt} {answer}".strip()
    return scorer_text_like(combined)


def load_records(files: Iterable[Path]) -> pd.DataFrame:
    """Load JSON result files and return a DataFrame with text/label/metadata."""
    rows: List[Dict] = []
    for jf in files:
        try:
            payload = json.loads(jf.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON: {jf}") from exc

        meta = infer_metadata(jf, payload if isinstance(payload, dict) else {})

        if isinstance(payload, dict):
            record_list = payload.get("records", [])
        elif isinstance(payload, list):
            record_list = payload
        else:
            record_list = []

        for rec in record_list:
            text = build_text(rec)
            score = rec.get("score")
            cost = rec.get("cost")
            if not text or score is None or cost is None:
                continue

            record_index = rec.get("index")
            if record_index is None:
                record_index = rec.get("_id") or rec.get("id")

            dataset_name = meta["dataset_name"] or rec.get("dataset_name") or "unknown"
            model_name = meta["model_name"] or rec.get("model_name") or "unknown"
            split = meta["split"] or rec.get("split")

            if record_index is not None:
                sample_id = f"{dataset_name}::{record_index}"
            else:
                sample_id = f"{dataset_name}::{Path(jf).stem}:{len(rows)}"

            rows.append(
                {
                    "path": str(jf),
                    "record_index": record_index,
                    "sample_id": str(sample_id),
                    "text": text,
                    "score": float(score),
                    "cost": float(cost),
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "split": split,
                }
            )

    if not rows:
        raise ValueError("No valid records were found in the supplied inputs.")

    return pd.DataFrame(rows)


def load_jsonl_split(path: Path, split: str) -> pd.DataFrame:
    """Load LLMRouterBench JSONL split where each line stores per-model records."""
    path = Path(path).expanduser()
    rows: List[Dict] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSONL line {line_no} in {path}") from exc

            dataset_name = payload.get("dataset") or "unknown"
            record_index = payload.get("index")
            query = payload.get("query") or ""
            records = payload.get("records") or {}
            usages = payload.get("usages") or {}

            text_value = scorer_text_like(query)

            for model_name, score in records.items():
                usage = usages.get(model_name) or {}
                cost = usage.get("cost")
                if score is None or cost is None:
                    continue

                if record_index is not None:
                    sample_id = f"{dataset_name}::{record_index}"
                else:
                    sample_id = f"{dataset_name}::{path.stem}:{line_no}"

                rows.append(
                    {
                        "path": str(path),
                        "record_index": record_index,
                        "sample_id": str(sample_id),
                        "text": text_value,
                        "score": float(score),
                        "cost": float(cost),
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "split": split,
                    }
                )

    if not rows:
        raise ValueError(f"No valid records were found in {path}")

    return pd.DataFrame(rows)


# --------------------------
# Dataset and model helpers
# --------------------------

class TextClsDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def set_seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PooledEncoderForSequenceClassification(nn.Module):
    SUPPORTED_POOLING = {"cls", "mean", "last-token"}

    def __init__(
        self,
        base_model_name_or_path: str,
        num_labels: int = 2,
        pooling: str = "cls",
        trust_remote_code: bool = False,
    ):
        super().__init__()
        pooling = pooling.lower()
        if pooling not in self.SUPPORTED_POOLING:
            raise ValueError(f"Unsupported pooling strategy: {pooling}")
        self.pooling = pooling
        self.config = AutoConfig.from_pretrained(
            str(base_model_name_or_path),
            trust_remote_code=trust_remote_code,
        )
        self.config.num_labels = num_labels
        self.config.router_pooling = pooling
        architectures = list(getattr(self.config, "architectures", []) or [])
        if self.__class__.__name__ not in architectures:
            architectures.append(self.__class__.__name__)
        self.config.architectures = architectures
        self.base_model = AutoModel.from_pretrained(
            str(base_model_name_or_path),
            config=self.config,
            trust_remote_code=trust_remote_code,
        )
        self._forward_arg_names = set(inspect.signature(self.base_model.forward).parameters)
        hidden_size = (
            getattr(self.config, "hidden_size", None)
            or getattr(self.config, "d_model", None)
            or getattr(self.config, "model_dim", None)
        )
        if hidden_size is None:
            raise AttributeError("Could not infer hidden size from base model config.")
        dropout_prob = getattr(
            self.config,
            "classifier_dropout",
            getattr(self.config, "hidden_dropout_prob", 0.1),
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def resize_token_embeddings(self, new_num_tokens: int):
        if hasattr(self.base_model, "resize_token_embeddings"):
            return self.base_model.resize_token_embeddings(new_num_tokens)
        return None

    def _pool(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones(
                hidden_states.shape[:2], device=hidden_states.device, dtype=hidden_states.dtype
            )
        if self.pooling == "cls":
            return hidden_states[:, 0]
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).type(hidden_states.dtype)
            masked = hidden_states * mask
            denom = mask.sum(dim=1).clamp(min=1e-6)
            return masked.sum(dim=1) / denom
        if self.pooling == "last-token":
            indices = attention_mask.long().sum(dim=1) - 1
            indices = indices.clamp(min=0)
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_indices, indices]
        raise ValueError(f"Unsupported pooling strategy: {self.pooling}")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        base_kwargs = {}
        if "input_ids" in self._forward_arg_names and input_ids is not None:
            base_kwargs["input_ids"] = input_ids
        if "attention_mask" in self._forward_arg_names and attention_mask is not None:
            base_kwargs["attention_mask"] = attention_mask
        if "token_type_ids" in self._forward_arg_names and token_type_ids is not None:
            base_kwargs["token_type_ids"] = token_type_ids
        for key, value in kwargs.items():
            if value is None:
                continue
            if key in self._forward_arg_names:
                base_kwargs[key] = value
        if "return_dict" in self._forward_arg_names:
            base_kwargs["return_dict"] = True
        outputs = self.base_model(**base_kwargs)
        hidden_history = getattr(outputs, "hidden_states", None)
        attentions = getattr(outputs, "attentions", None)

        last_hidden = None
        if hasattr(outputs, "last_hidden_state"):
            maybe_hidden = outputs.last_hidden_state
            if isinstance(maybe_hidden, torch.Tensor):
                last_hidden = maybe_hidden
        elif isinstance(outputs, (tuple, list)) and outputs:
            maybe_hidden = outputs[0]
            if isinstance(maybe_hidden, torch.Tensor):
                last_hidden = maybe_hidden

        pooled = None
        sentence_embedding = getattr(outputs, "sentence_embedding", None)
        if sentence_embedding is None and isinstance(outputs, dict):
            sentence_embedding = outputs.get("sentence_embedding")
        if isinstance(sentence_embedding, torch.Tensor):
            pooled = sentence_embedding
        else:
            pooler_output = getattr(outputs, "pooler_output", None)
            if pooler_output is None and isinstance(outputs, dict):
                pooler_output = outputs.get("pooler_output")
            if isinstance(pooler_output, torch.Tensor):
                pooled = pooler_output
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 1:
                candidate = outputs[1]
                if isinstance(candidate, torch.Tensor) and candidate.ndim >= 2:
                    pooled = candidate

        if pooled is None:
            if last_hidden is None:
                raise ValueError("Base model outputs do not contain hidden states or sentence embeddings.")
            pooled = self._pool(last_hidden, attention_mask)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits.view(-1), labels.view(-1).float())
            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_history,
            attentions=attentions,
        )

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        self.config.save_pretrained(save_path)
        extra_path = save_path / "router_config.json"
        extra_payload = {"router_pooling": self.pooling}
        extra_path.write_text(json.dumps(extra_payload, indent=2))

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        pooling: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "PooledEncoderForSequenceClassification":
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
        effective_pooling = pooling or getattr(config, "router_pooling", "cls")
        num_labels = kwargs.get("num_labels", getattr(config, "num_labels", 2))
        model = cls(
            model_dir,
            num_labels=num_labels,
            pooling=effective_pooling,
            trust_remote_code=trust_remote_code,
        )
        state_path = Path(model_dir) / "pytorch_model.bin"
        if state_path.exists():
            state_dict = torch.load(state_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
        return model


def load_base_model_and_tokenizer(
    local_base: Path,
    local_tokenizer: Optional[Path] = None,
    num_labels: int = 2,
    backbone_type: str = "sequence-classification",
    pooling: str = "cls",
    trust_remote_code: bool = False,
    truncation_side: str = "right",
):
    """Load a local base HF encoder and initialize a classification head."""
    tok_path = str(local_tokenizer or local_base)
    tokenizer_kwargs = {"trust_remote_code": trust_remote_code}
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True, **tokenizer_kwargs)
    except Exception as exc:
        print(f"Falling back to slow tokenizer for {tok_path}: {exc}")
        tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=False, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if getattr(tokenizer, "padding_side", None) is None:
        tokenizer.padding_side = "right"
    truncation_side = truncation_side.lower()
    if truncation_side not in ("left", "right"):
        raise ValueError(f"Unsupported truncation_side: {truncation_side}")
    if getattr(tokenizer, "truncation_side", None) is None or tokenizer.truncation_side != truncation_side:
        tokenizer.truncation_side = truncation_side

    backbone = backbone_type.lower()
    if backbone == "sequence-classification":
        config = AutoConfig.from_pretrained(
            str(local_base),
            num_labels=num_labels,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(local_base),
            config=config,
            trust_remote_code=trust_remote_code,
        )
    elif backbone == "embedding":
        model = PooledEncoderForSequenceClassification(
            str(local_base),
            num_labels=num_labels,
            pooling=pooling,
            trust_remote_code=trust_remote_code,
        )
    else:
        raise ValueError(f"Unsupported backbone_type: {backbone_type}")

    if hasattr(model, "resize_token_embeddings") and tokenizer.pad_token_id is not None:
        try:
            model.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass

    return model, tokenizer


# --------------------------
# Evaluation utilities
# --------------------------

def logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    """Convert logits (N,2) to probability of the positive class as 1D (N,)."""
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    logits = np.asarray(logits)
    # Handle cases: (N,2) expected; if already probabilities, squeeze
    if logits.ndim == 2 and logits.shape[1] == 2:
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        denom = np.clip(exp.sum(axis=1), 1e-9, None)  # (N,)
        prob = exp[:, 1] / denom  # (N,)
        return prob
    # If model outputs a single logit (N,), apply sigmoid
    if logits.ndim == 1:
        return 1.0 / (1.0 + np.exp(-logits))
    # If model outputs (N,1), squeeze to (N,) and sigmoid
    if logits.ndim == 2 and logits.shape[1] == 1:
        v = logits[:, 0]
        return 1.0 / (1.0 + np.exp(-v))
    # Fallback: argmax to class 1 indicator then cast to float
    return (np.argmax(logits, axis=-1) == 1).astype(float)


def evaluate_scores(test_df: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> Dict[str, float]:
    labels = test_df["label"].to_numpy()
    preds = (probabilities >= threshold).astype(int)

    metrics: Dict[str, float] = {
        "record_accuracy": accuracy_score(labels, preds)
    }
    try:
        metrics["record_roc_auc"] = roc_auc_score(labels, probabilities)
    except ValueError:
        metrics["record_roc_auc"] = float("nan")

    enriched = test_df.copy()
    enriched["probability"] = probabilities
    enriched["prediction"] = preds
    enriched["sample_id"] = enriched["sample_id"].astype(str)

    prompt_probs = enriched.groupby("sample_id")["probability"].max()
    prompt_labels = enriched.groupby("sample_id")["label"].max()
    prompt_preds = enriched.groupby("sample_id")["prediction"].max()

    metrics["prompt_accuracy"] = accuracy_score(prompt_labels, prompt_preds)
    try:
        metrics["prompt_roc_auc"] = roc_auc_score(prompt_labels, prompt_probs)
    except ValueError:
        metrics["prompt_roc_auc"] = float("nan")

    print("=== Overall Metrics ===")
    for k, v in metrics.items():
        print(f"{k:>16}: {v:.4f}")
    print("=== Record-level Classification Report ===")
    print(classification_report(labels, preds, digits=4))

    print("=== Prompt-level Classification Report ===")
    print(classification_report(prompt_labels, prompt_preds, digits=4))
    return metrics


def evaluate_per_dataset(test_df: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> None:
    """Prompt-level evaluation without cascade (cheapest positive else most confident)."""
    df = test_df.copy()
    df["probability"] = probabilities
    df["prediction"] = (probabilities >= threshold).astype(int)
    df["dataset_name"] = df["dataset_name"].fillna("unknown")
    df["sample_id"] = df["sample_id"].astype(str)

    def select_model(group: pd.DataFrame) -> pd.Series:
        sorted_group = group.sort_values(["cost", "probability"], ascending=[True, False])
        positives = sorted_group[sorted_group["prediction"] == 1]
        if not positives.empty:
            chosen = positives.iloc[0]
        else:
            fallback = sorted_group.sort_values(["probability", "cost"], ascending=[False, True])
            chosen = fallback.iloc[0]
        return chosen[["dataset_name", "sample_id", "label", "prediction", "probability", "cost", "model_name"]]

    selected = (
        df.groupby(["dataset_name", "sample_id"], dropna=False)
        .apply(select_model)
        .reset_index(drop=True)
    )

    total_samples = len(selected)
    total_cost = selected["cost"].sum()
    avg_cost = total_cost / max(total_samples, 1)

    print("\n=== Accuracy & cost by dataset (prompt-level) ===")
    per_dataset_acc: Dict[str, float] = {}
    for dataset, group in selected.groupby("dataset_name", dropna=False):
        dataset_key = dataset or "unknown"
        acc = float((group["label"] == 1).mean())
        per_dataset_acc[dataset_key] = acc
        cost_total = group["cost"].sum()
        cost_avg = cost_total / max(len(group), 1)
        print(
            f"{dataset_key}: acc={acc:.4f} prompts={len(group)} "
            f"total_cost={cost_total:.10f} avg_cost={cost_avg:.10f}"
        )

    if total_samples:
        macro_acc = float(np.mean(list(per_dataset_acc.values()))) if per_dataset_acc else float("nan")
        print(
            f"\nAll datasets: prompts={total_samples} "
            f"total_cost={total_cost:.10f} avg_cost={avg_cost:.10f} macro_acc={macro_acc:.4f}"
        )


def compute_model_average_costs(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty or "cost" not in df.columns:
        return {}
    cost_series = df[df["cost"].notna()].groupby("model_name")["cost"].mean()
    return {str(model): float(value) for model, value in cost_series.items()}


@dataclass
class CascadeStrategy:
    model_order: List[str]
    thresholds: List[float]
    quantile: List[float]
    budget: float


def _parse_order_arg(order_arg: Optional[str]) -> Optional[List[str]]:
    if not order_arg:
        return None
    txt = str(order_arg).strip()
    if "," in txt:
        names = [t.strip() for t in txt.split(",") if t.strip()]
        return names or None
    return None


def _accept_mask(d_mat: np.ndarray, thresholds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mimic optimizer acceptance: accept at first position where distance < threshold."""
    if d_mat.ndim != 2 or thresholds.ndim != 1:
        raise ValueError("Invalid shapes for d_mat or thresholds.")
    if d_mat.shape[1] != thresholds.shape[0]:
        raise ValueError("Thresholds length must equal cascade depth.")

    raw_accept = d_mat < thresholds  # (N, depth)
    any_accept = raw_accept.any(axis=1)
    if raw_accept.shape[1] == 0:
        return np.zeros_like(raw_accept, dtype=bool), any_accept

    first_hits = np.argmax(raw_accept, axis=1)
    accept = np.zeros_like(raw_accept, dtype=bool)
    rows = np.arange(raw_accept.shape[0])
    accept[rows[any_accept], first_hits[any_accept]] = True
    return accept, any_accept


def _compute_acc_and_cost(L_mat: np.ndarray, C_mat: np.ndarray, d_mat: np.ndarray, thresholds: np.ndarray, budget: float) -> Tuple[float, float]:
    accept, _ = _accept_mask(d_mat, thresholds)
    acc_sum = float(np.sum(accept * L_mat))
    cost_sum = float(np.sum(accept * C_mat))
    over_budget = cost_sum > budget * max(L_mat.shape[0], 1)
    return acc_sum, cost_sum if not over_budget else float("inf")


def _quantiles_to_thresholds(qual: np.ndarray, d_mat: np.ndarray) -> np.ndarray:
    depth = d_mat.shape[1]
    if depth == 0:
        return np.array([])
    thresholds = np.zeros(depth, dtype=float)
    thresholds[-1] = 1.0
    mask = np.ones(d_mat.shape[0], dtype=bool)
    for i in range(depth - 1):
        qi = float(np.clip(qual[i], 1e-5, 1 - 1e-5))
        data = d_mat[mask, i]
        if data.size == 0:
            thresholds[i] = 1.0
            mask = np.zeros_like(mask, dtype=bool)
            continue
        thr = float(np.quantile(data, 1 - qi))
        thresholds[i] = thr
        mask = np.logical_and(mask, d_mat[:, i] >= thr)
    return thresholds


def optimize_thresholds(L_mat: np.ndarray, C_mat: np.ndarray, d_mat: np.ndarray, budget: float, brute_samples: int = 15) -> Tuple[float, np.ndarray, np.ndarray]:
    """Port of original optimizer.optimize without external deps."""
    n, depth = L_mat.shape
    if depth == 0 or n == 0:
        return float("-inf"), np.array([]), np.array([])
    if float(np.average(C_mat[:, 0])) > float(budget):
        return float("-inf"), np.array([]), np.array([])

    if depth == 1:
        thresholds = np.array([1.0], dtype=float)
        acc_sum, cost_sum = _compute_acc_and_cost(L_mat, C_mat, d_mat, thresholds, budget)
        acc = acc_sum / max(n, 1)
        if cost_sum == float("inf"):
            acc = float("-inf")
        return acc, thresholds, np.array([1.0])

    def objective(qual_flat: np.ndarray) -> float:
        qual = np.asarray(qual_flat, dtype=float)
        if qual.ndim == 0:
            qual = np.array([float(qual)])
        if not np.all(np.diff(qual) >= 0):
            return 1e6
        thresholds = _quantiles_to_thresholds(qual, d_mat)
        acc_sum, cost_sum = _compute_acc_and_cost(L_mat, C_mat, d_mat, thresholds, budget)
        if cost_sum == float("inf"):
            return 1e6
        return -acc_sum

    ranges = [(1e-5, 1 - 1e-5)] * (depth - 1)
    resbrute = optimize.brute(
        objective,
        ranges,
        full_output=True,
        finish=optimize.fmin,
        Ns=max(5, brute_samples),
        disp=False,
    )
    best_qual = np.asarray(resbrute[0], dtype=float)
    best_thresholds = _quantiles_to_thresholds(best_qual, d_mat)
    best_acc = -float(resbrute[1]) / max(n, 1)
    return best_acc, best_thresholds, best_qual


def _pivot_by_model(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return df.pivot_table(index="sample_id", columns="model_name", values=value_col)


def _enumerate_orders(models: List[str], depth: int, user_order: Optional[List[str]], perm_limit: int = 5000) -> List[Tuple[str, ...]]:
    if user_order:
        filtered = [m for m in user_order if m in models]
        if not filtered:
            return []
        return [tuple(filtered[:depth])]
    depth = max(1, min(depth, len(models)))
    total = math.perm(len(models), depth) if hasattr(math, "perm") else math.factorial(len(models)) // math.factorial(len(models) - depth)
    if total > perm_limit:
        # Avoid factorial explosion: fall back to single greedy order (cost-ascending)
        return [tuple(sorted(models))]
    return list(itertools.permutations(models, depth))


def compute_cascade_strategy(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    budget: float,
    cascade_max_depth: Optional[int],
    cascade_order_arg: Optional[str],
    brute_samples: int = 40,
    perm_limit: int = 5000,
) -> Optional[CascadeStrategy]:
    if df.empty or probabilities is None or probabilities.size == 0:
        return None
    work = df.copy().reset_index(drop=True)
    work["probability"] = probabilities
    work["sample_id"] = work["sample_id"].astype(str)
    work["dataset_name"] = work["dataset_name"].fillna("unknown")

    label_piv = _pivot_by_model(work, "label")
    prob_piv = _pivot_by_model(work, "probability")
    cost_piv = _pivot_by_model(work, "cost")
    models = [m for m in prob_piv.columns.tolist() if m in label_piv.columns and m in cost_piv.columns]
    if not models:
        return None

    depth = cascade_max_depth if cascade_max_depth and cascade_max_depth > 0 else len(models)
    user_order = _parse_order_arg(cascade_order_arg)
    orders = _enumerate_orders(models, depth, user_order, perm_limit=perm_limit)
    best: Optional[CascadeStrategy] = None
    best_acc = float("-inf")

    for order in orders:
        order = list(order)
        sub_labels = label_piv[order]
        sub_probs = prob_piv[order]
        sub_costs = cost_piv[order]
        mask = sub_labels.notna().all(axis=1) & sub_probs.notna().all(axis=1) & sub_costs.notna().all(axis=1)
        if mask.sum() == 0:
            continue
        L_mat = sub_labels[mask].to_numpy(dtype=float)
        P_mat = sub_probs[mask].to_numpy(dtype=float)
        C_mat = np.cumsum(sub_costs[mask].to_numpy(dtype=float), axis=1)
        d_mat = 1.0 - P_mat
        acc, thresholds, quantile = optimize_thresholds(L_mat, C_mat, d_mat, budget, brute_samples=brute_samples)
        if acc > best_acc:
            best_acc = acc
            best = CascadeStrategy(model_order=order, thresholds=thresholds.tolist(), quantile=np.asarray(quantile).tolist(), budget=float(budget))

    return best


def evaluate_cascade_with_strategy(
    test_df: pd.DataFrame,
    probabilities: np.ndarray,
    strategy: CascadeStrategy,
    cascade_max_depth: Optional[int],
) -> None:
    if strategy is None or not strategy.model_order:
        print("[cascade] No strategy to evaluate.")
        return

    df = test_df.copy()
    df["probability"] = probabilities
    df["dataset_name"] = df["dataset_name"].fillna("unknown")
    df["sample_id"] = df["sample_id"].astype(str)

    depth = min(len(strategy.model_order), cascade_max_depth) if cascade_max_depth and cascade_max_depth > 0 else len(strategy.model_order)
    thresholds = strategy.thresholds[:depth] + [1.0] * max(0, depth - len(strategy.thresholds))

    selections: List[Dict] = []
    for (dataset, sample), group in df.groupby(["dataset_name", "sample_id"], dropna=False):
        rows = []
        for name in strategy.model_order[:depth]:
            sub = group[group["model_name"] == name]
            if not sub.empty:
                rows.append(sub.iloc[0])
        if not rows:
            continue

        cumulative_cost = 0.0
        chosen_row = rows[-1]
        for idx, row in enumerate(rows):
            cumulative_cost += float(row["cost"]) if pd.notna(row["cost"]) else 0.0
            thr = thresholds[idx] if idx < len(thresholds) else 1.0
            if float(row["probability"]) > (1.0 - float(thr)):
                chosen_row = row
                break
            chosen_row = row

        selections.append(
            {
                "dataset_name": dataset,
                "sample_id": sample,
                "model_name": chosen_row["model_name"],
                "label": int(chosen_row["label"]),
                "probability": float(chosen_row["probability"]),
                "cum_cost": cumulative_cost,
            }
        )

    if not selections:
        print("[cascade] No selections to report.")
        return

    sel_df = pd.DataFrame(selections)
    print("\n=== Cascade (prompt-level): accuracy & cumulative cost by dataset ===")
    per_dataset_acc: Dict[str, float] = {}
    total_prompts = 0
    total_cost = 0.0
    for dataset, group in sel_df.groupby("dataset_name", dropna=False):
        acc = float((group["label"] == 1).mean())
        cost_sum = float(group["cum_cost"].sum())
        prompts = int(len(group))
        total_prompts += prompts
        total_cost += cost_sum
        per_dataset_acc[dataset or "unknown"] = acc
        print(
            f"{dataset or 'unknown'}: acc={acc:.4f} prompts={prompts} total_cost={cost_sum:.10f} avg_cost={cost_sum/max(prompts,1):.10f}"
        )

    macro_acc = float(np.mean(list(per_dataset_acc.values()))) if per_dataset_acc else float("nan")
    print(
        f"\nAll datasets (cascade): prompts={total_prompts} total_cost={total_cost:.10f} "
        f"avg_cost={total_cost/max(total_prompts,1):.10f} macro_acc={macro_acc:.4f}"
    )


def train_single_model(
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    model_output_dir: Optional[Path],
) -> Tuple[np.ndarray, np.ndarray]:
    """Train one scorer for a specific model_name and return train/test probabilities aligned to provided dataframes."""
    if train_df.empty:
        raise ValueError(f"No training data for model {model_name}")

    print(f"\n=== Training scorer for model: {model_name} (train={len(train_df)} test={len(test_df)}) ===")

    # Load base model/tokenizer fresh per model to keep heads independent
    model, tokenizer = load_base_model_and_tokenizer(
        args.local_base,
        args.local_tokenizer,
        num_labels=2,
        backbone_type=args.backbone_type,
        pooling=args.pooling,
        trust_remote_code=args.trust_remote_code,
        truncation_side=args.truncation_side,
    )

    train_ds = TextClsDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_len=args.max_length)
    test_ds = TextClsDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_len=args.max_length)

    using_deepspeed = args.deepspeed is not None
    ds_engine = None
    ds_config = None
    rank = 0
    world_size = 1

    if using_deepspeed:
        ds_config_path = Path(args.deepspeed).expanduser()
        if not ds_config_path.exists():
            raise SystemExit(f"DeepSpeed config not found: {ds_config_path}")
        try:
            import deepspeed
        except ImportError as exc:
            raise SystemExit("DeepSpeed requested (--deepspeed) but the deepspeed package is not installed.") from exc
        ds_config = json.loads(ds_config_path.read_text())
        if not torch.cuda.is_available():
            raise SystemExit("DeepSpeed requires at least one CUDA device.")
        if not torch.distributed.is_available():
            raise SystemExit("torch.distributed is required for DeepSpeed runs.")
        if not torch.distributed.is_initialized():
            deepspeed.init_distributed()
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = args.local_rank if isinstance(args.local_rank, int) else -1
        if isinstance(local_rank, int) and local_rank >= 0:
            rank = local_rank
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed_all(args.seed + int(rank))

    train_sampler = None
    if using_deepspeed:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
    )

    steps_per_epoch = max(1, len(train_loader))
    grad_accum_steps = max(1, args.grad_accum)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = max(1, math.ceil(steps_per_epoch / grad_accum_steps))
    if args.max_steps is not None and args.max_steps > 0:
        total_training_steps = args.max_steps
        max_epochs = max(1, math.ceil(args.max_steps / updates_per_epoch))
    else:
        total_training_steps = updates_per_epoch * args.epochs
        max_epochs = args.epochs

    warmup_steps = int(total_training_steps * args.warmup_ratio) if total_training_steps else 0
    scheduler = (
        get_linear_schedule_with_warmup(optimizer, warmup_steps, total_training_steps)
        if total_training_steps > 0
        else None
    )

    use_cuda_amp = device.type == "cuda" and not using_deepspeed
    use_bf16 = use_cuda_amp and args.bf16
    use_fp16 = use_cuda_amp and not args.bf16
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    if rank == 0:
        msg = (
            f"[{model_name}] Training for up to {max_epochs} epoch(s); updates_per_epoch={updates_per_epoch}, "
            f"total_training_steps={total_training_steps}"
        )
        if using_deepspeed:
            msg += f" (DeepSpeed world_size={world_size})"
        print(msg)

    def _autocast():
        return torch.cuda.amp.autocast(dtype=autocast_dtype) if use_cuda_amp else nullcontext()

    model_engine = model
    if using_deepspeed:
        ds_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=ds_config,
        )
        model_engine = ds_engine
        train_device = device
    else:
        model.to(device)
        train_device = device

    global_step = 0
    log_interval = max(1, args.log_interval)
    stop_training = False
    for epoch in range(max_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model_engine.train()
        if not using_deepspeed:
            optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        interval_loss = 0.0
        interval_count = 0
        pbar = None
        if rank == 0:
            pbar = tqdm(
                total=steps_per_epoch,
                desc=f"[{model_name}] Epoch {epoch + 1}/{max_epochs}",
                leave=True,
                dynamic_ncols=True,
            )
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(train_device) for k, v in batch.items()}
            with _autocast():
                outputs = model_engine(**batch)
                loss_tensor = getattr(outputs, "loss", None)
                if loss_tensor is None:
                    logits = getattr(outputs, "logits", None)
                    if logits is None and isinstance(outputs, (tuple, list)) and outputs:
                        logits = outputs[0]
                    if logits is None:
                        raise ValueError("Model forward did not return logits.")
                    loss_tensor = nn.CrossEntropyLoss()(logits, batch["labels"])
            loss_value = float(loss_tensor.detach().cpu())
            interval_loss += loss_value
            interval_count += 1
            if using_deepspeed:
                model_engine.backward(loss_tensor)
                model_engine.step()
                global_step = int(getattr(model_engine, "global_steps", global_step + 1))
            else:
                loss_to_backprop = loss_tensor / grad_accum_steps
                if use_fp16:
                    scaler.scale(loss_to_backprop).backward()
                else:
                    loss_to_backprop.backward()
                should_step = (step + 1) % grad_accum_steps == 0 or (step + 1) == steps_per_epoch
                if should_step:
                    if use_fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
            stop_training = bool(total_training_steps and global_step >= total_training_steps)
            if rank == 0:
                epoch_loss += loss_value
                if pbar is not None:
                    pbar.update(1)
                should_log = (
                    ((step + 1) % log_interval == 0)
                    or (step + 1 == steps_per_epoch)
                    or stop_training
                )
                if should_log and pbar is not None:
                    avg_interval_loss = interval_loss / max(1, interval_count)
                    pbar.set_postfix({
                        "loss": f"{avg_interval_loss:.4f}",
                        "gstep": global_step,
                    })
                    interval_loss = 0.0
                    interval_count = 0
            if stop_training:
                break
        if pbar is not None:
            pbar.close()
        if stop_training:
            if rank == 0:
                print(f"[{model_name}] Reached max steps ({global_step}); stopping training.")
            break

    if using_deepspeed and torch.distributed.is_initialized():
        torch.distributed.barrier()
        is_primary = torch.distributed.get_rank() == 0
    else:
        is_primary = True

    eval_device = train_device if isinstance(train_device, torch.device) and train_device.type == "cuda" else torch.device("cpu")
    if using_deepspeed:
        model_for_eval_engine = model_engine
        model_for_eval_engine.eval()
        base_model_for_save = model_engine.module
    else:
        model_for_eval_engine = model_engine
        model_for_eval_engine.to(eval_device)
        model_for_eval_engine.eval()
        base_model_for_save = model_for_eval_engine

    def collect_logits(dataset: TextClsDataset, batch_size: int) -> Optional[np.ndarray]:
        distributed_eval = using_deepspeed and torch.distributed.is_initialized() and world_size > 1
        total_samples = len(dataset)

        if distributed_eval:
            shard_size = math.ceil(total_samples / world_size) if total_samples > 0 else 0
            start_idx = shard_size * rank
            end_idx = min(total_samples, start_idx + shard_size)
            shard_indices = list(range(start_idx, end_idx))
            dataset_for_rank = Subset(dataset, shard_indices)
            index_source = shard_indices
        else:
            dataset_for_rank = dataset
            index_source = list(range(total_samples))

        loader = DataLoader(dataset_for_rank, batch_size=batch_size, shuffle=False)
        logits_list: List[torch.Tensor] = []
        local_indices: List[int] = []
        pointer = 0
        first_shape: Optional[torch.Size] = None

        for batch in loader:
            current_batch_size = batch["labels"].shape[0]
            if distributed_eval:
                current_indices = index_source[pointer : pointer + current_batch_size]
            else:
                current_indices = list(range(pointer, pointer + current_batch_size))
            pointer += current_batch_size

            inputs = {k: v.to(eval_device) for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                outputs = model_for_eval_engine(**inputs)
            logits = getattr(outputs, "logits", None)
            if logits is None and isinstance(outputs, (tuple, list)) and outputs:
                logits = outputs[0]
            if logits is None:
                raise ValueError("Model forward did not return logits.")
            logits = logits.detach()
            if logits.dtype in (torch.float16, torch.bfloat16):
                logits = logits.float()
            logits = logits.to(eval_device)
            logits_list.append(logits)
            local_indices.extend(current_indices)
            if first_shape is None:
                first_shape = logits.shape[1:]

        if logits_list:
            local_logits_tensor = torch.cat(logits_list, dim=0)
        else:
            empty_shape = (0,) + tuple(first_shape or ())
            local_logits_tensor = torch.empty(empty_shape, dtype=torch.float32, device=eval_device)

        if distributed_eval:
            local_indices_tensor = (
                torch.tensor(local_indices, dtype=torch.long, device=eval_device)
                if local_indices
                else torch.zeros((0,), dtype=torch.long, device=eval_device)
            )

            def _all_gather_with_sizes(tensor: torch.Tensor) -> tuple[List[torch.Tensor], List[int]]:
                local_count = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
                count_list = [torch.zeros_like(local_count) for _ in range(world_size)]
                torch.distributed.all_gather(count_list, local_count)
                max_count = int(torch.stack(count_list).max().item())
                if tensor.shape[0] < max_count:
                    pad_shape = (max_count - tensor.shape[0],) + tensor.shape[1:]
                    pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
                    tensor_to_send = torch.cat([tensor, pad_tensor], dim=0)
                else:
                    tensor_to_send = tensor
                gathered = [torch.zeros_like(tensor_to_send) for _ in range(world_size)]
                torch.distributed.all_gather(gathered, tensor_to_send)
                sizes = [int(c.item()) for c in count_list]
                return gathered, sizes

            gathered_logits, logits_sizes = _all_gather_with_sizes(local_logits_tensor)
            gathered_indices, index_sizes = _all_gather_with_sizes(local_indices_tensor)

            if is_primary:
                final_shape = (total_samples,) + tuple(local_logits_tensor.shape[1:])
                final_logits = torch.empty(final_shape, dtype=torch.float32)
                for worker_rank in range(world_size):
                    take = min(logits_sizes[worker_rank], index_sizes[worker_rank])
                    if take <= 0:
                        continue
                    worker_logits = gathered_logits[worker_rank][:take].cpu().float()
                    worker_indices = gathered_indices[worker_rank][:take].cpu().long()
                    final_logits[worker_indices] = worker_logits
                return final_logits.numpy()
            return None

        return local_logits_tensor.detach().cpu().float().numpy()

    def collect_probabilities(dataset: TextClsDataset, batch_size: int) -> Optional[np.ndarray]:
        logits = collect_logits(dataset, batch_size)
        if logits is None:
            return None
        return logits_to_probabilities(logits)

    test_probabilities = collect_probabilities(test_ds, args.eval_batch_size)
    train_probabilities = collect_probabilities(train_ds, args.eval_batch_size)

    if is_primary and model_output_dir:
        model_output_dir.mkdir(parents=True, exist_ok=True)
        model_for_save = base_model_for_save.to("cpu")
        model_for_save.save_pretrained(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)
        meta = {
            "source": str(args.local_base),
            "model_name": model_name,
            "epochs": args.epochs,
            "max_length": args.max_length,
            "backbone_type": args.backbone_type,
            "pooling": args.pooling,
            "bf16": args.bf16,
        }
        (model_output_dir / "scorer_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[{model_name}] Saved fine-tuned scorer to {model_output_dir}")

    return train_probabilities, test_probabilities





# --------------------------
# Argument parsing
# --------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune a local encoder model (BERT or pooled-embedding LLM) as a FrugalGPT scorer."
    )
    p.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Directories or JSON files containing result records.",
    )
    p.add_argument(
        "--train-jsonl",
        type=Path,
        default=None,
        help="Optional LLMRouterBench train split JSONL (records + usages).",
    )
    p.add_argument(
        "--test-jsonl",
        type=Path,
        default=None,
        help="Optional LLMRouterBench test split JSONL (records + usages).",
    )
    p.add_argument(
        "--local-base",
        type=Path,
        required=True,
        help="Path to a local base HF model directory (BERT-style encoder or pooled embedding LLM).",
    )
    p.add_argument(
        "--local-tokenizer",
        type=Path,
        default=None,
        help="Optional path to tokenizer directory (defaults to --local-base).",
    )
    p.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bfloat16 mixed precision when supported.",
    )
    p.add_argument(
        "--truncation-side",
        choices=("left", "right"),
        default="right",
        help="Tokenization truncation side (left keeps tail, right keeps head).",
    )
    p.add_argument(
        "--backbone-type",
        choices=("sequence-classification", "embedding"),
        default="sequence-classification",
        help="Backbone mode: use HF sequence classification heads or pooled embedding encoders.",
    )
    p.add_argument(
        "--pooling",
        choices=("cls", "mean", "last-token"),
        default="cls",
        help="Pooling strategy when --backbone-type=embedding.",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading models/tokenizers that require trust_remote_code (use with caution).",
    )
    p.add_argument(
        "--local_rank",
        "--local-rank",
        dest="local_rank",
        type=int,
        default=-1,
        help="Compatibility shim for torch.distributed launchers; value is ignored.",
    )
    p.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Label generator: JSON 'score' >= this threshold => label=1.",
    )
    p.add_argument(
        "--prob-threshold",
        type=float,
        default=0.5,
        help="Reporting threshold: prob >= this => predict positive.",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio.",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/test split.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on total samples for quick experiments.",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Tokenizer max_length (BERT-family typically 512).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device train batch size.",
    )
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Per-device eval batch size.",
    )
    p.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Run evaluation every N training steps (when supported).",
    )
    p.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Number of optimizer steps between progress prints (primary rank only).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on total optimizer steps (overrides epochs if set).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay.",
    )
    p.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.06,
        help="Warmup ratio for LR scheduler.",
    )
    p.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    p.add_argument(
        "--deepspeed",
        type=Path,
        default=None,
        help="Optional DeepSpeed config JSON to enable ZeRO/offload distributed training.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for torch.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to save fine-tuned model/tokenizer.",
    )
    p.add_argument(
        "--cascade",
        action="store_true",
        help="Enable cascade-style offline evaluation (simulate cumulative cost).",
    )
    p.add_argument(
        "--cascade-order",
        type=str,
        default=None,
        help=(
            "Cascade order: comma-separated model names to fix the order; "
            "if omitted, permutations up to --max-permutations are searched."
        ),
    )
    p.add_argument(
        "--cascade-max-depth",
        type=int,
        default=None,
        help="Optional cap on number of models tried per prompt in cascade.",
    )
    p.add_argument(
        "--cascade-config",
        type=Path,
        default=None,
        help="Optional path to write cascade order/threshold JSON for inference.",
    )
    p.add_argument(
        "--budget",
        type=float,
        default=0.1,
        help="Per-prompt budget for cascade optimizer (matches original FrugalGPT).",
    )
    p.add_argument(
        "--max-permutations",
        type=int,
        default=5000,
        help="Safety cap on cascade order permutations searched (original logic enumerates all permutations).",
    )
    p.add_argument(
        "--brute-samples",
        type=int,
        default=40,
        help="Grid size per dimension for brute-force quantile search (original uses 40).",
    )
    return p.parse_args()


# --------------------------
# Main
# --------------------------



def main() -> None:
    args = parse_args()

    using_jsonl = args.train_jsonl is not None or args.test_jsonl is not None

    if using_jsonl:
        if args.train_jsonl is None or args.test_jsonl is None:
            raise SystemExit("Both --train-jsonl and --test-jsonl must be provided together.")
        if args.inputs:
            raise SystemExit("Use either --inputs or --train-jsonl/--test-jsonl, not both.")
        train_df = load_jsonl_split(args.train_jsonl, split="train")
        test_df = load_jsonl_split(args.test_jsonl, split="test")
        df = pd.concat([train_df, test_df], ignore_index=True)
    else:
        if not args.inputs:
            raise SystemExit("Either --inputs or both --train-jsonl/--test-jsonl must be provided.")
        files = list(iter_result_files(args.inputs))
        if not files:
            raise SystemExit("No JSON files were found underneath the provided inputs.")
        df = load_records(files)

    if args.max_samples is not None and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.random_state).reset_index(drop=True)

    df["label"] = (df["score"] >= args.score_threshold).astype(int)
    unique_prompts = df["sample_id"].astype(str).nunique()
    print(f"Collected {len(df)} records ({unique_prompts} unique prompts) across {df['model_name'].nunique()} models.")
    print(f"Positive label rate (per record, score >= {args.score_threshold}): {df['label'].mean():.4f}")
    sample_pos_rate = df.groupby("sample_id")["label"].max().mean()
    print(f"Positive label rate (per prompt max over models): {sample_pos_rate:.4f}")

    df["dataset_name"] = df["dataset_name"].fillna("unknown")
    if "sample_id" not in df.columns:
        df["sample_id"] = (
            df["dataset_name"].astype(str)
            + "::"
            + df["record_index"].astype(str)
        )

    if using_jsonl:
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        test_df = df[df["split"] == "test"].reset_index(drop=True)
        if train_df.empty or test_df.empty:
            raise SystemExit("Train/test JSONL splits did not yield any samples after preprocessing.")
    else:
        groups = df["sample_id"].astype(str).to_numpy()
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
        train_idx, test_idx = next(gss.split(df, df["label"], groups))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

    # Keep only models that have training data; drop test-only models to avoid NaN probabilities later.
    train_models = set(train_df["model_name"].dropna().unique().tolist())
    test_models = set(test_df["model_name"].dropna().unique().tolist())
    missing_train_models = sorted(test_models - train_models)
    if missing_train_models:
        before = len(test_df)
        print(f"[warn] Dropping test rows for models without training data: {missing_train_models}")
        test_df = test_df[test_df["model_name"].isin(train_models)].reset_index(drop=True)
        removed = before - len(test_df)
        print(f"[warn] Removed {removed} test rows lacking trained models.")
        if test_df.empty:
            raise SystemExit("No test samples remain after removing models without training data. Check your splits.")

    cascade_avg_costs: Dict[str, float] = compute_model_average_costs(train_df if not train_df.empty else df) if args.cascade else {}

    def current_rank() -> int:
        """Best-effort rank detection before/after distributed init."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                return int(torch.distributed.get_rank())
            except Exception:
                pass
        env_rank = os.environ.get("RANK")
        if env_rank is not None:
            try:
                return int(env_rank)
            except ValueError:
                return 0
        return 0

    dist_rank = current_rank()

    models = sorted(train_df["model_name"].dropna().unique().tolist())
    if not models:
        raise SystemExit("No model_name values found for training.")

    train_probs_full = np.full(len(train_df), np.nan, dtype=float)
    test_probs_full = np.full(len(test_df), np.nan, dtype=float)

    for model_name in models:
        sub_train = train_df[train_df["model_name"] == model_name].reset_index(drop=True)
        sub_test = test_df[test_df["model_name"] == model_name].reset_index(drop=True)
        if sub_train.empty:
            print(f"[skip] No training data for model {model_name}")
            continue
        output_dir_model = (args.output_dir / model_name) if args.output_dir else None
        train_probs, test_probs = train_single_model(args, sub_train, sub_test, model_name, output_dir_model)
        if train_probs is not None and len(train_probs) == len(sub_train):
            mask = train_df["model_name"] == model_name
            train_probs_full[mask] = train_probs
        if test_probs is not None and len(test_probs) == len(sub_test):
            mask = test_df["model_name"] == model_name
            test_probs_full[mask] = test_probs

    # Re-evaluate rank after potential distributed init during training, and skip
    # evaluation/NaN checks on non-primary ranks (they do not gather probabilities).
    dist_rank = current_rank()
    if dist_rank != 0:
        return

    if np.isnan(test_probs_full).any():
        missing_mask = np.isnan(test_probs_full)
        missing_counts = (
            test_df.loc[missing_mask, "model_name"]
            .value_counts()
            .to_dict()
        )
        raise RuntimeError(f"Missing probabilities for some test rows; ensure each model was trained. Missing counts={missing_counts}")

    metrics = evaluate_scores(test_df, test_probs_full, args.prob_threshold)
    print(f"[record_accuracy]={metrics.get('record_accuracy'):.4f}")

    cascade_strategy: Optional[CascadeStrategy] = None
    if args.cascade:
        if np.isnan(train_probs_full).any():
            print("[cascade] Skip strategy learning: missing train probabilities.")
        else:
            train_with_probs = train_df.copy().reset_index(drop=True)
            train_with_probs["probability"] = train_probs_full
            cascade_strategy = compute_cascade_strategy(
                train_with_probs,
                train_probs_full,
                budget=args.budget,
                cascade_max_depth=args.cascade_max_depth,
                cascade_order_arg=args.cascade_order,
                brute_samples=args.brute_samples,
                perm_limit=args.max_permutations,
            )
            if cascade_strategy:
                print(f"[cascade] Learned strategy: order={cascade_strategy.model_order} thresholds={cascade_strategy.thresholds} budget={cascade_strategy.budget}")
            else:
                print("[cascade] Failed to learn cascade strategy.")

    if args.cascade and cascade_strategy:
        evaluate_cascade_with_strategy(
            test_df,
            test_probs_full,
            strategy=cascade_strategy,
            cascade_max_depth=args.cascade_max_depth,
        )
    elif args.cascade:
        print("[cascade] No valid strategy; falling back to non-cascade selection.")
        evaluate_per_dataset(test_df, test_probs_full, args.prob_threshold)
    else:
        evaluate_per_dataset(test_df, test_probs_full, args.prob_threshold)

    cascade_config_path = args.cascade_config
    if args.cascade:
        config_path = cascade_config_path or (args.output_dir / "cascade_config.json" if args.output_dir else None)
        if config_path and cascade_strategy:
            config_payload = {
                "model_order": cascade_strategy.model_order,
                "thresholds": cascade_strategy.thresholds,
                "quantile": cascade_strategy.quantile,
                "budget": cascade_strategy.budget,
                "cascade_max_depth": args.cascade_max_depth,
                "average_cost": cascade_avg_costs,
            }
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config_payload, indent=2))
            print(f"\nSaved cascade config to {config_path}")
        elif config_path and not cascade_strategy:
            print("[cascade] Skipped writing cascade config: no strategy learned.")
        else:
            print("[cascade] Skipped writing cascade config: provide --cascade-config or --output-dir.")

    print("\nDone. Per-model scorers saved inside output dir if provided.")


if __name__ == "__main__":
    main()
