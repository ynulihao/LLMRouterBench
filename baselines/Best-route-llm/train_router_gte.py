# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Modified version for GTE (General Text Embeddings) models
# This version uses last-token pooling instead of source prefix token pooling

import os
import json
import time
import shutil
import numpy as np
import torch
import logging
import argparse

from uuid import uuid4

from typing import List, Tuple
from transformers import PreTrainedTokenizer, TrainingArguments, TrainerCallback
import inspect
from hybrid_llm.pair_ranker.config import NClassRankerConfig
from hybrid_llm.pair_ranker.data import load_data, Dataset
from hybrid_llm.pair_ranker.collator import NClassCollator
from hybrid_llm.pair_ranker.ranker import NClassReranker, NClassRerankerLossType
from hybrid_llm.pair_ranker.trainer import (compute_metrics_for_prob_2class, compute_metrics_for_det_nclass,
                                            compute_metrics_for_det_nlabel, compute_metrics_for_prob_nlabel)
from hybrid_llm.pair_ranker.model_util import (
    build_nclass_ranker,
    build_nclass_ranker_from_checkpoint,
)

from llm_blender.pair_ranker.model_util import build_tokenizer
from llm_blender.pair_ranker.trainer import RerankerTrainer
from llm_blender.common.utils import empty2None, str2bool, seed_everything

loss2metric = {
    "det_2cls": compute_metrics_for_prob_2class,  # reuse 2-class metric for deterministic 2-class
    "det_ncls": compute_metrics_for_det_nclass,
    "det_nlabels": compute_metrics_for_det_nlabel,
    "prob_2cls": compute_metrics_for_prob_2class,
    "prob_nlabels": compute_metrics_for_prob_nlabel,
}


def _ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def _to_serializable(val):
    if isinstance(val, (np.generic,)):
        return val.item()
    if torch.is_tensor(val):
        return val.detach().cpu().item() if val.dim() == 0 else val.detach().cpu().tolist()
    return val


def append_jsonl(path: str, record: dict):
    dir_path = os.path.dirname(path)
    if dir_path:
        _ensure_dir(dir_path)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


class JsonlMetricsCallback(TrainerCallback):
    """
    Log every Trainer on_log event to a JSONL file and optionally run train-set eval each epoch.
    """

    def __init__(
        self,
        log_path: str,
        train_dataset: Dataset = None,
        eval_dataset: Dataset = None,
        enable_train_eval: bool = True,
        trainer: RerankerTrainer = None,
        cli_args=None,
    ):
        self.log_path = log_path
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.enable_train_eval = enable_train_eval
        self.trainer = trainer
        self.cli_args = cli_args
        dir_path = os.path.dirname(log_path)
        if dir_path:
            _ensure_dir(dir_path)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        split = "train"
        if any(k.startswith("train_") for k in logs):
            split = "train_eval"
        elif any(k.startswith("eval_") for k in logs):
            split = "eval"
        record = {
            "step": state.global_step,
            "epoch": state.epoch,
            "split": split,
        }
        record.update({k: _to_serializable(v) for k, v in logs.items()})
        append_jsonl(self.log_path, record)

    def on_epoch_end(self, args, state, control, **kwargs):
        if not self.enable_train_eval or self.train_dataset is None:
            return
        trainer = self.trainer
        if trainer is None:
            return
        trainer.evaluate(eval_dataset=self.train_dataset, metric_key_prefix="train")
        if self.cli_args is not None:
            log_task_metrics(
                split="train",
                trainer=trainer,
                dataset=self.train_dataset,
                args=self.cli_args,
                step=state.global_step,
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        trainer = self.trainer
        if trainer is None or self.eval_dataset is None:
            return
        log_task_metrics(
            split="eval",
            trainer=trainer,
            dataset=self.eval_dataset,
            args=self.cli_args if self.cli_args is not None else trainer.args,
            step=state.global_step,
        )


def configure_logging(output_dir: str, log_level: str):
    """
    Attach a file handler under output_dir/train.log while keeping console logs.
    """
    _ensure_dir(output_dir)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # Ensure at least one stream handler
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(log_format)
        root_logger.addHandler(sh)
    log_path = os.path.join(output_dir, "train.log")
    if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        fh = logging.FileHandler(log_path)
        fh.setFormatter(log_format)
        root_logger.addHandler(fh)
    for h in root_logger.handlers:
        h.setLevel(log_level.upper())


def plot_metrics(log_path: str, output_dir: str, plot_filename: str = "metrics.png"):
    """
    Generate simple loss/metric curves from the JSONL log.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        logging.warning(f"Matplotlib not available, skip plotting: {e}")
        return

    if not os.path.exists(log_path):
        logging.warning(f"No metrics log found at {log_path}, skip plotting.")
        return

    train_steps, train_losses = [], []
    eval_loss_steps, eval_losses = [], []
    eval_score_steps, eval_scores = [], []
    train_eval_steps, train_eval_losses = [], []
    test_steps, test_losses = [], []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            split = rec.get("split")
            step = rec.get("step")
            if split == "train" and "loss" in rec:
                train_steps.append(step)
                train_losses.append(rec["loss"])
            if split == "eval":
                if "eval_loss" in rec:
                    eval_loss_steps.append(step)
                    eval_losses.append(rec["eval_loss"])
                # dev_score may be named eval_dev_score
                for key in ("eval_dev_score", "dev_score"):
                    if key in rec:
                        eval_score_steps.append(step)
                        eval_scores.append(rec[key])
                        break
            if split == "train_eval":
                if "train_loss" in rec:
                    train_eval_steps.append(step)
                    train_eval_losses.append(rec["train_loss"])
            if split == "test" and "test_loss" in rec:
                test_steps.append(step)
                test_losses.append(rec["test_loss"])

    if not train_steps and not eval_loss_steps and not eval_score_steps and not train_eval_steps and not test_steps:
        logging.warning("No metrics found to plot.")
        return

    plt.figure(figsize=(10, 6))
    if train_losses:
        plt.plot(train_steps, train_losses, label="train loss", alpha=0.8)
    if train_eval_losses:
        plt.plot(train_eval_steps, train_eval_losses, label="train eval loss", alpha=0.8)
    if eval_losses:
        plt.plot(eval_loss_steps, eval_losses, label="eval loss", alpha=0.8)
    if test_losses:
        plt.plot(test_steps, test_losses, label="test loss", linestyle="--", alpha=0.8)
    plt.xlabel("step")
    plt.ylabel("loss / metric")
    if eval_scores:
        plt.plot(eval_score_steps, eval_scores, label="eval dev_score", linestyle=":", alpha=0.8)
    plt.legend()
    plt.title("Training / Eval / Test metrics")
    plt.tight_layout()
    _ensure_dir(output_dir)
    fig_path = os.path.join(output_dir, plot_filename)
    plt.savefig(fig_path)
    plt.close()
    logging.info(f"Saved metrics plot to {fig_path}")


def main(args):
    """
    Main function for training and predicting using the hybrid-ml model.

    Args:
        args (argparse.Namespace): Command-line arguments parsed by argparse.

    Returns:
        None
    """
    if args.save_predictions and args.output_dir is None:
        raise ValueError("output_dir must be set to save predictions")

    _ensure_dir(args.output_dir)
    configure_logging(args.output_dir, args.log_level)
    seed_everything(args.seed)

    setup_device()
    tokenizer = build_tokenizer(args.model_name, cache_dir=args.cache_dir)
    data_collator = NClassCollator(
        args.source_maxlength, tokenizer, args.candidate_maxlength)
    train_dataset, eval_dataset, predict_dataset = load_datasets(args)
    model = initialize_model(args, tokenizer)
    trainer = initialize_training(
        args, model, train_dataset, eval_dataset, tokenizer, data_collator)

    if args.do_train:
        train(args, trainer, model)
    if args.do_predict:
        predict(args, trainer, predict_dataset, train_dataset=train_dataset)


def setup_device() -> torch.device:
    """
    Sets up the device for training.

    Returns:
        device (torch.device): The device to be used for training.
    """
    logging.info("Setting up device")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logging.info(f"device: {device}, n_gpu: {n_gpu}")
    return device


def initialize_model(args, tokenizer: PreTrainedTokenizer) -> NClassReranker:
    """
    Initializes the reranker model using the provided arguments and tokenizer.

    Args:
        args (argparse.Namespace): The command-line arguments.
        tokenizer (PreTrainedTokenizer): The tokenizer used for tokenization.

    Returns:
        NClassReranker: The initialized reranker model.
    """
    config = NClassRankerConfig()
    config.__dict__.update(
        {k: v for k, v in args.__dict__.items() if k in config.__dict__})

    if args.load_checkpoint:
        config = torch.load(os.path.join(args.load_checkpoint, "config.bin"))
        model = build_nclass_ranker_from_checkpoint(
            args.load_checkpoint, config, tokenizer
        )
    else:
        model = build_nclass_ranker(
            config.model_type,
            args.model_name,
            args.cache_dir,
            config,
            tokenizer,
        )
    return model


def initialize_training(
    args, model: NClassReranker, train_dataset: Dataset,
    eval_dataset: Dataset, tokenizer: PreTrainedTokenizer,
    data_collator: NClassCollator
) -> RerankerTrainer:
    """
    Initializes the training process for the reranker model.

    Args:
        args (argparse.Namespace): The command-line arguments.
        model (NClassReranker): The reranker model.
        train_dataset (Dataset): The training dataset.
        eval_dataset (Dataset): The evaluation dataset.
        tokenizer (PreTrainedTokenizer): The tokenizer used for tokenization.
        data_collator (NClassCollator): The data collator used for batching and padding.

    Returns:
        RerankerTrainer: The trainer object for training the reranker model.
    """
    _ensure_dir(args.output_dir)
    _ensure_dir(os.path.join(args.output_dir, "logs"))
    # Prepare TrainingArguments in a backward-compatible way by filtering kwargs
    ta_kwargs = dict(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        logging_first_step=args.logging_first_step,
        logging_strategy=args.logging_strategy,
        logging_dir=os.path.join(args.output_dir, "logs"),
        log_level=args.log_level,
        report_to=args.report_to,
        run_name=args.run_name,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        seed=args.seed,
        local_rank=args.local_rank,
        fp16=args.fp16,
        deepspeed=args.deepspeed,
        label_names=args.label_names,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        adafactor=args.adafactor,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        disable_tqdm=True,
        greater_is_better=True,
    )
    if not args.save_checkpoints:
        ta_kwargs["save_strategy"] = "no"
        ta_kwargs["save_steps"] = 0
        ta_kwargs["load_best_model_at_end"] = False
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard('self')
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in allowed}
    training_args = TrainingArguments(**ta_kwargs)

    trainer = RerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=loss2metric[args.loss_type],
    )
    metrics_log_path = os.path.join(args.output_dir, args.metrics_log_filename)
    callback = JsonlMetricsCallback(
        log_path=metrics_log_path,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        enable_train_eval=args.train_eval_each_epoch,
        trainer=trainer,
        cli_args=args,
    )
    trainer.add_callback(callback)

    return trainer


def load_datasets(args) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load datasets for training, evaluation, and prediction.

    Args:
        args: An object containing the arguments for loading datasets.

    Returns:
        A tuple containing the training dataset, evaluation dataset, and prediction dataset.
    """
    train_dataset = None
    eval_dataset = None
    predict_dataset = None

    def load_dataset(dataset_name: str, dataset_path: str, args, max_size: int = None):
        logging.info(
            f"Loading {dataset_name} dataset from path: {dataset_path}")
        examples = load_data(dataset_path, args, max_size=max_size)
        dataset = Dataset(examples, args.n_candidates)
        logging.info(
            f"{dataset_name.capitalize()} dataset loaded successfully")
        return dataset

    if args.do_train:
        train_dataset = load_dataset(
            "training", args.train_data_path, args, max_size=args.max_train_data_size)

    if args.do_eval:
        eval_dataset = load_dataset(
            "evaluation", args.eval_data_path, args, max_size=args.max_eval_data_size)
    else:
        args.evaluation_strategy = "no"
        args.save_strategy = "no"

    if args.do_predict:
        predict_dataset = load_dataset(
            "test", args.test_data_path, args, max_size=args.max_predict_data_size)

    if args.do_train:
        if args.do_eval:
            assert train_dataset.n_tasks == eval_dataset.n_tasks
        args.n_tasks = train_dataset.n_tasks
    elif args.do_predict:
        args.n_tasks = predict_dataset.n_tasks

    return train_dataset, eval_dataset, predict_dataset


def train(args, trainer: RerankerTrainer, model: NClassReranker):
    """
    Trains the model using the provided trainer and model instances.

    Args:
        args (object): The arguments object containing the training configuration.
        trainer (RerankerTrainer): The trainer instance used for training.
        model (NClassReranker): The model instance to be trained.

    Returns:
        None
    """
    if args.evaluate_before_training:
        metrics = trainer.evaluate()
        logging.info(f"Evaluate first step: \n{metrics}")

    logging.info("Start training")
    outputs = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    logging.info("Training finished")
    global_step, training_loss = outputs.global_step, outputs.training_loss
    metrics = outputs.metrics
    logging.info(
        f"global_step = {global_step}, average loss = {training_loss}")
    for key, value in metrics.items():
        logging.info(f"{key} = {value}")

    if args.save_checkpoints:
        logging.info("Saving model")
        best_checkpoint_folder = os.path.join(args.output_dir, "checkpoint-best")
        _ensure_dir(best_checkpoint_folder)
        trainer.save_model(best_checkpoint_folder)
        torch.save(model.args, os.path.join(best_checkpoint_folder, "config.bin"))
        logging.info(f"Model saved to {best_checkpoint_folder}")
    else:
        logging.info("Skipping checkpoint saving (save_checkpoints=False)")
    logging.info("Training finished")
    if args.plot_metrics:
        plot_metrics(
            log_path=os.path.join(args.output_dir, args.metrics_log_filename),
            output_dir=args.output_dir,
            plot_filename=args.plot_filename,
        )


def predict(args, trainer: RerankerTrainer, predict_dataset: Dataset, train_dataset: Dataset = None):
    """
    Perform prediction using the provided trainer on the given dataset.

    Args:
        args (object): The arguments object containing configuration options.
        trainer (object): The trainer object used for prediction.
        predict_dataset (object): The dataset used for prediction.

    Returns:
        None
    """
    logging.info("Start predicting")
    outputs = trainer.predict(predict_dataset)
    predictions = outputs.predictions
    labels = outputs.label_ids
    metrics = outputs.metrics
    logging.info(f"metrics: {metrics}")
    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved test metrics to {metrics_path}")
    append_jsonl(
        os.path.join(args.output_dir, args.metrics_log_filename),
        {
            "split": "test",
            "step": getattr(trainer.state, "global_step", None),
            **{k: _to_serializable(v) for k, v in metrics.items()},
        },
    )

    # save predictions
    if args.save_predictions:
        save_predictions(predictions, labels, args.output_dir)

    # Compute extended test report (dataset-wise metrics and costs)
    try:
        report = build_test_report(
            predictions=predictions,
            labels=labels,
            dataset=predict_dataset,
            loss_type=args.loss_type,
            inference_mode=args.inference_mode,
        )

        # Merge trainer-computed overall metrics for reference
        report["overall"].setdefault("trainer_metrics", metrics)

        # Log overall and macro cost statistics to console
        overall_cost = report.get("overall", {}).get("cost", {})
        macro_cost = report.get("macro", {}).get("cost", {})
        logging.info(
            f"Test overall cost: total={overall_cost.get('total')}, "
            f"avg_per_sample={overall_cost.get('avg_per_sample')}"
        )
        logging.info(
            f"Test macro cost: avg_total_per_dataset={macro_cost.get('avg_total_per_dataset')}, "
            f"avg_avg_per_sample={macro_cost.get('avg_avg_per_sample')}"
        )

        if args.save_test_report:
            os.makedirs(args.output_dir, exist_ok=True)
            report_path = os.path.join(args.output_dir, "test_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved test report to {report_path}")

        if args.track_test_best:
            best_path = os.path.join(args.output_dir, "test_best.json")
            primary_metric = args.test_primary_metric
            higher_is_better = (
                args.test_higher_is_better
                if args.test_higher_is_better is not None
                else (args.loss_type != NClassRerankerLossType.DET_NLABELS.value)
            )

            current_value = report.get("overall", {}).get(primary_metric)
            if current_value is None:
                logging.warning(
                    f"Primary metric '{primary_metric}' not found in overall report; skip best tracking."
                )
            else:
                best = None
                if os.path.exists(best_path):
                    try:
                        with open(best_path, "r", encoding="utf-8") as f:
                            best = json.load(f)
                    except Exception:
                        best = None

                def is_better(cv, bv):
                    if bv is None:
                        return True
                    return (cv > bv) if higher_is_better else (cv < bv)

                best_value = best.get("value") if isinstance(best, dict) else None
                if is_better(current_value, best_value):
                    payload = {
                        "primary_metric": primary_metric,
                        "value": current_value,
                        "higher_is_better": higher_is_better,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        "report": report,
                    }
                    with open(best_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                    # Save a copy of the report as best snapshot
                    best_report_path = os.path.join(args.output_dir, "test_report.best.json")
                    with open(best_report_path, "w", encoding="utf-8") as f:
                        json.dump(report, f, ensure_ascii=False, indent=2)
                    logging.info(
                        f"New best test performance: {primary_metric}={current_value}. Saved to {best_path}"
                    )
    except Exception as e:
        logging.exception(f"Failed to build/save extended test report: {e}")

    logging.info("Predicting finished")
    if args.plot_metrics:
        plot_metrics(
            log_path=os.path.join(args.output_dir, args.metrics_log_filename),
            output_dir=args.output_dir,
            plot_filename=args.plot_filename,
        )
    # Log task-averaged accuracies for test (and optionally train) splits
    log_task_metrics(
        split="test",
        trainer=trainer,
        dataset=predict_dataset,
        args=args,
        step=getattr(trainer.state, "global_step", None),
        predictions=predictions,
        labels=labels,
    )
    if args.log_train_metrics_on_predict and train_dataset is not None:
        log_task_metrics(
            split="train",
            trainer=trainer,
            dataset=train_dataset,
            args=args,
            step=getattr(trainer.state, "global_step", None),
        )


def save_predictions(predictions: torch.Tensor, labels: torch.Tensor, output_dir: str):
    """
    Save the predictions and labels to the specified output directory.

    Args:
        predictions (torch.Tensor): The predictions to be saved.
        labels (torch.Tensor): The labels to be saved.
        output_dir (str): The directory where the predictions and labels will be saved.

    Returns:
        None
    """
    logging.info("Saving predictions")
    with open(os.path.join(output_dir, "predictions.pt"), "wb") as f:
        torch.save(predictions, f)
    with open(os.path.join(output_dir, "labels.pt"), "wb") as f:
        torch.save(labels, f)
    logging.info(f"predictions saved to {output_dir}")


# ---------------------
# Extended reporting
# ---------------------

def _extract_dataset_name(sample) -> str:
    """
    Extract dataset name from sample id. Expected formats like 'dataset/1234'.
    Fallback to 'unknown' if not present.
    """
    try:
        sid = sample.get("id")
        if isinstance(sid, str) and "/" in sid:
            return sid.split("/")[0]
        return str(sid) if sid is not None else "unknown"
    except Exception:
        return "unknown"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _selected_indices(
    predictions,
    loss_type: str,
    inference_mode: str,
    n_candidates: int,
):
    """
    Derive selected candidate index per sample from raw predictions according to loss/inference semantics.
    Returns an array of shape [B] with integer indices in [0, n_candidates-1].
    """
    # Convert predictions to numpy
    def to_np(x):
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.array(x)

    if loss_type in (
        NClassRerankerLossType.DET_2CLS.value,
        NClassRerankerLossType.DET_NCLS.value,
        NClassRerankerLossType.PROB_2CLS.value,
    ):
        # predictions: [B, C]
        logits = to_np(predictions)
        return np.argmax(logits, axis=-1)

    if loss_type == NClassRerankerLossType.DET_NLABELS.value:
        # predictions: [B, n_candidates] logits for each candidate (multi-label)
        logits = to_np(predictions)
        prob = _sigmoid(logits)
        if inference_mode == "bubble":
            # pick first index with prob>=0.5 else last (reference)
            sel = np.full((prob.shape[0],), fill_value=n_candidates - 1, dtype=int)
            thresh = (prob >= 0.5)
            for i in range(prob.shape[0]):
                idx = np.argmax(thresh[i]) if np.any(thresh[i]) else (n_candidates - 1)
                sel[i] = int(idx)
            return sel
        else:  # full
            return np.argmax(prob, axis=-1)

    if loss_type == NClassRerankerLossType.PROB_NLABELS.value:
        # predictions: list/tuple of length n_candidates-1, each [B, 2] logits
        head_margins = []
        for p in predictions:
            arr = to_np(p)
            # margin: advantage over the last (reference) candidate
            head_margins.append(arr[:, 1] - arr[:, 0])
        margins = np.stack(head_margins, axis=1)  # [B, n_candidates-1]
        max_margin = margins.max(axis=1)
        argmax_margin = margins.argmax(axis=1)
        # If all margins < 0 -> choose last (reference), else choose best head index
        sel = np.where(max_margin < 0, n_candidates - 1, argmax_margin)
        return sel

    raise ValueError(f"Unknown loss_type for selection: {loss_type}")


def _compute_selection_metrics(scores_3d: np.ndarray, selected_idx: np.ndarray):
    """
    scores_3d: [B, n_candidates, window]
    selected_idx: [B]

    Returns dict with sel/oracle metrics and dev_score placeholder (to be filled by caller if needed).
    """
    B, C, W = scores_3d.shape
    # Aggregate per candidate per sample across window
    agg_scores = np.mean(scores_3d, axis=-1)  # [B, C]

    # ranks: 0 is best
    sort_indices = np.flip(np.argsort(agg_scores, axis=-1), axis=-1)
    ranks = np.zeros_like(sort_indices)
    rows = np.arange(B)[:, None]
    ranks[rows, sort_indices] = np.arange(C)

    sel_scores = scores_3d[np.arange(B), selected_idx]  # [B, W]
    oracle_idx = np.argmax(agg_scores, axis=1)  # [B]
    oracle_scores = scores_3d[np.arange(B), oracle_idx]

    sel_acc = float(np.mean(ranks[np.arange(B), selected_idx] == 0))
    oracle_acc = 1.0  # by definition selecting best

    metrics = {
        "sel": {
            "avg_score": float(np.mean(sel_scores)),
            "acc": sel_acc,
            "avg_ind": float(np.mean(selected_idx)),
        },
        "oracle": {
            "avg_score": float(np.mean(oracle_scores)),
            "acc": oracle_acc,
            "avg_ind": float(np.mean(oracle_idx)),
        },
    }
    return metrics, agg_scores


def _dev_score_for_group(loss_type: str, predictions, labels_group):
    """
    Compute dev_score consistent with existing compute_metrics_* but for a subset group.
    labels_group: numpy array with shape like original labels, sliced to group.
    """
    # Align with existing metrics implementations:
    # labels[..., 0] picks first metric; mean over last axis averages window
    scores = labels_group[..., 0]
    agg_scores = np.mean(scores, axis=-1)  # [B, C]

    if loss_type == NClassRerankerLossType.PROB_NLABELS.value:
        # average Spearman across heads
        from scipy.stats import spearmanr

        pred_scores = predictions  # list of [B, 2]
        pred_diff_list = []
        agg_score_diff_list = []
        for i, p in enumerate(pred_scores):
            p_arr = p if isinstance(p, np.ndarray) else (
                p.detach().cpu().numpy() if torch.is_tensor(p) else np.array(p)
            )
            pred_diff_list.append(p_arr[:, 1] - p_arr[:, 0])  # [B]
            agg_score_diff_list.append(agg_scores[:, -1] - agg_scores[:, i])
        # average correlation across heads (ignore NaN)
        corr = []
        for a, b in zip(pred_diff_list, agg_score_diff_list):
            c = spearmanr(a, b).correlation
            if not (c is None or np.isnan(c)):
                corr.append(c)
        return float(np.mean(corr)) if len(corr) > 0 else float("nan")

    if loss_type == NClassRerankerLossType.DET_NCLS.value:
        from sklearn.metrics import accuracy_score
        pred = predictions if isinstance(predictions, np.ndarray) else (
            predictions.detach().cpu().numpy() if torch.is_tensor(predictions) else np.array(predictions)
        )
        sel = np.argmax(pred, axis=-1)
        true_labels = np.argmax(agg_scores, axis=-1)
        return float(accuracy_score(true_labels, sel))

    if loss_type == NClassRerankerLossType.DET_NLABELS.value:
        # MSE(sigmoid(pred), true_multi_hot)
        pred = predictions if isinstance(predictions, np.ndarray) else (
            predictions.detach().cpu().numpy() if torch.is_tensor(predictions) else np.array(predictions)
        )
        prob = _sigmoid(pred)
        true = np.array([[(1.0 if _ >= s[-1] else 0.0) for _ in s] for s in agg_scores])
        return float(np.mean((prob - true) ** 2))

    if loss_type in (NClassRerankerLossType.DET_2CLS.value, NClassRerankerLossType.PROB_2CLS.value):
        from scipy.stats import spearmanr
        pred = predictions if isinstance(predictions, np.ndarray) else (
            predictions.detach().cpu().numpy() if torch.is_tensor(predictions) else np.array(predictions)
        )
        # Flatten pairwise differences of predicted vs agg
        pred_diff_list = []
        agg_score_diff_list = []
        for p, s in zip(pred, agg_scores):
            pred_diff_list += [p[-1] - _ for _ in p[:-1]]
            agg_score_diff_list += [s[-1] - _ for _ in s[:-1]]
        corr = spearmanr(np.array(pred_diff_list), np.array(agg_score_diff_list)).correlation
        return float(corr) if corr is not None else float("nan")

    raise ValueError(f"Unknown loss_type for dev_score: {loss_type}")


def _compute_costs(selected_idx: np.ndarray, dataset: Dataset):
    """
    Compute cost per-sample (mean over window), per-dataset totals and overall totals.
    Returns (per_sample_costs, by_dataset_cost, total_cost)
    """
    per_sample = []
    by_dataset = {}
    data_ref = dataset.data
    for i, idx in enumerate(selected_idx):
        sample = data_ref[i]
        ds_name = _extract_dataset_name(sample)
        cands = sample.get("candidates", [])
        if not cands or idx >= len(cands):
            cost = 0.0
        else:
            cost_list = cands[idx].get("cost", [])
            # average over window if available
            cost = float(np.mean(cost_list)) if len(cost_list) > 0 else 0.0
        per_sample.append(cost)
        by_dataset.setdefault(ds_name, 0.0)
        by_dataset[ds_name] += cost
    total_cost = float(np.sum(per_sample))
    return np.array(per_sample, dtype=float), by_dataset, total_cost


def build_test_report(predictions, labels, dataset: Dataset, loss_type: str, inference_mode: str):
    """
    Build a comprehensive test report including:
    - per-dataset performance (sel/oracle/dev_score) and costs
    - macro (dataset-level average)
    - micro (overall sample-level average)
    """
    # Normalize labels to numpy
    if torch.is_tensor(labels):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.array(labels)

    # labels[..., 0] is the chosen metric, last dim is metrics; mean over last is over window
    scores = labels_np[..., 0]  # [B, C, W]
    if scores.ndim != 3:
        # Attempt to reshape if missing window dimension (edge cases)
        # Fallback: treat as single-window
        if scores.ndim == 2:
            scores = scores[..., None]
        else:
            raise ValueError(f"Unexpected scores shape: {scores.shape}")

    n_samples, n_candidates, _ = scores.shape

    # Compute selected indices from predictions
    sel_idx = _selected_indices(
        predictions=predictions,
        loss_type=loss_type,
        inference_mode=inference_mode,
        n_candidates=n_candidates,
    )

    # Overall selection metrics (micro)
    overall_sel_metrics, agg_scores = _compute_selection_metrics(scores, sel_idx)
    overall_dev_score = _dev_score_for_group(loss_type, predictions, labels_np)

    per_sample_cost, by_dataset_cost_total, total_cost = _compute_costs(sel_idx, dataset)
    overall = {
        **overall_sel_metrics,
        "dev_score": float(overall_dev_score),
        "cost": {
            "total": float(total_cost),
            "avg_per_sample": float(np.mean(per_sample_cost)),
        },
        "n_samples": int(n_samples),
    }

    # Group by dataset
    groups = {}
    for i, sample in enumerate(dataset.data):
        ds = _extract_dataset_name(sample)
        groups.setdefault(ds, []).append(i)

    by_dataset = {}
    macro_accumulate = {
        "sel_avg_score": [],
        "sel_acc": [],
        "sel_avg_ind": [],
        "oracle_avg_score": [],
        "oracle_acc": [],
        "oracle_avg_ind": [],
        "dev_score": [],
        "cost_total": [],
        "cost_avg_per_sample": [],
        "n_samples": [],
    }

    for ds, idxs in groups.items():
        idxs_arr = np.array(idxs, dtype=int)
        scores_g = scores[idxs_arr]  # [b, C, W]
        sel_idx_g = sel_idx[idxs_arr]
        labels_g = labels_np[idxs_arr]

        sel_metrics_g, _ = _compute_selection_metrics(scores_g, sel_idx_g)
        dev_g = _dev_score_for_group(loss_type, (
            # Slice predictions for group
            [p[idxs_arr] if (isinstance(p, np.ndarray) or torch.is_tensor(p)) else np.array(p)[idxs_arr] for p in predictions]
            if loss_type == NClassRerankerLossType.PROB_NLABELS.value
            else (predictions[idxs_arr] if (isinstance(predictions, np.ndarray) or torch.is_tensor(predictions)) else np.array(predictions)[idxs_arr])
        ), labels_g)

        # Cost for this group
        cost_group_samples = per_sample_cost[idxs_arr]
        cost_total = float(np.sum(cost_group_samples))
        cost_avg = float(np.mean(cost_group_samples)) if len(cost_group_samples) > 0 else 0.0

        by_dataset[ds] = {
            **sel_metrics_g,
            "dev_score": float(dev_g),
            "cost": {"total": cost_total, "avg_per_sample": cost_avg},
            "n_samples": int(len(idxs_arr)),
        }

        # accumulate for macro
        macro_accumulate["sel_avg_score"].append(sel_metrics_g["sel"]["avg_score"])
        macro_accumulate["sel_acc"].append(sel_metrics_g["sel"]["acc"])
        macro_accumulate["sel_avg_ind"].append(sel_metrics_g["sel"]["avg_ind"])
        macro_accumulate["oracle_avg_score"].append(sel_metrics_g["oracle"]["avg_score"])
        macro_accumulate["oracle_acc"].append(sel_metrics_g["oracle"]["acc"])
        macro_accumulate["oracle_avg_ind"].append(sel_metrics_g["oracle"]["avg_ind"])
        macro_accumulate["dev_score"].append(float(dev_g))
        macro_accumulate["cost_total"].append(cost_total)
        macro_accumulate["cost_avg_per_sample"].append(cost_avg)
        macro_accumulate["n_samples"].append(int(len(idxs_arr)))

    # Macro: dataset-level average (equal weight per dataset)
    def _avg(arr):
        return float(np.mean(arr)) if len(arr) > 0 else 0.0

    macro = {
        "sel": {
            "avg_score": _avg(macro_accumulate["sel_avg_score"]),
            "acc": _avg(macro_accumulate["sel_acc"]),
            "avg_ind": _avg(macro_accumulate["sel_avg_ind"]),
        },
        "oracle": {
            "avg_score": _avg(macro_accumulate["oracle_avg_score"]),
            "acc": _avg(macro_accumulate["oracle_acc"]),
            "avg_ind": _avg(macro_accumulate["oracle_avg_ind"]),
        },
        "dev_score": _avg(macro_accumulate["dev_score"]),
        "cost": {
            "avg_total_per_dataset": _avg(macro_accumulate["cost_total"]),
            "avg_avg_per_sample": _avg(macro_accumulate["cost_avg_per_sample"]),
        },
        "n_datasets": len(groups),
    }

    report = {
        "overall": overall,
        "by_dataset": by_dataset,
        "macro": macro,
    }
    return report


def split_comma_separated_arg(arg: str) -> List[str]:
    """
    Splits a comma-separated string into a list of individual values.

    Args:
        arg (str): The comma-separated string to be split.

    Returns:
        list: A list of individual values extracted from the comma-separated string.
              Returns None if the input string is None.
    """
    return arg.split(",") if arg is not None else None


def log_task_metrics(
    split: str,
    trainer: RerankerTrainer,
    dataset: Dataset,
    args,
    step: int = None,
    predictions=None,
    labels=None,
):
    """
    Run predict (if needed), build test report, and append macro / per-dataset accuracies to metrics log.
    """
    if dataset is None:
        return
    # Use provided preds/labels or run inference
    if predictions is None or labels is None:
        outputs = trainer.predict(dataset)
        predictions = outputs.predictions
        labels = outputs.label_ids
    report = build_test_report(
        predictions=predictions,
        labels=labels,
        dataset=dataset,
        loss_type=args.loss_type,
        inference_mode=args.inference_mode,
    )
    macro_acc = report.get("macro", {}).get("sel", {}).get("acc")
    overall_acc = report.get("overall", {}).get("sel", {}).get("acc")
    overall_cost = report.get("overall", {}).get("cost", {}) or {}
    macro_cost = report.get("macro", {}).get("cost", {}) or {}
    by_ds = {
        ds: metrics.get("sel", {}).get("acc")
        for ds, metrics in report.get("by_dataset", {}).items()
    }
    payload = {
        "split": split,
        "step": step,
        "macro_sel_acc": _to_serializable(macro_acc),
        "overall_sel_acc": _to_serializable(overall_acc),
        "by_dataset_sel_acc": {k: _to_serializable(v) for k, v in by_ds.items()},
        "overall_cost": {
            "total": _to_serializable(overall_cost.get("total")),
            "avg_per_sample": _to_serializable(overall_cost.get("avg_per_sample")),
        },
        "macro_cost": {
            "avg_total_per_dataset": _to_serializable(macro_cost.get("avg_total_per_dataset")),
            "avg_avg_per_sample": _to_serializable(macro_cost.get("avg_avg_per_sample")),
        },
    }
    append_jsonl(os.path.join(args.output_dir, args.metrics_log_filename), payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument("--model_name", type=str,
                        default="/fs-computility-new/Uma4agi/shared/models/gte_Qwen2-7B-instruct")
    parser.add_argument("--load_checkpoint", type=empty2None, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=[loss_type.value for loss_type in NClassRerankerLossType],
        default=NClassRerankerLossType.DET_2CLS.value,
    )

    # data config
    parser.add_argument("--match_t", type=float, default=0.0)
    parser.add_argument("--upsample_major", type=str, default="")
    parser.add_argument("--upsample_minor", type=str, default="")
    parser.add_argument(
        "--candidate_decoding_method",
        type=empty2None,
        default=None,
        help="separted by comma. Empty string for all methods",
    )
    parser.add_argument("--n_candidates", type=int, default=-1)
    parser.add_argument(
        "--candidate_models",
        type=empty2None,
        default=None,
        help="Small and Large model separted by comma",
    )
    parser.add_argument("--source_maxlength", type=int, default=128)
    parser.add_argument("--candidate_maxlength", type=int, default=128)
    parser.add_argument("--num_pos", type=int, default=5)
    parser.add_argument("--num_neg", type=int, default=5)
    parser.add_argument("--sub_sampling_ratio", type=float, default=0.4)
    parser.add_argument(
        "--sub_sampling_mode",
        type=str,
        choices=[
            "uniform",
            "top_bottom",
            "top_random",
            "random_bottom",
            "random",
            "uniform",
            "all_pair",
        ],
        default="all_pair",
    )
    parser.add_argument("--max_train_data_size", type=int, default=-1)
    parser.add_argument("--max_eval_data_size", type=int, default=-1)
    parser.add_argument("--max_predict_data_size", type=int, default=-1)
    parser.add_argument(
        "--quality_metric",
        type=str,
        default="bartscore",
        help="Response quality metric, default: bartscore",
    )

    # running config
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", type=str2bool, default=True)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank. Necessary for using the torch.distributed.launch utility.",
    )

    # mode
    parser.add_argument("--do_train", type=str2bool, default=True)
    parser.add_argument("--do_eval", type=str2bool, default=True)
    parser.add_argument("--do_predict", type=str2bool, default=True)

    # training hyperparameters
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=10e10)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
        default="linear",
    )
    parser.add_argument("--adafactor", type=bool, default=True)

    # evaluation hyperparameters
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--evaluate_before_training",
                        type=str2bool, default=False)
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        choices=["steps", "epoch", "no"],
        default="epoch",
    )
    parser.add_argument("--eval_steps", type=int, default=0)

    # predict config
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--save_predictions", type=str2bool, default=True)
    parser.add_argument("--save_test_report", type=str2bool, default=True)
    parser.add_argument("--track_test_best", type=str2bool, default=True)
    parser.add_argument("--test_primary_metric", type=str, default="dev_score")
    parser.add_argument(
        "--test_higher_is_better",
        type=lambda s: None if s in (None, "", "none", "None") else str2bool(s),
        default=None,
        help="Override comparison direction for best test metric (true/false). Default: infer from loss_type (det_nlabels -> minimize).",
    )

    # logging
    parser.add_argument("--logging_first_step", type=str2bool, default=True)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument(
        "--logging_strategy",
        type=str,
        choices=["steps", "epoch", "no"],
        default="steps",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="info",
        choices=["info", "debug", "warning", "error", "critical"],
    )
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--run_name", type=str,
                        default="seed42_split0.7")  # wandb run name
    parser.add_argument("--metrics_log_filename", type=str, default="metrics_log.jsonl")
    parser.add_argument("--plot_filename", type=str, default="metrics.png")
    parser.add_argument("--plot_metrics", type=str2bool, default=True)
    parser.add_argument("--train_eval_each_epoch", type=str2bool, default=True)
    parser.add_argument("--log_train_metrics_on_predict", type=str2bool, default=True)

    # save config
    parser.add_argument("--output_dir", type=str, default="./outputs/seed42_split0.7")
    parser.add_argument("--overwrite_output_dir", type=str2bool, default=True)
    parser.add_argument(
        "--save_strategy", type=str, choices=["steps", "epoch", "no"], default="epoch"
    )
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--save_total_limit", type=int, default=4)
    parser.add_argument(
        "--save_checkpoints",
        type=str2bool,
        default=False,
        help="Whether to write checkpoint weights to disk. Default: False (no checkpoint saving).",
    )

    # metrics config
    parser.add_argument("--load_best_model_at_end",
                        type=str2bool, default=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--metric_for_best_model",
                        type=str, default="dev_score")

    # inference config
    parser.add_argument(
        "--inference_mode", type=str, default="bubble", choices=["bubble", "full"]
    )

    # init args
    args = parser.parse_args()
    args.ranker_type = "nclass"
    # Note: Removed hardcoded model_type="deberta" to allow auto-detection
    # The model type will be automatically detected based on the model_name

    args.load_best_model_at_end = args.do_train and args.do_predict and args.save_checkpoints

    # set up default output dir
    if args.output_dir is None:
        args.output_dir = f"outputs/{args.run_name}"
    args.cache_dir = f"./hf_models/{args.model_name.split('/')[-1]}/"
    args.label_names = ["scores"]
    args.candidate_decoding_methods = split_comma_separated_arg(
        args.candidate_decoding_method
    )
    args.candidate_models = split_comma_separated_arg(args.candidate_models)
    args.n_candidates = (
        len(args.candidate_models)
        if args.candidate_models is not None
        else args.n_candidates
    )
    args.local_rank = os.environ.get("LOCAL_RANK", args.local_rank)
    args.metrics = split_comma_separated_arg(args.quality_metric)
    # args.is_prob = args.loss_type == NClassRerankerLossType.PROB_2CLS.value

    # set up logging
    logging.basicConfig(level=args.log_level.upper())

    main(args)
