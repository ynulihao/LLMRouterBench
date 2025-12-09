"""Batch evaluation helper for the Matrix Factorization router.

This utility loads a trained MF checkpoint via :class:`Controller`,
then sweeps over a pairwise comparison dataset (e.g. adaptor output)
to report routing accuracy and related statistics.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from loguru import logger

from baselines.RouteLLM.controller import Controller


@dataclass
class EvaluationResult:
    total: int
    ties: int
    decisive_total: int
    selected_correct_total: int
    selected_correct_decisive: int
    routing_correct: int
    selection_accuracy: float
    selection_accuracy_decisive: float
    routing_accuracy: float
    strong_selected: int
    weak_selected: int
    avg_winrate: float
    winrate_std: float
    total_cost: float
    avg_cost: float
    datasets: Dict[str, "DatasetEvaluationResult"]
    indomain_avg: Optional[float] = None
    ood_avg: Optional[float] = None
    indomain_sample_avg: Optional[float] = None
    ood_sample_avg: Optional[float] = None
    all_dataset_avg: Optional[float] = None
    sample_avg: Optional[float] = None
    indomain_total_cost: Optional[float] = None
    indomain_avg_cost: Optional[float] = None
    ood_total_cost: Optional[float] = None
    ood_avg_cost: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        return {
            "total": self.total,
            "ties": self.ties,
            "decisive_total": self.decisive_total,
            "selected_correct_total": self.selected_correct_total,
            "selected_correct_decisive": self.selected_correct_decisive,
            "routing_correct": self.routing_correct,
            "selection_accuracy": self.selection_accuracy,
            "selection_accuracy_decisive": self.selection_accuracy_decisive,
            "routing_accuracy": self.routing_accuracy,
            "strong_selected": self.strong_selected,
            "weak_selected": self.weak_selected,
            "avg_winrate": self.avg_winrate,
            "winrate_std": self.winrate_std,
            "total_cost": self.total_cost,
            "avg_cost": self.avg_cost,
            "indomain_avg": self.indomain_avg,
            "ood_avg": self.ood_avg,
            "indomain_sample_avg": self.indomain_sample_avg,
            "ood_sample_avg": self.ood_sample_avg,
            "all_dataset_avg": self.all_dataset_avg,
            "sample_avg": self.sample_avg,
            "indomain_total_cost": self.indomain_total_cost,
            "indomain_avg_cost": self.indomain_avg_cost,
            "ood_total_cost": self.ood_total_cost,
            "ood_avg_cost": self.ood_avg_cost,
            "datasets": {
                dataset: result.to_dict() for dataset, result in self.datasets.items()
            },
        }


@dataclass
class DatasetEvaluationResult:
    total: int
    ties: int
    decisive_total: int
    selected_correct_total: int
    selected_correct_decisive: int
    routing_correct: int
    selection_accuracy: float
    selection_accuracy_decisive: float
    routing_accuracy: float
    strong_selected: int
    weak_selected: int
    avg_winrate: float
    winrate_std: float
    total_cost: float
    avg_cost: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "total": self.total,
            "ties": self.ties,
            "decisive_total": self.decisive_total,
            "selected_correct_total": self.selected_correct_total,
            "selected_correct_decisive": self.selected_correct_decisive,
            "routing_correct": self.routing_correct,
            "selection_accuracy": self.selection_accuracy,
            "selection_accuracy_decisive": self.selection_accuracy_decisive,
            "routing_accuracy": self.routing_accuracy,
            "strong_selected": self.strong_selected,
            "weak_selected": self.weak_selected,
            "avg_winrate": self.avg_winrate,
            "winrate_std": self.winrate_std,
            "total_cost": self.total_cost,
            "avg_cost": self.avg_cost,
        }


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def infer_pairwise_path(data_dir: Path | None, explicit_file: Path | None) -> Path:
    if explicit_file:
        return explicit_file
    if not data_dir:
        raise ValueError("Either --pairwise-file or --data-dir must be provided.")
    return data_dir / "pairwise_test.json"


def evaluate(
    controller: Controller,
    samples: List[Dict[str, object]],
    router: str,
    threshold: float,
) -> EvaluationResult:
    if not samples:
        raise ValueError("No pairwise samples provided for evaluation.")

    router_instance = controller.routers[router]
    total = len(samples)
    ties = 0
    decisive_total = 0
    selected_correct_total = 0
    selected_correct_decisive = 0
    routing_correct = 0
    strong_selected = 0
    weak_selected = 0
    winrates: List[float] = []
    total_cost = 0.0
    dataset_stats = defaultdict(
        lambda: {
            "total": 0,
            "ties": 0,
            "decisive_total": 0,
            "selected_correct_total": 0,
            "selected_correct_decisive": 0,
            "routing_correct": 0,
            "strong": 0,
            "weak": 0,
            "winrates": [],
            "cost": 0.0,
        }
    )

    def is_correct(score: object) -> bool:
        try:
            return float(score) > 0
        except (TypeError, ValueError):
            return False

    for sample in samples:
        prompt = sample.get("prompt") or sample.get("origin_query")
        if not prompt:
            raise KeyError("Sample missing 'prompt' (or fallback 'origin_query').")

        winrate = router_instance.calculate_strong_win_rate(prompt)
        winrates.append(winrate)
        predicted = "model_a" if winrate >= threshold else "model_b"

        dataset_id = sample.get("dataset_id", "unknown")
        stats = dataset_stats[dataset_id]
        stats["total"] += 1
        stats["winrates"].append(winrate)

        if predicted == "model_a":
            strong_selected += 1
            stats["strong"] += 1
        else:
            weak_selected += 1
            stats["weak"] += 1

        winner = sample.get("winner")
        if winner not in {"model_a", "model_b", "tie"}:
            raise ValueError(f"Unexpected winner label: {winner}")

        score_a = sample.get("score_model_a")
        score_b = sample.get("score_model_b")

        model_a_correct = is_correct(score_a)
        model_b_correct = is_correct(score_b)
        predicted_correct = model_a_correct if predicted == "model_a" else model_b_correct

        if predicted_correct:
            selected_correct_total += 1
            stats["selected_correct_total"] += 1

        if winner == "tie":
            ties += 1
            stats["ties"] += 1
        else:
            decisive_total += 1
            stats["decisive_total"] += 1
            if predicted == winner:
                routing_correct += 1
                stats["routing_correct"] += 1
            if predicted_correct:
                selected_correct_decisive += 1
                stats["selected_correct_decisive"] += 1

        strong_cost = float(sample.get("cost_model_a") or 0.0)
        weak_cost = float(sample.get("cost_model_b") or 0.0)
        sample_cost = strong_cost if predicted == "model_a" else weak_cost
        total_cost += sample_cost
        stats["cost"] += sample_cost

    selection_accuracy = selected_correct_total / total
    selection_accuracy_decisive = (
        selected_correct_decisive / decisive_total if decisive_total else 0.0
    )
    routing_accuracy = routing_correct / decisive_total if decisive_total else 0.0
    avg_winrate = sum(winrates) / total
    # Use population variance (ddof=0) to avoid dependency on numpy
    variance = (
        sum((w - avg_winrate) ** 2 for w in winrates) / total if total > 1 else 0.0
    )
    winrate_std = variance**0.5

    dataset_results: Dict[str, DatasetEvaluationResult] = {}
    for dataset_id, stats in sorted(dataset_stats.items()):
        ds_total = stats["total"]
        ds_avg_winrate = sum(stats["winrates"]) / ds_total if ds_total else 0.0
        ds_variance = (
            sum((w - ds_avg_winrate) ** 2 for w in stats["winrates"]) / ds_total
            if ds_total > 1
            else 0.0
        )
        ds_std = ds_variance**0.5
        ds_total_cost = stats["cost"]
        ds_ties = stats["ties"]
        ds_decisive_total = stats["decisive_total"]
        ds_selected_correct_total = stats["selected_correct_total"]
        ds_selected_correct_decisive = stats["selected_correct_decisive"]
        ds_routing_correct = stats["routing_correct"]
        ds_selection_accuracy = (
            ds_selected_correct_total / ds_total if ds_total else 0.0
        )
        ds_selection_accuracy_decisive = (
            ds_selected_correct_decisive / ds_decisive_total
            if ds_decisive_total
            else 0.0
        )
        ds_routing_accuracy = (
            ds_routing_correct / ds_decisive_total if ds_decisive_total else 0.0
        )
        dataset_results[dataset_id] = DatasetEvaluationResult(
            total=ds_total,
            ties=ds_ties,
            decisive_total=ds_decisive_total,
            selected_correct_total=ds_selected_correct_total,
            selected_correct_decisive=ds_selected_correct_decisive,
            routing_correct=ds_routing_correct,
            selection_accuracy=ds_selection_accuracy,
            selection_accuracy_decisive=ds_selection_accuracy_decisive,
            routing_accuracy=ds_routing_accuracy,
            strong_selected=stats["strong"],
            weak_selected=stats["weak"],
            avg_winrate=ds_avg_winrate,
            winrate_std=ds_std,
            total_cost=ds_total_cost,
            avg_cost=ds_total_cost / ds_total if ds_total else 0.0,
        )

    def is_ood(dataset_id: str) -> bool:
        return dataset_id.startswith("arenahard")

    def average_accuracy(dataset_ids):
        accuracies = [
            dataset_results[d].selection_accuracy
            for d in dataset_ids
            if dataset_results[d].total > 0
        ]
        if not accuracies:
            return None
        return sum(accuracies) / len(accuracies)

    def sample_accuracy(dataset_ids):
        total_correct = 0
        total_samples = 0
        for d in dataset_ids:
            result = dataset_results[d]
            total_correct += result.selected_correct_total
            total_samples += result.total
        if total_samples == 0:
            return None
        return total_correct / total_samples

    dataset_ids = sorted(dataset_results.keys())
    indomain_ids = [d for d in dataset_ids if not is_ood(d)]
    ood_ids = [d for d in dataset_ids if is_ood(d)]

    indomain_avg = average_accuracy(indomain_ids)
    ood_avg = average_accuracy(ood_ids)
    indomain_sample_avg = sample_accuracy(indomain_ids)
    ood_sample_avg = sample_accuracy(ood_ids)
    all_dataset_avg = average_accuracy(dataset_ids)
    sample_avg = selected_correct_total / total if total else None

    def cost_stats(dataset_ids):
        total = sum(dataset_results[d].total_cost for d in dataset_ids)
        count = sum(1 for d in dataset_ids if dataset_results[d].total > 0)
        avg = total / count if count else None
        if not dataset_ids:
            total = None
        return total, avg

    indomain_total_cost, indomain_avg_cost = cost_stats(indomain_ids)
    ood_total_cost, ood_avg_cost = cost_stats(ood_ids)

    return EvaluationResult(
        total=total,
        ties=ties,
        decisive_total=decisive_total,
        selected_correct_total=selected_correct_total,
        selected_correct_decisive=selected_correct_decisive,
        routing_correct=routing_correct,
        selection_accuracy=selection_accuracy,
        selection_accuracy_decisive=selection_accuracy_decisive,
        routing_accuracy=routing_accuracy,
        strong_selected=strong_selected,
        weak_selected=weak_selected,
        avg_winrate=avg_winrate,
        winrate_std=winrate_std,
        total_cost=total_cost,
        avg_cost=total_cost / total if total else 0.0,
        datasets=dataset_results,
        indomain_avg=indomain_avg,
        ood_avg=ood_avg,
        indomain_sample_avg=indomain_sample_avg,
        ood_sample_avg=ood_sample_avg,
        all_dataset_avg=all_dataset_avg,
        sample_avg=sample_avg,
        indomain_total_cost=indomain_total_cost,
        indomain_avg_cost=indomain_avg_cost,
        ood_total_cost=ood_total_cost,
        ood_avg_cost=ood_avg_cost,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MF router on pairwise comparison data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="baselines/RouteLLM/router_eval_config.json",
        help="Path to router configuration JSON (Controller-compatible).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing pairwise_{train,test}.json (adaptor output).",
    )
    parser.add_argument(
        "--pairwise-file",
        type=str,
        default=None,
        help="Explicit path to pairwise comparison JSON (overrides --data-dir).",
    )
    parser.add_argument(
        "--router",
        type=str,
        default="mf",
        help="Router key to evaluate (must exist in the config).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Routing threshold for deciding strong vs. weak model.",
    )
    parser.add_argument(
        "--strong-model",
        type=str,
        required=True,
        help="Strong model identifier used during adaptor conversion.",
    )
    parser.add_argument(
        "--weak-model",
        type=str,
        required=True,
        help="Weak model identifier used during adaptor conversion.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save evaluation metrics as JSON.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Enable Controller progress bars.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Router config not found: {config_path}")
    router_config = load_json(config_path)

    pairwise_path = infer_pairwise_path(
        Path(args.data_dir).expanduser() if args.data_dir else None,
        Path(args.pairwise_file).expanduser() if args.pairwise_file else None,
    )
    if not pairwise_path.exists():
        raise FileNotFoundError(f"Pairwise comparison file not found: {pairwise_path}")

    samples = load_json(pairwise_path)
    logger.info(
        "Loaded {} pairwise samples from {}",
        len(samples),
        pairwise_path,
    )

    controller = Controller(
        routers=[args.router],
        strong_model=args.strong_model,
        weak_model=args.weak_model,
        config=router_config,
        progress_bar=args.progress,
    )

    result = evaluate(controller, samples, args.router, args.threshold)
    logger.info("Evaluation complete:")
    logger.info(
        "  Selection accuracy (all prompts)      : {:.2f}% ({}/{})",
        result.selection_accuracy * 100,
        result.selected_correct_total,
        result.total,
    )
    if result.decisive_total:
        logger.info(
            "  Selection accuracy (decisive prompts): {:.2f}% ({}/{})",
            result.selection_accuracy_decisive * 100,
            result.selected_correct_decisive,
            result.decisive_total,
        )
        logger.info(
            "  Routing accuracy (decisive prompts)  : {:.2f}% ({}/{})",
            result.routing_accuracy * 100,
            result.routing_correct,
            result.decisive_total,
        )
    else:
        logger.info("  Selection/Routing accuracy (decisive) : N/A (0 decisive samples)")
    logger.info("  Ties                                 : {}", result.ties)
    logger.info("  Strong selections                    : {}", result.strong_selected)
    logger.info("  Weak selections                      : {}", result.weak_selected)
    logger.info("  Avg winrate                          : {:.4f}", result.avg_winrate)
    logger.info("  Winrate std dev                      : {:.4f}", result.winrate_std)
    logger.info("  Total cost                           : ${:.4f}", result.total_cost)
    logger.info("  Avg cost per q                       : ${:.4f}", result.avg_cost)

    logger.info("Aggregate dataset metrics:")
    if result.all_dataset_avg is not None:
        logger.info("  All-dataset avg accuracy             : {:.2f}%", result.all_dataset_avg * 100)
    else:
        logger.info("  All-dataset avg accuracy             : N/A")
    if result.sample_avg is not None:
        logger.info("  Sample avg accuracy                  : {:.2f}%", result.sample_avg * 100)
    else:
        logger.info("  Sample avg accuracy                  : N/A")
    if result.indomain_avg is not None:
        logger.info("  In-domain dataset avg accuracy       : {:.2f}%", result.indomain_avg * 100)
    else:
        logger.info("  In-domain dataset avg accuracy       : N/A")
    if result.indomain_sample_avg is not None:
        logger.info(
            "  In-domain sample avg accuracy        : {:.2f}%",
            result.indomain_sample_avg * 100,
        )
    else:
        logger.info("  In-domain sample avg accuracy        : N/A")
    if result.ood_avg is not None:
        logger.info("  OOD dataset avg accuracy             : {:.2f}%", result.ood_avg * 100)
    else:
        logger.info("  OOD dataset avg accuracy             : N/A")
    if result.ood_sample_avg is not None:
        logger.info("  OOD sample avg accuracy              : {:.2f}%", result.ood_sample_avg * 100)
    else:
        logger.info("  OOD sample avg accuracy              : N/A")
    if result.indomain_total_cost is not None:
        logger.info(
            "  In-domain total cost                 : ${:.4f}", result.indomain_total_cost
        )
    else:
        logger.info("  In-domain total cost                 : N/A")
    if result.indomain_avg_cost is not None:
        logger.info(
            "  In-domain avg cost                   : ${:.4f}", result.indomain_avg_cost
        )
    else:
        logger.info("  In-domain avg cost                   : N/A")
    if result.ood_total_cost is not None:
        logger.info("  OOD total cost                       : ${:.4f}", result.ood_total_cost)
    else:
        logger.info("  OOD total cost                       : N/A")
    if result.ood_avg_cost is not None:
        logger.info("  OOD avg cost                         : ${:.4f}", result.ood_avg_cost)
    else:
        logger.info("  OOD avg cost                         : N/A")

    if result.datasets:
        logger.info("Per-dataset metrics:")
        for dataset_id, ds in result.datasets.items():
            if ds.decisive_total:
                selection_dec = (
                    f"{ds.selection_accuracy_decisive * 100:.2f}% "
                    f"({ds.selected_correct_decisive}/{ds.decisive_total})"
                )
                routing_dec = (
                    f"{ds.routing_accuracy * 100:.2f}% "
                    f"({ds.routing_correct}/{ds.decisive_total})"
                )
            else:
                selection_dec = "N/A (0 decisive)"
                routing_dec = "N/A (0 decisive)"
            logger.info(
                "  {} -> Selection acc: {:.2f}% ({}/{}), Selection acc (dec): {}, Routing acc (dec): {}, Ties: {}, Strong: {}, Weak: {}, Cost: ${:.4f}, Avg cost: ${:.4f}",
                dataset_id,
                ds.selection_accuracy * 100,
                ds.selected_correct_total,
                ds.total,
                selection_dec,
                routing_dec,
                ds.ties,
                ds.strong_selected,
                ds.weak_selected,
                ds.total_cost,
                ds.avg_cost,
            )

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(result.to_dict(), fp, indent=2)
        logger.info("Saved metrics to {}", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
