"""
Setup utilities for external benchmarks
处理导入路径和结果存储
"""

import sys
import yaml
import time
from pathlib import Path
from typing import Dict, Any, List


def load_cache_config(config_path: str) -> Dict[str, Any]:
    """Load cache config from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('cache', {})
    except Exception:
        return {"enabled": False}


def add_project_path():
    """Add project root to sys.path"""
    current = Path.cwd()
    # Go up until we find the project root
    for _ in range(5):
        if (current / 'generators').exists():
            sys.path.insert(0, str(current))
            return str(current)
        current = current.parent
    # Fallback: just add current directory
    sys.path.insert(0, str(Path.cwd()))
    return str(Path.cwd())


def save_result(benchmark_result, dataset_id: str, split: str, model_name: str, base_dir: str = "results") -> str:
    """Save benchmark result using framework storage format"""
    # Import here to avoid circular imports
    sys.path.append(str(Path(__file__).parent.parent))
    from data_collector.storage import ResultsStorage

    storage = ResultsStorage(base_dir=base_dir)
    # External bench may not have standard datasets, pass empty fingerprint
    storage.save_result(benchmark_result, dataset_id, split, model_name, data_fingerprint="")
    return str(storage.get_result_path(dataset_id, split, model_name))


def finish_benchmark(record_results: List, model_name: str,
                    dataset_name: str = "external_bench",
                    split: str = "test",
                    base_dir: str = "results/bench/external_bench") -> float:
    """
    Complete benchmark by calculating metrics, printing results, and saving to storage

    Args:
        record_results: List of RecordResult objects
        model_name: Model name string
        dataset_name: Name of the dataset
        split: Data split name
        base_dir: Base directory for results storage

    Returns:
        Accuracy score
    """
    # Import here to avoid circular imports
    sys.path.append(str(Path(__file__).parent.parent))
    from data_collector.storage import BenchmarkResult

    # Calculate final metrics
    start_time = getattr(finish_benchmark, '_start_time', time.time())
    end_time = time.time()
    total_time = end_time - start_time

    correct_count = sum(1 for r in record_results if r.score > 0.0)
    accuracy = correct_count / len(record_results) if record_results else 0.0
    total_prompt_tokens = sum(r.prompt_tokens for r in record_results)
    total_completion_tokens = sum(r.completion_tokens for r in record_results)
    total_cost = sum(r.cost for r in record_results)

    # Print results
    print("\n" + "=" * 40)
    print("Benchmark Results:")
    print(f"Accuracy: {accuracy:.1%} ({correct_count}/{len(record_results)})")
    print(f"Time: {total_time:.2f}s")
    print(f"Total tokens: {total_prompt_tokens + total_completion_tokens}")
    print(f"Total cost: ${total_cost:.6f}")

    # Create BenchmarkResult and save
    benchmark_result = BenchmarkResult(
        performance=accuracy,
        time_taken=total_time,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        cost=total_cost,
        counts=len(record_results),
        model_name=model_name,
        dataset_name=dataset_name,
        split=split,
        records=record_results
    )

    # Save using framework storage format
    result_path = save_result(benchmark_result, dataset_name, split, model_name, base_dir)
    print(f"Results saved to: {result_path}")

    return accuracy


def start_timer():
    """Mark the start time for benchmark timing"""
    finish_benchmark._start_time = time.time()