import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class RecordResult:
    index: int
    origin_query: str
    prompt: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    score: Optional[float]
    prediction: str
    ground_truth: str
    raw_output: Any
    extra_fields: Dict[str, Any] = field(default_factory=dict)  # Extracted fields from response


@dataclass
class BenchmarkResult:
    performance: float
    time_taken: float
    prompt_tokens: int
    completion_tokens: int
    cost: float
    counts: int
    model_name: str
    dataset_name: str
    split: str
    records: List[RecordResult]
    demo: bool = False
    extra_metrics: Dict[str, Any] = field(default_factory=dict)  # Aggregated extra metrics


class ResultsStorage:
    """Manage benchmark results storage and indexing"""
    
    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self.bench_dir = self.base_dir / "bench"

        # Ensure directories exist
        self.bench_dir.mkdir(parents=True, exist_ok=True)
    
    def get_result_path(self, dataset_id: str, split: str, model_name: str) -> Path:
        """Get the file path for a specific result with timestamp"""
        import time
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"{dataset_id}-{split}-{model_name}-{timestamp}.json"
        return self.bench_dir / dataset_id / split / model_name / filename
    
    def exists(self, dataset_id: str, split: str, model_name: str) -> bool:
        """Check if a result already exists"""
        result_dir = self.bench_dir / dataset_id / split / model_name
        if not result_dir.exists():
            return False
        
        # Check if any result file exists in the directory
        file_pattern = f"{dataset_id}-{split}-{model_name}-*.json"
        return any(result_dir.glob(file_pattern))
    
    def save_result(self, result: BenchmarkResult, dataset_id: str, split: str, model_name: str, data_fingerprint: str = ""):
        """Save a benchmark result to storage"""
        result_path = self.get_result_path(dataset_id, split, model_name)
        result_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for JSON serialization
        result_dict = {
            "performance": result.performance,
            "time_taken": result.time_taken,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cost": result.cost,
            "counts": result.counts,
            "model_name": result.model_name,
            "dataset_name": result.dataset_name,
            "split": result.split,
            "demo": result.demo,
            "extra_metrics": result.extra_metrics,
            "data_fingerprint": data_fingerprint,
            "records": [
                {
                    "index": record.index,
                    "origin_query": record.origin_query,
                    "prompt": record.prompt,
                    "prompt_tokens": record.prompt_tokens,
                    "completion_tokens": record.completion_tokens,
                    "cost": record.cost,
                    "score": record.score,
                    "prediction": record.prediction,
                    "ground_truth": record.ground_truth,
                    "raw_output": record.raw_output,
                    "extra_fields": record.extra_fields
                }
                for record in result.records
            ]
        }
        
        # Atomic write using temporary file
        temp_path = result_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        temp_path.rename(result_path)
        logger.info(f"Saved result: {result_path}")
    
    def load_result(self, dataset_id: str, split: str, model_name: str) -> Optional[BenchmarkResult]:
        """Load a benchmark result from storage"""
        result_dir = self.bench_dir / dataset_id / split / model_name
        if not result_dir.exists():
            return None
        
        # Find the most recent result file
        file_pattern = f"{dataset_id}-{split}-{model_name}-*.json"
        result_files = list(result_dir.glob(file_pattern))
        
        # Also check for legacy result.json files
        legacy_file = result_dir / "result.json"
        if legacy_file.exists():
            result_files.append(legacy_file)
        
        if not result_files:
            return None
        
        # Sort by modification time and get the most recent
        result_path = max(result_files, key=lambda f: f.stat().st_mtime)
        
        with open(result_path, 'r', encoding='utf-8') as f:
            result_dict = json.load(f)
        
        # Convert records back to RecordResult objects
        records = [
            RecordResult(**record_data) 
            for record_data in result_dict["records"]
        ]
        
        # Remove records from dict to avoid duplicate argument
        result_dict_clean = {k: v for k, v in result_dict.items() if k != "records"}
        
        return BenchmarkResult(**result_dict_clean, records=records)
    
    def calculate_data_fingerprint(self, data) -> str:
        """Calculate SHA256 fingerprint of loaded data content"""
        try:
            # Convert data to a consistent string representation for hashing
            if hasattr(data, 'to_dict'):  # Dataset object
                content = json.dumps(data.to_dict(), sort_keys=True, ensure_ascii=False)
            elif isinstance(data, list):
                content = json.dumps(data, sort_keys=True, ensure_ascii=False)
            elif isinstance(data, dict):
                content = json.dumps(data, sort_keys=True, ensure_ascii=False)
            else:
                content = str(data)
            
            return hashlib.sha256(content.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate data fingerprint: {e}")
            return ""

    def _get_latest_non_demo_file(self, files: List[Path]) -> Optional[Path]:
        """
        Select the latest non-demo file from a list of files.
        Sort by timestamp in filename (YYYYMMDD_HHMMSS), not file modification time.

        Args:
            files: List of file paths to choose from

        Returns:
            Path to the latest non-demo file, or None if no valid files found
        """
        import re

        non_demo_files = []
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                if not data.get("demo", False):
                    non_demo_files.append(f)
            except Exception as e:
                logger.warning(f"Error reading file {f}: {e}")
                continue

        if not non_demo_files:
            return None

        # Sort by timestamp in filename (YYYYMMDD_HHMMSS)
        def extract_timestamp(file_path: Path) -> int:
            match = re.search(r'(\d{8})_(\d{6})\.json$', file_path.name)
            if match:
                # Convert to integer: YYYYMMDDHHMMSS
                return int(match.group(1) + match.group(2))
            # Fallback to file modification time if no timestamp in filename
            return int(file_path.stat().st_mtime)

        return max(non_demo_files, key=extract_timestamp)

    def list_results(self) -> List[Dict[str, Any]]:
        """List all available results"""
        results = []
        
        for dataset_dir in self.bench_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            for split_dir in dataset_dir.iterdir():
                if not split_dir.is_dir():
                    continue
                
                for model_dir in split_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    
                    # Look for both new timestamped files and legacy result.json
                    dataset_id = dataset_dir.name
                    split_name = split_dir.name
                    model_name = model_dir.name
                    
                    # Find timestamped files
                    file_pattern = f"{dataset_id}-{split_name}-{model_name}-*.json"
                    result_files = list(model_dir.glob(file_pattern))

                    # Also check for legacy result.json
                    legacy_file = model_dir / "result.json"
                    if legacy_file.exists():
                        result_files.append(legacy_file)

                    # Get only the latest non-demo file
                    latest_file = self._get_latest_non_demo_file(result_files)

                    if latest_file:
                        # Load and add the latest result
                        try:
                            with open(latest_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)

                            # Extract timestamp from filename, fallback to file stat
                            filename = latest_file.name
                            import re
                            match = re.search(r'(\d{8})_(\d{6})\.json$', filename)
                            if match:
                                date_str = match.group(1)  # YYYYMMDD
                                time_str = match.group(2)  # HHMMSS
                                updated_at = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
                            else:
                                # Fallback to file stat
                                updated_at = time.ctime(latest_file.stat().st_mtime)

                            results.append({
                                "dataset_id": dataset_id,
                                "split": split_name,
                                "model_name": model_name,
                                "result_path": str(latest_file),
                                "updated_at": updated_at,
                                "counts": data.get("counts", 0)
                            })
                        except Exception as e:
                            logger.warning(f"Failed to load result file {latest_file}: {e}")
                            continue
        
        return sorted(results, key=lambda x: x["updated_at"], reverse=True)
    
    def needs_run(self, dataset_id: str, split: str, model_name: str,
                 current_fingerprint: str = "", overwrite: bool = False) -> bool:
        """Check if a run is needed based on file system and fingerprint"""
        if overwrite:
            return True

        # Check if result files exist
        result_dir = self.bench_dir / dataset_id / split / model_name
        if not result_dir.exists():
            return True

        # Find timestamped files
        file_pattern = f"{dataset_id}-{split}-{model_name}-*.json"
        result_files = list(result_dir.glob(file_pattern))

        # Also check for legacy result.json
        legacy_file = result_dir / "result.json"
        if legacy_file.exists():
            result_files.append(legacy_file)

        # Get the latest non-demo file
        latest_file = self._get_latest_non_demo_file(result_files)
        if not latest_file:
            return True

        # Check fingerprint if provided
        if current_fingerprint:
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)

                stored_fingerprint = result_data.get('data_fingerprint', '')

                if not stored_fingerprint:
                    logger.warning(f"Legacy result without fingerprint: {latest_file.name}")
                    return False  # Keep existing result, don't re-run

                if stored_fingerprint != current_fingerprint:
                    logger.warning(
                        f"Data changed for {dataset_id}/{split} "
                        f"({stored_fingerprint[:8]}... → {current_fingerprint[:8]}...), will re-run"
                    )
                    return True
            except Exception as e:
                logger.error(f"Failed to check fingerprint for {latest_file}: {e}")
                return False  # Conservative: don't re-run on error

        return False