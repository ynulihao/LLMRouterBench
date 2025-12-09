import os
import yaml
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv


@dataclass
class ModelConfig:
    name: str
    api_model_name: str
    base_url: str
    api_key: str
    temperature: float = 0.0
    top_p: float = 1.0
    timeout: int = 500
    generator_type: str = "direct"
    reasoning_effort: Optional[str] = None
    extra_body: Dict[str, Any] = None
    pricing: Dict[str, Any] = None
    extract_fields: Dict[str, str] = None  # Field extraction config: {field_name: response_path}

    def __post_init__(self):
        if self.extra_body is None:
            self.extra_body = {}
        if self.pricing is None:
            self.pricing = {}
        if self.extract_fields is None:
            self.extract_fields = {}


@dataclass
class DatasetConfig:
    dataset_id: str
    splits: Optional[List[str]] = None


@dataclass
class RunConfig:
    kind: str = "bench"
    output_dir: str = "./results"
    overwrite: bool = False
    concurrency: int = 8
    log_level: str = "INFO"
    demo_mode: bool = False
    demo_limit: int = 10


@dataclass
class BenchmarkConfig:
    models: List[ModelConfig]
    datasets: List[DatasetConfig]
    run: RunConfig
    cache_config: Optional[Dict[str, Any]] = None
    grader_cache_config: Optional[Dict[str, Any]] = None


class ConfigLoader:
    """Load and validate benchmark configuration from YAML files"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load environment variables from .env file
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
            logger.debug("Loaded environment variables from .env file")
        else:
            logger.debug("No .env file found, using system environment variables")
    
    def load(self) -> BenchmarkConfig:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        
        # Validate required top-level keys
        required_keys = ['models', 'datasets']
        for key in required_keys:
            if key not in raw_config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Parse models
        models = []
        for model_data in raw_config['models']:
            model_config = model_data
            
            # Validate required model fields
            required_model_keys = ['name', 'api_model_name', 'base_url', 'api_key']
            for key in required_model_keys:
                if key not in model_config:
                    raise ValueError(f"Missing required model field '{key}' in model: {model_data.get('name', 'unnamed')}")

            # Validate model name doesn't contain forward slash
            if '/' in model_config['name']:
                raise ValueError(
                    f"Invalid model name '{model_config['name']}': "
                    f"Model name cannot contain forward slash ('/') as it's used for directory paths."
                )

            # Resolve environment variables
            model_config['api_key'] = self._resolve_env_var(model_config['api_key'])
            
            models.append(ModelConfig(**model_config))
        
        # Parse datasets
        datasets = []
        for dataset_data in raw_config['datasets']:
            if 'dataset_id' not in dataset_data:
                raise ValueError(f"Missing required field 'dataset_id' in dataset config")
            datasets.append(DatasetConfig(**dataset_data))
        
        # Parse run configuration
        run_config = RunConfig(**raw_config.get('run', {}))

        # Parse cache configuration
        cache_config = raw_config.get('cache')

        # Parse grader cache configuration
        grader_cache_config = raw_config.get('grader_cache')

        return BenchmarkConfig(
            models=models,
            datasets=datasets,
            run=run_config,
            cache_config=cache_config,
            grader_cache_config=grader_cache_config
        )
    
    def _resolve_env_var(self, value: str) -> str:
        """Resolve environment variable references in config values"""
        if isinstance(value, str) and value.upper() in os.environ:
            resolved = os.environ[value.upper()]
            logger.debug(f"Resolved environment variable {value} -> {resolved[:8]}...")
            return resolved
        return value
    
    @staticmethod
    def get_evaluator_splits(dataset_id: str) -> List[str]:
        """Get valid splits from evaluator"""
        from evaluation.factory import EvaluatorFactory
        factory = EvaluatorFactory()
        try:
            evaluator = factory.get_evaluator(dataset_id)
            return evaluator.get_valid_splits()
        except Exception as e:
            logger.warning(f"Failed to get splits for {dataset_id}: {e}")
            return []
    
    @staticmethod
    def filter_splits(all_splits: List[str], 
                     target_splits: Optional[List[str]] = None) -> List[str]:
        """Filter splits based on target_splits specification"""
        
        # If target_splits specified, use only those
        if target_splits:
            result = [s for s in all_splits if s in target_splits]
            return result
        else:
            # If no target_splits, return all splits
            return all_splits.copy()