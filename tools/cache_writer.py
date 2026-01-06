#!/usr/bin/env python3
"""
Cache Writer Script for LLMRouterBench

This script writes existing result data into cache to avoid repeated API calls.
It processes result files from the Avengers_pro directory and writes them
directly into the MySQL cache system.

Usage:
    python -m tools.cache_writer config/cache_write_config.yaml [--dry-run] [--verbose]
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import asdict

# Add the project root to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from data_collector.config_loader import ConfigLoader
from generators.generator import GeneratorOutput
from common.cache.config import CacheConfig
from common.cache.mysql_store import MySQLCacheStore
from common.cache.key_generator import CacheKeyGenerator


class CacheWriter:
    """Write existing result data into cache"""
    
    def __init__(self, config_path: str, dry_run: bool = False, verbose: bool = False, 
                 include_models: Optional[List[str]] = None, exclude_models: Optional[List[str]] = None):
        self.config_path = config_path
        self.dry_run = dry_run
        self.verbose = verbose
        
        # Setup logging first
        log_level = "DEBUG" if verbose else "INFO"
        logger.remove()  # Remove default logger
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # Load raw configuration directly
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            self.raw_config = yaml.safe_load(f)
        
        # Cache writer specific config
        self.writer_config = self.raw_config.get('cache_writer', {})
        self.results_directory = Path(self.writer_config.get('results_directory', './results/Avengers_pro'))
        self.skip_benchmarks = set(self.writer_config.get('skip_benchmarks', []))
        self.include_datasets = set(self.writer_config.get('include_datasets', []))
        self.temperature = self.writer_config.get('temperature', 0.2)
        self.top_p = self.writer_config.get('top_p', 1.0)
        self.batch_size = self.writer_config.get('batch_size', 100)
        self.show_progress = self.writer_config.get('show_progress', True)
        self.max_records = self.writer_config.get('max_records', None)
        
        # Cache writing retry configuration
        self.max_retries = self.writer_config.get('max_retries', 3)
        self.retry_delay = self.writer_config.get('retry_delay', 0.1)
        self.enable_write_verification = self.writer_config.get('enable_write_verification', True)
        
        # Model filtering configuration
        # Command line arguments override config file settings
        config_include_models = self.writer_config.get('include_models', [])
        config_exclude_models = self.writer_config.get('exclude_models', [])
        
        self.include_models = set(include_models) if include_models else set(config_include_models)
        self.exclude_models = set(exclude_models) if exclude_models else set(config_exclude_models)
        
        # Clean model names (strip whitespace)
        self.include_models = {model.strip() for model in self.include_models}
        self.exclude_models = {model.strip() for model in self.exclude_models}
        
        # Initialize cache components
        self.cache_config = CacheConfig.from_dict(self.raw_config.get('cache', {}))
        if not self.cache_config.enabled and not dry_run:
            raise ValueError("Cache must be enabled in configuration")
            
        # Always create key generator (needed for dry run too)
        self.key_generator = CacheKeyGenerator(self.cache_config.key_generator)
        
        if not dry_run:
            self.cache_store = MySQLCacheStore(self.cache_config.mysql)
        else:
            self.cache_store = None
            
        # Statistics
        self.stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'records_processed': 0,
            'records_cached_new': 0,      # Successfully inserted new records
            'records_cached_updated': 0,  # Successfully updated existing records
            'records_duplicate_keys': 0,  # Records with duplicate cache keys
            'records_failed': 0,
            'records_retried': 0,
            'verification_failures': 0,
            'errors': []
        }
        
        # Cache key tracking for duplicate detection
        self.seen_cache_keys = set()  # Track all cache keys we've processed
        self.duplicate_cache_keys = {}  # Map cache_key -> [record_info_list] for duplicates
        
    def extract_model_info(self, full_model_name: str) -> tuple[str, dict]:
        """Extract model name and additional parameters from full model name
        
        Examples:
        'gpt-5-medium' -> ('gpt-5', {'reasoning_effort': 'medium'})
        'gpt-5-high' -> ('gpt-5', {'reasoning_effort': 'high'})
        'gpt-5-chat' -> ('gpt-5-chat', {})
        'qwen/qwen3-235b-a22b-2507' -> ('qwen3-235b-a22b-2507', {})
        """
        # Handle GPT-5 series models
        if full_model_name == "gpt-5-medium":
            return "gpt-5", {"reasoning_effort": "medium"}
        elif full_model_name == "gpt-5-high":
            return "gpt-5", {"reasoning_effort": "high"}
        elif full_model_name == "gpt-5-chat":
            return "gpt-5-chat", {}
        
        # Handle other models with provider prefix
        if '/' in full_model_name:
            return full_model_name.split('/', 1)[1], {}
        
        return full_model_name, {}
        
    def extract_dataset_name(self, file_path: Path) -> str:
        """Extract dataset name from result file path
        
        Examples:
        'aime-experiment_claude-opus-4.1-20250814-085351.json' -> 'aime-experiment'
        'arc-agi-v1_anthropic_claude-opus-4.1.json' -> 'arc-agi-v1'
        'gpqa_claude-opus-4.1.json' -> 'gpqa'
        'tau2_airline_claude-opus-4.1.json' -> 'tau2_airline'
        'tau2_retail_claude-opus-4.1.json' -> 'tau2_retail'
        """
        filename = file_path.name
        
        # Remove .json extension
        base_name = filename.replace('.json', '')
        
        # Handle special patterns first
        
        # Handle tau2 variants (tau2_airline, tau2_retail, tau2_telecom)
        if base_name.startswith('tau2_'):
            parts = base_name.split('_')
            if len(parts) >= 2:
                return f"{parts[0]}_{parts[1]}"  # tau2_airline, tau2_retail, etc.
        
        # Handle experiment suffixes (like aime-experiment, hle-experiment)
        if '-experiment_' in base_name:
            dataset_part = base_name.split('-experiment_')[0]
            return f"{dataset_part}-experiment"
        
        # Handle provider prefixes (like arc-agi-v1_anthropic_...)
        if '_anthropic_' in base_name:
            return base_name.split('_anthropic_')[0]
        elif '_openai_' in base_name:
            return base_name.split('_openai_')[0]
        elif '_google_' in base_name:
            return base_name.split('_google_')[0]
        elif '_qwen_' in base_name:
            return base_name.split('_qwen_')[0]
        
        # Default: take everything before the first model name indicator
        # Common model indicators in filenames
        model_indicators = ['claude', 'gpt', 'gemini', 'qwen']
        
        for indicator in model_indicators:
            if f'_{indicator}' in base_name:
                return base_name.split(f'_{indicator}')[0]
        
        # Fallback: take everything before the first underscore
        if '_' in base_name:
            return base_name.split('_')[0]
        
        # Final fallback: return the base name without extension
        return base_name
        
    def extract_model_name(self, full_model_name: str) -> str:
        """Extract cache-friendly model name from full model name (legacy method)
        
        Example: 'qwen/qwen3-235b-a22b-2507' -> 'qwen3-235b-a22b-2507'
        """
        model_name, _ = self.extract_model_info(full_model_name)
        return model_name
        
    def should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped based on JSON content filters"""
        try:
            # Read JSON file to get dataset_name and model_name
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            dataset_name = data.get('dataset_name')
            model_name = data.get('model_name')

            if not dataset_name or not model_name:
                logger.debug(f"Skipping file {file_path.name} - missing dataset_name or model_name")
                return True

            # Check skip_benchmarks filter
            if self.skip_benchmarks and dataset_name in self.skip_benchmarks:
                logger.debug(f"Skipping file {file_path.name} - dataset '{dataset_name}' in skip list")
                return True

            # Check include_datasets filter
            if self.include_datasets and dataset_name not in self.include_datasets:
                logger.debug(f"Skipping file {file_path.name} - dataset '{dataset_name}' not in include list")
                return True

            # Check include_models filter
            if self.include_models and model_name not in self.include_models:
                logger.debug(f"Skipping file {file_path.name} - model '{model_name}' not in include list")
                return True

            # Check exclude_models filter
            if self.exclude_models and model_name in self.exclude_models:
                logger.debug(f"Skipping file {file_path.name} - model '{model_name}' in exclude list")
                return True

            logger.debug(f"Including file {file_path.name} - dataset='{dataset_name}', model='{model_name}'")
            return False

        except Exception as e:
            logger.warning(f"Error reading file {file_path.name} for filtering: {e}")
            return True
        
    def find_result_files(self) -> List[Path]:
        """Find all result JSON files to process"""
        result_files = []

        if not self.results_directory.exists():
            logger.error(f"Results directory does not exist: {self.results_directory}")
            return result_files

        # New directory structure: results_directory/<dataset>/<split>/<model>/*.json
        # Collect all JSON files first, then filter based on file content
        for dataset_dir in self.results_directory.iterdir():
            if not dataset_dir.is_dir():
                continue

            # Iterate through split directories
            for split_dir in dataset_dir.iterdir():
                if not split_dir.is_dir():
                    continue

                # Iterate through model directories
                for model_dir in split_dir.iterdir():
                    if not model_dir.is_dir():
                        continue

                    # Find JSON files in model directory
                    for json_file in model_dir.glob('*.json'):
                        # Filter based on JSON content
                        if self.should_skip_file(json_file):
                            self.stats['files_skipped'] += 1
                            continue

                        result_files.append(json_file)

        logger.info(f"Found {len(result_files)} result files to process")
        return result_files
        
    def is_benchmark_result_file(self, file_path: Path) -> bool:
        """Check if a file looks like a benchmark result file"""
        filename = file_path.name.lower()
        
        # Common benchmark patterns in filename
        benchmark_indicators = [
            'aime', 'gpqa', 'hle', 'livecodebench', 'simpleqa', 
            'experiment', 'claude', 'gpt', 'qwen', 'gemini'
        ]
        
        return any(indicator in filename for indicator in benchmark_indicators)
    
    def _validate_models(self) -> None:
        """Validate that specified model directories exist"""
        if not self.results_directory.exists():
            return

        # Get all available models and datasets from JSON file content
        # Structure: results_directory/<dataset>/<split>/<model>/*.json
        available_models = set()
        available_datasets = set()

        for dataset_dir in self.results_directory.iterdir():
            if not dataset_dir.is_dir():
                continue
            for split_dir in dataset_dir.iterdir():
                if not split_dir.is_dir():
                    continue
                for model_dir in split_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    # Read JSON files to extract model_name and dataset_name
                    for json_file in model_dir.glob('*.json'):
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            model_name = data.get('model_name')
                            dataset_name = data.get('dataset_name')
                            if model_name:
                                available_models.add(model_name)
                            if dataset_name:
                                available_datasets.add(dataset_name)
                        except Exception:
                            continue

        # Check included models
        if self.include_models:
            missing_models = self.include_models - available_models
            if missing_models:
                logger.warning(f"Specified include models not found: {sorted(missing_models)}")
                logger.info(f"Available models: {sorted(available_models)}")

        # Check excluded models
        if self.exclude_models:
            missing_excluded = self.exclude_models - available_models
            if missing_excluded:
                logger.warning(f"Specified exclude models not found: {sorted(missing_excluded)}")

        # Check included datasets
        if self.include_datasets:
            missing_datasets = self.include_datasets - available_datasets
            if missing_datasets:
                logger.warning(f"Specified include datasets not found: {sorted(missing_datasets)}")
                logger.info(f"Available datasets: {sorted(available_datasets)}")

        # Show which models will actually be processed
        filtered_models = available_models.copy()
        if self.include_models:
            filtered_models &= self.include_models
        if self.exclude_models:
            filtered_models -= self.exclude_models

        if filtered_models:
            logger.info(f"Models to be processed: {sorted(filtered_models)}")
        else:
            logger.warning("No models will be processed after applying filters!")

        # Show which datasets will be processed
        filtered_datasets = available_datasets.copy()
        if self.include_datasets:
            filtered_datasets &= self.include_datasets
        if self.skip_benchmarks:
            filtered_datasets -= self.skip_benchmarks

        if filtered_datasets:
            logger.info(f"Datasets to be processed: {sorted(filtered_datasets)}")
        else:
            logger.warning("No datasets will be processed after applying filters!")
        
    def load_result_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load a result file and return its contents"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Validate that it has the expected structure
            if 'records' not in data:
                logger.warning(f"File {file_path} does not contain 'records' field")
                return None
                
            if not isinstance(data['records'], list):
                logger.warning(f"File {file_path} 'records' field is not a list")
                return None
                
            return data
            
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            self.stats['errors'].append(f"Load error in {file_path}: {e}")
            return None
            
    def process_record(self, record: Dict[str, Any], file_path: Path, model_name: str) -> Optional[Tuple[str, GeneratorOutput, Dict[str, Any]]]:
        """Process a single record and return cache key, GeneratorOutput, and record info"""
        try:
            # Extract record index for better error reporting
            record_index = record.get('index', 'unknown')

            # Extract required fields - try 'prompt' first (new format), then 'query' (old format)
            prompt = record.get('prompt')
            if not prompt:
                prompt = record.get('query')
                if not prompt:
                    logger.warning(f"Record missing 'prompt' and 'query' in {file_path}, record index {record_index}")
                    return None

            raw_output = record.get('raw_output')
            if not raw_output:
                logger.warning(f"Record missing 'raw_output' in {file_path}, record index {record_index}")
                return None

            # Handle raw_output as list or string
            if isinstance(raw_output, list):
                if not raw_output:
                    logger.warning(f"Empty 'raw_output' list in {file_path}, record index {record_index}")
                    return None
                output_text = raw_output[0]  # Take first element
            else:
                output_text = str(raw_output)

            # Extract token usage - new format has tokens directly in record, old format has 'usage' dict
            if 'usage' in record:
                # Old format: tokens in 'usage' dict
                usage = record.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                cost = usage.get('cost', 0.0)
            else:
                # New format: tokens directly in record
                prompt_tokens = record.get('prompt_tokens', 0)
                completion_tokens = record.get('completion_tokens', 0)
                cost = record.get('cost', 0.0)
            
            # Extract model name and additional parameters
            # 特殊处理 GPT-5：从文件路径判断变体
            if model_name == "gpt-5":
                if "gpt-5-medium" in str(file_path):
                    cache_model_name, extra_params = "gpt-5", {"reasoning_effort": "medium"}
                elif "gpt-5-high" in str(file_path):
                    cache_model_name, extra_params = "gpt-5", {"reasoning_effort": "high"}
                elif "gpt-5-chat" in str(file_path):
                    cache_model_name, extra_params = "gpt-5-chat", {}
                else:
                    cache_model_name, extra_params = "gpt-5", {}
            else:
                # 对于其他模型，使用原有逻辑
                cache_model_name, extra_params = self.extract_model_info(model_name)
            
            # Generate cache key with all parameters
            cache_key = self.key_generator.generate_key(
                model_name=cache_model_name,
                question=prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                **extra_params
            )
            
            # Create record info for duplicate tracking
            record_info = {
                'file_path': str(file_path),
                'record_index': record_index,
                'model_name': model_name,
                'cache_model_name': cache_model_name,
                'question_preview': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                'extra_params': extra_params
            }
            
            # Check for duplicate cache keys
            if cache_key in self.seen_cache_keys:
                if cache_key not in self.duplicate_cache_keys:
                    self.duplicate_cache_keys[cache_key] = []
                self.duplicate_cache_keys[cache_key].append(record_info)
                logger.warning(f"Duplicate cache key detected: key={cache_key[:16]}... for record {record_index} in {file_path.name}")
            else:
                self.seen_cache_keys.add(cache_key)
            
            # Create GeneratorOutput
            generator_output = GeneratorOutput(
                output=output_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost
            )
            
            return cache_key, generator_output, record_info
            
        except Exception as e:
            logger.warning(f"Failed to process record in {file_path}, record index {record.get('index', 'unknown')}: {e}")
            self.stats['errors'].append(f"Process error in {file_path}, record {record.get('index', 'unknown')}: {e}")
            return None
            
    def write_to_cache(self, cache_key: str, generator_output: GeneratorOutput, record_info: Dict[str, Any]) -> str:
        """Write a GeneratorOutput to cache with strict verification
        
        Returns:
            'new': Successfully inserted new record
            'updated': Successfully updated existing record  
            'duplicate': Cache key already exists (no operation performed)
            'failed': Failed to write after all retries
        """
        if self.dry_run:
            # Check if we've already seen this cache key
            if cache_key in self.seen_cache_keys:
                logger.debug(f"[DRY RUN] Duplicate cache key: key={cache_key[:16]}..., record {record_info.get('record_index')} in {record_info.get('file_path')}")
                return 'duplicate'
            else:
                logger.debug(f"[DRY RUN] Would cache: key={cache_key[:16]}..., tokens={generator_output.prompt_tokens}+{generator_output.completion_tokens}, cost=${generator_output.cost:.4f}")
                return 'new'  # Assume new for dry run
            
        # Convert GeneratorOutput to dictionary for caching
        generator_output_dict = asdict(generator_output)
        
        # Check if this is a duplicate cache key we've already processed
        if cache_key in self.duplicate_cache_keys:
            logger.debug(f"Processing record with known duplicate cache key: key={cache_key[:16]}..., record {record_info.get('record_index')}")
        
        for attempt in range(self.max_retries):
            try:
                # Check if cache key already exists in database
                existing_data = self.cache_store.get(cache_key)
                key_existed_in_db = existing_data is not None
                
                if key_existed_in_db:
                    # Compare with existing data to see if update is needed
                    if self._compare_generator_output(existing_data, generator_output_dict):
                        # Data is identical
                        if not self.cache_config.force_override_cache:
                            # Skip update if force_override_cache is false
                            logger.debug(f"Cache key already exists with identical data: key={cache_key[:16]}...")
                            return 'duplicate'
                        else:
                            # Force update even if data is identical
                            logger.debug(f"Force overriding cache (force_override_cache=true): key={cache_key[:16]}...")
                            # Continue to the write operation below
                
                # Attempt to write to cache (this will INSERT or UPDATE as needed)
                success = self.cache_store.put(cache_key, generator_output_dict)
                
                if success:
                    # Verify write was successful with strict verification
                    if self.enable_write_verification:
                        cached_data = self.cache_store.get(cache_key)
                        if cached_data is not None:
                            # Verify the data matches what we wrote
                            if self._compare_generator_output(cached_data, generator_output_dict):
                                # Success - Now we need to determine if it was INSERT or UPDATE
                                # by checking if the key existed in DB before our write
                                operation_type = 'updated' if key_existed_in_db else 'new'
                                logger.debug(f"Successfully cached and verified ({operation_type}): key={cache_key[:16]}..., tokens={generator_output.prompt_tokens}+{generator_output.completion_tokens}")
                                
                                if attempt > 0:
                                    self.stats['records_retried'] += 1
                                return operation_type
                            else:
                                # Data doesn't match - verification failed
                                self.stats['verification_failures'] += 1
                                logger.warning(f"Write verification failed - data mismatch for key={cache_key[:16]}..., attempt {attempt + 1}/{self.max_retries}")
                        else:
                            # Write reported success but verification failed
                            self.stats['verification_failures'] += 1
                            logger.warning(f"Write verification failed - no data found for key={cache_key[:16]}..., attempt {attempt + 1}/{self.max_retries}")
                    else:
                        # Verification disabled, trust the put() result and DB existence check
                        operation_type = 'updated' if key_existed_in_db else 'new'
                        logger.debug(f"Successfully cached ({operation_type}): key={cache_key[:16]}..., tokens={generator_output.prompt_tokens}+{generator_output.completion_tokens}")
                        if attempt > 0:
                            self.stats['records_retried'] += 1
                        return operation_type
                else:
                    # Cache put returned False
                    logger.warning(f"Cache put returned False for key={cache_key[:16]}..., attempt {attempt + 1}/{self.max_retries}")
                    
            except Exception as e:
                logger.warning(f"Cache write exception for key={cache_key[:16]}..., attempt {attempt + 1}/{self.max_retries}: {e}")
                self.stats['errors'].append(f"Cache write error attempt {attempt + 1}: {e}")
            
            # Wait before retry (except for last attempt)
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
        
        # All attempts failed
        logger.error(f"Failed to cache after {self.max_retries} attempts: key={cache_key[:16]}...")
        return 'failed'
        
    def _compare_generator_output(self, cached_data: Dict[str, Any], new_data: Dict[str, Any]) -> bool:
        """Compare two GeneratorOutput dictionaries for equality"""
        # Compare key fields that matter for caching
        key_fields = ['output', 'prompt_tokens', 'completion_tokens', 'cost']
        
        for field in key_fields:
            if cached_data.get(field) != new_data.get(field):
                return False
        return True
            
    def process_file(self, file_path: Path) -> None:
        """Process a single result file"""
        logger.info(f"Processing file: {file_path}")

        data = self.load_result_file(file_path)
        if not data:
            return

        # Extract model_name from file-level data (new format)
        model_name = data.get('model_name')
        if not model_name:
            logger.warning(f"File {file_path} missing 'model_name' at file level")
            return

        records = data['records']
        logger.info(f"Found {len(records)} records in {file_path}")

        batch_count = 0
        for record in records:
            # Check max_records limit
            if self.max_records and self.stats['records_processed'] >= self.max_records:
                logger.info(f"Reached maximum records limit: {self.max_records}")
                return

            result = self.process_record(record, file_path, model_name)
            if result:
                cache_key, generator_output, record_info = result

                # Write to cache and get the operation result
                cache_result = self.write_to_cache(cache_key, generator_output, record_info)

                if cache_result == 'new':
                    self.stats['records_cached_new'] += 1
                elif cache_result == 'updated':
                    self.stats['records_cached_updated'] += 1
                elif cache_result == 'duplicate':
                    # Count duplicates properly
                    self.stats['records_duplicate_keys'] += 1
                else:  # 'failed'
                    self.stats['records_failed'] += 1
            else:
                self.stats['records_failed'] += 1

            self.stats['records_processed'] += 1
            batch_count += 1

            # Progress reporting
            if self.show_progress and batch_count % self.batch_size == 0:
                logger.info(f"Processed {batch_count}/{len(records)} records in {file_path.name}")

        logger.info(f"Completed file: {file_path} ({len(records)} records)")
        
    def run(self) -> None:
        """Main execution method"""
        logger.info("Starting cache writer...")
        logger.info(f"Results directory: {self.results_directory}")
        logger.info(f"Skip benchmarks: {self.skip_benchmarks}")
        
        if self.include_datasets:
            logger.info(f"Include datasets: {self.include_datasets}")
        else:
            logger.info("Include datasets: ALL (no filter specified)")
        
        # Model filtering feedback
        if self.include_models:
            logger.info(f"Include models: {sorted(self.include_models)}")
        else:
            logger.info("Include models: ALL (no filter specified)")
            
        if self.exclude_models:
            logger.info(f"Exclude models: {sorted(self.exclude_models)}")
            
        logger.info(f"Temperature: {self.temperature}")
        logger.info(f"Dry run: {self.dry_run}")
        
        if self.max_records:
            logger.info(f"Maximum records: {self.max_records}")
            
        # Validate model directories exist
        self._validate_models()
        
        # Find all files to process
        result_files = self.find_result_files()
        if not result_files:
            logger.error("No result files found to process")
            return
            
        # Process each file
        for i, file_path in enumerate(result_files, 1):
            logger.info(f"Processing file {i}/{len(result_files)}: {file_path.name}")
            
            try:
                self.process_file(file_path)
                self.stats['files_processed'] += 1
                
            except KeyboardInterrupt:
                logger.warning("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error processing {file_path}: {e}")
                self.stats['errors'].append(f"File processing error {file_path}: {e}")
                
        # Print final statistics
        self.print_statistics()
        
    def print_statistics(self) -> None:
        """Print final processing statistics"""
        logger.info("=== CACHE WRITER STATISTICS ===")
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Files skipped: {self.stats['files_skipped']}")
        logger.info(f"Records processed: {self.stats['records_processed']}")
        
        # Detailed cache operation statistics
        logger.info(f"Records cached (new): {self.stats['records_cached_new']}")
        logger.info(f"Records cached (updated): {self.stats['records_cached_updated']}")
        logger.info(f"Records with duplicate keys: {self.stats['records_duplicate_keys']}")
        logger.info(f"Records failed: {self.stats['records_failed']}")
        
        # Total successful operations (for database count comparison)
        total_cached = self.stats['records_cached_new'] + self.stats['records_cached_updated']
        logger.info(f"Total records in cache operations: {total_cached}")
        logger.info(f"Expected new database records: {self.stats['records_cached_new']}")
        
        # Cache key uniqueness analysis
        unique_cache_keys = len(self.seen_cache_keys)
        logger.info(f"Unique cache keys generated: {unique_cache_keys}")
        
        if self.duplicate_cache_keys:
            logger.warning(f"CACHE KEY COLLISIONS DETECTED: {len(self.duplicate_cache_keys)} duplicate keys found")
            logger.warning("This explains the discrepancy between processed records and database records!")
            
            # Show details about duplicate cache keys
            if self.verbose:
                logger.warning("Duplicate cache key details:")
                for cache_key, record_list in list(self.duplicate_cache_keys.items())[:5]:  # Show first 5
                    logger.warning(f"  Cache key {cache_key[:16]}... has {len(record_list)} duplicate records:")
                    for record_info in record_list[:3]:  # Show first 3 records for each key
                        logger.warning(f"    - Record {record_info['record_index']} in {record_info['file_path']} (model: {record_info['model_name']})")
                        logger.warning(f"      Question: {record_info['question_preview']}")
                    if len(record_list) > 3:
                        logger.warning(f"    ... and {len(record_list) - 3} more records")
                        
                if len(self.duplicate_cache_keys) > 5:
                    logger.warning(f"  ... and {len(self.duplicate_cache_keys) - 5} more duplicate keys")
        else:
            logger.info("No cache key collisions detected - all keys are unique")
        
        # Additional retry and verification statistics
        if self.stats['records_retried'] > 0:
            logger.info(f"Records retried: {self.stats['records_retried']}")
        if self.stats['verification_failures'] > 0:
            logger.warning(f"Verification failures: {self.stats['verification_failures']}")
            
        if self.stats['records_processed'] > 0:
            success_rate = (total_cached / self.stats['records_processed']) * 100
            logger.info(f"Cache success rate: {success_rate:.1f}%")
            
        if self.stats['errors']:
            logger.warning(f"Total errors: {len(self.stats['errors'])}")
            if self.verbose:
                logger.warning("Error details:")
                for error in self.stats['errors'][:10]:  # Show first 10 errors
                    logger.warning(f"  - {error}")
                if len(self.stats['errors']) > 10:
                    logger.warning(f"  ... and {len(self.stats['errors']) - 10} more")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Write existing result data into cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tools.cache_writer config/cache_write_config.yaml
  python -m tools.cache_writer config/cache_write_config.yaml --dry-run
  python -m tools.cache_writer config/cache_write_config.yaml --verbose
  python -m tools.cache_writer config/cache_write_config.yaml --models claude_4_sonnet,gpt-5-chat
  python -m tools.cache_writer config/cache_write_config.yaml --exclude-models claude_4.1_opus
        """
    )
    
    parser.add_argument(
        "config_path",
        help="Path to the cache writer configuration file"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actually writing to cache"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of model names to process (e.g., 'claude_4_sonnet,gpt-5-chat')"
    )
    
    parser.add_argument(
        "--exclude-models",
        type=str,
        help="Comma-separated list of model names to exclude from processing"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config_path):
        print(f"Configuration file not found: {args.config_path}")
        sys.exit(1)
        
    try:
        writer = CacheWriter(
            config_path=args.config_path,
            dry_run=args.dry_run,
            verbose=args.verbose,
            include_models=args.models.split(',') if args.models else None,
            exclude_models=args.exclude_models.split(',') if args.exclude_models else None
        )
        writer.run()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()