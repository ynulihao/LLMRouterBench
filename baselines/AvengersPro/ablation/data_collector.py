"""
Ablation Data Collector

Manages data collection, caching, and storage for ablation experiments.
Provides efficient experiment execution with result caching and error handling.
"""

import json
import os
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import threading
import fcntl
from tqdm import tqdm

# Import parent modules
from ..balance_cluster_router import BalanceClusterRouter
from ..config import SimpleClusterConfig


def _run_single_experiment_worker(config_dict: Dict[str, Any], experiment_type: str, 
                                cache_dir: str, max_retries: int) -> Dict[str, Any]:
    """
    Worker function for running single experiment in parallel process.
    
    This function must be defined at module level to be pickled for multiprocessing.
    """
    import logging
    
    # Set up minimal logging for worker process
    logger = logging.getLogger('AblationWorker')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('Worker - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)  # Reduce verbosity in workers
    
    # Create a temporary data collector for this worker
    collector = AblationDataCollector(cache_dir=cache_dir, max_retries=max_retries)
    
    try:
        result = collector.run_single_experiment(config_dict, experiment_type)
        return {
            'success': True,
            'result': result,
            'config': config_dict,
            'error': None
        }
    except Exception as e:
        logger.error(f"Worker experiment failed: {e}")
        return {
            'success': False,
            'result': None,
            'config': config_dict,
            'error': str(e)
        }


class AblationDataCollector:
    """
    Collects and manages experimental data for ablation studies.
    
    Features:
    - Automatic result caching based on configuration hash
    - Progress tracking and logging
    - Error handling and retry mechanisms
    - JSON-based storage compatible with existing pipeline
    """
    
    def __init__(self, cache_dir: str = "ablation/results", max_retries: int = 3):
        """
        Initialize the data collector.
        
        Args:
            cache_dir: Directory to store cached results
            max_retries: Maximum retries for failed experiments
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup logging if not already configured
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _generate_config_hash(self, config: Dict[str, Any]) -> str:
        """
        Generate a hash for configuration to enable caching.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            MD5 hash of the configuration
        """
        # Create a normalized config string for hashing
        config_str = json.dumps(config, sort_keys=True, separators=(',', ':'))
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    def _get_cache_path(self, experiment_type: str, config_hash: str) -> Path:
        """
        Get the cache file path for a configuration.
        
        Args:
            experiment_type: Type of experiment ('cluster' or 'weight')  
            config_hash: Configuration hash
            
        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{experiment_type}_{config_hash}.json"
    
    def _is_cached(self, experiment_type: str, config: Dict[str, Any]) -> bool:
        """
        Check if results are already cached for this configuration.
        
        Args:
            experiment_type: Type of experiment
            config: Configuration dictionary
            
        Returns:
            True if results are cached
        """
        config_hash = self._generate_config_hash(config)
        cache_path = self._get_cache_path(experiment_type, config_hash)
        return cache_path.exists()
    
    def _load_cached_result(self, experiment_type: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Load cached results for a configuration.
        
        Args:
            experiment_type: Type of experiment
            config: Configuration dictionary
            
        Returns:
            Cached results or None if not found
        """
        config_hash = self._generate_config_hash(config)
        cache_path = self._get_cache_path(experiment_type, config_hash)
        
        try:
            if cache_path.exists():
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load cached result from {cache_path}: {e}")
            
        return None
    
    def _save_result(self, experiment_type: str, config: Dict[str, Any], 
                    results: Dict[str, Any]) -> None:
        """
        Save experimental results to cache.
        
        Args:
            experiment_type: Type of experiment
            config: Configuration dictionary
            results: Experimental results
        """
        config_hash = self._generate_config_hash(config)
        cache_path = self._get_cache_path(experiment_type, config_hash)
        
        # Prepare data for storage
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'experiment_type': experiment_type,
            'config': config,
            'results': results,
            'config_hash': config_hash
        }
        
        try:
            # Use file locking for concurrent write safety
            lock_path = cache_path.with_suffix('.lock')
            with open(lock_path, 'w') as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"Results cached to {cache_path}")
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            # Clean up lock file
            lock_path.unlink(missing_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to save results to {cache_path}: {e}")
    
    def run_single_experiment(self, config_dict: Dict[str, Any], 
                            experiment_type: str = "ablation") -> Dict[str, Any]:
        """
        Run a single experiment with the given configuration.
        
        Args:
            config_dict: Configuration dictionary for the experiment
            experiment_type: Type of experiment for caching
            
        Returns:
            Experimental results dictionary
        """
        # Check cache first
        if self._is_cached(experiment_type, config_dict):
            self.logger.info(f"Loading cached result for {experiment_type}")
            cached = self._load_cached_result(experiment_type, config_dict)
            if cached:
                return cached['results']
        
        # Run experiment with retries
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Running experiment (attempt {attempt + 1}/{self.max_retries})")
                self.logger.info(f"Config: n_clusters={config_dict.get('n_clusters', 'N/A')}, "
                               f"performance_weight={config_dict.get('performance_weight', 'N/A')}, "
                               f"cost_sensitivity={config_dict.get('cost_sensitivity', 'N/A')}")
                
                # Create config object
                config = SimpleClusterConfig(**config_dict)
                
                # Initialize and run router
                router = BalanceClusterRouter(config)
                results = router.run_routing()
                
                # Save to cache
                self._save_result(experiment_type, config_dict, results)
                
                self.logger.info(f"Experiment completed successfully. Accuracy: {results.get('accuracy', 0):.3f}")
                return results
                
            except Exception as e:
                self.logger.error(f"Experiment failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    raise e
                
        raise RuntimeError(f"Experiment failed after {self.max_retries} attempts")
    
    def run_parameter_sweep(self, base_config: Dict[str, Any], 
                          parameter_name: str, parameter_values: List[Any],
                          experiment_type: str = "parameter_sweep") -> List[Dict[str, Any]]:
        """
        Run a parameter sweep experiment.
        
        Args:
            base_config: Base configuration dictionary
            parameter_name: Name of parameter to sweep
            parameter_values: List of parameter values to test
            experiment_type: Type of experiment for caching
            
        Returns:
            List of results for each parameter value
        """
        results = []
        total_experiments = len(parameter_values)
        
        self.logger.info(f"Starting parameter sweep: {parameter_name}")
        self.logger.info(f"Testing {total_experiments} values: {parameter_values}")
        
        for i, param_value in enumerate(parameter_values):
            self.logger.info(f"Progress: {i+1}/{total_experiments} ({(i+1)/total_experiments*100:.1f}%)")
            
            # Create config for this parameter value
            config_dict = base_config.copy()
            config_dict[parameter_name] = param_value
            
            try:
                result = self.run_single_experiment(config_dict, 
                                                  f"{experiment_type}_{parameter_name}_{param_value}")
                
                # Add parameter info to result
                result_with_param = {
                    'parameter_name': parameter_name,
                    'parameter_value': param_value,
                    'config': config_dict,
                    **result
                }
                
                results.append(result_with_param)
                
            except Exception as e:
                self.logger.error(f"Failed experiment for {parameter_name}={param_value}: {e}")
                # Add failed result placeholder
                results.append({
                    'parameter_name': parameter_name,
                    'parameter_value': param_value,
                    'config': config_dict,
                    'error': str(e),
                    'accuracy': 0.0,
                    'failed': True
                })
        
        self.logger.info(f"Parameter sweep completed. {len([r for r in results if not r.get('failed', False)])} successful runs")
        return results
    
    def run_multi_parameter_sweep(self, base_config: Dict[str, Any],
                                parameter_configs: List[Tuple[str, List[Any]]],
                                experiment_type: str = "multi_parameter_sweep") -> List[Dict[str, Any]]:
        """
        Run a multi-parameter sweep experiment.
        
        Args:
            base_config: Base configuration dictionary
            parameter_configs: List of (parameter_name, parameter_values) tuples
            experiment_type: Type of experiment for caching
            
        Returns:
            List of results for each parameter combination
        """
        results = []
        
        # Generate all parameter combinations
        import itertools
        parameter_names = [pc[0] for pc in parameter_configs]
        parameter_value_lists = [pc[1] for pc in parameter_configs]
        parameter_combinations = list(itertools.product(*parameter_value_lists))
        
        total_experiments = len(parameter_combinations)
        self.logger.info(f"Starting multi-parameter sweep: {parameter_names}")
        self.logger.info(f"Testing {total_experiments} combinations")
        
        for i, param_values in enumerate(parameter_combinations):
            self.logger.info(f"Progress: {i+1}/{total_experiments} ({(i+1)/total_experiments*100:.1f}%)")
            
            # Create config for this parameter combination
            config_dict = base_config.copy()
            param_dict = {}
            for param_name, param_value in zip(parameter_names, param_values):
                config_dict[param_name] = param_value
                param_dict[param_name] = param_value
                
            try:
                result = self.run_single_experiment(config_dict, 
                                                  f"{experiment_type}_{'_'.join(map(str, param_values))}")
                
                # Add parameter info to result
                result_with_params = {
                    'parameter_combination': param_dict,
                    'config': config_dict,
                    **result
                }
                
                results.append(result_with_params)
                
            except Exception as e:
                self.logger.error(f"Failed experiment for {param_dict}: {e}")
                # Add failed result placeholder
                results.append({
                    'parameter_combination': param_dict,
                    'config': config_dict,
                    'error': str(e),
                    'accuracy': 0.0,
                    'failed': True
                })
        
        successful_runs = len([r for r in results if not r.get('failed', False)])
        self.logger.info(f"Multi-parameter sweep completed. {successful_runs} successful runs")
        return results
    
    def save_sweep_results(self, results: List[Dict[str, Any]], 
                          filename: str) -> str:
        """
        Save sweep results to a JSON file.
        
        Args:
            results: List of experimental results
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.cache_dir / filename
        
        sweep_data = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(results),
            'successful_experiments': len([r for r in results if not r.get('failed', False)]),
            'results': results
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sweep_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Sweep results saved to {output_path}")
            return str(output_path)
        except Exception as e:
            self.logger.error(f"Failed to save sweep results: {e}")
            raise e
    
    def run_parameter_sweep_parallel(self, base_config: Dict[str, Any], 
                                   parameter_name: str, parameter_values: List[Any],
                                   experiment_type: str = "parameter_sweep",
                                   max_workers: Optional[int] = None,
                                   quiet: bool = False) -> List[Dict[str, Any]]:
        """
        Run a parameter sweep experiment in parallel.
        
        Args:
            base_config: Base configuration dictionary
            parameter_name: Name of parameter to sweep
            parameter_values: List of parameter values to test
            experiment_type: Type of experiment for caching
            max_workers: Maximum number of parallel workers (default: CPU count - 1)
            quiet: If True, minimize output and show progress bar
            
        Returns:
            List of results for each parameter value
        """
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)
        
        total_experiments = len(parameter_values)
        
        if not quiet:
            self.logger.info(f"Starting parallel parameter sweep: {parameter_name}")
            self.logger.info(f"Testing {total_experiments} values with {max_workers} workers")
            self.logger.info(f"Parameter values: {parameter_values}")
        
        # Prepare experiment configurations
        experiment_configs = []
        for param_value in parameter_values:
            config_dict = base_config.copy()
            config_dict[parameter_name] = param_value
            experiment_configs.append({
                'config_dict': config_dict,
                'experiment_type': f"{experiment_type}_{parameter_name}_{param_value}",
                'param_name': parameter_name,
                'param_value': param_value
            })
        
        results = []
        
        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all experiments
            future_to_config = {
                executor.submit(
                    _run_single_experiment_worker,
                    exp_config['config_dict'],
                    exp_config['experiment_type'],
                    str(self.cache_dir),
                    self.max_retries
                ): exp_config for exp_config in experiment_configs
            }
            
            # Create progress bar if in quiet mode
            if quiet:
                progress_bar = tqdm(
                    total=total_experiments,
                    desc=f"Parameter Sweep ({parameter_name})",
                    unit="exp",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                )
            
            completed = 0
            # Process completed experiments as they finish
            for future in as_completed(future_to_config):
                exp_config = future_to_config[future]
                completed += 1
                
                try:
                    worker_result = future.result()
                    
                    if worker_result['success']:
                        result = {
                            'parameter_name': exp_config['param_name'],
                            'parameter_value': exp_config['param_value'],
                            'config': exp_config['config_dict'],
                            **worker_result['result']
                        }
                        
                        if not quiet:
                            accuracy = result.get('accuracy', 0)
                            self.logger.info(f"Completed {exp_config['param_name']}={exp_config['param_value']} "
                                           f"(accuracy: {accuracy:.3f}) [{completed}/{total_experiments}]")
                    else:
                        if not quiet:
                            self.logger.error(f"Failed {exp_config['param_name']}={exp_config['param_value']}: "
                                            f"{worker_result['error']}")
                        result = {
                            'parameter_name': exp_config['param_name'],
                            'parameter_value': exp_config['param_value'],
                            'config': exp_config['config_dict'],
                            'error': worker_result['error'],
                            'accuracy': 0.0,
                            'failed': True
                        }
                    
                    results.append(result)
                    
                    if quiet:
                        progress_bar.update(1)
                        # Update progress bar description with best result so far
                        best_acc = max([r.get('accuracy', 0) for r in results if not r.get('failed', False)], default=0)
                        progress_bar.set_postfix({'Best Acc': f'{best_acc:.3f}'})
                    
                except Exception as e:
                    if not quiet:
                        self.logger.error(f"Exception in experiment {exp_config['param_name']}={exp_config['param_value']}: {e}")
                    
                    result = {
                        'parameter_name': exp_config['param_name'],
                        'parameter_value': exp_config['param_value'],
                        'config': exp_config['config_dict'],
                        'error': str(e),
                        'accuracy': 0.0,
                        'failed': True
                    }
                    results.append(result)
                    
                    if quiet:
                        progress_bar.update(1)
            
            if quiet:
                progress_bar.close()
        
        # Sort results by parameter value to maintain order
        param_value_to_result = {r['parameter_value']: r for r in results}
        sorted_results = [param_value_to_result[pv] for pv in parameter_values if pv in param_value_to_result]
        
        successful_runs = len([r for r in sorted_results if not r.get('failed', False)])
        if not quiet:
            self.logger.info(f"Parallel parameter sweep completed. {successful_runs}/{total_experiments} successful runs")
        
        return sorted_results
    
    def run_multi_parameter_sweep_parallel(self, base_config: Dict[str, Any],
                                         parameter_configs: List[Tuple[str, List[Any]]],
                                         experiment_type: str = "multi_parameter_sweep",
                                         max_workers: Optional[int] = None,
                                         quiet: bool = False) -> List[Dict[str, Any]]:
        """
        Run a multi-parameter sweep experiment in parallel.
        
        Args:
            base_config: Base configuration dictionary
            parameter_configs: List of (parameter_name, parameter_values) tuples
            experiment_type: Type of experiment for caching
            max_workers: Maximum number of parallel workers (default: CPU count - 1)
            quiet: If True, minimize output and show progress bar
            
        Returns:
            List of results for each parameter combination
        """
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)
        
        # Generate all parameter combinations
        import itertools
        parameter_names = [pc[0] for pc in parameter_configs]
        parameter_value_lists = [pc[1] for pc in parameter_configs]
        parameter_combinations = list(itertools.product(*parameter_value_lists))
        
        total_experiments = len(parameter_combinations)
        
        if not quiet:
            self.logger.info(f"Starting parallel multi-parameter sweep: {parameter_names}")
            self.logger.info(f"Testing {total_experiments} combinations with {max_workers} workers")
        
        # Prepare experiment configurations
        experiment_configs = []
        for i, param_values in enumerate(parameter_combinations):
            config_dict = base_config.copy()
            param_dict = {}
            for param_name, param_value in zip(parameter_names, param_values):
                config_dict[param_name] = param_value
                param_dict[param_name] = param_value
            
            experiment_configs.append({
                'config_dict': config_dict,
                'experiment_type': f"{experiment_type}_{'_'.join(map(str, param_values))}",
                'param_dict': param_dict,
                'param_values': param_values,
                'index': i
            })
        
        results = []
        
        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all experiments
            future_to_config = {
                executor.submit(
                    _run_single_experiment_worker,
                    exp_config['config_dict'],
                    exp_config['experiment_type'],
                    str(self.cache_dir),
                    self.max_retries
                ): exp_config for exp_config in experiment_configs
            }
            
            # Create progress bar if in quiet mode
            if quiet:
                progress_bar = tqdm(
                    total=total_experiments,
                    desc=f"Multi-Parameter Sweep",
                    unit="exp",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                )
            
            completed = 0
            # Process completed experiments as they finish
            for future in as_completed(future_to_config):
                exp_config = future_to_config[future]
                completed += 1
                
                try:
                    worker_result = future.result()
                    
                    if worker_result['success']:
                        result = {
                            'parameter_combination': exp_config['param_dict'],
                            'config': exp_config['config_dict'],
                            'index': exp_config['index'],
                            **worker_result['result']
                        }
                        
                        if not quiet:
                            accuracy = result.get('accuracy', 0)
                            param_str = ', '.join([f"{k}={v}" for k, v in exp_config['param_dict'].items()])
                            self.logger.info(f"Completed {param_str} (accuracy: {accuracy:.3f}) [{completed}/{total_experiments}]")
                    else:
                        if not quiet:
                            param_str = ', '.join([f"{k}={v}" for k, v in exp_config['param_dict'].items()])
                            self.logger.error(f"Failed {param_str}: {worker_result['error']}")
                        result = {
                            'parameter_combination': exp_config['param_dict'],
                            'config': exp_config['config_dict'],
                            'index': exp_config['index'],
                            'error': worker_result['error'],
                            'accuracy': 0.0,
                            'failed': True
                        }
                    
                    results.append(result)
                    
                    if quiet:
                        progress_bar.update(1)
                        # Update progress bar description with best result so far
                        best_acc = max([r.get('accuracy', 0) for r in results if not r.get('failed', False)], default=0)
                        progress_bar.set_postfix({'Best Acc': f'{best_acc:.3f}'})
                    
                except Exception as e:
                    if not quiet:
                        param_str = ', '.join([f"{k}={v}" for k, v in exp_config['param_dict'].items()])
                        self.logger.error(f"Exception in experiment {param_str}: {e}")
                    
                    result = {
                        'parameter_combination': exp_config['param_dict'],
                        'config': exp_config['config_dict'],
                        'index': exp_config['index'],
                        'error': str(e),
                        'accuracy': 0.0,
                        'failed': True
                    }
                    results.append(result)
                    
                    if quiet:
                        progress_bar.update(1)
            
            if quiet:
                progress_bar.close()
        
        # Sort results by original index to maintain parameter combination order
        sorted_results = sorted(results, key=lambda x: x['index'])
        # Remove index from final results
        for result in sorted_results:
            result.pop('index', None)
        
        successful_runs = len([r for r in sorted_results if not r.get('failed', False)])
        if not quiet:
            self.logger.info(f"Parallel multi-parameter sweep completed. {successful_runs}/{total_experiments} successful runs")
        
        return sorted_results
