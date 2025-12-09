"""
N-Clusters Ablation Study

Conducts systematic ablation studies on the number of clusters parameter
while keeping all other parameters constant. Analyzes performance trends
and identifies optimal cluster configurations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np

# Import internal modules
from .data_collector import AblationDataCollector
from .visualization import AblationVisualizer


class ClusterAblation:
    """
    Conducts n_clusters ablation experiments for the Balance Cluster Router.
    
    This class systematically varies the number of clusters while keeping all other
    parameters constant to understand the effect of cluster granularity on 
    routing performance and cost efficiency.
    """
    
    def __init__(self, output_dir: str = "ablation"):
        """
        Initialize cluster ablation experiment.
        
        Args:
            output_dir: Directory to save results and figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_collector = AblationDataCollector(cache_dir=str(self.output_dir / "results"))
        self.visualizer = AblationVisualizer(output_dir=str(self.output_dir / "figures"))
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup logging if not configured
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def generate_cluster_configurations(self, base_config: Dict[str, Any],
                                      cluster_range: Optional[Tuple[int, int, int]] = None) -> List[int]:
        """
        Generate list of cluster counts to test.
        
        Args:
            base_config: Base configuration dictionary
            cluster_range: Optional tuple of (min_clusters, max_clusters, step)
            
        Returns:
            List of cluster counts to test
        """
        if cluster_range is None:
            # Default: comprehensive range from 8 to 80
            # Use logarithmic-like spacing for better coverage
            cluster_counts = [8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80]
        else:
            min_clusters, max_clusters, step = cluster_range
            cluster_counts = list(range(min_clusters, max_clusters + 1, step))
        
        self.logger.info(f"Testing cluster counts: {cluster_counts}")
        return cluster_counts
    
    def run_cluster_ablation(self, base_config: Dict[str, Any], 
                           cluster_range: Optional[Tuple[int, int, int]] = None,
                           load_baseline: bool = True,
                           parallel: bool = False,
                           max_workers: Optional[int] = None,
                           quiet: bool = False) -> Dict[str, Any]:
        """
        Run complete cluster ablation study.
        
        Args:
            base_config: Base configuration for experiments
            cluster_range: Optional tuple of (min_clusters, max_clusters, step)
            load_baseline: Whether to load baseline performance data
            parallel: Whether to run experiments in parallel
            max_workers: Maximum number of parallel workers (default: CPU count - 1)
            quiet: Whether to minimize output and show progress bar
            
        Returns:
            Dictionary containing all experimental results and analysis
        """
        self.logger.info("Starting n_clusters ablation study")
        
        # Generate cluster configurations
        cluster_counts = self.generate_cluster_configurations(base_config, cluster_range)
        
        # Run parameter sweep (parallel or sequential)
        if parallel:
            if not quiet:
                self.logger.info(f"Running {len(cluster_counts)} cluster experiments in parallel with {max_workers or 'auto'} workers")
            results = self.data_collector.run_parameter_sweep_parallel(
                base_config=base_config,
                parameter_name='n_clusters',
                parameter_values=cluster_counts,
                experiment_type='cluster_ablation',
                max_workers=max_workers,
                quiet=quiet
            )
        else:
            self.logger.info(f"Running experiments for {len(cluster_counts)} cluster configurations")
            results = self.data_collector.run_parameter_sweep(
                base_config=base_config,
                parameter_name='n_clusters',
                parameter_values=cluster_counts,
                experiment_type='cluster_ablation'
            )
        
        # Load baseline data if requested
        baseline_data = None
        if load_baseline:
            baseline_data = self._load_baseline_data()
        
        # Analyze results
        analysis = self._analyze_cluster_results(results, baseline_data)
        
        # Generate visualizations
        if not quiet:
            self.logger.info("Generating visualizations")
        figure_paths = self._create_visualizations(results, baseline_data)
        
        # Compile final results
        final_results = {
            'experiment_type': 'cluster_ablation',
            'timestamp': datetime.now().isoformat(),
            'base_config': base_config,
            'cluster_range': cluster_counts,
            'total_experiments': len(results),
            'successful_experiments': len([r for r in results if not r.get('failed', False)]),
            'results': results,
            'analysis': analysis,
            'baseline_data': baseline_data,
            'figure_paths': figure_paths
        }
        
        # Save results
        results_path = self.output_dir / "results" / "cluster_ablation_complete.json"
        self.data_collector.save_sweep_results(results, "cluster_ablation_sweep.json")
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        # Create concise results export
        concise_results_path = self._export_concise_results(results, baseline_data)
        if not quiet:
            self.logger.info(f"Concise results exported to {concise_results_path}")
        
        if not quiet:
            self.logger.info(f"Cluster ablation study completed. Results saved to {results_path}")
        return final_results
    
    def _analyze_cluster_results(self, results: List[Dict[str, Any]], 
                               baseline_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze cluster ablation results.
        
        Args:
            results: List of experimental results
            baseline_data: Optional baseline performance data
            
        Returns:
            Analysis dictionary with insights and statistics
        """
        analysis = {
            'performance_analysis': {},
            'cost_analysis': {},
            'efficiency_analysis': {},
            'optimal_configurations': {},
            'statistical_analysis': {}
        }
        
        # Filter successful results
        successful_results = [r for r in results if not r.get('failed', False)]
        
        if not successful_results:
            self.logger.warning("No successful results to analyze")
            return analysis
        
        # Extract key metrics
        cluster_counts = [r.get('parameter_value') for r in successful_results]
        accuracies = [r.get('accuracy', 0.0) for r in successful_results]
        total_costs = [r.get('cost_analysis', {}).get('total_cost', 0.0) for r in successful_results]
        avg_costs = [r.get('cost_analysis', {}).get('avg_cost_per_query', 0.0) for r in successful_results]
        cost_efficiencies = [r.get('cost_analysis', {}).get('cost_efficiency', 0.0) for r in successful_results]
        
        # Performance analysis
        best_acc_idx = np.argmax(accuracies)
        worst_acc_idx = np.argmin(accuracies)
        
        analysis['performance_analysis'] = {
            'best_accuracy': {
                'value': accuracies[best_acc_idx],
                'n_clusters': cluster_counts[best_acc_idx],
                'config_index': best_acc_idx
            },
            'worst_accuracy': {
                'value': accuracies[worst_acc_idx], 
                'n_clusters': cluster_counts[worst_acc_idx],
                'config_index': worst_acc_idx
            },
            'accuracy_range': max(accuracies) - min(accuracies),
            'accuracy_std': np.std(accuracies),
            'accuracy_trend': self._calculate_trend(cluster_counts, accuracies)
        }
        
        # Cost analysis
        if any(cost > 0 for cost in total_costs):
            min_cost_idx = np.argmin([c for c in total_costs if c > 0])
            valid_cost_indices = [i for i, c in enumerate(total_costs) if c > 0]
            
            if valid_cost_indices:
                min_cost_idx = valid_cost_indices[np.argmin([total_costs[i] for i in valid_cost_indices])]
                max_cost_idx = valid_cost_indices[np.argmax([total_costs[i] for i in valid_cost_indices])]
                
                analysis['cost_analysis'] = {
                    'lowest_cost': {
                        'value': total_costs[min_cost_idx],
                        'n_clusters': cluster_counts[min_cost_idx],
                        'config_index': min_cost_idx
                    },
                    'highest_cost': {
                        'value': total_costs[max_cost_idx],
                        'n_clusters': cluster_counts[max_cost_idx],
                        'config_index': max_cost_idx
                    },
                    'cost_trend': self._calculate_trend(cluster_counts, total_costs)
                }
        
        # Efficiency analysis
        if any(eff > 0 for eff in cost_efficiencies):
            best_eff_idx = np.argmax(cost_efficiencies)
            
            analysis['efficiency_analysis'] = {
                'best_efficiency': {
                    'value': cost_efficiencies[best_eff_idx],
                    'n_clusters': cluster_counts[best_eff_idx],
                    'accuracy': accuracies[best_eff_idx],
                    'avg_cost': avg_costs[best_eff_idx],
                    'config_index': best_eff_idx
                },
                'efficiency_trend': self._calculate_trend(cluster_counts, cost_efficiencies)
            }
        
        # Find optimal configurations
        analysis['optimal_configurations'] = self._find_optimal_configs(successful_results)
        
        # Sensitivity analysis
        analysis['sensitivity_analysis'] = self._analyze_cluster_sensitivity(
            cluster_counts, accuracies, avg_costs, cost_efficiencies
        )
        
        # Statistical analysis
        if len(accuracies) > 2:
            # Calculate correlation between cluster count and performance
            corr_clusters_acc = np.corrcoef(cluster_counts, accuracies)[0, 1]
            if not np.isnan(corr_clusters_acc):
                analysis['statistical_analysis']['cluster_accuracy_correlation'] = corr_clusters_acc
            
            if any(cost > 0 for cost in total_costs):
                valid_costs = [c for c in total_costs if c > 0]
                valid_clusters = [cluster_counts[i] for i, c in enumerate(total_costs) if c > 0]
                
                if len(valid_costs) > 2:
                    corr_clusters_cost = np.corrcoef(valid_clusters, valid_costs)[0, 1]
                    if not np.isnan(corr_clusters_cost):
                        analysis['statistical_analysis']['cluster_cost_correlation'] = corr_clusters_cost
        
        # Baseline comparison
        if baseline_data:
            analysis['baseline_comparison'] = self._compare_with_baselines(successful_results, baseline_data)
        
        return analysis
    
    def _calculate_trend(self, x_values: List[float], y_values: List[float]) -> str:
        """
        Calculate trend direction using linear regression.
        
        Args:
            x_values: Independent variable values
            y_values: Dependent variable values
            
        Returns:
            Trend description ('increasing', 'decreasing', 'stable', 'unknown')
        """
        try:
            if len(x_values) < 2 or len(y_values) < 2:
                return 'unknown'
                
            # Simple linear regression slope
            slope = np.polyfit(x_values, y_values, 1)[0]
            
            if abs(slope) < 1e-6:
                return 'stable'
            elif slope > 0:
                return 'increasing'
            else:
                return 'decreasing'
        except:
            return 'unknown'
    
    def _find_optimal_configs(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Find optimal configurations based on different criteria.
        
        Args:
            results: List of experimental results
            
        Returns:
            Dictionary of optimal configurations
        """
        optimal_configs = {}
        
        if not results:
            return optimal_configs
        
        # Best accuracy
        best_acc_result = max(results, key=lambda x: x.get('accuracy', 0.0))
        optimal_configs['best_accuracy'] = {
            'n_clusters': best_acc_result.get('parameter_value'),
            'accuracy': best_acc_result.get('accuracy', 0.0),
            'cost_analysis': best_acc_result.get('cost_analysis', {})
        }
        
        # Lowest cost (with reasonable accuracy > 0.3)
        reasonable_results = [r for r in results if r.get('accuracy', 0.0) > 0.3]
        if reasonable_results:
            lowest_cost_result = min(reasonable_results, 
                                   key=lambda x: x.get('cost_analysis', {}).get('avg_cost_per_query', float('inf')))
            
            optimal_configs['lowest_cost'] = {
                'n_clusters': lowest_cost_result.get('parameter_value'),
                'accuracy': lowest_cost_result.get('accuracy', 0.0),
                'cost_analysis': lowest_cost_result.get('cost_analysis', {})
            }
        
        # Best cost efficiency
        cost_efficiencies = [(r, r.get('cost_analysis', {}).get('cost_efficiency', 0.0)) for r in results]
        best_eff_result = max(cost_efficiencies, key=lambda x: x[1])[0]
        
        optimal_configs['best_cost_efficiency'] = {
            'n_clusters': best_eff_result.get('parameter_value'),
            'accuracy': best_eff_result.get('accuracy', 0.0),
            'cost_efficiency': best_eff_result.get('cost_analysis', {}).get('cost_efficiency', 0.0),
            'cost_analysis': best_eff_result.get('cost_analysis', {})
        }
        
        # Balanced configuration (accuracy * cost_efficiency)
        balanced_scores = []
        for r in results:
            acc = r.get('accuracy', 0.0)
            eff = r.get('cost_analysis', {}).get('cost_efficiency', 0.0)
            balanced_score = acc * eff if eff > 0 else acc
            balanced_scores.append((r, balanced_score))
        
        if balanced_scores:
            best_balanced_result = max(balanced_scores, key=lambda x: x[1])[0]
            optimal_configs['best_balanced'] = {
                'n_clusters': best_balanced_result.get('parameter_value'),
                'accuracy': best_balanced_result.get('accuracy', 0.0),
                'cost_efficiency': best_balanced_result.get('cost_analysis', {}).get('cost_efficiency', 0.0),
                'balanced_score': max(balanced_scores, key=lambda x: x[1])[1],
                'cost_analysis': best_balanced_result.get('cost_analysis', {})
            }
        
        # Most stable configuration (middle range cluster count)
        cluster_counts = [r.get('parameter_value') for r in results]
        if cluster_counts:
            median_cluster = sorted(cluster_counts)[len(cluster_counts) // 2]
            
            # Find result closest to median cluster count
            median_results = [(r, abs(r.get('parameter_value', 0) - median_cluster)) for r in results]
            most_stable_result = min(median_results, key=lambda x: x[1])[0]
            
            optimal_configs['most_stable'] = {
                'n_clusters': most_stable_result.get('parameter_value'),
                'accuracy': most_stable_result.get('accuracy', 0.0),
                'cost_efficiency': most_stable_result.get('cost_analysis', {}).get('cost_efficiency', 0.0),
                'cost_analysis': most_stable_result.get('cost_analysis', {})
            }
        
        return optimal_configs
    
    def _load_baseline_data(self) -> Optional[Dict[str, Any]]:
        """
        Load baseline performance data from configuration.
        
        Returns:
            Baseline data dictionary or None if not found
        """
        try:
            baseline_path = Path(__file__).parent.parent / "config" / "baseline.json"
            if baseline_path.exists():
                with open(baseline_path, 'r', encoding='utf-8') as f:
                    baseline_scores = json.load(f)
                
                # Process baseline data similar to balance_cluster_router
                model_summaries = []
                for model, scores in baseline_scores.items():
                    valid_scores = [score / 100.0 if score > 1.0 else score 
                                  for score in scores.values() if score is not None]
                    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
                    
                    model_summaries.append({
                        'model': model,
                        'avg_score': avg_score,
                        'total_cost': 0.0,  # Would need actual cost data
                        'dataset_coverage': f"{len(valid_scores)}/{len(scores)}"
                    })
                
                model_summaries.sort(key=lambda x: x['avg_score'], reverse=True)
                
                return {
                    'model_summaries': model_summaries,
                    'best_overall_baseline': model_summaries[0] if model_summaries else None
                }
        except Exception as e:
            self.logger.warning(f"Failed to load baseline data: {e}")
        
        return None
    
    def _compare_with_baselines(self, results: List[Dict[str, Any]], 
                              baseline_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare cluster ablation results with baseline models.
        
        Args:
            results: Experimental results
            baseline_data: Baseline model data
            
        Returns:
            Baseline comparison dictionary
        """
        comparison = {}
        
        if not results or not baseline_data:
            return comparison
        
        # Get best router performance
        best_result = max(results, key=lambda x: x.get('accuracy', 0.0))
        best_accuracy = best_result.get('accuracy', 0.0)
        best_cost = best_result.get('cost_analysis', {}).get('avg_cost_per_query', 0.0)
        
        # Get baseline performance
        best_baseline = baseline_data.get('best_overall_baseline')
        if best_baseline:
            baseline_acc = best_baseline.get('avg_score', 0.0)
            
            comparison['best_router_vs_best_baseline'] = {
                'router_accuracy': best_accuracy,
                'baseline_accuracy': baseline_acc,
                'improvement': best_accuracy - baseline_acc,
                'improvement_percentage': ((best_accuracy - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0,
                'router_cost': best_cost,
                'router_n_clusters': best_result.get('parameter_value')
            }
        
        # Count how many router configs beat baseline
        model_summaries = baseline_data.get('model_summaries', [])
        if model_summaries:
            baseline_accuracies = [m.get('avg_score', 0.0) for m in model_summaries]
            max_baseline_acc = max(baseline_accuracies)
            
            better_than_baseline = [r for r in results if r.get('accuracy', 0.0) > max_baseline_acc]
            
            comparison['configurations_beating_baseline'] = {
                'total_configs': len(results),
                'configs_beating_best_baseline': len(better_than_baseline),
                'percentage': (len(better_than_baseline) / len(results) * 100) if results else 0,
                'best_baseline_threshold': max_baseline_acc
            }
            
            # Find which cluster counts work best vs baselines
            if better_than_baseline:
                winning_clusters = [r.get('parameter_value') for r in better_than_baseline]
                comparison['winning_cluster_range'] = {
                    'min_clusters': min(winning_clusters),
                    'max_clusters': max(winning_clusters),
                    'median_clusters': sorted(winning_clusters)[len(winning_clusters) // 2]
                }
        
        return comparison
    
    def _analyze_cluster_sensitivity(self, cluster_counts: List[int], accuracies: List[float],
                                   costs: List[float], cost_efficiencies: List[float]) -> Dict[str, Any]:
        """
        Analyze sensitivity of metrics to cluster count changes.
        
        Args:
            cluster_counts: List of cluster count values
            accuracies: List of accuracy values
            costs: List of cost values
            cost_efficiencies: List of cost efficiency values
            
        Returns:
            Sensitivity analysis dictionary
        """
        sensitivity = {}
        
        if len(cluster_counts) < 3:
            return sensitivity
        
        # Calculate derivatives (gradients) using finite differences
        def calculate_gradient(x_vals, y_vals):
            if len(x_vals) != len(y_vals) or len(x_vals) < 2:
                return []
            gradients = []
            for i in range(1, len(x_vals)):
                dx = x_vals[i] - x_vals[i-1]
                dy = y_vals[i] - y_vals[i-1]
                if dx != 0:
                    gradients.append(dy / dx)
            return gradients
        
        # Sort by cluster count for gradient calculation
        sorted_data = sorted(zip(cluster_counts, accuracies, costs, cost_efficiencies))
        sorted_clusters, sorted_accs, sorted_costs, sorted_effs = zip(*sorted_data)
        
        # Calculate gradients
        acc_gradients = calculate_gradient(sorted_clusters, sorted_accs)
        cost_gradients = calculate_gradient(sorted_clusters, sorted_costs)
        eff_gradients = calculate_gradient(sorted_clusters, sorted_effs)
        
        if acc_gradients:
            sensitivity['accuracy_sensitivity'] = {
                'mean_gradient': float(np.mean(acc_gradients)),
                'std_gradient': float(np.std(acc_gradients)),
                'max_gradient': float(max(acc_gradients)),
                'min_gradient': float(min(acc_gradients)),
                'direction': 'increasing' if np.mean(acc_gradients) > 0 else 'decreasing'
            }
        
        if cost_gradients and any(c > 0 for c in sorted_costs):
            valid_cost_gradients = [g for g, c in zip(cost_gradients, sorted_costs[1:]) if c > 0]
            if valid_cost_gradients:
                sensitivity['cost_sensitivity'] = {
                    'mean_gradient': float(np.mean(valid_cost_gradients)),
                    'std_gradient': float(np.std(valid_cost_gradients)),
                    'max_gradient': float(max(valid_cost_gradients)),
                    'min_gradient': float(min(valid_cost_gradients)),
                    'direction': 'increasing' if np.mean(valid_cost_gradients) > 0 else 'decreasing'
                }
        
        if eff_gradients:
            sensitivity['efficiency_sensitivity'] = {
                'mean_gradient': float(np.mean(eff_gradients)),
                'std_gradient': float(np.std(eff_gradients)),
                'max_gradient': float(max(eff_gradients)),
                'min_gradient': float(min(eff_gradients)),
                'direction': 'increasing' if np.mean(eff_gradients) > 0 else 'decreasing'
            }
        
        # Identify optimal cluster range based on sensitivity
        if acc_gradients:
            stability_threshold = np.std(acc_gradients) * 0.5  # Low variance indicates stability
            stable_regions = []
            
            for i, grad in enumerate(acc_gradients):
                if abs(grad) < stability_threshold:
                    cluster_val = sorted_clusters[i+1]  # +1 because gradients are between points
                    stable_regions.append(cluster_val)
            
            if stable_regions:
                sensitivity['stable_cluster_ranges'] = {
                    'stable_points': stable_regions,
                    'recommended_range': [min(stable_regions), max(stable_regions)]
                }
        
        return sensitivity
    
    def _create_visualizations(self, results: List[Dict[str, Any]], 
                             baseline_data: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Create all visualizations for cluster ablation.
        
        Args:
            results: Experimental results
            baseline_data: Optional baseline data
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        figure_paths = {}
        
        try:
            # Main cluster ablation plot
            cluster_path = self.visualizer.plot_cluster_ablation(results, baseline_data)
            figure_paths['cluster_ablation'] = cluster_path
        except Exception as e:
            self.logger.error(f"Failed to create cluster ablation plot: {e}")
        
        return figure_paths
    
    def _export_concise_results(self, results: List[Dict[str, Any]], 
                               baseline_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Export concise cluster ablation results to a separate file.
        
        Args:
            results: List of experimental results
            baseline_data: Optional baseline model data
            
        Returns:
            Path to exported concise results file
        """
        concise_data = {
            'timestamp': datetime.now().isoformat(),
            'experiment_type': 'cluster_ablation',
            'router_results': [],
            'baseline_results': []
        }
        
        # Process router experiment results
        for result in results:
            if result.get('failed', False):
                continue
                
            cost_analysis = result.get('cost_analysis', {})
            
            concise_data['router_results'].append({
                'n_clusters': result.get('parameter_value'),
                'accuracy': result.get('accuracy', 0.0),
                'total_cost': cost_analysis.get('total_cost', 0.0),
                'avg_cost_per_query': cost_analysis.get('avg_cost_per_query', 0.0),
                'cost_efficiency': cost_analysis.get('cost_efficiency', 0.0)
            })
        
        # Process baseline results if available
        if baseline_data:
            model_summaries = baseline_data.get('model_summaries', [])
            for model_info in model_summaries:
                concise_data['baseline_results'].append({
                    'model': model_info.get('model', 'Unknown'),
                    'accuracy': model_info.get('avg_score', 0.0),
                    'total_cost': model_info.get('total_cost', 0.0),
                    'cost_per_query': 0.0,  # Baseline models don't have per-query cost data
                    'dataset_coverage': model_info.get('dataset_coverage', 'Unknown')
                })
        
        # Sort router results by n_clusters
        concise_data['router_results'].sort(key=lambda x: x.get('n_clusters', 0))
        
        # Sort baseline results by accuracy (descending)
        concise_data['baseline_results'].sort(key=lambda x: x.get('accuracy', 0), reverse=True)
        
        # Save concise results
        concise_path = self.output_dir / "results" / "cluster_ablation_concise.json"
        concise_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(concise_path, 'w', encoding='utf-8') as f:
            json.dump(concise_data, f, indent=2, ensure_ascii=False)
        
        # Also create a CSV version for easy analysis
        csv_path = self.output_dir / "results" / "cluster_ablation_results.csv"
        self._export_csv_results(concise_data, csv_path)
        
        return str(concise_path)
    
    def _export_csv_results(self, concise_data: Dict[str, Any], csv_path: Path):
        """
        Export results to CSV format for easy analysis.
        
        Args:
            concise_data: Concise results data
            csv_path: Path to save CSV file
        """
        import csv
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write router results
            writer.writerow(['# Router Experiment Results'])
            writer.writerow(['n_clusters', 'accuracy', 'total_cost', 'avg_cost_per_query', 'cost_efficiency'])
            
            for result in concise_data['router_results']:
                writer.writerow([
                    result.get('n_clusters', ''),
                    result.get('accuracy', ''),
                    result.get('total_cost', ''),
                    result.get('avg_cost_per_query', ''),
                    result.get('cost_efficiency', '')
                ])
            
            # Write baseline results if available
            if concise_data['baseline_results']:
                writer.writerow([])  # Empty row
                writer.writerow(['# Baseline Model Results'])
                writer.writerow(['model', 'accuracy', 'total_cost', 'dataset_coverage'])
                
                for baseline in concise_data['baseline_results']:
                    writer.writerow([
                        baseline.get('model', ''),
                        baseline.get('accuracy', ''),
                        baseline.get('total_cost', ''),
                        baseline.get('dataset_coverage', '')
                    ])