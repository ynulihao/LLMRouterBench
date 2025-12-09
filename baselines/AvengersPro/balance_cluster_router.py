"""
Balance Cluster Router

A cost-efficiency aware version of the cluster-based routing system that:
- Balances model performance with cost considerations using configurable weights
- Provides detailed cost-efficiency analysis and reporting
- Maintains compatibility with SimpleClusterRouter interface

Features:
- Multi-objective optimization (accuracy vs cost) with configurable weights
- Budget constraint management
- Cost-efficiency analytics and reporting
"""
import json
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import statistics

from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from .simple_cluster_router import SimpleClusterRouter
from .config import SimpleClusterConfig


@dataclass
class CostEfficiencyMetrics:
    """
    Cost-efficiency metrics for model evaluation.
    
    Attributes:
        model_name (str): Name of the model
        accuracy (float): Model accuracy in cluster
        avg_cost (float): Average cost per query
        cost_efficiency (float): Accuracy per unit cost ratio
        total_queries (int): Number of queries processed
        total_cost (float): Total cost incurred
        cost_percentile (float): Cost percentile among all models (0-100)
    """
    model_name: str
    accuracy: float
    avg_cost: float
    cost_efficiency: float
    total_queries: int
    total_cost: float
    cost_percentile: float = 0.0


class CostEfficiencyAnalyzer:
    """
    Analyzes cost-efficiency patterns across models and clusters.
    
    Provides methods for computing cost-efficiency metrics, identifying 
    high-value models, and generating cost optimization recommendations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def analyze_cluster_cost_efficiency(self, cluster_records: List[Dict], 
                                      available_models: List[str], cluster_id: int = -1) -> Dict[str, CostEfficiencyMetrics]:
        """
        Analyze cost-efficiency for all models in a cluster.
        
        Args:
            cluster_records: List of training records for the cluster
            available_models: List of available model names
            
        Returns:
            Dictionary mapping model names to CostEfficiencyMetrics
        """
        model_metrics = {}
        
        # Collect performance and cost data for each model
        model_data = defaultdict(lambda: {'successes': [], 'costs': []})
        penalty_count = defaultdict(int)  # Track penalty assignments
        total_records = defaultdict(int)  # Track total records per model
        
        # Analyze cluster data for cost efficiency
        
        for record in cluster_records:
            records = record.get('records', {})
            usages = record.get('usages')  # Don't provide default empty dict
                
            for model_name in available_models:
                if model_name in records:
                    score = records[model_name]
                    if score is not None:  # Skip None values
                        total_records[model_name] += 1
                        
                        # Get cost for this model, use high penalty if no cost data
                        cost = None
                        if (usages is not None and 
                            model_name in usages and 
                            isinstance(usages[model_name], dict) and 
                            'cost' in usages[model_name]):
                            cost = usages[model_name]['cost']
                            
                        # If no cost data available, assign zero cost
                        if cost is None or not isinstance(cost, (int, float)) or cost < 0:
                            cost = 0.0  # No cost penalty for missing data
                            penalty_count[model_name] += 1
                            
                        model_data[model_name]['successes'].append(float(score))
                        model_data[model_name]['costs'].append(cost)
        
        # Optional: Report penalty statistics for debugging
        # penalty analysis available in penalty_count and total_records if needed
        
        # Calculate cost percentiles for normalization
        all_costs = []
        for model_name in available_models:
            if model_data[model_name]['costs']:
                all_costs.extend(model_data[model_name]['costs'])
        
        # Compute metrics for each model
        for model_name in available_models:
            successes = model_data[model_name]['successes']
            costs = model_data[model_name]['costs']
            
            if successes and costs:
                accuracy = np.mean(successes)
                avg_cost = np.mean(costs)
                total_queries = len(successes)
                total_cost = sum(costs)
                
                # Cost-efficiency: accuracy per unit cost
                cost_efficiency = accuracy / (avg_cost + 1e-8)  # Add small epsilon to avoid division by zero
                
                # Calculate cost percentile using correct algorithm
                cost_percentile = 0.0
                if all_costs:
                    sorted_costs = sorted(all_costs)
                    cost_percentile = (np.searchsorted(sorted_costs, avg_cost, side='left') / len(all_costs)) * 100
                
                model_metrics[model_name] = CostEfficiencyMetrics(
                    model_name=model_name,
                    accuracy=accuracy,
                    avg_cost=avg_cost,
                    cost_efficiency=cost_efficiency,
                    total_queries=total_queries,
                    total_cost=total_cost,
                    cost_percentile=cost_percentile
                )
            else:
                # Model has no data in this cluster
                model_metrics[model_name] = CostEfficiencyMetrics(
                    model_name=model_name,
                    accuracy=0.0,
                    avg_cost=float('inf'),
                    cost_efficiency=0.0,
                    total_queries=0,
                    total_cost=0.0,
                    cost_percentile=100.0
                )
        
        return model_metrics
    
    def compute_balance_score(self, metrics: CostEfficiencyMetrics, 
                            performance_weight: float, cost_sensitivity: float,
                            max_cost_in_cluster: float, 
                            max_accuracy_in_cluster: float, min_accuracy_in_cluster: float) -> float:
        """
        Compute balanced score considering both performance and cost.
        
        Args:
            metrics: Cost-efficiency metrics for the model
            performance_weight: Weight for performance component (0.0-1.0)
            cost_sensitivity: Weight for cost component (0.0-1.0)  
            max_cost_in_cluster: Maximum cost in the cluster for normalization
            max_accuracy_in_cluster: Maximum accuracy in the cluster for normalization
            min_accuracy_in_cluster: Minimum accuracy in the cluster for normalization
            
        Returns:
            Balanced score (higher is better)
        """
        if metrics.total_queries == 0:
            return 0.0
        
        # Normalize accuracy to 0-1 range within cluster
        accuracy_range = max_accuracy_in_cluster - min_accuracy_in_cluster
        if accuracy_range > 0:
            normalized_accuracy = (metrics.accuracy - min_accuracy_in_cluster) / accuracy_range
        else:
            normalized_accuracy = 1.0  # All models have same accuracy
            
        # Normalize cost to 0-1 range (lower cost = higher normalized score)
        if max_cost_in_cluster > 0:
            normalized_cost = metrics.avg_cost / max_cost_in_cluster
            cost_score = 1.0 - normalized_cost  # Invert: lower cost = higher score
        else:
            cost_score = 1.0
            
        # Combine normalized performance and cost with weights
        balance_score = performance_weight * normalized_accuracy + cost_sensitivity * cost_score
        
        return balance_score
    
    def find_pareto_optimal_models(self, metrics_dict: Dict[str, CostEfficiencyMetrics]) -> List[str]:
        """
        Find Pareto optimal models (non-dominated in accuracy-cost space).
        
        Args:
            metrics_dict: Dictionary of model metrics
            
        Returns:
            List of Pareto optimal model names
        """
        models = [(name, metrics.accuracy, metrics.avg_cost) 
                 for name, metrics in metrics_dict.items() 
                 if metrics.total_queries > 0]
        
        pareto_optimal = []
        
        for i, (name_i, acc_i, cost_i) in enumerate(models):
            is_dominated = False
            
            for j, (name_j, acc_j, cost_j) in enumerate(models):
                if i != j:
                    # j dominates i if j is better or equal in both dimensions
                    # and strictly better in at least one
                    if (acc_j >= acc_i and cost_j <= cost_i and 
                        (acc_j > acc_i or cost_j < cost_i)):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_optimal.append(name_i)
        
        return pareto_optimal


class BalanceClusterRouter(SimpleClusterRouter):
    """
    Balance-aware cluster router that considers both performance and cost.
    
    Extends SimpleClusterRouter with cost-efficiency optimization capabilities.
    Routes queries to models that provide the best balance of accuracy and cost
    based on configurable preferences and constraints.
    """
    
    def __init__(self, config: SimpleClusterConfig):
        """
        Initialize the balance cluster router.
        
        Args:
            config: Configuration with balance routing parameters
        """
        super().__init__(config)
        self.cost_analyzer = CostEfficiencyAnalyzer()
        self.cluster_cost_metrics: Optional[Dict[int, Dict[str, CostEfficiencyMetrics]]] = None
        
        self.logger.info("Initialized BalanceClusterRouter")
        self.logger.info(f"Performance weight: {config.performance_weight}, Cost sensitivity: {config.cost_sensitivity}")
    
    def load_and_split_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load data and extract cost information along with performance data.
        
        Extends parent method to validate cost data availability.
        """
        train_data, test_data = super().load_and_split_data()
        
        # Validate that cost data is available
        cost_data_count = 0
        for item in train_data:
            if 'usages' in item and item['usages']:
                cost_data_count += 1
        
        if cost_data_count == 0:
            raise ValueError("No cost data found in training set. Balance routing requires 'usages' field with cost information.")
        
        cost_coverage = cost_data_count / len(train_data) * 100
        self.logger.info(f"Cost data coverage: {cost_coverage:.1f}% of training samples")
        
        if cost_coverage < 50:
            self.logger.warning("Low cost data coverage may affect balance routing quality")
        
        return train_data, test_data
    
    def build_cluster_model(self, train_data: List[Dict]):
        """Build cluster model and compute cluster-wise rankings from training data with cost information."""
        print(f"\n=== Building Cluster Model (n_clusters={self.config.n_clusters}) ===")
        
        queries = [item["query"] for item in train_data]

        # Discover available models from training records (apply exclusions)
        model_set = set()
        for item in train_data:
            rec = item.get('records', {})
            if isinstance(rec, dict):
                model_set.update([m for m in rec.keys() if m is not None])

        # Apply excluded_models filter
        if self.config.excluded_models:
            model_set = {m for m in model_set if m not in self.config.excluded_models}

        self.available_models = sorted(model_set)
        if not self.available_models:
            raise ValueError("No available models found in training data after filtering. Check records/exclusions.")
        print(f"Discovered {len(self.available_models)} available models for routing")
        
        # Generate embeddings
        print(f"Generating embeddings for {len(queries)} training queries...")
        train_embeddings = self._generate_embeddings_concurrent(queries)
        train_embeddings = np.array(train_embeddings)
        
        # Create and fit normalizer
        self.normalizer = Normalizer(norm='l2')
        train_embeddings = self.normalizer.fit_transform(train_embeddings)
        
        print(f"Built embeddings matrix: {train_embeddings.shape}")
        
        # Perform K-means clustering
        print(f"Performing K-means clustering with {self.config.n_clusters} clusters...")
        self.kmeans_model = KMeans(
            n_clusters=self.config.n_clusters,
            random_state=self.config.seed,
            n_init=10
        )
        cluster_labels = self.kmeans_model.fit_predict(train_embeddings)
        self.cluster_centers = self.kmeans_model.cluster_centers_
        
        print(f"Clustering completed. Cluster centers shape: {self.cluster_centers.shape}")
        
        # Log cluster sample distribution
        cluster_counts = defaultdict(int)
        for label in cluster_labels:
            cluster_counts[label] += 1
        
        print(f"Cluster sample distribution:")
        for cluster_id in sorted(cluster_counts.keys()):
            count = cluster_counts[cluster_id]
            print(f"  Cluster {cluster_id:2d}: {count:4d} samples")
        
        # Compute cluster-wise rankings with full data (including usages)
        print("Computing cluster-wise expert rankings...")
        self.cluster_rankings = self._compute_cluster_rankings(cluster_labels, train_data)
        
        print("Cluster model built successfully")
        print(f"Clusters with data: {len(self.cluster_rankings)}")
        
        # Export cluster models if configured
        if self.config.export_cluster:
            self.export_cluster_models(self.config.export_cluster)
    
    def _compute_cluster_rankings(self, cluster_labels: np.ndarray, train_data: List[Dict]) -> Dict[int, Dict]:
        """
        Compute cluster rankings with cost-efficiency considerations.
        
        Extends parent method to include cost-efficiency analysis and balance scoring.
        """
        cluster_rankings = {}
        cluster_cost_metrics = {}
        
        # Group data by cluster (full data including usages)
        cluster_data = defaultdict(list)
        for i, cluster_id in enumerate(cluster_labels):
            cluster_data[cluster_id].append(train_data[i])
        
        # Compute rankings for each cluster
        self.logger.info(f"Computing balance rankings for {len(cluster_data)} clusters")
        for cluster_id, records_list in cluster_data.items():
            if len(records_list) == 0:
                continue
            
            # Analyze cost-efficiency for this cluster
            metrics_dict = self.cost_analyzer.analyze_cluster_cost_efficiency(
                records_list, self.available_models, cluster_id
            )
            cluster_cost_metrics[cluster_id] = metrics_dict
            
            # Find min/max values in cluster for normalization (only among models with data)
            valid_metrics = [metrics for metrics in metrics_dict.values() 
                           if metrics.total_queries > 0 and metrics.avg_cost != float('inf')]
            
            if valid_metrics:
                max_cost = max(m.avg_cost for m in valid_metrics)
                max_accuracy = max(m.accuracy for m in valid_metrics)
                min_accuracy = min(m.accuracy for m in valid_metrics)
            else:
                max_cost = max_accuracy = min_accuracy = 1.0
            
            # Compute balance scores for all models
            balance_scores = {}
            for model_name, metrics in metrics_dict.items():
                balance_score = self.cost_analyzer.compute_balance_score(
                    metrics, 
                    self.config.performance_weight,
                    self.config.cost_sensitivity,
                    max_cost,
                    max_accuracy,
                    min_accuracy
                )
                
                # Apply minimum accuracy threshold
                if metrics.accuracy >= self.config.min_accuracy_threshold:
                    balance_scores[model_name] = balance_score
                else:
                    balance_scores[model_name] = 0.0  # Exclude models below threshold
            
            # Sort models by balance score (higher is better)
            sorted_models = sorted(balance_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Store cluster ranking information
            cluster_rankings[cluster_id] = {
                'total': len(records_list),
                'balance_scores': dict(sorted_models),
                'ranking': [model_name for model_name, _ in sorted_models],
                'metrics': metrics_dict,
                'pareto_optimal': self.cost_analyzer.find_pareto_optimal_models(metrics_dict)
            }
        
        # Store cost metrics for later analysis
        self.cluster_cost_metrics = cluster_cost_metrics
        
        self.logger.info("Balance cluster rankings computed successfully")
        return cluster_rankings
    
    def evaluate_routing(self, test_data: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate routing performance with cost-efficiency metrics.
        
        Extends parent evaluation to include additional balance-specific cost analysis.
        """
        # Parent class now handles all baseline analysis and cost analysis
        results = super().evaluate_routing(test_data)
        
        # Balance-specific analysis could be added here if needed in the future
        
        return results
    
    def _analyze_routing_costs(self, test_data: List[Dict], routing_details: List[Dict], results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the cost implications of routing decisions.
        
        Args:
            test_data: Original test dataset with cost information
            routing_details: Detailed routing results
            results: Overall routing results for context
            
        Returns:
            Cost analysis dictionary
        """
        total_cost = 0.0
        total_queries = len(test_data)
        model_costs = defaultdict(float)
        model_queries = defaultdict(int)
        cost_per_dataset = defaultdict(float)
        correct_cost = 0.0
        incorrect_cost = 0.0
        ood_total_cost = 0.0
        non_ood_total_cost = 0.0
        
        for i, (test_item, routing_result) in enumerate(zip(test_data, routing_details)):
            selected_models = routing_result['selected_models']
            is_correct = routing_result['is_correct']
            dataset = routing_result['dataset']
            
            # Get cost for selected models
            query_cost = 0.0
            usages = test_item.get('usages')
            
            for model_name in selected_models:
                cost = None
                if (usages is not None and 
                    model_name in usages and 
                    isinstance(usages[model_name], dict) and
                    'cost' in usages[model_name]):
                    cost = usages[model_name]['cost']
                
                # If no cost data available, assign zero cost
                if cost is None or not isinstance(cost, (int, float)) or cost < 0:
                    cost = 0.0  # No cost penalty for missing data
                    
                query_cost += cost
                model_costs[model_name] += cost
                model_queries[model_name] += 1
            
            total_cost += query_cost
            cost_per_dataset[dataset] += query_cost

            # Track OOD vs non-OOD costs
            if dataset in self.config.ood_datasets:
                ood_total_cost += query_cost
            else:
                non_ood_total_cost += query_cost

            if is_correct:
                correct_cost += query_cost
            else:
                incorrect_cost += query_cost
        
        # Get accuracy from results for cost efficiency calculation
        accuracy = results.get('accuracy', 0.0)
        correct_routes = results.get('correct_routes', 0)
        
        # Calculate cost-efficiency metrics
        avg_cost_per_query = total_cost / total_queries if total_queries > 0 else 0.0
        cost_per_correct = correct_cost / correct_routes if correct_routes > 0 else 0.0
        
        # Cost efficiency: accuracy per unit cost
        overall_cost_efficiency = accuracy / (avg_cost_per_query + 1e-8)
        
        cost_analysis = {
            'total_cost': total_cost,
            'avg_cost_per_query': avg_cost_per_query,
            'cost_per_correct_prediction': cost_per_correct,
            'cost_efficiency': overall_cost_efficiency,
            'model_costs': dict(model_costs),
            'dataset_costs': dict(cost_per_dataset),
            'cost_distribution': {
                'correct_predictions': correct_cost,
                'incorrect_predictions': incorrect_cost
            },
            'ood_total_cost': ood_total_cost,
            'non_ood_total_cost': non_ood_total_cost
        }
        
        return cost_analysis
    
    def print_evaluation_results(self, results: Dict[str, Any], test_data: List[Dict] = None):
        """
        Print detailed evaluation results including cost-efficiency analysis.
        """
        # Display prominent balance parameters
        print("\n" + "="*80)
        print("🔄 BALANCE MODE - COST/PERFORMANCE PARAMETERS")
        print("="*80)
        print(f"📊 Performance Weight: {self.config.performance_weight:.1f} ({self.config.performance_weight*100:.0f}%)")
        print(f"💰 Cost Sensitivity:   {self.config.cost_sensitivity:.1f} ({self.config.cost_sensitivity*100:.0f}%)")
        print(f"🎯 Min Accuracy Threshold: {self.config.min_accuracy_threshold:.2f}")
        if hasattr(self.config, 'budget_limit') and self.config.budget_limit:
            print(f"💳 Budget Limit: ${self.config.budget_limit:.4f}")
        print("="*80)
        
        # Print the basic results with baseline comparison and cost analysis
        # (cost analysis is now handled in the parent class)
        super().print_evaluation_results(results, test_data)
    
    def export_cluster_models(self, export_path: str):
        """
        Export trained balance cluster models to disk.
        
        Args:
            export_path: Directory path to save the models
            
        Saves (extends parent method):
            - normalizer.joblib: L2 normalizer  
            - cluster_centers.npy: K-means cluster centers
            - cluster_rankings.json: Enhanced rankings with balance scores
            - balance_metrics.json: Cost-efficiency metrics per cluster/model
            - metadata.json: Enhanced metadata with balance configuration
        """
        if not all([self.normalizer, self.kmeans_model, self.cluster_rankings]):
            raise ValueError("Models must be trained before exporting. Call build_cluster_model() first.")
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Export basic components using parent method
            super().export_cluster_models(export_path)
            
            # Export balance-specific cost metrics
            if hasattr(self, 'cluster_cost_metrics') and self.cluster_cost_metrics:
                balance_metrics = {}
                for cluster_id, model_metrics in self.cluster_cost_metrics.items():
                    balance_metrics[str(cluster_id)] = {}
                    for model_name, metrics in model_metrics.items():
                        # Convert CostEfficiencyMetrics to dict for JSON serialization
                        balance_metrics[str(cluster_id)][model_name] = {
                            'model_name': metrics.model_name,
                            'accuracy': metrics.accuracy,
                            'avg_cost': metrics.avg_cost,
                            'cost_efficiency': metrics.cost_efficiency,
                            'total_queries': metrics.total_queries,
                            'total_cost': metrics.total_cost,
                            'cost_percentile': metrics.cost_percentile
                        }
                
                balance_path = export_dir / "balance_metrics.json"
                with open(balance_path, 'w', encoding='utf-8') as f:
                    json.dump(balance_metrics, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Exported balance metrics to {balance_path}")
            
            # Update metadata with balance-specific configuration
            metadata_path = export_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Add balance-specific config
                metadata['balance_config'] = {
                    'cost_sensitivity': self.config.cost_sensitivity,
                    'performance_weight': self.config.performance_weight,
                    'min_accuracy_threshold': self.config.min_accuracy_threshold,
                    'budget_limit': self.config.budget_limit
                }
                metadata['router_type'] = 'BalanceClusterRouter'
                
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Updated metadata with balance configuration")
            
            print(f"✅ Successfully exported balance cluster models to: {export_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to export balance cluster models: {e}")
            raise RuntimeError(f"Balance export failed: {e}")


def main():
    """Main entry point for Balance Cluster Router testing."""
    from datetime import datetime
    import argparse
    
    parser = argparse.ArgumentParser(description='Balance Cluster Router - Performance vs Cost Optimization')
    parser.add_argument('--train-data', type=str, help='Path to train JSONL file')
    parser.add_argument('--test-data', type=str, help='Path to test JSONL file')
    parser.add_argument('--output', type=str, help='Output file path for results')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--performance_weight', type=float, default=0.7,
                       help='Performance weight (0.0-1.0)')
    parser.add_argument('--cost_sensitivity', type=float, default=0.3,
                       help='Cost sensitivity weight (0.0-1.0)')
    parser.add_argument('--min_accuracy', type=float, default=0.0,
                       help='Minimum accuracy threshold (0.0-1.0)')
    
    # Include all SimpleClusterRouter arguments
    parser.add_argument('--clusters', type=int, default=32, help='Number of clusters')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_router', type=int, default=1, help='Number of models to select')
    parser.add_argument('--excluded_models', type=str,
                       help='Comma-separated list of models to exclude from routing')
    parser.add_argument('--excluded_datasets', type=str,
                       help='Comma-separated list of datasets to exclude from evaluation')
    parser.add_argument('--ood_datasets', type=str,
                       help='Comma-separated list of out-of-distribution datasets for separate evaluation')
    parser.add_argument('--dataset_exclusion_mode', choices=['soft', 'hard'], default='hard',
                       help='Dataset exclusion mode: soft (exclude from eval only) or hard (exclude completely)')
    parser.add_argument('--export_cluster', type=str,
                       help='Directory path to export trained cluster models (normalizer, centers, rankings, balance metrics)')
    
    args = parser.parse_args()
    
    # Setup logging
    from .config import setup_logging
    setup_logging('INFO')
    logger = logging.getLogger(__name__)
    
    try:
        # Create configuration with balance parameters
        if args.config:
            # Load base configuration from file
            config = SimpleClusterConfig.from_file(args.config)
            
            # Override with command line arguments if provided
            if args.performance_weight != 0.7:  # Default value check
                config.performance_weight = args.performance_weight
            if args.cost_sensitivity != 0.3:  # Default value check
                config.cost_sensitivity = args.cost_sensitivity
            if args.min_accuracy != 0.0:  # Default value check
                config.min_accuracy_threshold = args.min_accuracy
            if args.export_cluster is not None:
                config.export_cluster = args.export_cluster
            if args.excluded_models:
                excluded_models = [model.strip() for model in args.excluded_models.split(",") if model.strip()]
                config.excluded_models = excluded_models
            if args.excluded_datasets:
                excluded_datasets = [dataset.strip() for dataset in args.excluded_datasets.split(",") if dataset.strip()]
                config.excluded_datasets = excluded_datasets
            if args.ood_datasets:
                ood_datasets = [dataset.strip() for dataset in args.ood_datasets.split(",") if dataset.strip()]
                config.ood_datasets = ood_datasets
            if args.dataset_exclusion_mode != 'hard':  # Default value check
                config.dataset_exclusion_mode = args.dataset_exclusion_mode
                
        else:
            if not args.train_data or not args.test_data:
                parser.error("--train-data and --test-data are required when not using --config")
            
            # Parse excluded models and datasets from command line
            excluded_models = []
            if args.excluded_models:
                excluded_models = [model.strip() for model in args.excluded_models.split(",") if model.strip()]
            
            excluded_datasets = []
            if args.excluded_datasets:
                excluded_datasets = [dataset.strip() for dataset in args.excluded_datasets.split(",") if dataset.strip()]
            
            ood_datasets = []
            if args.ood_datasets:
                ood_datasets = [dataset.strip() for dataset in args.ood_datasets.split(",") if dataset.strip()]
            
            config = SimpleClusterConfig.from_env(
                train_data_path=args.train_data,
                test_data_path=args.test_data,
                n_clusters=args.clusters,
                seed=args.seed,
                max_router=args.max_router,
                performance_weight=args.performance_weight,
                cost_sensitivity=args.cost_sensitivity,
                min_accuracy_threshold=args.min_accuracy,
                excluded_models=excluded_models,
                excluded_datasets=excluded_datasets,
                ood_datasets=ood_datasets,
                dataset_exclusion_mode=args.dataset_exclusion_mode,
                export_cluster=args.export_cluster
            )
        
        # Initialize and run balance router
        router = BalanceClusterRouter(config)
        results = router.run_routing()
        
        # Prepare results for serialization
        results_serializable = {
            'timestamp': datetime.now().isoformat(),
            'config': config.to_dict(),
            'results': {
                'accuracy': results['accuracy'],
                'correct_routes': results['correct_routes'],
                'total_queries': results['total_queries'],
                'dataset_performance': dict(results['dataset_performance']),
                'model_selection_stats': dict(results['model_selection_stats']),
                # Add OOD/non-OOD metrics
                'ood_accuracy': results.get('ood_accuracy', 0.0),
                'non_ood_accuracy': results.get('non_ood_accuracy', 0.0),
                'ood_sample_avg': results.get('ood_sample_avg', 0.0),
                'non_ood_sample_avg': results.get('non_ood_sample_avg', 0.0),
                'all_sample_avg': results.get('all_sample_avg', 0.0),
                # Add cost analysis
                'cost_analysis': results.get('cost_analysis', {}),
                # Add baseline analysis if available
                'baseline_analysis': results.get('baseline_analysis', {})
            }
        }
        
        # Save results to JSON if output specified
        json_path = None
        if args.output:
            # Ensure output is in results directory
            json_path = Path("results") / Path(args.output).name
            json_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results_serializable, f, indent=2, ensure_ascii=False)
            
            print(f"\nResults saved to: {json_path}")
        
        # Always auto-export to markdown format
        try:
            from experiment_exporter import ExperimentExporter
            
            # Create markdown filename
            if args.output:
                md_filename = Path(args.output).stem + ".md"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                md_filename = f"balance_experiment_{timestamp}.md"
            
            exporter = ExperimentExporter("experiment_reports")
            md_path = exporter.export_balance_results(results_serializable, config.to_dict(), md_filename)
            
            print(f"📄 Markdown report exported to: {md_path}")
            
        except Exception as e:
            logger.warning(f"Failed to export markdown report: {e}")
            if json_path:
                logger.info(f"You can manually export using: python experiment_exporter.py --results {json_path}")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()
