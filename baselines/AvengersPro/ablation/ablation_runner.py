"""
Ablation Study Runner

Main entry point for running ablation experiments on the Balance Cluster Router.
Provides command-line interface and orchestrates different types of ablation studies.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .cluster_ablation import ClusterAblation
from .weight_ablation import WeightAblation
from .visualization import AblationVisualizer
from ..config import SimpleClusterConfig, setup_logging


class AblationRunner:
    """
    Main orchestrator for ablation studies.
    
    Provides unified interface for running different types of ablation experiments
    and generating comprehensive reports with visualizations.
    """
    
    def __init__(self, output_dir: str = "ablation"):
        """
        Initialize ablation runner.
        
        Args:
            output_dir: Base directory for all ablation outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize experiment modules
        self.cluster_ablation = ClusterAblation(output_dir=str(self.output_dir))
        self.weight_ablation = WeightAblation(output_dir=str(self.output_dir))
        self.visualizer = AblationVisualizer(output_dir=str(self.output_dir / "figures"))
    
    def run_cluster_ablation(self, config: Dict[str, Any], 
                           cluster_range: Optional[tuple] = None) -> Dict[str, Any]:
        """
        Run n_clusters ablation study.
        
        Args:
            config: Base configuration dictionary
            cluster_range: Optional (min, max, step) for cluster counts
            
        Returns:
            Complete results dictionary
        """
        self.logger.info("Starting cluster ablation study")
        
        results = self.cluster_ablation.run_cluster_ablation(
            base_config=config,
            cluster_range=cluster_range,
            load_baseline=True
        )
        
        self.logger.info("Cluster ablation study completed successfully")
        return results
    
    def run_weight_ablation(self, config: Dict[str, Any],
                          num_points: int = 11,
                          weight_range: Optional[tuple] = None) -> Dict[str, Any]:
        """
        Run cost/performance weight ablation study.
        
        Args:
            config: Base configuration dictionary
            num_points: Number of weight combinations to test
            weight_range: Optional (min_perf_weight, max_perf_weight)
            
        Returns:
            Complete results dictionary
        """
        self.logger.info("Starting weight ablation study")
        
        results = self.weight_ablation.run_weight_ablation(
            base_config=config,
            num_points=num_points,
            weight_range=weight_range,
            load_baseline=True
        )
        
        self.logger.info("Weight ablation study completed successfully")
        return results
    
    def run_comprehensive_study(self, config: Dict[str, Any],
                              cluster_range: Optional[tuple] = None,
                              num_weight_points: int = 11,
                              weight_range: Optional[tuple] = None) -> Dict[str, Any]:
        """
        Run both cluster and weight ablation studies.
        
        Args:
            config: Base configuration dictionary
            cluster_range: Optional (min, max, step) for cluster counts
            num_weight_points: Number of weight combinations to test
            weight_range: Optional (min_perf_weight, max_perf_weight)
            
        Returns:
            Combined results dictionary
        """
        self.logger.info("Starting comprehensive ablation study")
        
        # Run cluster ablation
        cluster_results = self.run_cluster_ablation(config, cluster_range)
        
        # Run weight ablation
        weight_results = self.run_weight_ablation(config, num_weight_points, weight_range)
        
        # Create summary visualization
        self.logger.info("Creating comprehensive summary visualization")
        try:
            summary_path = self.visualizer.create_summary_plot(
                cluster_results['results'],
                weight_results['results'],
                cluster_results.get('baseline_data'),
                save_path=str(self.output_dir / "figures" / "comprehensive_summary.png")
            )
            
            self.logger.info(f"Summary plot saved to {summary_path}")
        except Exception as e:
            self.logger.error(f"Failed to create summary plot: {e}")
            summary_path = None
        
        # Combine results
        comprehensive_results = {
            'experiment_type': 'comprehensive_ablation',
            'timestamp': datetime.now().isoformat(),
            'base_config': config,
            'cluster_ablation': cluster_results,
            'weight_ablation': weight_results,
            'summary_plot': summary_path
        }
        
        # Save comprehensive results
        results_path = self.output_dir / "results" / "comprehensive_ablation.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Comprehensive ablation study completed. Results saved to {results_path}")
        return comprehensive_results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate markdown report from ablation results.
        
        Args:
            results: Ablation results dictionary
            
        Returns:
            Path to generated report
        """
        experiment_type = results.get('experiment_type', 'unknown')
        timestamp = results.get('timestamp', datetime.now().isoformat())
        
        report_lines = []
        
        # Header
        report_lines.extend([
            f"# Ablation Study Report: {experiment_type.replace('_', ' ').title()}",
            f"",
            f"**Generated:** {timestamp}",
            f"**Experiment Type:** {experiment_type}",
            f"",
            f"---",
            f""
        ])
        
        # Base configuration
        base_config = results.get('base_config', {})
        if base_config:
            report_lines.extend([
                f"## Base Configuration",
                f"",
                f"- **Data Path:** {base_config.get('data_path', 'N/A')}",
                f"- **Training Ratio:** {base_config.get('train_ratio', 'N/A')}",
                f"- **Random Seed:** {base_config.get('seed', 'N/A')}",
                f"- **Max Router:** {base_config.get('max_router', 'N/A')}",
                f"- **Embedding Model:** {base_config.get('embedding_model', 'N/A')}",
                f"",
                f"---",
                f""
            ])
        
        # Experiment-specific sections
        if experiment_type == 'cluster_ablation':
            report_lines.extend(self._generate_cluster_report_section(results))
        elif experiment_type == 'weight_ablation':
            report_lines.extend(self._generate_weight_report_section(results))
        elif experiment_type == 'comprehensive_ablation':
            report_lines.extend(self._generate_comprehensive_report_section(results))
        
        # Recommendations
        report_lines.extend([
            f"## 🎯 Recommendations",
            f"",
            f"Based on the ablation study results:",
            f"",
            *self._generate_recommendations(results),
            f"",
            f"---",
            f""
        ])
        
        # Footer
        report_lines.extend([
            f"## 📝 Technical Notes",
            f"",
            f"- This report was automatically generated by the Balance Cluster Router Ablation System",
            f"- All experiments use the same base configuration with systematic parameter variations", 
            f"- Statistical significance and confidence intervals should be considered for production deployment",
            f"- Results may vary with different datasets and baseline models",
            f"",
            f"**Report generated at:** {datetime.now().isoformat()}",
            f""
        ])
        
        # Save report
        report_filename = f"{experiment_type}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path = self.output_dir / "reports" / report_filename
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Report generated: {report_path}")
        return str(report_path)
    
    def _generate_cluster_report_section(self, results: Dict[str, Any]) -> list:
        """Generate cluster ablation specific report section."""
        lines = []
        
        analysis = results.get('analysis', {})
        cluster_range = results.get('cluster_range', [])
        successful_experiments = results.get('successful_experiments', 0)
        
        lines.extend([
            f"## 🔍 N-Clusters Ablation Results",
            f"",
            f"**Cluster Range Tested:** {min(cluster_range)} to {max(cluster_range)} ({len(cluster_range)} configurations)",
            f"**Successful Experiments:** {successful_experiments}/{results.get('total_experiments', 0)}",
            f""
        ])
        
        # Performance analysis
        perf_analysis = analysis.get('performance_analysis', {})
        if perf_analysis:
            best_acc = perf_analysis.get('best_accuracy', {})
            lines.extend([
                f"### Performance Analysis",
                f"",
                f"- **Best Accuracy:** {best_acc.get('value', 0):.1%} (n_clusters={best_acc.get('n_clusters', 'N/A')})",
                f"- **Accuracy Range:** {perf_analysis.get('accuracy_range', 0):.3f}",
                f"- **Accuracy Std:** {perf_analysis.get('accuracy_std', 0):.3f}",
                f"- **Performance Trend:** {perf_analysis.get('accuracy_trend', 'unknown')}",
                f""
            ])
        
        # Optimal configurations
        optimal_configs = analysis.get('optimal_configurations', {})
        if optimal_configs:
            lines.extend([
                f"### Optimal Configurations",
                f""
            ])
            
            for config_name, config_data in optimal_configs.items():
                config_title = config_name.replace('_', ' ').title()
                lines.extend([
                    f"**{config_title}:**",
                    f"- N-Clusters: {config_data.get('n_clusters')}",
                    f"- Accuracy: {config_data.get('accuracy', 0):.1%}",
                    f""
                ])
        
        return lines
    
    def _generate_weight_report_section(self, results: Dict[str, Any]) -> list:
        """Generate weight ablation specific report section."""
        lines = []
        
        analysis = results.get('analysis', {})
        weight_configs = results.get('weight_configurations', [])
        successful_experiments = results.get('successful_experiments', 0)
        
        lines.extend([
            f"## ⚖️ Weight Ablation Results",
            f"",
            f"**Weight Configurations Tested:** {len(weight_configs)} combinations",
            f"**Successful Experiments:** {successful_experiments}/{results.get('total_experiments', 0)}",
            f""
        ])
        
        # Pareto analysis
        pareto_analysis = analysis.get('pareto_analysis', {})
        if pareto_analysis:
            pareto_points = pareto_analysis.get('pareto_frontier_points', 0)
            lines.extend([
                f"### Pareto Frontier Analysis",
                f"",
                f"- **Pareto Optimal Configurations:** {pareto_points}",
                f"- **Trade-off Efficiency:** Multiple optimal solutions found",
                f""
            ])
        
        # Optimal configurations
        optimal_configs = analysis.get('optimal_configurations', {})
        if optimal_configs:
            lines.extend([
                f"### Optimal Weight Configurations",
                f""
            ])
            
            for config_name, config_data in optimal_configs.items():
                config_title = config_name.replace('_', ' ').title()
                perf_weight = config_data.get('performance_weight', 0)
                cost_sens = config_data.get('cost_sensitivity', 0)
                lines.extend([
                    f"**{config_title}:**",
                    f"- Performance Weight: {perf_weight:.2f} ({perf_weight*100:.0f}%)",
                    f"- Cost Sensitivity: {cost_sens:.2f} ({cost_sens*100:.0f}%)",
                    f"- Accuracy: {config_data.get('accuracy', 0):.1%}",
                    f""
                ])
        
        return lines
    
    def _generate_comprehensive_report_section(self, results: Dict[str, Any]) -> list:
        """Generate comprehensive ablation report section."""
        lines = []
        
        cluster_results = results.get('cluster_ablation', {})
        weight_results = results.get('weight_ablation', {})
        
        lines.extend([
            f"## 📊 Comprehensive Ablation Results",
            f"",
            f"This study combines both n-clusters and weight ablation experiments.",
            f""
        ])
        
        # Cluster results summary
        if cluster_results:
            cluster_analysis = cluster_results.get('analysis', {})
            cluster_optimal = cluster_analysis.get('optimal_configurations', {})
            
            if cluster_optimal.get('best_accuracy'):
                best_cluster_config = cluster_optimal['best_accuracy']
                lines.extend([
                    f"### Cluster Ablation Highlights",
                    f"",
                    f"- **Optimal N-Clusters:** {best_cluster_config.get('n_clusters')}",
                    f"- **Best Cluster Accuracy:** {best_cluster_config.get('accuracy', 0):.1%}",
                    f""
                ])
        
        # Weight results summary
        if weight_results:
            weight_analysis = weight_results.get('analysis', {})
            weight_optimal = weight_analysis.get('optimal_configurations', {})
            
            if weight_optimal.get('best_accuracy'):
                best_weight_config = weight_optimal['best_accuracy']
                lines.extend([
                    f"### Weight Ablation Highlights",
                    f"",
                    f"- **Optimal Performance Weight:** {best_weight_config.get('performance_weight', 0):.2f}",
                    f"- **Optimal Cost Sensitivity:** {best_weight_config.get('cost_sensitivity', 0):.2f}",
                    f"- **Best Weight Accuracy:** {best_weight_config.get('accuracy', 0):.1%}",
                    f""
                ])
        
        return lines
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> list:
        """Generate recommendations based on results."""
        recommendations = []
        
        experiment_type = results.get('experiment_type', '')
        
        if 'cluster' in experiment_type:
            # Cluster-specific recommendations
            analysis = results.get('analysis', {})
            if 'cluster_ablation' in results:
                analysis = results['cluster_ablation'].get('analysis', {})
            
            optimal_configs = analysis.get('optimal_configurations', {})
            if optimal_configs.get('best_accuracy'):
                best_n_clusters = optimal_configs['best_accuracy'].get('n_clusters')
                recommendations.append(f"- **Recommended n_clusters:** {best_n_clusters} for optimal accuracy")
            
            perf_analysis = analysis.get('performance_analysis', {})
            if perf_analysis.get('accuracy_trend') == 'increasing':
                recommendations.append(f"- **Consider higher cluster counts** - performance trend is increasing")
            elif perf_analysis.get('accuracy_trend') == 'decreasing':
                recommendations.append(f"- **Consider lower cluster counts** - performance trend is decreasing")
        
        if 'weight' in experiment_type:
            # Weight-specific recommendations
            analysis = results.get('analysis', {})
            if 'weight_ablation' in results:
                analysis = results['weight_ablation'].get('analysis', {})
            
            optimal_configs = analysis.get('optimal_configurations', {})
            if optimal_configs.get('best_cost_efficiency'):
                best_config = optimal_configs['best_cost_efficiency']
                perf_weight = best_config.get('performance_weight', 0.7)
                cost_sens = best_config.get('cost_sensitivity', 0.3)
                recommendations.extend([
                    f"- **Recommended weight balance:** performance_weight={perf_weight:.2f}, cost_sensitivity={cost_sens:.2f}",
                    f"- **Cost efficiency optimization** achieved with balanced approach"
                ])
        
        # General recommendations
        recommendations.extend([
            f"- **Validate results** on different datasets before production deployment",
            f"- **Monitor performance** when changing configurations in production",
            f"- **Consider ensemble approaches** combining multiple optimal configurations"
        ])
        
        return recommendations


def main():
    """Command-line interface for ablation studies."""
    parser = argparse.ArgumentParser(
        description='Run ablation studies for Balance Cluster Router - DEPRECATED, use specific scripts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DEPRECATED: This unified runner is being phased out. Please use specific scripts:

  # Run cluster ablation only
  python ablation/run_cluster_ablation.py --data data/dataset.json

  # Run weight ablation only  
  python ablation/run_weight_ablation.py --data data/dataset.json

  # Run both experiments independently
  python ablation/run_cluster_ablation.py --data data/dataset.json
  python ablation/run_weight_ablation.py --data data/dataset.json

Legacy usage (will be removed in future versions):
  python ablation_runner.py --type cluster --data data/dataset.json
        """
    )
    
    parser.add_argument('--type', choices=['cluster', 'weight', 'comprehensive'], 
                       default='comprehensive', help='Type of ablation study to run')
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to dataset JSON file')
    parser.add_argument('--config', type=str, 
                       help='Path to configuration file (optional)')
    parser.add_argument('--output', type=str, default='ablation',
                       help='Output directory for results')
    
    # Cluster ablation parameters
    parser.add_argument('--cluster-range', nargs=3, type=int, metavar=('MIN', 'MAX', 'STEP'),
                       help='Cluster range as min max step (e.g., 8 80 8)')
    
    # Weight ablation parameters  
    parser.add_argument('--weight-points', type=int, default=11,
                       help='Number of weight combination points to test')
    parser.add_argument('--weight-range', nargs=2, type=float, metavar=('MIN', 'MAX'),
                       help='Performance weight range (e.g., 0.0 1.0)')
    
    # General parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--clusters', type=int, default=32, 
                       help='Base number of clusters (for weight ablation)')
    parser.add_argument('--performance-weight', type=float, default=0.7,
                       help='Base performance weight (for cluster ablation)')
    parser.add_argument('--cost-sensitivity', type=float, default=0.3,
                       help='Base cost sensitivity (for cluster ablation)')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate markdown report after experiments')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Create base configuration
        if args.config:
            config = SimpleClusterConfig.from_file(args.config, args.data)
            base_config = config.to_dict()
        else:
            # Create config from command line arguments
            base_config = {
                'data_path': args.data,
                'seed': args.seed,
                'n_clusters': args.clusters,
                'performance_weight': args.performance_weight,
                'cost_sensitivity': args.cost_sensitivity,
                'train_ratio': 0.7,
                'max_router': 1,
                'top_k': 1,
                'beta': 9.0,
                'max_workers': 4,
                'cluster_batch_size': 1000,
                'max_tokens': 30000,
                'embedding_model': "Qwen3-Embedding-8B",
                'min_accuracy_threshold': 0.0,
                'budget_limit': None,
                'excluded_models': [],
                'excluded_datasets': []
            }
        
        # Initialize runner
        runner = AblationRunner(output_dir=args.output)
        
        # Run requested ablation study
        if args.type == 'cluster':
            logger.info("Running cluster ablation study")
            cluster_range = tuple(args.cluster_range) if args.cluster_range else None
            results = runner.run_cluster_ablation(base_config, cluster_range)
            
        elif args.type == 'weight':
            logger.info("Running weight ablation study")
            weight_range = tuple(args.weight_range) if args.weight_range else None
            results = runner.run_weight_ablation(base_config, args.weight_points, weight_range)
            
        elif args.type == 'comprehensive':
            logger.info("Running comprehensive ablation study")
            cluster_range = tuple(args.cluster_range) if args.cluster_range else None
            weight_range = tuple(args.weight_range) if args.weight_range else None
            results = runner.run_comprehensive_study(
                base_config, cluster_range, args.weight_points, weight_range)
        
        # Generate report if requested
        if args.generate_report:
            logger.info("Generating markdown report")
            report_path = runner.generate_report(results)
            print(f"\n📄 Report generated: {report_path}")
        
        # Print summary
        experiment_type = results.get('experiment_type', args.type)
        successful_experiments = results.get('successful_experiments', 0)
        total_experiments = results.get('total_experiments', 0)
        
        print(f"\n✅ {experiment_type.replace('_', ' ').title()} study completed successfully!")
        print(f"📊 Experiments: {successful_experiments}/{total_experiments} successful")
        print(f"📁 Results saved to: {args.output}/")
        
        if results.get('figure_paths'):
            print(f"📈 Visualizations saved to: {args.output}/figures/")
        
    except Exception as e:
        logger.error(f"Ablation study failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
