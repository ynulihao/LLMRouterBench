"""
Experiment Results Exporter

Exports experiment results to markdown format for better documentation and sharing.
Supports various export formats and customizable templates.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import statistics
from collections import defaultdict


class ExperimentExporter:
    """
    Exports experiment results to structured markdown format.
    
    Features:
    - Automatic markdown generation from results
    - Cost/performance analysis tables  
    - Model comparison charts
    - Dataset-specific breakdowns
    - Configurable output formatting
    """
    
    def __init__(self, output_dir: str = "experiment_reports"):
        """
        Initialize the experiment exporter.
        
        Args:
            output_dir: Directory to save markdown reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_balance_results(self, results: Dict[str, Any], config: Dict[str, Any], 
                              filename: Optional[str] = None) -> str:
        """
        Export balance cluster router results to markdown format.
        
        Args:
            results: Results dictionary from balance router
            config: Configuration dictionary
            filename: Optional custom filename (auto-generated if None)
            
        Returns:
            Path to the generated markdown file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"balance_experiment_{timestamp}.md"
        
        filepath = self.output_dir / filename
        
        # Generate markdown content
        markdown_content = self._generate_balance_markdown(results, config)
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
            
        return str(filepath)
    
    def _generate_balance_markdown(self, results: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate complete markdown content for balance results."""
        
        # Extract key metrics
        main_results = results.get('results', {})
        cost_analysis = main_results.get('cost_analysis', {})
        dataset_performance = main_results.get('dataset_performance', {})
        model_stats = main_results.get('model_selection_stats', {})
        
        # Use baseline analysis from results if available
        baseline_analysis = main_results.get('baseline_analysis', {})
        
        accuracy = main_results.get('accuracy', 0.0)
        total_queries = main_results.get('total_queries', 0)
        correct_routes = main_results.get('correct_routes', 0)
        
        # Start building markdown
        md_lines = []
        
        # Header
        md_lines.extend([
            f"# Balance Cluster Router Experiment Report",
            f"",
            f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Configuration:** Balance Mode with Cost-Performance Optimization",
            f"",
            f"---",
            f""
        ])
        
        # Key Results Summary - put most important info first
        md_lines.extend([
            f"## 🎯 Key Results",
            f"",
            f"**Router Performance:** {accuracy:.4%} ({correct_routes}/{total_queries})"
        ])
        
        # Add baseline comparison prominently
        if baseline_analysis:
            best_baseline = baseline_analysis.get('best_overall_baseline')
            if best_baseline:
                baseline_acc = best_baseline.get('avg_score', 0.0)
                baseline_model = best_baseline.get('model', 'Unknown').split('/')[-1]  # Remove prefix
                improvement = (accuracy - baseline_acc) / baseline_acc
                improvement_str = f"{improvement:+.2%}" if improvement != 0 else "0.0%"
                
                md_lines.extend([
                    f"",
                    f"**Best Baseline:** {baseline_model} ({baseline_acc:.4%})",
                    f"**Improvement:** {improvement_str}"
                ])
        
        # Balance Configuration
        md_lines.extend([
            f"",
            f"**Balance Settings:**",
            f"- Performance Weight: {config.get('performance_weight', 0.7):.4%}",
            f"- Cost Sensitivity: {config.get('cost_sensitivity', 0.3):.4%}"
        ])
        
        # Add cost metrics if available
        if cost_analysis:
            total_cost = cost_analysis.get('total_cost', 0.0)
            avg_cost = cost_analysis.get('avg_cost_per_query', 0.0)
            cost_efficiency = cost_analysis.get('cost_efficiency', 0.0)
            
            md_lines.extend([
                f"| **Total Cost** | ${total_cost:.4f} |",
                f"| **Avg Cost per Query** | ${avg_cost:.4f} |",
                f"| **Cost Efficiency** | {cost_efficiency:.4f} |"
            ])
        
        md_lines.extend([f"", f"---", f""])
        
        # Configuration Details
        md_lines.extend([
            f"## ⚙️ Configuration",
            f"",
            f"### Balance Parameters",
            f"- **Performance Weight:** {config.get('performance_weight', 0.7):.4%} (emphasis on accuracy)",
            f"- **Cost Sensitivity:** {config.get('cost_sensitivity', 0.3):.4%} (emphasis on cost optimization)", 
            f"- **Minimum Accuracy Threshold:** {config.get('min_accuracy_threshold', 0.0):.4%}",
            f"",
            f"### Clustering Configuration", 
            f"- **Number of Clusters:** {config.get('n_clusters', 32)}",
            f"- **Training Ratio:** {config.get('train_ratio', 0.7):.4%}",
            f"- **Max Routers per Query:** {config.get('max_router', 1)}",
            f"- **Random Seed:** {config.get('seed', 42)}",
            f""
        ])
        
        # Excluded models/datasets if any
        excluded_models = config.get('excluded_models', [])
        excluded_datasets = config.get('excluded_datasets', [])
        
        if excluded_models or excluded_datasets:
            md_lines.extend([f"### Exclusions"])
            if excluded_models:
                md_lines.append(f"- **Excluded Models:** {', '.join(excluded_models)}")
            if excluded_datasets:
                md_lines.append(f"- **Excluded Datasets:** {', '.join(excluded_datasets)}")
            md_lines.append(f"")
        
        md_lines.extend([f"---", f""])
        
        # Baseline Analysis Section
        if baseline_analysis:
            md_lines.extend([
                f"## 🎯 Baseline Performance Comparison",
                f"",
                self._generate_baseline_comparison_section(baseline_analysis, dataset_performance),
                f"---",
                f""
            ])
        
        # Cost Analysis Section  
        if cost_analysis:
            md_lines.extend([
                f"## 💰 Cost Analysis",
                f"",
                self._generate_cost_analysis_section(cost_analysis),
                f"---",
                f""
            ])
        
        # Dataset Performance
        if dataset_performance:
            md_lines.extend([
                f"## 📈 Dataset Performance",
                f"",
                self._generate_dataset_performance_section(dataset_performance),
                f"---", 
                f""
            ])
        
        # Model Selection Statistics
        if model_stats:
            md_lines.extend([
                f"## 🤖 Model Selection Statistics",
                f"",
                self._generate_model_stats_section(model_stats),
                f"---",
                f""
            ])
        
        # Technical Details
        md_lines.extend([
            f"## 🔧 Technical Details",
            f"",
            f"### Algorithm",
            f"The Balance Cluster Router uses a multi-objective optimization approach that:",
            f"",
            f"1. **Clusters queries** using K-means on embedding vectors",
            f"2. **Analyzes cost-efficiency** for each model in each cluster", 
            f"3. **Computes balance scores** combining accuracy and cost considerations",
            f"4. **Routes queries** to optimal models based on configurable trade-offs",
            f"",
            f"### Balance Scoring Formula",
            f"```",
            f"balance_score = performance_weight × accuracy + cost_sensitivity × (1 - normalized_cost)",
            f"```",
            f"",
            f"### Data Processing",
            f"- **Embedding Model:** text-embedding-3-large",
            f"- **Normalization:** L2 normalization of embedding vectors",  
            f"- **Cost Handling:** Zero-cost assignment for missing cost data",
            f"",
            f"---",
            f""
        ])
        
        # Recommendations
        md_lines.extend([
            f"## 🎯 Recommendations & Insights",
            f"",
            self._generate_recommendations(results, config),
            f"",
            f"---",
            f""
        ])
        
        # Footer
        md_lines.extend([
            f"## 📝 Notes",
            f"",
            f"- This report was automatically generated by the Balance Cluster Router",
            f"- Cost data coverage and quality may affect routing decisions",
            f"- Performance may vary with different parameter configurations",
            f"",
            f"**Experiment completed at:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        ])
        
        return "\n".join(md_lines)
    
    def _generate_cost_analysis_section(self, cost_analysis: Dict[str, Any]) -> str:
        """Generate cost analysis section."""
        lines = []
        
        total_cost = cost_analysis.get('total_cost', 0.0)
        avg_cost = cost_analysis.get('avg_cost_per_query', 0.0) 
        cost_per_correct = cost_analysis.get('cost_per_correct_prediction', 0.0)
        cost_efficiency = cost_analysis.get('cost_efficiency', 0.0)
        
        lines.extend([
            f"### Overall Cost Metrics",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|", 
            f"| Total Cost | ${total_cost:.4f} |",
            f"| Average Cost per Query | ${avg_cost:.4f} |",
            f"| Cost per Correct Prediction | ${cost_per_correct:.4f} |", 
            f"| Cost Efficiency (Accuracy/Cost) | {cost_efficiency:.4f} |",
            f""
        ])
        
        # Model cost breakdown
        model_costs = cost_analysis.get('model_costs', {})
        if model_costs:
            lines.extend([
                f"### Model Cost Distribution",
                f"",
                f"| Model | Total Cost | Percentage |",
                f"|-------|------------|------------|"
            ])
            
            sorted_models = sorted(model_costs.items(), key=lambda x: x[1], reverse=True)
            for model, cost in sorted_models[:10]:  # Top 10 models by cost
                percentage = (cost / total_cost * 100) if total_cost > 0 else 0
                lines.append(f"| {model} | ${cost:.4f} | {percentage:.4f}% |")
            
            lines.append(f"")
        
        # Dataset cost breakdown
        dataset_costs = cost_analysis.get('dataset_costs', {})
        if dataset_costs:
            lines.extend([
                f"### Dataset Cost Distribution",
                f"",
                f"| Dataset | Total Cost | Avg Cost per Query |",
                f"|---------|------------|-------------------|"
            ])
            
            for dataset, cost in sorted(dataset_costs.items()):
                # This would need query counts per dataset to calculate avg
                lines.append(f"| {dataset} | ${cost:.4f} | - |")
            
            lines.append(f"")
        
        return "\n".join(lines)
    
    def _generate_baseline_comparison_section(self, baseline_analysis: Dict[str, Any], 
                                            dataset_performance: Dict[str, Any]) -> str:
        """Generate baseline comparison section using baseline_analysis data."""
        lines = []
        
        model_summaries = baseline_analysis.get('model_summaries', [])
        dataset_comparisons = baseline_analysis.get('dataset_comparisons', [])
        best_baseline = baseline_analysis.get('best_overall_baseline')
        
        if model_summaries:
            # Generate baseline model performance summary in markdown table format
            lines.extend([
                f"### Baseline Model Performance Summary",
                f"",
                f"| Model | Average Score | Total Cost | Dataset Coverage |",
                f"|-------|---------------|------------|------------------|"
            ])
            
            for model_info in model_summaries:
                model_name = model_info.get('model', 'Unknown').split('/')[-1]  # Remove prefix
                avg_score = model_info.get('avg_score', 0.0)
                dataset_coverage = model_info.get('dataset_coverage', '0/0')
                total_cost = model_info.get('total_cost', 0.0)
                
                # Format with proper cost display
                cost_str = f"${total_cost:.4f}" if total_cost > 0 else "N/A"
                lines.append(f"| {model_name} | {avg_score:.4f} | {cost_str} | {dataset_coverage} |")
            
            lines.append(f"")
            
            if best_baseline:
                best_model_name = best_baseline.get('model', 'Unknown').split('/')[-1]
                lines.extend([
                    f"**Best Overall Baseline:** {best_model_name} (avg: {best_baseline['avg_score']:.4f})",
                    f""
                ])
        
        # Generate per-dataset comparison using dataset_comparisons data
        if dataset_comparisons:
            lines.extend([
                f"### Per-Dataset Performance",
                f"",
                f"| Dataset | Router | Baseline | Best Model | Improvement |",
                f"|---------|--------|----------|------------|-------------|"
            ])
            
            for comparison in dataset_comparisons:
                dataset = comparison.get('dataset', 'Unknown')
                router_accuracy = comparison.get('router_accuracy', 0.0)
                best_baseline_score = comparison.get('best_baseline_score', 0.0)
                best_model_name = comparison.get('best_baseline_model', 'Unknown').split('/')[-1]
                improvement = comparison.get('improvement', 0.0)
                
                improvement_str = f"{improvement:+.4f}" if improvement != 0 else "0.000"
                lines.append(f"| {dataset} | {router_accuracy:.4f} | {best_baseline_score:.4f} | {best_model_name} | {improvement_str} |")
            
            lines.append(f"")
        
        return "\n".join(lines) if lines else "No baseline comparison data available."
    
    def _generate_dataset_performance_section(self, dataset_performance: Dict[str, Any]) -> str:
        """Generate dataset performance breakdown."""
        lines = []
        
        if not dataset_performance:
            return "No dataset performance data available."
        
        lines.extend([
            f"| Dataset | Accuracy | Correct | Total |",
            f"|---------|----------|---------|-------|"
        ])
        
        # Sort by accuracy descending (calculate accuracy from correct/total)
        sorted_datasets = sorted(dataset_performance.items(), 
                               key=lambda x: (x[1].get('correct', 0) / max(x[1].get('total', 1), 1)) if isinstance(x[1], dict) else 0, 
                               reverse=True)
        
        for dataset, stats in sorted_datasets:
            if isinstance(stats, dict):
                correct = stats.get('correct', 0)
                total = stats.get('total', 0)
                accuracy = correct / max(total, 1)  # Calculate accuracy from correct/total
                lines.append(f"| {dataset} | {accuracy:.4%} | {correct} | {total} |")
        
        return "\n".join(lines)
    
    def _generate_model_stats_section(self, model_stats: Dict[str, Any]) -> str:
        """Generate model selection statistics."""
        lines = []
        
        if not model_stats:
            return "No model selection statistics available."
        
        lines.extend([
            f"| Model | Selection Count | Percentage |",
            f"|-------|-----------------|------------|"
        ])
        
        total_selections = sum(model_stats.values())
        sorted_models = sorted(model_stats.items(), key=lambda x: x[1], reverse=True)
        
        for model, count in sorted_models:
            percentage = (count / total_selections * 100) if total_selections > 0 else 0
            lines.append(f"| {model} | {count} | {percentage:.4f}% |")
        
        return "\n".join(lines)
    
    def _generate_recommendations(self, results: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate recommendations based on results."""
        lines = []
        
        main_results = results.get('results', {})
        accuracy = main_results.get('accuracy', 0.0)
        cost_analysis = main_results.get('cost_analysis', {})
        
        # Performance assessment
        if accuracy < 0.3:
            lines.append(f"⚠️  **Low Accuracy Alert**: Current accuracy ({accuracy:.4%}) is below 30%. Consider:")
        elif accuracy < 0.5:
            lines.append(f"📊 **Moderate Performance**: Accuracy is {accuracy:.4%}. Potential improvements:")
        else:
            lines.append(f"✅ **Good Performance**: Accuracy of {accuracy:.4%} is solid. Fine-tuning suggestions:")
        
        lines.append(f"")
        
        # Parameter tuning suggestions
        perf_weight = config.get('performance_weight', 0.7)
        cost_sens = config.get('cost_sensitivity', 0.3)
        
        if accuracy < 0.4 and perf_weight < 0.8:
            lines.append(f"- **Increase Performance Weight** to 0.8+ (currently {perf_weight:.4f}) to prioritize accuracy")
        
        if cost_analysis.get('avg_cost_per_query', 0) > 0.1:
            lines.append(f"- **Increase Cost Sensitivity** to 0.4+ (currently {cost_sens:.4f}) to reduce query costs")
        
        # Clustering recommendations  
        n_clusters = config.get('n_clusters', 32)
        if accuracy < 0.4:
            lines.append(f"- **Experiment with cluster count**: Try {n_clusters//2} or {n_clusters*2} clusters")
        
        # Model recommendations
        model_stats = main_results.get('model_selection_stats', {})
        if model_stats:
            top_model = max(model_stats.items(), key=lambda x: x[1])
            lines.append(f"- **Top performing model**: {top_model[0]} (selected {top_model[1]} times)")
        
        lines.append(f"")
        lines.append(f"💡 **Next Steps**:")
        lines.append(f"- Run parameter sensitivity analysis")  
        lines.append(f"- Test with different model exclusion sets")
        lines.append(f"- Analyze cluster quality and query distribution")
        
        return "\n".join(lines)


def main():
    """Example usage of the experiment exporter."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export experiment results to markdown')
    parser.add_argument('--results', type=str, required=True, help='Results JSON file')
    parser.add_argument('--output', type=str, help='Output markdown filename')
    parser.add_argument('--output_dir', type=str, default='experiment_reports', 
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Load results
    with open(args.results, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract results and config
    results = data
    config = data.get('config', {})
    
    # Export to markdown
    exporter = ExperimentExporter(args.output_dir)
    output_path = exporter.export_balance_results(results, config, args.output)
    
    print(f"Experiment report exported to: {output_path}")


if __name__ == "__main__":
    main()