"""
Academic Visualization for Ablation Studies

Provides high-quality, publication-ready visualizations for ablation experiments.
Follows academic standards with proper typography, colors, and statistical representation.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import json
import logging
from collections import defaultdict

# Configure matplotlib for high-DPI output
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

# Academic color palette (colorblind-friendly)
ACADEMIC_COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Pink/Purple  
    'tertiary': '#F18F01',     # Orange
    'quaternary': '#C73E1D',   # Red
    'success': '#4CAF50',      # Green
    'warning': '#FF9800',      # Amber
    'baseline': '#6C757D',     # Gray
    'grid': '#E0E0E0',         # Light gray
    'background': '#FAFAFA'    # Very light gray
}

BASELINE_MARKERS = ['o', 's', '^', 'v', 'D', 'P', 'X', '*']


class AblationVisualizer:
    """
    Creates academic-quality visualizations for ablation experiments.
    
    Features:
    - High-DPI output (300+ DPI)
    - Colorblind-friendly palette
    - Statistical error representation
    - Professional typography
    - Publication-ready formatting
    """
    
    def __init__(self, output_dir: str = "ablation/figures", style: str = "academic"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save figures
            style: Visualization style ('academic', 'presentation')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup matplotlib style
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib style for academic publications."""
        if self.style == "academic":
            # Academic paper style
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams.update({
                'figure.figsize': (8, 6),
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.grid': True,
                'grid.alpha': 0.3,
                'grid.linestyle': '-',
                'grid.linewidth': 0.5,
                'axes.axisbelow': True
            })
        elif self.style == "presentation":
            # Presentation style with larger fonts
            plt.rcParams.update({
                'figure.figsize': (10, 7),
                'font.size': 14,
                'axes.titlesize': 16,
                'axes.labelsize': 14,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12
            })
    
    def plot_cluster_ablation(self, results: List[Dict[str, Any]], 
                            baseline_data: Optional[Dict[str, Any]] = None,
                            save_path: Optional[str] = None) -> str:
        """
        Plot n_clusters ablation results.
        
        Args:
            results: List of experimental results with cluster configurations
            baseline_data: Optional baseline model performance data
            save_path: Optional custom save path
            
        Returns:
            Path to saved figure
        """
        # Extract data
        cluster_counts = []
        accuracies = []
        costs = []
        cost_efficiencies = []
        
        for result in results:
            if result.get('failed', False):
                continue
                
            cluster_counts.append(result.get('parameter_value', result['config'].get('n_clusters')))
            accuracies.append(result.get('accuracy', 0.0))
            
            cost_analysis = result.get('cost_analysis', {})
            costs.append(cost_analysis.get('avg_cost_per_query', 0.0))
            cost_efficiencies.append(cost_analysis.get('cost_efficiency', 0.0))
        
        if not cluster_counts:
            raise ValueError("No valid results found for cluster ablation")
        
        # Sort by cluster count
        sorted_data = sorted(zip(cluster_counts, accuracies, costs, cost_efficiencies))
        cluster_counts, accuracies, costs, cost_efficiencies = zip(*sorted_data)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Performance vs Number of Clusters
        ax1.plot(cluster_counts, accuracies, 'o-', color=ACADEMIC_COLORS['primary'], 
                linewidth=2, markersize=8, label='Balance Router')
        
        # Add baseline if available
        if baseline_data:
            best_baseline = baseline_data.get('best_overall_baseline')
            if best_baseline:
                baseline_acc = best_baseline.get('avg_score', 0.0)
                ax1.axhline(y=baseline_acc, color=ACADEMIC_COLORS['baseline'], 
                           linestyle='--', linewidth=2, label=f'Best Baseline ({best_baseline.get("model", "").split("/")[-1]})')
        
        ax1.set_xlabel('Number of Clusters', fontweight='bold')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Performance vs Number of Clusters', fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        
        # Format y-axis as percentage
        ax1.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        
        # Plot 2: Cost Efficiency vs Number of Clusters
        ax2.plot(cluster_counts, cost_efficiencies, 'o-', color=ACADEMIC_COLORS['secondary'],
                linewidth=2, markersize=8, label='Cost Efficiency')
        
        ax2.set_xlabel('Number of Clusters', fontweight='bold')
        ax2.set_ylabel('Cost Efficiency (Accuracy/Cost)', fontweight='bold')
        ax2.set_title('Cost Efficiency vs Number of Clusters', fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=True, fancybox=True, shadow=True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / "cluster_ablation.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Cluster ablation plot saved to {save_path}")
        return str(save_path)
    
    def plot_weight_ablation(self, results: List[Dict[str, Any]],
                           baseline_data: Optional[Dict[str, Any]] = None,
                           save_path: Optional[str] = None) -> str:
        """
        Plot cost/performance weight ablation results.
        
        Args:
            results: List of experimental results with weight configurations
            baseline_data: Optional baseline model performance data  
            save_path: Optional custom save path
            
        Returns:
            Path to saved figure
        """
        # Extract data
        performance_weights = []
        cost_sensitivities = []
        accuracies = []
        total_costs = []
        
        for result in results:
            if result.get('failed', False):
                continue
                
            config = result.get('config', {})
            param_combo = result.get('parameter_combination', {})
            
            perf_weight = param_combo.get('performance_weight', config.get('performance_weight'))
            cost_sens = param_combo.get('cost_sensitivity', config.get('cost_sensitivity'))
            
            if perf_weight is not None and cost_sens is not None:
                performance_weights.append(perf_weight)
                cost_sensitivities.append(cost_sens)
                accuracies.append(result.get('non_ood_accuracy', 0.0))

                cost_analysis = result.get('cost_analysis', {})
                total_costs.append(cost_analysis.get('non_ood_total_cost', 0.0))
        
        if not performance_weights:
            raise ValueError("No valid results found for weight ablation")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot colored by performance weight
        scatter = ax.scatter(total_costs, accuracies, c=performance_weights, 
                           s=120, alpha=0.8, cmap='viridis', edgecolors='black', linewidth=1)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Performance Weight', fontweight='bold')
        
        # Connect points to show Pareto frontier
        sorted_data = sorted(zip(total_costs, accuracies, performance_weights))
        sorted_costs, sorted_accs, _ = zip(*sorted_data)
        ax.plot(sorted_costs, sorted_accs, '-', color=ACADEMIC_COLORS['primary'], 
               alpha=0.6, linewidth=2, label='Weight Trade-off Curve')
        
        # Add baseline models if available
        if baseline_data:
            model_summaries = baseline_data.get('model_summaries', [])
            
            # Calculate average cost for baseline placement if no cost data available
            avg_cost = np.mean(total_costs) if total_costs else 0.0
            cost_range = max(total_costs) - min(total_costs) if total_costs else 1.0
            
            for i, model_info in enumerate(model_summaries[:6]):  # Show top 6 models
                model_name = model_info.get('model', 'Unknown').split('/')[-1]
                baseline_acc = model_info.get('avg_score', 0.0)
                baseline_cost = model_info.get('total_cost', 0.0)
                
                # If no cost data, distribute baseline models along x-axis for visibility
                if baseline_cost <= 0:
                    baseline_cost = avg_cost + (i - 2.5) * cost_range * 0.1
                
                marker = BASELINE_MARKERS[i % len(BASELINE_MARKERS)]
                ax.scatter(baseline_cost, baseline_acc, marker=marker, 
                         s=200, color=ACADEMIC_COLORS['baseline'], 
                         edgecolors='black', linewidth=2, alpha=0.9,
                         label=f'Baseline: {model_name}', zorder=5)
        
        ax.set_xlabel('Non-OOD Total Cost ($)', fontweight='bold')
        ax.set_ylabel('Non-OOD Accuracy', fontweight='bold')
        ax.set_title('Non-OOD Performance-Cost Trade-off: Weight Ablation Study', fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / "weight_ablation.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Weight ablation plot saved to {save_path}")
        return str(save_path)
    
    def plot_pareto_frontier(self, results: List[Dict[str, Any]],
                           baseline_data: Optional[Dict[str, Any]] = None,
                           save_path: Optional[str] = None) -> str:
        """
        Plot Pareto frontier of performance vs cost.
        
        Args:
            results: List of experimental results
            baseline_data: Optional baseline model performance data
            save_path: Optional custom save path
            
        Returns:
            Path to saved figure
        """
        # Extract performance and cost data
        points = []
        for result in results:
            if result.get('failed', False):
                continue
                
            accuracy = result.get('accuracy', 0.0)
            cost_analysis = result.get('cost_analysis', {})
            avg_cost = cost_analysis.get('avg_cost_per_query', 0.0)
            
            if avg_cost > 0:  # Only include points with valid cost data
                points.append((avg_cost, accuracy))
        
        if not points:
            raise ValueError("No valid cost-performance points found")
        
        # Find Pareto frontier
        pareto_points = self._find_pareto_frontier(points, maximize_y=True, minimize_x=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot all points
        costs, accs = zip(*points)
        ax.scatter(costs, accs, alpha=0.6, s=80, color=ACADEMIC_COLORS['tertiary'], 
                  label='Experiment Points', zorder=3)
        
        # Plot Pareto frontier
        if len(pareto_points) > 1:
            pareto_costs, pareto_accs = zip(*sorted(pareto_points))
            ax.plot(pareto_costs, pareto_accs, 'o-', color=ACADEMIC_COLORS['primary'], 
                   linewidth=3, markersize=10, label='Pareto Frontier', zorder=4)
        
        # Add baseline models
        if baseline_data:
            model_summaries = baseline_data.get('model_summaries', [])
            
            # Calculate average cost for baseline placement if no cost data available
            avg_cost = np.mean(costs) if costs else 0.0
            cost_range = max(costs) - min(costs) if costs else 1.0
            
            for i, model_info in enumerate(model_summaries[:6]):  # Show top 6 models
                model_name = model_info.get('model', 'Unknown').split('/')[-1]
                baseline_acc = model_info.get('avg_score', 0.0)
                baseline_cost = model_info.get('total_cost', 0.0)
                
                if baseline_cost > 0:
                    # Approximate per-query cost (divide by typical query count)
                    baseline_cost_per_query = baseline_cost / 1000  # Rough estimate
                else:
                    # If no cost data, distribute baseline models along x-axis
                    baseline_cost_per_query = avg_cost + (i - 2.5) * cost_range * 0.1
                
                marker = BASELINE_MARKERS[i % len(BASELINE_MARKERS)]
                ax.scatter(baseline_cost_per_query, baseline_acc, marker=marker,
                         s=200, color=ACADEMIC_COLORS['baseline'],
                         edgecolors='black', linewidth=2, alpha=0.9,
                         label=f'Baseline: {model_name}', zorder=5)
        
        ax.set_xlabel('Average Cost per Query ($)', fontweight='bold')
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title('Pareto Frontier: Performance vs Cost Efficiency', fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / "pareto_frontier.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Pareto frontier plot saved to {save_path}")
        return str(save_path)
    
    def _find_pareto_frontier(self, points: List[Tuple[float, float]], 
                            maximize_y: bool = True, minimize_x: bool = True) -> List[Tuple[float, float]]:
        """
        Find Pareto frontier points.
        
        Args:
            points: List of (x, y) coordinate tuples
            maximize_y: Whether to maximize y values
            minimize_x: Whether to minimize x values
            
        Returns:
            List of Pareto optimal points
        """
        pareto_points = []
        
        for i, (x1, y1) in enumerate(points):
            is_pareto = True
            
            for j, (x2, y2) in enumerate(points):
                if i != j:
                    # Check if point j dominates point i
                    x_better = (x2 <= x1) if minimize_x else (x2 >= x1)
                    y_better = (y2 >= y1) if maximize_y else (y2 <= y1)
                    
                    x_strictly_better = (x2 < x1) if minimize_x else (x2 > x1)
                    y_strictly_better = (y2 > y1) if maximize_y else (y2 < y1)
                    
                    if (x_better and y_better) and (x_strictly_better or y_strictly_better):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_points.append((x1, y1))
        
        return pareto_points
    
    def create_summary_plot(self, cluster_results: List[Dict[str, Any]], 
                          weight_results: List[Dict[str, Any]],
                          baseline_data: Optional[Dict[str, Any]] = None,
                          save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive summary plot with both ablation studies.
        
        Args:
            cluster_results: Results from cluster ablation
            weight_results: Results from weight ablation  
            baseline_data: Optional baseline model data
            save_path: Optional custom save path
            
        Returns:
            Path to saved figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Cluster ablation - Performance
        cluster_counts = [r.get('parameter_value', r['config'].get('n_clusters')) 
                         for r in cluster_results if not r.get('failed', False)]
        accuracies = [r.get('accuracy', 0.0) for r in cluster_results if not r.get('failed', False)]
        
        if cluster_counts and accuracies:
            sorted_cluster_data = sorted(zip(cluster_counts, accuracies))
            cluster_counts, accuracies = zip(*sorted_cluster_data)
            
            ax1.plot(cluster_counts, accuracies, 'o-', color=ACADEMIC_COLORS['primary'], 
                    linewidth=2, markersize=6)
            ax1.set_xlabel('Number of Clusters')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('N-Clusters Ablation', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        
        # Plot 2: Weight ablation - Performance vs Cost
        weight_accuracies = []
        weight_costs = []
        performance_weights = []
        
        for result in weight_results:
            if result.get('failed', False):
                continue
            config = result.get('config', {})
            param_combo = result.get('parameter_combination', {})
            perf_weight = param_combo.get('performance_weight', config.get('performance_weight'))
            
            if perf_weight is not None:
                weight_accuracies.append(result.get('accuracy', 0.0))
                cost_analysis = result.get('cost_analysis', {})
                weight_costs.append(cost_analysis.get('avg_cost_per_query', 0.0))
                performance_weights.append(perf_weight)
        
        if weight_accuracies and weight_costs:
            scatter = ax2.scatter(weight_costs, weight_accuracies, c=performance_weights, 
                                cmap='viridis', s=60, alpha=0.8)
            ax2.set_xlabel('Cost per Query ($)')
            ax2.set_ylabel('Accuracy')  
            ax2.set_title('Weight Ablation', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
            
            # Add small colorbar
            cbar = fig.colorbar(scatter, ax=ax2, shrink=0.8)
            cbar.set_label('Perf. Weight', fontsize=8)
        
        # Plot 3: Cost efficiency trends
        if cluster_counts:
            cost_effs = []
            for r in cluster_results:
                if not r.get('failed', False):
                    cost_analysis = r.get('cost_analysis', {})
                    cost_effs.append(cost_analysis.get('cost_efficiency', 0.0))
            
            if cost_effs:
                ax3.plot(cluster_counts, cost_effs, 'o-', color=ACADEMIC_COLORS['secondary'],
                        linewidth=2, markersize=6)
                ax3.set_xlabel('Number of Clusters')
                ax3.set_ylabel('Cost Efficiency')
                ax3.set_title('Cost Efficiency vs Clusters', fontweight='bold')
                ax3.grid(True, alpha=0.3)
        
        # Plot 4: Baseline comparison
        if baseline_data:
            model_summaries = baseline_data.get('model_summaries', [])
            model_names = [m.get('model', 'Unknown').split('/')[-1] for m in model_summaries[:8]]
            model_accs = [m.get('avg_score', 0.0) for m in model_summaries[:8]]
            
            bars = ax4.bar(range(len(model_names)), model_accs, 
                          color=ACADEMIC_COLORS['baseline'], alpha=0.7)
            ax4.set_xlabel('Baseline Models')
            ax4.set_ylabel('Accuracy')
            ax4.set_title('Baseline Model Performance', fontweight='bold')
            ax4.set_xticks(range(len(model_names)))
            ax4.set_xticklabels(model_names, rotation=45, ha='right')
            ax4.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add best router performance line
            if cluster_results:
                best_router_acc = max([r.get('accuracy', 0.0) for r in cluster_results 
                                     if not r.get('failed', False)])
                ax4.axhline(y=best_router_acc, color=ACADEMIC_COLORS['primary'], 
                           linestyle='--', linewidth=2, label='Best Router')
                ax4.legend(fontsize=8)
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / "ablation_summary.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Summary plot saved to {save_path}")
        return str(save_path)