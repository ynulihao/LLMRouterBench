"""
Ablation Study Module for Balance Cluster Router

This module provides comprehensive ablation studies for analyzing the performance
of the Balance Cluster Router under different parameter configurations.

Features:
- N-cluster ablation studies  
- Cost/performance weight ablation studies
- Academic-quality visualization and reporting
- Reproducible experimental framework
"""

__version__ = "1.0.0"
__author__ = "Balance Cluster Router Team"

# Only import core components without matplotlib dependencies
from .data_collector import AblationDataCollector

# Conditional imports to handle missing matplotlib
try:
    from .cluster_ablation import ClusterAblation
    from .weight_ablation import WeightAblation
    from .ablation_runner import AblationRunner
    __all__ = [
        "ClusterAblation",
        "WeightAblation", 
        "AblationRunner",
        "AblationDataCollector"
    ]
except ImportError as e:
    # matplotlib or other visualization dependencies missing
    import warnings
    warnings.warn(f"Visualization components not available: {e}", ImportWarning)
    __all__ = ["AblationDataCollector"]

# Try to import visualizer separately
try:
    from .visualization import AblationVisualizer
    __all__.append("AblationVisualizer")
except ImportError:
    pass