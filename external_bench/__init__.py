"""
External Benchmark Tool - 极简版
只解决导入路径问题，直接使用主框架的类
"""

# Core utilities
from .utils import load_cache_config, add_project_path, save_result, finish_benchmark, start_timer

# Re-export main framework components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from generators.generator import DirectGenerator, MultimodalGenerator, GeneratorOutput
from data_collector.storage import RecordResult, BenchmarkResult


def setup(verbose: bool = False):
    """
    Usage:
        from external_bench import setup; setup()
    """
    try:
        project_root = add_project_path()
        if verbose:
            print(f"Added project path: {project_root}")
        return {"status": "success", "project_root": project_root}
    except Exception as e:
        if verbose:
            print(f"Setup warning: {e}")
        return {"status": "warning", "error": str(e)}


# Export everything needed
__all__ = [
    'DirectGenerator',
    'MultimodalGenerator',
    'GeneratorOutput',
    'RecordResult',
    'BenchmarkResult',
    'setup',
    'load_cache_config',
    'save_result',
    'finish_benchmark',
    'start_timer'
]