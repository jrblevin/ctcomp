#!/usr/bin/env python
"""
Shared utilities for benchmark modules.
"""

import time
import psutil
import numpy as np
import platform
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod


@dataclass
class BenchmarkConfig:
    """Centralized benchmark configuration."""
    model_sizes: Dict[str, List[Tuple[int, int]]]
    test_parameters: Dict[str, float]
    output_formats: List[str] = None

    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ['json', 'latex', 'png', 'pdf']


@dataclass
class TimingStats:
    """Statistics from multiple timing runs."""
    times: List[float]
    mean: float
    median: float
    std: float
    min: float
    max: float
    n_runs: int


class BaseBenchmarkResult(ABC):
    """Base class for all benchmark results."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        pass

    def get_size_key(self) -> str:
        """Get standardized size key."""
        return f"{self.model_size[0]}x{self.model_size[1]}"

    @staticmethod
    def timing_stats_to_dict(stats: TimingStats, prefix: str = "") -> dict:
        """Convert TimingStats to dictionary with optional prefix."""
        return {
            f"{prefix}times": stats.times,
            f"{prefix}mean": stats.mean,
            f"{prefix}median": stats.median,
            f"{prefix}std": stats.std,
            f"{prefix}min": stats.min,
            f"{prefix}max": stats.max,
            f"{prefix}n_runs": stats.n_runs
        }


def get_model_sizes() -> Dict[str, List[Tuple[int, int]]]:
    """Get model sizes for different benchmark types."""
    return {
        'very_small': [(2, 2), (3, 2), (4, 2)],
        'small': [(4, 3), (5, 3), (6, 3)],
        'medium': [(6, 4), (7, 4), (7, 5)],
        'large': [(8, 4), (8, 5), (8, 6)],
        'very_large': [(9, 5), (9, 6), (10, 6)],
    }


def get_test_parameters() -> Dict[str, float]:
    """Get standard test parameters."""
    return {
        'theta_ec': -1.0,
        'theta_rn': -0.5,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 0.5
    }




def time_operation(func, n_runs: int = 1, *args, **kwargs):
    """
    Time a function call with multiple runs and return statistics.

    Parameters
    ----------
    func : callable
        Function to time
    n_runs : int
        Number of runs to perform (default: 1)
    *args, **kwargs
        Arguments passed to func

    Returns
    -------
    result : any
        Result from the last function call
    stats : TimingStats
        Timing statistics from all runs
    """
    times = []
    result = None

    for _ in range(n_runs):
        start_time = time.time()
        result = func(*args, **kwargs)
        times.append(time.time() - start_time)

    times_array = np.array(times)
    stats = TimingStats(
        times=times,
        mean=float(np.mean(times_array)),
        median=float(np.median(times_array)),
        std=float(np.std(times_array, ddof=1) if len(times) > 1 else 0.0),
        min=float(np.min(times_array)),
        max=float(np.max(times_array)),
        n_runs=n_runs
    )

    return result, stats


def print_header(title: str, char: str = "="):
    """Print formatted header."""
    print(f"\n{title}")
    print(char * len(title))


def print_size_group_subheader(size_group: str):
    """Print formatted header for a model size group."""
    title = size_group.replace('_', ' ').title()
    print_header(title, char='-')


def get_system_info() -> dict:
    """Get comprehensive system information for benchmark results."""
    try:
        # Basic system info
        info = {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'python_implementation': platform.python_implementation(),
        }

        # CPU information
        info['cpu_count_logical'] = psutil.cpu_count(logical=True)
        info['cpu_count_physical'] = psutil.cpu_count(logical=False)

        # Memory information
        memory = psutil.virtual_memory()
        info['total_memory_gb'] = round(memory.total / (1024**3), 1)
        info['available_memory_gb'] = round(memory.available / (1024**3), 1)

        # Additional details
        try:
            info['cpu_freq_max'] = round(psutil.cpu_freq().max / 1000, 2) if psutil.cpu_freq() else None
        except:
            info['cpu_freq_max'] = None

        return info
    except Exception as e:
        return {'error': f"Could not gather system info: {e}"}


def print_system_info():
    """Print formatted system information."""
    print_header("System Information")

    info = get_system_info()

    if 'error' in info:
        print(f"Warning: {info['error']}")
        return

    print(f"Platform: {info['system']} {info['release']} ({info['machine']})")
    print(f"Processor: {info['processor']}")
    print(f"CPU Cores: {info['cpu_count_physical']} physical, {info['cpu_count_logical']} logical")
    if info['cpu_freq_max']:
        print(f"CPU Max Frequency: {info['cpu_freq_max']} GHz")
    print(f"Memory: {info['total_memory_gb']} GB total, {info['available_memory_gb']} GB available")
    print(f"Python: {info['python_implementation']} {platform.python_version()}")

    # Add library versions for key dependencies
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except:
        pass

    try:
        import scipy
        print(f"SciPy: {scipy.__version__}")
    except:
        pass


def create_benchmark_config() -> BenchmarkConfig:
    """Create standard benchmark configuration."""
    return BenchmarkConfig(
        model_sizes=get_model_sizes(),
        test_parameters=get_test_parameters()
    )
