#!/usr/bin/env python
"""
Benchmark for value function and choice probability computation.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .benchmark_utils import BaseBenchmarkResult, TimingStats, time_operation, print_size_group_subheader
from optimization_config import OptimizationConfig
from model_configurable import ConfigurableModel


@dataclass
class ComputationBenchmarkResult(BaseBenchmarkResult):
    """Results from a computation benchmark run."""
    config: OptimizationConfig
    model_size: Tuple[int, int]  # (n_players, n_demand)
    n_states: int
    setup_stats: TimingStats
    value_function_stats: TimingStats
    choice_prob_stats: TimingStats
    total_stats: TimingStats
    vf_iterations: int
    error: Optional[str] = None

    def to_dict(self) -> dict:
        result = {
            'config': str(self.config),
            'n_players': self.model_size[0],
            'n_demand': self.model_size[1],
            'n_states': self.n_states,
            'vf_iterations': self.vf_iterations,
            'error': self.error,
            # Include individual config flags for reconstruction
            'vectorize': self.config.vectorize,
            'sparse': self.config.sparse,
            'polyalgorithm': self.config.polyalgorithm,
            'cython': self.config.cython,
            'derivatives': self.config.derivatives
        }

        # Add timing statistics with prefixes
        result.update(self.timing_stats_to_dict(self.setup_stats, "setup_"))
        result.update(self.timing_stats_to_dict(self.value_function_stats, "value_function_"))
        result.update(self.timing_stats_to_dict(self.choice_prob_stats, "choice_prob_"))
        result.update(self.timing_stats_to_dict(self.total_stats, "total_"))

        return result


def get_computation_configs() -> List[Tuple[str, OptimizationConfig]]:
    """Get configurations for computation benchmarks."""
    # Use the standard optimization sequence, but only use first 4
    configs = OptimizationConfig.sequential()
    return configs[:4]


def run_computation_benchmark(
    n_players: int,
    n_demand: int,
    config_name: str,
    config: OptimizationConfig,
    test_parameters: dict,
    n_runs: int = 1,
    reference_values: Optional[None] = None
) -> ComputationBenchmarkResult:
    """Run a single computation benchmark."""
    n_states = (2 ** n_players) * n_demand

    try:
        # Setup timing
        def setup_func():
            return ConfigurableModel(n_players, n_demand, test_parameters, rho=0.05, verbose=False, config=config)

        model, setup_stats = time_operation(setup_func, n_runs)

        # Value function computation
        def vf_func():
            v, _ = model.value_function(vf_max_iter=5000, vf_tol=1e-13)
            return v

        v, vf_stats = time_operation(vf_func, n_runs)

        # Choice probability computation
        def cp_func():
            return model.choice_probabilities(v)

        p, cp_stats = time_operation(cp_func, n_runs)

        # Calculate total time statistics
        total_times = [
            setup_stats.times[i] + vf_stats.times[i] + cp_stats.times[i]
            for i in range(n_runs)
        ]
        import numpy as np
        total_times_array = np.array(total_times)
        total_stats = TimingStats(
            times=total_times,
            mean=float(np.mean(total_times_array)),
            median=float(np.median(total_times_array)),
            std=float(np.std(total_times_array, ddof=1) if len(total_times) > 1 else 0.0),
            min=float(np.min(total_times_array)),
            max=float(np.max(total_times_array)),
            n_runs=n_runs
        )

        return ComputationBenchmarkResult(
            config=config,
            model_size=(n_players, n_demand),
            n_states=n_states,
            setup_stats=setup_stats,
            value_function_stats=vf_stats,
            choice_prob_stats=cp_stats,
            total_stats=total_stats,
            vf_iterations=0  # Not tracked
        )

    except Exception as e:
        # Create zero-filled stats for error cases
        zero_stats = TimingStats(
            times=[0.0] * n_runs,
            mean=0.0,
            median=0.0,
            std=0.0,
            min=0.0,
            max=0.0,
            n_runs=n_runs
        )

        return ComputationBenchmarkResult(
            config=config,
            model_size=(n_players, n_demand),
            n_states=n_states,
            setup_stats=zero_stats,
            value_function_stats=zero_stats,
            choice_prob_stats=zero_stats,
            total_stats=zero_stats,
            vf_iterations=0,
            error=str(e)
        )


def run_computation_benchmarks(size_groups: List[str], model_sizes: dict, test_parameters: dict, is_quick: bool = False, verbose: bool = True) -> List[ComputationBenchmarkResult]:
    """Run computation benchmarks for specified size groups."""
    if verbose:
        print("Computation Benchmarks")
        print("======================")

    # Determine number of runs
    n_runs = 1 if is_quick else 5
    if verbose:
        print(f"Running {n_runs} replications{'s' if n_runs > 1 else ''} per benchmark")

    results = []
    configs = get_computation_configs()

    for size_group in size_groups:
        if size_group not in model_sizes:
            continue

        if verbose:
            print_size_group_subheader(size_group)

        for n_players, n_demand in model_sizes[size_group]:
            n_states = (2 ** n_players) * n_demand
            if verbose:
                print(f"\nModel: {n_players}×{n_demand} ({n_states} states)")

            for config_name, config in configs:
                if verbose:
                    print(f"  {config_name}...", end=' ', flush=True)
                result = run_computation_benchmark(n_players, n_demand, config_name, config, test_parameters, n_runs)

                if result.error:
                    if verbose:
                        print(f"ERROR: {result.error}")
                else:
                    if verbose:
                        if n_runs == 1:
                            print(f"{result.total_stats.median:.3f}s")
                        else:
                            print(f"{result.total_stats.median:.3f}s (±{result.total_stats.std:.3f})")
                    results.append(result)

    return results
