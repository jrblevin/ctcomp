#!/usr/bin/env python
"""
Benchmark module for parameter estimation.
"""

import numpy as np
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .benchmark_utils import BaseBenchmarkResult, TimingStats, time_operation, print_size_group_subheader
from optimization_config import OptimizationConfig
from model_configurable import ConfigurableModel
from model import Model


@dataclass
class EstimationBenchmarkResult(BaseBenchmarkResult):
    """Results from an estimation benchmark run."""
    config: str  # Configuration name
    model_size: Tuple[int, int]
    n_states: int
    n_obs: int
    estimation_stats: TimingStats
    n_iterations: int
    n_likelihood_evals: int
    final_likelihood: float
    converged: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        result = {
            'config': self.config,
            'n_players': self.model_size[0],
            'n_demand': self.model_size[1],
            'n_states': self.n_states,
            'n_obs': self.n_obs,
            'n_iterations': self.n_iterations,
            'n_likelihood_evals': self.n_likelihood_evals,
            'final_likelihood': self.final_likelihood,
            'converged': self.converged,
            'error': self.error
        }

        # Add timing statistics
        result.update(self.timing_stats_to_dict(self.estimation_stats, "estimation_"))


        return result


def get_estimation_configs() -> List[Tuple[str, OptimizationConfig]]:
    """Get configurations for estimation benchmarks."""
    configs = OptimizationConfig.sequential()
    # For estimation, we want cython, sparse, derivatives
    return [configs[3], configs[4], configs[5]]


def run_estimation_benchmark(
    n_players: int,
    n_demand: int,
    config_name: str,
    config: OptimizationConfig,
    test_parameters: dict,
    n_runs: int = 1,
    n_obs: int = 100
) -> EstimationBenchmarkResult:
    """Run a single estimation benchmark."""
    n_states = (2 ** n_players) * n_demand

    try:
        # Create Model instance for DGP
        dgp_model = Model(n_players, n_demand, test_parameters, verbose=False)

        # Generate sample data using model's DGP
        sample = dgp_model.discrete_time_dgp(n_obs=n_obs, Delta=1.0, seed=42)

        # Create estimation model with the specified configuration
        model = ConfigurableModel(
            n_players, n_demand, test_parameters, verbose=False, config=config
        )

        # Parameter starting values (perturb true parameters slightly)
        param_keys = model.param_keys
        theta_start = np.array([test_parameters[key] * 1.1 for key in param_keys])

        # Run estimation using statistical timing
        def estimation_func():
            return model.estimate_parameters(
                sample=sample,
                start=theta_start,
                max_iter=100,
                use_grad=config.derivatives,
            )

        result, estimation_stats = time_operation(estimation_func, n_runs)

        return EstimationBenchmarkResult(
            config=config_name,
            model_size=(n_players, n_demand),
            n_states=n_states,
            n_obs=n_obs,
            estimation_stats=estimation_stats,
            n_iterations=result.nit if hasattr(result, 'nit') else 0,
            n_likelihood_evals=result.nfev if hasattr(result, 'nfev') else 0,
            final_likelihood=-result.fun if hasattr(result, 'fun') else 0,
            converged=result.success if hasattr(result, 'success') else False
        )

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"

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

        return EstimationBenchmarkResult(
            config=config_name,
            model_size=(n_players, n_demand),
            n_states=n_states,
            n_obs=n_obs,
            estimation_stats=zero_stats,
            n_iterations=0,
            n_likelihood_evals=0,
            final_likelihood=0,
            converged=False,
            error=error_msg
        )


def run_estimation_benchmarks(size_groups: List[str], model_sizes: dict, test_parameters: dict, is_quick: bool = False, verbose: bool = True) -> List[EstimationBenchmarkResult]:
    """Run estimation benchmarks for specified size groups."""
    if verbose:
        print()
        print("Estimation Benchmarks")
        print("=====================")

    # Determine number of runs
    n_runs = 1 if is_quick else 5
    if verbose:
        print(f"Running {n_runs} replications{'s' if n_runs > 1 else ''} per benchmark")

    results = []
    configs = get_estimation_configs()

    n_obs_by_size = {
        'very_small': 1000,
        'small': 1000,
        'medium': 1000,
        'large': 1000,
        'very_large': 1000,
    }

    for size_group in size_groups:
        if size_group not in model_sizes:
            continue

        if verbose:
            print_size_group_subheader(size_group)

        n_obs = n_obs_by_size.get(size_group, 100)

        for n_players, n_demand in model_sizes[size_group]:
            n_states = (2 ** n_players) * n_demand
            if verbose:
                print(f"\nModel: {n_players}×{n_demand} ({n_states} states, {n_obs:,} observations)")

            for config_name, config in configs:
                if verbose:
                    print(f"  {config_name}...", end=' ', flush=True)
                result = run_estimation_benchmark(
                    n_players, n_demand, config_name, config, test_parameters, n_runs, n_obs
                )

                if result.error:
                    if verbose:
                        print(f"ERROR: {result.error}")
                else:
                    if verbose:
                        if n_runs == 1:
                            print(f"{result.estimation_stats.median:.2f}s (LL: {result.final_likelihood:.3f}, neval: {result.n_likelihood_evals})")
                        else:
                            print(f"{result.estimation_stats.median:.2f}s (±{result.estimation_stats.std:.2f}) (LL: {result.final_likelihood:.3f}, neval: {result.n_likelihood_evals})")
                    results.append(result)

    return results
