#!/usr/bin/env python
"""
Benchmark module for matrix exponential computation.
"""

import numpy as np
from scipy.linalg import expm
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .benchmark_utils import BaseBenchmarkResult, TimingStats, time_operation, print_size_group_subheader
from sparse import vexpm
from model import Model


@dataclass
class ExpmBenchmarkResult(BaseBenchmarkResult):
    """Results from a matrix exponential benchmark run."""
    model_size: Tuple[int, int]  # (n_players, n_demand)
    n_states: int
    method: str  # 'dense' or 'sparse'

    # Timing results
    setup_stats: TimingStats
    full_expm_stats: TimingStats
    columns_expm_stats: TimingStats

    # Accuracy results (compared to dense)
    max_absolute_error: float
    max_relative_error: float

    # Additional info
    n_columns: int
    error: Optional[str] = None

    def to_dict(self) -> dict:
        result = {
            'n_players': self.model_size[0],
            'n_demand': self.model_size[1],
            'n_states': self.n_states,
            'method': self.method,
            'max_absolute_error': self.max_absolute_error,
            'max_relative_error': self.max_relative_error,
            'n_columns': self.n_columns,
            'error': self.error
        }

        # Add timing statistics with prefixes
        result.update(self.timing_stats_to_dict(self.setup_stats, "setup_"))
        result.update(self.timing_stats_to_dict(self.full_expm_stats, "full_expm_"))
        result.update(self.timing_stats_to_dict(self.columns_expm_stats, "columns_expm_"))


        return result


def run_expm_benchmark(
    n_players: int,
    n_demand: int,
    method: str,
    test_parameters: dict,
    n_runs: int = 1,
    reference_result: Optional[np.ndarray] = None
) -> ExpmBenchmarkResult:
    """
    Run matrix exponential benchmark for a specific method.

    Parameters
    ----------
    n_players : int
        Number of players
    n_demand : int
        Number of demand states
    method : str
        'dense' or 'sparse'
    test_parameters : dict
        Model parameters
    reference_result : np.ndarray, optional
        Reference result from dense method for error calculation

    Returns
    -------
    result : ExpmBenchmarkResult
        Benchmark results
    """
    n_states = (2 ** n_players) * n_demand
    Delta = 1.0

    try:
        # Setup: Create model and get intensity matrix
        def setup_func():
            model = Model(n_players, n_demand, test_parameters, verbose=False)
            Q, _ = model.intensity_matrix()
            return model, Q

        (model, Q), setup_stats = time_operation(setup_func, n_runs)

        # Convert to appropriate format
        if method == 'dense':
            if hasattr(Q, 'toarray'):
                Q_test = Q.toarray()
            else:
                Q_test = Q
        else:  # sparse
            Q_test = Q

        # Full matrix exponential timing
        if method == 'dense':
            def full_expm_func():
                return expm(Q_test * Delta)

            P_result, full_stats = time_operation(full_expm_func, n_runs)
        else:  # sparse
            def full_expm_func():
                P_result = np.zeros((n_states, n_states))
                e_j = np.zeros(n_states)
                for j in range(n_states):
                    e_j[j] = 1.0
                    P_result[:, j] = vexpm(Q_test, Delta, e_j)
                    e_j[j] = 0.0
                return P_result

            P_result, full_stats = time_operation(full_expm_func, n_runs)

        # Column-based timing (simulate log-likelihood computation)
        n_columns = min(200, n_states)

        if method == 'dense':
            def columns_func():
                P_full = expm(Q_test * Delta)
                return P_full[:, :n_columns]

            _, columns_stats = time_operation(columns_func, n_runs)
        else:  # sparse
            def columns_func():
                P_cols = np.zeros((n_states, n_columns))
                e_j = np.zeros(n_states)
                for j in range(n_columns):
                    e_j[j] = 1.0
                    P_cols[:, j] = vexpm(Q_test, Delta, e_j)
                    e_j[j] = 0.0
                return P_cols

            _, columns_stats = time_operation(columns_func, n_runs)

        # Calculate errors (if reference provided)
        if reference_result is not None and method != 'dense':
            abs_error = np.abs(P_result - reference_result)
            max_abs_error = np.max(abs_error)

            # Relative error (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_error = abs_error / np.maximum(np.abs(reference_result), 1e-15)
            max_rel_error = np.max(rel_error[np.isfinite(rel_error)])

        else:
            max_abs_error = 0.0
            max_rel_error = 0.0

        return ExpmBenchmarkResult(
            model_size=(n_players, n_demand),
            n_states=n_states,
            method=method,
            setup_stats=setup_stats,
            full_expm_stats=full_stats,
            columns_expm_stats=columns_stats,
            max_absolute_error=max_abs_error,
            max_relative_error=max_rel_error,
            n_columns=n_columns
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

        return ExpmBenchmarkResult(
            model_size=(n_players, n_demand),
            n_states=n_states,
            method=method,
            setup_stats=zero_stats,
            full_expm_stats=zero_stats,
            columns_expm_stats=zero_stats,
            max_absolute_error=float('inf'),
            max_relative_error=float('inf'),
            n_columns=0,
            error=str(e)
        )


def run_expm_benchmarks(size_groups: List[str], model_sizes: dict, test_parameters: dict, is_quick: bool = False, verbose: bool = True) -> List[ExpmBenchmarkResult]:
    """Run matrix exponential benchmarks for specified size groups."""
    if verbose:
        print()
        print("Matrix Exponential Benchmarks")
        print("=============================")

    # Determine number of runs: 1 for quick, 5 for full
    n_runs = 1 if is_quick else 5
    if verbose:
        print(f"Running {n_runs} replications{'s' if n_runs > 1 else ''} per benchmark")

    results = []
    methods = ['dense', 'sparse']

    for size_group in size_groups:
        if size_group not in model_sizes:
            continue

        if verbose:
            print_size_group_subheader(size_group)

        for n_players, n_demand in model_sizes[size_group]:
            n_states = (2 ** n_players) * n_demand
            if verbose:
                print(f"\nModel: {n_players}×{n_demand} ({n_states} states)")

            # Run dense first (reference)
            dense_result = None
            dense_matrix = None

            for method in methods:
                if verbose:
                    print(f"  {method}...", end=' ', flush=True)

                if method == 'dense':
                    result = run_expm_benchmark(n_players, n_demand, method, test_parameters, n_runs)
                    dense_result = result
                    # Store the matrix result for error calculation
                    model = Model(n_players, n_demand, test_parameters, verbose=False)
                    Q, _ = model.intensity_matrix()
                    Q_dense = Q.toarray() if hasattr(Q, 'toarray') else Q
                    dense_matrix = expm(Q_dense * 1.0)
                else:
                    result = run_expm_benchmark(n_players, n_demand, method, test_parameters, n_runs, dense_matrix)

                if result.error:
                    if verbose:
                        print(f"ERROR: {result.error}")
                else:
                    if verbose:
                        if method == 'dense':
                            if n_runs == 1:
                                print(f"full: {result.full_expm_stats.median:.4f}s, {result.n_columns} cols: {result.columns_expm_stats.median:.4f}s")
                            else:
                                print(f"full: {result.full_expm_stats.median:.4f}s (±{result.full_expm_stats.std:.4f}), {result.n_columns} cols: {result.columns_expm_stats.median:.4f}s (±{result.columns_expm_stats.std:.4f})")
                        else:
                            full_speedup = dense_result.full_expm_stats.median / result.full_expm_stats.median if result.full_expm_stats.median > 0 else 0
                            cols_speedup = dense_result.columns_expm_stats.median / result.columns_expm_stats.median if result.columns_expm_stats.median > 0 else 0
                            if n_runs == 1:
                                print(f"full: {result.full_expm_stats.median:.4f}s ({full_speedup:.2f}x), {result.n_columns} cols: {result.columns_expm_stats.median:.4f}s ({cols_speedup:.2f}x), max_err: {result.max_absolute_error:.2e}")
                            else:
                                print(f"full: {result.full_expm_stats.median:.4f}s (±{result.full_expm_stats.std:.4f}) ({full_speedup:.2f}x), {result.n_columns} cols: {result.columns_expm_stats.median:.4f}s (±{result.columns_expm_stats.std:.4f}) ({cols_speedup:.2f}x), max_err: {result.max_absolute_error:.2e}")

                    results.append(result)

    return results
