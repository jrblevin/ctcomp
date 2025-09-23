#!/usr/bin/env python
"""
Benchmark module for matrix sparsity analysis.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .benchmark_utils import BaseBenchmarkResult, print_size_group_subheader
from model import Model


@dataclass
class SparsityBenchmarkResult(BaseBenchmarkResult):
    """Results from a matrix sparsity analysis."""
    model_size: Tuple[int, int]  # (n_players, n_demand)
    n_states: int
    q_matrix_size: int
    q_matrix_nonzeros: int
    q_matrix_sparsity: float
    dt_dv_matrix_size: int
    dt_dv_matrix_nonzeros: int
    dt_dv_matrix_sparsity: float
    computation_time: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'n_players': self.model_size[0],
            'n_demand': self.model_size[1],
            'n_states': self.n_states,
            'q_matrix_size': self.q_matrix_size,
            'q_matrix_nonzeros': self.q_matrix_nonzeros,
            'q_matrix_sparsity': self.q_matrix_sparsity,
            'dt_dv_matrix_size': self.dt_dv_matrix_size,
            'dt_dv_matrix_nonzeros': self.dt_dv_matrix_nonzeros,
            'dt_dv_matrix_sparsity': self.dt_dv_matrix_sparsity,
            'computation_time': self.computation_time,
            'error': self.error
        }


def run_sparsity_benchmark(
    n_players: int,
    n_demand: int,
    test_parameters: dict
) -> SparsityBenchmarkResult:
    """Run a single sparsity analysis benchmark."""
    n_states = (2 ** n_players) * n_demand

    try:
        # Setup timing
        start_time = time.time()

        # Create model using the production Model class for accurate sparsity
        model = Model(n_players, n_demand, test_parameters, rho=0.05, verbose=False)

        # Get converged value function
        v, _ = model.value_function(vf_max_iter=1000, vf_tol=1e-13)

        # Analyze Q matrix (intensity matrix)
        Q, _ = model.intensity_matrix()
        q_matrix_size = Q.shape[0] * Q.shape[1]
        q_matrix_nonzeros = Q.nnz
        q_matrix_sparsity = 1.0 - (q_matrix_nonzeros / q_matrix_size)

        # Analyze dT/dv matrix (Bellman operator Jacobian)
        dt_dv = model.dbellman_operator_dv(v)
        dt_dv_matrix_size = dt_dv.shape[0] * dt_dv.shape[1]
        dt_dv_matrix_nonzeros = dt_dv.nnz
        dt_dv_matrix_sparsity = 1.0 - (dt_dv_matrix_nonzeros / dt_dv_matrix_size)

        computation_time = time.time() - start_time

        return SparsityBenchmarkResult(
            model_size=(n_players, n_demand),
            n_states=n_states,
            q_matrix_size=q_matrix_size,
            q_matrix_nonzeros=q_matrix_nonzeros,
            q_matrix_sparsity=q_matrix_sparsity,
            dt_dv_matrix_size=dt_dv_matrix_size,
            dt_dv_matrix_nonzeros=dt_dv_matrix_nonzeros,
            dt_dv_matrix_sparsity=dt_dv_matrix_sparsity,
            computation_time=computation_time
        )

    except Exception as e:
        return SparsityBenchmarkResult(
            model_size=(n_players, n_demand),
            n_states=n_states,
            q_matrix_size=0,
            q_matrix_nonzeros=0,
            q_matrix_sparsity=0,
            dt_dv_matrix_size=0,
            dt_dv_matrix_nonzeros=0,
            dt_dv_matrix_sparsity=0,
            computation_time=0,
            error=str(e)
        )


def run_sparsity_benchmarks(size_groups: List[str], model_sizes: dict, test_parameters: dict, is_quick: bool = False, verbose: bool = True) -> List[SparsityBenchmarkResult]:
    """Run sparsity analysis benchmarks for specified size groups."""
    if verbose:
        print()
        print("Sparsity Analysis")
        print("=================")

    results = []

    for size_group in size_groups:
        if size_group not in model_sizes:
            continue

        if verbose:
            print_size_group_subheader(size_group)

        for n_players, n_demand in model_sizes[size_group]:
            n_states = (2 ** n_players) * n_demand
            if verbose:
                print(f"\nModel: {n_players}×{n_demand} ({n_states} states)")
                print("  Analyzing sparsity...", end=' ', flush=True)

            result = run_sparsity_benchmark(n_players, n_demand, test_parameters)

            if result.error:
                if verbose:
                    print(f"ERROR: {result.error}")
            else:
                if verbose:
                    print(f"Q: {result.q_matrix_sparsity:.1%} sparse, dT/dV: {result.dt_dv_matrix_sparsity:.1%} sparse ({result.computation_time:.3f}s)")
                results.append(result)

    return results
