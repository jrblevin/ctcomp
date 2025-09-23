#!/usr/bin/env python
"""
Benchmark runner - sequential or parallel execution.
"""

import os
from typing import List, Dict, Any
from multiprocessing import Pool

from benchmarks import (
    create_benchmark_config, print_system_info,
    run_computation_benchmarks, run_estimation_benchmarks,
    run_sparsity_benchmarks, run_expm_benchmarks
)


def setup_worker_environment(n_workers: int, total_cores: int = None):
    """Configure worker process threading."""
    if total_cores is None:
        import psutil
        # Use physical cores for accurate core count
        total_cores = psutil.cpu_count(logical=False)
    threads_per_worker = max(1, total_cores // n_workers)

    os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads_per_worker)
    os.environ['MKL_NUM_THREADS'] = str(threads_per_worker)
    os.environ['NUMEXPR_NUM_THREADS'] = str(threads_per_worker)

    # Suppress worker output to avoid console clutter
    import sys
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


def execute_benchmark_task(args):
    """Execute a single benchmark task."""
    benchmark_type, size_groups, model_sizes, config, is_quick, verbose = args

    try:
        # Call the appropriate benchmark function
        if benchmark_type == 'computation':
            return run_computation_benchmarks(size_groups, model_sizes, config.test_parameters, is_quick, verbose=verbose)
        elif benchmark_type == 'estimation':
            return run_estimation_benchmarks(size_groups, model_sizes, config.test_parameters, is_quick, verbose=verbose)
        elif benchmark_type == 'sparsity':
            return run_sparsity_benchmarks(size_groups, model_sizes, config.test_parameters, is_quick, verbose=verbose)
        elif benchmark_type == 'expm':
            return run_expm_benchmarks(size_groups, model_sizes, config.test_parameters, is_quick, verbose=verbose)
        else:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")

    except Exception as e:
        return f"ERROR in {benchmark_type}: {e}"


class BenchmarkRunner:
    def __init__(self, parallel: bool = False, n_workers: int = None, total_cores: int = None):
        self.parallel = parallel
        # Auto-detect total cores if not specified
        if total_cores is None:
            import psutil
            # Use physical cores for accurate core count
            total_cores = psutil.cpu_count(logical=False)
        self.total_cores = total_cores

        # Auto-determine workers for parallel mode
        # Use all physical cores by default
        if parallel and n_workers is None:
            self.n_workers = total_cores
        else:
            self.n_workers = n_workers or 1

        self.config = create_benchmark_config()

        # Print system information at the start
        print_system_info()
        if parallel:
            print(f"Parallel mode: {self.n_workers} workers, {total_cores // self.n_workers} threads each")
        else:
            print("Sequential mode")

    def run_benchmarks(self, benchmark_types: List[str], size_groups: List[str],
                       is_quick: bool = False) -> Dict[str, List[Any]]:
        print(f"Running {len(benchmark_types)} benchmark types on {len(size_groups)} size groups\n")

        # Filter valid size groups
        valid_size_groups = [sg for sg in size_groups if sg in self.config.model_sizes]

        if self.parallel:
            return self._run_parallel(benchmark_types, valid_size_groups, is_quick)
        else:
            return self._run_sequential(benchmark_types, valid_size_groups, is_quick)

    def _run_sequential(self, benchmark_types: List[str], size_groups: List[str], is_quick: bool) -> Dict[str, List[Any]]:
        all_results = {'computation': [], 'estimation': [], 'sparsity': [], 'expm': []}

        for benchmark_type in benchmark_types:
            # Prepare model sizes dict for this benchmark
            model_sizes = {sg: self.config.model_sizes[sg] for sg in size_groups}

            # Execute benchmark
            results = execute_benchmark_task((
                benchmark_type, size_groups, model_sizes, self.config, is_quick, True
            ))

            if isinstance(results, str) and results.startswith("ERROR"):
                print(f"Error in {benchmark_type}: {results}")
            else:
                all_results[benchmark_type].extend(results)

        return all_results

    def _run_parallel(self, benchmark_types: List[str], size_groups: List[str], is_quick: bool) -> Dict[str, List[Any]]:
        # Prepare task arguments - one task per benchmark type
        model_sizes = {sg: self.config.model_sizes[sg] for sg in size_groups}
        task_args = [
            (benchmark_type, size_groups, model_sizes, self.config, is_quick, False)
            for benchmark_type in benchmark_types
        ]

        all_results = {'computation': [], 'estimation': [], 'sparsity': [], 'expm': []}

        with Pool(processes=self.n_workers,
                  initializer=setup_worker_environment,
                  initargs=(self.n_workers, self.total_cores)) as pool:

            for i, (benchmark_type, results) in enumerate(zip(benchmark_types, pool.map(execute_benchmark_task, task_args)), 1):
                if isinstance(results, str) and results.startswith("ERROR"):
                    print(f"[{i}/{len(benchmark_types)}] ✗ {benchmark_type} - {results}")
                else:
                    print(f"[{i}/{len(benchmark_types)}] ✓ {benchmark_type}")
                    all_results[benchmark_type].extend(results)

        return all_results
