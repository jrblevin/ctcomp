#!/usr/bin/env python
"""
Benchmark suite orchestrator.

This script benchmarks both computation and estimation performance across
different model sizes and optimization configurations.

Usage:
    python run_benchmarks.py --quick           # Quick test (small models)
    python run_benchmarks.py --computation     # Computation benchmarks only
    python run_benchmarks.py --estimation      # Estimation benchmarks only
    python run_benchmarks.py --sparsity        # Sparsity analysis only
    python run_benchmarks.py --expm            # Matrix expm benchmarks only
    python run_benchmarks.py --full            # Full benchmark suite
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Any

from benchmarks import (
    create_benchmark_config,
    run_computation_benchmarks, run_estimation_benchmarks,
    run_sparsity_benchmarks, run_expm_benchmarks,
    save_all_results, print_summary, BenchmarkRunner
)


class BenchmarkSuite:
    """Main benchmark orchestrator."""

    def __init__(self):
        self.config = create_benchmark_config()

    def run_benchmarks(self, benchmark_types: List[str], size_groups: List[str], is_quick: bool = False) -> Dict[str, List[Any]]:
        """Run specified benchmarks and return results."""
        results = {}

        # Suppress warnings during benchmarks
        warnings.filterwarnings('ignore')

        if 'computation' in benchmark_types:
            results['computation'] = run_computation_benchmarks(
                size_groups, self.config.model_sizes, self.config.test_parameters, is_quick
            )

        if 'estimation' in benchmark_types:
            results['estimation'] = run_estimation_benchmarks(
                size_groups, self.config.model_sizes, self.config.test_parameters, is_quick
            )

        if 'sparsity' in benchmark_types:
            results['sparsity'] = run_sparsity_benchmarks(
                size_groups, self.config.model_sizes, self.config.test_parameters, is_quick
            )

        if 'expm' in benchmark_types:
            results['expm'] = run_expm_benchmarks(
                size_groups, self.config.model_sizes, self.config.test_parameters, is_quick
            )

        return results


def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Benchmark computation and estimation performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmarks.py --quick           # Quick test with small models
  python run_benchmarks.py --computation     # Computation benchmarks only
  python run_benchmarks.py --estimation      # Estimation benchmarks only
  python run_benchmarks.py --sparsity        # Sparsity analysis benchmarks only
  python run_benchmarks.py --expm            # Matrix exponential benchmarks only
  python run_benchmarks.py --full            # Full benchmarks for paper
  python run_benchmarks.py --regenerate      # Regenerate tables/figures from saved results
  python run_benchmarks.py --parallel --full # Parallel execution with 48 cores
        """
    )

    parser.add_argument('--quick', action='store_true',
                        help='Quick test with small models only')
    parser.add_argument('--computation', action='store_true',
                        help='Run computation benchmarks only')
    parser.add_argument('--estimation', action='store_true',
                        help='Run estimation benchmarks only')
    parser.add_argument('--sparsity', action='store_true',
                        help='Run sparsity analysis benchmarks only')
    parser.add_argument('--expm', action='store_true',
                        help='Run matrix exponential benchmarks only')
    parser.add_argument('--full', action='store_true',
                        help='Run full benchmarks for paper')
    parser.add_argument('--regenerate', action='store_true',
                        help='Regenerate tables and figures from existing results')
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel execution with multiprocessing')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoints (automatically enabled with --parallel)')
    parser.add_argument('--max-workers', type=int,
                        help='Maximum number of worker processes (default: auto-detect)')
    parser.add_argument('--output', type=str, default='benchmark_results',
                        help='Output directory for results (or input for --regenerate)')
    parser.add_argument('--input', type=str,
                        help='Input directory for --regenerate (defaults to --output)')

    return parser


def determine_benchmark_scope(args):
    """Determine what benchmarks to run and what size groups to use."""
    if args.quick:
        size_groups = ['very_small']
        benchmark_types = ['computation', 'sparsity', 'expm', 'estimation']
    elif args.computation:
        size_groups = ['very_small', 'small', 'medium', 'large', 'very_large']
        benchmark_types = ['computation']
    elif args.estimation:
        size_groups = ['very_small', 'small', 'medium', 'large', 'very_large']
        benchmark_types = ['estimation']
    elif args.sparsity:
        size_groups = ['very_small', 'small', 'medium', 'large', 'very_large']
        benchmark_types = ['sparsity']
    elif args.expm:
        size_groups = ['very_small', 'small', 'medium', 'large', 'very_large']
        benchmark_types = ['expm']
    elif args.full:
        size_groups = ['very_small', 'small', 'medium', 'large', 'very_large']
        benchmark_types = ['computation', 'estimation', 'sparsity', 'expm']
    else:
        return None, None

    return benchmark_types, size_groups


def handle_regenerate(args):
    """Handle regeneration of tables and figures from existing results."""
    from benchmarks import load_results_from_json

    input_dir = Path(args.input) if args.input else Path(args.output)
    output_dir = Path(args.output)

    print("Regenerating tables and figures from existing results...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    try:
        # Load existing results
        results = load_results_from_json(input_dir)

        if not any(results.values()):
            print(f"Error: No valid results found in {input_dir}")
            print("Expected files: computation_results.json, estimation_results.json, sparsity_results.json, expm_results.json")
            sys.exit(1)

        result_counts = {k: len(v) for k, v in results.items() if v}
        print(f"Loaded results: {result_counts}")

        # Create config for model sizes
        config = create_benchmark_config()

        # Regenerate outputs without running benchmarks (don't resave JSON files)
        save_all_results(results, output_dir, config.model_sizes, save_raw_data=False)
        print_summary(results)
        print("\nTables and figures regenerated successfully!")

    except Exception as e:
        print(f"Error during regeneration: {e}")
        sys.exit(1)


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate arguments
    benchmark_flags = [args.quick, args.computation, args.estimation,
                       args.sparsity, args.expm, args.full, args.regenerate]
    if not any(benchmark_flags):
        parser.print_help()
        sys.exit(1)

    # Handle regenerate option
    if args.regenerate:
        handle_regenerate(args)
        return

    # Determine benchmark scope
    benchmark_types, size_groups = determine_benchmark_scope(args)
    if benchmark_types is None:
        parser.print_help()
        sys.exit(1)

    # Run benchmarks
    output_dir = Path(args.output)
    is_quick = args.quick

    # Use benchmark runner
    runner = BenchmarkRunner(
        parallel=args.parallel,
        n_workers=getattr(args, 'max_workers', None)
    )
    results = runner.run_benchmarks(benchmark_types, size_groups, is_quick)

    # Save results and generate outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    config = create_benchmark_config()
    save_all_results(results, output_dir, config.model_sizes)
    print_summary(results)

    print("\nBenchmarking complete!")


if __name__ == "__main__":
    main()
