#!/usr/bin/env python
"""
Benchmark output generation - tables, figures, and result management.
"""

import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple

from benchmarks.computation import ComputationBenchmarkResult
from benchmarks.estimation import EstimationBenchmarkResult
from benchmarks.sparsity import SparsityBenchmarkResult
from benchmarks.expm import ExpmBenchmarkResult
from benchmarks.benchmark_utils import TimingStats, print_header, get_system_info
from optimization_config import OptimizationConfig
from benchmarks.plot_style import setup_plot_style, save_figure


# ==============================================================================
# Constants
# ==============================================================================

# Model size categories
CATEGORY_NAMES = ['very_small', 'small', 'medium', 'large', 'very_large']
CATEGORY_LABELS = {
    'very_small': 'Very Small',
    'small': 'Small',
    'medium': 'Medium',
    'large': 'Large',
    'very_large': 'Very Large'
}


# ==============================================================================
# LaTeX Table Generation
# ==============================================================================

def create_computation_latex_table(results: List[ComputationBenchmarkResult]) -> str:
    """Create LaTeX table for computation benchmarks."""
    # Get model size categories
    from benchmarks.benchmark_utils import get_model_sizes
    size_categories = get_model_sizes()

    # Group results by model size
    by_size = {}
    for r in results:
        if r.error:
            continue
        key = r.model_size
        if key not in by_size:
            by_size[key] = {}
        # Match config to name based on settings
        if not r.config.vectorize:
            by_size[key]["baseline"] = r
        elif r.config.vectorize and not r.config.polyalgorithm:
            by_size[key]["vectorize"] = r
        elif r.config.polyalgorithm and not r.config.cython:
            by_size[key]["polyalgorithm"] = r
        elif r.config.cython and not r.config.sparse:
            by_size[key]["cython"] = r

    # Create table header with configurations as columns
    latex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Computation Performance Comparison}",
        r"\label{tab:computation}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{1}{c}{Baseline} & \multicolumn{2}{c}{Vectorize} & \multicolumn{3}{c}{Polyalgorithm} & \multicolumn{3}{c}{Cython} \\",
        r"\cmidrule(lr){2-2} \cmidrule(lr){3-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}",
        r"Model & VF & VF & vs.~Base & VF & vs.~Base & vs.~Vec & VF & vs.~Base & vs.~Poly \\",
        r"\midrule"
    ]

    # Process each size category
    first_category = True
    for cat_name in CATEGORY_NAMES:
        if cat_name not in size_categories:
            continue

        # Filter sizes for this category
        cat_sizes = [(size, data) for size, data in by_size.items()
                     if size in size_categories[cat_name]]

        if not cat_sizes:
            continue

        # Add category header
        if not first_category:
            latex.append(r"\addlinespace")
        latex.append(rf"\multicolumn{{10}}{{l}}{{\textit{{{CATEGORY_LABELS[cat_name]} Models}}}} \\")
        first_category = False

        # Process each model size in the category
        for size, size_data in sorted(cat_sizes):
            n_players, n_demand = size

            # Get times for each configuration
            baseline_time = size_data.get("baseline").value_function_stats.median if "baseline" in size_data else None
            vectorize_time = size_data.get("vectorize").value_function_stats.median if "vectorize" in size_data else None
            poly_time = size_data.get("polyalgorithm").value_function_stats.median if "polyalgorithm" in size_data else None
            cython_time = size_data.get("cython").value_function_stats.median if "cython" in size_data else None

            # Skip if we don't have baseline
            if baseline_time is None:
                continue

            # Calculate speedups
            vec_vs_base = baseline_time / vectorize_time if vectorize_time else 0
            poly_vs_base = baseline_time / poly_time if poly_time else 0
            poly_vs_vec = vectorize_time / poly_time if vectorize_time and poly_time else 0
            cython_vs_base = baseline_time / cython_time if cython_time else 0
            cython_vs_poly = poly_time / cython_time if poly_time and cython_time else 0

            # Format row
            row = f"${n_players} \\times {n_demand}$ & "

            # Baseline column
            row += f"{baseline_time:.3f} & "

            # Vectorize columns
            if vectorize_time:
                row += f"{vectorize_time:.3f} & {vec_vs_base:.1f}$\\times$ & "
            else:
                row += "-- & -- & "

            # Polyalgorithm columns
            if poly_time:
                row += f"{poly_time:.3f} & {poly_vs_base:.1f}$\\times$ & {poly_vs_vec:.1f}$\\times$ & "
            else:
                row += "-- & -- & -- & "

            # Cython columns
            if cython_time:
                row += f"{cython_time:.3f} & {cython_vs_base:.1f}$\\times$ & {cython_vs_poly:.1f}$\\times$ \\\\"
            else:
                row += "-- & -- & -- \\\\"

            latex.append(row)

    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}\newline",
        r"\footnotesize",
        r"``VF'' denotes time in seconds to compute the value function.",
        r"``vs. Base'' denotes the speedup of value function computation over the baseline.",
        r"Similarly, ``vs. Vec'' and ``vs. Poly'' denote speedups relative to the prior optimizations.",
        r"\end{table}"
    ])

    return "\n".join(latex)


def create_estimation_latex_table(results: List[EstimationBenchmarkResult]) -> str:
    """Create LaTeX table for estimation benchmarks."""
    # Get model size categories
    from benchmarks.benchmark_utils import get_model_sizes
    size_categories = get_model_sizes()

    # Group by model size and config
    by_size = {}
    for r in results:
        if r.error:
            continue
        if r.model_size not in by_size:
            by_size[r.model_size] = {}
        by_size[r.model_size][r.config] = r

    # Create table with configurations as columns
    latex = [
        r"\begin{landscape}",
        r"\begin{table}[p]",
        r"\centering",
        r"\caption{Estimation Performance}",
        r"\label{tab:estimation}",
        r"\begin{tabular}{lrrrrrrrrrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Cython} & \multicolumn{5}{c}{Sparse} & \multicolumn{5}{c}{Derivatives} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-10} \cmidrule(lr){11-15}",
        r"Model & Time & Iter & Evals & LL & Time & Iter & Evals & LL & vs. Cython & Time & Iter & Evals & LL & vs. Cython & vs. Sparse \\",
        r"\midrule"
    ]

    # Process each size category
    first_category = True
    for cat_name in CATEGORY_NAMES:
        if cat_name not in size_categories:
            continue

        # Filter sizes for this category
        cat_sizes = [(size, data) for size, data in by_size.items()
                     if size in size_categories[cat_name]]

        if not cat_sizes:
            continue

        # Add category header
        if not first_category:
            latex.append(r"\addlinespace")
        latex.append(rf"\multicolumn{{15}}{{l}}{{\textit{{{CATEGORY_LABELS[cat_name]} Models}}}} \\")
        first_category = False

        # Process each model size in the category
        for size, size_data in sorted(cat_sizes):
            n_players, n_demand = size

            # Get results for each configuration
            cython_result = size_data.get("cython")
            sparse_result = size_data.get("sparse")
            derivatives_result = size_data.get("derivatives")

            # Skip if we don't have cython baseline
            if not cython_result:
                continue

            # Get baseline time for speedup calculations
            baseline_time = cython_result.estimation_stats.median

            # Calculate speedups
            sparse_vs_cython = baseline_time / sparse_result.estimation_stats.median if sparse_result else 0
            derivatives_vs_cython = baseline_time / derivatives_result.estimation_stats.median if derivatives_result else 0
            derivatives_vs_sparse = sparse_result.estimation_stats.median if sparse_result else 0 / derivatives_result.estimation_stats.median if derivatives_result else 0

            # Format row
            row = f"${n_players} \\times {n_demand}$ & "

            # Cython columns (baseline, no speedup)
            row += f"{cython_result.estimation_stats.median:.2f} & "
            row += f"{cython_result.n_iterations} & "
            row += f"{cython_result.n_likelihood_evals} & "
            row += f"{cython_result.final_likelihood:.3f} & "

            # Sparse columns
            if sparse_result:
                row += f"{sparse_result.estimation_stats.median:.2f} & "
                row += f"{sparse_result.n_iterations} & "
                row += f"{sparse_result.n_likelihood_evals} & "
                row += f"{sparse_result.final_likelihood:.3f} & "
                row += f"{sparse_vs_cython:.1f}$\\times$ & "
            else:
                row += "-- & -- & -- & -- & -- & "

            # Derivatives columns
            if derivatives_result:
                row += f"{derivatives_result.estimation_stats.median:.2f} & "
                row += f"{derivatives_result.n_iterations} & "
                row += f"{derivatives_result.n_likelihood_evals} & "
                row += f"{derivatives_result.final_likelihood:.3f} & "
                row += f"{derivatives_vs_cython:.1f}$\\times$ &"
                row += f"{derivatives_vs_sparse:.1f}$\\times$ \\\\"
            else:
                row += "-- & -- & -- & -- & -- & -- \\\\"

            latex.append(row)

    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\footnotesize",
        r"Note: Time in seconds, LL = Log-likelihood, Evals = Function evaluations, Speedup = Cython time / Config time.",
        r"\end{table}",
        r"\end{landscape}"
    ])

    return "\n".join(latex)


def create_sparsity_latex_table(results: List[SparsityBenchmarkResult]) -> str:
    """Create LaTeX table for sparsity analysis."""
    # Get model size categories
    from benchmarks.benchmark_utils import get_model_sizes
    size_categories = get_model_sizes()

    # Group by model size
    by_size = {}
    for r in results:
        if r.error:
            continue
        if r.model_size not in by_size:
            by_size[r.model_size] = r

    # Create table
    latex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Matrix Sparsity Analysis}",
        r"\label{tab:sparsity}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r" &  & \multicolumn{2}{c}{Intensity Matrix $Q$} & \multicolumn{2}{c}{Jacobian $\partial T/\partial V$} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"Model & States ($K$) & Size ($K^2$) & Sparsity & Size ($K^2 \Nplayers^2$) & Sparsity \\",
        r"\midrule"
    ]

    # Process each size category
    first_category = True
    for cat_name in CATEGORY_NAMES:
        if cat_name not in size_categories:
            continue

        # Filter sizes for this category
        cat_sizes = [(size, by_size[size]) for size in by_size.keys()
                     if size in size_categories[cat_name]]

        if not cat_sizes:
            continue

        # Add category header
        if not first_category:
            latex.append(r"\addlinespace")
        latex.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{CATEGORY_LABELS[cat_name]} Models}}}} \\")
        first_category = False

        # Process each model size in the category
        for size, r in sorted(cat_sizes):
            n_players, n_demand = size

            # Format matrix sizes with commas
            q_size_str = f"{r.q_matrix_size:,}"
            dt_dv_size_str = f"{r.dt_dv_matrix_size:,}"

            latex.append(
                f"${n_players} \\times {n_demand}$ & "
                f"{r.n_states:,} & "
                f"{q_size_str} & "
                f"{r.q_matrix_sparsity*100:.2f}\\% & "
                f"{dt_dv_size_str} & "
                f"{r.dt_dv_matrix_sparsity*100:.2f}\\% \\\\"
            )

    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}\newline",
        r"\footnotesize",
        r"Sparsity denotes the percentage of zero elements.",
        r"\end{table}"
    ])

    return "\n".join(latex)


def create_expm_latex_table(results: List[ExpmBenchmarkResult]) -> str:
    """Create LaTeX table for matrix exponential benchmarks."""
    # Get model size categories
    from benchmarks.benchmark_utils import get_model_sizes
    size_categories = get_model_sizes()

    # Group by model size
    by_size = {}
    for r in results:
        if r.error:
            continue
        if r.model_size not in by_size:
            by_size[r.model_size] = {}
        by_size[r.model_size][r.method] = r

    # Create table with methods as columns
    latex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Matrix Exponential Performance Comparison}",
        r"\label{tab:matrix_expm}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Full $\exp(Q)$} & \multicolumn{3}{c}{Columns of $\exp(Q)$} & \multicolumn{2}{c}{Difference} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-9}",
        r"Model & Dense & Sparse & Speedup & Dense & Sparse & Speedup & Absolute & Relative \\",
        r"\midrule"
    ]

    # Process each size category
    first_category = True
    for cat_name in CATEGORY_NAMES:
        if cat_name not in size_categories:
            continue

        # Filter sizes for this category
        cat_sizes = [(size, by_size[size]) for size in by_size.keys()
                     if size in size_categories[cat_name] and size in by_size]

        if not cat_sizes:
            continue

        # Add category header
        if not first_category:
            latex.append(r"\addlinespace")
        latex.append(rf"\multicolumn{{9}}{{l}}{{\textit{{{CATEGORY_LABELS[cat_name]} Models}}}} \\")
        first_category = False

        # Process each model size in the category
        for size, size_data in sorted(cat_sizes):
            n_players, n_demand = size

            if 'dense' not in size_data or 'sparse' not in size_data:
                continue

            dense = size_data['dense']
            sparse = size_data['sparse']

            # Calculate speedups (dense time / sparse time)
            full_speedup = dense.full_expm_stats.median / sparse.full_expm_stats.median if sparse.full_expm_stats.median > 0 else 0
            columns_speedup = dense.columns_expm_stats.median / sparse.columns_expm_stats.median if sparse.columns_expm_stats.median > 0 else 0

            # Convert to milliseconds for display
            dense_full_ms = dense.full_expm_stats.median * 1000
            dense_columns_ms = dense.columns_expm_stats.median * 1000
            sparse_full_ms = sparse.full_expm_stats.median * 1000
            sparse_columns_ms = sparse.columns_expm_stats.median * 1000

            # Format row
            row = f"${n_players} \\times {n_demand}$ & "

            # Full expm columns
            row += f"{dense_full_ms:,.2f} & {sparse_full_ms:,.2f} & {full_speedup:.2f}$\\times$ & "

            # Columns expm columns
            row += f"{dense_columns_ms:,.2f} & {sparse_columns_ms:,.2f} & {columns_speedup:.2f}$\\times$ & "

            # Error columns
            row += f"{sparse.max_absolute_error:.1e} & {sparse.max_relative_error:.1e} \\\\"

            latex.append(row)

    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1ex}",
        r"\footnotesize",
        r"Note: Full = complete matrix exponential $\exp(Q)$, Columns = $\min\lbrace 200, K \rbrace$ columns of $\exp(Q)$,",
        r"Times in milliseconds. Speedup = Dense time / Sparse time ($>1$ means Sparse is faster).",
        r"\end{table}"
    ])

    return "\n".join(latex)


# ==============================================================================
# Figure Generation Functions
# ==============================================================================

def create_computation_speedup_figure(results: List[ComputationBenchmarkResult], output_path: str):
    """Create figure showing computation speedups."""
    # Apply academic style
    setup_plot_style()

    # Group by model size
    by_size = {}
    for r in results:
        if r.error:
            continue
        if r.model_size not in by_size:
            by_size[r.model_size] = []
        by_size[r.model_size].append(r)

    # Get model sizes grouped by size category
    from benchmarks.benchmark_utils import create_benchmark_config
    config = create_benchmark_config()
    size_groups = list(config.model_sizes.items())

    # Configuration names
    config_names = ["baseline", "vectorize", "polyalgorithm", "cython"]
    n_configs = len(config_names)

    # Collect speedup data for each size group and configuration
    speedup_data = {config: [] for config in config_names}
    size_labels = []

    for size_name, sizes in size_groups:
        # Collect all speedups for this size group
        group_speedups = {config: [] for config in config_names}

        for size in sizes:
            if size not in by_size:
                continue

            # Get baseline time for this specific size
            baseline_time = None
            for r in by_size[size]:
                if (not r.config.vectorize and
                    not r.config.polyalgorithm and not r.config.cython and
                    not r.config.sparse and not r.config.derivatives):
                    baseline_time = r.total_stats.median
                    break

            if baseline_time is None:
                continue

            # Calculate speedups for each configuration for this size
            for r in by_size[size]:
                speedup = baseline_time / r.total_stats.median if r.total_stats.median > 0 else 0

                if not r.config.vectorize:
                    group_speedups["baseline"].append(speedup)
                elif r.config.vectorize and not r.config.polyalgorithm:
                    group_speedups["vectorize"].append(speedup)
                elif r.config.polyalgorithm and not r.config.cython:
                    group_speedups["polyalgorithm"].append(speedup)
                elif r.config.cython and not r.config.sparse:
                    group_speedups["cython"].append(speedup)

        # Calculate average speedups for this size group
        if any(group_speedups[config] for config in config_names):
            size_labels.append(size_name.replace('_', ' ').title())
            for config in config_names:
                if group_speedups[config]:
                    avg_speedup = np.mean(group_speedups[config])
                else:
                    avg_speedup = 0.0
                speedup_data[config].append(avg_speedup)

    # Also collect timing data for the left panel
    timing_data = {config: [] for config in config_names}

    for size_name, sizes in size_groups:
        # Collect all times for this size group
        group_times = {config: [] for config in config_names}

        for size in sizes:
            if size not in by_size:
                continue

            # Calculate times for each configuration for this size
            for r in by_size[size]:
                time_val = r.total_stats.median

                if not r.config.vectorize:
                    group_times["baseline"].append(time_val)
                elif r.config.vectorize and not r.config.polyalgorithm:
                    group_times["vectorize"].append(time_val)
                elif r.config.polyalgorithm and not r.config.cython:
                    group_times["polyalgorithm"].append(time_val)
                elif r.config.cython and not r.config.sparse:
                    group_times["cython"].append(time_val)

        # Calculate average times for this size group
        for config in config_names:
            if group_times[config]:
                avg_time = np.mean(group_times[config])
            else:
                avg_time = 0.0
            timing_data[config].append(avg_time)

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Set up grouped bar chart parameters
    x = np.arange(len(size_labels))
    width = 0.2  # Width of individual bars

    # Get grayscale colors (no hatching)
    colors = ['0.9', '0.6', '0.4', '0.2']  # Light to dark gray

    # Panel 1: Computation times
    for i, (config, color) in enumerate(zip(config_names, colors)):
        offset = (i - n_configs/2 + 0.5) * width
        bars1 = ax1.bar(x + offset, timing_data[config], width,
                        label=config.capitalize(),
                        color=color,
                        edgecolor='black',
                        linewidth=1.0)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                # Format: no decimals if >= 1, keep decimal if < 1
                if height <= 0.1:
                    label = f'{height:.2f}s'
                elif height <= 1:
                    label = f'{height:.1f}s'
                else:
                    label = f'{height:.0f}s'
                ax1.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                         label, ha='center', va='bottom',
                         fontsize=7, color='black')

    # Customize panel 1
    ax1.set_xlabel('Model Size', fontsize=12)
    ax1.set_ylabel('Average Time in Seconds (Log Scale)', fontsize=12)
    ax1.set_title('Computation Time', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(size_labels)
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3, linestyle=':')

    # Panel 2: Speedups
    for i, (config, color) in enumerate(zip(config_names, colors)):
        offset = (i - n_configs/2 + 0.5) * width
        bars2 = ax2.bar(x + offset, speedup_data[config], width,
                        label=config.capitalize(),
                        color=color,
                        edgecolor='black',
                        linewidth=1.0)

        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                         f'{height:.0f}×', ha='center', va='bottom',
                         fontsize=7, color='black')

    # Customize panel 2
    ax2.set_xlabel('Model Size', fontsize=12)
    ax2.set_ylabel('Speedup vs. Baseline', fontsize=12)
    ax2.set_title('Computation Speedup', fontsize=12)
    ax2.set_xticks(x)
    # ax2.set_yscale('log')
    ax2.set_xticklabels(size_labels)
    ax2.grid(axis='y', alpha=0.3, linestyle=':')

    # Add legend to the first panel
    ax1.legend(loc='upper left', frameon=True, fancybox=False)

    # Set main title
    fig.suptitle('Computation Performance (Relative to Sequential Baseline)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, Path(output_path).with_suffix(''))
    plt.close()


def create_estimation_comparison_figure(results: List[EstimationBenchmarkResult], output_path: str):
    """Create figure comparing estimation performance."""
    # Apply academic style
    setup_plot_style()

    # Group by model size and config
    by_size_config = {}
    for r in results:
        if r.error:
            continue
        key = (r.model_size, r.config)
        if key not in by_size_config:
            by_size_config[key] = []
        by_size_config[key].append(r)

    # Get model sizes grouped by size category
    from benchmarks.benchmark_utils import create_benchmark_config
    config = create_benchmark_config()
    size_groups = list(config.model_sizes.items())

    # Configuration names - these are the actual estimation configs
    config_names = ["cython", "sparse", "derivatives"]
    n_configs = len(config_names)

    # Collect timing and speedup data for each size group and configuration
    timing_data = {config: [] for config in config_names}
    speedup_data = {config: [] for config in config_names}
    size_labels = []

    for size_name, sizes in size_groups:
        # Collect all times for this size group
        group_times = {config: [] for config in config_names}

        # First pass: collect times for each config
        for size in sizes:
            for config_name in config_names:
                key = (size, config_name)
                if key in by_size_config:
                    # Take the median time from all runs for this size/config combination
                    times = [r.estimation_stats.median for r in by_size_config[key]]
                    if times:
                        group_times[config_name].extend(times)

        # Check if we have data for this size group
        if any(group_times[config] for config in config_names):
            size_labels.append(size_name.replace('_', ' ').title())

            # Calculate average times and speedups for this size group
            baseline_time = None
            group_avg_times = {}

            for config_name in config_names:
                if group_times[config_name]:
                    avg_time = np.mean(group_times[config_name])
                    group_avg_times[config_name] = avg_time

                    # Use first config as baseline
                    if baseline_time is None:
                        baseline_time = avg_time
                else:
                    group_avg_times[config_name] = 0.0

            # Add timing data and calculate speedups
            for config_name in config_names:
                timing_data[config_name].append(group_avg_times[config_name])

                if group_avg_times[config_name] > 0 and baseline_time > 0:
                    speedup = baseline_time / group_avg_times[config_name]
                else:
                    speedup = 0.0
                speedup_data[config_name].append(speedup)

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Set up grouped bar chart parameters
    x = np.arange(len(size_labels))
    width = 0.25  # Width of individual bars

    # Get grayscale colors (no hatching)
    colors = ['0.6', '0.4', '0.2']  # Medium to dark gray for 3 configs

    # Panel 1: Estimation times
    for i, (config, color) in enumerate(zip(config_names, colors)):
        offset = (i - n_configs/2 + 0.5) * width
        bars1 = ax1.bar(x + offset, timing_data[config], width,
                        label=config.capitalize(),
                        color=color,
                        edgecolor='black',
                        linewidth=1.0)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                # Format: no decimals if >= 1, keep decimal if < 1
                label = f'{height:.0f}s' if height >= 1 else f'{height:.1f}s'
                ax1.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                         label, ha='center', va='bottom',
                         fontsize=7, color='black')

    # Customize panel 1
    ax1.set_xlabel('Model Size', fontsize=12)
    ax1.set_ylabel('Average Time in Seconds (Log Scale)', fontsize=12)
    ax1.set_title('Estimation Time', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(size_labels)
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3, linestyle=':')

    # Panel 2: Speedups
    for i, (config, color) in enumerate(zip(config_names, colors)):
        offset = (i - n_configs/2 + 0.5) * width
        bars2 = ax2.bar(x + offset, speedup_data[config], width,
                        label=config.capitalize(),
                        color=color,
                        edgecolor='black',
                        linewidth=1.0)

        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                         f'{height:.1f}×', ha='center', va='bottom',
                         fontsize=7, color='black')

    # Customize panel 2
    ax2.set_xlabel('Model Size', fontsize=12)
    ax2.set_ylabel('Speedup vs. Optimized Cython', fontsize=12)
    ax2.set_title('Estimation Speedup', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(size_labels)
    ax2.grid(axis='y', alpha=0.3, linestyle=':')

    # Add legend to the first panel
    ax1.legend(loc='upper left', frameon=True, fancybox=False)

    # Set main title
    fig.suptitle('Estimation Performance (Relative to Optimized Cython)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, Path(output_path).with_suffix(''))
    plt.close()


def create_expm_figures(results: List[ExpmBenchmarkResult], output_dir: Path):
    """Create matrix exponential performance figures."""
    # Apply academic style
    setup_plot_style()

    # Group by model size and method
    by_size_method = {}
    for r in results:
        if r.error:
            continue
        key = (r.model_size, r.method)
        if key not in by_size_method:
            by_size_method[key] = []
        by_size_method[key].append(r)

    # Get model sizes grouped by size category
    from benchmarks.benchmark_utils import create_benchmark_config
    config = create_benchmark_config()
    size_groups = list(config.model_sizes.items())

    # Method names
    method_names = ["dense", "sparse"]
    n_methods = len(method_names)

    # Collect timing and speedup data for each size group and method
    full_timing_data = {method: [] for method in method_names}
    full_speedup_data = {method: [] for method in method_names}
    columns_timing_data = {method: [] for method in method_names}
    columns_speedup_data = {method: [] for method in method_names}
    size_labels = []

    for size_name, sizes in size_groups:
        # Collect all times for this size group
        group_full_times = {method: [] for method in method_names}
        group_columns_times = {method: [] for method in method_names}

        # First pass: collect times for each method
        for size in sizes:
            for method_name in method_names:
                key = (size, method_name)
                if key in by_size_method:
                    # Take the median time from all runs for this size/method combination
                    full_times = [r.full_expm_stats.median * 1000 for r in by_size_method[key]]  # Convert to ms
                    columns_times = [r.columns_expm_stats.median * 1000 for r in by_size_method[key]]
                    if full_times:
                        group_full_times[method_name].extend(full_times)
                        group_columns_times[method_name].extend(columns_times)

        # Check if we have data for this size group
        if any(group_full_times[method] for method in method_names):
            size_labels.append(size_name.replace('_', ' ').title())

            # Calculate average times and speedups for this size group
            dense_full_time = None
            dense_columns_time = None
            group_avg_full_times = {}
            group_avg_columns_times = {}

            for method_name in method_names:
                if group_full_times[method_name]:
                    avg_full_time = np.mean(group_full_times[method_name])
                    avg_columns_time = np.mean(group_columns_times[method_name])
                    group_avg_full_times[method_name] = avg_full_time
                    group_avg_columns_times[method_name] = avg_columns_time

                    # Use dense as baseline
                    if method_name == "dense":
                        dense_full_time = avg_full_time
                        dense_columns_time = avg_columns_time
                else:
                    group_avg_full_times[method_name] = 0.0
                    group_avg_columns_times[method_name] = 0.0

            # Add timing data and calculate speedups
            for method_name in method_names:
                full_timing_data[method_name].append(group_avg_full_times[method_name])
                columns_timing_data[method_name].append(group_avg_columns_times[method_name])

                if group_avg_full_times[method_name] > 0 and dense_full_time > 0:
                    full_speedup = dense_full_time / group_avg_full_times[method_name]
                else:
                    full_speedup = 0.0
                full_speedup_data[method_name].append(full_speedup)

                if group_avg_columns_times[method_name] > 0 and dense_columns_time > 0:
                    columns_speedup = dense_columns_time / group_avg_columns_times[method_name]
                else:
                    columns_speedup = 0.0
                columns_speedup_data[method_name].append(columns_speedup)

    if not size_labels:
        return

    # Create full expm figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Set up grouped bar chart parameters
    x = np.arange(len(size_labels))
    width = 0.35  # Width of individual bars

    # Get grayscale colors (no hatching)
    colors = ['0.6', '0.3']  # Medium and dark gray for 2 methods

    # Panel 1: Full expm times
    for i, (method, color) in enumerate(zip(method_names, colors)):
        offset = (i - n_methods/2 + 0.5) * width
        bars1 = ax1.bar(x + offset, full_timing_data[method], width,
                        label=f'{method.capitalize()} (scipy.expm)' if method == 'dense' else f'{method.capitalize()} (vexpm)',
                        color=color,
                        edgecolor='black',
                        linewidth=1.0)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                # Format: seconds if > 1000ms, otherwise milliseconds
                if height > 1000:
                    label = f'{height/1000:.1f}s'
                elif height <= 0.1:
                    label = f'{height:.2f}ms'
                elif height <= 1:
                    label = f'{height:.1f}ms'
                else:
                    label = f'{height:.0f}ms'
                ax1.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                         label, ha='center', va='bottom',
                         fontsize=7, color='black')

    # Customize panel 1
    ax1.set_xlabel('Model Size', fontsize=12)
    ax1.set_ylabel('Average Time in Milliseconds (Log Scale)', fontsize=12)
    ax1.set_title('Full Matrix Exponential Time', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(size_labels)
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3, linestyle=':')

    # Panel 2: Full expm speedups
    for i, (method, color) in enumerate(zip(method_names, colors)):
        offset = (i - n_methods/2 + 0.5) * width
        bars2 = ax2.bar(x + offset, full_speedup_data[method], width,
                        label=f'{method.capitalize()}',
                        color=color,
                        edgecolor='black',
                        linewidth=1.0)

        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{height:.1f}×', ha='center', va='bottom',
                         fontsize=7, color='black')

    # Customize panel 2
    ax2.set_xlabel('Model Size', fontsize=12)
    ax2.set_ylabel('Speedup vs. Dense', fontsize=12)
    ax2.set_title('Full Matrix Exponential Speedup', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(size_labels)
    ax2.grid(axis='y', alpha=0.3, linestyle=':')

    # Add legend to the first panel
    ax1.legend(loc='upper left', frameon=True, fancybox=False)

    # Set main title
    fig.suptitle('Full Matrix Exponential Performance', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, output_dir / 'matrix_expm_full')
    plt.close()

    # Create columns expm figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Columns expm times
    for i, (method, color) in enumerate(zip(method_names, colors)):
        offset = (i - n_methods/2 + 0.5) * width
        bars1 = ax1.bar(x + offset, columns_timing_data[method], width,
                        label=f'{method.capitalize()} (scipy.expm)' if method == 'dense' else f'{method.capitalize()} (vexpm)',
                        color=color,
                        edgecolor='black',
                        linewidth=1.0)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                # Format: seconds if > 1000ms, otherwise milliseconds
                if height > 1000:
                    label = f'{height/1000:.1f}s'
                elif height <= 0.1:
                    label = f'{height:.2f}ms'
                elif height <= 1:
                    label = f'{height:.1f}ms'
                else:
                    label = f'{height:.0f}ms'
                ax1.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                         label, ha='center', va='bottom',
                         fontsize=7, color='black')

    # Customize panel 1
    ax1.set_xlabel('Model Size', fontsize=12)
    ax1.set_ylabel('Average Time in Milliseconds (Log Scale)', fontsize=12)
    ax1.set_title('Columns Matrix Exponential Time', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(size_labels)
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3, linestyle=':')

    # Panel 2: Columns expm speedups
    for i, (method, color) in enumerate(zip(method_names, colors)):
        offset = (i - n_methods/2 + 0.5) * width
        bars2 = ax2.bar(x + offset, columns_speedup_data[method], width,
                        label=f'{method.capitalize()}',
                        color=color,
                        edgecolor='black',
                        linewidth=1.0)

        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                         f'{height:.1f}×', ha='center', va='bottom',
                         fontsize=7, color='black')

    # Customize panel 2
    ax2.set_xlabel('Model Size', fontsize=12)
    ax2.set_ylabel('Speedup vs. Dense', fontsize=12)
    ax2.set_title('Columns Matrix Exponential Speedup', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(size_labels)
    ax2.grid(axis='y', alpha=0.3, linestyle=':')

    # Add legend to the first panel
    ax1.legend(loc='upper left', frameon=True, fancybox=False)

    # Set main title
    fig.suptitle('Columns Matrix Exponential Performance', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, output_dir / 'matrix_expm_columns')
    plt.close()


# ==============================================================================
# Result Management
# ==============================================================================

def load_results_from_json(input_dir: Path) -> Dict[str, List[Any]]:
    """Load previously saved results from JSON files."""
    results = {
        'computation': [],
        'estimation': [],
        'sparsity': [],
        'expm': []
    }

    # Load computation results
    compute_file = input_dir / 'computation_results.json'
    if compute_file.exists():
        with open(compute_file, 'r') as f:
            compute_data = json.load(f)

        for result_dict in compute_data['results']:
            # Reconstruct OptimizationConfig from the result
            config = OptimizationConfig(
                vectorize=result_dict.get('vectorize', False),
                polyalgorithm=result_dict.get('polyalgorithm', False),
                cython=result_dict.get('cython', False),
                sparse=result_dict.get('sparse', False),
                derivatives=result_dict.get('derivatives', False)
            )

            # Create TimingStats objects from the saved data
            def create_timing_stats(prefix):
                return TimingStats(
                    times=result_dict[f'{prefix}times'],
                    mean=result_dict[f'{prefix}mean'],
                    median=result_dict[f'{prefix}median'],
                    std=result_dict[f'{prefix}std'],
                    min=result_dict[f'{prefix}min'],
                    max=result_dict[f'{prefix}max'],
                    n_runs=result_dict[f'{prefix}n_runs']
                )

            result = ComputationBenchmarkResult(
                config=config,
                model_size=(result_dict['n_players'], result_dict['n_demand']),
                n_states=result_dict['n_states'],
                setup_stats=create_timing_stats('setup_'),
                value_function_stats=create_timing_stats('value_function_'),
                choice_prob_stats=create_timing_stats('choice_prob_'),
                total_stats=create_timing_stats('total_'),
                vf_iterations=result_dict['vf_iterations'],
                error=result_dict.get('error')
            )
            results['computation'].append(result)

    # Load estimation results
    estimate_file = input_dir / 'estimation_results.json'
    if estimate_file.exists():
        with open(estimate_file, 'r') as f:
            estimate_data = json.load(f)

        for result_dict in estimate_data['results']:
            result = EstimationBenchmarkResult(
                config=result_dict['config'],
                model_size=(result_dict['n_players'], result_dict['n_demand']),
                n_states=result_dict['n_states'],
                n_obs=result_dict['n_obs'],
                estimation_stats=create_timing_stats('estimation_'),
                n_iterations=result_dict['n_iterations'],
                n_likelihood_evals=result_dict['n_likelihood_evals'],
                final_likelihood=result_dict['final_likelihood'],
                converged=result_dict['converged'],
                error=result_dict.get('error')
            )
            results['estimation'].append(result)

    # Load sparsity results
    sparsity_file = input_dir / 'sparsity_results.json'
    if sparsity_file.exists():
        with open(sparsity_file, 'r') as f:
            sparsity_data = json.load(f)

        for result_dict in sparsity_data['results']:
            result = SparsityBenchmarkResult(
                model_size=(result_dict['n_players'], result_dict['n_demand']),
                n_states=result_dict['n_states'],
                q_matrix_size=result_dict['q_matrix_size'],
                q_matrix_nonzeros=result_dict['q_matrix_nonzeros'],
                q_matrix_sparsity=result_dict['q_matrix_sparsity'],
                dt_dv_matrix_size=result_dict['dt_dv_matrix_size'],
                dt_dv_matrix_nonzeros=result_dict['dt_dv_matrix_nonzeros'],
                dt_dv_matrix_sparsity=result_dict['dt_dv_matrix_sparsity'],
                computation_time=result_dict['computation_time'],
                error=result_dict.get('error')
            )
            results['sparsity'].append(result)

    # Load matrix exponential results
    expm_file = input_dir / 'expm_results.json'
    if expm_file.exists():
        with open(expm_file, 'r') as f:
            expm_data = json.load(f)

        for result_dict in expm_data['results']:
            # Create TimingStats objects from the saved data
            def create_timing_stats(prefix):
                return TimingStats(
                    times=result_dict[f'{prefix}times'],
                    mean=result_dict[f'{prefix}mean'],
                    median=result_dict[f'{prefix}median'],
                    std=result_dict[f'{prefix}std'],
                    min=result_dict[f'{prefix}min'],
                    max=result_dict[f'{prefix}max'],
                    n_runs=result_dict[f'{prefix}n_runs']
                )

            result = ExpmBenchmarkResult(
                model_size=(result_dict['n_players'], result_dict['n_demand']),
                n_states=result_dict['n_states'],
                method=result_dict['method'],
                setup_stats=create_timing_stats('setup_'),
                full_expm_stats=create_timing_stats('full_expm_'),
                columns_expm_stats=create_timing_stats('columns_expm_'),
                max_absolute_error=result_dict['max_absolute_error'],
                max_relative_error=result_dict['max_relative_error'],
                n_columns=result_dict['n_columns'],
                error=result_dict.get('error')
            )
            results['expm'].append(result)

    return results


def save_all_results(
    results: Dict[str, List[Any]],
    output_dir: Path,
    model_sizes: Dict[str, List[Tuple[int, int]]],
    save_raw_data: bool = True
):
    """Save all benchmark results to files."""
    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Save raw results as JSON (only if requested)
    if save_raw_data:
        for benchmark_type, benchmark_results in results.items():
            if benchmark_results:
                data = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'system_info': get_system_info(),
                    'results': [r.to_dict() for r in benchmark_results]
                }
                with open(output_dir / f'{benchmark_type}_results.json', 'w') as f:
                    json.dump(data, f, indent=2)

    # Generate LaTeX tables
    if 'computation' in results and results['computation']:
        latex_table = create_computation_latex_table(results['computation'])
        with open(output_dir / 'tab_computation.tex', 'w') as f:
            f.write(latex_table)

    if 'estimation' in results and results['estimation']:
        latex_table = create_estimation_latex_table(results['estimation'])
        with open(output_dir / 'tab_estimation.tex', 'w') as f:
            f.write(latex_table)

    if 'sparsity' in results and results['sparsity']:
        latex_table = create_sparsity_latex_table(results['sparsity'])
        with open(output_dir / 'tab_sparsity.tex', 'w') as f:
            f.write(latex_table)

    if 'expm' in results and results['expm']:
        latex_table = create_expm_latex_table(results['expm'])
        with open(output_dir / 'tab_matrix_expm.tex', 'w') as f:
            f.write(latex_table)

    # Generate figures
    if 'computation' in results and results['computation']:
        try:
            create_computation_speedup_figure(results['computation'], str(output_dir / 'computation_comparison.png'))
            create_computation_speedup_figure(results['computation'], str(output_dir / 'computation_comparison.pdf'))
        except Exception as e:
            print(f"Warning: Could not generate computation speedup figure: {e}")

    if 'estimation' in results and results['estimation']:
        try:
            create_estimation_comparison_figure(results['estimation'], str(output_dir / 'estimation_comparison.png'))
            create_estimation_comparison_figure(results['estimation'], str(output_dir / 'estimation_comparison.pdf'))
        except Exception as e:
            print(f"Warning: Could not generate estimation comparison figure: {e}")

    if 'expm' in results and results['expm']:
        try:
            create_expm_figures(results['expm'], output_dir)
        except Exception as e:
            print(f"Warning: Could not generate matrix exponential figures: {e}")

    print(f"\nResults saved to {output_dir}/")


def print_summary(results: Dict[str, List[Any]]):
    """Print summary of all benchmark results."""
    print_header('Summary')

    for benchmark_type, benchmark_results in results.items():
        if not benchmark_results:
            continue

        print_header(benchmark_type.title() + ' Performance', '-')

        if benchmark_type == 'computation':
            _print_computation_summary(benchmark_results)
        elif benchmark_type == 'estimation':
            _print_estimation_summary(benchmark_results)
        elif benchmark_type == 'sparsity':
            _print_sparsity_summary(benchmark_results)
        elif benchmark_type == 'expm':
            _print_expm_summary(benchmark_results)


def _print_computation_summary(results: List[ComputationBenchmarkResult]):
    """Print computation benchmark summary."""
    # Find best speedups
    by_size = {}
    for r in results:
        if r.error:
            continue
        if r.model_size not in by_size:
            by_size[r.model_size] = {'baseline': None, 'best': None}

        # Use total_stats.median for timing
        timing = r.total_stats.median

        if (not r.config.vectorize and
            not r.config.polyalgorithm and not r.config.cython and
            not r.config.sparse and not r.config.derivatives):
            by_size[r.model_size]['baseline'] = timing
        elif r.config.cython and not r.config.sparse:
            by_size[r.model_size]['best'] = timing

    for size in sorted(by_size.keys()):
        n_players, n_demand = size
        n_states = (2 ** n_players) * n_demand
        baseline = by_size[size]['baseline']
        best = by_size[size]['best']

        if baseline and best:
            speedup = baseline / best
            print(f"{n_players}×{n_demand} ({n_states:4d} states): "
                  f"{baseline:7.3f}s → {best:7.3f}s "
                  f"(speedup: {speedup:5.1f}×)")


def _print_estimation_summary(results: List[EstimationBenchmarkResult]):
    """Print estimation benchmark summary."""
    # Group by model size
    by_size = {}
    for r in results:
        if r.error:
            continue
        if r.model_size not in by_size:
            by_size[r.model_size] = []
        by_size[r.model_size].append(r)

    for size in sorted(by_size.keys()):
        n_players, n_demand = size
        results_for_size = by_size[size]

        # Find the slowest and fastest configs for this size
        # Estimation only runs cython, sparse, derivatives
        baseline = next((r for r in results_for_size if str(r.config) == "cython"), None)
        best = next((r for r in results_for_size if str(r.config) == "derivatives"), None)

        if baseline and best:
            baseline_time = baseline.estimation_stats.median
            best_time = best.estimation_stats.median
            speedup = baseline_time / best_time
            print(f"{n_players}×{n_demand}: "
                  f"{baseline_time:6.2f}s → {best_time:6.2f}s "
                  f"(speedup: {speedup:4.1f}×)")


def _print_sparsity_summary(results: List[SparsityBenchmarkResult]):
    """Print sparsity benchmark summary."""
    for r in sorted(results, key=lambda x: x.n_states):
        if r.error:
            continue
        n_players, n_demand = r.model_size
        print(f"{n_players}×{n_demand} ({r.n_states:4d} states): "
              f"Q {r.q_matrix_sparsity:.1%} sparse, "
              f"∂T/∂V {r.dt_dv_matrix_sparsity:.1%} sparse "
              f"({r.computation_time:.3f}s)")


def _print_expm_summary(results: List[ExpmBenchmarkResult]):
    """Print matrix exponential benchmark summary."""
    # Group by model size
    by_size = {}
    for r in results:
        if r.error:
            continue
        if r.model_size not in by_size:
            by_size[r.model_size] = {}
        by_size[r.model_size][r.method] = r

    print(f"{'Model':<9} {'States':<8} {'Dense(ms)':<12} {'Sparse(ms)':<12} {'Speedup':<10} {'Max Error':<12}")
    print("-" * 68)

    for size in sorted(by_size.keys()):
        if 'dense' not in by_size[size] or 'sparse' not in by_size[size]:
            continue

        n_players, n_demand = size
        n_states = (2 ** n_players) * n_demand
        dense = by_size[size]['dense']
        sparse = by_size[size]['sparse']

        speedup = dense.columns_expm_stats.median / sparse.columns_expm_stats.median if sparse.columns_expm_stats.median > 0 else 0

        print(f"{n_players}×{n_demand:<7} {n_states:<8} "
              f"{dense.columns_expm_stats.median:<12.2f} "
              f"{sparse.columns_expm_stats.median:<12.2f} "
              f"{speedup:<10.2f} "
              f"{sparse.max_absolute_error:<12.2e}")
