import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from pathlib import Path


def calculate_robust_global_scale(
        all_data, include_reference_lines=None, reference_values=None):
    """
    Calculate global scale for box plots that handles extreme outliers.

    This function is designed to work with parameter-grouped data where each
    parameter should have its own whisker calculation, then the global scale
    is determined from the union of all individual parameter extents.
    """
    if len(all_data) == 0:
        return None

    # Calculate whisker extents for each parameter separately
    all_extents = []

    for param_idx, param_data in enumerate(all_data):
        if len(param_data) == 0:
            continue

        param_array = np.array(param_data)

        # Only filter truly extreme outliers (orders of magnitude different).
        # Check if any values are more than 10x the IQR away from median.
        q1 = np.percentile(param_array, 25)
        q3 = np.percentile(param_array, 75)
        iqr = q3 - q1

        if iqr > 0:
            median = np.median(param_array)
            very_extreme_threshold = 10 * iqr
            ex_large = param_array > (median + very_extreme_threshold)
            ex_small = param_array < (median - very_extreme_threshold)

            # Only filter if we have extreme outliers
            if np.any(ex_large) or np.any(ex_small):
                reasonable_data = param_array[~(ex_large | ex_small)]
                # If at least 50% of data remains
                if len(reasonable_data) > len(param_array) * 0.5:
                    param_array = reasonable_data

        # Calculate whisker extents for this parameter using all remaining data
        if len(param_array) == 0:
            continue

        # Calculate quartiles on the (possibly filtered) data
        q1_filtered = np.percentile(param_array, 25)
        q3_filtered = np.percentile(param_array, 75)
        iqr_filtered = q3_filtered - q1_filtered

        # Standard boxplot whisker bounds for this parameter
        lower_whisker_bound = q1_filtered - 1.5 * iqr_filtered
        upper_whisker_bound = q3_filtered + 1.5 * iqr_filtered

        # Find actual whisker extents and normal outliers for this parameter
        whisker_data = param_array[(param_array >= lower_whisker_bound)
                                   & (param_array <= upper_whisker_bound)]
        normal_outliers = param_array[(param_array < lower_whisker_bound)
                                      | (param_array > upper_whisker_bound)]

        # Determine bounds that include whiskers and normal outliers
        if len(whisker_data) > 0:
            min_extent = np.min(whisker_data)
            max_extent = np.max(whisker_data)
        else:
            min_extent = q1_filtered
            max_extent = q3_filtered

        if len(normal_outliers) > 0:
            min_extent = min(min_extent, np.min(normal_outliers))
            max_extent = max(max_extent, np.max(normal_outliers))

        all_extents.extend([min_extent, max_extent])

    if len(all_extents) == 0:
        return None

    # Global bounds from union of all parameter extents
    global_min = min(all_extents)
    global_max = max(all_extents)

    # Add padding
    data_range = global_max - global_min
    if data_range == 0:
        data_range = max(0.1, abs((global_min + global_max) / 2) * 0.1)

    # Scale-aware padding: relative for small ranges, absolute for normal
    if data_range < 1e-3:
        # Very small range (precision-level data)
        annotation_space = 0.3 * data_range  # 30% of range for annotations
        plot_padding = 0.1 * data_range      # 10% of range for padding
    else:
        # Normal range
        annotation_space = max(0.02, 0.08 * data_range)
        plot_padding = max(0.01, 0.02 * data_range)

    lower_bound = global_min - annotation_space - plot_padding
    upper_bound = global_max + annotation_space + plot_padding

    # Include reference values if specified
    if include_reference_lines and reference_values:
        for ref_val in reference_values:
            lower_bound = min(lower_bound, ref_val - 0.05 * data_range)
            upper_bound = max(upper_bound, ref_val + 0.05 * data_range)

    return (lower_bound, upper_bound)


class MonteCarloResults:

    def __init__(self, param_keys, param_latex=None):
        self.param_keys = param_keys
        self.param_latex = param_latex or {key: key for key in param_keys}
        self.results = {}
        self.metadata = {
            'param_keys': param_keys,
            'param_latex': self.param_latex
        }

    def store_experiment(self, experiment_type, sample_size, results, theta_true):
        key = (experiment_type, sample_size)

        # Extract parameter estimates
        estimates = np.array([result.x for result in results])

        # Extract computational statistics
        wall_times = np.array([getattr(result, 'wall_time', np.nan) for result in results])
        iterations = np.array([result.nit for result in results])
        function_evals = np.array([result.nfev for result in results])
        log_likelihoods = np.array([-result.fun for result in results])
        success_flags = np.array([result.success for result in results])

        self.results[key] = {
            'estimates': estimates,
            'theta_true': np.array(theta_true),
            'wall_times': wall_times,
            'iterations': iterations,
            'function_evals': function_evals,
            'log_likelihoods': log_likelihoods,
            'success_flags': success_flags,
            'n_replications': len(results)
        }

    def get_summary_statistics(self, experiment_type, sample_size):
        key = (experiment_type, sample_size)
        if key not in self.results:
            raise ValueError(f"No results found for {experiment_type} with {sample_size} observations")

        data = self.results[key]
        estimates = data['estimates']
        theta_true = data['theta_true']
        success_flags = data['success_flags']
        # Filter to only successful replications for bias/RMSE calculations
        successful_estimates = estimates[success_flags]

        stats = {}
        for i, param in enumerate(self.param_keys):
            true_value = theta_true[i]

            # Statistics on successful replications only
            if len(successful_estimates) > 0:
                param_estimates_success = successful_estimates[:, i]
                stats[param] = {
                    'true_value': true_value,
                    'mean': np.mean(param_estimates_success),
                    'median': np.median(param_estimates_success),
                    'std': np.std(param_estimates_success),
                    'rmse': np.sqrt(np.mean((param_estimates_success - true_value)**2)),
                    'mean_bias': np.mean(param_estimates_success - true_value),
                    'median_bias': np.median(param_estimates_success - true_value),
                    'n_successful': len(param_estimates_success)
                }
            else:
                # No successful replications
                stats[param] = {
                    'true_value': true_value,
                    'mean': np.nan,
                    'median': np.nan,
                    'std': np.nan,
                    'rmse': np.nan,
                    'mean_bias': np.nan,
                    'median_bias': np.nan,
                    'n_successful': 0
                }

        # Add computational statistics (filtered to successful runs)
        if len(successful_estimates) > 0:
            successful_wall_times = data['wall_times'][success_flags]
            successful_iterations = data['iterations'][success_flags]
            successful_function_evals = data['function_evals'][success_flags]
            successful_log_likelihoods = data['log_likelihoods'][success_flags]

            stats['computational'] = {
                'mean_wall_time': np.nanmean(successful_wall_times),
                'median_wall_time': np.nanmedian(successful_wall_times),
                'std_wall_time': np.nanstd(successful_wall_times),
                'mean_iterations': np.mean(successful_iterations),
                'median_iterations': np.median(successful_iterations),
                'std_iterations': np.std(successful_iterations),
                'mean_function_evals': np.mean(successful_function_evals),
                'median_function_evals': np.median(successful_function_evals),
                'std_function_evals': np.std(successful_function_evals),
                'mean_log_likelihood': np.mean(successful_log_likelihoods),
                'median_log_likelihood': np.median(successful_log_likelihoods),
                'std_log_likelihood': np.std(successful_log_likelihoods),
                'success_rate': np.mean(success_flags),
                'n_successful': len(successful_estimates)
            }
        else:
            # No successful replications
            stats['computational'] = {
                'mean_wall_time': np.nan,
                'median_wall_time': np.nan,
                'std_wall_time': np.nan,
                'mean_iterations': np.nan,
                'median_iterations': np.nan,
                'std_iterations': np.nan,
                'mean_function_evals': np.nan,
                'median_function_evals': np.nan,
                'std_function_evals': np.nan,
                'mean_log_likelihood': np.nan,
                'median_log_likelihood': np.nan,
                'std_log_likelihood': np.nan,
                'success_rate': np.mean(success_flags),
                'n_successful': 0
            }

        return stats

    def generate_box_plots(self, output_dir='figures', save_formats=['pdf', 'png'],
                           n_players=None, n_demand=None):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Get unique experiment types and sample sizes
        experiment_types = ['numerical', 'analytical', 'infeasible']
        sample_sizes = sorted(set(key[1] for key in self.results.keys()))

        # Map experiment types to academic labels
        type_labels = {
            'analytical': 'Analytical',
            'numerical': 'Numerical',
            'infeasible': 'Infeasible'
        }

        # Set up academic plotting style
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14,
            'text.color': 'black',
            'axes.labelcolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black'
        })

        print("Producing box plots of estimates...")

        # Collect relative bias data grouped by parameter across sample sizes
        all_relative_bias_by_param = []
        for param_idx, param in enumerate(self.param_keys):
            param_bias_data = []
            for sample_size in sample_sizes:
                for exp_type in experiment_types:
                    key = (exp_type, sample_size)
                    if key not in self.results:
                        continue

                    estimates = self.results[key]['estimates'][:, param_idx]
                    success_flags = self.results[key]['success_flags']
                    true_value = self.results[key]['theta_true'][param_idx]
                    estimates = estimates[success_flags]
                    if len(estimates) > 0:
                        rel_bias = (estimates - true_value) / abs(true_value)
                        param_bias_data.extend(rel_bias)

            if param_bias_data:
                all_relative_bias_by_param.append(param_bias_data)

        # Calculate robust global scale using parameter-grouped data
        global_ylim = calculate_robust_global_scale(
            all_relative_bias_by_param,
            include_reference_lines=True,
            reference_values=[0]  # Zero bias reference line
        )

        if global_ylim is None:
            global_ylim = (-0.2, 0.2)  # Default range

        # Create one plot per sample size
        for sample_size in sample_sizes:
            n_exp_types = len([exp for exp in experiment_types
                              if (exp, sample_size) in self.results])
            if n_exp_types == 0:
                continue

            # Create figure with subplots for each experiment type
            fig, axes = plt.subplots(1, n_exp_types,
                                     figsize=(4 * n_exp_types, 6))
            if n_exp_types == 1:
                axes = [axes]

            # Define grayscale colors for print
            color_map = {
                'analytical': '0.3',      # Dark gray
                'numerical': '0.6',       # Medium gray
                'infeasible': '0.9'       # Light gray
            }

            subplot_idx = 0
            for exp_type in experiment_types:
                key = (exp_type, sample_size)
                if key not in self.results:
                    continue

                ax = axes[subplot_idx]

                # Collect data for all parameters for this experiment type and sample size
                box_data = []
                param_labels = []
                true_values = []

                for param_idx, param in enumerate(self.param_keys):
                    estimates = self.results[key]['estimates'][:, param_idx]
                    success_flags = self.results[key]['success_flags']
                    true_value = self.results[key]['theta_true'][param_idx]
                    estimates = estimates[success_flags]
                    if len(estimates) > 0:
                        rel_bias = (estimates - true_value) / abs(true_value)
                    else:
                        rel_bias = np.array([])

                    box_data.append(rel_bias)
                    param_labels.append(self.param_latex[param])  # LaTeX name
                    true_values.append(0.0)

                # Create box plot with outlier control
                bp = ax.boxplot(box_data, labels=param_labels,
                                patch_artist=True,
                                boxprops=dict(linewidth=1.2),
                                medianprops=dict(linewidth=1.5, color='black'),
                                whiskerprops=dict(linewidth=1.2),
                                capprops=dict(linewidth=1.2),
                                flierprops=dict(marker='o', markersize=3,
                                                alpha=0.6),
                                whis=1.5, showfliers=True)

                # Set colors (same color for all parameters in each panel)
                color = color_map.get(exp_type, '0.5')
                for patch in bp['boxes']:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                # Count outliers beyond the global scale for each parameter
                outlier_counts = []
                for param_idx, param in enumerate(self.param_keys):
                    estimates = self.results[key]['estimates'][:, param_idx]
                    true_value = self.results[key]['theta_true'][param_idx]
                    rel_bias = (estimates - true_value) / abs(true_value)

                    # Count outliers beyond global limits
                    n_outliers = np.sum((rel_bias < global_ylim[0]) | (rel_bias > global_ylim[1]))
                    outlier_counts.append(n_outliers)

                # Set global y-axis limits for all subplots
                ax.set_ylim(global_ylim)

                # Add outlier count annotations if there are significant outliers
                for i, (param, n_outliers) in enumerate(zip(param_labels, outlier_counts)):
                    if n_outliers > 0:
                        # Add annotation above the plot
                        ax.text(i + 1, global_ylim[1] * 0.9, f'{n_outliers}',
                                ha='center', va='center', fontsize=8,
                                bbox=dict(boxstyle='round,pad=0.3',
                                          facecolor='white', alpha=0.8,
                                          edgecolor='gray', linewidth=0.5))

                # Set subplot title
                experiment_type = type_labels.get(exp_type, exp_type)
                if experiment_type.startswith('infeasible'):
                    title = "Infeasible"
                else:
                    title = experiment_type.title() + " Gradient"
                ax.set_title(title)

                ax.grid(True, alpha=0.3, linestyle=':')

                # Add reference lines for interpretation
                ax.axhline(y=0, color='gray', linestyle='-',
                           linewidth=0.8, alpha=0.8)
                ax.axhline(y=0.10, color='gray', linestyle='--',
                           linewidth=0.8, alpha=0.5)
                ax.axhline(y=-0.10, color='gray', linestyle='--',
                           linewidth=0.8, alpha=0.5)

                # Only show y-label on leftmost subplot
                if subplot_idx == 0:
                    ax.set_ylabel('Relative Bias')

                subplot_idx += 1

            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], color='gray', linestyle='-',
                           linewidth=0.8, alpha=0.8, label='No Bias'),
                plt.Line2D([0], [0], color='gray', linestyle='--',
                           linewidth=0.8, alpha=0.5, label='±10% Bias')
            ]

            # Check if any outliers exist across all experiments for this n_obs
            has_outliers = False
            for exp_type in experiment_types:
                key = (exp_type, sample_size)
                if key not in self.results:
                    continue
                for param_idx, param in enumerate(self.param_keys):
                    estimates = self.results[key]['estimates'][:, param_idx]
                    true_value = self.results[key]['theta_true'][param_idx]
                    rel_bias = (estimates - true_value) / abs(true_value)
                    n_outliers = np.sum((rel_bias < global_ylim[0]) |
                                        (rel_bias > global_ylim[1]))
                    if n_outliers > 0:
                        has_outliers = True
                        break
                if has_outliers:
                    break

            if has_outliers:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='s', color='white',
                               markerfacecolor='white', markeredgecolor='gray',
                               markersize=6, linestyle='None',
                               label='# Extreme Outliers')
                )

            fig.legend(handles=legend_elements, loc='lower right',
                       bbox_to_anchor=(0.97, 0.08))

            plt.tight_layout()
            plt.subplots_adjust(top=0.85)

            # Save figure with requested filename format
            for fmt in save_formats:
                if n_players is not None and n_demand is not None:
                    filename = f'boxplot-{n_players}x{n_demand}-{sample_size}.{fmt}'
                else:
                    filename = f'boxplot-{sample_size}.{fmt}'
                filepath = output_path / filename
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Saved {filepath}")

            plt.close()

    def generate_computational_analysis(self, output_dir='figures',
                                        save_formats=['pdf', 'png'],
                                        n_players=None, n_demand=None):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        sample_sizes = sorted(set(key[1] for key in self.results.keys()))

        # Set up academic plotting style
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14,
            'text.color': 'black',
            'axes.labelcolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black'
        })

        # Calculate global scales for each metric using the robust function
        global_scales = {}

        for metric in ['ll_gap', 'iterations', 'function_evals', 'wall_times']:
            # Collect all data for this metric
            all_data = []
            for sample_size in sample_sizes:
                required_keys = [('analytical', sample_size),
                                 ('numerical', sample_size),
                                 ('infeasible', sample_size)]
                if not all(key in self.results for key in required_keys):
                    continue
                infeas_data = self.results[('infeasible', sample_size)]
                for method in ['analytical', 'numerical']:
                    method_data = self.results[(method, sample_size)]
                    success_flags = method_data['success_flags']
                    if metric == 'll_gap':
                        infeas_ll_filtered = infeas_data['log_likelihoods'][success_flags]
                        method_ll_filtered = method_data['log_likelihoods'][success_flags]
                        data = infeas_ll_filtered - method_ll_filtered
                    else:
                        data = method_data[metric][success_flags]
                    all_data.extend(data)

            # Use robust scale calculation with appropriate reference values
            # For computational metrics, we don't need parameter grouping
            if metric == 'll_gap':
                reference_values = [0]  # Gap = 0 reference line
            else:
                reference_values = None

            global_scales[metric] = calculate_robust_global_scale(
                [all_data],
                include_reference_lines=True,
                reference_values=reference_values
            )

        # Create computational efficiency analysis plots
        for sample_size in sample_sizes:
            # Check if we have all three experiment types for this sample size
            required_keys = [('analytical', sample_size),
                             ('numerical', sample_size),
                             ('infeasible', sample_size)]
            if not all(key in self.results for key in required_keys):
                continue

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Colors and labels
            colors = {'analytical': '0.3', 'numerical': '0.6'}
            labels = {'analytical': 'Analytical Gradients', 'numerical': 'Numerical Gradients'}

            # Get baseline (infeasible) data
            infeas_data = self.results[('infeasible', sample_size)]
            infeas_ll = infeas_data['log_likelihoods']

            # Panel a: Log-likelihood gap (infeasible LL - method LL)
            ax1 = axes[0, 0]
            ll_gaps = []
            method_names = []

            for method in ['analytical', 'numerical']:
                method_data = self.results[(method, sample_size)]
                success_flags = method_data['success_flags']
                method_ll = method_data['log_likelihoods'][success_flags]
                infeas_ll_filtered = infeas_ll[success_flags]
                # Calculate gap: how much worse is this method's LL compared to infeasible
                gap = infeas_ll_filtered - method_ll
                ll_gaps.append(gap)
                method_names.append(labels[method])

            bp1 = ax1.boxplot(ll_gaps, labels=method_names, patch_artist=True,
                              showfliers=True, whis=1.5,
                              boxprops=dict(linewidth=1.2),
                              medianprops=dict(linewidth=1.5, color='black'),
                              whiskerprops=dict(linewidth=1.2, color='black'),
                              capprops=dict(linewidth=1.2, color='black'),
                              flierprops=dict(marker='o', markersize=3, alpha=0.6, color='black'))
            for i, patch in enumerate(bp1['boxes']):
                patch.set_facecolor(list(colors.values())[i])
                patch.set_alpha(0.7)

            # Set global scale and count outliers
            if global_scales['ll_gap'] is not None:
                ax1.set_ylim(global_scales['ll_gap'])

                # Count and annotate outliers
                for i, (method, gap_data) in enumerate(zip(['analytical', 'numerical'], ll_gaps)):
                    n_outliers = np.sum((gap_data < global_scales['ll_gap'][0]) |
                                        (gap_data > global_scales['ll_gap'][1]))
                    if n_outliers > 0:
                        # Position annotation just below the top of the plot
                        y_pos = global_scales['ll_gap'][1] - 0.05 * (global_scales['ll_gap'][1] - global_scales['ll_gap'][0])
                        ax1.text(i + 1, y_pos, f'{n_outliers}',
                                 ha='center', va='center', fontsize=8,
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                           alpha=0.8, edgecolor='gray', linewidth=0.5))

            ax1.set_ylabel('Log-Likelihood Difference')
            ax1.set_title('(a) Infeasible LL - Method LL')
            ax1.grid(True, alpha=0.3, linestyle=':')

            # Panel b: Iteration counts
            ax2 = axes[0, 1]
            iter_data = []

            for method in ['analytical', 'numerical']:
                method_data = self.results[(method, sample_size)]
                success_flags = method_data['success_flags']
                method_iter = method_data['iterations'][success_flags]
                iter_data.append(method_iter)

            bp2 = ax2.boxplot(iter_data, labels=method_names, patch_artist=True,
                              showfliers=True, whis=1.5,
                              boxprops=dict(linewidth=1.2),
                              medianprops=dict(linewidth=1.5, color='black'),
                              whiskerprops=dict(linewidth=1.2, color='black'),
                              capprops=dict(linewidth=1.2, color='black'),
                              flierprops=dict(marker='o', markersize=3, alpha=0.6, color='black'))
            for i, patch in enumerate(bp2['boxes']):
                patch.set_facecolor(list(colors.values())[i])
                patch.set_alpha(0.7)

            # Set global scale and count outliers
            if global_scales['iterations'] is not None:
                ax2.set_ylim(global_scales['iterations'])

                # Count and annotate outliers
                for i, (method, ratio_data) in enumerate(zip(['analytical', 'numerical'], iter_data)):
                    n_outliers = np.sum((ratio_data < global_scales['iterations'][0]) |
                                        (ratio_data > global_scales['iterations'][1]))
                    if n_outliers > 0:
                        # Position annotation just below the top of the plot
                        y_pos = global_scales['iterations'][1] - 0.05 * (global_scales['iterations'][1] - global_scales['iterations'][0])
                        ax2.text(i + 1, y_pos, f'{n_outliers}',
                                 ha='center', va='center', fontsize=8,
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                           alpha=0.8, edgecolor='gray', linewidth=0.5))

            ax2.set_ylabel('Count')
            ax2.set_title('(b) Iterations')
            ax2.grid(True, alpha=0.3, linestyle=':')

            # Panel c: Function evaluations
            ax3 = axes[1, 0]
            nfev_data = []

            for method in ['analytical', 'numerical']:
                method_data = self.results[(method, sample_size)]
                success_flags = method_data['success_flags']
                method_nfev = method_data['function_evals'][success_flags]
                nfev_data.append(method_nfev)

            bp3 = ax3.boxplot(nfev_data, labels=method_names, patch_artist=True,
                              showfliers=True, whis=1.5,
                              boxprops=dict(linewidth=1.2),
                              medianprops=dict(linewidth=1.5, color='black'),
                              whiskerprops=dict(linewidth=1.2, color='black'),
                              capprops=dict(linewidth=1.2, color='black'),
                              flierprops=dict(marker='o', markersize=3, alpha=0.6, color='black'))
            for i, patch in enumerate(bp3['boxes']):
                patch.set_facecolor(list(colors.values())[i])
                patch.set_alpha(0.7)

            # Set global scale and count outliers
            if global_scales['function_evals'] is not None:
                ax3.set_ylim(global_scales['function_evals'])

                # Count and annotate outliers
                for i, (method, ratio_data) in enumerate(zip(['analytical', 'numerical'], nfev_data)):
                    n_outliers = np.sum((ratio_data < global_scales['function_evals'][0]) |
                                        (ratio_data > global_scales['function_evals'][1]))
                    if n_outliers > 0:
                        # Position annotation just below the top of the plot
                        y_pos = global_scales['function_evals'][1] - 0.05 * (global_scales['function_evals'][1] - global_scales['function_evals'][0])
                        ax3.text(i + 1, y_pos, f'{n_outliers}',
                                 ha='center', va='center', fontsize=8,
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                           alpha=0.8, edgecolor='gray', linewidth=0.5))

            ax3.set_ylabel('Count')
            ax3.set_title('(c) Function Evaluations')
            ax3.grid(True, alpha=0.3, linestyle=':')

            # Panel d: Wall time
            ax4 = axes[1, 1]
            time_data = []

            for method in ['analytical', 'numerical']:
                method_data = self.results[(method, sample_size)]
                success_flags = method_data['success_flags']
                method_time = method_data['wall_times'][success_flags]
                time_data.append(method_time)

            bp4 = ax4.boxplot(time_data, labels=method_names, patch_artist=True,
                              showfliers=True, whis=1.5,
                              boxprops=dict(linewidth=1.2),
                              medianprops=dict(linewidth=1.5, color='black'),
                              whiskerprops=dict(linewidth=1.2, color='black'),
                              capprops=dict(linewidth=1.2, color='black'),
                              flierprops=dict(marker='o', markersize=3, alpha=0.6, color='black'))
            for i, patch in enumerate(bp4['boxes']):
                patch.set_facecolor(list(colors.values())[i])
                patch.set_alpha(0.7)

            # Set global scale and count outliers
            if global_scales['wall_times'] is not None:
                ax4.set_ylim(global_scales['wall_times'])

                # Count and annotate outliers
                for i, (method, ratio_data) in enumerate(zip(['analytical', 'numerical'], time_data)):
                    n_outliers = np.sum((ratio_data < global_scales['wall_times'][0]) |
                                        (ratio_data > global_scales['wall_times'][1]))
                    if n_outliers > 0:
                        # Position annotation just below the top of the plot
                        y_pos = global_scales['wall_times'][1] - 0.05 * (global_scales['wall_times'][1] - global_scales['wall_times'][0])
                        ax4.text(i + 1, y_pos, f'{n_outliers}',
                                 ha='center', va='center', fontsize=8,
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                           alpha=0.8, edgecolor='gray', linewidth=0.5))

            ax4.set_ylabel('Seconds')
            ax4.set_title('(d) Wall Time')
            ax4.axhline(y=1, color='black', linestyle='-', linewidth=1, alpha=0.8)
            ax4.grid(True, alpha=0.3, linestyle=':')

            plt.tight_layout()
            plt.subplots_adjust(top=0.90)

            # Save computational analysis plot
            for fmt in save_formats:
                if n_players is not None and n_demand is not None:
                    filename = f'computational-analysis-{n_players}x{n_demand}-{sample_size}.{fmt}'
                else:
                    filename = f'computational-analysis-{sample_size}.{fmt}'
                filepath = output_path / filename
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Saved {filepath}")

            plt.close()

    def generate_latex_table(self, output_dir='figures', n_players=None, n_demand=None):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Get unique sample sizes
        sample_sizes = sorted(set(key[1] for key in self.results.keys()))

        for sample_size in sample_sizes:
            # Check which experiment types we have for this sample size
            available_types = []
            for exp_type in ['numerical', 'analytical', 'infeasible']:
                if (exp_type, sample_size) in self.results:
                    available_types.append(exp_type)

            if not available_types:
                continue

            # Create filename
            if n_players is not None and n_demand is not None:
                filename = f'tab_mc_{n_players}x{n_demand}_{sample_size}.tex'
            else:
                filename = f'tab_mc_{sample_size}.tex'
            filepath = output_path / filename

            with open(filepath, 'w') as f:
                # Start table
                f.write(f"\\begin{{table}}[htbp]\n")
                f.write(f"\\centering\n")
                f.write(f"\\caption{{Monte Carlo Results ({sample_size:,} Observations)}}\n")

                # Add label
                if n_players is not None and n_demand is not None:
                    f.write(f"\\label{{tab:mc:{n_players}x{n_demand}:{sample_size}}}\n")
                else:
                    f.write(f"\\label{{tab:mc:{sample_size}}}\n")

                # Create column specification
                n_cols = 2 + 2 * len(available_types)  # Parameter, True, then Mean & S.D. for each type
                col_spec = "l" + "r" + "rr" * len(available_types)
                f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
                f.write("\\hline\\hline\n")

                # Header row
                header = "Parameter & True"
                for exp_type in available_types:
                    if exp_type == 'numerical':
                        header += " & \\multicolumn{2}{c}{Numerical Gradient}"
                    elif exp_type == 'analytical':
                        header += " & \\multicolumn{2}{c}{Analytical Gradient}"
                    elif exp_type == 'infeasible':
                        header += " & \\multicolumn{2}{c}{Infeasible Start}"
                header += " \\\\\n"
                f.write(header)

                # Subheader row
                subheader = " & Value"
                for _ in available_types:
                    subheader += " & Mean & S.D."
                subheader += " \\\\\n"
                f.write(subheader)
                f.write("\\hline\n")

                # Parameter estimates
                for param in self.param_keys:
                    param_display = self.param_latex[param]
                    row = f"{param_display}"

                    # Get true value from first available type
                    first_type = available_types[0]
                    stats = self.get_summary_statistics(first_type, sample_size)
                    true_val = stats[param]['true_value']
                    row += f" & {true_val:.3f}"

                    # Add results for each experiment type
                    for exp_type in available_types:
                        stats = self.get_summary_statistics(exp_type, sample_size)
                        s = stats[param]
                        row += f" & {s['mean']:.3f} & {s['std']:.3f}"

                    row += " \\\\\n"
                    f.write(row)

                f.write("\\hline\n")

                # Computational statistics
                comp_metrics = [
                    ('Time (s)', 'mean_wall_time', 'std_wall_time', '.1f'),
                    ('Iterations', 'mean_iterations', 'std_iterations', '.1f'),
                    ('Func. Eval.', 'mean_function_evals', 'std_function_evals', '.1f'),
                    ('Log-likelihood', 'mean_log_likelihood', 'std_log_likelihood', '.4f')
                ]

                for metric_name, mean_key, std_key, fmt in comp_metrics:
                    row = f"{metric_name} & "

                    for exp_type in available_types:
                        stats = self.get_summary_statistics(exp_type, sample_size)
                        comp = stats['computational']
                        mean_val = comp[mean_key]
                        std_val = comp[std_key]
                        row += f" & {mean_val:{fmt}} & {std_val:{fmt}}"

                    row += " \\\\\n"
                    f.write(row)

                f.write(f"\\hline\n")
                f.write(f"\\end{{tabular}}\n")
                f.write(f"\\end{{table}}\n\n")

            print(f"Saved table to {filepath}")

    def save_results(self, filename):
        filepath = Path(filename)

        if filepath.suffix == '.pkl':
            with open(filepath, 'wb') as f:
                pickle.dump({'results': self.results, 'metadata': self.metadata}, f)
        elif filepath.suffix == '.json':
            # Convert numpy arrays to lists for JSON serialization
            json_data = {'metadata': self.metadata, 'results': {}}
            for key, data in self.results.items():
                json_key = f"{key[0]}_{key[1]}"
                json_data['results'][json_key] = {
                    'estimates': data['estimates'].tolist(),
                    'theta_true': data['theta_true'].tolist(),
                    'wall_times': np.nan_to_num(data['wall_times']).tolist(),
                    'iterations': data['iterations'].tolist(),
                    'function_evals': data['function_evals'].tolist(),
                    'log_likelihoods': data['log_likelihoods'].tolist(),
                    'success_flags': data['success_flags'].tolist(),
                    'n_replications': data['n_replications']
                }

            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2)
        else:
            raise ValueError("Unsupported file format. Use .pkl or .json")

        print(f"Saved results to {filepath}")

    @classmethod
    def load_results(cls, filename):
        filepath = Path(filename)

        if filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            param_latex = data['metadata'].get('param_latex', {})
            obj = cls(data['metadata']['param_keys'], param_latex)
            obj.results = data['results']
            obj.metadata = data['metadata']

        elif filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)

            param_latex = data['metadata'].get('param_latex', {})
            obj = cls(data['metadata']['param_keys'], param_latex)
            obj.metadata = data['metadata']

            for json_key, result_data in data['results'].items():
                parts = json_key.rsplit('_', 1)
                experiment_type = parts[0]
                sample_size = int(parts[1])
                key = (experiment_type, sample_size)

                obj.results[key] = {
                    'estimates': np.array(result_data['estimates']),
                    'theta_true': np.array(result_data['theta_true']),
                    'wall_times': np.array(result_data['wall_times']),
                    'iterations': np.array(result_data['iterations']),
                    'function_evals': np.array(result_data['function_evals']),
                    'log_likelihoods': np.array(result_data['log_likelihoods']),
                    'success_flags': np.array(result_data['success_flags']),
                    'n_replications': result_data['n_replications']
                }
        else:
            raise ValueError("Unsupported file format. Use .pkl or .json")

        return obj
