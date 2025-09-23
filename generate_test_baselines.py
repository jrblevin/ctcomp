#!/usr/bin/env python3
"""
Generate baseline test results for regression testing.

This script generates reference values for all model components at various
parameter values.  The results are saved to a JSON file that is used by the
`test_regression.py` test suite to ensure numerical consistency.
"""

import numpy as np
import json
import os
from model import Model


def generate_model_baselines(verbose=False):
    """Generate baseline results for different model configurations."""

    # Define model specifications
    model_specifications = {
        'small': {
            'description': 'Small 2x2 model for basic testing',
            'n_players': 2,
            'n_demand': 2,
            'params': {
                'theta_ec': -1.0,
                'theta_rn': -1.0,
                'theta_d': 1.0,
                'lambda': 1.0,
                'gamma': 1.0,
            },
            'rho': 0.1
        },
        'baseline': {
            'description': 'Standard 3x2 baseline model',
            'n_players': 3,
            'n_demand': 2,
            'params': {
                'theta_ec': -1.5,
                'theta_rn': -0.5,
                'theta_d': 1.0,
                'lambda': 1.0,
                'gamma': 0.5,
            },
            'rho': 0.05
        },
        'large': {
            'description': 'Larger 5x3 model',
            'n_players': 5,
            'n_demand': 3,
            'params': {
                'theta_ec': -2.0,
                'theta_rn': -0.8,
                'theta_d': 1.5,
                'lambda': 0.8,
                'gamma': 0.3,
            },
            'rho': 0.1
        },
        'high_competition': {
            'description': 'Strong competitive effects',
            'n_players': 4,
            'n_demand': 2,
            'params': {
                'theta_ec': -1.0,
                'theta_rn': -2.0,
                'theta_d': 1.2,
                'lambda': 1.2,
                'gamma': 0.6,
            },
            'rho': 0.05
        },
        'low_entry_cost': {
            'description': 'Low entry cost',
            'n_players': 3,
            'n_demand': 3,
            'params': {
                'theta_ec': -0.5,
                'theta_rn': -0.3,
                'theta_d': 0.8,
                'lambda': 1.5,
                'gamma': 0.4,
            },
            'rho': 0.08
        },
        'strong_competition': {
            'description': 'Strong competition, low entry cost',
            'n_players': 4,
            'n_demand': 2,
            'params': {
                'theta_ec': -0.1,
                'theta_rn': -3.0,
                'theta_d': 1.0,
                'lambda': 5.0,
                'gamma': 0.5,
            },
            'rho': 0.01
        },
        'near_divergent': {
            'description': 'Near-divergent model',
            'n_players': 5,
            'n_demand': 2,
            'params': {
                'theta_ec': -0.05,
                'theta_rn': -4.0,
                'theta_d': 1.2,
                'lambda': 8.0,
                'gamma': 0.8,
            },
            'rho': 0.005
        }
    }

    # Storage for all baseline results
    baselines = {}

    # Generate baseline results for each specification
    for name, spec in model_specifications.items():
        print(f"\nGenerating baselines for: {name}")
        print(f"  Description: {spec['description']}")
        print(f"  Model size: {spec['n_players']} players, {spec['n_demand']} demand states")
        print(f"  Discount rate: {spec['rho']}")
        print(f"  Parameters: {spec['params']}")

        # Baseline settings
        model_config = {
            'vf_algorithm': 'polyalgorithm',   # Polyalgorithm for robustness
            'vf_max_iter': 5000,               # High value iteration limit
            'vf_tol': 1e-16,                   # Tight tolerance
            'vf_rtol': 0.1,            # NFXP switching ratio
            'vf_max_newton_iter': 100,         # High Newton iteration limit
            'vf_newton_solver': 'direct',      # Direct solver
        }
        model = Model(
            n_players=spec['n_players'],
            n_demand=spec['n_demand'],
            param=spec['params'],
            rho=spec['rho'],
            verbose=verbose,
            config=model_config,
        )

        # Store configuration
        baseline = {
            'config': spec,
            'results': {}
        }

        # Value function and derivatives
        # ------------------------------
        print("  Computing value function...")
        v, dv = model.value_function()
        baseline['results']['value_function'] = {
            'values': v.tolist(),
            'converged': True,
            'derivatives': {k: dv_k.tolist() for k, dv_k in dv.items()},
        }

        # Store a few specific values for quick checking
        baseline['results']['value_function']['sample_values'] = {
            'v[0]': float(v.flat[0]),
            'v[n_states//2]': float(v.flat[len(v)//2]),
            'v[-1]': float(v.flat[-1]),
            'mean': float(np.mean(v)),
            'std': float(np.std(v)),
        }

        # Choice probabilities
        # --------------------
        print("  Computing choice probabilities...")
        p, dp = model.choice_probabilities(v, dv)
        baseline['results']['choice_probabilities'] = {
            'values': p.tolist(),
            'derivatives': {k: dp_k.tolist() for k, dp_k in dp.items()}
        }

        # Sample statistics
        baseline['results']['choice_probabilities']['sample_values'] = {
            'p[0]': float(p.flat[0]),
            'p[n_states//2]': float(p.flat[len(p)//2]),
            'p[-1]': float(p.flat[-1]),
            'mean': float(np.mean(p)),
            'std': float(np.std(p)),
            'min': float(np.min(p)),
            'max': float(np.max(p)),
        }

        # Intensity matrix
        # ----------------
        print("  Computing intensity matrix...")
        Q, dQ = model.intensity_matrix()
        Q_dense = Q.toarray()
        baseline['results']['intensity_matrix'] = {
            'shape': Q.shape,
            'nnz': Q.nnz,
            'diagonal': Q.diagonal().tolist(),
            'row_sums': Q_dense.sum(axis=1).tolist(),  # Should be close to 0
            'derivatives_nnz': {k: dQ_k.nnz for k, dQ_k in dQ.items()},
        }

        # Sample off-diagonal elements of Q and dQ
        n_states = Q.shape[0]
        sample_indices = [
            (0, 1),
            (1, 2),
            (1, 3),
            (2, 1),
            (2, 3),
            (2, 4),
            (3, 2),
            (n_states//2, 1),
            (n_states//2, n_states//2 - 2),
            (n_states//2, n_states//2 - 1),
            (n_states//2, n_states//2 + 1),
            (n_states//2, n_states//2 + 2),
            (n_states//2, n_states-1),
            (n_states-2, n_states-4),
            (n_states-2, n_states-3),
            (n_states-2, n_states-1),
            (n_states-1, n_states-3),
            (n_states-1, n_states-2),
        ]
        baseline['results']['intensity_matrix']['sample_values'] = {}
        for i, j in sample_indices:
            if i < n_states and j < n_states:
                baseline['results']['intensity_matrix']['sample_values'][f'Q[{i},{j}]'] = float(Q_dense[i, j])
        baseline['results']['intensity_matrix']['sample_derivatives'] = {}
        for i, j in sample_indices:
            for k, dQ_k in dQ.items():
                if i < n_states and j < n_states:
                    baseline['results']['intensity_matrix']['sample_derivatives'][f'dQ/d{k}[{i},{j}]'] = float(dQ_k[i, j])

        # Log-likelihood function
        # -----------------------
        print("  Generating test data and computing log-likelihood...")
        dgp_seed = 42
        Delta = 1.0
        test_data = model.discrete_time_dgp(n_obs=10, Delta=1.0, seed=dgp_seed)

        # Compute log-likelihood
        param_array = np.array([spec['params'][key] for key in model.param_keys])
        ll, grad = model.log_likelihood(param_array, test_data, Delta=Delta, grad=True)
        baseline['results']['log_likelihood'] = {
            'n_obs': len(test_data),
            'dgp_seed': dgp_seed,
            'Delta': Delta,
            'sample': [int(s) for s in test_data],
            'value': float(ll),
            'gradient': grad.tolist(),
        }

        # Bellman operator
        # ----------------
        print("  Computing Bellman operator...")
        v_new = model.bellman_operator(v)
        baseline['results']['bellman_operator'] = {
            'values': v_new.tolist(),
            'max_change': float(np.max(np.abs(v_new - v))),
            'mean_change': float(np.mean(np.abs(v_new - v))),
            'sample_values': {
                'v_new[0]': float(v_new.flat[0]),
                'v_new[-1]': float(v_new.flat[-1]),
            }
        }

        # Derivative with respect to value function (Jacobian)
        print("  Computing Bellman operator derivatives ∂T/∂v...")
        dT_dv = model.dbellman_operator_dv(v)
        baseline['results']['dbellman_operator_dv'] = {
            'shape': dT_dv.shape,
            'nnz': dT_dv.nnz,
            'format': dT_dv.format,
            'values': dT_dv.toarray().tolist(),  # sparse -> dense -> list
            'sample_values': {
                'dT_dv[0,0]': float(dT_dv[0, 0]),
                'dT_dv[0,1]': float(dT_dv[0, 1]),
                'dT_dv[1,2]': float(dT_dv[1, 2]),
                'dT_dv[-1,-1]': float(dT_dv[-1, -1]),
                'max_diagonal': float(np.max(dT_dv.diagonal())),
                'min_diagonal': float(np.min(dT_dv.diagonal())),
                'max_abs': float(np.max(np.abs(dT_dv.data))),
            }
        }

        # Derivatives with respect to parameters
        print("  Computing Bellman operator derivatives ∂T/∂θ...")
        dT_dtheta = model.dbellman_operator_dtheta(v)
        baseline['results']['dbellman_operator_dtheta'] = {}
        for param_name in model.param_keys:
            param_deriv = dT_dtheta[param_name]
            baseline['results']['dbellman_operator_dtheta'][param_name] = {
                'values': param_deriv.tolist(),
                'sample_values': {
                    f'dT_d{param_name}[0,0]': float(param_deriv[0, 0]),
                    f'dT_d{param_name}[0,-1]': float(param_deriv[0, -1]),
                    f'dT_d{param_name}[-1,0]': float(param_deriv[-1, 0]),
                    f'dT_d{param_name}[-1,-1]': float(param_deriv[-1, -1]),
                    'mean': float(np.mean(param_deriv)),
                    'std': float(np.std(param_deriv)),
                    'max_abs': float(np.max(np.abs(param_deriv))),
                }
            }

        # Store this parameter set's baselines
        baselines[name] = baseline

        print(f"  Completed baseline generation for {name}")

    final_output = {
        'baselines': baselines,
    }

    # Save to JSON file in the tests directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, 'tests', 'test_baselines.json')
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2)

    print(f"\nBaseline results saved to {output_file}")
    print(f"Generated baselines for {len(baselines)} parameter sets")

    return final_output


def generate_algorithm_baselines(verbose=False):
    """Generate baseline results for different algorithm configurations."""

    # Define test model
    test_config = {
        'description': 'Small model for algorithm comparison',
        'n_players': 3,
        'n_demand': 2,
        'params': {
            'theta_ec': -1.5,
            'theta_rn': -0.5,
            'theta_d': 1.0,
            'lambda': 1.0,
            'gamma': 0.5,
        },
        'rho': 0.05,
    }

    # Define algorithm configurations to test
    algorithm_configs = {
        'value_iteration': {
            'vf_algorithm': 'value_iteration',
            'vf_max_iter': 5000,
            'vf_tol': 1e-14,
        },
        'polyalgorithm_auto': {
            'vf_algorithm': 'polyalgorithm',
            'vf_max_iter': 1000,
            'vf_tol': 1e-14,
            'vf_rtol': 0.1,
            'vf_max_newton_iter': 10,
            'vf_newton_solver': 'auto',
        },
        'polyalgorithm_direct': {
            'vf_algorithm': 'polyalgorithm',
            'vf_max_iter': 1000,
            'vf_tol': 1e-14,
            'vf_rtol': 0.05,
            'vf_max_newton_iter': 15,
            'vf_newton_solver': 'direct',
        },
        'polyalgorithm_gmres': {
            'vf_algorithm': 'polyalgorithm',
            'vf_max_iter': 1000,
            'vf_tol': 1e-13,
            'vf_rtol': 0.15,
            'vf_max_newton_iter': 20,
            'vf_newton_solver': 'gmres',
        },
    }

    # Storage for algorithm baselines
    algorithm_baselines = {}

    print("Generating algorithm-specific baselines...")
    print(f"Test model: {test_config['n_players']} players, {test_config['n_demand']} demand states")

    # Generate baselines for each algorithm configuration
    for algo_name, algo_config in algorithm_configs.items():
        print(f"\n  Running algorithm: {algo_name}")

        # Create model with specific algorithm configuration
        model = Model(
            n_players=test_config['n_players'],
            n_demand=test_config['n_demand'],
            param=test_config['params'],
            rho=test_config['rho'],
            verbose=verbose,
            config=algo_config
        )

        # Compute value function
        print("    Computing value function...")
        v, dv = model.value_function()

        # Store algorithm-specific results
        baseline = {
            'algorithm_config': algo_config,
            'test_config': test_config,
            'results': {
                'value_function': {
                    'values': v.tolist(),
                    'derivatives': {k: dv_k.tolist() for k, dv_k in dv.items()},
                    'sample_values': {
                        'v[0]': float(v.flat[0]),
                        'v[-1]': float(v.flat[-1]),
                        'mean': float(np.mean(v)),
                        'std': float(np.std(v)),
                        'max': float(np.max(v)),
                        'min': float(np.min(v)),
                    }
                }
            }
        }

        # Choice probabilities
        print("    Computing choice probabilities...")
        p, dp = model.choice_probabilities(v, dv)
        baseline['results']['choice_probabilities'] = {
            'values': p.tolist(),
            'derivatives': {k: dp_k.tolist() for k, dp_k in dp.items()},
            'sample_values': {
                'p[0]': float(p.flat[0]),
                'p[-1]': float(p.flat[-1]),
                'mean': float(np.mean(p)),
                'std': float(np.std(p)),
                'min': float(np.min(p)),
                'max': float(np.max(p)),
            }
        }

        # Intensity matrix
        print("    Computing intensity matrix...")
        Q, dQ = model.intensity_matrix()
        Q_dense = Q.toarray()
        baseline['results']['intensity_matrix'] = {
            'shape': Q.shape,
            'nnz': Q.nnz,
            'diagonal_sum': float(np.sum(Q.diagonal())),
            'max_off_diagonal': float(np.max(Q_dense[Q_dense != Q.diagonal()])),
            'row_sums_max_abs': float(np.max(np.abs(Q_dense.sum(axis=1)))),
            'derivatives_nnz': {k: dQ_k.nnz for k, dQ_k in dQ.items()},
        }

        # Bellman operator
        print("    Computing Bellman operator...")
        v_new = model.bellman_operator(v)
        baseline['results']['bellman_operator'] = {
            'values': v_new.tolist(),
            'max_change': float(np.max(np.abs(v_new - v))),
            'mean_change': float(np.mean(np.abs(v_new - v))),
            'sample_values': {
                'v_new[0]': float(v_new.flat[0]),
                'v_new[-1]': float(v_new.flat[-1]),
            }
        }

        # Test data and log-likelihood (same seed for all algorithms)
        print("    Computing log-likelihood...")
        seed = 12345
        np.random.seed(seed)  # Fixed seed for consistency across algorithms
        test_data = model.discrete_time_dgp(n_obs=10, Delta=1.0, seed=seed)
        param_array = np.array([test_config['params'][key] for key in model.param_keys])
        ll, grad = model.log_likelihood(param_array, test_data, Delta=1.0, grad=True)

        baseline['results']['log_likelihood'] = {
            'dgp_seed': seed,
            'n_obs': len(test_data),
            'sample': [int(s) for s in test_data],
            'value': float(ll),
            'gradient': grad.tolist(),
        }

        algorithm_baselines[algo_name] = baseline
        print("    Completed {algo_name}")

    return algorithm_baselines


def compare_algorithm_results(algorithm_baselines, tolerance):
    """Compare results across different algorithms to ensure consistency."""

    print("\nAlgorithm Consistency Check")
    print("===========================")

    algorithms = list(algorithm_baselines.keys())
    reference_algo = algorithms[0]  # Use first algorithm as reference

    print(f"Using '{reference_algo}' as reference algorithm")
    print(f"Comparing against: {', '.join(algorithms[1:])}")

    reference_results = algorithm_baselines[reference_algo]['results']

    issues_found = False

    for algo_name in algorithms[1:]:
        print(f"\nComparing {algo_name} vs {reference_algo}:")

        results = algorithm_baselines[algo_name]['results']

        # Compare value functions
        ref_v = np.array(reference_results['value_function']['values'])
        test_v = np.array(results['value_function']['values'])
        v_diff = np.max(np.abs(ref_v - test_v))

        if v_diff > tolerance:
            print(f"  Value function differs by {v_diff:.2e} (tolerance: {tolerance:.1e})")
            issues_found = True
        else:
            print(f"  Value function matches within {tolerance:.1e}")

        # Compare choice probabilities
        ref_p = np.array(reference_results['choice_probabilities']['values'])
        test_p = np.array(results['choice_probabilities']['values'])
        p_diff = np.max(np.abs(ref_p - test_p))

        if p_diff > tolerance:
            print(f"  Choice probabilities differ by {p_diff:.2e} (tolerance: {tolerance:.1e})")
            issues_found = True
        else:
            print(f"  Choice probabilities match within {tolerance:.1e}")

        # Compare log-likelihood values
        ref_ll = reference_results['log_likelihood']['value']
        test_ll = results['log_likelihood']['value']
        ll_diff = abs(ref_ll - test_ll)

        if ll_diff > tolerance:
            print(f"  Log-likelihood differs by {ll_diff:.2e}")
            issues_found = True
        else:
            print(f"  Log-likelihood matches within {tolerance:.1e}")

    if not issues_found:
        print("\nSUCCESS: All algorithms produce consistent results within tolerance!")
    else:
        print("\nWARNING: Some algorithms show differences beyond tolerance!")

    return not issues_found


def generate_baselines(verbose=False):
    """Generate all baseline results for regression testing."""

    print("Test Baseline Generation")
    print("========================")

    # Generate model baselines (different parameter sets)
    print("\nModel configuration baselines...")
    model_baselines = generate_model_baselines(verbose)

    # Generate algorithm baselines (different algorithms)
    print("\nAlgorithm configuration baselines...")
    algorithm_baselines = generate_algorithm_baselines(verbose)

    # Compare algorithm results for consistency
    consistency_ok = compare_algorithm_results(algorithm_baselines, tolerance=1e-11)

    # Combine all results
    combined_output = {
        'metadata': {
            'version': '2.0',
            'description': 'Comprehensive baseline test results for continuous-time entry model',
            'generated_by': 'generate_test_baselines.py',
            'numpy_version': np.__version__,
            'parameter_order': ['theta_ec', 'theta_rn', 'theta_d', 'lambda', 'gamma'],
        },
        'baselines': model_baselines['baselines'],
        'algorithm_baselines': algorithm_baselines,
    }

    # Save model baselines to original file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Save comprehensive baselines
    comprehensive_path = os.path.join(script_dir, 'tests', 'test_baselines.json')
    with open(comprehensive_path, 'w') as f:
        json.dump(combined_output, f, indent=2)

    print("Test Baseline Generation Complete")
    print("=================================")
    print("\nFiles generated:")
    print(f"  - {comprehensive_path}")

    # Summary
    print(f"\nModel parameter sets: {len(model_baselines['baselines'])}")
    for name, baseline in model_baselines['baselines'].items():
        config = baseline['config']
        results = baseline['results']
        print(f"  - {name}")
        print(f"    - {config['n_players']} players, {config['n_demand']} demand states")
        print(f"    - Q shape: {results['intensity_matrix']['shape']}")
        print(f"    - Log-likelihood: {results['log_likelihood']['value']:.6f}")
        print(f"    - Value function mean: {results['value_function']['sample_values']['mean']:.6f}")

    print(f"\nAlgorithm configurations: {len(algorithm_baselines)}")
    for algo_name in algorithm_baselines.keys():
        config = algorithm_baselines[algo_name]['algorithm_config']
        print(f"  - {algo_name}")

    if consistency_ok:
        print("\nSUCCESS: All algorithms are numerically consistent!")
    else:
        print("\nWARNING: Some algorithms show numerical differences")

    return combined_output


if __name__ == "__main__":
    baselines = generate_baselines(verbose=False)
