#!/usr/bin/env python3
"""
Search for multiple equilibria using extensive parameter grids
with many starting values for Newton's method.
"""

import numpy as np
from model import Model
import itertools
from multiprocessing import Pool
import json
import argparse
import os
import pickle
from datetime import datetime
import time
import psutil

CHECKPOINT_INTERVAL = 1000
MAX_NEWTON_ITER = 500
N_RUNS = 2000


def test_single_case(args):
    """Search for equilibria for a single parameter combination."""
    params, rho, n_players, n_demand, n_runs = args

    model = Model(n_players, n_demand, params, rho=rho, verbose=False)

    solutions = []
    residuals = []
    newton_iter = []
    failures = []  # Track convergence failures
    errors = []    # Track exceptions

    # Compute economically meaningful bounds for value function
    # Flow payoffs for active players: theta_rn * n_active + theta_d * demand_state
    # Value function bounds: [min_payoff/rho, max_payoff/rho]
    theta_rn = params['theta_rn']
    theta_d = params['theta_d']
    flow_payoffs = theta_rn * model.n_active + theta_d * model.demand_states
    min_payoff = np.min(flow_payoffs)
    max_payoff = np.max(flow_payoffs)
    v_lower = min_payoff / rho
    v_upper = max_payoff / rho

    # Use many different random starting values to explore solution space
    for run in range(n_runs):
        try:
            # Reset model state
            model._value_cache = None
            model._v_prev = None

            # Set random seed for reproducibility
            np.random.seed(run)

            # Generate random initial value function
            # Use uniform distribution over economically meaningful range
            v_init = np.random.uniform(v_lower, v_upper, size=(model.n_players, model.K))

            v, dv = model.value_function(
                vf_max_iter=0,  # Start Newton directly from random initialization
                vf_tol=1e-13,
                vf_algorithm='polyalgorithm',
                vf_newton_solver='direct',
                vf_max_newton_iter=MAX_NEWTON_ITER,
                v_init=v_init,
                compute_derivatives=False,  # Skip derivative computation for speed
            )
            newton_iter.append(model._last_phase2_iter)

            # Check if we found a valid equilibrium using stored Bellman evaluation
            residual = np.max(np.abs(model._last_bellman_eval - v))
            if residual < 1e-10:  # Valid equilibrium
                p = model.choice_probabilities(v)
                solutions.append(p.copy())
                residuals.append(residual)
            else:
                # Record failure details
                failures.append({
                    'run': run,
                    'residual': float(residual),
                    'newton_iterations': model._last_phase2_iter
                })

        except Exception as e:
            # Record error details
            errors.append({
                'run': run,
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
            print(f"  Error in run {run}: {type(e).__name__}: {str(e)}")
            continue

    # Check for distinct solutions with tight tolerance
    distinct_equilibria = []
    equilibrium_residuals = []
    equilibrium_newton_iter = []
    tolerance = 1e-3

    for i, sol in enumerate(solutions):
        is_new = True
        for existing_sol in distinct_equilibria:
            if np.max(np.abs(sol - existing_sol)) < tolerance:
                is_new = False
                break
        if is_new:
            distinct_equilibria.append(sol)
            equilibrium_residuals.append(residuals[i])
            equilibrium_newton_iter.append(newton_iter[i])

    if len(distinct_equilibria) > 1:
        print("Found multiple equilibria!")
        print("n_players:", n_players)
        print("n_demand:", n_demand)
        print("params:", params)
        print("rho:", rho)

    return {
        'params': params,
        'rho': rho,
        'n_players': n_players,
        'n_demand': n_demand,
        'n_runs': n_runs,
        'num_solutions': len(solutions),
        'num_distinct_equilibria': len(distinct_equilibria),
        'distinct_equilibria': distinct_equilibria,
        'equilibrium_residuals': equilibrium_residuals,
        'equilibrium_newton_iter': equilibrium_newton_iter,
        'num_failures': len(failures),
        'num_errors': len(errors),
    }


def generate_parameter_grid():
    """Generate comprehensive parameter grid for testing."""

    # Parameter ranges
    theta_ec_values = [0.0, -0.01, -0.1, -0.5, -1.0, -2.0, -3.0]
    theta_rn_values = [0.0, -0.25, -0.5, -1.0, -2.0, -3.0, -4.0, -5.0]
    theta_d_values = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0]
    lambda_values = [0.1, 0.25, 0.5, 1.0, 2.0]
    gamma_values = [0.1, 0.25, 0.5, 1.0, 2.0]
    rho_values = [0.01, 0.03, 0.05, 0.1, 0.5, 1.0]
    n_players = [2, 3, 4]
    n_demand = [1, 2, 3]

    # Generate all combinations
    all_combinations = list(itertools.product(
        theta_ec_values, theta_rn_values, theta_d_values,
        lambda_values, gamma_values, rho_values,
        n_players, n_demand
    ))

    combinations = []
    for theta_ec, theta_rn, theta_d, lam, gamma, rho, n_players, n_demand in all_combinations:
        params = {
            'theta_ec': theta_ec,
            'theta_rn': theta_rn,
            'theta_d': theta_d,
            'lambda': lam,
            'gamma': gamma,
        }
        combinations.append((params, rho, n_players, n_demand, N_RUNS))

    return combinations


def save_checkpoint(all_runs, completed_count, checkpoint_file="checkpoint.pkl"):
    """Save checkpoint with all results and current completion count."""
    checkpoint_data = {
        'all_runs': all_runs,
        'completed_count': completed_count,
        'timestamp': datetime.now().isoformat()
    }
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)

    # Count multiplicity cases
    multiple_equilibria_cases = sum(1 for case in all_runs if case['num_distinct_equilibria'] > 1)

    print(f"  Checkpoint saved: {len(all_runs)} results, {completed_count} completed")
    print(f"  Multiple equilibria cases found so far: {multiple_equilibria_cases}")


def load_checkpoint(checkpoint_file="checkpoint.pkl"):
    """Load checkpoint data if it exists and verify parameter consistency."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)

            completed_count = checkpoint_data.get('completed_count', len(checkpoint_data['all_runs']))
            print(f"Checkpoint loaded: {len(checkpoint_data['all_runs'])} previous results")
            print(f"Resuming from task {completed_count + 1}")
            return checkpoint_data['all_runs'], completed_count
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return [], 0
    return [], 0


def save_results(all_runs, output_file="multiple_equilibria_search.json"):
    """Save final results to JSON file."""
    with open(output_file, "w") as f:
        json_compatible_runs = []
        for run in all_runs:
            json_run = {}
            for key, value in run.items():
                if isinstance(value, list) and len(value) > 0 and hasattr(value[0], 'tolist'):
                    json_run[key] = [item.tolist() for item in value]
                elif hasattr(value, 'tolist'):
                    json_run[key] = value.tolist()
                else:
                    json_run[key] = value
            json_compatible_runs.append(json_run)

        json.dump(json_compatible_runs, f, indent=4)


def main():
    """Comprehensive search for multiple equilibria."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Search for multiple equilibria in continuous-time entry games")
    parser.add_argument('--restart', action='store_true', help='Restart from checkpoint if available')
    parser.add_argument('--checkpoint-file', default='multiple_equilibria_search_checkpoint.pkl', help='Checkpoint file')
    parser.add_argument('--output-file', default='multiple_equilibria_search.json', help='Output file path')
    parser.add_argument('--workers', type=int, default=psutil.cpu_count(logical=False), help='Number of worker processes')
    args = parser.parse_args()

    print("Multiple Equilibrium Search")
    print("===========================")

    parameter_combinations = generate_parameter_grid()

    print(f"\nTesting {len(parameter_combinations):,} parameter combinations...")
    print(f"Each with {N_RUNS:,} different Newton starting points...")
    print(f"Using {args.workers} parallel processes...")
    print(f"Total Newton's method runs: {len(parameter_combinations) * N_RUNS:,}")

    # Load checkpoint if restarting
    if args.restart:
        all_runs, start_idx = load_checkpoint(args.checkpoint_file)
        remaining_combinations = parameter_combinations[start_idx:]
    else:
        all_runs = []
        start_idx = 0
        remaining_combinations = parameter_combinations

    if not remaining_combinations:
        print("All tasks already completed!")
        return

    print(f"Processing {len(remaining_combinations):,} remaining parameter combinations...")

    # Create single pool and process with streaming results
    completed_count = start_idx
    last_checkpoint_time = time.time()

    try:
        with Pool(args.workers) as pool:
            # Use imap_unordered for immediate result processing
            results_iter = pool.imap_unordered(test_single_case, remaining_combinations, chunksize=1)

            for result in results_iter:
                if result is not None:
                    all_runs.append(result)
                    if result['num_distinct_equilibria'] > 1:
                        print(f"*** Multiple equilibria found! Task {completed_count + 1} ***")

                completed_count += 1

                # Progress update and checkpointing
                if completed_count % 100 == 0:  # Progress every 100 tasks
                    progress = (completed_count - start_idx) / len(remaining_combinations) * 100
                    print(f"Progress: {completed_count - start_idx}/{len(remaining_combinations)} ({progress:.1f}%)")

                # Checkpoint periodically
                if (completed_count % CHECKPOINT_INTERVAL == 0 or
                        time.time() - last_checkpoint_time > 600):
                    save_checkpoint(all_runs, completed_count, args.checkpoint_file)
                    last_checkpoint_time = time.time()
                    print(f"  Checkpoint saved at task {completed_count}")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving checkpoint...")
        save_checkpoint(all_runs, completed_count, args.checkpoint_file)
        return
    except Exception as e:
        print(f"Error in processing: {e}")
        save_checkpoint(all_runs, completed_count, args.checkpoint_file)
        raise

    # Final checkpoint
    save_checkpoint(all_runs, completed_count, args.checkpoint_file)

    # Results summary
    print("Equilibrium Search Results")
    print("==========================")
    print()

    print(f"Total Newton's method runs: {len(parameter_combinations) * N_RUNS:,}")
    print()

    multiple_equilibria_cases = [case for case in all_runs if case['num_distinct_equilibria'] > 1]

    if multiple_equilibria_cases:
        print(f"Found {len(multiple_equilibria_cases)} cases with multiple equilibria!")
        for i, case in enumerate(multiple_equilibria_cases[:10]):  # Show first 10
            print(f"Case {i+1}: {case['num_distinct_equilibria']} distinct equilibria")
            print(f"  Players: {case['n_players']}, Demand: {case['n_demand']}")
            print(f"  Parameters: {case['params']}")
            print(f"  Discount rate: {case['rho']}")
            print(f"  Valid solutions found: {case['num_solutions']}")
            print()
    else:
        print("No multiple equilibria found.")

    # Save the final results to a file
    save_results(all_runs, args.output_file)
    print(f"\nResults saved to: {args.output_file}")

    # Clean up checkpoint file on successful completion
    if os.path.exists(args.checkpoint_file):
        os.remove(args.checkpoint_file)


if __name__ == "__main__":
    main()
