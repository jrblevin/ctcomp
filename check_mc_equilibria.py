#!/usr/bin/env python3
"""
Search for multiple equilibria.
"""

import numpy as np
from model import Model
from multiprocessing import Pool
import psutil

# True parameter values
PARAM = {
    'theta_ec': -2.0,
    'theta_rn': -0.5,
    'theta_d': 2.0,
    'lambda': 1.0,
    'gamma': 0.3,
}

N_PLAYERS = 7
N_DEMAND = 5
RHO = 0.05

# Admits 3 equilibria with N_PLAYERS=2, N_DEMAND=2
# PARAM = {
#     'theta_ec': -1.0,
#     'theta_rn': -1.0,
#     'theta_d': 3.0,
#     'lambda': 0.2,
#     'gamma': 1.6,
# }
#
# N_PLAYERS = 2
# N_DEMAND = 2
# RHO = 0.05

SEED = 20180120
MAX_NEWTON_ITER = 500


def solve_single_equilibrium(args):
    """
    Worker function to solve for a single equilibrium.

    Parameters
    ----------
    args : tuple
        (i, n_players, n_demand, param, rho, v_lower, v_upper, seed, max_newton_iter)

    Returns
    -------
    result : dict or None
        Dict with 'v' and 'choice_probs' if successful, None otherwise
    """
    i, n_players, n_demand, param, rho, v_lower, v_upper, seed, max_newton_iter = args

    # Create model for each worker
    model = Model(n_players=n_players, n_demand=n_demand,
                  param=param, rho=rho, verbose=False)

    # Random starting values
    np.random.seed(seed + i)
    v_init = np.random.uniform(v_lower, v_upper, size=(n_players, model.K))

    # Solve for equilibrium
    try:
        v_eq, _ = model.value_function(
            vf_max_iter=0,  # Start Newton directly (skip value iteration)
            vf_tol=1e-13,
            vf_algorithm='polyalgorithm',
            vf_newton_solver='direct',
            vf_max_newton_iter=max_newton_iter,
            v_init=v_init,
            compute_derivatives=False,  # Skip derivative computation for speed
        )

        # Compute choice probabilities
        choice_probs = model.choice_probabilities(v=v_eq)

        return {
            'iteration': i,
            'v': v_eq.copy(),
            'choice_probs': choice_probs.copy(),
            'success': True
        }
    except Exception as e:
        return {
            'iteration': i,
            'success': False,
            'error': str(e)
        }


def find_equilibria(model, n_searches=10000, tol=1e-3, n_jobs=-1, verbose=True):
    """
    Find all equilibria for the model (parallelized).

    Parameters
    ----------
    model : Model
        The model with parameters set
    n_searches : int
        Number of random starting values
    tol : float
        Tolerance for considering two equilibria as identical
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    verbose : bool
        Print progress

    Returns
    -------
    equilibria : list of dict
        Each dict contains:
        - 'v': value function (n_players, K)
        - 'choice_probs': conditional choice probabilities (n_players, K, n_actions)
        - 'id': equilibrium index
    """
    # n_jobs = -1 means use all physical cores
    if n_jobs == -1:
        n_jobs = psutil.cpu_count(logical=False)

    # Compute bounds for value function initialization
    theta_rn = model.param['theta_rn']
    theta_d = model.param['theta_d']
    flow_payoffs = theta_rn * model.n_active + theta_d * model.demand_states
    min_payoff = np.min(flow_payoffs)
    max_payoff = np.max(flow_payoffs)
    v_lower = min_payoff / model.rho
    v_upper = max_payoff / model.rho

    if verbose:
        print(f"Searching for equilibria with {n_searches} random initializations...", flush=True)
        print(f"Using {n_jobs} parallel workers", flush=True)
        print(f"Value function bounds: [{v_lower:.2f}, {v_upper:.2f}]", flush=True)
        print(flush=True)

    # Build tasks
    tasks = [
        (i, model.n_players, model.n_demand, model.param, model.rho,
         v_lower, v_upper, SEED, MAX_NEWTON_ITER)
        for i in range(n_searches)
    ]

    # Run in parallel and process results as they complete
    unique_equilibria = []
    n_failed = 0

    with Pool(processes=n_jobs) as pool:
        # Use imap_unordered to get results as they complete
        for result in pool.imap_unordered(solve_single_equilibrium, tasks):
            i = result['iteration']

            if not result['success']:
                n_failed += 1
                if verbose:
                    print(f"Iteration {i:4d}: Failed to converge: {result['error']}", flush=True)
                continue

            choice_probs = result['choice_probs']

            # Check if this is a new equilibrium
            is_new = True
            matched_eq_id = None
            for eq in unique_equilibria:
                if np.allclose(choice_probs, eq['choice_probs'], atol=tol):
                    is_new = False
                    matched_eq_id = eq['id']
                    break

            if is_new:
                unique_equilibria.append({
                    'v': result['v'].copy(),
                    'choice_probs': choice_probs.copy(),
                    'id': len(unique_equilibria)
                })
                if verbose:
                    print(f"Iteration {i:4d}: New equilibrium {len(unique_equilibria)} found! (Total unique: {len(unique_equilibria)})", flush=True)
            else:
                if verbose:
                    print(f"Iteration {i:4d}: Converged to existing equilibrium {matched_eq_id}", flush=True)

    if verbose:
        print(f"\nCompleted {n_searches} searches", flush=True)
        print(f"Found {len(unique_equilibria)} unique equilibria", flush=True)
        print(f"Failed convergences: {n_failed}", flush=True)

    return unique_equilibria


def main():
    import argparse

    # Set multiprocessing start method
    import multiprocessing
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    # Parse arguments
    parser = argparse.ArgumentParser(description='Check for multiple equilibria')
    parser.add_argument('--n_searches', type=int, default=10000,
                        help='Number of random starting values')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 for all cores)')
    args = parser.parse_args()

    print("Multiple Equilibria Search", flush=True)
    print("==========================", flush=True)
    print("\nParameters:", flush=True)
    for key, val in PARAM.items():
        print(f"  {key:12s} = {val:8.3f}", flush=True)
    print(f"  n_players    = {N_PLAYERS}", flush=True)
    print(f"  n_demand     = {N_DEMAND}", flush=True)
    print(f"  rho          = {RHO}", flush=True)
    print(f"  n_searches   = {args.n_searches}", flush=True)
    print(f"  n_jobs       = {args.n_jobs if args.n_jobs != -1 else 'auto'}", flush=True)
    print(flush=True)

    # Create model
    model = Model(n_players=N_PLAYERS, n_demand=N_DEMAND,
                  param=PARAM, rho=RHO, verbose=False)

    print(f"Space size K = {model.K}", flush=True)
    print(flush=True)

    # Find all equilibria
    equilibria = find_equilibria(model, n_searches=args.n_searches, n_jobs=args.n_jobs, verbose=True)

    # Report results
    if len(equilibria) == 0:
        print("\nNo equilibria found!")
    elif len(equilibria) == 1:
        print("\nUnique equilibrium found")
    else:
        print("\nMultiple equilibria found: {len(equilibria)} distinct equilibria")
        print("\nChoice probability differences between equilibria:")
        for i in range(len(equilibria)):
            for j in range(i+1, len(equilibria)):
                diff = np.abs(equilibria[i]['choice_probs'] - equilibria[j]['choice_probs'])
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                print(f"  Eq {i} vs Eq {j}: max diff = {max_diff:.4f}, mean diff = {mean_diff:.4f}")


if __name__ == "__main__":
    main()
