from mc import MonteCarloResults
import numpy as np
from multiprocessing import Pool, cpu_count
import time
from pathlib import Path
import argparse

# Parameter keys in order
PARAM_KEYS = ['theta_ec', 'theta_rn', 'theta_d', 'lambda', 'gamma']

# True parameter values
PARAM = {
    'theta_ec': -2.0,
    'theta_rn': -0.5,
    'theta_d': 2.0,
    'lambda': 1.0,
    'gamma': 0.3,
}

# Initial parameter values
START = {
    'theta_ec': -1.0,
    'theta_rn': -0.1,
    'theta_d': 1.0,
    'lambda': 0.2,
    'gamma': 1.0,
}

SEED = 20180120


def format_array(x):
    return np.array2string(x, formatter={'float_kind': lambda x: "%8.3f" % x})


def run_mc_replication(i, theta_start, n_obs, grad, n_players,
                       n_demand, param_dict, rho=0.05, delta=1.0,
                       seed=SEED, model='model', config={}, verbose=False):
    if model == 'model':
        from model import Model
    elif model == 'model_reduced':
        from model_reduced import ReducedFormModel as Model
    else:
        raise ValueError(f"Invalid model name: {model}")

    start_time = time.time()

    try:
        # Instantiate a model for this replication
        model_obj = Model(n_players=n_players, n_demand=n_demand, param=param_dict,
                          rho=rho, config=config, verbose=verbose)

        # Generate simulated data
        sample = model_obj.discrete_time_dgp(n_obs=n_obs, Delta=delta, seed=seed + 2024 * i)

        # Estimate parameters
        result = model_obj.estimate_parameters(sample=sample, Delta=delta, start=theta_start, use_grad=grad)

    except Exception as e:
        # If any exception is thrown, create a failed result object
        from scipy.optimize import OptimizeResult
        end_time = time.time()
        result = OptimizeResult(
            x=theta_start,
            success=False,
            fun=float('inf'),
            nfev=0,
            nit=0,
            message=f"Monte Carlo replication failed: {str(e)}"
        )
        result.wall_time = end_time - start_time
        print(f"Replication {i+1:5g}: FAILED - {str(e)}")
        return result

    # Record elapsed time
    end_time = time.time()
    result.wall_time = end_time - start_time

    # Print result with error message if optimization failed
    success_str = "" if result.success else f" [FAILED: {result.message}]"
    print(f"Replication {i+1:5g}:      {format_array(result.x)}, "
          f"LL = {-result.fun:10.6f}, Iter = {result.nit:6d}, "
          f"NFev = {result.nfev:4d}{success_str}")
    return result


def run_parallel_tasks(tasks, n_jobs, func):
    """Generic function to run tasks either sequentially or in parallel."""
    if n_jobs == 1:
        return [func(*task) for task in tasks]
    else:
        with Pool(processes=n_jobs) as pool:
            return pool.starmap(func, tasks)


def run_parallel_mc(theta_true, theta_start, n_obs, n_players, n_demand,
                    param_dict, n_mc=100, n_jobs=-1, rho=0.05, delta=1.0,
                    seed=SEED, model='model', config={}, mc_results=None):
    # Handle n_jobs like joblib: -1 means use all cores
    if n_jobs == -1:
        n_jobs = cpu_count()
    # Import specified model
    if model == 'model':
        from model import Model
    elif model == 'model_reduced':
        from model_reduced import ReducedFormModel as Model
    else:
        raise ValueError(f"Invalid model name: {model}")

    print(f"\nExperiments with analytical gradient (n_obs = {n_obs})...")
    tasks = [
        (i, theta_start, n_obs, True, n_players, n_demand, param_dict, rho, delta, seed, model, config)
        for i in range(n_mc)
    ]
    res_grad = run_parallel_tasks(tasks, n_jobs, run_mc_replication)

    print(f"\nExperiments with numerical gradient (n_obs = {n_obs})...")
    tasks = [
        (i, theta_start, n_obs, False, n_players, n_demand, param_dict, rho, delta, seed, model, config)
        for i in range(n_mc)
    ]
    res_nograd = run_parallel_tasks(tasks, n_jobs, run_mc_replication)

    # Determine best starting points for each replication
    print("\nDetermining starting points for infeasible runs...")
    best_starts = []

    for i in range(n_mc):
        # Compare log-likelihoods found so far
        analytical_ll = -res_grad[i].fun
        numerical_ll = -res_nograd[i].fun

        # Create model to evaluate true parameter log-likelihood
        model_obj = Model(n_players=n_players, n_demand=n_demand,
                          param=param_dict, rho=rho, config=config,
                          verbose=False)
        sample = model_obj.discrete_time_dgp(n_obs=n_obs, Delta=delta,
                                             seed=seed + 2024 * i)
        true_ll = model_obj.log_likelihood(theta_true, sample, delta,
                                           grad=False)

        # Find which method gives highest log-likelihood
        if analytical_ll >= numerical_ll and analytical_ll >= true_ll:
            best_params = res_grad[i].x.copy()
            best_source = 'analytical'
        elif numerical_ll >= analytical_ll and numerical_ll >= true_ll:
            best_params = res_nograd[i].x.copy()
            best_source = 'numerical'
        else:
            best_params = theta_true.copy()
            best_source = 'true'

        best_starts.append(best_params)

        print(f"Rep {i+1:3d}: Analytical LL={analytical_ll:10.6f}, "
              f"Numerical LL={numerical_ll:10.6f}, "
              f"True LL={float(true_ll):10.6f} -> Best: {best_source}")

    print(f"\nExperiments with infeasible start (n_obs = {n_obs})...")
    tasks = [
        (i, best_starts[i], n_obs, True, n_players, n_demand, param_dict, rho, delta, seed, model, config)
        for i in range(n_mc)
    ]
    res_inf = run_parallel_tasks(tasks, n_jobs, run_mc_replication)

    # Store results if storage object provided
    if mc_results is not None:
        mc_results.store_experiment('analytical', n_obs, res_grad, theta_true)
        mc_results.store_experiment('numerical', n_obs, res_nograd, theta_true)
        mc_results.store_experiment('infeasible', n_obs, res_inf, theta_true)

    return res_grad, res_nograd, res_inf


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Monte Carlo experiments for continuous-time '
        'dynamic discrete choice games',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--n_players', type=int, default=5,
                        help='Number of players')
    parser.add_argument('--n_demand', type=int, default=5,
                        help='Number of demand states')
    parser.add_argument('--n_mc', type=int, default=100,
                        help='Number of Monte Carlo replications')
    parser.add_argument('--n_obs', type=str, default='1000,4000,8000',
                        help='List of sample sizes (e.g., "1000,4000,8000")')
    parser.add_argument('--rho', type=float, default=0.05,
                        help='Discount rate')
    parser.add_argument('--delta', type=float, default=1.0,
                        help='Observation interval')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed base')
    parser.add_argument('--n_jobs', type=int, default=-1,  # auto
                        help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--model', type=str, default='model',
                        help='Which model to use (model, model_reduced)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results and figures')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')

    return parser.parse_args()


def main():
    # Set multiprocessing start method to avoid ResourceTracker issues on macOS
    import multiprocessing
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass  # Already set

    # Parse command line arguments
    args = parse_arguments()

    # Parse sample sizes
    try:
        sample_sizes = [int(x.strip()) for x in args.n_obs.split(',')]
    except ValueError:
        print("Error: n_obs must be a comma-separated list of integers")
        return

    # Model configuration
    model_config = {}

    # Parameter values (using defaults from constants)
    theta_true = np.array([PARAM[key] for key in PARAM_KEYS])
    theta_start = np.array([START[key] for key in PARAM_KEYS])

    # Define LaTeX parameter names for tables and figures
    PARAM_LATEX = {
        'theta_ec': r'$\theta_{EC}$',
        'theta_rn': r'$\theta_{RN}$',
        'theta_d': r'$\theta_{D}$',
        'lambda': r'$\lambda$',
        'gamma': r'$\gamma$'
    }

    # Create results storage object
    mc_results = MonteCarloResults(PARAM_KEYS, PARAM_LATEX)

    jobs_str = f"{args.n_jobs}" if args.n_jobs != -1 else "auto"
    if args.model == 'model':
        model_str = 'Structural dynamic model'
    else:
        model_str = 'Reduced-form model'

    print("\n\nCTCOMP Monte Carlo\n")
    print(f"Model:                    {model_str}")
    print(f"Number of players:        {args.n_players}")
    print(f"Number of demand states:  {args.n_demand}")
    print(f"Number of observations:   {sample_sizes}")
    print(f"Observation interval:     {args.delta}")
    print(f"Initial RNG seed:         {args.seed}")
    print(f"Number of replications:   {args.n_mc}")
    print(f"Parallel jobs:            {jobs_str}")
    print(f"True parameters:          {format_array(theta_true)}")
    print(f"Starting values:          {format_array(theta_start)}")
    print("")

    # Run experiments and store results
    for n_obs in sample_sizes:
        run_parallel_mc(
            theta_true=theta_true,
            theta_start=theta_start,
            n_obs=n_obs,
            n_players=args.n_players,
            n_demand=args.n_demand,
            param_dict=PARAM,
            n_mc=args.n_mc,
            n_jobs=args.n_jobs,
            rho=args.rho,
            delta=args.delta,
            seed=args.seed,
            model=args.model,
            config=model_config,
            mc_results=mc_results
        )

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y-%m-%d")
        output_dir = Path(f"{timestamp}-entry-mc")
    output_dir.mkdir(exist_ok=True)

    # Save results in both pkl and json format
    print("\nSaving results and generating figures and tables...")
    results_filename = output_dir / f'mc-results-{args.n_players}x{args.n_demand}'
    mc_results.save_results(f"{results_filename}.pkl")
    mc_results.save_results(f"{results_filename}.json")

    # Generate box plots of estimates
    mc_results.generate_box_plots(output_dir=output_dir, n_players=args.n_players, n_demand=args.n_demand)

    # Generate computational efficiency analysis
    mc_results.generate_computational_analysis(output_dir=output_dir, n_players=args.n_players, n_demand=args.n_demand)

    # Generate combined LaTeX tables
    mc_results.generate_latex_table(output_dir=output_dir, n_players=args.n_players, n_demand=args.n_demand)

    print(f"Results, figures, and tables saved to: {output_dir}")


if __name__ == "__main__":
    main()
