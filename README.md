# A Continuous-Time Dynamic Entry-Exit Game using Python, NumPy, and Cython

This directory contains the Python implementation of a continuous-time oligopoly
model of industry dynamics, focusing on strategic interaction between multiple
players in market entry/exit decisions under varying demand conditions.  This
code focuses on the more difficult case where only discrete time data is
available.  With continuous time data, estimation becomes much easier.

This is the source code accompanying the following paper, which contains full
details on the model and methods considered:

- Blevins, J.R. (2025).
  [Leveraging Uniformization and Sparsity for Estimation and Computation of Continuous-Time Dynamic Discrete Choice Games](https://jblevins.org/research/ctcomp).
  Working Paper,
  The Ohio State University.

## Table of Contents

- [Quick Start](#quick-start)
- [Dependencies](#dependencies)
- [Overview](#overview)
- [Cython Configuration](#cython-configuration)
- [Model Configuration](#model-configuration)
- [Monte Carlo Experiments](#monte-carlo-experiments)
- [Benchmarking](#benchmarking)
- [Multiple Equilibria Search](#multiple-equilibria-search)
- [Testing](#testing)

-----

## Quick Start

Most users will use the production-ready `Model` class:

```python
from model import Model

# Define model parameters
param = {
    'theta_ec': -0.5,  # Entry cost (negative)
    'theta_rn': -0.1,  # Rival effect (negative)
    'theta_d': 0.2,    # Demand effect (positive)
    'lambda': 2.0,     # Player action rate
    'gamma': 1.0,      # Demand transition rate
}

# Create and solve model with true parameters
model = Model(n_players=3, n_demand=3, param=param)
v, dv = model.value_function()  # Solve for value functions

# Generate simulated discrete-time data
sample = model.discrete_time_dgp(n_obs=1000, Delta=1.0, seed=1234)

# Compute log likelihood
ll = model.log_likelihood(sample, Delta=1.0)

# Estimate parameters from simulated data
results = model.estimate_parameters(sample, Delta=1.0)
print(f"Estimated parameters: {results.x}")
print(f"Log-likelihood: {-results.fun}")
```

-----

## Dependencies

**Required:**

- numpy
- scipy

**Optional:**

- cython (for high-performance acceleration on large models)
- pytest (for running tests)
- pytest-cov (for coverage reports)
- matplotlib (for benchmarking plots)

-----

## Overview

The codebase implements both a clean production version of the Model class
and a flexible version for benchmarking:

### Production Code

- **`model.py`**: Main production `Model` class with full optimizations

  - Continuous-time Markov jump process formulation
  - Value function algorithm selection: value iteration or polyalgorithm
  - Polyalgorithm (value iteration + Newton-Kantorovich)
  - Sparse matrix implementation for efficiency
  - Automatic Cython acceleration for large models
  - Maximum likelihood estimation with L-BFGS-B optimization
  - Analytical derivatives

- **`sparse.py`**: Specialized sparse matrix utilities, including sparse matrix
  exponential with derivatives

### Monte Carlo Simulations

- **`run_mc.py`**: Monte Carlo simulation entry point
- **`mc/`**: Monte Carlo infrastructure modules

### Benchmarking

- **`run_benchmarks.py`**: Main benchmarking entry point
- **`model_configurable.py`**: Configurable `Model` class with progressive optimization levels: `baseline` → `vectorize` → `polyalgorithm` → `cython` → `sparse` → `derivatives`
- **`optimization_config.py`**: Optimization configuration system
- **`benchmarks/`**: Benchmarking infrastructure modules

### Multiple Equilibria Search

- **`check_mc_equilibria.py`**: Search for equilibria with fixed parameter values used in the Monte Carlo experiments

- **`multiple_equilibria_search.py`**: Multiple equilibria search which systematically explores the parameter space in search of multiplicity, solving the model using the Newton-Kantorovich method with a large number of random starting points

-----

## Cython Configuration

For large-scale models with many players, Cython optimization provides
substantial performance gains.  The `Model` class automatically uses Cython
when:

- The Cython module is built and available
- The model has over 200 states
- The `use_cython` configuration is not explicitly disabled

### Prerequisites

**C compiler**:

- macOS: `xcode-select --install`
- Linux: `sudo apt-get install build-essential`
- Windows: Microsoft Visual C++ 14.0+

**Install Cython**:

```bash
pip install cython
```

### Build Instructions

```bash
python setup_cython.py build_ext --inplace
```

This creates `model_cython.cpython-*.so` (or `.pyd` on Windows).

To verify the setup works:

```bash
python -c "import model_cython; print('Cython ready')"
```

-----

## Model Configuration

The `Model` class requires the following structural parameters in the `param`
dictionary:

- `theta_ec`: Entry cost θ_EC (negative)
- `theta_rn`: Rival effect θ_RN (negative)
- `theta_d`: Demand effect θ_D (positive)
- `lambda`: Player action rate λ (positive)
- `gamma`: Demand transition rate γ (positive)

The `Model` class supports flexible configuration through the `config` parameter:

**Optimization Control for Estimation:**

- `opt_max_iter`: Maximum optimization iterations (default: 100)

**Value Function Algorithm:**

- `vf_algorithm`: Algorithm choice ('value_iteration', 'polyalgorithm')
    - `'value_iteration'`: Pure value iteration only
    - `'polyalgorithm'`: Value iteration + Newton-Kantorovich (default)
- `vf_max_iter`: Maximum value function iterations (default: 5000)
- `vf_tol`: Convergence tolerance (default: 1e-13)
- `vf_rtol`: NFXP switching ratio for polyalgorithm (default: 0.1)
- `vf_max_newton_iter`: Maximum Newton-Kantorovich iterations (default: 10)
- `vf_newton_solver`: Newton solver ('auto', 'direct', 'gmres')

**Performance:**

- `use_cython`: Cython control ('auto', True, False)
- `cython_threshold`: State threshold for auto Cython (default: 200)

### Usage Examples

```python
# Default configuration (polyalgorithm, recommended)
model = Model(n_players=3, n_demand=3, param=param)

# Pure value iteration (no Newton acceleration)
v, dv = model.value_function(algorithm='value_iteration')

# Custom polyalgorithm settings
config = {
    'vf_algorithm': 'polyalgorithm',
    'vf_max_iter': 1000,
    'vf_rtol': 0.05,
    'vf_max_newton_iter': 20,
}
model = Model(n_players=3, n_demand=3, param=param, config=config)

# Estimation with custom settings
results = model.estimate_parameters(
    sample,
    max_iter=200,              # Optimizer iterations
    vf_max_iter=100,           # Value iterations per evaluation
    vf_algorithm='polyalgorithm',
)
```

-----

## Monte Carlo Experiments

The included `run_mc.py` script generates simulated data and carries out Monte
Carlo experiments and produces an array of tables and figures based on the
results:

```bash
# Default configuration: 5 players, 5 demand states, 100 replications,
# with 1000, 4000, and 8000 observations each
python run_mc.py

# Custom configuration
python run_mc.py --n_players 3 --n_demand 3 --n_mc 50 --n_obs "1000,4000"

# Quick test with small model
python run_mc.py --n_players 2 --n_demand 2 --n_mc 10 --n_obs "500"

# See all options
python run_mc.py --help
```

**Command Line Arguments:**

- `--n_players`: Number of players (default: 5)
- `--n_demand`: Number of demand states (default: 5)
- `--n_mc`: Number of Monte Carlo replications (default: 100)
- `--n_obs`: Comma-separated list of sample sizes (default: "1000,4000,8000")
- `--rho`: Discount rate (default: 0.05)
- `--delta`: Discrete time observation interval (default: 1.0)
- `--seed`: Random seed base (default: 20180120)
- `--n_jobs`: Parallel jobs, -1 for all cores (default: -1)
- `--model`: Which model to use (dynamic structural model `model`, or reduced form model `model_reduced`)
- `--output_dir`: Name of a directory in which to store the results
- `--verbose`: Enable verbose output

**Results Storage and Visualization:**

The Monte Carlo experiments automatically store results and generate figures:

- **Data storage**: Results saved in both pickle (.pkl) and JSON formats.
- **Box plots**: Relative bias by specification (analytical/numerical/infeasible start) and sample size.
- **Method comparison**: Computational efficiency plots comparing analytical vs. numerical gradients.
- **LaTeX tables**: Summary statistics with bias, RMSE, and computational metrics.

**Output files generated:**

- `boxplot-{N_PLAYERS}x{N_DEMAND}-{N_OBS}.{pdf,png}`: Box plots for each sample size
- `computational-analysis-{N_PLAYERS}x{N_DEMAND}.{pdf,png}`: Comparison of computational accuracy and efficiency
- `tab_mc-{N_PLAYERS}x{N_DEMAND}_{N_OBS}.tex`: LaTeX summary tables
- `mc-results-{N_PLAYERS}x{N_DEMAND}.{pkl,json}`: Stored results data

-----

## Benchmarking

The benchmarking infrastructure provides systematic performance analysis across
various optimization levels and model sizes with streamlined, production-ready code.
We implement benchmarks in four key areas:

1. **Computation**: Value function and choice probability performance
2. **Estimation**: Full maximum likelihood estimation pipeline
3. **Sparsity**: Matrix operations and sparse optimization benefits
4. **Matrix Exponential**: Dense vs sparse exponential computation

The benchmarking system tests a series of progressive optimizations:

1. **Baseline**: Sequential reference implementations with explicit state lookups and loops
2. **Vectorize**: NumPy vectorization, value iteration only
3. **Polyalgorithm**: NFXP-style value iteration with Newton-Kantorovich switching for faster convergence
4. **Cython**: Cython acceleration for primary computational routines
5. **Sparse**: Sparse matrix optimizations
6. **Derivatives**: Analytical gradients for maximum likelihood estimation

The implementation of this progressive optimization strategy delegates methods
between `ConfigurableModel` and `Model`:

| Optimization Level | Value Function & CCPs | Q & Log Likelihood  |
|--------------------|-----------------------|---------------------|
| `baseline`         | `ConfigurableModel`   | `ConfigurableModel` |
| `vectorize`        | `Model`               | `ConfigurableModel` |
| `polyalgorithm`    | `Model`               | `ConfigurableModel` |
| `cython`           | `Model`               | `ConfigurableModel` |
| `sparse`           | `Model`               | `Model`             |
| `derivatives`      | `Model`               | `Model`             |

These optimization levels are compared across several model size categories

- **Very Small**: 2-4 players, 2 demand states (8-32 total states)
- **Small**: 4-6 players, 3 demand states (48-192 total states)
- **Medium**: 6-7 players, 4-5 demand states (256-640 total states)
- **Large**: 8 players, 4-6 demand states (1,024-1,536 total states)
- **Very Large**: 9-10 players, 5-6 demand states (2,560-6,144 total states)

Benchmarks can be run individually or in combination:

```bash
# Quick benchmarks (small models for development)
python run_benchmarks.py --quick

# Individual benchmark categories
python run_benchmarks.py --compute      # Value function performance
python run_benchmarks.py --estimate     # Estimation pipeline
python run_benchmarks.py --sparsity     # Sparse matrix benefits
python run_benchmarks.py --expm         # Matrix exponential comparison

# Combined benchmarks
python run_benchmarks.py --compute --estimate
python run_benchmarks.py --all          # All categories
python run_benchmarks.py --full         # All categories, all sizes

# Parallel execution (recommended for large benchmarks)
python run_benchmarks.py --parallel --max_workers 8

# Run everything with parallelization
python run_benchmarks.py --parallel --full --max-workers 24
```

-----

## Multiple Equilibria Search

The code also includes two scripts that search for multiple equilibria. The
first, `check_mc_equilibria.py`, carries out a search for equilibria with fixed
parameter values used in the Monte Carlo experiments. The script produces 10,000
random starting value functions drawn from a uniform distribution over the range
[min(u)/rho, max(u)/rho], where u represents the flow payoffs across all states
and rho is the continuous time discount rate. Newton-Kantorovich iterations from
each starting value are continued until convergence in the sup norm of the value
function differences with absolute tolerance 1e-13, up to a maximum of 500
iterations. Similar solutions are clustered by their implied choice
probabilities with a tolerance of 1e-3 to identify distinct equilibria. The
results suggest there is a unique equilibrium in this specification. To
replicate this search:

```bash
python check_mc_equilibria.py
```

The second script, `multiple_equilibria_search.py`, carries out a broad search
over a grid of 529,200 parameter vectors, each with 2,000 random value functions
used to initialize Newton-Kantorovich iterations to solve the system of
equilibrium equations. The parameter grid spans simple two-player models with no
strategic interaction to models with up to four very patient players with a very
high degree of competition. Starting values are drawn in the same way as the
first script, and the same convergence and clustering criteria are used.  This
procedure found multiple equilibria for 1,752 parameter values (0.33% of cases).
To run this search:

```bash
# Default settings
python multiple_equilibria_search.py

# Parallel execution
python multiple_equilibria_search.py --workers 48

# Resume from checkpoint
python multiple_equilibria_search.py --restart
```

-----

## Testing

The code includes a comprehensive `pytest` unit test suite.  Tests are organized
into specialized test files in the `tests/` directory:

- `test_model.py`: Tests of core `Model` functionality (state space encoding/decoding, precomputed transitions, model initialization, expected vector and array shapes and mathematical properties, data preprocessing, parameter estimation).
- `test_sparse.py`: Tests of sparse matrix utilities (matrix exponential and derivatives, sparse linear solvers).
- `test_regression.py`: Regression tests with fixed numerical baseline results including value functions, choice probabilities, intensity matrix, and log likelihood values.
- `test_derivatives.py`: Tests of analytical derivatives including value function derivatives, choice probability derivatives, Bellman operator derivatives, and log-likelihood gradients.
- `test_cython_sync.py`: Ensure Python/Cython implementation synchronization.
- `test_model_configurable.py`: Test `ConfigurableModel` implementation for benchmarking optimizations.
- `test_optimization.py`: Unit tests for `OptimizationConfig` and optimization sequence.
- `test_algorithm.py`: Tests of value function algorithms (value iteration and polyalgorithm).

To run the full test suite:

```bash
pytest
```
