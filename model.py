import itertools
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, identity
from scipy.sparse.linalg import spsolve, gmres
from scipy.linalg import eig
from scipy.optimize import minimize, OptimizeResult
import scipy

from sparse import vexpm, vexpm_deriv, spsolve_multiple_rhs

# Detect SciPy version for GMRES parameter compatibility
_SCIPY_VERSION = tuple(map(int, scipy.__version__.split('.')[:2]))
_GMRES_TOL_PARAM = 'rtol' if _SCIPY_VERSION >= (1, 12) else 'tol'

# Model constants
EPSILON = 1e-16                 # Numerical stability (div. by zero, log(0))
CYTHON_STATE_THRESHOLD = 200    # Auto-enable Cython above this state count
SPARSE_STATE_THRESHOLD = 200    # State threshold for direct vs GMRES solver
VF_DIVERGENCE_THRESHOLD = 1.05  # Convergence rate for divergence detection
GMRES_TOLERANCE = 1e-12         # Tolerance for GMRES iterative solver
MIN_MONITORING_ITERS = 10       # Minimum iterations before Newton switching

# Default configuration for all model parameters
DEFAULT_CONFIG = {
    'rho': 0.05,                                # Discount rate
    'vf_tol': 1e-13,                            # Convergence tolerance for value iteration
    'vf_max_iter': 5000,                        # Maximum value function iterations
    'vf_algorithm': 'polyalgorithm',            # VF algorithm ('value_iteration', 'polyalgorithm')
    'vf_rtol': 0.1,                             # NFXP switching threshold
    'vf_max_newton_iter': 10,                   # Maximum Newton-Kantorovich iterations
    'vf_newton_solver': 'auto',                 # Newton solver ('auto', 'direct', 'gmres')
    'opt_max_iter': 100,                        # Maximum optimization iterations
    'opt_tolerance': 1e-12,                     # Default optimization tolerance
    'use_cython': 'auto',                       # Cython usage ('auto', True, False)
    'cython_threshold': CYTHON_STATE_THRESHOLD  # Threshold for auto Cython usage
}

# Try to import Cython optimization
try:
    from model_cython import (
        _bellman_components_cython,
        bellman_operator_cython,
        choice_probabilities_cython,
        dbellman_operator_dtheta_cython,
        dbellman_operator_dv_cython,
    )
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


class CythonFallbackWarning(UserWarning):
    """Warning raised when Cython optimization fails and falls back to Python."""
    pass


class Model:
    """
    Continuous-time dynamic discrete choice entry/exit game.

    This class implements a model of strategic interaction between multiple
    players in a market with stochastic demand.  Players make entry and exit
    decisions in continuous time and the overall market dynamics are
    characterized by a continuous-time Markov jump process.

    This class implements the following features:

    - Value function iteration and Newton-Kantorovich algorithms
      for computing equilibria.
    - Continuous-time Markov jump process formulation.
    - Parameter estimation using maximum likelihood.
    - Sparse matrix implementation for efficiency.
    - Analytical computation of parameter derivatives.
    - Cython acceleration for large models.

    Parameters
    ----------
    n_players : int
        Number of players in the market
    n_demand : int
        Number of demand states
    param : dict
        Model parameters including:
        - 'theta_ec': Entry cost θ_EC (negative)
        - 'theta_rn': Rival effect θ_RN (negative)
        - 'theta_d': Demand effect θ_D (positive)
        - 'lambda': Player action rate λ (positive)
        - 'gamma': Demand transition rate γ (positive)
    rho : float, optional
        Discount rate ρ
    verbose : bool, optional
        Whether to print progress information
    config : dict, optional
        Configuration for algorithms.  If None, uses default values.

        **Optimization Configuration:**

        - 'opt_max_iter' : int, default 100
            Maximum iterations for parameter estimation optimization

        **Value Function Configuration:**

        - 'vf_algorithm' : default 'polyalgorithm'
            Algorithm for computing value functions:
            - 'value_iteration': Value iteration only
            - 'polyalgorithm': Value iteration + Newton-Kantorovich

        - 'vf_max_iter' : int, default 5000
            Maximum value function iterations

        - 'vf_tol' : float, default 1e-13
            Convergence tolerance for value function

        - 'vf_rtol' : float, default 0.1
            For polyalgorithm: threshold for switching to Newton method

        - 'vf_max_newton_iter' : int, default 10
            Maximum Newton-Kantorovich iterations

        - 'vf_newton_solver' : {'auto', 'direct', 'gmres'}, default 'auto'
            Solver for Newton steps:
            - 'auto': Choose based on model size (<200 states: direct, ≥200: gmres)
            - 'direct': Sparse direct solver (small models)
            - 'gmres': Iterative GMRES solver (large models)

        **Performance Configuration:**

        - 'use_cython' : {'auto', True, False}, default 'auto'
            Control Cython acceleration:
            - 'auto': Use Cython for models with >200 states
            - True: Always use Cython (if available)
            - False: Never use Cython

        - 'cython_threshold' : int, default 200
            State count threshold for automatic Cython enabling

        **Configuration Precedence:**

        1. Method parameters
        2. Instance config
        3. Built-in defaults

        **Configuration Examples:**

        # Instance-level configuration
        config = {
            'vf_max_iter': 1000,
            'vf_algorithm': 'value_iteration',
            'use_cython': True,
        }
        model = Model(n_players=2, n_demand=2, param=params, config=config)

        # Method-level override (for this call only)
        v, dv = model.value_function(vf_max_iter=100, vf_algorithm='polyalgorithm')
    """

    # ------------------------------------------------------------------------
    # Initialization & Setup
    # ------------------------------------------------------------------------

    def __init__(self, n_players: int, n_demand: int,
                 param: Dict[str, float], rho: float = DEFAULT_CONFIG['rho'],
                 verbose: bool = False,
                 config: Optional[Dict[str, Any]] = None) -> None:
        # Model parameters
        self.param_keys = ['theta_ec', 'theta_rn', 'theta_d', 'lambda', 'gamma']
        self._validate_parameters(n_players, n_demand, param, rho)
        self.param = param

        # Setup configuration: merge config with defaults
        self.config = self._setup_configuration(config)

        # Set basic attributes
        self.n_players = n_players
        self.n_demand = n_demand
        self.nobs = 0
        self.D = None
        self.verbose = verbose
        self.rho = rho
        self._convergence_history = None

        # Number of player configurations
        self.n_configs = 2**n_players
        # Total model states (combinations of player and demand states)
        self.K = self.n_configs * self.n_demand
        # Set up state space with precomputed addresses
        self._setup_state_space()
        self._setup_encodings()
        self._precompute_transitions()
        # Allocate sparse matrix storage
        self.nnz_intensity = self._count_intensity_nonzeros()
        self.nnz_dbellman_dv = self._count_dbellman_operator_dv_nonzeros()
        # Standard deviation of error
        self.sigma_eps = 1.0
        # Parameter bounds
        self.theta_bounds = [(-10.0, 0.0),  # θ_EC
                             (-5.0, 0.0),   # θ_RN
                             (0.0, 10.0),   # θ_D
                             (0.01, 10.0),  # λ
                             (0.01, 5.0)]   # γ
        # Print basic model information
        if verbose:
            print(self)

    def _validate_parameters(self, n_players: int, n_demand: int,
                             param: Dict[str, float], rho: float) -> None:
        """
        Validate model parameters.

        Parameters
        ----------
        n_players : int
            Number of players in the market
        n_demand : int
            Number of demand states
        param : dict
            Model parameters dictionary
        rho : float
            Discount rate
        """
        # Input validation
        if not isinstance(n_players, int) or n_players <= 0:
            raise ValueError(f"n_players must be a positive integer, got {n_players}")
        if not isinstance(n_demand, int) or n_demand <= 0:
            raise ValueError(f"n_demand must be a positive integer, got {n_demand}")
        if not isinstance(param, dict):
            raise TypeError(f"param must be a dictionary, got {type(param)}")

        # Validate required parameters
        missing_params = [p for p in self.param_keys if p not in param]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        # Validate parameter types and basic bounds
        for key in self.param_keys:
            value = param[key]
            if not isinstance(value, float) or not np.isfinite(value):
                raise ValueError(f"Parameter {key} must be a finite float, got {value}")
        if param['lambda'] <= 0:
            raise ValueError(f"lambda must be positive, got {param['lambda']}")
        if param['gamma'] <= 0:
            raise ValueError(f"gamma must be positive, got {param['gamma']}")

        # Validate discount rate
        if not isinstance(rho, float) or not np.isfinite(rho) or rho < 0:
            raise ValueError(f"rho must be a non-negative finite float, got {rho}")

    def _setup_configuration(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Setup configuration using simple dictionary merge with default config.

        Parameters
        ----------
        config : dict or None
            Configuration dictionary

        Returns
        -------
        config : dict
            Merged configuration dictionary
        """
        merged_config = DEFAULT_CONFIG.copy()
        if config is not None:
            if not isinstance(config, dict):
                raise TypeError(f"config must be a dictionary, got {type(config)}")
            merged_config.update(config)
        return merged_config

    def _setup_state_space(self):
        """
        Generate the complete state space.

        Each state is structured as:
        - A demand state (ranging from 0 to n_demand-1)
        - A tuple of player-specific states (each being either 0 or 1)

        The state space is organized with demand states grouped together:
        [(d=0, players), (d=0, players), ..., (d=1, players), ...]
        """
        # Generate all possible player configurations (2^n_players combinations)
        player_configs = np.array(list(itertools.product([0, 1], repeat=self.n_players)))

        # Create full state space: each demand state paired with each player config
        self.state_space = list(itertools.product(np.arange(self.n_demand), player_configs))

        # Extract demand and player information into separate arrays for efficient access
        self.demand_states = np.array([state[0] for state in self.state_space])
        self.player_states = np.array([state[1] for state in self.state_space])

        # Count number of active players per state
        self.n_active = np.sum(self.player_states, axis=1).astype(np.int32)

        # Create boolean activity indicators (shape: n_players × K)
        self.is_active = self.player_states.transpose().astype(np.bool_)
        self.is_inactive = ~self.is_active

    def _setup_encodings(self):
        """
        Set up mappings between states and integers.

        Creates bidirectional mappings for quick encoding and look-up.
        States are grouped by demand states.
        """
        n_states = self.K
        self.state_to_int = {}
        self.int_to_state = {}
        for k in range(n_states):
            demand_state = self.demand_states[k]
            player_state = tuple(self.player_states[k])
            state = (demand_state, player_state)
            self.state_to_int[state] = k
            self.int_to_state[k] = (demand_state, player_state)

    def _precompute_transitions(self):
        """
        Precompute transitions for efficient matrix construction.

        Populates switch_indices and demand transition mappings k_demand_up and
        k_demand_down.
        """
        # Initialize arrays to -1 to catch errors
        n_states = self.K
        switch_indices = np.full(shape=(self.n_players, n_states), fill_value=-1, dtype=np.int32)
        k_demand_up = np.full(shape=(n_states,), fill_value=-1, dtype=np.int32)
        k_demand_down = np.full(shape=(n_states,), fill_value=-1, dtype=np.int32)

        # Compute transition mappings
        for k in range(n_states):
            current_demand = self.demand_states[k]
            current_active = self.player_states[k]

            # Demand increase
            if current_demand < self.n_demand - 1:
                next_state = (current_demand + 1, tuple(current_active))
                next_state_index = self.state_to_int[next_state]
                k_demand_up[k] = next_state_index

            # Demand decrease
            if current_demand > 0:
                next_state = (current_demand - 1, tuple(current_active))
                next_state_index = self.state_to_int[next_state]
                k_demand_down[k] = next_state_index

            # Player transitions
            for m in range(self.n_players):
                next_active = current_active.copy()
                next_active[m] = 1 - next_active[m]
                next_state = (current_demand, tuple(next_active))
                next_state_index = self.state_to_int[next_state]

                # Store the next state index for player switching
                switch_indices[m, k] = next_state_index

        # Store switch_indices
        self.switch_indices = switch_indices

        # Continuation states after demand increase or decrease
        self.k_demand_up = k_demand_up
        self.k_demand_down = k_demand_down

        # Pre-compute transition masks (after k_demand arrays are populated)
        self._demand_up_valid = k_demand_up >= 0
        self._demand_down_valid = k_demand_down >= 0

    def update_parameters(self, theta: Union[List[float], np.ndarray]) -> None:
        """
        Update model parameters.

        Parameters
        ----------
        theta : array-like
            Parameter values in the same order as self.param_keys
        """
        if len(theta) != len(self.param_keys):
            raise ValueError(f"Expected {len(self.param_keys)} parameters, got {len(theta)}")

        for i, (key, value) in enumerate(zip(self.param_keys, theta)):
            if not isinstance(value, float) or not np.isfinite(value):
                raise ValueError(f"Parameter {key} (index {i}) must be a finite number, got {value}")
            if key == 'lambda' and value <= 0:
                raise ValueError(f"lambda must be positive, got {value}")
            if key == 'gamma' and value <= 0:
                raise ValueError(f"gamma must be positive, got {value}")
            self.param[key] = value

    # ------------------------------------------------------------------------
    # Value function solution
    # ------------------------------------------------------------------------

    def value_function(self, vf_max_iter: Optional[int] = None,
                       vf_tol: Optional[float] = None,
                       vf_algorithm: Optional[str] = None,
                       vf_rtol: Optional[float] = None,
                       vf_max_newton_iter: Optional[int] = None,
                       vf_newton_solver: Optional[str] = None,
                       v_init: Optional[np.ndarray] = None,
                       compute_derivatives: bool = True
                       ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """
        Compute the value function and its derivatives.

        Parameters
        ----------
        vf_max_iter : int, optional
            Maximum number of iterations.
        vf_tol : float, optional
            Convergence tolerance.
        vf_algorithm : {'value_iteration', 'polyalgorithm'}, optional
            Algorithm to use.
        vf_rtol : float, optional
            For polyalgorithm: norm ratio threshold for switching to Newton method.
        vf_max_newton_iter : int, optional
            Maximum Newton-Kantorovich iterations.
        vf_newton_solver : {'auto', 'direct', 'gmres'}, optional
            Solver for Newton steps and implicit derivatives.
        v_init : numpy.ndarray, optional
            Initial value function for iteration, shape (n_players, K).
            If None, initializes to zeros.
        compute_derivatives : bool, optional
            Whether to compute partial derivatives of v with respect to parameters.
            Default is True. Set to False to skip derivative computation.

        Returns
        -------
        v : numpy.ndarray
            Value function, shape (n_players, K)
        dv : dict or None
            Dictionary of partial derivatives of v with respect to model parameters.
            Keys are parameter names ('theta_ec', 'theta_rn', 'theta_d', 'lambda', 'gamma').
            Returns None if compute_derivatives is False.
        """
        # Resolve configuration: method params > instance config
        vf_max_iter = vf_max_iter if vf_max_iter is not None else self.config['vf_max_iter']
        vf_tol = vf_tol if vf_tol is not None else self.config['vf_tol']
        vf_algorithm = vf_algorithm if vf_algorithm is not None else self.config['vf_algorithm']
        vf_rtol = vf_rtol if vf_rtol is not None else self.config['vf_rtol']
        vf_max_newton_iter = vf_max_newton_iter if vf_max_newton_iter is not None else self.config['vf_max_newton_iter']
        vf_newton_solver = vf_newton_solver if vf_newton_solver is not None else self.config['vf_newton_solver']
        if vf_newton_solver != 'auto':
            selected_solver = vf_newton_solver
        else:
            selected_solver = 'direct' if self.K <= SPARSE_STATE_THRESHOLD else 'gmres'
        if self.verbose:
            print(f"Using algorithm: {vf_algorithm}, solver: {selected_solver}")

        # Use v_init if set and no explicit v_init passed
        if v_init is None and hasattr(self, '_v_init') and self._v_init is not None:
            v_init = self._v_init

        # Execute selected algorithm
        if vf_algorithm == 'value_iteration':
            v = self.value_iteration(vf_max_iter, vf_tol, v_init=v_init)
        else:
            v = self.polyalgorithm(vf_max_iter, vf_tol, vf_rtol,
                                   vf_max_newton_iter, selected_solver, v_init=v_init)

        # Solve for derivatives
        if compute_derivatives:
            dv = self.solve_implicit_derivatives(v, solver=selected_solver)
        else:
            dv = None

        return v, dv

    def value_iteration(self, vf_max_iter: Optional[int] = None,
                        vf_tol: Optional[float] = None,
                        v_init: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve the model using value iteration.

        Parameters
        ----------
        vf_max_iter : int, optional
            Maximum number of iterations
        vf_tol : float, optional
            Convergence tolerance for relative difference
        v_init : numpy.ndarray, optional
            Initial value function, shape (n_players, K).
            If None, initializes to zeros.

        Returns
        -------
        v : numpy.ndarray
            Value function, shape (n_players, K)
        """
        # Resolve parameters
        if vf_max_iter is None:
            vf_max_iter = self.config['vf_max_iter']
        if vf_tol is None:
            vf_tol = self.config['vf_tol']

        # Initialize value function
        if v_init is not None:
            v = v_init.copy()
        else:
            v = np.zeros(shape=(self.n_players, self.K))
        v_old = np.zeros(shape=(self.n_players, self.K))
        v_diff = np.inf
        iter_idx = 0

        # Print iteration status
        if self.verbose:
            print("Iter\t|v_diff|")
            print("----\t-------")

        # Iterate until convergence
        while (v_diff > vf_tol and iter_idx < vf_max_iter):
            # Store the old value function
            v_old = v.copy()

            # Apply the Bellman operator
            v = self.bellman_operator(v)

            # Calculate relative error
            v_diff = np.max(np.abs(v - v_old) / np.maximum(np.abs(v_old), EPSILON))

            # Print iteration status
            if self.verbose:
                print(f"{iter_idx}\t{v_diff:.6f}")

            iter_idx += 1

        if self.verbose:
            if iter_idx < vf_max_iter:
                print(f"Converged after {iter_idx} iterations")
                print(f"Final v_diff: {v_diff:.8e}")
            else:
                print(f"Maximum iterations ({vf_max_iter}) reached without convergence")
                print(f"Final v_diff: {v_diff:.8e}")

        # Store the last Bellman evaluation for reuse
        self._last_bellman_eval = v.copy()

        return v

    def polyalgorithm(self, vf_max_iter: int, vf_tol: float, vf_rtol: float,
                      vf_max_newton_iter: int, vf_newton_solver: str,
                      track_convergence: bool = False,
                      v_init: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Polyalgorithm: value iteration + Newton-Kantorovich.

        Runs value iteration first, then switches to Newton-Kantorovich
        if convergence is slow or fails.

        Parameters
        ----------
        vf_max_iter : int
            Maximum number of value iteration iterations
        vf_tol : float
            Convergence tolerance
        vf_rtol : float
            NFXP switching ratio
        vf_max_newton_iter : int
            Maximum Newton-Kantorovich iterations
        vf_newton_solver : str
            Linear solver for Newton steps
        track_convergence : bool, default False
            If True, track residuals and switching in self._convergence_history
        v_init : numpy.ndarray, optional
            Initial value function, shape (n_players, K).
            If None, initializes to zeros.

        Returns
        -------
        v : numpy.ndarray
            Converged value function, shape (n_players, K)
        """
        if self.verbose:
            print("Using algorithm: polyalgorithm")

        if track_convergence:
            tracking_data = {
                'residuals': [],
                'switch_iteration': None
            }

        # Phase 1: Value iteration
        if v_init is not None:
            v = v_init.copy()
        else:
            v = np.zeros((self.n_players, self.K))
        converged = False
        vi_iter = 0

        # Expected contraction rate for NFXP switching
        unifrate = self.param['lambda'] * self.n_players + 2 * self.param['gamma']
        beta = unifrate / (self.rho + unifrate)
        prev_diff = np.inf

        for vi_iter in range(vf_max_iter):
            v_old = v.copy()
            v = self.bellman_operator(v)
            diff = np.max(np.abs(v - v_old))

            if track_convergence:
                tracking_data['residuals'].append(diff)

            if diff < vf_tol:
                converged = True
                break

            # NFXP switching conditions
            if vi_iter >= MIN_MONITORING_ITERS:
                # Small absolute errors (relative to tolerance)
                if diff < 1e6 * vf_tol:
                    if track_convergence:
                        tracking_data['switch_iteration'] = vi_iter + 1
                    break

                # Core NFXP condition: ratio close to theoretical rate
                rate = diff / prev_diff if prev_diff > EPSILON else np.inf
                if not np.isnan(rate) and abs(beta - rate) < vf_rtol:
                    if track_convergence:
                        tracking_data['switch_iteration'] = vi_iter + 1
                    break

            prev_diff = diff

        # Phase 2: Newton-Kantorovich
        newton_iter = 0
        if not converged:
            if self.verbose:
                print("Phase 2: Newton-Kantorovich")

            n_states = self.n_players * self.K
            eye = identity(n_states, format='csc', dtype=np.float64)

            # Compute initial Bellman evaluation for Newton phase
            Tv = self.bellman_operator(v)

            for newton_iter in range(vf_max_newton_iter):
                residual = v - Tv
                residual_norm = np.max(np.abs(residual))
                if residual_norm < vf_tol:
                    break

                jacobian = self.dbellman_operator_dv(v)
                system_matrix = eye - jacobian
                system_matrix.eliminate_zeros()
                residual_flat = residual.ravel()
                try:
                    if vf_newton_solver == 'direct':
                        delta_flat = spsolve(system_matrix, residual_flat)
                    else:  # gmres
                        delta_flat, info = gmres(system_matrix, residual_flat,
                                                 **{_GMRES_TOL_PARAM: GMRES_TOLERANCE})
                        if info != 0:
                            delta_flat = spsolve(system_matrix, residual_flat)

                    v = v - delta_flat.reshape(v.shape)
                    newton_iter += 1

                    # Compute Tv for next iteration
                    Tv = self.bellman_operator(v)

                    if track_convergence:
                        new_residual = np.max(np.abs(v - Tv))
                        tracking_data['residuals'].append(new_residual)

                except Exception:
                    break

        # Store iteration counts for reporting
        self._last_phase1_iter = vi_iter + 1
        self._last_phase2_iter = newton_iter

        # Store the last Bellman evaluation for reuse
        if not converged:
            # Newton phase was run, Tv is already computed
            self._last_bellman_eval = Tv
        else:
            # Converged during value iteration, compute final Tv
            self._last_bellman_eval = self.bellman_operator(v)

        if track_convergence:
            self._convergence_history = {
                'residuals': np.array(tracking_data['residuals']),
                'switch_iteration': tracking_data['switch_iteration']
            }

        return v

    # ------------------------------------------------------------------------
    # Choice probabilities
    # ------------------------------------------------------------------------

    def choice_probabilities(self,
                             v: np.ndarray,
                             dv: Optional[Dict[str, np.ndarray]] = None
                             ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Calculate probabilities of switching for each player and state.

        Optionally computes derivatives of the probabilities.

        Must match choice_probabilities_cython in model_cython.pyx.

        Parameters
        ----------
        v : numpy.ndarray
            Array of utilities for continuing, shape (n_players, K)
        dv : dict, optional
            Dictionary of derivatives of v with respect to parameters

        Returns
        -------
        p : numpy.ndarray
            Array of switching probabilities, shape (n_players, K)
        dp : dict, optional
            Dictionary of derivatives of p with respect to parameters.
            Only returned if dv is provided.
        """
        # Use Cython if available; fall through to Python implementation
        if self.is_cython_enabled:
            try:
                p, dp, _ = choice_probabilities_cython(
                    v, dv, self.param['theta_ec'], self.n_players, self.K,
                    self.is_inactive, self.switch_indices.astype(np.int32)
                )
                return (p, dp) if dv is not None else p
            except (ImportError, AttributeError, TypeError, ValueError) as e:
                import warnings
                warnings.warn(
                    f"Cython choice_probabilities failed: {e}. Falling back to Python implementation.",
                    CythonFallbackWarning,
                    stacklevel=2
                )

        # Initialize storage
        p = np.zeros(shape=(self.n_players, self.K), dtype=np.float64)
        if dv is not None:
            dp = {key: np.zeros(shape=(self.n_players, self.K), dtype=np.float64)
                  for key in dv.keys()}

        # Instantaneous payoff parameters (entry cost)
        psi = np.zeros(shape=(self.n_players, self.K), dtype=np.float64)
        psi[self.is_inactive] = self.param['theta_ec']

        # Assign utilities for 'continue' action directly from v and utilities
        # for 'switch' action using precomputed switch indices
        for m in range(self.n_players):
            # Get indices of states following 'switch' action j = 1
            k_switch = self.switch_indices[m, :]

            # Calculate probabilities via logit formula
            v0 = v[m, :]
            v1 = v[m, k_switch] + psi[m, :]

            # Use numerically stable computation
            max_utility = np.maximum(v0, v1)
            exp_v0 = np.exp(v0 - max_utility)
            exp_v1 = np.exp(v1 - max_utility)
            denom = exp_v0 + exp_v1

            # Switch probability
            p[m, :] = exp_v1 / denom

            # Calculate derivatives if requested
            if dv is not None:
                # For the logistic function p = exp(u1)/(exp(u0) + exp(u1)),
                # the derivative is p*(1-p)*(dv1/dθ - dv0/dθ)
                pmp = p[m, :] * (1 - p[m, :])

                for key in dv.keys():
                    # Chain rule: ∂p/∂θ = ∂p/∂v_switch × ∂v_switch/∂θ
                    #                   - ∂p/∂v_continue × ∂v_continue/∂θ
                    dv0 = dv[key][m, :]
                    dv1 = dv[key][m, k_switch]

                    # Add direct partial derivative of entry cost ψ with respect to θ_EC
                    if key == 'theta_ec':
                        # ∂ψ/∂θ_EC = 1 for inactive players, 0 for active players
                        dv1_vec = dv1 + self.is_inactive[m, :].astype(float)
                        dp[key][m, :] = pmp * (dv1_vec - dv0)
                    else:
                        dp[key][m, :] = pmp * (dv1 - dv0)

        if dv is not None:
            return p, dp
        return p

    # ------------------------------------------------------------------------
    # Bellman operator
    # ------------------------------------------------------------------------

    def _bellman_components(
            self,
            v: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the numerator, rate sum, etc. for the Bellman operator.

        This is the core computation that both bellman_operator and
        dbellman_operator_dv need.  Separating it eliminates code
        duplication.

        Must match _bellman_components_cython in model_cython.pyx.

        Parameters
        ----------
        v : numpy.ndarray
            Current value function, shape (n_players, K)

        Returns
        -------
        numerator : numpy.ndarray
            Numerator values, shape (n_players, K)
        ratesum : numpy.ndarray
            Rate sum values, shape (K,)
        p : numpy.ndarray
            Choice probabilities, shape (n_players, K)
        psi : numpy.ndarray
            Entry costs, shape (n_players, K)
        """
        # Use Cython if available; fall through to Python implementation
        if self.is_cython_enabled:
            try:
                return _bellman_components_cython(
                    v=v,
                    theta_ec=self.param['theta_ec'],
                    theta_rn=self.param['theta_rn'],
                    theta_d=self.param['theta_d'],
                    gamma=self.param['gamma'],
                    lam=self.param['lambda'],
                    rho=self.rho,
                    n_players=self.n_players,
                    K=self.K,
                    is_active=self.is_active,
                    is_inactive=self.is_inactive,
                    n_active=self.n_active.astype(np.int32),
                    demand_states=self.demand_states.astype(np.int32),
                    k_demand_up=self.k_demand_up.astype(np.int32),
                    k_demand_down=self.k_demand_down.astype(np.int32),
                    switch_indices=self.switch_indices.astype(np.int32),
                )
            except (ImportError, AttributeError, TypeError, ValueError) as e:
                import warnings
                warnings.warn(
                    f"Cython _bellman_components failed: {e}. Falling back to Python implementation.",
                    CythonFallbackWarning,
                    stacklevel=2
                )

        # Named parameters
        theta_ec = self.param['theta_ec']
        theta_rn = self.param['theta_rn']
        theta_d = self.param['theta_d']
        gam = self.param['gamma']
        lam = self.param['lambda']

        # 1. Precompute ratesum (not including ρ):
        # ratesum[k] = sum of all player decision rates + sum of nature rates
        ratesum = np.full(self.K, self.n_players * lam, dtype=np.float64)
        ratesum[self._demand_up_valid] += gam
        ratesum[self._demand_down_valid] += gam

        # 2. Instantaneous payoffs ψ (entry costs)
        psi = np.zeros(shape=(self.n_players, self.K), dtype=np.float64)
        psi[self.is_inactive] = theta_ec

        # 3. Choice probabilities
        p = self.choice_probabilities(v)

        # 4. Vectorized flow payoffs for all players
        numerator = (self.is_active *
                     (theta_rn * self.n_active + theta_d * self.demand_states))

        # 5. Main Bellman update loop over players
        for i in range(self.n_players):
            # 6. Demand transitions
            mask = self._demand_up_valid
            numerator[i, mask] += gam * v[i, self.k_demand_up[mask]]

            mask = self._demand_down_valid
            numerator[i, mask] += gam * v[i, self.k_demand_down[mask]]

            # 7. Player transitions
            for m in range(self.n_players):
                if m != i:
                    # 8. Rival player actions
                    k_switch = self.switch_indices[m, :]
                    numerator[i, :] += lam * (p[m, :] * v[i, k_switch] + (1 - p[m, :]) * v[i, :])

                else:
                    # 9. Own actions
                    k_switch = self.switch_indices[i, :]
                    v0 = v[i, :]
                    v1 = v[i, k_switch] + psi[i, :]

                    # Calculate the log sum of exponentials
                    max_v = np.maximum(v0, v1)
                    logsumexp = max_v + np.log(np.exp(v0 - max_v) + np.exp(v1 - max_v))
                    numerator[i, :] += lam * logsumexp

        return numerator, ratesum, p, psi

    def bellman_operator(self, v: np.ndarray) -> np.ndarray:
        """
        Update the value function using the Bellman operator.

        Parameters
        ----------
        v : numpy.ndarray
            Current value function, shape (n_players, K)

        Returns
        -------
        v_new : numpy.ndarray
            Updated value function
        """
        # Use Cython if available; fall through to Python implementation
        if self.is_cython_enabled:
            try:
                v_new, p = bellman_operator_cython(
                    v=v,
                    theta_ec=self.param['theta_ec'],
                    theta_rn=self.param['theta_rn'],
                    theta_d=self.param['theta_d'],
                    gamma=self.param['gamma'],
                    lam=self.param['lambda'],
                    rho=self.rho,
                    n_players=self.n_players,
                    K=self.K,
                    is_active=self.is_active,
                    is_inactive=self.is_inactive,
                    n_active=self.n_active.astype(np.int32),
                    demand_states=self.demand_states.astype(np.int32),
                    k_demand_up=self.k_demand_up.astype(np.int32),
                    k_demand_down=self.k_demand_down.astype(np.int32),
                    switch_indices=self.switch_indices.astype(np.int32),
                )
                return v_new
            except (ImportError, AttributeError, TypeError, ValueError) as e:
                import warnings
                warnings.warn(
                    f"Cython bellman_operator failed: {e}. Falling back to Python implementation.",
                    CythonFallbackWarning,
                    stacklevel=2
                )

        # Use the core computation function
        numerator, ratesum, _, _ = self._bellman_components(v)

        # Normalization
        v_new = numerator / (self.rho + ratesum)

        return v_new

    def _count_dbellman_operator_dv_nonzeros(self) -> int:
        """
        Count the exact number of non-zero elements in the Jacobian ∂T/∂v.

        Returns
        -------
        nnz : int
            Exact number of non-zero elements
        """
        nnz = 0

        # Within-player effects for each player i
        for i in range(self.n_players):
            # Demand transitions
            nnz += np.sum(self._demand_up_valid)    # Demand up transitions
            nnz += np.sum(self._demand_down_valid)  # Demand down transitions

            # Player transitions for each state
            for k in range(self.K):
                # Other players' actions: 1 entry per other player
                nnz += (self.n_players - 1)

                # Own actions: 2 entries (effect on own state + effect on switch state)
                nnz += 2

        # Cross-player effects: for each pair (i, m) where i != m
        # Each state k contributes up to 2 entries (dp_dv0 and dp_dv1)
        cross_player_pairs = self.n_players * (self.n_players - 1)
        nnz += cross_player_pairs * self.K * 2

        return nnz

    def dbellman_operator_dv(self, v: np.ndarray) -> csc_matrix:
        """
        Compute the analytical Jacobian ∂T/∂v of the Bellman operator.

        Must match dbellman_operator_dv_cython in model_cython.pyx.

        Parameters
        ----------
        v : numpy.ndarray
            Value function, shape (n_players, K)

        Returns
        -------
        dT_dv : scipy.sparse.csc_matrix
            The Jacobian matrix ∂T/∂v, shape (K*n_players, K*n_players)
        """
        n_states = self.K * self.n_players

        # Use Cython if available; fall through to Python implementation
        if self.is_cython_enabled:
            try:
                # Get triplet arrays from Cython function
                rows, cols, data, nnz = dbellman_operator_dv_cython(
                    v=v,
                    theta_ec=self.param['theta_ec'],
                    theta_rn=self.param['theta_rn'],
                    theta_d=self.param['theta_d'],
                    lam=self.param['lambda'],
                    gam=self.param['gamma'],
                    rho=self.rho,
                    n_players=self.n_players,
                    K=self.K,
                    is_active=self.is_active,
                    is_inactive=self.is_inactive,
                    n_active=self.n_active.astype(np.int32),
                    demand_states=self.demand_states.astype(np.int32),
                    k_demand_up=self.k_demand_up.astype(np.int32),
                    k_demand_down=self.k_demand_down.astype(np.int32),
                    switch_indices=self.switch_indices.astype(np.int32),
                    demand_up_valid=self._demand_up_valid,
                    demand_down_valid=self._demand_down_valid,
                )
                # Create sparse matrix from triplets
                dT_dv = csc_matrix((data, (rows, cols)), shape=(n_states, n_states))
                return dT_dv

            except (ImportError, AttributeError, TypeError, ValueError) as e:
                import warnings
                warnings.warn(
                    f"Cython dbellman_operator_dv failed: {e}. Falling back to Python implementation.",
                    CythonFallbackWarning,
                    stacklevel=2
                )

        # Extract named parameters
        lam = self.param['lambda']
        gam = self.param['gamma']

        # Use the core computation function
        numerator_all, ratesum, p, psi = self._bellman_components(v)

        # Pre-allocate arrays for sparse matrix triplets
        nnz = self.nnz_dbellman_dv
        rows = np.empty(nnz, dtype=np.int32)
        cols = np.empty(nnz, dtype=np.int32)
        data = np.empty(nnz, dtype=np.float64)

        # Main loop for within-player effects
        nnz_idx = 0  # Current position in arrays
        for i in range(self.n_players):
            row_offset = i * self.K

            # Demand up transitions
            mask = self._demand_up_valid
            if np.any(mask):
                k_indices = np.where(mask)[0]
                for k in k_indices:
                    current_idx = row_offset + k
                    neighbor_idx = row_offset + self.k_demand_up[k]
                    rows[nnz_idx] = current_idx
                    cols[nnz_idx] = neighbor_idx
                    data[nnz_idx] = gam / (self.rho + ratesum[k])
                    nnz_idx += 1

            # Demand down transitions
            mask = self._demand_down_valid
            if np.any(mask):
                k_indices = np.where(mask)[0]
                for k in k_indices:
                    current_idx = row_offset + k
                    neighbor_idx = row_offset + self.k_demand_down[k]
                    rows[nnz_idx] = current_idx
                    cols[nnz_idx] = neighbor_idx
                    data[nnz_idx] = gam / (self.rho + ratesum[k])
                    nnz_idx += 1

            # Player transitions
            for k in range(self.K):
                current_idx = row_offset + k
                denom = self.rho + ratesum[k]
                diag_contrib = 0.0  # Diagonal contribution (derivatives w.r.t. v[i,k])

                # Other players' actions
                for m in range(self.n_players):
                    if m != i:
                        k_switch = self.switch_indices[m, k]
                        switch_idx = row_offset + k_switch

                        # Effect on switch state (when m switches)
                        rows[nnz_idx] = current_idx
                        cols[nnz_idx] = switch_idx
                        data[nnz_idx] = (lam * p[m, k]) / denom
                        nnz_idx += 1

                        # Effect on own state (when m continues)
                        diag_contrib += (lam * (1 - p[m, k])) / denom

                # Own action effects
                k_switch = self.switch_indices[i, k]
                switch_idx = row_offset + k_switch

                # Effect on own state (when i continues)
                diag_contrib += (lam * (1 - p[i, k])) / denom
                rows[nnz_idx] = current_idx
                cols[nnz_idx] = current_idx
                data[nnz_idx] = diag_contrib
                nnz_idx += 1

                # Effect on switch state (when i switches)
                rows[nnz_idx] = current_idx
                cols[nnz_idx] = switch_idx
                data[nnz_idx] = (lam * p[i, k]) / denom
                nnz_idx += 1

        # Cross-player effects
        for i in range(self.n_players):
            for m in range(self.n_players):
                if m != i:
                    nnz_idx = self._dbellman_operator_dv_cross_player_effects(
                        rows, cols, data, nnz_idx, v, i, m, lam, p, psi, ratesum
                    )

        # Create sparse matrix from triplets (use only filled portion)
        dT_dv = csc_matrix((data[:nnz_idx], (rows[:nnz_idx], cols[:nnz_idx])), shape=(n_states, n_states))

        return dT_dv

    def _dbellman_operator_dv_cross_player_effects(
            self, rows: np.ndarray, cols: np.ndarray, data: np.ndarray,
            nnz_idx: int, v: np.ndarray, i: int, m: int, lam: float,
            p: np.ndarray, psi: np.ndarray, ratesum: np.ndarray) -> int:
        """
        Compute cross-player derivative terms ∂T_i/∂v_m.

        The Bellman operator has form: T_i[k] = numerator_i[k] / (ρ + ratesum[k])

        Cross-player effects contribute to numerator:
        - numerator_i[k] += λ * p_m[k] * v_i[k_switch_m[k]] + λ * (1-p_m[k]) * v_i[k]

        The derivative simplifies to:
        ∂T_i[k]/∂v_m = (∂numerator_i[k]/∂v_m) / (ρ + ratesum[k])

        Where
        ∂numerator_i[k]/∂v_m = λ * (∂p_m/∂v_m) * (v_i[k_switch_m[k]] - v_i[k])

        Parameters
        ----------
        rows : numpy.ndarray
            Row indices for sparse matrix construction
        cols : numpy.ndarray
            Column indices for sparse matrix construction
        data : numpy.ndarray
            Data values for sparse matrix construction
        nnz_idx : int
            Current position in the triplet arrays
        v : numpy.ndarray
            Value function
        i : int
            Player i index
        m : int
            Player m index
        lam : float
            Lambda parameter
        p : numpy.ndarray
            Choice probabilities
        psi : numpy.ndarray
            Entry costs
        ratesum : numpy.ndarray
            Uniformized rates, shape (K,)

        Returns
        -------
        nnz_idx : int
            Updated nnz_idx
        """
        i_offset = i * self.K
        m_offset = m * self.K

        # States after player m switches
        k_switch_m = self.switch_indices[m, :]  # Shape (K,)

        # Use pre-computed choice probabilities
        p_m = p[m, :]

        # Partial derivatives of p_m with respect to v_m
        dp_dv0 = -p_m * (1 - p_m)  # ∂p_m/∂v_m(k)
        dp_dv1 = p_m * (1 - p_m)   # ∂p_m/∂v_m(k_switch)

        # Player i's values
        v_i_stay = v[i, :]
        v_i_switch = v[i, k_switch_m]
        v_i_diff = v_i_switch - v_i_stay

        # Vectorized computation of all cross effects
        # Compute denominators for all states
        denom = self.rho + ratesum  # Shape (K,)

        # Compute all values for both derivative terms
        # Term 1: ∂p_m/∂v_m(k) - affects column k
        vals_dv0 = lam * dp_dv0 * v_i_diff / denom  # Shape (K,)
        rows_dv0 = np.arange(i_offset, i_offset + self.K)
        cols_dv0 = m_offset + np.arange(self.K)

        # Term 2: ∂p_m/∂v_m(k_switch) - affects column k_switch_m[k]
        vals_dv1 = lam * dp_dv1 * v_i_diff / denom  # Shape (K,)
        rows_dv1 = np.arange(i_offset, i_offset + self.K)
        cols_dv1 = m_offset + k_switch_m

        # Stack both sets of triplets
        all_vals = np.concatenate([vals_dv0, vals_dv1])
        all_rows = np.concatenate([rows_dv0, rows_dv1])
        all_cols = np.concatenate([cols_dv0, cols_dv1])

        # Filter out zeros (important for sparsity)
        nonzero_mask = all_vals != 0
        all_vals = all_vals[nonzero_mask]
        all_rows = all_rows[nonzero_mask]
        all_cols = all_cols[nonzero_mask]

        # Add to triplet arrays
        n_new = len(all_vals)
        rows[nnz_idx:nnz_idx + n_new] = all_rows
        cols[nnz_idx:nnz_idx + n_new] = all_cols
        data[nnz_idx:nnz_idx + n_new] = all_vals
        nnz_idx += n_new

        return nnz_idx

    def dbellman_operator_dtheta(self, v: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute analytical partial derivatives ∂T/∂θ of the Bellman operator.

        Must match dbellman_operator_dtheta_cython in model_cython.pyx.

        Parameters
        ----------
        v : numpy.ndarray
            Value function, shape (n_players, K)

        Returns
        -------
        dT_dtheta : dict
            Dictionary of partial derivatives ∂T/∂θ for each parameter,
            each with shape (n_players, K)
        """
        # Use Cython if available; fall through to Python implementation
        if self.is_cython_enabled:
            try:
                dT_theta_ec, dT_theta_rn, dT_theta_d, dT_lambda, dT_gamma = dbellman_operator_dtheta_cython(
                    v=v,
                    theta_ec=self.param['theta_ec'],
                    theta_rn=self.param['theta_rn'],
                    theta_d=self.param['theta_d'],
                    gamma=self.param['gamma'],
                    lam=self.param['lambda'],
                    rho=self.rho,
                    n_players=self.n_players,
                    K=self.K,
                    is_active=self.is_active,
                    is_inactive=self.is_inactive,
                    n_active=self.n_active.astype(np.int32),
                    demand_states=self.demand_states.astype(np.int32),
                    k_demand_up=self.k_demand_up.astype(np.int32),
                    k_demand_down=self.k_demand_down.astype(np.int32),
                    switch_indices=self.switch_indices.astype(np.int32),
                )
                return {
                    'theta_ec': dT_theta_ec,
                    'theta_rn': dT_theta_rn,
                    'theta_d': dT_theta_d,
                    'lambda': dT_lambda,
                    'gamma': dT_gamma
                }
            except (ImportError, AttributeError, TypeError, ValueError) as e:
                import warnings
                warnings.warn(
                    f"Cython dbellman_operator_dtheta failed: {e}. Falling back to Python implementation.",
                    CythonFallbackWarning,
                    stacklevel=2
                )

        # Extract parameters
        lam = self.param['lambda']

        # Reuse computations from _bellman_components
        numerator, ratesum, p, psi = self._bellman_components(v)

        # Initialize results for all parameters
        dT_dtheta = {key: np.zeros_like(v) for key in self.param_keys}

        # Denominator (broadcast for all players)
        denom = self.rho + ratesum  # shape (K,)

        # ∂T/∂θ_RN: Only affects numerator (flow payoffs)
        dT_dtheta['theta_rn'] = (self.is_active * self.n_active) / denom

        # ∂T/∂θ_D: Only affects numerator (flow payoffs)
        dT_dtheta['theta_d'] = (self.is_active * self.demand_states) / denom

        # ∂T/∂γ: Affects demand transitions and denominator
        d_numerator_gam = np.zeros_like(v)
        mask_up = self._demand_up_valid
        mask_down = self._demand_down_valid
        if np.any(mask_up):
            d_numerator_gam[:, mask_up] = v[:, self.k_demand_up[mask_up]]
        if np.any(mask_down):
            d_numerator_gam[:, mask_down] += v[:, self.k_demand_down[mask_down]]

        # ∂ratesum/∂γ = (# of valid demand transitions for state k)
        d_ratesum_gam = np.zeros(self.K, dtype=np.float64)
        d_ratesum_gam[mask_up] = 1.0
        d_ratesum_gam[mask_down] += 1.0
        # Quotient rule
        dT_dtheta['gamma'] = (denom * d_numerator_gam - numerator * d_ratesum_gam) / (denom ** 2)

        # ∂T/∂λ: Affects player transitions and denominator
        d_numerator_lam = np.zeros_like(v)
        for m in range(self.n_players):
            k_switch = self.switch_indices[m, :]
            p_m = p[m, :]

            # Continuation value for rivals and own player
            for i in range(self.n_players):
                if i != m:
                    # Rival m: p_m * v_switch + (1-p_m) * v_stay
                    d_numerator_lam[i, :] += p_m * v[i, k_switch] + (1 - p_m) * v[i, :]
                else:
                    # Own player gets logsumexp
                    v0 = v[m, :]
                    v1 = v[m, k_switch] + psi[m, :]
                    max_v = np.maximum(v0, v1)
                    logsumexp = max_v + np.log(np.exp(v0 - max_v) + np.exp(v1 - max_v))
                    d_numerator_lam[m, :] += logsumexp

        # ∂ratesum/∂λ = n_players (constant for all states)
        d_ratesum_lam = self.n_players
        # Quotient rule
        dT_dtheta['lambda'] = (denom * d_numerator_lam - numerator * d_ratesum_lam) / (denom ** 2)

        # ∂T/∂θ_EC: Only affects numerator (via choice probabilities)
        d_numerator_ec = np.zeros_like(v)
        for m in range(self.n_players):
            k_switch_m = self.switch_indices[m, :]
            p_m = p[m, :]
            dp_m_dtheta_ec = p_m * (1 - p_m) * self.is_inactive[m, :]

            # Effect on other players
            for i in range(self.n_players):
                if i != m:
                    # Rival m's probability change affects expected continuation
                    d_numerator_ec[i, :] += lam * dp_m_dtheta_ec * (v[i, k_switch_m] - v[i, :])
                else:
                    # Own player effect: direct entry cost effect
                    d_numerator_ec[m, :] += lam * p_m * self.is_inactive[m, :]

        dT_dtheta['theta_ec'] = d_numerator_ec / denom

        return dT_dtheta

    def solve_implicit_derivatives(self, v: np.ndarray,
                                   solver: Optional[str] = 'direct'
                                   ) -> Dict[str, np.ndarray]:
        """
        Solve for value function derivatives using implicit differentiation.

        For the fixed point v = T(v, θ), we have:
        dv/dθ = (I - ∂T/∂v)^(-1) * ∂T/∂θ

        Parameters
        ----------
        v : numpy.ndarray
            Converged value function, shape (n_players, K)
        solver : str, optional
            Solver method: 'direct' (sparse direct) or 'gmres'

        Returns
        -------
        dv : dict
            Dictionary of value function partial derivatives ∂v/∂θ with
            respect to all parameters
        """
        K = self.K
        n_players = self.n_players
        jacobian_size = n_players * K

        # Compute analytical Jacobian ∂T/∂v.  This returns CSC format,
        # which we want for LU factorization.
        jacobian = self.dbellman_operator_dv(v)

        # Form system matrix (I - ∂T/∂v) in same format
        eye = identity(jacobian_size, format=jacobian.format, dtype=jacobian.dtype)
        system_matrix = eye - jacobian
        system_matrix.eliminate_zeros()

        # Compute analytical RHS: ∂T/∂θ for all parameters
        dT_dtheta_all = self.dbellman_operator_dtheta(v)

        # Solve for derivatives with respect to each parameter
        if solver == 'gmres':
            dv = {}
            for param in self.param_keys:
                rhs = dT_dtheta_all[param].ravel()
                solution, info = gmres(system_matrix, rhs,
                                       **{_GMRES_TOL_PARAM: GMRES_TOLERANCE})

                if info != 0:
                    raise RuntimeError(
                        f"GMRES solver failed to converge for parameter '{param}'. "
                        f"Try using solver='direct' or increase tolerance."
                    )

                dv[param] = solution.reshape((n_players, K))
        else:
            # Use direct sparse solve (default)
            # Ensure CSC format for efficient splu factorization
            if system_matrix.format != 'csc':
                system_matrix = system_matrix.tocsc()
            dv = spsolve_multiple_rhs(system_matrix, dT_dtheta_all)
            for param in self.param_keys:
                dv[param] = dv[param].reshape((n_players, K))

        return dv

    # ------------------------------------------------------------------------
    # Markov jump process
    # ------------------------------------------------------------------------

    def intensity_matrix(self) -> Tuple[csr_matrix, Dict[str, csr_matrix]]:
        """
        Compute the intensity matrix Q and its partial derivatives ∂Q/∂θ.

        This method constructs the continuous-time Markov chain transition
        intensity matrix and calculates partial derivatives with respect to all
        model parameters, accounting for the indirect effects through the value
        function derivatives via the chain rule.

        Uses triplet approach with pre-allocated arrays for efficiency.

        Returns
        -------
        Q : scipy.sparse.csr_matrix
            Intensity matrix
        dQ : dict
            Dictionary of partial derivatives of Q with respect to each θ
        """
        # Extract named parameters for clarity
        lam = self.param['lambda']
        gam = self.param['gamma']

        # Get the value function and its derivatives
        v, dv = self.value_function()

        # Compute choice probabilities and their derivatives using chain rule
        p, dp = self.choice_probabilities(v, dv)

        # Use pre-calculated number of non-zero elements
        nnz = self.nnz_intensity

        # Pre-allocate triplet arrays
        rows = np.zeros(nnz, dtype=np.int32)
        cols = np.zeros(nnz, dtype=np.int32)
        data = np.zeros(nnz, dtype=np.float64)
        dQ_data = {key: np.zeros(nnz, dtype=np.float64) for key in self.param_keys}

        # Track current position in arrays
        idx = 0

        # Track row sums for diagonal elements
        row_sums = np.zeros(self.K, dtype=np.float64)
        row_sums_deriv = {key: np.zeros(self.K, dtype=np.float64) for key in self.param_keys}

        # Build off-diagonal elements
        for from_idx in range(self.K):
            from_demand = self.demand_states[from_idx]
            from_players = self.player_states[from_idx]

            # Demand transitions
            if from_demand < self.n_demand - 1:
                # Demand increase
                to_idx = self.k_demand_up[from_idx]
                rows[idx] = from_idx
                cols[idx] = to_idx
                data[idx] = gam
                dQ_data['gamma'][idx] = 1.0
                row_sums[from_idx] += gam
                row_sums_deriv['gamma'][from_idx] += 1.0
                idx += 1

            if from_demand > 0:
                # Demand decrease
                to_idx = self.k_demand_down[from_idx]
                rows[idx] = from_idx
                cols[idx] = to_idx
                data[idx] = gam
                dQ_data['gamma'][idx] = 1.0
                row_sums[from_idx] += gam
                row_sums_deriv['gamma'][from_idx] += 1.0
                idx += 1

            # Player transitions
            for player_idx in range(self.n_players):
                to_idx = self.switch_indices[player_idx, from_idx]
                rows[idx] = from_idx
                cols[idx] = to_idx

                if from_players[player_idx] == 0:
                    # Entry: inactive -> active
                    rate = lam * p[player_idx, from_idx]
                    data[idx] = rate
                    row_sums[from_idx] += rate

                    # Partial derivatives
                    for key in self.param_keys:
                        if key == 'lambda':
                            # Direct effect + indirect effect via ∂p/∂λ
                            deriv = p[player_idx, from_idx] + lam * dp[key][player_idx, from_idx]
                            dQ_data[key][idx] = deriv
                            row_sums_deriv[key][from_idx] += deriv
                        else:
                            # Only indirect effect via ∂p/∂θ
                            deriv = lam * dp[key][player_idx, from_idx]
                            dQ_data[key][idx] = deriv
                            row_sums_deriv[key][from_idx] += deriv

                else:
                    # Exit: active -> inactive
                    rate = lam * (1 - p[player_idx, from_idx])
                    data[idx] = rate
                    row_sums[from_idx] += rate

                    # Partial derivatives
                    for key in self.param_keys:
                        if key == 'lambda':
                            # Direct effect - indirect effect via ∂p/∂λ (note negative sign)
                            deriv = (1 - p[player_idx, from_idx]) - lam * dp[key][player_idx, from_idx]
                            dQ_data[key][idx] = deriv
                            row_sums_deriv[key][from_idx] += deriv
                        else:
                            # Only negative indirect effect via ∂p/∂θ
                            deriv = -lam * dp[key][player_idx, from_idx]
                            dQ_data[key][idx] = deriv
                            row_sums_deriv[key][from_idx] += deriv

                idx += 1

        # Add diagonal elements (negative row sums)
        for i in range(self.K):
            rows[idx] = i
            cols[idx] = i
            data[idx] = -row_sums[i]
            for key in self.param_keys:
                dQ_data[key][idx] = -row_sums_deriv[key][i]
            idx += 1

        # Create sparse matrices from triplets
        Q = csr_matrix((data, (rows, cols)), shape=(self.K, self.K))

        dQ = {}
        for key in self.param_keys:
            dQ[key] = csr_matrix((dQ_data[key], (rows, cols)), shape=(self.K, self.K))

        return Q, dQ

    def _count_intensity_nonzeros(self) -> int:
        """
        Count the number of non-zero elements in the intensity matrix.

        Returns
        -------
        nnz : int
            Number of non-zero elements
        """
        nnz = 0

        for from_idx in range(self.K):
            from_demand = self.demand_states[from_idx]

            # Count demand transitions
            if from_demand < self.n_demand - 1:  # Demand up
                nnz += 1
            if from_demand > 0:  # Demand down
                nnz += 1

            # Count player transitions (always n_players transitions per state)
            nnz += self.n_players

            # Count diagonal element
            nnz += 1

        return nnz

    # ------------------------------------------------------------------------
    # Data generating process
    # ------------------------------------------------------------------------

    def discrete_time_dgp(self, n_obs: int,
                          Delta: Optional[float] = 1.0,
                          seed: Optional[int] = 20180120) -> List[int]:
        """
        Generate discrete-time simulated data from the model.

        Parameters
        ----------
        n_obs : int
            Number of observations to generate
        Delta : float, optional
            Time between observations
        seed : int, optional
            Random seed

        Returns
        -------
        sample : list
            Sequence of simulated state indices
        """
        # Input validation
        if not isinstance(n_obs, int) or n_obs <= 0:
            raise ValueError(f"n_obs must be a positive integer, got {n_obs}")
        if not isinstance(Delta, float) or not np.isfinite(Delta) or Delta <= 0.0:
            raise ValueError(f"Delta must be a positive finite number, got {Delta}")
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise ValueError(f"seed must be a non-negative integer or None, got {seed}")

        # Calculate the transition matrix using matrix exponential
        Q, _ = self.intensity_matrix()
        P, _ = vexpm_deriv(Q, {}, Delta, np.eye(self.K))

        # Compute the invariant distribution as the left eigenvector corresponding to eigenvalue 1
        values, left_vectors = eig(P.T, left=True, right=False)
        invariant_distribution = np.abs(left_vectors[:, np.isclose(values, 1.0)].flatten().real)
        invariant_distribution /= invariant_distribution.sum()

        # Generate the initial state
        np.random.seed(seed)
        initial_state = np.random.choice(np.arange(self.K), p=invariant_distribution)

        # Simulate the Markov Chain
        sample = [initial_state]
        for _ in range(1, n_obs):
            current_state = sample[-1]
            next_state = np.random.choice(np.arange(self.K), p=P[current_state])
            sample.append(next_state)

        return sample

    # ------------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------------

    def preprocess_data(self, sample: List[int]) -> None:
        """
        Preprocess data to compute the transition count matrix.

        This function is called automatically from the log_likelihood function.

        Parameters
        ----------
        sample : array-like
            Sequence of observed states
        """
        # Caching - skip if already processed
        if hasattr(self, 'D') and self.D is not None:
            return

        # Transition count matrix D
        D = lil_matrix((self.K, self.K), dtype=int)
        for i in range(len(sample) - 1):
            from_state = sample[i]
            to_state = sample[i + 1]
            D[from_state, to_state] += 1

        # Convert to csc_matrix for efficient column access
        self.D = D.tocsc()
        self.nobs = len(sample)

    def log_likelihood(self, theta: np.ndarray, sample: List[int],
                       Delta: Optional[float] = 1.0,
                       grad: Optional[bool] = True
                       ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Compute the log-likelihood and its gradient for discrete time data.

        This function implements the log-likelihood for discrete-time
        observations of a continuous-time Markov process, using the matrix
        exponential with analytical partial derivatives.

        Parameters
        ----------
        theta : array-like
            Parameter vector
        sample : array-like
            Sequence of observed states
        Delta : float, optional
            Time between observations
        grad : bool, optional
            Whether to compute the gradient

        Returns
        -------
        ll : float
            Log-likelihood value
        g : array-like, optional
            Gradient of log-likelihood, returned if grad=True
        """
        # Input validation
        if len(theta) != len(self.param_keys):
            raise ValueError(f"theta must have length {len(self.param_keys)}, got length {len(theta)}")
        if not all(isinstance(t, float) and np.isfinite(t) for t in theta):
            raise ValueError("theta must contain finite numbers")
        if len(sample) < 2:
            raise ValueError("sample must contain at least 2 observations")
        if not all(isinstance(s, (int, np.integer)) and 0 <= s < self.K for s in sample):
            raise ValueError(f"sample must contain valid state indices (0 to {self.K-1})")
        if not isinstance(Delta, float) or not np.isfinite(Delta) or Delta <= 0:
            raise ValueError(f"Delta must be a positive finite number, got {Delta}")
        if not isinstance(grad, bool):
            raise ValueError(f"grad must be a boolean, got {grad}")

        # Update model parameters
        if self.verbose:
            print(f"Log likelihood evaluation for theta = {theta}...")
        self.update_parameters(theta)

        # Preprocess data
        self.preprocess_data(sample)

        # Compute the log likelihood and its partial derivatives
        ll = 0
        if grad:
            dll = np.zeros(len(self.param_keys), dtype=np.float64)

        # Find states with > 0 incoming transitions
        _, nonzero_cols = self.D.nonzero()
        to_states = np.unique(nonzero_cols)

        # Precompute and store the intensity matrix and its partial derivatives
        Q, dQ = self.intensity_matrix()

        # Basis vector
        e_l = np.zeros(self.K, dtype=np.float64)

        # Process only states that have transitions
        for l in to_states:
            # Compute the l-th column of the transition probability matrix P
            # and, optionally, its partial derivatives
            e_l[l] = 1.0  # Basis vector for state l
            if grad:
                P_l, dP_l = vexpm_deriv(Q, dQ, Delta, e_l)
            else:
                P_l = vexpm(Q, Delta, e_l)
            e_l[l] = 0  # Reset basis vector

            # Get transition counts for this column (sparse)
            counts_l = self.D[:,l]

            # Add to log-likelihood, handling numerical stability
            log_P_l = np.log(np.maximum(P_l, EPSILON))

            # Efficient sparse computation: only access non-zero elements
            start_idx = counts_l.indptr[0]
            end_idx = counts_l.indptr[1]
            if end_idx <= start_idx:  # No non-zero elements
                continue
            indices = counts_l.indices[start_idx:end_idx]
            data = counts_l.data[start_idx:end_idx]
            ll += np.sum(data * log_P_l[indices])

            # Compute gradient if needed
            if grad:
                # For each parameter
                for i, param_key in enumerate(self.param_keys):
                    # The partial derivative of log(P) is ∂P/P,
                    # so be careful with small values of P
                    valid_idx = P_l > EPSILON
                    if np.any(valid_idx):
                        dlog_P = np.zeros_like(P_l)
                        dlog_P[valid_idx] = dP_l[param_key][valid_idx] / P_l[valid_idx]

                        # Efficient sparse computation for gradient
                        if end_idx > start_idx:  # Has non-zero elements
                            dll[i] += np.sum(data * dlog_P[indices])

        # Normalize by the number of observations
        ll /= self.nobs
        if grad:
            dll /= self.nobs

        if self.verbose:
            print(f"LL = {ll}")
            if grad:
                print(f"grad = {dll}")

        if grad:
            return ll, dll
        else:
            return ll

    def estimate_parameters(self, sample: List[int], Delta: float = 1.0,
                            max_iter: Optional[int] = None,
                            start: Optional[np.ndarray] = None,
                            use_grad: bool = True,
                            v_init: Optional[np.ndarray] = None,
                            ) -> OptimizeResult:
        """
        Estimate model parameters from observed data.

        Parameters
        ----------
        sample : array-like
            Sequence of observed states
        Delta : float, optional
            Time interval between observations
        max_iter : int, optional
            Maximum number of optimization iterations.
        start : array-like, optional
            Initial parameter values. If None, uses 0.1 for all parameters.
        use_grad : bool, optional
            Whether to use analytical gradients in optimization
        v_init : numpy.ndarray, optional
            Initial value function for equilibrium solver, shape (n_players, K).
            If provided, will be used as starting point for value function iteration.

        Returns
        -------
        results : scipy.optimize.OptimizeResult
            Optimization results containing:
            - x : array, estimated parameters in order
            - fun : float, negative log-likelihood at optimum
            - success : bool, whether optimization converged
            - nit : int, number of iterations used
            - message : str, optimization termination message
        """
        # Input validation
        if len(sample) == 0:
            raise ValueError("sample must be a non-empty sequence of states")
        if not all(isinstance(s, (int, np.integer)) and 0 <= s < self.K for s in sample):
            raise ValueError(f"sample must contain valid state indices (0 to {self.K-1})")
        if not isinstance(Delta, float) or not np.isfinite(Delta) or Delta <= 0:
            raise ValueError(f"Delta must be a positive finite number, got {Delta}")
        if max_iter is not None and (not isinstance(max_iter, int) or max_iter <= 0):
            raise ValueError(f"max_iter must be a positive integer or None, got {max_iter}")
        if start is not None:
            if len(start) != len(self.param_keys):
                raise ValueError(f"start must have length {len(self.param_keys)}, got {len(start)}")
            if not all(isinstance(s, (int, float)) and np.isfinite(s) for s in start):
                raise ValueError("start must contain finite numbers")
        if not isinstance(use_grad, bool):
            raise ValueError(f"use_grad must be a boolean, got {use_grad}")

        # Store v_init for use by value_function()
        self._v_init = v_init

        # Preprocess data
        self.preprocess_data(sample)

        # Get optimization max_iter from parameter or instance config
        opt_max_iter = max_iter if max_iter is not None else self.config.get('opt_max_iter', DEFAULT_CONFIG['opt_max_iter'])

        # Initial parameter values
        if start is not None:
            initial_params = start
        else:
            initial_params = np.array([-0.1, -0.1, 0.1, 0.1, 0.1])

        # Define objective function for optimization
        def objective(theta, sample, Delta, grad=True):
            if grad:
                ll, g = self.log_likelihood(theta, sample, Delta, grad=True)
                return -ll, -g
            else:
                ll = self.log_likelihood(theta, sample, Delta, grad=False)
                return -ll

        # Optimization options
        options = {
            'disp': None,
            'maxiter': opt_max_iter,
            'gtol': DEFAULT_CONFIG['opt_tolerance'],
            'ftol': DEFAULT_CONFIG['opt_tolerance'],
        }

        # Run optimization
        results = minimize(
            objective,
            initial_params,
            args=(sample, Delta, use_grad),
            method='L-BFGS-B',
            jac=use_grad,
            bounds=self.theta_bounds,
            options=options
        )

        return results

    # ------------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------------

    @property
    def is_cython_enabled(self) -> bool:
        """
        Check if Cython optimization is available and will be used.

        Returns
        -------
        enabled : bool
            True if Cython is available and should be used based on config
        """
        if not CYTHON_AVAILABLE:
            return False

        use_cython = self.config.get('use_cython', 'auto')

        if use_cython is True:
            return True
        elif use_cython is False:
            return False
        else:  # use_cython == 'auto'
            threshold = self.config.get('cython_threshold', CYTHON_STATE_THRESHOLD)
            return self.K > threshold

    def __str__(self) -> str:
        """Return string representation of the model."""
        str = f"""
n_players: {self.n_players}
n_demand: {self.n_demand}

K: {self.K}
Q dimension: {self.K} x {self.K} ({self.K * self.K} elements)
Q non-zeros: {self.nnz_intensity} ({self.nnz_intensity / (self.K * self.K) * 100:.2f}%)

Configuration:
"""
        for key in self.config:
            key_colon = key + ":"
            str += f"  {key_colon:<20}{self.config[key]}\n"
        str += "\n"

        if CYTHON_AVAILABLE:
            if self.is_cython_enabled:
                use_cython = self.config.get('use_cython', 'auto')
                if use_cython is True:
                    str += f"Cython enabled (by config) for {self.K} states"
                else:
                    str += f"Cython enabled for {self.K} states"
            else:
                use_cython = self.config.get('use_cython', 'auto')
                if use_cython is False:
                    str += f"Cython disabled (by config) for {self.K} states"
                else:
                    str += f"Cython available but not used for small model ({self.K} states)"
        else:
            str += "Cython not available\n"
            str += "Build with: python setup_cython.py build_ext --inplace"

        return str

    def __repr__(self) -> str:
        """Return repr string for the model."""
        return (f"Model(n_players={self.n_players}, n_demand={self.n_demand})")
