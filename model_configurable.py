"""
Configurable Model implementation for benchmarking optimizations.
"""

import itertools
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from scipy.linalg import expm
from scipy.optimize import minimize, OptimizeResult
from scipy.sparse import lil_matrix

from optimization_config import OptimizationConfig
from model import (
    Model,
    DEFAULT_CONFIG,
    EPSILON,
)


class ConfigurableModel:
    """
    Model with selectable optimizations for benchmarking.

    This class implements the continuous-time dynamic discrete choice entry/exit
    game with configurable optimizations.  It provides its own sequential
    and dense-matrix implementations.  For later optimizations, it delegates
    methods to the optimized Model class.  Since this class is not for
    production use, we do very little input validation.

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
    config : OptimizationConfig, optional
        Configuration specifying which optimizations to enable.
        If None, uses OptimizationConfig.full() (all optimizations).
    """

    def __init__(self, n_players: int, n_demand: int,
                 param: Dict[str, float], rho: float = DEFAULT_CONFIG['rho'],
                 verbose: bool = False,
                 config: Optional[OptimizationConfig] = None):
        # Store basic attributes
        self.n_players = n_players
        self.n_demand = n_demand
        self.param = param.copy()
        self.rho = rho
        self.verbose = verbose

        # Set configuration
        self.config = config or OptimizationConfig.full()

        # Parameter keys and bounds for estimation
        self.param_keys = ['theta_ec', 'theta_rn', 'theta_d', 'lambda', 'gamma']
        self.theta_bounds = [(-10.0, 0.0),  # θ_EC
                             (-5.0, 0.0),   # θ_RN
                             (0.0, 10.0),   # θ_D
                             (0.01, 10.0),  # λ
                             (0.01, 5.0)]   # γ

        # Data preprocessing attributes (will be set by preprocess_data)
        self.D = None  # Transition count matrix
        self.nobs = None  # Number of observations

        # Compute state space dimensions
        self.K = self.n_demand * (2 ** self.n_players)
        self.n_configs = 2 ** self.n_players

        # Initialize Model implementation if needed (i.e., if any optimization
        # beyond baseline is enabled)
        if self._uses_model_for_vf_or_ll():
            model_config = self._create_model_config()
            self._impl = Model(n_players, n_demand, param, rho, verbose, model_config)
        else:
            self._impl = None  # baseline: No Model delegation
        self.setup_state_space()

    def _uses_model_for_vf_or_ll(self) -> bool:
        """
        Determine if we need Model instance for any operations.
        We need Model if any optimization beyond baseline is enabled.
        """
        return (self.config.vectorize or
                self.config.polyalgorithm or
                self.config.cython or
                self.config.sparse or
                self.config.derivatives)

    def _use_model_for_vf_ccp(self) -> bool:
        """
        Determine routing for VF & CCP methods (bellman_operator,
        value_function, choice_probabilities).

        According to routing table:
        - baseline: ConfigurableModel
        - vectorize, polyalgorithm, cython, sparse, derivatives: Model
        """
        return (self.config.vectorize or
                self.config.polyalgorithm or
                self.config.cython or
                self.config.sparse or
                self.config.derivatives)

    def _use_model_for_q_ll(self) -> bool:
        """
        Determine routing for log_likelihood and intensity_matrix.

        According to routing table:
        - baseline, vectorize, polyalgorithm, cython: ConfigurableModel
        - sparse, derivatives: Model
        """
        return (self.config.sparse or
                self.config.derivatives)

    def _create_model_config(self) -> Dict[str, Any]:
        """
        Create configuration dict for Model class based on OptimizationConfig.
        """
        model_config = {}

        # Set specific configurations based on highest optimization level
        if self.config.derivatives:
            # derivatives level (highest)
            model_config['vf_algorithm'] = 'polyalgorithm'
            model_config['use_cython'] = True
            model_config['cython_threshold'] = 0
        elif self.config.sparse:
            # sparse level
            model_config['vf_algorithm'] = 'polyalgorithm'
            model_config['use_cython'] = True
            model_config['cython_threshold'] = 0
        elif self.config.cython:
            # cython level
            model_config['vf_algorithm'] = 'polyalgorithm'
            model_config['use_cython'] = True
            model_config['cython_threshold'] = 0
        elif self.config.polyalgorithm:
            # polyalgorithm level
            model_config['vf_algorithm'] = 'polyalgorithm'
            model_config['use_cython'] = False
        elif self.config.vectorize:
            model_config['vf_algorithm'] = 'value_iteration'
            model_config['use_cython'] = False

        # Choose solver based on size for polyalgorithm
        if model_config.get('vf_algorithm') == 'polyalgorithm':
            if self.K < 200:
                model_config['vf_newton_solver'] = 'direct'
            else:
                model_config['vf_newton_solver'] = 'gmres'

        return model_config

    def setup_state_space(self):
        """Setup state space using simple dictionaries."""
        # Generate all possible states
        player_configs = list(itertools.product([0, 1], repeat=self.n_players))
        demand_states = list(range(self.n_demand))

        # Create state space: (demand, players)
        self.state_space = []
        for demand in demand_states:
            for players in player_configs:
                self.state_space.append((demand, players))

        # Create state lookup dictionaries
        self.state_to_int = {state: i for i, state in enumerate(self.state_space)}
        self.int_to_state = {i: state for i, state in enumerate(self.state_space)}

        if self.verbose:
            print(f"ConfigurableModel (baseline): {self.n_players} players, "
                  f"{self.n_demand} demand states, {self.K} total states")

    def decode_state(self, k: int) -> Tuple[int, Tuple[int, ...]]:
        """Decode integer state index to (demand, players) tuple."""
        if self._impl is not None:
            return self._impl.decode_state(k)
        else:
            return self.int_to_state[k]

    def encode_state(self, demand: int, players: Tuple[int, ...]) -> int:
        """Encode (demand, players) tuple to integer state index."""
        if self._impl is not None:
            return self._impl.encode_state(demand, players)
        else:
            return self.state_to_int[(demand, players)]

    def choice_probabilities(
            self, v: np.ndarray,
            dv: Optional[Dict[str, np.ndarray]] = None
            ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Compute choice probabilities with optional derivatives.
        """
        if self._use_model_for_vf_ccp():
            # Delegate to Model
            if dv is None:
                return self._impl.choice_probabilities(v)
            else:
                return self._impl.choice_probabilities(v, dv)

        # Sequential implementation (baseline only)
        p = np.zeros((self.n_players, self.K))

        for i in range(self.n_players):
            for k in range(self.K):
                state = self.decode_state(k)
                demand, players = state

                # Value of staying
                v_stay = v[i, k]

                # Value of switching - use explicit lookup
                new_players = list(players)
                new_players[i] = 1 - new_players[i]
                k_switch = self.encode_state(demand, tuple(new_players))
                v_switch = v[i, k_switch]

                # Add entry cost if inactive
                if players[i] == 0:
                    v_switch += self.param['theta_ec']

                # Compute probability (stable)
                v_max = max(v_stay, v_switch)
                exp_stay = np.exp(v_stay - v_max)
                exp_switch = np.exp(v_switch - v_max)
                p[i, k] = exp_switch / (exp_stay + exp_switch)

        return p

    def bellman_operator(self, v: np.ndarray) -> np.ndarray:
        """
        Apply Bellman operator to update value function.
        """
        if self._use_model_for_vf_ccp():
            # Delegate to Model's bellman_operator
            v_new = self._impl.bellman_operator(v)
            return v_new

        # Sequential implementation (baseline only)
        v_new = np.zeros_like(v)

        # Get choice probabilities
        p = self.choice_probabilities(v)

        # Parameters
        theta_ec = self.param['theta_ec']
        theta_rn = self.param['theta_rn']
        theta_d = self.param['theta_d']
        lam = self.param['lambda']
        gamma = self.param['gamma']

        # Main Bellman update loop
        for i in range(self.n_players):
            for k in range(self.K):
                state = self.decode_state(k)
                demand, players = state

                # Initialize accumulator
                value = 0.0
                rate_sum = self.rho

                # Flow payoffs (if active)
                if players[i] == 1:
                    n_active = sum(players)
                    value += theta_rn * n_active + theta_d * demand

                # Demand transitions
                if demand < self.n_demand - 1:
                    k_up = self.encode_state(demand + 1, players)
                    value += gamma * v[i, k_up]
                    rate_sum += gamma
                if demand > 0:
                    k_down = self.encode_state(demand - 1, players)
                    value += gamma * v[i, k_down]
                    rate_sum += gamma

                # Player transitions
                for m in range(self.n_players):
                    if m != i:
                        # Other players
                        new_players = list(players)
                        new_players[m] = 1 - new_players[m]
                        k_switch = self.encode_state(demand, tuple(new_players))
                        value += lam * p[m, k] * v[i, k_switch]
                        rate_sum += lam * p[m, k]
                    else:
                        # Own transitions
                        new_players = list(players)
                        new_players[i] = 1 - new_players[i]
                        k_switch = self.encode_state(demand, tuple(new_players))
                        v_stay = v[i, k]
                        v_switch = v[i, k_switch]
                        if players[i] == 0:
                            v_switch += theta_ec

                        # Log-sum-exp
                        v_max = max(v_stay, v_switch)
                        logsumexp = v_max + np.log(np.exp(v_stay - v_max) + np.exp(v_switch - v_max))
                        value += lam * logsumexp
                        rate_sum += lam

                # Normalize
                v_new[i, k] = value / rate_sum

        return v_new

    def value_function(self,
                       vf_max_iter: int = DEFAULT_CONFIG['vf_max_iter'],
                       vf_tol: float = DEFAULT_CONFIG['vf_tol'],
                       vf_algorithm: Optional[str] = None,
                       vf_rtol: Optional[float] = None,
                       vf_max_newton_iter: Optional[int] = None,
                       vf_newton_solver: Optional[str] = None,
                       **kwargs) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute the value function and its derivatives.

        This method provides a consistent API with Model.value_function().
        Depending on the optimization configuration, it either uses baseline
        value iteration (levels 0-1) or delegates to the optimized Model class.

        Parameters
        ----------
        vf_max_iter : int, optional
            Maximum number of iterations. Default: 5000
        vf_tol : float, optional
            Convergence tolerance. Default: 1e-13
        vf_algorithm : str, optional
            Algorithm to use ('value_iteration' or 'polyalgorithm').
            Only used when delegating to Model class.
        vf_rtol : float, optional
            NFXP switching ratio for polyalgorithm. Only used with Model.
        vf_max_newton_iter : int, optional
            Maximum Newton iterations. Only used with Model.
        vf_newton_solver : str, optional
            Solver for Newton steps. Only used with Model.

        Returns
        -------
        v : numpy.ndarray
            Value function, shape (n_players, K)
        dv : dict
            Dictionary of derivatives. Empty dict for baseline implementation,
            full derivatives when delegating to Model class.
        """
        if self._use_model_for_vf_ccp():
            # Delegate to Model
            return self._impl.value_function(
                vf_max_iter=vf_max_iter,
                vf_tol=vf_tol,
                vf_algorithm=vf_algorithm,
                vf_rtol=vf_rtol,
                vf_max_newton_iter=vf_max_newton_iter,
                vf_newton_solver=vf_newton_solver,
                **kwargs)

        # Sequential implementation (baseline only)

        # Initialize value iteration
        v = np.zeros((self.n_players, self.K))

        # Value iteration
        for iteration in range(vf_max_iter):
            v_new = self.bellman_operator(v)
            # Check convergence
            diff = np.max(np.abs(v_new - v))
            if diff < vf_tol:
                if self.verbose:
                    print(f"Value function converged in {iteration + 1} iterations")
                break
            v = v_new
        else:
            if self.verbose:
                print(f"Value function did not converge in {vf_max_iter} iterations")

        return v, {}

    def intensity_matrix(self) -> np.array:
        """
        Compute (dense) intensity matrix Q.
        """
        # Route according to Q & LL routing table (same as log_likelihood)
        if self._use_model_for_q_ll():
            # sparse and derivatives use Model's intensity_matrix
            Q, _ = self._impl.intensity_matrix()  # Model returns (Q, dQ) tuple
            # Return a dense matrix for our dense log_likelihood implementation
            if hasattr(Q, 'toarray'):
                return Q.toarray()
            return Q

        # ConfigurableModel implementation (baseline, vectorize, polyalgorithm, cython)
        v, _ = self.value_function()
        p = self.choice_probabilities(v)

        # Build intensity matrix using explicit state encoding/decoding
        Q = np.zeros((self.K, self.K))
        lam = self.param['lambda']
        gam = self.param['gamma']

        for k in range(self.K):
            demand, players = self.int_to_state[k]

            # Demand transitions
            if demand < self.n_demand - 1:
                k_up = self.state_to_int[(demand + 1, players)]
                Q[k, k_up] = gam

            if demand > 0:
                k_down = self.state_to_int[(demand - 1, players)]
                Q[k, k_down] = gam

            # Player transitions
            for i in range(self.n_players):
                new_players = list(players)
                new_players[i] = 1 - new_players[i]
                k_switch = self.state_to_int[(demand, tuple(new_players))]
                if players[i] == 0:
                    # Entry: inactive -> active
                    rate = lam * p[i, k]
                else:
                    # Exit: active -> inactive
                    rate = lam * (1 - p[i, k])

                Q[k, k_switch] = rate

            # Diagonal element
            Q[k, k] = -np.sum(Q[k, :])

        return Q

    def preprocess_data(self, sample: List[int]) -> None:
        """
        Preprocess data to compute the transition count matrix.

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
                       grad: Optional[bool] = False
                       ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Compute the log-likelihood and its gradient for discrete time data.

        Parameters
        ----------
        theta : array-like
            Parameter vector (will be mapped to parameter dictionary)
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
        # Route according to Q & LL routing table
        if self._use_model_for_q_ll():
            # sparse and derivatives use Model's log_likelihood
            return self._impl.log_likelihood(theta, sample, Delta, grad=grad)

        # ConfigurableModel implementation (baseline, vectorize, polyalgorithm, cython)

        # Update parameters from theta vector
        self.update_parameters(theta)

        # Preprocess data
        self.preprocess_data(sample)

        # Compute intensity matrix
        Q = self.intensity_matrix()

        # Compute transition probability matrix using dense matrix exponential
        P = expm(Q * Delta)

        # Find states with > 0 incoming transitions
        _, nonzero_cols = self.D.nonzero()
        to_states = np.unique(nonzero_cols)

        # Compute the log likelihood
        ll = 0
        # Process only states that have transitions
        for l in to_states:
            # Get transition counts for this column
            counts_l = self.D[:, l]

            # Get the l-th column of transition probability matrix
            P_l = P[:, l]

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

        # Normalize by the number of observations
        ll /= self.nobs

        if self.verbose:
            print(f"ConfigurableModel LL = {ll}")

        return ll

    def estimate_parameters(self, sample: List[int], Delta: float = 1.0,
                            max_iter: Optional[int] = None,
                            start: Optional[np.ndarray] = None,
                            use_grad: bool = False,
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
            Maximum number of optimization iterations. If None, uses instance default.
        start : array-like, optional
            Initial parameter values. If None, uses 0.1 for all parameters.
        use_grad : bool, optional
            Whether to use analytical gradients in optimization

        Returns
        -------
        results : scipy.optimize.OptimizeResult
            Optimization results containing:
            - x : array, estimated parameters in order [θ_EC, θ_RN, θ_D, λ, γ]
            - fun : float, negative log-likelihood at optimum
            - success : bool, whether optimization converged
            - nit : int, number of iterations used
            - message : str, optimization termination message
        """
        # Route according to Q & LL routing table
        if self._use_model_for_q_ll():
            # sparse and derivatives use Model's estimate_parameters
            result = self._impl.estimate_parameters(
                sample, Delta, max_iter, start, use_grad=self.config.derivatives
            )
            # Sync parameters back to ConfigurableModel
            for i, key in enumerate(self.param_keys):
                self.param[key] = result.x[i]
            return result

        # ConfigurableModel implementation (baseline, vectorize, polyalgorithm, cython)
        self.preprocess_data(sample)

        # Set default values for optional arguments
        if max_iter is None:
            max_iter = DEFAULT_CONFIG['opt_max_iter']

        # Initial parameter values
        if start is not None:
            initial_params = start
        else:
            initial_params = np.array([-0.1, -0.1, 0.1, 0.1, 0.1])

        # Define objective function
        def objective(theta):
            # Compute negative log-likelihood
            result = self.log_likelihood(theta, sample, Delta, grad=False)
            return -result

        # Optimization options
        options = {
            'disp': None,
            'maxiter': max_iter,
            'gtol': DEFAULT_CONFIG['opt_tolerance'],
            'ftol': DEFAULT_CONFIG['opt_tolerance'],
        }

        # Optimize
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            jac=None,  # Use finite differences for gradients
            bounds=self.theta_bounds,
            options=options,
        )

        return result

    def __str__(self) -> str:
        """String representation of the model."""
        return (f"ConfigurableModel(n_players={self.n_players}, "
                f"n_demand={self.n_demand}, K={self.K}, "
                f"config={self.config})")

    def update_parameters(self, theta: np.ndarray) -> None:
        """
        Update model parameters from a parameter vector.

        Parameters
        ----------
        theta : array-like
            Parameter vector in the order specified by param_keys
        """
        # Check parameter size
        if len(theta) != len(self.param_keys):
            raise ValueError(f"Expected {len(self.param_keys)} parameters, got {len(theta)}")

        # Validate parameter values
        for i, (key, value) in enumerate(zip(self.param_keys, theta)):
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"Parameter {key} (index {i}) must be a finite number, got {value}")

            # Validate specific parameter constraints
            if key == 'lambda' and value <= 0:
                raise ValueError(f"lambda must be positive, got {value}")
            if key == 'gamma' and value <= 0:
                raise ValueError(f"gamma must be positive, got {value}")

        # Update parameters
        for key, value in zip(self.param_keys, theta):
            self.param[key] = value

        # If using Model implementation, update its parameters too
        if self._impl is not None:
            self._impl.update_parameters(theta)
