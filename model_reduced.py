import itertools
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.linalg import eig
from scipy.optimize import minimize

from sparse import vexpm, vexpm_deriv
from model import DEFAULT_CONFIG


def get_csr_indices(csr_matrix):
    row_indices = []
    col_indices = csr_matrix.indices  # Directly use the indices array

    # Build the row_indices array
    for i in range(len(csr_matrix.indptr) - 1):
        # Number of elements in row i
        row_start = csr_matrix.indptr[i]
        row_end = csr_matrix.indptr[i + 1]
        # Append the current row index i, repeated for each non-zero element in the row
        row_indices.extend([i] * (row_end - row_start))

    return row_indices, col_indices


class ReducedFormModel:
    def __init__(self, n_players, n_demand, param, rho=None, config=None, verbose=True):
        self.n_players = n_players
        self.n_demand = n_demand
        self.D = None
        self.verbose = verbose
        self.setup_state_space()
        self.setup_encodings()
        # Number of player configurations
        self.n_configs = 2**n_players
        # Total number of model states (combinations of player and demand states)
        self.K = self.n_configs * self.n_demand
        # Precomputed transition addresses
        self.precompute_transitions()
        # Model parameters
        self.param = param
        self.param_keys = ['theta_ec', 'theta_rn', 'theta_d', 'lambda', 'gamma']
        # Parameter bounds
        self.theta_bounds = [(-10.0, 0.0),  # θ_EC
                             (-5.0, 0.0),   # θ_RN
                             (0.0, 10.0),   # θ_D
                             (0.01, 10.0),  # λ
                             (0.01, 5.0)]   # γ
        # Print basic model information
        if verbose:
            print(self)

    def __str__(self):
        str = f"""
n_players: {self.n_players}
n_demand: {self.n_demand}

K: {self.K}
Q dimension: {self.Q_tmpl.shape[0]} x {self.Q_tmpl.shape[1]} ({self.Q_tmpl.shape[0] * self.Q_tmpl.shape[1]} elements)
Q non-zeros: {self.Q_tmpl.nnz} ({self.Q_tmpl.nnz / (self.K*self.K) * 100:.2f}%)
"""
        return str

    def __repr__(self):
        return (f"ReducedFormModel(n_players={self.n_players}, n_demand={self.n_demand}, param={self.param})")

    def setup_state_space(self):
        """
        Generates the complete state space, where each state is structured as:
        - A demand state (ranging from 0 to n_demand-1)
        - A tuple of player states (each player's state being either 0 or 1)
        """
        # Iterate over demand states first to group by them
        demand_states = np.arange(self.n_demand)
        player_configs = np.array(list(itertools.product([0, 1], repeat=self.n_players)))
        self.state_space = list(itertools.product(demand_states, player_configs))

        # Separate arrays for demand states and player states
        self.demand_states = np.array([state[0] for state in self.state_space])
        self.player_states = np.array([state[1] for state in self.state_space])

        # Count number of active players (sum of player states)
        self.n_active = np.sum(self.player_states, axis=1)

    def setup_encodings(self):
        """
        Sets up mappings from states to integers and integers to states for
        quick encoding and look-up. States are grouped by demand states.
        """
        self.state_to_int = {}
        self.int_to_state = {}

        # Generate binary numbers from player states
        # Each row in player_states is a binary tuple;
        # convert them to integer using dot product with powers of 2.
        powers_of_two = 2 ** np.arange(self.n_players)
        binary_numbers = np.dot(self.player_states, powers_of_two[::-1])

        # Calculate state indices
        state_indices = self.demand_states * (2 ** self.n_players) + binary_numbers

        # Populate the dictionaries
        for index, state_index in enumerate(state_indices):
            demand_state = self.demand_states[index]
            player_state_tuple = tuple(self.player_states[index])
            self.state_to_int[(demand_state, player_state_tuple)] = state_index
            self.int_to_state[state_index] = (demand_state, player_state_tuple)

    def precompute_transitions(self):
        """
        Precomputes the structure of the intensity matrix.
        """
        n_states = len(self.state_space)
        row_indices = []
        col_indices = []
        values = []

        # Demand changes and player transitions
        for k in range(n_states):
            current_demand = self.demand_states[k]

            # Diagonal elements
            row_indices.append(k)
            col_indices.append(k)
            values.append(-1)  # -1 for diagonal

            # Demand increase
            if current_demand < self.n_demand - 1:
                kp = k + (2 ** self.n_players)  # Shift down one demand level
                row_indices.append(k)
                col_indices.append(kp)
                values.append(self.n_players + 1)  # nature

            # Demand decrease
            if current_demand > 0:
                kp = k - (2 ** self.n_players)  # Shift up one demand level
                row_indices.append(k)
                col_indices.append(kp)
                values.append(self.n_players + 1)  # nature

            # Player transitions
            for m in range(self.n_players):
                toggle_index = 1 << m  # Toggle m-th bit
                kp = k ^ toggle_index  # XOR to flip the m-th player's state
                row_indices.append(k)
                col_indices.append(kp)
                values.append(m+1)  # Player m

        # Create a CSR format matrix
        self.Q_tmpl = csr_matrix((np.array(values), (row_indices, col_indices)),
                                          shape=(n_states, n_states))

        # Reconstruct and store implied row and column indices in case they have changed
        row_indices, col_indices = get_csr_indices(self.Q_tmpl)
        self.row_indices = np.array(row_indices)
        self.col_indices = np.array(col_indices)

    def flow_utilities(self):
        return self.param['theta_ec'] \
            + self.param['theta_rn'] * self.n_active \
            + self.param['theta_d'] * self.demand_states

    def choice_probabilities(self):
        u = self.flow_utilities()
        p = 1 / (1 + np.exp(-u))
        return p

    def intensity_matrix(self):
        # Calculate choice probabilities
        p = self.choice_probabilities()

        # p * (1 - p) term for logistic derivatives
        pmp = p * (1 - p)

        # Initialize the data array with zeros matching the sparsity pattern
        Q_data = np.zeros_like(self.Q_tmpl.data, dtype=np.float64)

        # Initialize storage for derivative values
        dQ_data = {key: np.zeros_like(Q_data, dtype=np.float64) for key in self.param}

        # Update the data array with the correct transition rates
        for i, (from_idx, to_idx) in enumerate(zip(self.row_indices, self.col_indices)):

            # Demand vs player state transitions
            from_demand = self.demand_states[from_idx]
            to_demand = self.demand_states[to_idx]
            from_players = self.player_states[from_idx]
            to_players = self.player_states[to_idx]
            n_active = self.n_active[from_idx]
            n_entrant = self.n_players - n_active

            if from_demand != to_demand:

                # Demand transitions
                Q_data[i] = self.param['gamma']
                dQ_data['gamma'][i] = 1

            elif any(from_players != to_players):

                # Player state transition
                player_diff = from_players ^ to_players
                player_idx = np.where(player_diff)[0][0]
                if from_players[player_idx] == 0:
                    # Entry
                    Q_data[i] = self.param['lambda'] * p[from_idx]
                    dQ_data['theta_ec'][i] = (1-p[from_idx]) * Q_data[i]
                    dQ_data['theta_rn'][i] = (1-p[from_idx]) * n_active * Q_data[i]
                    dQ_data['theta_d'][i] = (1-p[from_idx]) * from_demand * Q_data[i]
                    dQ_data['lambda'][i] = p[from_idx]
                else:
                    # Exit
                    Q_data[i] = self.param['lambda'] * (1 - p[from_idx])
                    dQ_data['theta_ec'][i] = -p[from_idx] * Q_data[i]
                    dQ_data['theta_rn'][i] = -p[from_idx] * n_active * Q_data[i]
                    dQ_data['theta_d'][i] = -p[from_idx] * from_demand * Q_data[i]
                    dQ_data['lambda'][i] = 1 - p[from_idx]

            else:

                # Diagonal element
                # Contribution of n_entrant entry rates
                Q_data[i] -= n_entrant * self.param['lambda'] * p[from_idx]
                dQ_data['theta_ec'][i] = -n_entrant * self.param['lambda'] * pmp[from_idx]
                dQ_data['theta_rn'][i] = -n_entrant * self.param['lambda'] * n_active * pmp[from_idx]
                dQ_data['theta_d'][i] = -n_entrant * self.param['lambda'] * from_demand * pmp[from_idx]
                dQ_data['lambda'][i] = -n_entrant * p[from_idx]

                # Contribution of n_active exit rates
                Q_data[i] -= n_active * self.param['lambda'] * (1 - p[from_idx])
                dQ_data['theta_ec'][i] += n_active * self.param['lambda'] * pmp[from_idx]
                dQ_data['theta_rn'][i] += n_active * self.param['lambda'] * n_active * pmp[from_idx]
                dQ_data['theta_d'][i] += n_active * self.param['lambda'] * from_demand * pmp[from_idx]
                dQ_data['lambda'][i] -= n_active * (1 - p[from_idx])

                # Contribution of demand increase
                if from_demand < (self.n_demand - 1):
                    Q_data[i] -= self.param['gamma']
                    dQ_data['gamma'][i] -= 1

                # Contribution of demand decrease
                if from_demand > 0:
                    Q_data[i] -= self.param['gamma']
                    dQ_data['gamma'][i] -= 1

        # Use the template matrix to form Q
        Q = self.Q_tmpl.copy()
        Q.data = Q_data

        # Store sparse CSR matrices for each derivative
        dQ = {}
        for key, value in dQ_data.items():
            # Only update the data of the matrix, not the structure
            dQ[key] = Q.copy()
            dQ[key].data = value

        return Q, dQ

    def discrete_time_dgp(self, n_obs, Delta=1.0, seed=20180120):
        # Calculate the transition matrix using matrix exponential
        Q, _ = self.intensity_matrix()
        P, _ = vexpm_deriv(Q, {}, Delta, np.eye(self.K))

        # Invariant distribution (left eigenvector for eigenvalue 1)
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

    def update_parameters(self, theta):
        # Update model parameters from a list of parameter values
        for key, value in zip(self.param_keys, theta):
            self.param[key] = value

    def preprocess_data(self, sample):
        """
        Preprocess data to compute the transition count matrix.
        This function is called automatically from the log_likelihood function.
        """
        # Caching
        if hasattr(self, 'D') and self.D is not None:
            return
        # Transition count matrix D
        D = lil_matrix((self.K, self.K), dtype=int)
        for i in range(len(sample) - 1):
            from_state = sample[i]
            to_state = sample[i + 1]
            D[from_state, to_state] += 1
        self.D = D

    def log_likelihood(self, theta, sample, Delta=1.0, grad=True):
        # Update model parameters
        self.update_parameters(theta)

        # Number of states
        K = self.K

        # Preprocess data
        self.preprocess_data(sample)

        # Compute the log likelihood and its derivatives
        ll = 0
        if grad:
            dll = np.zeros(len(self.param_keys), dtype=np.float64)

        # Precompute and store the intensity matrix and its derivatives
        Q, dQ = self.intensity_matrix()

        for l in range(K):
            # Basis vector for state l
            e_l = np.zeros(K)
            e_l[l] = 1

            # Compute the l-th column of the transition probability matrix P
            # and its derivatives
            if grad:
                P_l, dP_l = vexpm_deriv(Q, dQ, Delta, e_l)
            else:
                P_l = vexpm(Q, Delta, e_l)

            # Logarithm of the l-th column
            log_P_l = np.log(P_l)

            # Get the l-th column of D
            D_col_l = self.D.getcol(l).toarray().flatten()

            # Contribution to the log likelihood from the l-th column
            ll += float(np.dot(D_col_l, log_P_l))

            # Compute and accumulate derivatives
            if grad:
                # For each parameter
                for i, param_key in enumerate(self.param_keys):
                    # The derivative of log P with respect to parameter 'param'
                    dlogP_l_param = dP_l[param_key] / P_l  # Element-wise
                    # Contribution to the derivative of the log likelihood with
                    # respect to 'param'
                    dll[i] += float(np.dot(D_col_l, dlogP_l_param))

        # Normalize the log likelihood and gradient
        ll = ll / len(sample)
        if grad:
            dll /= len(sample)
            return ll, dll

        return ll

    def estimate_parameters(self, sample, Delta=1.0,
                            max_iter=None, start=None, use_grad=True):
        # Preprocess data
        self.preprocess_data(sample)

        # Get optimization max_iter from parameter or instance config
        opt_max_iter = max_iter if max_iter is not None else DEFAULT_CONFIG['opt_max_iter']

        # Initial parameter values
        if start is not None:
            initial_params = start
        else:
            initial_params = np.array([-0.1, -0.1, 0.1, 0.1, 0.1])

        # Define objective function for optimization
        def objective(theta, Delta, sample, grad):
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
            args=(Delta, sample, use_grad),
            method='L-BFGS-B',
            jac=use_grad,
            bounds=self.theta_bounds,
            options=options,
        )

        return results
