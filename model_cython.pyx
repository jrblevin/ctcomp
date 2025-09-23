# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Complete Cython implementation of Bellman operator with derivatives.

This implementation must match the Python version in model.py.

Compile with:
    python setup_cython.py build_ext --inplace
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, log, fmax

# Type definitions
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t INT32_t

# Python object type for dictionaries
from cpython.object cimport PyObject

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double logsumexp(double v0, double v1) nogil:
    """Compute log(exp(v0) + exp(v1)) in a numerically stable way."""
    cdef double max_v = fmax(v0, v1)
    return max_v + log(exp(v0 - max_v) + exp(v1 - max_v))

@cython.boundscheck(False)
@cython.wraparound(False)
def choice_probabilities_cython(
    np.ndarray[DTYPE_t, ndim=2] v,
    dv_dict,  # Python dict or None
    double theta_ec,
    int n_players,
    int K,
    np.ndarray[np.uint8_t, ndim=2, cast=True] is_inactive,
    np.ndarray[INT32_t, ndim=2] switch_indices
):
    """
    Compute choice probabilities and optionally derivatives using Cython.

    This implementation must match choice_probabilities in model.py.

    Parameters
    ----------
    v : numpy.ndarray
        Value function, shape (n_players, K)
    dv_dict : dict or None
        Dictionary of value function derivatives, or None if not needed
    theta_ec : float
        Entry cost parameter
    n_players : int
        Number of players
    K : int
        Number of states
    is_inactive : numpy.ndarray
        Boolean array indicating inactive players
    switch_indices : numpy.ndarray
        Precomputed switch indices

    Returns
    -------
    p : numpy.ndarray
        Choice probabilities, shape (n_players, K)
    dp_dict : dict or None
        Dictionary of derivatives (only if dv_dict provided)
    psi_array : numpy.ndarray
        Entry costs, shape (n_players, K)
    """
    cdef int i, k, m
    cdef double v0, v1, psi, max_utility, exp_v0, exp_v1, denom
    cdef double pmp, dv0_val, dv1_val, dv1_vec_val

    # Pre-allocate arrays
    cdef np.ndarray[DTYPE_t, ndim=2] p = np.zeros((n_players, K), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] psi_array = np.zeros((n_players, K), dtype=np.float64)

    # Instantaneous payoffs ψ (entry costs)
    psi_array = is_inactive.astype(np.float64) * theta_ec

    # Initialize derivatives dictionary if needed
    dp_dict = None
    cdef dict dp_dict_typed = None
    if dv_dict is not None:
        dp_dict = {}
        dp_dict_typed = dp_dict
        for key in dv_dict.keys():
            dp_dict[key] = np.zeros((n_players, K), dtype=np.float64)

    # Main computation loop
    for m in range(n_players):
        # Get switch indices for player m
        for k in range(K):
            # Calculate probabilities via logit formula
            v0 = v[m, k]
            v1 = v[m, switch_indices[m, k]] + psi_array[m, k]

            # Use numerically stable computation
            max_utility = fmax(v0, v1)
            exp_v0 = exp(v0 - max_utility)
            exp_v1 = exp(v1 - max_utility)
            denom = exp_v0 + exp_v1

            # Switch probability
            p[m, k] = exp_v1 / denom

            # Calculate derivatives if requested
            if dv_dict is not None:
                # For the logistic function p = exp(u1)/(exp(u0) + exp(u1)),
                # the derivative is p*(1-p)*(dv1/dθ - dv0/dθ)
                pmp = p[m, k] * (1.0 - p[m, k])

                for key in dv_dict.keys():
                    # Chain rule: ∂p/∂θ = ∂p/∂v_switch × ∂v_switch/∂θ
                    #                   - ∂p/∂v_continue × ∂v_continue/∂θ
                    dv0_val = dv_dict[key][m, k]
                    dv1_val = dv_dict[key][m, switch_indices[m, k]]

                    # Add direct partial derivative of entry cost ψ with respect to θ_EC
                    if key == 'theta_ec':
                        # ∂ψ/∂θ_EC = 1 for inactive players, 0 for active players
                        if is_inactive[m, k]:
                            dv1_vec_val = dv1_val + 1.0
                        else:
                            dv1_vec_val = dv1_val
                        dp_dict_typed[key][m, k] = pmp * (dv1_vec_val - dv0_val)
                    else:
                        dp_dict_typed[key][m, k] = pmp * (dv1_val - dv0_val)

    return p, dp_dict, psi_array

@cython.boundscheck(False)
@cython.wraparound(False)
def _bellman_components_cython(
    np.ndarray[DTYPE_t, ndim=2] v,
    double theta_ec,
    double theta_rn,
    double theta_d,
    double gamma,
    double lam,
    double rho,
    int n_players,
    int K,
    np.ndarray[np.uint8_t, ndim=2, cast=True] is_active,
    np.ndarray[np.uint8_t, ndim=2, cast=True] is_inactive,
    np.ndarray[INT32_t, ndim=1] n_active,
    np.ndarray[INT32_t, ndim=1] demand_states,
    np.ndarray[INT32_t, ndim=1] k_demand_up,
    np.ndarray[INT32_t, ndim=1] k_demand_down,
    np.ndarray[INT32_t, ndim=2] switch_indices
):
    """
    Core Cython computation of Bellman numerator and rate sum.

    Must match _bellman_components in model.py.

    Returns:
        numerator: Numerator values
        ratesum: Rate sum values, shape (K,)
        p: Choice probabilities
        psi_array: Entry costs
    """
    cdef int i, k, m, k_switch
    cdef double v0, v1, psi

    # Pre-allocate arrays
    cdef np.ndarray[DTYPE_t, ndim=2] numerator = np.zeros((n_players, K), dtype=np.float64)

    # 1. Compute ratesum for denominator
    # ratesum[k] = sum of all player rates + sum of nature rates
    cdef np.ndarray[DTYPE_t, ndim=1] ratesum = np.full(K, n_players * lam, dtype=np.float64)
    for k in range(K):
        if k_demand_up[k] >= 0:
            ratesum[k] += gamma
        if k_demand_down[k] >= 0:
            ratesum[k] += gamma

    # 2. and 3. Get choice probabilities and ψ (entry costs)
    p, _, psi_array = choice_probabilities_cython(v, None, theta_ec, n_players, K, is_inactive, switch_indices)

    # 4. Vectorized flow payoffs for all players
    numerator = np.multiply(
        is_active.astype(np.float64),
        (theta_rn * n_active[np.newaxis, :] + theta_d * demand_states[np.newaxis, :])
    )

    # 5. Main Bellman update loop over players
    for i in range(n_players):
        # 6. Demand transitions
        for k in range(K):
            # Up transitions
            if k_demand_up[k] >= 0:
                numerator[i, k] += gamma * v[i, k_demand_up[k]]

            # Down transitions
            if k_demand_down[k] >= 0:
                numerator[i, k] += gamma * v[i, k_demand_down[k]]

        # 7. Player transitions
        for m in range(n_players):
            if m != i:
                # 8. Rival player switching actions
                for k in range(K):
                    k_switch = switch_indices[m, k]
                    numerator[i, k] += lam * (p[m, k] * v[i, k_switch] + (1.0 - p[m, k]) * v[i, k])
            else:
                # 9. Own actions
                for k in range(K):
                    k_switch = switch_indices[i, k]
                    v0 = v[i, k]
                    v1 = v[i, k_switch] + psi_array[i, k]

                    # Calculate the log sum of exponentials
                    numerator[i, k] += lam * logsumexp(v0, v1)

    return numerator, ratesum, p, psi_array

@cython.boundscheck(False)
@cython.wraparound(False)
def bellman_operator_cython(
    np.ndarray[DTYPE_t, ndim=2] v,
    double theta_ec,
    double theta_rn,
    double theta_d,
    double gamma,
    double lam,
    double rho,
    int n_players,
    int K,
    np.ndarray[np.uint8_t, ndim=2, cast=True] is_active,
    np.ndarray[np.uint8_t, ndim=2, cast=True] is_inactive,
    np.ndarray[INT32_t, ndim=1] n_active,
    np.ndarray[INT32_t, ndim=1] demand_states,
    np.ndarray[INT32_t, ndim=1] k_demand_up,
    np.ndarray[INT32_t, ndim=1] k_demand_down,
    np.ndarray[INT32_t, ndim=2] switch_indices
):
    """
    Cython Bellman operator matching Model.bellman_operator.

    Returns:
        v_new: Updated value function
        p: Choice probabilities
    """
    # Use the core computation function
    numerator, ratesum, p, psi_array = _bellman_components_cython(
        v, theta_ec, theta_rn, theta_d, gamma, lam, rho, n_players, K,
        is_active, is_inactive, n_active, demand_states,
        k_demand_up, k_demand_down, switch_indices
    )

    # Normalization
    cdef np.ndarray[DTYPE_t, ndim=2] v_new = numerator / (rho + ratesum)

    return v_new, p


@cython.boundscheck(False)
@cython.wraparound(False)
def dbellman_operator_dtheta_cython(
    np.ndarray[DTYPE_t, ndim=2] v,
    double theta_ec,
    double theta_rn,
    double theta_d,
    double gamma,
    double lam,
    double rho,
    int n_players,
    int K,
    np.ndarray[np.uint8_t, ndim=2, cast=True] is_active,
    np.ndarray[np.uint8_t, ndim=2, cast=True] is_inactive,
    np.ndarray[INT32_t, ndim=1] n_active,
    np.ndarray[INT32_t, ndim=1] demand_states,
    np.ndarray[INT32_t, ndim=1] k_demand_up,
    np.ndarray[INT32_t, ndim=1] k_demand_down,
    np.ndarray[INT32_t, ndim=2] switch_indices
):
    """
    Optimized version that returns arrays instead of dictionary.

    Returns arrays in a fixed order to avoid dictionary overhead.
    Dictionary creation is done on the Python side.
    """
    cdef int i, k, m
    cdef double p_m, dp_m_dtheta_ec, v0, v1, max_v, logsumexp_val
    cdef double denom, quotient_denom

    # Pre-allocate all result arrays - no dictionary operations in main loops
    cdef np.ndarray[DTYPE_t, ndim=2] dT_theta_ec = np.zeros((n_players, K), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] dT_theta_rn = np.zeros((n_players, K), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] dT_theta_d = np.zeros((n_players, K), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] dT_lambda = np.zeros((n_players, K), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] dT_gamma = np.zeros((n_players, K), dtype=np.float64)

    # Pre-allocate working arrays
    cdef np.ndarray[DTYPE_t, ndim=2] d_numerator_gam = np.zeros((n_players, K), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] d_ratesum_gam = np.zeros(K, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] d_numerator_lam = np.zeros((n_players, K), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] d_numerator_ec = np.zeros((n_players, K), dtype=np.float64)

    # Use the core computation function
    numerator, ratesum, p, psi_array = _bellman_components_cython(
        v, theta_ec, theta_rn, theta_d, gamma, lam, rho, n_players, K,
        is_active, is_inactive, n_active, demand_states,
        k_demand_up, k_demand_down, switch_indices
    )

    # ∂T/∂θ_RN and ∂T/∂θ_D: Only affect numerator (flow payoffs)
    for i in range(n_players):
        for k in range(K):
            denom = rho + ratesum[k]
            if is_active[i, k]:
                dT_theta_rn[i, k] = n_active[k] / denom
                dT_theta_d[i, k] = demand_states[k] / denom

    # ∂T/∂γ: Affects BOTH numerator and denominator (ratesum), need quotient rule
    for k in range(K):
        # Demand up
        if k_demand_up[k] >= 0:
            for i in range(n_players):
                d_numerator_gam[i, k] = v[i, k_demand_up[k]]
            d_ratesum_gam[k] = 1.0

        # Demand down
        if k_demand_down[k] >= 0:
            for i in range(n_players):
                d_numerator_gam[i, k] += v[i, k_demand_down[k]]
            d_ratesum_gam[k] += 1.0

    # ∂T/∂λ: Affects both numerator and denominator (ratesum), need quotient rule
    for m in range(n_players):
        for k in range(K):
            p_m = p[m, k]

            # Expected continuation value for rivals and own player
            for i in range(n_players):
                if i != m:
                    # Rival m: expected value is p_m * v_switch + (1-p_m) * v_stay
                    d_numerator_lam[i, k] += p_m * v[i, switch_indices[m, k]] + (1.0 - p_m) * v[i, k]
                else:
                    # Own player gets logsumexp
                    v0 = v[m, k]
                    v1 = v[m, switch_indices[m, k]] + psi_array[m, k]
                    max_v = fmax(v0, v1)
                    logsumexp_val = max_v + log(exp(v0 - max_v) + exp(v1 - max_v))
                    d_numerator_lam[m, k] += logsumexp_val

    # ∂T/∂θ_EC: Only affects numerator (via choice probabilities)
    for m in range(n_players):
        for k in range(K):
            p_m = p[m, k]
            if is_inactive[m, k]:
                dp_m_dtheta_ec = p_m * (1.0 - p_m)
            else:
                dp_m_dtheta_ec = 0.0

            # Effect on other players
            for i in range(n_players):
                if i != m:
                    # Rival m: p_m * v_switch + (1-p_m) * v_stay
                    d_numerator_ec[i, k] += lam * dp_m_dtheta_ec * (v[i, switch_indices[m, k]] - v[i, k])
                else:
                    # Own player effect: direct entry cost effect
                    if is_inactive[m, k]:
                        d_numerator_ec[m, k] += lam * p_m

    # Apply derivatives
    for i in range(n_players):
        for k in range(K):
            denom = rho + ratesum[k]
            denom_sq = denom * denom

            # θ_EC: simple division (no quotient rule)
            dT_theta_ec[i, k] = d_numerator_ec[i, k] / denom

            # γ: quotient rule
            dT_gamma[i, k] = (denom * d_numerator_gam[i, k] - numerator[i, k] * d_ratesum_gam[k]) / denom_sq

            # λ: quotient rule
            dT_lambda[i, k] = (denom * d_numerator_lam[i, k] - numerator[i, k] * n_players) / denom_sq

    # Return arrays in fixed order - dictionary creation in Python
    return dT_theta_ec, dT_theta_rn, dT_theta_d, dT_lambda, dT_gamma


@cython.boundscheck(False)
@cython.wraparound(False)
def dbellman_operator_dv_cython(
    np.ndarray[DTYPE_t, ndim=2] v,
    double theta_ec,
    double theta_rn,
    double theta_d,
    double lam,
    double gam,
    double rho,
    int n_players,
    int K,
    np.ndarray[np.uint8_t, ndim=2, cast=True] is_active,
    np.ndarray[np.uint8_t, ndim=2, cast=True] is_inactive,
    np.ndarray[INT32_t, ndim=1] n_active,
    np.ndarray[INT32_t, ndim=1] demand_states,
    np.ndarray[INT32_t, ndim=1] k_demand_up,
    np.ndarray[INT32_t, ndim=1] k_demand_down,
    np.ndarray[INT32_t, ndim=2] switch_indices,
    np.ndarray[np.uint8_t, ndim=1, cast=True] demand_up_valid,
    np.ndarray[np.uint8_t, ndim=1, cast=True] demand_down_valid
):
    """
    Compute the analytical Jacobian ∂T/∂v of the Bellman operator.

    Returns triplet arrays (rows, cols, data) for sparse matrix construction.

    Parameters
    ----------
    v : numpy.ndarray
        Value function, shape (n_players, K)
    theta_ec : float
        Entry cost parameter
    lam : float
        Lambda parameter
    gam : float
        Gamma parameter
    rho : float
        Discount rate
    n_players : int
        Number of players
    K : int
        Number of states
    is_active : numpy.ndarray
        Active player indicators
    is_inactive : numpy.ndarray
        Inactive player indicators
    n_active : numpy.ndarray
        Number of active players per state
    demand_states : numpy.ndarray
        Demand state indices
    k_demand_up : numpy.ndarray
        Demand up transition indices
    k_demand_down : numpy.ndarray
        Demand down transition indices
    switch_indices : numpy.ndarray
        Player switch indices
    demand_up_valid : numpy.ndarray
        Valid demand up transitions
    demand_down_valid : numpy.ndarray
        Valid demand down transitions

    Returns
    -------
    rows : numpy.ndarray
        Row indices for sparse matrix
    cols : numpy.ndarray
        Column indices for sparse matrix
    data : numpy.ndarray
        Data values for sparse matrix
    nnz : int
        Number of non-zero elements
    """
    cdef int i, k, m, current_idx, neighbor_idx, switch_idx, k_switch
    cdef int n_states = K * n_players
    cdef int row_offset
    cdef double val

    # Conservative estimate for array sizes
    cdef int estimated_nnz = 2 * n_players * K * (2 + n_players + n_players * n_players)

    # Pre-allocate triplet arrays
    cdef np.ndarray[INT32_t, ndim=1] rows = np.empty(estimated_nnz, dtype=np.int32)
    cdef np.ndarray[INT32_t, ndim=1] cols = np.empty(estimated_nnz, dtype=np.int32)
    cdef np.ndarray[DTYPE_t, ndim=1] data = np.empty(estimated_nnz, dtype=np.float64)

    cdef int nnz_idx = 0  # Current position in arrays

    # Vectorization arrays (declared here, used in loop)
    cdef np.ndarray[DTYPE_t, ndim=1] denom_all
    cdef np.ndarray[DTYPE_t, ndim=1] vals_up
    cdef np.ndarray[DTYPE_t, ndim=1] vals_down
    cdef np.ndarray[DTYPE_t, ndim=1] inv_denom
    cdef np.ndarray[DTYPE_t, ndim=1] diag_contrib
    cdef np.ndarray[DTYPE_t, ndim=1] own_switch_vals

    # Get core components - call the Cython version directly
    numerator_all, ratesum_all, p, psi_array = _bellman_components_cython(
        v, theta_ec, theta_rn, theta_d, gam, lam, rho, n_players, K,
        is_active, is_inactive, n_active, demand_states,
        k_demand_up, k_demand_down, switch_indices
    )

    # Main loop for within-player effects
    denom_all = rho + ratesum_all  # (K,)

    for i in range(n_players):
        row_offset = i * K

        # Demand up transitions
        vals_up = gam / denom_all  # (K,)
        for k in range(K):
            if demand_up_valid[k]:
                rows[nnz_idx] = row_offset + k
                cols[nnz_idx] = row_offset + k_demand_up[k]
                data[nnz_idx] = vals_up[k]
                nnz_idx += 1

        # Demand down transitions
        vals_down = gam / denom_all  # (K,)
        for k in range(K):
            if demand_down_valid[k]:
                rows[nnz_idx] = row_offset + k
                cols[nnz_idx] = row_offset + k_demand_down[k]
                data[nnz_idx] = vals_down[k]
                nnz_idx += 1

        # Pre-compute all denominators
        inv_denom = 1.0 / denom_all  # (K,)

        # Compute diagonal contributions from all players vectorized
        # Sum over all other players' contributions: sum_m [lam * (1 - p[m,:])] / denom
        diag_contrib = np.zeros(K, dtype=np.float64)
        for m in range(n_players):
            if m != i:
                diag_contrib += lam * (1.0 - p[m, :]) * inv_denom

        # Add own player i's contribution to diagonal
        diag_contrib += lam * (1.0 - p[i, :]) * inv_denom

        # Compute off-diagonal values for own switch
        own_switch_vals = lam * p[i, :] * inv_denom  # (K,)

        # Fill sparse arrays with within-player effects
        for k in range(K):
            current_idx = row_offset + k

            # Other players' switch effects (off-diagonal within same player block)
            for m in range(n_players):
                if m != i:
                    k_switch = switch_indices[m, k]
                    switch_idx = row_offset + k_switch
                    val = lam * p[m, k] * inv_denom[k]

                    rows[nnz_idx] = current_idx
                    cols[nnz_idx] = switch_idx
                    data[nnz_idx] = val
                    nnz_idx += 1

            # Diagonal entry
            rows[nnz_idx] = current_idx
            cols[nnz_idx] = current_idx
            data[nnz_idx] = diag_contrib[k]
            nnz_idx += 1

            # Own switch state entry
            k_switch = switch_indices[i, k]
            switch_idx = row_offset + k_switch
            rows[nnz_idx] = current_idx
            cols[nnz_idx] = switch_idx
            data[nnz_idx] = own_switch_vals[k]
            nnz_idx += 1

    # Cross-player effects
    for i in range(n_players):
        for m in range(n_players):
            if m != i:
                nnz_idx = _dbellman_operator_dv_cross_player_effects_cython(
                    rows, cols, data, nnz_idx, v, i, m, lam, rho, p, psi_array,
                    ratesum_all, K, switch_indices
                )

    # Return only the filled portion of the arrays
    return rows[:nnz_idx], cols[:nnz_idx], data[:nnz_idx], nnz_idx


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _dbellman_operator_dv_cross_player_effects_cython(
    np.ndarray[INT32_t, ndim=1] rows,
    np.ndarray[INT32_t, ndim=1] cols,
    np.ndarray[DTYPE_t, ndim=1] data,
    int nnz_idx,
    np.ndarray[DTYPE_t, ndim=2] v,
    int i,
    int m,
    double lam,
    double rho,
    np.ndarray[DTYPE_t, ndim=2] p,
    np.ndarray[DTYPE_t, ndim=2] psi,
    np.ndarray[DTYPE_t, ndim=1] ratesum,
    int K,
    np.ndarray[INT32_t, ndim=2] switch_indices
):
    """
    Compute cross-player derivative terms ∂T_i/∂v_m.

    This is the Cython version of _dbellman_operator_dv_cross_player_effects.
    """
    cdef int k, col_k
    cdef int i_offset = i * K
    cdef int m_offset = m * K
    cdef int current_idx
    cdef double val

    # Vectorized computation of all derivatives using NumPy operations
    # Extract player m's choice probabilities (K,)
    cdef np.ndarray[DTYPE_t, ndim=1] p_m = p[m, :]

    # Derivative of logit choice probability: p * (1 - p)
    cdef np.ndarray[DTYPE_t, ndim=1] dp_base = p_m * (1.0 - p_m)
    cdef np.ndarray[DTYPE_t, ndim=1] dp_dv0 = -dp_base  # ∂p_m/∂v_m(k)
    cdef np.ndarray[DTYPE_t, ndim=1] dp_dv1 = dp_base   # ∂p_m/∂v_m(k_switch)

    # Player i's value differences (K,)
    cdef np.ndarray[INT32_t, ndim=1] k_switch_m = switch_indices[m, :]
    cdef np.ndarray[DTYPE_t, ndim=1] v_i_stay = v[i, :]
    cdef np.ndarray[DTYPE_t, ndim=1] v_i_switch = v[i, k_switch_m]
    cdef np.ndarray[DTYPE_t, ndim=1] v_i_diff = v_i_switch - v_i_stay

    # Compute denominators (K,)
    cdef np.ndarray[DTYPE_t, ndim=1] denom = rho + ratesum

    # Compute all derivative values vectorized
    cdef np.ndarray[DTYPE_t, ndim=1] vals_dv0 = lam * dp_dv0 * v_i_diff / denom
    cdef np.ndarray[DTYPE_t, ndim=1] vals_dv1 = lam * dp_dv1 * v_i_diff / denom

    # Sparse array filling
    for k in range(K):
        current_idx = i_offset + k

        # Add term 1: ∂p_m/∂v_m(k)
        val = vals_dv0[k]
        if val != 0.0:
            rows[nnz_idx] = current_idx
            cols[nnz_idx] = m_offset + k
            data[nnz_idx] = val
            nnz_idx += 1

        # Add term 2: ∂p_m/∂v_m(k_switch)
        val = vals_dv1[k]
        if val != 0.0:
            col_k = k_switch_m[k]
            rows[nnz_idx] = current_idx
            cols[nnz_idx] = m_offset + col_k
            data[nnz_idx] = val
            nnz_idx += 1

    return nnz_idx
