import numpy as np
from scipy.stats import poisson
from scipy.sparse import eye, csr_matrix
from scipy.sparse.linalg import splu


def vexpm(Q, Delta, v, epsilon=1e-12):
    """
    Compute the matrix exponential-vector product using uniformization.

    This function calculates exp(Q*Delta)*v efficiently for sparse matrices.

    Parameters
    ----------
    Q : scipy.sparse.csr_matrix
        Intensity matrix
    Delta : float
        Time interval
    v : numpy.ndarray
        Vector to multiply with exp(Q*Delta)
    epsilon : float, optional
        Tolerance for truncation of the uniformization series

    Returns
    -------
    mu : numpy.ndarray
        Result of exp(Q*Delta)*v
    """
    # Problem dimension
    n = Q.shape[0]

    # Check that Q is square
    if Q.shape[1] != n:
        raise ValueError(f"Q is not square: {Q.shape}")

    # Check that v is conformable to Q (rows of v have same size as columns of Q)
    if v.shape[0] != n:
        raise ValueError(f"v has {v.shape[0]} rows, but Q has {n} columns")

    # Prepare the rate matrix and the identity matrix
    unif_rate = max(abs(Q.diagonal())) + epsilon
    Sigma_tilde = Delta * Q + unif_rate * Delta * eye(n, format='csc')

    # Compute J_epsilon for the Poisson truncation
    J_epsilon = poisson.ppf(1 - epsilon, unif_rate * Delta)

    # Initialize variables
    nu = v.copy()
    mu = nu.copy()

    # Uniformization algorithm for matrix exponential with derivatives
    for j in range(1, int(J_epsilon) + 1):
        # Matrix exponential
        nu = csr_matrix.dot(Sigma_tilde, nu) / j
        mu += nu

    # Scale results by exp(-unif_rate * Delta)
    exp_scale = np.exp(-unif_rate * Delta)
    mu *= exp_scale
    return mu


def vexpm_deriv(Q, dQ, Delta, v, epsilon=1e-12):
    """
    Compute the matrix exponential-vector product and its derivatives.

    This implements the uniformization algorithm for computing exp(Q*Delta)*v
    and its derivatives with respect to parameters.

    Parameters
    ----------
    Q : scipy.sparse.csr_matrix
        Intensity matrix
    dQ : dict
        Dictionary of derivatives of Q with respect to parameters
    Delta : float
        Time interval
    v : numpy.ndarray
        Vector to multiply with exp(Q*Delta)
    epsilon : float, optional
        Tolerance for truncation of the uniformization series

    Returns
    -------
    mu : numpy.ndarray
        Result of exp(Q*Delta)*v
    dmu : dict
        Dictionary of derivatives of exp(Q*Delta)*v with respect to parameters
    """
    # Problem dimension
    n = Q.shape[0]
    params = dQ.keys()

    # Check that Q is square
    if Q.shape[1] != n:
        raise ValueError(f"Q is not square: {Q.shape}")

    # Check that v is conformable to Q (rows of v have same size as columns of Q)
    if v.shape[0] != n:
        raise ValueError(f"v has {v.shape[0]} rows, but Q has {n} columns")

    # Check that the elements of dQ have the same shape as Q
    for alpha in params:
        if dQ[alpha].shape != Q.shape:
            raise ValueError(f"dQ[{alpha}] has shape {dQ[alpha].shape}, but Q has shape {Q.shape}")

    # Prepare the rate matrix and the identity matrix
    unif_rate = max(abs(Q.diagonal())) + epsilon
    Sigma_tilde = Delta * Q + unif_rate * Delta * eye(n, format='csc')
    dSigma_tilde = {alpha: Delta * dQ_alpha for alpha, dQ_alpha in dQ.items()}

    # Compute J_epsilon for the Poisson truncation
    J_epsilon = poisson.ppf(1 - epsilon, unif_rate * Delta)

    # Initialize variables
    nu = v.copy()
    mu = nu.copy()
    delta = {alpha: v.copy() for alpha in params}
    dmu = {alpha: np.zeros(shape=v.shape) for alpha in params}

    # Uniformization algorithm for matrix exponential with derivatives
    for j in range(1, int(J_epsilon) + 1):
        # Derivatives
        for alpha in params:
            if j == 1:
                delta[alpha] = csr_matrix.dot(dSigma_tilde[alpha], v)
            else:
                delta[alpha] = (csr_matrix.dot(dSigma_tilde[alpha], nu) + csr_matrix.dot(Sigma_tilde, delta[alpha])) / j
            dmu[alpha] += delta[alpha]
        # Matrix exponential
        nu = csr_matrix.dot(Sigma_tilde, nu) / j
        mu += nu

    # Scale results by exp(-unif_rate * Delta)
    exp_scale = np.exp(-unif_rate * Delta)
    mu *= exp_scale
    for alpha in params:
        dmu[alpha] *= exp_scale

    return mu, dmu


def spsolve_multiple_rhs(A, rhs_all):
    """
    Solve system with multiple RHS vectors efficiently.

    Expects matrix A to be in CSC format.

    Parameters
    ----------
    A : scipy.sparse.csc_matrix
        System matrix
    rhs_all : dict
        Dictionary of RHS vectors

    Returns
    -------
    x : dict
        Dictionary of solutions for each input rhs
    """
    x = {}

    try:
        # Ensure CSC format for efficient splu factorization
        if A.format != 'csc':
            A = A.tocsc()

        # Pre-factor the system matrix (LU decomposition)
        lu_factor = splu(A)

        # Solve for each parameter using the pre-factored matrix
        for key in rhs_all:
            rhs = rhs_all[key].flatten()

            # Ensure RHS is the right type
            if not isinstance(rhs, np.ndarray):
                rhs = np.array(rhs, dtype=np.float64)

            # Solve using pre-factored matrix
            x[key] = lu_factor.solve(rhs)

    except Exception as e:
        print(f"Error: LU factorization failed: {e}")
        return {}

    return x
