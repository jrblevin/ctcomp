#!/usr/bin/env python3
"""
Unit tests for analytical derivatives.

This test suite verifies the correctness of analytical derivatives by
comparing them to numerical approximations using finite differences.
"""

import pytest
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
from model import Model
from sparse import vexpm, vexpm_deriv


@pytest.fixture
def default_params():
    """Default test parameters."""
    return {
        'theta_ec': -1.5,
        'theta_rn': -0.5,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 0.5,
    }


@pytest.fixture
def derivative_settings():
    """Numerical derivative settings."""
    return {
        'h': 1e-6,      # Step size for finite differences
        'rtol': 1e-6,   # Relative tolerance
        'atol': 1e-9,   # Absolute tolerance
    }


def create_model(n_players=2, n_demand=2, params=None, rho=0.1):
    """Create a model instance."""
    if params is None:
        params = {
            'theta_ec': -1.5,
            'theta_rn': -0.5,
            'theta_d': 1.0,
            'lambda': 1.0,
            'gamma': 0.5,
        }
    return Model(
        n_players=n_players,
        n_demand=n_demand,
        param=params,
        rho=rho,
        verbose=False
    )


def numerical_derivative(func, param_name, params, h=1e-6):
    """
    Compute numerical derivative using central differences.

    Parameters:
    -----------
    func : callable
        Function that takes a model instance and returns the quantity of interest
    param_name : str
        Name of parameter to differentiate with respect to
    params : dict
        Base parameter values
    h : float
        Step size
    """
    # Get original parameter value
    original_value = params[param_name]

    # Forward step
    params_plus = params.copy()
    params_plus[param_name] = original_value + h
    model_plus = create_model(params=params_plus)
    f_plus = func(model_plus)

    # Backward step
    params_minus = params.copy()
    params_minus[param_name] = original_value - h
    model_minus = create_model(params=params_minus)
    f_minus = func(model_minus)

    # Central difference
    return (f_plus - f_minus) / (2 * h)


#
# model.value_function
#

def test_value_function_derivatives(default_params, derivative_settings):
    """Test derivatives of value function."""
    model = create_model(params=default_params)
    v, dv = model.value_function()

    # Test each parameter
    for param_name in model.param_keys:
        # Analytical derivative
        analytical = dv[param_name]

        # Numerical derivative
        def value_func(m):
            v_num, _ = m.value_function()
            return v_num

        numerical = numerical_derivative(value_func, param_name, default_params,
                                         h=derivative_settings['h'])

        # Compare
        np.testing.assert_allclose(
            analytical, numerical,
            rtol=derivative_settings['rtol'],
            atol=derivative_settings['atol'],
            err_msg=f"Value function derivative mismatch for {param_name}"
        )


@pytest.mark.parametrize("n_players,n_demand", [
    (2, 2),  # Small
    (6, 2),  # Medium
    (2, 3),  # Different demand states
])
def test_value_function_derivatives_shape(n_players, n_demand, default_params):
    """Test analytical derivatives across different model sizes."""
    model = create_model(n_players=n_players, n_demand=n_demand, params=default_params)

    # Check that derivatives have correct shapes
    v, dv = model.value_function()

    assert v.shape == (model.n_players, model.K), \
        f"Value function shape mismatch for ({n_players}, {n_demand})"

    for param_name in model.param_keys:
        assert dv[param_name].shape == v.shape, \
            f"Derivative shape mismatch for {param_name} in ({n_players}, {n_demand})"


def test_value_function_derivative_step_sizes(default_params):
    """Test numerical derivatives across several step sizes."""
    model = create_model(params=default_params)
    v, dv = model.value_function()

    # Test different step sizes
    h_values = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    param_name = 'theta_ec'  # Test one parameter

    errors = []
    for h in h_values:
        # Numerical derivative with this step size
        def value_func(m):
            v_num, _ = m.value_function()
            return v_num

        numerical = numerical_derivative(value_func, param_name, default_params, h=h)

        # Calculate relative error
        analytical = dv[param_name]
        rel_error = np.max(np.abs(analytical - numerical) / (np.abs(analytical) + 1e-16))
        errors.append(rel_error)

    # Find the best step size and verify accuracy
    best_error = np.min(errors)
    best_idx = np.argmin(errors)
    h_optimal = h_values[best_idx]

    # Analytical derivatives should be accurate for at least one step size
    error_info = [(h, err) for h, err in zip(h_values, errors)]
    assert best_error < 1e-8, (
        f"Best relative error {best_error:.2e} too large. "
        f"Optimal step size: {h_optimal:.1e}. "
        f"All errors: {error_info}"
    )


def test_value_function_derivative_near_bound(default_params):
    """Test derivatives near parameter bounds."""
    # Test near entry cost bound (theta_ec close to 0)
    params_near_bound = default_params.copy()
    params_near_bound['theta_ec'] = -0.00001

    model = create_model(params=params_near_bound)

    # Should still be able to compute derivatives
    v, dv = model.value_function()

    # Check that derivatives exist and are finite
    for param_name in model.param_keys:
        assert np.all(np.isfinite(dv[param_name])), \
            f"Non-finite derivatives for {param_name} near bound"


#
# model.choice_probabilities
#

def test_choice_probability_derivatives(default_params, derivative_settings):
    """Test derivatives of choice probabilities."""
    model = create_model(params=default_params)
    v, dv = model.value_function()
    p, dp = model.choice_probabilities(v, dv)

    # Test each parameter
    for param_name in model.param_keys:
        # Analytical derivative
        analytical = dp[param_name]

        # Numerical derivative
        def choice_prob_func(m):
            v_num, _ = m.value_function()
            p_num = m.choice_probabilities(v_num)
            return p_num

        numerical = numerical_derivative(choice_prob_func, param_name, default_params,
                                         h=derivative_settings['h'])

        # Compare
        np.testing.assert_allclose(
            analytical, numerical,
            rtol=derivative_settings['rtol'],
            atol=derivative_settings['atol'],
            err_msg=f"Choice probability derivative mismatch for {param_name}"
        )


#
# model.dbellman_operator_dtheta
#

def test_dbellman_operator_dtheta(default_params, derivative_settings):
    """Test partial derivatives of Bellman operator with respect to θ."""
    model = create_model(params=default_params)

    # Get a test value function (not converged)
    np.random.seed(42)
    v_test = np.random.randn(model.n_players, model.K)

    # Get analytical partials
    dT_analytical = model.dbellman_operator_dtheta(v_test)

    # Test each parameter
    for param_name in model.param_keys:
        # Analytical partial
        analytical = dT_analytical[param_name]

        # Numerical partial
        def bellman_func(m):
            Tv = m.bellman_operator(v_test)
            return Tv

        numerical = numerical_derivative(bellman_func, param_name,
                                         default_params,
                                         h=derivative_settings['h'])

        # Compare
        np.testing.assert_allclose(
            analytical, numerical,
            rtol=derivative_settings['rtol'],
            atol=derivative_settings['atol'],
            err_msg=f"Bellman operator partial derivative mismatch for {param_name}"
        )


#
# model.dbellman_operator_dv
#

def test_dbellman_operator_dv(default_params, derivative_settings):
    """Test partial derivatives of Bellman operator with respect to v."""
    model = create_model(params=default_params)

    # Use a random value function
    np.random.seed(42)
    v_test = np.random.randn(model.n_players, model.K)

    # Get analytical derivative
    dT_dv_analytical = model.dbellman_operator_dv(v_test)

    # Numerical derivative
    def bellman_func(v):
        return model.bellman_operator(v)

    # Step size and tolerances
    h = derivative_settings['h']
    rtol = derivative_settings['rtol']
    atol = derivative_settings['atol']

    for i in range(model.n_players):
        for k in range(model.K):
            v_plus = v_test.copy()
            v_plus[i, k] += h
            v_minus = v_test.copy()
            v_minus[i, k] -= h
            numerical = (bellman_func(v_plus) - bellman_func(v_minus)) / (2 * h)

            # Convert numerical result to flat index for comparison
            # Each column in the sparse matrix corresponds to input variable ∂T/∂v[i,k]
            input_idx = i * model.K + k  # Column index for ∂T/∂v[i,k]

            # Extract the analytical derivative column for this input variable
            analytical_column = dT_dv_analytical[:, input_idx].toarray().flatten()

            # Compare with numerical derivative (which is also flattened)
            numerical_flat = numerical.flatten()

            np.testing.assert_allclose(
                analytical_column, numerical_flat,
                rtol=rtol,
                atol=atol,
                err_msg=f"Bellman operator value derivative mismatch at i={i}, k={k}"
            )


#
# model.log_likelihood
#

def test_log_likelihood_gradient(default_params, derivative_settings):
    """Test gradient of log-likelihood function."""
    model = create_model(params=default_params)

    # Generate test data
    np.random.seed(42)
    test_data = model.discrete_time_dgp(n_obs=100, Delta=1.0, seed=42)

    # Get analytical gradient
    param_array = np.array([default_params[key] for key in model.param_keys])
    ll, analytical_grad = model.log_likelihood(param_array, test_data, Delta=1.0, grad=True)

    # Test each parameter
    for i, param_name in enumerate(model.param_keys):
        # Analytical gradient component
        analytical = analytical_grad[i]

        # Numerical gradient component
        def ll_func(m):
            param_array_num = np.array([m.param[key] for key in m.param_keys])
            ll_num = m.log_likelihood(param_array_num, test_data, Delta=1.0, grad=False)
            return ll_num

        numerical = numerical_derivative(ll_func, param_name, default_params,
                                         h=derivative_settings['h'])

        # Compare
        assert np.abs(analytical - numerical) < derivative_settings['atol'], \
            f"Log-likelihood gradient mismatch for {param_name}"
        assert np.abs(analytical - numerical) / (np.abs(analytical) + 1e-16) < derivative_settings['rtol'], \
            f"Log-likelihood gradient mismatch for {param_name}"


#
# model.intensity_matrix
#

def test_intensity_matrix_derivative_sparsity(default_params):
    """Test that intensity matrix derivatives maintain correct sparsity pattern."""
    model = create_model(params=default_params)
    Q, dQ = model.intensity_matrix()

    # Original matrix sparsity
    Q_nnz = Q.nnz

    # Check derivative sparsity
    for param_name in model.param_keys:
        dQ_param = dQ[param_name]

        # Derivatives should have same sparsity
        assert dQ_param.nnz <= Q_nnz, \
            f"Derivative for {param_name} too dense"

        # Derivatives should have same shape
        assert dQ_param.shape == Q.shape, \
            f"Shape mismatch for {param_name} derivative"


def test_intensity_matrix_derivatives(default_params, derivative_settings):
    """Test intensity matrix derivatives using finite differences."""
    model = create_model(params=default_params)
    Q_original, derivatives_original = model.intensity_matrix()

    # Test each parameter
    for param_name in model.param_keys:
        # Store original value
        original_value = model.param[param_name]
        h = derivative_settings['h']

        # Perturb parameter
        model.param[param_name] = original_value + h
        Q_perturbed, _ = model.intensity_matrix()

        # Compute finite difference approximation
        finite_diff = (Q_perturbed - Q_original) / h

        # Analytical derivative
        analytical_derivative = derivatives_original[param_name]

        # Reset parameter
        model.param[param_name] = original_value

        # Compute difference norm (using sparse matrix norm)
        difference = finite_diff - analytical_derivative
        difference_norm = norm(difference)

        # Relative error (avoid division by zero)
        analytical_norm = norm(analytical_derivative)
        if analytical_norm > 0:
            relative_error = difference_norm / analytical_norm
        else:
            relative_error = difference_norm

        # Check accuracy
        assert relative_error < derivative_settings['rtol'], \
            f"Intensity matrix derivative error too large for {param_name}: " \
            f"relative error = {relative_error:.2e}"


#
# sparse.vexpm_deriv
#

def test_vexpm_derivatives(default_params, derivative_settings):
    """Test matrix exponential derivatives using finite differences."""
    model = create_model(params=default_params)
    Delta = 1.0
    v = np.ones(model.K, dtype=np.float64)

    # Compute analytical derivatives
    Q_original, dQ_original = model.intensity_matrix()
    expQv_original, analytical_derivatives = vexpm_deriv(
        Q_original, dQ_original, Delta, v, epsilon=1e-12
    )

    # Test each parameter
    for param_name in model.param_keys:
        # Store original value
        original_value = model.param[param_name]
        h = derivative_settings['h']

        # Perturb parameter
        model.param[param_name] = original_value + h
        Q_perturbed, _ = model.intensity_matrix()

        # Compute matrix exponential for perturbed matrix
        expQv_perturbed = vexpm(Q_perturbed, Delta, v)

        # Numerical derivative
        derivative_numerical = (expQv_perturbed - expQv_original) / h

        # Analytical derivative
        analytical_derivative = analytical_derivatives[param_name]

        # Reset parameter
        model.param[param_name] = original_value

        # Check accuracy
        np.testing.assert_allclose(
            derivative_numerical,
            analytical_derivative,
            rtol=derivative_settings['rtol'],
            atol=derivative_settings['atol'],
            err_msg=f"vexpm derivative mismatch for {param_name}"
        )


@pytest.mark.parametrize("Delta", [0.1, 0.5, 2.0, 4.0, 10.0])
def test_vexpm_derivatives_various_delta(default_params, derivative_settings, Delta):
    """Test matrix exponential derivatives for various time intervals."""
    model = create_model(params=default_params)
    v = np.ones(model.K, dtype=np.float64)
    v = v / np.sum(v)

    # Compute analytical derivatives
    Q_original, dQ_original = model.intensity_matrix()
    expQv_original, analytical_derivatives = vexpm_deriv(
        Q_original, dQ_original, Delta, v, epsilon=1e-12
    )

    # Test lambda parameter (most sensitive to Delta)
    param_name = 'lambda'
    original_value = model.param[param_name]
    h = derivative_settings['h']

    # Perturb parameter
    model.param[param_name] = original_value + h
    Q_perturbed, _ = model.intensity_matrix()

    # Compute matrix exponential for perturbed matrix
    expQv_perturbed = vexpm(Q_perturbed, Delta, v)

    # Numerical derivative
    derivative_numerical = (expQv_perturbed - expQv_original) / h

    # Analytical derivative
    analytical_derivative = analytical_derivatives[param_name]

    # Reset parameter
    model.param[param_name] = original_value

    # Check accuracy
    np.testing.assert_allclose(
        derivative_numerical,
        analytical_derivative,
        rtol=derivative_settings['rtol'],
        atol=derivative_settings['atol'],
        err_msg=f"vexpm derivative mismatch for {param_name} with Delta={Delta}"
    )


def test_vexpm_derivatives_2x2(derivative_settings):
    """Test matrix exponential derivatives for a simple 2x2 case."""
    # Parameters for 2x2 test
    alpha = 1.0
    beta = 0.5
    Delta = 1.0
    h = derivative_settings['h']

    # Define Q matrix and derivatives
    Q = np.array([[-alpha, alpha],
                  [beta, -beta]], dtype=np.float64)
    Q_csr = csr_matrix(Q)

    # Analytical derivatives of Q
    dQ_alpha = np.array([[-1, 1],
                         [0, 0]], dtype=np.float64)
    dQ_alpha_csr = csr_matrix(dQ_alpha)
    dQ_beta = np.array([[0, 0],
                        [1, -1]], dtype=np.float64)
    dQ_beta_csr = csr_matrix(dQ_beta)
    dQ = {'alpha': dQ_alpha_csr, 'beta': dQ_beta_csr}

    # Test vector
    v = np.array([0.75, 0.25], dtype=np.float64)

    # Compute analytical derivatives
    expQv_original, analytical_derivatives = vexpm_deriv(
        Q_csr, dQ, Delta, v, epsilon=1e-12
    )

    # Test alpha derivative
    Q_alpha_perturbed = np.array([[-alpha - h, alpha + h],
                                  [beta, -beta]], dtype=np.float64)
    expQv_alpha_perturbed = vexpm(csr_matrix(Q_alpha_perturbed), Delta, v)
    dexpQv_alpha_numerical = (expQv_alpha_perturbed - expQv_original) / h

    np.testing.assert_allclose(
        dexpQv_alpha_numerical,
        analytical_derivatives['alpha'],
        rtol=derivative_settings['rtol'],
        atol=derivative_settings['atol'],
        err_msg="vexpm derivative mismatch for alpha in 2x2 case"
    )

    # Test beta derivative
    Q_beta_perturbed = np.array([[-alpha, alpha],
                                 [beta + h, -beta - h]], dtype=np.float64)
    expQv_beta_perturbed = vexpm(csr_matrix(Q_beta_perturbed), Delta, v)
    dexpQv_beta_numerical = (expQv_beta_perturbed - expQv_original) / h

    np.testing.assert_allclose(
        dexpQv_beta_numerical,
        analytical_derivatives['beta'],
        rtol=derivative_settings['rtol'],
        atol=derivative_settings['atol'],
        err_msg="vexpm derivative mismatch for beta in 2x2 case"
    )


def test_solve_implicit_derivatives_fixed_point():
    """Test that implicit derivatives satisfy fixed point condition."""
    param = {
        'theta_ec': -1.0,
        'theta_rn': -0.5,
        'theta_d': 0.5,
        'lambda': 1.0,
        'gamma': 0.5,
    }
    model = Model(n_players=2, n_demand=2, param=param, rho=0.1, verbose=False)

    # Get converged value function
    v, _ = model.value_function(vf_max_iter=200, vf_tol=1e-12)

    # Solve implicit derivatives
    dv = model.solve_implicit_derivatives(v)

    # Compute dT/dtheta
    dT_dtheta = model.dbellman_operator_dtheta(v)

    # Compute dT/dv
    dTv_dv = model.dbellman_operator_dv(v)

    # Check fixed point: dv = dT/dtheta + dT/dv * dv
    # This is approximately: (I - dT/dv) * dv = dT/dtheta
    from scipy.sparse import eye

    for param_key in model.param_keys:
        # Flatten dv for matrix operations
        dv_flat = dv[param_key].flatten()

        # Compute (I - dT/dv) * dv using sparse matrix operations
        id = eye(dTv_dv.shape[0], format='csc')
        lhs = (id - dTv_dv) @ dv_flat

        # Right side: dT/dtheta (flatten)
        rhs = dT_dtheta[param_key].flatten()

        # They should be equal within numerical tolerance
        relative_error = np.linalg.norm(lhs - rhs) / (np.linalg.norm(rhs) + 1e-10)
        assert relative_error < 0.001  # 0.1% tolerance
