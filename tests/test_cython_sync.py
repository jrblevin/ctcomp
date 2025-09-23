#!/usr/bin/env python3
"""
Verify Python and Cython implementations produce identical results.
"""

import pytest
import numpy as np
from unittest.mock import patch
from model import Model
import model as model_module
from model_cython import (
    bellman_operator_cython,
    choice_probabilities_cython,
    _bellman_components_cython,
    dbellman_operator_dtheta_cython,
    dbellman_operator_dv_cython,
)

# Skip all tests if Cython is not available
pytestmark = pytest.mark.skipif(
    not hasattr(model_module, 'CYTHON_AVAILABLE') or not model_module.CYTHON_AVAILABLE,
    reason="Cython module not available"
)


@pytest.fixture
def test_params():
    """Standard test parameters."""
    return {
        'theta_ec': -0.5,
        'theta_rn': -0.7,
        'theta_d': 0.3,
        'gamma': 1.0,
        'lambda': 2.0
    }


@pytest.mark.parametrize("n_players,n_demand", [
    (2, 2),
    (3, 3),
    (4, 4),
])
def test_bellman_operator_sync(n_players, n_demand, test_params):
    """Test that Python and Cython bellman_operator produce identical results."""

    # Create model instance
    model = Model(n_players=n_players, n_demand=n_demand, param=test_params, verbose=False)

    # Get state space size
    K = model.K

    # Create random value function and derivatives
    np.random.seed(42)
    v = np.random.randn(model.n_players, K)

    # Store original Cython values
    original_cython = model_module.CYTHON_AVAILABLE
    try:
        # Force Python implementation
        model_module.CYTHON_AVAILABLE = False
        v_py = model.bellman_operator(v)

        # Force Cython implementation
        model_module.CYTHON_AVAILABLE = True
        v_cy, _ = bellman_operator_cython(
            v=v,
            theta_ec=model.param['theta_ec'],
            theta_rn=model.param['theta_rn'],
            theta_d=model.param['theta_d'],
            gamma=model.param['gamma'],
            lam=model.param['lambda'],
            rho=model.rho,
            n_players=model.n_players,
            K=model.K,
            is_active=model.is_active,
            is_inactive=model.is_inactive,
            n_active=model.n_active.astype(np.int32),
            demand_states=model.demand_states.astype(np.int32),
            k_demand_up=model.k_demand_up.astype(np.int32),
            k_demand_down=model.k_demand_down.astype(np.int32),
            switch_indices=model.switch_indices.astype(np.int32),
        )

        # Compare results
        v_diff = np.max(np.abs(v_py - v_cy))

        # Verify results are identical up to tol
        tol = 1e-13
        assert v_diff < tol, f"Value functions differ by {v_diff}"

    finally:
        # Restore original setting
        model_module.CYTHON_AVAILABLE = original_cython


def test_convergence_sync(test_params):
    """Test that both implementations converge to the same value function."""

    # Create model instance
    model = Model(n_players=3, n_demand=3, param=test_params, verbose=False)

    # Store original Cython values
    original_cython = model_module.CYTHON_AVAILABLE
    try:
        # Solve with Python implementation
        model_module.CYTHON_AVAILABLE = False
        v_py = model.value_iteration(vf_max_iter=200, vf_tol=1e-13)

        # Solve with Cython implementation
        model_module.CYTHON_AVAILABLE = True
        v_cy = model.value_iteration(vf_max_iter=200, vf_tol=1e-13)

        # Compare converged values
        v_diff = np.max(np.abs(v_py - v_cy))
        assert v_diff < 1e-13, f"Converged values differ by {v_diff}"

    finally:
        # Restore original setting
        model_module.CYTHON_AVAILABLE = original_cython


def test_bellman_operator_ones(test_params):
    """Test with all ones initial values."""

    # Create model instance
    model = Model(n_players=2, n_demand=2, param=test_params, verbose=False)

    # Ones value function
    v = np.ones((model.n_players, model.K))

    # Store original Cython values
    original_cython = model_module.CYTHON_AVAILABLE
    try:
        # Get Python result
        model_module.CYTHON_AVAILABLE = False
        v_py = model.bellman_operator(v)

        # Get Cython result
        model_module.CYTHON_AVAILABLE = True
        v_cy, _ = bellman_operator_cython(
            v=v,
            theta_ec=model.param['theta_ec'],
            theta_rn=model.param['theta_rn'],
            theta_d=model.param['theta_d'],
            gamma=model.param['gamma'],
            lam=model.param['lambda'],
            rho=model.rho,
            n_players=model.n_players,
            K=model.K,
            is_active=model.is_active,
            is_inactive=model.is_inactive,
            n_active=model.n_active.astype(np.int32),
            demand_states=model.demand_states.astype(np.int32),
            k_demand_up=model.k_demand_up.astype(np.int32),
            k_demand_down=model.k_demand_down.astype(np.int32),
            switch_indices=model.switch_indices.astype(np.int32),
        )

        # Compare
        assert np.allclose(v_py, v_cy, rtol=1e-12, atol=1e-12)

    finally:
        # Restore original setting
        model_module.CYTHON_AVAILABLE = original_cython


@pytest.mark.parametrize("n_players,n_demand", [
    (2, 2),
    (3, 3),
    (4, 4),
])
def test_bellman_components_sync(n_players, n_demand, test_params):
    """Test that Python and Cython _bellman_components produce identical results."""

    # Create model instance
    model = Model(n_players=n_players, n_demand=n_demand, param=test_params, verbose=False)

    # Create random value function
    np.random.seed(42)
    v = np.random.randn(model.n_players, model.K)

    # Store original Cython values
    original_cython = model_module.CYTHON_AVAILABLE
    try:
        # Force Python implementation
        model_module.CYTHON_AVAILABLE = False
        numerator_py, ratesum_py, p_py, psi_py = model._bellman_components(v)

        # Force Cython implementation
        model_module.CYTHON_AVAILABLE = True
        numerator_cy, ratesum_cy, p_cy, psi_cy = _bellman_components_cython(
            v=v,
            theta_ec=model.param['theta_ec'],
            theta_rn=model.param['theta_rn'],
            theta_d=model.param['theta_d'],
            gamma=model.param['gamma'],
            lam=model.param['lambda'],
            rho=model.rho,
            n_players=model.n_players,
            K=model.K,
            is_active=model.is_active,
            is_inactive=model.is_inactive,
            n_active=model.n_active.astype(np.int32),
            demand_states=model.demand_states.astype(np.int32),
            k_demand_up=model.k_demand_up.astype(np.int32),
            k_demand_down=model.k_demand_down.astype(np.int32),
            switch_indices=model.switch_indices.astype(np.int32),
        )

        # Compare results
        tol = 1e-13

        numerator_diff = np.max(np.abs(numerator_py - numerator_cy))
        assert numerator_diff < tol, f"Numerators differ by {numerator_diff}"

        ratesum_diff = np.max(np.abs(ratesum_py - ratesum_cy))
        assert ratesum_diff < tol, f"Rate sums differ by {ratesum_diff}"

        p_diff = np.max(np.abs(p_py - p_cy))
        assert p_diff < tol, f"Choice probabilities differ by {p_diff}"

        psi_diff = np.max(np.abs(psi_py - psi_cy))
        assert psi_diff < tol, f"Entry costs differ by {psi_diff}"

    finally:
        # Restore original setting
        model_module.CYTHON_AVAILABLE = original_cython


@pytest.mark.parametrize("n_players,n_demand", [
    (2, 2),
    (3, 3),
    (4, 4),
])
def test_choice_probabilities_sync(n_players, n_demand, test_params):
    """Test that Python and Cython choice_probabilities produce identical results."""

    # Create model instance
    model = Model(n_players=n_players, n_demand=n_demand, param=test_params, verbose=False)

    # Create random value function
    np.random.seed(42)
    v = np.random.randn(model.n_players, model.K)

    # Store original Cython values
    original_cython = model_module.CYTHON_AVAILABLE
    try:
        # Force Python implementation
        model_module.CYTHON_AVAILABLE = False
        p_py = model.choice_probabilities(v)

        # Force Cython implementation
        model_module.CYTHON_AVAILABLE = True
        p_cy, _, psi_cy = choice_probabilities_cython(
            v=v,
            dv_dict=None,  # No derivatives for this test
            theta_ec=model.param['theta_ec'],
            n_players=model.n_players,
            K=model.K,
            is_inactive=model.is_inactive,
            switch_indices=model.switch_indices.astype(np.int32),
        )

        # Compare results
        tol = 1e-13
        p_diff = np.max(np.abs(p_py - p_cy))
        assert p_diff < tol, f"Choice probabilities differ by {p_diff}"

    finally:
        # Restore original setting
        model_module.CYTHON_AVAILABLE = original_cython


def test_choice_probabilities_with_derivatives_uses_cython(test_params):
    """Test that choice_probabilities with derivatives works with Cython."""

    model = Model(n_players=2, n_demand=2, param=test_params, verbose=False,
                  config={'use_cython': True})  # Force Cython to be enabled

    # Check that Cython is available
    assert model.is_cython_enabled
    assert model_module.CYTHON_AVAILABLE

    # Create random value function and derivatives
    np.random.seed(42)
    v = np.random.randn(model.n_players, model.K)
    dv = {'theta_ec': np.random.randn(model.n_players, model.K)}

    with patch('model.choice_probabilities_cython') as mock_cython:
        # Set up mock to return plausible values
        mock_return_p = np.random.randn(model.n_players, model.K)
        mock_return_dp = {key: np.random.randn(model.n_players, model.K) for key in model.param_keys}
        mock_return_psi = np.random.randn(model.n_players, model.K)
        mock_cython.return_value = mock_return_p, mock_return_dp, mock_return_psi

        # This should now use Cython implementation with derivatives
        p, dp = model.choice_probabilities(v, dv)

        # Verify that the Cython function was used
        assert mock_cython.called, "Cython implementation was not called"
        mock_cython.assert_called_once()

        # Verify we get both probabilities and derivatives
        assert p.shape == (model.n_players, model.K)
        assert 'theta_ec' in dp
        assert dp['theta_ec'].shape == (model.n_players, model.K)

        # Verify the return values
        assert np.allclose(p, mock_return_p)
        for i, key in enumerate(model.param_keys):
            assert np.allclose(dp[key], mock_return_dp[key])


@pytest.mark.parametrize("n_players,n_demand", [
    (2, 2),
    (3, 3),
    (4, 4),
])
def test_choice_probabilities_derivatives_sync(n_players, n_demand, test_params):
    """Test that Python and Cython choice probability derivatives are identical."""

    # Create model instance
    model = Model(n_players=n_players, n_demand=n_demand, param=test_params, verbose=False)

    # Create random value function and derivatives
    np.random.seed(42)
    v = np.random.randn(model.n_players, model.K)
    dv = {key: np.random.randn(model.n_players, model.K) for key in model.param_keys}

    # Store original Cython values
    original_cython = model_module.CYTHON_AVAILABLE
    try:
        # Force Python implementation
        model_module.CYTHON_AVAILABLE = False
        p_py, dp_py = model.choice_probabilities(v, dv)

        # Force Cython implementation
        model_module.CYTHON_AVAILABLE = True
        p_cy, dp_cy, psi_cy = choice_probabilities_cython(
            v=v,
            dv_dict=dv,
            theta_ec=model.param['theta_ec'],
            n_players=model.n_players,
            K=model.K,
            is_inactive=model.is_inactive,
            switch_indices=model.switch_indices.astype(np.int32),
        )

        # Compare probabilities
        tol = 1e-13
        p_diff = np.max(np.abs(p_py - p_cy))
        assert p_diff < tol, f"Choice probabilities differ by {p_diff}"

        # Compare derivatives for each parameter
        for key in model.param_keys:
            dp_diff = np.max(np.abs(dp_py[key] - dp_cy[key]))
            assert dp_diff < tol, f"Derivatives for {key} differ by {dp_diff}"

    finally:
        # Restore original setting
        model_module.CYTHON_AVAILABLE = original_cython


def test_choice_probabilities_derivatives_mixed_params(test_params):
    """Test derivatives with a subset of parameters."""

    model = Model(n_players=2, n_demand=2, param=test_params, verbose=False)

    np.random.seed(42)
    v = np.random.randn(model.n_players, model.K)

    # Only provide derivatives for some parameters
    dv_partial = {
        'theta_ec': np.random.randn(model.n_players, model.K),
        'theta_d': np.random.randn(model.n_players, model.K)
    }

    # Store original Cython values
    original_cython = model_module.CYTHON_AVAILABLE
    try:
        # Force Python implementation
        model_module.CYTHON_AVAILABLE = False
        p_py, dp_py = model.choice_probabilities(v, dv_partial)

        # Force Cython implementation
        model_module.CYTHON_AVAILABLE = True
        p_cy, dp_cy = model.choice_probabilities(v, dv_partial)

        # Compare results
        tol = 1e-13
        p_diff = np.max(np.abs(p_py - p_cy))
        assert p_diff < tol, f"Choice probabilities differ by {p_diff}"

        # Compare derivatives for each provided parameter
        for key in dv_partial.keys():
            dp_diff = np.max(np.abs(dp_py[key] - dp_cy[key]))
            assert dp_diff < tol, f"Derivatives for {key} differ by {dp_diff}"

        # Check that other parameters are not provided
        for key in model.param_keys:
            if key not in dv_partial:
                assert key not in dp_cy

    finally:
        # Restore original setting
        model_module.CYTHON_AVAILABLE = original_cython


def test_choice_probabilities_derivatives_theta_ec_special_case(test_params):
    """Test choice probability derivatives with respect to theta_ec."""

    # Create model instance
    model = Model(n_players=2, n_demand=2, param=test_params, verbose=False)

    # Create random value function and derivatives
    np.random.seed(42)
    v = np.random.randn(model.n_players, model.K)
    dv = {'theta_ec': np.random.randn(model.n_players, model.K)}

    # Store original Cython values
    original_cython = model_module.CYTHON_AVAILABLE
    try:
        # Force Python implementation
        model_module.CYTHON_AVAILABLE = False
        p_py, dp_py = model.choice_probabilities(v, dv)

        # Force Cython implementation
        model_module.CYTHON_AVAILABLE = True
        p_cy, dp_cy = model.choice_probabilities(v, dv)

        # Results should be identical
        tol = 1e-13
        p_diff = np.max(np.abs(p_py - p_cy))
        assert p_diff < tol, f"Choice probabilities differ by {p_diff}"

        dp_diff = np.max(np.abs(dp_py['theta_ec'] - dp_cy['theta_ec']))
        assert dp_diff < tol, f"theta_ec derivatives differ by {dp_diff}"

        # Verify derivatives for inactive players are nonzero where expected
        for m in range(model.n_players):
            for k in range(model.K):
                if model.is_inactive[m, k]:
                    assert abs(dp_py['theta_ec'][m, k]) > 1e-14, \
                        f"Expected non-zero derivative for inactive player {m}, state {k}"

    finally:
        model_module.CYTHON_AVAILABLE = original_cython


@pytest.mark.parametrize("n_players,n_demand", [
    (2, 2),
    (3, 3),
    (4, 4),
])
def test_dbellman_operator_dtheta_sync(n_players, n_demand, test_params):
    """Test that Python and Cython dbellman_operator_dtheta are identical."""

    # Create model instance
    model = Model(n_players=n_players, n_demand=n_demand, param=test_params, verbose=False)

    # Create random value function
    np.random.seed(42)
    v = np.random.randn(model.n_players, model.K)

    # Store original Cython values
    original_cython = model_module.CYTHON_AVAILABLE
    try:
        # Force Python implementation
        model_module.CYTHON_AVAILABLE = False
        dT_dtheta_py = model.dbellman_operator_dtheta(v)

        # Force Cython implementation
        model_module.CYTHON_AVAILABLE = True
        dT_theta_ec, dT_theta_rn, dT_theta_d, dT_lambda, dT_gamma = dbellman_operator_dtheta_cython(
            v=v,
            theta_ec=model.param['theta_ec'],
            theta_rn=model.param['theta_rn'],
            theta_d=model.param['theta_d'],
            gamma=model.param['gamma'],
            lam=model.param['lambda'],
            rho=model.rho,
            n_players=model.n_players,
            K=model.K,
            is_active=model.is_active,
            is_inactive=model.is_inactive,
            n_active=model.n_active.astype(np.int32),
            demand_states=model.demand_states.astype(np.int32),
            k_demand_up=model.k_demand_up.astype(np.int32),
            k_demand_down=model.k_demand_down.astype(np.int32),
            switch_indices=model.switch_indices.astype(np.int32),
        )

        # Create dictionary from arrays (same order as in model.py)
        dT_dtheta_cy = {
            'theta_ec': dT_theta_ec,
            'theta_rn': dT_theta_rn,
            'theta_d': dT_theta_d,
            'lambda': dT_lambda,
            'gamma': dT_gamma
        }

        # Compare results for each parameter
        tol = 1e-13
        for key in model.param_keys:
            diff = np.max(np.abs(dT_dtheta_py[key] - dT_dtheta_cy[key]))
            assert diff < tol, f"Parameter {key} derivatives differ by {diff}"

    finally:
        # Restore original setting
        model_module.CYTHON_AVAILABLE = original_cython


def test_dbellman_operator_dtheta_with_zero_values(test_params):
    """Test dbellman_operator_dtheta with ones initial values."""

    # Create model instance
    model = Model(n_players=2, n_demand=2, param=test_params, verbose=False)

    # Ones value function
    v = np.ones((model.n_players, model.K))

    # Store original Cython values
    original_cython = model_module.CYTHON_AVAILABLE
    try:
        # Get Python result
        model_module.CYTHON_AVAILABLE = False
        dT_dtheta_py = model.dbellman_operator_dtheta(v)

        # Get Cython result
        model_module.CYTHON_AVAILABLE = True
        # Get arrays from Cython function
        dT_theta_ec, dT_theta_rn, dT_theta_d, dT_lambda, dT_gamma = dbellman_operator_dtheta_cython(
            v=v,
            theta_ec=model.param['theta_ec'],
            theta_rn=model.param['theta_rn'],
            theta_d=model.param['theta_d'],
            gamma=model.param['gamma'],
            lam=model.param['lambda'],
            rho=model.rho,
            n_players=model.n_players,
            K=model.K,
            is_active=model.is_active,
            is_inactive=model.is_inactive,
            n_active=model.n_active.astype(np.int32),
            demand_states=model.demand_states.astype(np.int32),
            k_demand_up=model.k_demand_up.astype(np.int32),
            k_demand_down=model.k_demand_down.astype(np.int32),
            switch_indices=model.switch_indices.astype(np.int32),
        )

        # Create dictionary from arrays (same order as in model.py)
        dT_dtheta_cy = {
            'theta_ec': dT_theta_ec,
            'theta_rn': dT_theta_rn,
            'theta_d': dT_theta_d,
            'lambda': dT_lambda,
            'gamma': dT_gamma
        }

        # Compare for each parameter
        for key in model.param_keys:
            assert np.allclose(dT_dtheta_py[key], dT_dtheta_cy[key], rtol=1e-13, atol=1e-13)

    finally:
        model_module.CYTHON_AVAILABLE = original_cython


def test_dbellman_operator_dtheta_uses_cython(test_params):
    """Test that dbellman_operator_dtheta now uses Cython implementation."""

    # Create model instance
    model = Model(n_players=2, n_demand=2, param=test_params, verbose=False,
                  config={'use_cython': True})  # Force Cython to be enabled

    # Check that Cython is available
    assert model.is_cython_enabled
    assert model_module.CYTHON_AVAILABLE

    # Random value function
    np.random.seed(42)
    v = np.random.randn(model.n_players, model.K)

    with patch('model.dbellman_operator_dtheta_cython') as mock_cython:
        # Set up mock to return plausible values
        mock_return = (
            np.zeros((model.n_players, model.K)),  # theta_ec
            np.zeros((model.n_players, model.K)),  # theta_rn
            np.zeros((model.n_players, model.K)),  # theta_d
            np.zeros((model.n_players, model.K)),  # lambda
            np.zeros((model.n_players, model.K))   # gamma
        )
        mock_cython.return_value = mock_return

        # Call the Python model function
        dT_dtheta = model.dbellman_operator_dtheta(v)

        # Verify that the Cython function was used
        assert mock_cython.called, "Cython implementation was not called"
        mock_cython.assert_called_once()

        # Verify the dictionary keys
        assert set(dT_dtheta.keys()) == set(model.param_keys)

        # Verify the return values
        for i, key in enumerate(model.param_keys):
            assert np.allclose(dT_dtheta[key], mock_return[i])


@pytest.mark.parametrize("n_players,n_demand", [
    (2, 2),
    (3, 3),
    (4, 4),
])
def test_dbellman_operator_dv_sync(n_players, n_demand, test_params):
    """Test that Python and Cython dbellman_operator_dv are identical."""

    # Create model instance
    model = Model(n_players=n_players, n_demand=n_demand, param=test_params, verbose=False)

    # Create random value function
    np.random.seed(42)
    v = np.random.randn(model.n_players, model.K)

    # Store original Cython values
    original_cython = model_module.CYTHON_AVAILABLE
    try:
        # Force Python implementation
        model_module.CYTHON_AVAILABLE = False
        dT_dv_py = model.dbellman_operator_dv(v)

        # Force Cython implementation
        model_module.CYTHON_AVAILABLE = True
        rows, cols, data, nnz = dbellman_operator_dv_cython(
            v=v,
            theta_ec=model.param['theta_ec'],
            theta_rn=model.param['theta_rn'],
            theta_d=model.param['theta_d'],
            lam=model.param['lambda'],
            gam=model.param['gamma'],
            rho=model.rho,
            n_players=model.n_players,
            K=model.K,
            is_active=model.is_active,
            is_inactive=model.is_inactive,
            n_active=model.n_active.astype(np.int32),
            demand_states=model.demand_states.astype(np.int32),
            k_demand_up=model.k_demand_up.astype(np.int32),
            k_demand_down=model.k_demand_down.astype(np.int32),
            switch_indices=model.switch_indices.astype(np.int32),
            demand_up_valid=model._demand_up_valid,
            demand_down_valid=model._demand_down_valid,
        )

        # Create sparse matrix from triplets
        from scipy.sparse import csc_matrix
        n_states = model.K * model.n_players
        dT_dv_cy = csc_matrix((data, (rows, cols)), shape=(n_states, n_states))

        # Compare results
        tolerance = 1e-12

        # Compare matrix shapes
        assert dT_dv_py.shape == dT_dv_cy.shape, f"Matrix shapes differ: {dT_dv_py.shape} vs {dT_dv_cy.shape}"

        # Compare non-zero patterns
        assert dT_dv_py.nnz == dT_dv_cy.nnz, f"Non-zero counts differ: {dT_dv_py.nnz} vs {dT_dv_cy.nnz}"

        # Compare matrix data
        diff_matrix = dT_dv_py - dT_dv_cy
        if diff_matrix.nnz > 0:
            max_diff = np.max(np.abs(diff_matrix.data))
            assert max_diff < tolerance, f"Jacobian matrices differ by {max_diff}"
        # If both matrices are empty (nnz == 0), they are identical by definition

    finally:
        # Restore original setting
        model_module.CYTHON_AVAILABLE = original_cython


def test_dbellman_operator_dv_ones(test_params):
    """Test dbellman_operator_dv with ones as initial values."""

    # Create model instance
    model = Model(n_players=2, n_demand=2, param=test_params, verbose=False)

    # Ones value function
    v = np.ones((model.n_players, model.K))

    # Store original Cython values
    original_cython = model_module.CYTHON_AVAILABLE
    try:
        # Force Python implementation
        model_module.CYTHON_AVAILABLE = False
        dT_dv_py = model.dbellman_operator_dv(v)

        # Force Cython implementation
        model_module.CYTHON_AVAILABLE = True
        dT_dv_cy = model.dbellman_operator_dv(v)

        # Test that Cython returns a sparse matrix
        assert dT_dv_cy.format == 'csc', "Cython implementation did not return a CSC sparse matrix"

        # Compare matrices
        diff_matrix = dT_dv_py - dT_dv_cy
        max_diff = np.max(np.abs(diff_matrix.data)) if diff_matrix.nnz > 0 else 0.0
        assert max_diff < 1e-12, f"Jacobian matrices differ by {max_diff}"

    finally:
        model_module.CYTHON_AVAILABLE = original_cython


def test_dbellman_operator_dv_uses_cython(test_params):
    """Test that dbellman_operator_dv now uses Cython implementation."""

    # Create model instance
    model = Model(n_players=2, n_demand=2, param=test_params, verbose=False,
                  config={'use_cython': True})  # Force Cython to be enabled

    # Create random value function
    np.random.seed(42)
    v = np.random.randn(model.n_players, model.K)

    with patch('model.dbellman_operator_dv_cython') as mock_cython:
        # Set up mock to return a plausible CSC sparse matrix
        from scipy.sparse import csc_matrix
        n_states = model.n_players * model.K
        nnz = int(0.1 * n_states * n_states)  # 10% density
        rows = np.random.randint(0, n_states, nnz)
        cols = np.random.randint(0, n_states, nnz)
        data = np.random.randn(nnz)
        mock_return = rows, cols, data, nnz
        mock_cython.return_value = mock_return

        # Construct the expected sparse matrix
        expected = csc_matrix((data, (rows, cols)), shape=(n_states, n_states))

        # Call the Python model function
        dT_dv = model.dbellman_operator_dv(v)

        # Verify that the Cython function was used
        assert mock_cython.called, "Cython implementation was not called"
        mock_cython.assert_called_once()

        # Verify we get a sparse matrix with the right shape
        assert dT_dv.shape == (n_states, n_states)
        assert dT_dv.format == 'csc'  # Should be a CSC sparse matrix
        assert hasattr(dT_dv, 'nnz')  # Should be a sparse matrix
        assert dT_dv.nnz == expected.nnz  # Should have the same nnz
