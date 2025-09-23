#!/usr/bin/env python3
"""
Tests of value function algorithms.
"""

import pytest  # noqa: F401
import numpy as np
from model import Model


def test_algorithm_options():
    """Test the three algorithm options work correctly."""
    # Create a model to test algorithm options
    model = Model(
        n_players=2,
        n_demand=2,
        param={
            'theta_ec': -1.0,
            'theta_rn': -0.5,
            'theta_d': 1.0,
            'lambda': 1.0,
            'gamma': 1.0,
        },
        rho=0.05,
        verbose=False,
    )

    # Test that all three algorithm options work
    for algorithm in ['value_iteration', 'polyalgorithm']:
        v, dv = model.value_function(vf_algorithm=algorithm, vf_max_iter=10)
        assert v.shape == (2, model.K)
        assert isinstance(dv, dict)
        assert all(key in dv for key in model.param_keys)


def test_vfi_vs_polyalgorithm_behavior():
    """Test that 'value_iteration' and 'polyalgorithm' behave differently."""
    # Use verbose output to capture algorithm execution
    import io
    import contextlib

    # 9 x 5 model = 2560, large enough to show difference
    n_players = 9
    n_demand = 5
    param = {
        'theta_ec': -1.0,
        'theta_rn': -0.1,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 1.0,
    }

    # Initialize test model
    model = Model(
        n_players=n_players,
        n_demand=n_demand,
        param=param,
        rho=0.3,
        verbose=True,  # Enable verbose to capture output
        config={'vf_max_iter': 100}  # Enough iterations to converge
    )

    # Test 'value_iteration' and capture output
    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer):
        v_vfi, dv_vfi = model.value_function(vf_algorithm='value_iteration', vf_max_iter=3)
    output_vfi = output_buffer.getvalue()
    assert "Using algorithm: value_iteration" in output_vfi, "value_iteration algorithm should be used"

    # Clear any cached value function to ensure polyalgorithm runs fresh
    model._value_cache = None
    model._param_cache = None

    # Test 'polyalgorithm' and capture output
    output_buffer_poly = io.StringIO()
    with contextlib.redirect_stdout(output_buffer_poly):
        v_poly, dv_poly = model.value_function(vf_algorithm='polyalgorithm', vf_max_iter=3, vf_max_newton_iter=1)
    output_poly = output_buffer_poly.getvalue()

    # Polyalgorithm should include Newton phase when max_iter is low
    assert "Phase 2: Newton-Kantorovich" in output_poly, "polyalgorithm should have Newton phase when VI doesn't converge"


def test_newton_method_accuracy():
    """Test that Newton method achieves high accuracy."""
    # Use a moderately challenging case
    test_params = {
        'theta_ec': -0.12,
        'theta_rn': -2.5,
        'theta_d': 1.0,
        'lambda': 4.0,
        'gamma': 0.5,
    }
    vf_tol = 1e-13

    # Test with different solvers
    for solver in ['direct', 'gmres']:
        model = Model(
            n_players=3,
            n_demand=2,
            param=test_params,
            rho=0.01,
            verbose=False,
            config={
                'vf_algorithm': 'polyalgorithm',
                'vf_max_iter': 200,
                'vf_tol': vf_tol,
                'vf_newton_solver': solver,
                'vf_max_newton_iter': 15,
            }
        )
        v, dv = model.value_function()

        # Verify high accuracy achieved
        residual = np.max(np.abs(model.bellman_operator(v) - v))
        assert residual < vf_tol, \
            f"Newton method with {solver} solver should achieve high accuracy, got {residual:.2e}"

        # Verify derivatives are computed
        assert len(dv) == 5, "Should compute derivatives for all parameters"
        for param_name in model.param_keys:
            assert np.all(np.isfinite(dv[param_name])), \
                f"Derivative {param_name} should be finite"


def test_extreme_parameters(tolerances):
    """Test extreme parameter values."""
    cases = [
        {
            'name': 'Very Low Entry Cost',
            'params': {'theta_ec': -0.001, 'theta_rn': -1.0, 'theta_d': 1.0, 'lambda': 2.0, 'gamma': 0.5},
            'rho': 0.05,
        },
        {
            'name': 'Very High Competition',
            'params': {'theta_ec': -0.2, 'theta_rn': -8.0, 'theta_d': 1.5, 'lambda': 3.0, 'gamma': 0.8},
            'rho': 0.02,
        },
        {
            'name': 'High Action Rate',
            'params': {'theta_ec': -0.1, 'theta_rn': -1.5, 'theta_d': 1.0, 'lambda': 7.0, 'gamma': 1.0},
            'rho': 0.02,
        }
    ]
    vf_tol = tolerances['value_function']

    for case in cases:
        model = Model(
            n_players=5,
            n_demand=5,
            param=case['params'],
            rho=case['rho'],
            verbose=False,
            config={
                'vf_algorithm': 'polyalgorithm',
                'vf_max_iter': 1000,
                'vf_tol': vf_tol,
                'vf_rtol': 0.1,
                'vf_max_newton_iter': 30,
                'vf_newton_solver': 'auto',
            }
        )

        # Should complete without error
        v, dv = model.value_function()

        # Basic validation
        assert np.all(np.isfinite(v)), f"Value function should be finite for {case['name']}"
        assert np.all(np.isfinite([dv[p] for p in model.param_keys])), \
            f"Derivatives should be finite for {case['name']}"

        # Check convergence
        residual = np.max(np.abs(model.bellman_operator(v) - v))
        assert residual < vf_tol, \
            f"Should achieve strict tolerance for {case['name']}, got {residual:.2e}"


def test_vfi_vs_polyalgorithm_rescue():
    """Test that polyalgorithm succeeds where value iteration fails."""
    # strong_competition baseline
    config = {
        'n_players': 4,
        'n_demand': 2,
        'params': {
            'theta_ec': -0.1,
            'theta_rn': -3.0,
            'theta_d': 1.0,
            'lambda': 5.0,
            'gamma': 0.5,
        },
        'rho': 0.01,
    }
    vf_tol = 1e-13
    vf_max_iter = 50

    # Test 1: Value iteration should not converge within vf_max_iter iterations
    model_vi = Model(
        n_players=config['n_players'],
        n_demand=config['n_demand'],
        param=config['params'],
        rho=config['rho'],
        verbose=False,
        config={
            'vf_algorithm': 'value_iteration',
            'vf_max_iter': vf_max_iter,
            'vf_tol': vf_tol,
        }
    )
    v_vi = model_vi.value_iteration(vf_max_iter=vf_max_iter, vf_tol=vf_tol)

    # Check if VI reached convergence (it shouldn't with limited iterations)
    residual_vi = np.max(np.abs(model_vi.bellman_operator(v_vi) - v_vi))

    print(f"Value iteration residual: {residual_vi}")

    # Test 2: Polyalgorithm should succeed
    model_poly = Model(
        n_players=config['n_players'],
        n_demand=config['n_demand'],
        param=config['params'],
        rho=config['rho'],
        verbose=False,
        config={
            'vf_algorithm': 'polyalgorithm',
            'vf_max_iter': vf_max_iter,
            'vf_tol': vf_tol,
            'vf_rtol': 0.1,
            'vf_max_newton_iter': 20,
            'vf_newton_solver': 'auto',
        }
    )

    v_poly, _ = model_poly.value_function()
    residual_poly = np.max(np.abs(model_poly.bellman_operator(v_poly) - v_poly))

    print(f"Polyyalgorithm residual: {residual_poly}")

    # Polyalgorithm should achieve MUCH better convergence
    assert residual_poly < 1e-10 * residual_vi, \
        f"Polyalgorithm should achieve better convergence: {residual_poly:.2e} vs VI {residual_vi:.2e}"

    # Polyalgorithm should achieve target tolerance
    assert residual_poly < vf_tol, \
        f"Polyalgorithm should reach target tolerance, got residual {residual_poly:.2e}"
