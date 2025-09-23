#!/usr/bin/env python3
"""
Unit tests for ConfigurableModel.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from model_configurable import ConfigurableModel
from optimization_config import OptimizationConfig
from model import Model


# Test parameters
TEST_PARAMS = {
    'theta_ec': -1.0,
    'theta_rn': -0.5,
    'theta_d': 0.8,
    'lambda': 0.2,
    'gamma': 0.1,
}


def test_initialization_with_default_config():
    """Test ConfigurableModel initialization with default config."""
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS)
    assert model.n_players == 2
    assert model.n_demand == 2
    assert model.K == 8  # 2 * 2^2
    assert isinstance(model.config, OptimizationConfig)
    assert model.config == OptimizationConfig.full()


def test_initialization_with_custom_config():
    """Test ConfigurableModel initialization with custom config."""
    config = OptimizationConfig.baseline()
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)
    assert model.config == config
    assert model.config.vectorize is False


def test_state_space_setup_baseline():
    """Test state space setup with baseline configuration."""
    config = OptimizationConfig.baseline()
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)

    # Check state space was created
    assert hasattr(model, 'state_space')
    assert hasattr(model, 'state_to_int')
    assert hasattr(model, 'int_to_state')
    assert len(model.state_space) == model.K

    # Check we don't have precomputed arrays
    assert not hasattr(model, 'demand_states')
    assert not hasattr(model, 'player_states')
    assert not hasattr(model, 'switch_indices')


def test_encoding_decoding():
    """Test state encoding and decoding."""
    config = OptimizationConfig.baseline()
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)

    # Test encoding and decoding round trip
    for k in range(model.K):
        state = model.decode_state(k)
        assert isinstance(state, tuple)
        assert len(state) == 2  # (demand, players)
        demand, players = state
        assert 0 <= demand < model.n_demand
        assert len(players) == model.n_players
        assert all(p in [0, 1] for p in players)

        # Round trip
        k_encoded = model.encode_state(demand, players)
        assert k_encoded == k


def test_choice_probabilities_sequential():
    """Test sequential choice probability computation."""
    config = OptimizationConfig(vectorize=False, sparse=False,
                                cython=False, derivatives=False)
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)

    v = np.random.randn(model.n_players, model.K)
    p = model.choice_probabilities(v)

    # Check shape and bounds
    assert p.shape == (model.n_players, model.K)
    assert np.all(p >= 0)
    assert np.all(p <= 1)


def test_choice_probabilities_vectorized():
    """Test vectorized choice probability computation."""
    config = OptimizationConfig(vectorize=True, sparse=False,
                                cython=False, derivatives=False)
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)

    v = np.random.randn(model.n_players, model.K)
    p = model.choice_probabilities(v)

    # Check shape and bounds
    assert p.shape == (model.n_players, model.K)
    assert np.all(p >= 0)
    assert np.all(p <= 1)


def test_bellman_operator_sequential():
    """Test sequential Bellman operator."""
    config = OptimizationConfig(vectorize=False, sparse=False,
                                cython=False, derivatives=False)
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)

    v = np.zeros((model.n_players, model.K))
    v_new = model.bellman_operator(v)

    assert v_new.shape == v.shape
    assert np.all(np.isfinite(v_new))


def test_bellman_operator_vectorized():
    """Test vectorized Bellman operator."""
    config = OptimizationConfig(vectorize=True, sparse=False,
                                cython=False, derivatives=False)
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)

    v = np.zeros((model.n_players, model.K))
    v_new = model.bellman_operator(v)

    assert v_new.shape == v.shape
    assert np.all(np.isfinite(v_new))


def test_value_function_convergence():
    """Test value function computation converges."""
    config = OptimizationConfig.baseline()
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)

    v, dv = model.value_function(vf_max_iter=1000, vf_tol=1e-10)

    assert v.shape == (model.n_players, model.K)
    assert np.all(np.isfinite(v))

    # Without derivatives configured, dv should be empty
    assert dv == {}


def test_value_function_with_derivatives():
    """Test value function with derivatives enabled."""
    config = OptimizationConfig(vectorize=True, sparse=False,
                                cython=False, derivatives=True)
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)

    v, dv = model.value_function(vf_max_iter=1000, vf_tol=1e-10)

    assert v.shape == (model.n_players, model.K)
    assert np.all(np.isfinite(v))

    # Check derivatives
    assert len(dv) == len(model.param_keys)
    for key in model.param_keys:
        assert key in dv
        assert dv[key].shape == v.shape
        assert np.all(np.isfinite(dv[key]))


def test_consistency_across_configs():
    """Test that different configurations produce consistent results."""
    n_players, n_demand = 2, 2
    params = TEST_PARAMS.copy()
    tol = 1e-8

    # Create models with different configurations
    config_baseline = OptimizationConfig.baseline()
    config_vectorized = OptimizationConfig(
        vectorize=True, sparse=False,
        cython=False, derivatives=False
    )

    model_baseline = ConfigurableModel(n_players, n_demand, params, config=config_baseline)
    model_vectorized = ConfigurableModel(n_players, n_demand, params, config=config_vectorized)

    # Compute value functions
    v_baseline, _ = model_baseline.value_function(vf_max_iter=1000, vf_tol=1e-10)
    v_vectorized, _ = model_vectorized.value_function(vf_max_iter=1000, vf_tol=1e-10)

    # Check consistency
    assert np.allclose(v_baseline, v_vectorized, rtol=tol, atol=tol)


def test_consistency_with_reference_models():
    """Test ConfigurableModel matches Model with different configurations."""
    n_players, n_demand = 2, 2
    params = TEST_PARAMS.copy()
    tol = 1e-8

    # Create reference model
    model_ref = Model(n_players, n_demand, params, config={'use_cython': False})

    # Create configurable models
    config_full = OptimizationConfig(
        vectorize=True, sparse=True,
        cython=False, derivatives=True
    )
    config_baseline = OptimizationConfig.baseline()

    model_config_full = ConfigurableModel(n_players, n_demand, params, config=config_full)
    model_config_baseline = ConfigurableModel(n_players, n_demand, params, config=config_baseline)

    # Compute value functions
    v_ref, _ = model_ref.value_function(vf_max_iter=1000, vf_tol=1e-10)
    v_config_full, _ = model_config_full.value_function(vf_max_iter=1000, vf_tol=1e-10)
    v_config_baseline, _ = model_config_baseline.value_function(vf_max_iter=1000, vf_tol=1e-10)

    # Check consistency
    # ConfigurableModel with sparse should match Model
    assert np.allclose(v_ref, v_config_full, rtol=tol, atol=tol)
    # All implementations should be consistent
    assert np.allclose(v_ref, v_config_baseline, rtol=tol, atol=tol)


def test_intensity_matrix_dense():
    """Test intensity matrix computation with dense matrices."""
    config = OptimizationConfig(vectorize=True, cython=True,
                                sparse=False, derivatives=False)
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)

    Q = model.intensity_matrix()

    # Check Q properties
    assert Q.shape == (model.K, model.K)
    # Row sums should be approximately zero (generator matrix property)
    if hasattr(Q, 'toarray'):  # sparse matrix
        row_sums = np.sum(Q.toarray(), axis=1)
    else:  # dense matrix
        row_sums = np.sum(Q, axis=1)
    assert np.allclose(row_sums, 0, atol=1e-10)


def test_parameter_validation():
    """Test parameter validation in initialization."""
    # Test missing parameter
    params_missing = TEST_PARAMS.copy()
    del params_missing['lambda']
    with pytest.raises(ValueError, match="Missing required parameters"):
        ConfigurableModel(n_players=2, n_demand=2, param=params_missing)

    # Test invalid parameter value
    params_invalid = TEST_PARAMS.copy()
    params_invalid['lambda'] = -0.1
    with pytest.raises(ValueError, match="lambda must be positive"):
        ConfigurableModel(n_players=2, n_demand=2, param=params_invalid)

    # Test invalid n_players
    with pytest.raises(ValueError, match="n_players must be a positive integer"):
        ConfigurableModel(n_players=0, n_demand=2, param=TEST_PARAMS)

    # Test invalid n_demand
    with pytest.raises(ValueError, match="n_demand must be a positive integer"):
        ConfigurableModel(n_players=2, n_demand=-1, param=TEST_PARAMS)


def test_method_routing_baseline():
    """Test that baseline configuration uses ConfigurableModel methods."""

    config = OptimizationConfig.baseline()
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)

    # Verify it's not using Model implementation
    assert not model._use_model_for_vf_ccp()
    assert not model._use_model_for_q_ll()
    assert model._impl is None


def test_method_routing_vectorize():
    """Test that vectorize configuration uses Model methods."""
    config = OptimizationConfig(vectorize=True, sparse=False,
                                polyalgorithm=False, cython=False,
                                derivatives=False)
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)

    # Verify VF & CCP methods use Model, Q & LL methods use ConfigurableModel
    assert model._use_model_for_vf_ccp()
    assert not model._use_model_for_q_ll()
    assert isinstance(model._impl, Model)

    # Check Model configuration
    assert model._impl.config['vf_algorithm'] == 'value_iteration'
    assert model._impl.config['use_cython'] is False

    # Test that methods are delegated to Model
    v = np.random.randn(model.n_players, model.K)

    # Test choice_probabilities - should use Model's method
    with patch.object(model._impl, 'choice_probabilities') as mock_cp:
        mock_cp.return_value = np.zeros((model.n_players, model.K))
        model.choice_probabilities(v)
        mock_cp.assert_called_once_with(v)

    # Test bellman_operator - should use Model, not ConfigurableModel methods
    with patch.object(model._impl, 'bellman_operator') as mock_bo:
        # Call ConfigurableModel.bellman_operator
        model.bellman_operator(v)
        # Check that Model's bellman_operator was called
        mock_bo.assert_called_once_with(v)

    with patch.object(model._impl, 'value_function') as mock_vf:
        mock_vf.return_value = (np.zeros_like(v), {})
        model.value_function()
        mock_vf.assert_called_once()


def test_method_routing_polyalgorithm():
    """Test that polyalgorithm configuration uses Model with correct settings."""
    config = OptimizationConfig(vectorize=True, sparse=False,
                                polyalgorithm=True, cython=False,
                                derivatives=False)
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)

    # Verify Model configuration
    assert model._impl.config['vf_algorithm'] == 'polyalgorithm'
    assert model._impl.config['use_cython'] is False


def test_method_routing_cython():
    """Test that cython configuration uses Model with Cython enabled."""
    config = OptimizationConfig(vectorize=True, sparse=False,
                                polyalgorithm=True, cython=True,
                                derivatives=False)
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)

    # Verify Model configuration
    assert model._impl.config['vf_algorithm'] == 'polyalgorithm'
    assert model._impl.config['use_cython']
    assert model._impl.config['cython_threshold'] == 0  # Force Cython


def test_method_routing_sparse():
    """Test that sparse configuration uses Model methods for all operations."""
    config = OptimizationConfig(vectorize=True, sparse=True,
                                polyalgorithm=True, cython=True,
                                derivatives=False)
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)

    # Verify Model configuration
    assert model._impl.config['use_cython']
    assert model._impl.config['cython_threshold'] == 0  # Force Cython
    assert model._impl.config['vf_algorithm'] == 'polyalgorithm'

    # Create sample data
    sample = [0, 1, 2]  # Sequence of states

    # Test log_likelihood routing to Model (sparse uses Model's log_likelihood)
    with patch.object(model._impl, 'log_likelihood') as mock_ll_model, \
         patch('model_configurable.expm') as mock_expm:
        mock_ll_model.return_value = -10.0
        theta = np.array([model.param[key] for key in model.param_keys])
        _ = model.log_likelihood(theta, sample)
        mock_ll_model.assert_called_once_with(theta, sample, 1.0, grad=False)
        # Should NOT use ConfigurableModel's expm-based implementation
        mock_expm.assert_not_called()

    # Test estimate_parameters routing to Model for sparse (sparse uses Model for Q & LL)
    with patch.object(model._impl, 'estimate_parameters') as mock_ep_model:
        mock_ep_model.return_value = type('obj', (object,), {'x': np.zeros(5)})()
        # Should use Model's estimate_parameters for sparse
        _ = model.estimate_parameters(sample, Delta=1.0, max_iter=1)
        mock_ep_model.assert_called_once()


def test_method_routing_derivatives():
    """Test that derivatives configuration uses Model for everything."""
    config = OptimizationConfig.full()  # All optimizations incl. derivatives
    cm = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)

    # Test sample
    sample = [0, 1, 2]

    # Test log_likelihood routing to Model
    with patch.object(cm._impl, 'log_likelihood') as mock_ll, \
         patch('model_configurable.expm') as mock_expm:
        mock_ll.return_value = (-10.0, np.zeros(5))
        theta = np.array([cm.param[key] for key in cm.param_keys])
        _ = cm.log_likelihood(theta, sample, grad=True)
        # SHOULD call Model's log_likelihood
        mock_ll.assert_called_once_with(theta, sample, 1.0, grad=True)
        # Should NOT use ConfigurableModel's expm-based implementation
        mock_expm.assert_not_called()

    # Test estimate_parameters routing to Model
    with patch.object(cm._impl, 'estimate_parameters') as mock_ep, \
         patch('model_configurable.minimize') as mock_minimize:
        mock_ep.return_value = type('obj', (object,), {'x': np.zeros(5)})()
        start = np.ones(5)
        _ = cm.estimate_parameters(sample, Delta=1.0, max_iter=1, start=start, use_grad=True)
        mock_ep.assert_called_once()
        # Verify correct arguments are passed through to model._impl
        call_args = mock_ep.call_args
        assert call_args[0][0] == sample                 # sample
        assert call_args[0][1] == 1.0                    # Delta
        assert call_args[0][2] == 1                      # max_iter
        assert np.allclose(call_args[0][3], np.ones(5))  # start
        # use_grad is passed as keyword argument
        assert call_args[1]['use_grad']

        # Should NOT use ConfigurableModel's minimize
        mock_minimize.assert_not_called()


def test_log_likelihood_routing_by_config():
    """Test log_likelihood method routing based on configuration."""
    sample = [0, 1, 2]  # Sequence of states

    # Test 1: baseline/vectorize/polyalgorithm/cython should all
    # use ConfigurableModel's dense matrix exponential implementations of
    # log_likelihood function, without analytical derivatives.  However,
    # we should use Model's Cython implementation  and intensity
    # matrix.
    configs_using_configurable = [
        OptimizationConfig.baseline(),
        OptimizationConfig(vectorize=True, sparse=False,
                           polyalgorithm=False, cython=False,
                           derivatives=False),
        OptimizationConfig(vectorize=True, sparse=False,
                           polyalgorithm=True, cython=False,
                           derivatives=False),
        OptimizationConfig(vectorize=True, sparse=False,
                           polyalgorithm=True, cython=True,
                           derivatives=False),
    ]

    for config in configs_using_configurable:
        cm = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)

        # For non-Model implementations, patch intensity_matrix
        with patch.object(cm, 'intensity_matrix') as mock_im:
            Q_dense = np.zeros((cm.K, cm.K))
            np.fill_diagonal(Q_dense, -1.0)  # Make it a valid Q matrix
            mock_im.return_value = Q_dense  # Return just the matrix, not tuple

            # Patch expm to avoid computation
            with patch('model_configurable.expm') as mock_expm:
                mock_expm.return_value = np.eye(cm.K)
                theta = np.array([cm.param[key] for key in cm.param_keys])
                ll = cm.log_likelihood(theta, sample)

                # Verify intensity_matrix was called
                assert mock_im.called
                # Verify expm was called (ConfigurableModel's implementation)
                assert mock_expm.called

    # Test 2: sparse/derivatives use Model's log_likelihood
    configs_using_model = [
        OptimizationConfig(vectorize=True,
                           polyalgorithm=True, cython=True,

                           sparse=True, derivatives=False),
        OptimizationConfig.full(),  # derivatives=True
    ]

    for config in configs_using_model:
        cm = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)

        with patch.object(cm._impl, 'log_likelihood') as mock_m_ll, \
             patch('model_configurable.expm') as mock_cm_expm:
            mock_m_ll.return_value = (-10.0, np.zeros(5))
            mock_cm_expm.return_value = np.eye(cm.K)
            theta = np.array([cm.param[key] for key in cm.param_keys])
            ll, grad = cm.log_likelihood(theta, sample, 1.0, grad=config.derivatives)

            # ConfigurableModel's log_likelihood should hand off the call to
            # Model's log_likelihood.
            mock_m_ll.assert_called_once_with(theta, sample, 1.0, grad=config.derivatives)
            # ConfigurableModel's expm should NOT be called
            mock_cm_expm.assert_not_called()


def test_estimate_parameters_routing_by_config():
    """Test estimate_parameters method routing and use_grad setting."""
    sample = [0, 1, 2]  # Sequence of states

    # Test 1: baseline, vectorize, polyalgorithm, cython use ConfigurableModel's estimate_parameters
    configs_using_configurable = [
        # baseline
        OptimizationConfig.baseline(),
        # vectorize
        OptimizationConfig(vectorize=True,
                           polyalgorithm=False, cython=False,
                           sparse=False, derivatives=False),
        # polyalgorithm
        OptimizationConfig(vectorize=True,
                           polyalgorithm=True, cython=False,
                           sparse=False, derivatives=False),
        # cython
        OptimizationConfig(vectorize=True,
                           polyalgorithm=True, cython=True,
                           sparse=False, derivatives=False),
    ]

    for config in configs_using_configurable:
        model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)

        with patch('model_configurable.minimize') as mock_minimize:
            mock_minimize.return_value = type('obj', (object,), {
                'x': np.array([p for p in TEST_PARAMS.values()]),
                'success': True
            })()

            _ = model.estimate_parameters(sample, Delta=1.0, max_iter=1)

            # scipy.optimize.minimize should be called
            mock_minimize.assert_called_once()
            # Check jac=None (finite differences, not analytical gradient)
            call_args = mock_minimize.call_args
            assert call_args[1]['jac'] is None

    # Test 2: sparse config uses Model's estimate_parameters with use_grad=False
    config_sparse = OptimizationConfig(vectorize=True,
                                       polyalgorithm=True, cython=True,
                                       sparse=True, derivatives=False)
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config_sparse)

    with patch.object(model._impl, 'estimate_parameters') as mock_ep:
        mock_ep.return_value = type('obj', (object,), {
            'x': np.array([p for p in TEST_PARAMS.values()]),
            'success': True
        })()

        _ = model.estimate_parameters(sample, Delta=1.0, max_iter=1)

        # Model's estimate_parameters should be called
        mock_ep.assert_called_once()

    # Test 3: derivatives config uses Model's estimate_parameters with use_grad=True
    config = OptimizationConfig.full()  # derivatives=True
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)

    with patch.object(model._impl, 'estimate_parameters') as mock_ep:
        mock_ep.return_value = type('obj', (object,), {
            'x': np.array([p for p in TEST_PARAMS.values()]),
            'success': True
        })()

        _ = model.estimate_parameters(sample, Delta=1.0, max_iter=1)

        # Model's estimate_parameters should be called
        mock_ep.assert_called_once()


def test_configuration_hierarchy():
    """Test that configuration options follow the correct hierarchy."""
    # Each level should enable all previous optimizations

    # Baseline
    config = OptimizationConfig.baseline()
    assert not any([config.vectorize, config.sparse,
                    config.polyalgorithm, config.cython,
                    config.derivatives])

    # Sequential configs from OptimizationConfig.sequential()
    configs = OptimizationConfig.sequential()

    # Check each level
    assert configs[0][0] == "baseline"

    assert configs[1][0] == "vectorize"
    assert configs[1][1].vectorize and not configs[1][1].polyalgorithm

    assert configs[2][0] == "polyalgorithm"
    assert all([configs[2][1].vectorize,
                configs[2][1].polyalgorithm]) and not configs[2][1].cython

    assert configs[3][0] == "cython"
    assert all([configs[3][1].vectorize,
                configs[3][1].polyalgorithm, configs[3][1].cython]) and not configs[3][1].sparse

    assert configs[4][0] == "sparse"
    assert all([configs[4][1].vectorize,
                configs[4][1].polyalgorithm, configs[4][1].cython,
                configs[4][1].sparse]) and not configs[4][1].derivatives

    assert configs[5][0] == "derivatives"
    assert all([configs[5][1].vectorize,
                configs[5][1].polyalgorithm, configs[5][1].cython,
                configs[5][1].sparse, configs[5][1].derivatives])


def test_complete_routing_specification():
    """
    Test that the complete routing specification is correct.

    This is a comprehensive test that verifies each optimization level uses the
    correct implementation methods.
    """
    sample = [0, 1, 2]  # Sequence of observed states

    # Define optimization configs according to specification
    optimization_configs = {
        'baseline': {
            'config': OptimizationConfig.baseline(),
            'expected_settings': {},
            'uses_model_vf': False,    # Value function uses ConfigurableModel
            'uses_model_ll': False,    # Log likelihood uses ConfigurableModel
        },
        'vectorize': {
            'config': OptimizationConfig(vectorize=True, sparse=False,
                                         polyalgorithm=False, cython=False,
                                         derivatives=False),
            'expected_settings': {
                'vf_algorithm': 'value_iteration',
                'use_cython': False,
            },
            'uses_model_vf': True,     # Value function uses Model
            'uses_model_ll': False,    # Log likelihood uses ConfigurableModel
        },
        'polyalgorithm': {
            'config': OptimizationConfig(vectorize=True, sparse=False,
                                         polyalgorithm=True, cython=False,
                                         derivatives=False),
            'expected_settings': {
                'vf_algorithm': 'polyalgorithm',
                'use_cython': False,
            },
            'uses_model_vf': True,     # Value function uses Model
            'uses_model_ll': False,    # Log likelihood uses ConfigurableModel
        },
        'cython': {
            'config': OptimizationConfig(vectorize=True, sparse=False,
                                         polyalgorithm=True, cython=True,
                                         derivatives=False),
            'expected_settings': {
                'vf_algorithm': 'polyalgorithm',
                'use_cython': True,
                'cython_threshold': 0,
            },
            'uses_model_vf': True,     # Value function uses Model
            'uses_model_ll': False,    # Log likelihood uses ConfigurableModel
        },
        'sparse': {
            'config': OptimizationConfig(vectorize=True, sparse=True,
                                         polyalgorithm=True, cython=True,
                                         derivatives=False),
            'expected_settings': {
                'vf_algorithm': 'polyalgorithm',
                'use_cython': True,
                'cython_threshold': 0,
            },
            'uses_model_vf': True,     # Value function uses Model
            'uses_model_ll': True,     # Log likelihood uses Model
        },
        'derivatives': {
            'config': OptimizationConfig(vectorize=True, sparse=True,
                                         polyalgorithm=True, cython=True,
                                         derivatives=True),
            'expected_settings': {
                'vf_algorithm': 'polyalgorithm',
                'use_cython': True,
                'cython_threshold': 0,
            },
            'uses_model_vf': True,     # Value function uses Model
            'uses_model_ll': True,     # Log likelihood uses Model
        },
    }

    for name, spec in optimization_configs.items():
        model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS,
                                  config=spec['config'])

        # Check value function & CCP settings
        if spec['uses_model_vf']:
            assert model._use_model_for_vf_ccp(), f"{name}: Should use Model for VF & CCP"
            assert isinstance(model._impl, Model), f"{name}: Should have Model instance"

            # Check Model settings
            for key, expected_val in spec['expected_settings'].items():
                actual_val = model._impl.config.get(key)
                assert actual_val == expected_val, \
                    f"{name}: Expected {key}={expected_val}, got {actual_val}"
        else:
            assert not model._use_model_for_vf_ccp(), f"{name}: Should not use Model for VF & CCP"
            assert model._impl is None, f"{name}: Should not have Model instance"

        # Test bellman_operator, choice_probabilities, and value_function
        v = np.random.randn(model.n_players, model.K)
        if spec['uses_model_vf']:
            # Should use Model's methods
            with patch.object(model._impl, 'bellman_operator') as mock_bo, \
                 patch.object(model._impl, 'choice_probabilities') as mock_cp, \
                 patch.object(model._impl, 'value_function') as mock_vf:
                mock_bo.return_value = v
                mock_cp.return_value = np.zeros((model.n_players, model.K))
                mock_vf.return_value = (v, {})

                # Call the methods
                model.bellman_operator(v)
                model.choice_probabilities(v)
                model.value_function(vf_max_iter=10, vf_tol=1e-6)

                # Verify Model's methods were called
                mock_bo.assert_called_once_with(v)
                mock_cp.assert_called_once_with(v)
                mock_vf.assert_called_once()
        else:
            # Should use ConfigurableModel's own methods. We can't easily mock
            # these since they're the actual methods being tested, but we can
            # verify they don't delegate to Model.
            assert model._impl is None
            _ = model.bellman_operator(v)
            _ = model.choice_probabilities(v)

        # Test log_likelihood routing
        if spec['uses_model_ll']:
            # Should use Model's log_likelihood
            with patch.object(model._impl, 'log_likelihood') as mock_ll:
                mock_ll.return_value = (-10.0, np.zeros(5)) if name == 'derivatives' else -10.0
                theta = np.array([model.param[key] for key in model.param_keys])
                _ = model.log_likelihood(theta, sample)
                mock_ll.assert_called_once()
        else:
            # Should use ConfigurableModel's log_likelihood.
            # ConfigurableModel.log_likelihood() calls self.intensity_matrix()
            # which only delegates to Model for sparse/derivatives configs.
            Q_dense = np.zeros((model.K, model.K))
            np.fill_diagonal(Q_dense, -1.0)

            # Patch ConfigurableModel's intensity_matrix since that's what
            # ConfigurableModel.log_likelihood() calls directly.
            with patch.object(model, 'intensity_matrix') as mock_im:
                # ConfigurableModel.intensity_matrix returns Q (not dQ)
                mock_im.return_value = Q_dense
                # Patch expm in the module where it's used
                with patch('model_configurable.expm') as mock_expm:
                    mock_expm.return_value = np.eye(model.K)
                    theta = np.array([model.param[key] for key in model.param_keys])
                    _ = model.log_likelihood(theta, sample)
                    assert mock_expm.called
                    assert mock_im.called

        # Test estimate_parameters routing
        if name in ['sparse', 'derivatives']:
            # Should use Model's estimate_parameters
            with patch.object(model._impl, 'estimate_parameters') as mock_ep:
                # Return valid parameter values
                mock_ep.return_value = MagicMock(x=np.array([-1.0, -1.0, 1.0, 1.0, 1.0]))
                model.estimate_parameters(sample, Delta=1.0, max_iter=1)
                mock_ep.assert_called_once()
                # Check that use_grad=True is passed for derivatives
                if name == 'derivatives':
                    call_kwargs = mock_ep.call_args[1] if len(mock_ep.call_args) > 1 else {}
                    if 'use_grad' in call_kwargs:
                        assert call_kwargs['use_grad']
        else:
            # Should use ConfigurableModel's estimate_parameters, which calls
            # scipy.optimize.minimize imported by model_configurable.
            with patch('model_configurable.minimize') as mock_min:
                # Return valid parameter values
                mock_min.return_value = MagicMock(x=np.array([-1.0, -1.0, 1.0, 1.0, 1.0]), success=True)
                model.estimate_parameters(sample, Delta=1.0, max_iter=1)
                mock_min.assert_called()


def test_no_method_cross_contamination():
    """
    Test that methods are not called from the wrong implementation.

    This test ensures there's no cross-contamination between ConfigurableModel
    and Model implementations.
    """
    # Test baseline - should never use Model methods
    config = OptimizationConfig.baseline()
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)
    v = np.random.randn(model.n_players, model.K)

    # Since model._impl is None, we can't patch it, but let's verify it stays None
    assert model._impl is None

    # Test vectorize - should delegate to Model for VF & CCP methods
    config = OptimizationConfig(vectorize=True, sparse=False,
                                polyalgorithm=False, cython=False,
                                derivatives=False)
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)

    # Since vectorize uses Model for VF & CCP, check that Model methods are called
    with patch.object(model._impl, 'choice_probabilities') as mock_cp, \
         patch.object(model._impl, 'bellman_operator') as mock_bo:
        mock_cp.return_value = np.zeros((model.n_players, model.K))
        mock_bo.return_value = np.zeros((model.n_players, model.K))

        # Call the public methods
        _ = model.choice_probabilities(v)
        _ = model.bellman_operator(v)

        # Model methods should be called
        mock_cp.assert_called_once()
        mock_bo.assert_called_once()

    # Test sparse - ConfigurableModel's expm should not be used for log_likelihood
    config = OptimizationConfig(vectorize=True, sparse=True,
                                polyalgorithm=True, cython=True,
                                derivatives=False)
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)
    sample = [0, 1]  # Sequence of states

    with patch.object(model._impl, 'log_likelihood') as mock_ll, \
         patch('model_configurable.expm') as mock_expm, \
         patch.object(model, 'intensity_matrix') as mock_im:
        mock_ll.return_value = (-5.0, np.zeros(5))

        theta = np.array([model.param[key] for key in model.param_keys])
        _ = model.log_likelihood(theta, sample)

        # Model's log_likelihood should be called
        mock_ll.assert_called_once()
        # ConfigurableModel's expm and intensity_matrix should NOT be called
        mock_expm.assert_not_called()
        mock_im.assert_not_called()

    # Test derivatives - ConfigurableModel's minimize should not be used
    config = OptimizationConfig.full()
    model = ConfigurableModel(n_players=2, n_demand=2, param=TEST_PARAMS, config=config)

    with patch.object(model._impl, 'estimate_parameters') as mock_ep, \
         patch('model_configurable.minimize') as mock_min, \
         patch.object(model, 'log_likelihood') as mock_ll:
        mock_ep.return_value = MagicMock(x=np.zeros(5), success=True)

        _ = model.estimate_parameters(sample, Delta=1.0, max_iter=1)

        # Model's estimate_parameters should be called
        mock_ep.assert_called_once()
        # ConfigurableModel's minimize and log_likelihood should NOT be called
        mock_min.assert_not_called()
        mock_ll.assert_not_called()


def test_vf_ccp_method_routing_verification():
    """
    Explicitly verify that VF & CCP methods are routed correctly.
    """
    # Define expected routing for VF & CCP methods per specification table
    vf_ccp_routing = {
        'baseline': 'ConfigurableModel',
        'vectorize': 'Model',
        'polyalgorithm': 'Model',
        'cython': 'Model',
        'sparse': 'Model',
        'derivatives': 'Model'
    }

    for name, config in OptimizationConfig.sequential():
        model = ConfigurableModel(2, 2, TEST_PARAMS, config=config)
        expected_impl = vf_ccp_routing[name]

        print(f"\nTesting {name} - expected VF & CCP: {expected_impl}")

        if expected_impl == 'Model':
            # Should delegate to Model
            assert model._use_model_for_vf_ccp(), f"{name} should use Model for VF & CCP"
            assert model._impl is not None, f"{name} should have Model instance"

            # Test that Model methods are actually called
            v = np.random.randn(2, 8)

            with patch.object(model._impl, 'bellman_operator', return_value=v) as mock_bo, \
                 patch.object(model._impl, 'choice_probabilities', return_value=v) as mock_cp, \
                 patch.object(model._impl, 'value_function', return_value=(v, {})) as mock_vf:

                # Call methods
                model.bellman_operator(v)
                model.choice_probabilities(v)
                model.value_function(vf_max_iter=10)

                # Verify Model methods were called
                mock_bo.assert_called_once()
                mock_cp.assert_called_once()
                mock_vf.assert_called_once()

        else:  # ConfigurableModel
            # Should NOT delegate to Model
            assert not model._use_model_for_vf_ccp(), f"{name} should use ConfigurableModel for VF & CCP"
            assert model._impl is None, f"{name} should not have Model instance"


def test_q_ll_method_routing_verification():
    """
    Explicitly verify that Q & LL methods are routed correctly for each optimization level.
    """
    # Define expected routing for Q & LL methods per specification table
    q_ll_routing = {
        'baseline': 'ConfigurableModel',
        'vectorize': 'ConfigurableModel',
        'polyalgorithm': 'ConfigurableModel',
        'cython': 'ConfigurableModel',
        'sparse': 'Model',
        'derivatives': 'Model'
    }

    sample = [0, 1, 2]

    for name, config in OptimizationConfig.sequential():
        model = ConfigurableModel(2, 2, TEST_PARAMS, config=config)
        expected_impl = q_ll_routing[name]

        print(f"\nTesting {name} - expected Q & LL: {expected_impl}")

        if expected_impl == 'Model':
            # Should delegate to Model
            assert model._use_model_for_q_ll(), f"{name} should use Model for Q & LL"

            # Test intensity_matrix routing
            with patch.object(model._impl, 'intensity_matrix') as mock_im:
                mock_im.return_value = np.array([[1.0, 2.0], [3.0, 4.0]]), {}
                model.intensity_matrix()
                mock_im.assert_called_once()

            # Test log_likelihood routing
            with patch.object(model._impl, 'log_likelihood', return_value=-5.0) as mock_ll:
                theta = np.array([TEST_PARAMS[key] for key in model.param_keys])
                model.log_likelihood(theta, sample)
                mock_ll.assert_called_once()

            # Test estimate_parameters routing
            with patch.object(model._impl, 'estimate_parameters') as mock_ep:
                mock_ep.return_value = MagicMock(x=np.array([-1.0, -1.0, 1.0, 1.0, 1.0]))
                model.estimate_parameters(sample, max_iter=1)
                mock_ep.assert_called_once()

        else:  # ConfigurableModel
            # Should NOT delegate to Model for Q & LL
            assert not model._use_model_for_q_ll(), f"{name} should use ConfigurableModel for Q & LL"

            # For log_likelihood, should use ConfigurableModel's implementation
            # We need to patch intensity_matrix and expm to verify the path
            with patch.object(model, 'intensity_matrix', return_value=np.eye(8)) as mock_im, \
                 patch('model_configurable.expm', return_value=np.eye(8)) as mock_expm:

                theta = np.array([TEST_PARAMS[key] for key in model.param_keys])
                model.log_likelihood(theta, sample)

                # Should use ConfigurableModel's path (intensity_matrix + expm)
                mock_im.assert_called_once()
                mock_expm.assert_called_once()

            # For estimate_parameters, should use ConfigurableModel's minimize
            with patch('model_configurable.minimize') as mock_min:
                mock_min.return_value = MagicMock(x=np.array([-1.0, -1.0, 1.0, 1.0, 1.0]), success=True)
                model.estimate_parameters(sample, max_iter=1)
                mock_min.assert_called_once()


@pytest.mark.parametrize("n_players,n_demand", [
    (2, 2), (2, 3), (3, 2), (3, 3), (4, 2), (4, 3), (5, 2), (5, 3), (5, 5)
])
def test_intensity_matrix_consistency(n_players, n_demand):
    """
    Test that ConfigurableModel and Model produce identical intensity matrices.

    This is critical for ensuring baseline and cython configurations produce
    identical log-likelihood values during estimation benchmarks.
    """
    params = TEST_PARAMS.copy()

    # Create baseline ConfigurableModel (uses its own intensity_matrix)
    baseline_config = OptimizationConfig.baseline()
    baseline_model = ConfigurableModel(n_players, n_demand, params, config=baseline_config)

    # Create cython ConfigurableModel (delegates intensity_matrix to Model)
    cython_config = OptimizationConfig(vectorize=True, polyalgorithm=True,
                                       cython=True, sparse=False, derivatives=False)
    cython_model = ConfigurableModel(n_players, n_demand, params, config=cython_config)

    # Create reference Model directly
    ref_model = Model(n_players, n_demand, params, verbose=False)

    # Compute intensity matrices
    Q_baseline = baseline_model.intensity_matrix()
    Q_cython = cython_model.intensity_matrix()
    Q_ref, _ = ref_model.intensity_matrix()  # Model returns (Q, dQ) tuple

    # Convert sparse to dense if needed
    if hasattr(Q_ref, 'toarray'):
        Q_ref = Q_ref.toarray()
    if hasattr(Q_cython, 'toarray'):
        Q_cython = Q_cython.toarray()

    # All should be identical
    assert Q_baseline.shape == Q_cython.shape == Q_ref.shape
    assert np.allclose(Q_baseline, Q_cython, rtol=1e-13, atol=1e-15), \
        "Baseline and cython intensity matrices should be identical"
    assert np.allclose(Q_baseline, Q_ref, rtol=1e-13, atol=1e-15), \
        "Baseline and Model intensity matrices should be identical"

    # Verify intensity matrix properties
    for Q in [Q_baseline, Q_cython, Q_ref]:
        # Row sums should be zero (generator matrix property)
        row_sums = np.sum(Q, axis=1)
        assert np.allclose(row_sums, 0, atol=1e-10), "Row sums should be zero"

        # Off-diagonal elements should be non-negative
        Q_off_diag = Q.copy()
        np.fill_diagonal(Q_off_diag, 0)
        assert np.all(Q_off_diag >= -1e-15), "Off-diagonal elements should be non-negative"

        # Diagonal elements should be non-positive
        diag_elements = np.diag(Q)
        assert np.all(diag_elements <= 1e-15), "Diagonal elements should be non-positive"


def test_log_likelihood_numerical_consistency():
    """
    ConfigurableModel and Model should produce same log-likelihood values.
    """
    n_players, n_demand = 2, 2
    params = TEST_PARAMS.copy()

    # Generate a sample path
    np.random.seed(42)  # For reproducibility
    sample = np.random.randint(0, n_players * n_demand, size=50).tolist()
    Delta = 1.0

    # Create reference Model
    ref_model = Model(n_players, n_demand, params, verbose=False)
    theta = np.array([params[key] for key in ref_model.param_keys])
    ll_ref = ref_model.log_likelihood(theta, sample, Delta, grad=False)

    # Test all ConfigurableModel configurations
    configs_to_test = OptimizationConfig.sequential()

    # Check log likelihood function values
    log_likelihood_values = {}
    for name, config in configs_to_test:
        model = ConfigurableModel(n_players, n_demand, params, config=config)
        ll = model.log_likelihood(theta, sample, Delta, grad=False)
        log_likelihood_values[name] = ll

        # Check consistency with reference
        assert np.allclose(ll, ll_ref, rtol=1e-10, atol=1e-12), \
            f"{name} log-likelihood {ll} differs from reference {ll_ref}"

    # Additional cross-checks between configurations: baseline, vectorize,
    # polyalgorithm, cython should be identical.
    for config1 in ["baseline", "vectorize", "polyalgorithm", "cython"]:
        for config2 in ["baseline", "vectorize", "polyalgorithm", "cython"]:
            assert np.allclose(log_likelihood_values[config1],
                               log_likelihood_values[config2],
                               rtol=1e-13, atol=1e-15), \
                f"{config1} and {config2} should produce identical log-likelihood values"

    # sparse should match Model exactly
    assert np.allclose(log_likelihood_values["sparse"], ll_ref,
                       rtol=1e-13, atol=1e-15), \
        "Sparse configuration should match Model log-likelihood exactly"
