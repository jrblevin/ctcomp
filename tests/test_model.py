"""
Test functions for Model methods and internal state.

This module contains unit tests for the Model class's internal
state space encodings and precomputed transition addresses,
data preprocessing, parameter updating, parameter estimation,
and other internal methods.

Derivative accuracy and correctness is tested in `test_derivatives.py`.

Cython functions are tested in `test_cython_sync.py`.

Specific return values of structural functions are tested in
`test_regression.py`.
"""

import pytest
import numpy as np
from model import Model

# Import CYTHON_AVAILABLE for tests
try:
    from model_cython import bellman_operator_cython
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


@pytest.mark.parametrize("n_players,n_demand", [
    (2, 2), (2, 3), (3, 2), (3, 3), (4, 3), (4, 4), (5, 3), (6, 3),
])
def test_encoding_decoding(n_players, n_demand):
    """Test that encoding and decoding states is consistent."""
    # Create model with minimal parameters
    param = {
        'theta_ec': -1.0,
        'theta_rn': -1.0,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 1.0
    }
    model = Model(n_players, n_demand, param, verbose=False)

    # Test all states
    for k in range(model.K):
        state = model.int_to_state[k]
        int_state = model.state_to_int[state]
        assert k == int_state, f"Encoding mismatch: state {k} -> {state} -> {int_state}"


def test_state_space_structure():
    """Test that state space has correct structure."""
    param = {
        'theta_ec': -1.0,
        'theta_rn': -1.0,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 1.0,
    }
    model = Model(2, 3, param, verbose=False)

    # Should have 2^2 * 3 = 12 states
    assert model.K == 12
    assert len(model.state_space) == 12
    assert len(model.demand_states) == 12
    assert len(model.player_states) == 12

    # Check demand states range
    assert np.min(model.demand_states) == 0
    assert np.max(model.demand_states) == 2

    # Check player states are binary
    assert np.all(np.isin(model.player_states.flatten(), [0, 1]))

    # Check number of active players is correct
    expected_n_active = np.sum(model.player_states, axis=1)
    assert np.array_equal(model.n_active, expected_n_active)


@pytest.mark.parametrize("n_players,n_demand", [
    (2, 2), (2, 3), (3, 2), (3, 3), (4, 3), (4, 4), (5, 3), (6, 3),
])
def test_state_space_structure_general(n_players, n_demand):
    """Test the structure of the state space."""
    param = {
        'theta_ec': -1.0,
        'theta_rn': -1.0,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 1.0,
    }
    model = Model(n_players=n_players, n_demand=n_demand, param=param, rho=0.1, verbose=False)

    # Test dimensions
    assert model.K == model.n_configs * model.n_demand
    assert model.n_configs == 2**model.n_players
    assert len(model.state_space) == model.K

    # Test state space ordering (demand states should be grouped)
    for i in range(len(model.state_space) - 1):
        d_curr = model.state_space[i][0]
        d_next = model.state_space[i + 1][0]
        # Demand should either stay same or increase by 1
        assert d_next == d_curr or d_next == d_curr + 1

    # Check demand states range
    assert np.min(model.demand_states) == 0
    assert np.max(model.demand_states) == n_demand - 1

    # Check player states are binary
    assert np.all(np.isin(model.player_states.flatten(), [0, 1]))

    # Check number of active players is correct
    expected_n_active = np.sum(model.player_states, axis=1)
    assert np.array_equal(model.n_active, expected_n_active)


@pytest.mark.parametrize("n_players,n_demand", [
    (2, 2), (2, 3), (3, 2), (3, 3), (4, 3), (5, 2), (5, 3), (5, 5),
])
def test_precomputed_addresses(n_players, n_demand):
    """Test precomputed addresses for various model sizes."""
    param = {
        'theta_ec': -1.0,
        'theta_rn': -1.0,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 1.0,
    }
    model = Model(n_players, n_demand, param, verbose=False)

    # Test all states
    for k in range(model.K):
        current_state = model.int_to_state[k]
        current_demand = current_state[0]
        current_players = current_state[1]

        # Test player switching
        for m in range(n_players):
            new_players = list(current_players)
            new_players[m] = 1 - new_players[m]
            expected_state = (current_demand, tuple(new_players))
            expected_index = model.state_to_int[expected_state]

            assert model.switch_indices[m, k] == expected_index, \
                f"Player {m} switch from state {k}: expected {expected_index}, got {model.switch_indices[m, k]}"

        # Test demand transitions
        if current_demand < n_demand - 1:
            expected_state = (current_demand + 1, current_players)
            expected_index = model.state_to_int[expected_state]
            assert model.k_demand_up[k] == expected_index, \
                f"Demand up from state {k}: expected {expected_index}, got {model.k_demand_up[k]}"

        if current_demand > 0:
            expected_state = (current_demand - 1, current_players)
            expected_index = model.state_to_int[expected_state]
            assert model.k_demand_down[k] == expected_index, \
                f"Demand down from state {k}: expected {expected_index}, got {model.k_demand_down[k]}"


def test_demand_transition_masks():
    """Test that demand transition masks are computed correctly."""
    param = {
        'theta_ec': -1.0,
        'theta_rn': -1.0,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 1.0
    }
    model = Model(3, 4, param, verbose=False)

    # Test demand up mask
    expected_up_mask = model.k_demand_up >= 0
    assert np.array_equal(model._demand_up_valid, expected_up_mask)

    # Test demand down mask
    expected_down_mask = model.k_demand_down >= 0
    assert np.array_equal(model._demand_down_valid, expected_down_mask)

    # Verify masks make sense
    # States with demand = n_demand - 1 should not have valid up transitions
    max_demand_states = model.demand_states == (model.n_demand - 1)
    assert not np.any(model._demand_up_valid[max_demand_states])

    # States with demand = 0 should not have valid down transitions
    min_demand_states = model.demand_states == 0
    assert not np.any(model._demand_down_valid[min_demand_states])


def test_switch_indices_shape():
    """Test that switch indices have correct shape and values."""
    param = {
        'theta_ec': -1.0,
        'theta_rn': -1.0,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 1.0,
    }
    model = Model(4, 3, param, verbose=False)

    # Check shape
    assert model.switch_indices.shape == (model.n_players, model.K)

    # Check that all indices are valid state indices
    assert np.all(model.switch_indices >= 0)
    assert np.all(model.switch_indices < model.K)

    # Check that switch indices are different from original state
    # (a player switching should go to a different state)
    for m in range(model.n_players):
        for k in range(model.K):
            switch_idx = model.switch_indices[m, k]
            assert switch_idx != k, f"Player {m} switch from state {k} leads to same state"


@pytest.mark.parametrize("n_players,n_demand", [
    (2, 2), (2, 3), (3, 2), (3, 3), (4, 2), (4, 3), (5, 2), (5, 3), (5, 5)
])
def test_model_initialization_consistency(n_players, n_demand):
    """Test that model initializes consistently for different sizes."""
    param = {
        'theta_ec': -1.0,
        'theta_rn': -1.0,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 1.0,
    }

    # Model should initialize without errors
    model = Model(n_players, n_demand, param, verbose=False)

    # Basic size checks
    assert model.n_players == n_players
    assert model.n_demand == n_demand
    assert model.n_configs == 2 ** n_players
    assert model.K == model.n_configs * n_demand

    # Arrays should have correct sizes
    assert len(model.state_space) == model.K
    assert model.demand_states.shape == (model.K,)
    assert model.player_states.shape == (model.K, n_players)
    assert model.n_active.shape == (model.K,)
    assert model.is_active.shape == (n_players, model.K)
    assert model.is_inactive.shape == (n_players, model.K)
    assert model.k_demand_up.shape == (model.K,)
    assert model.k_demand_down.shape == (model.K,)
    assert model.switch_indices.shape == (n_players, model.K)

    # Check that is_active and is_inactive are complements
    assert np.array_equal(model.is_active, ~model.is_inactive)

    # Check that n_active matches sum of player states
    expected_n_active = np.sum(model.player_states, axis=1)
    assert np.array_equal(model.n_active, expected_n_active)


def test_dbellman_operator_dv_shape():
    """Test that dbellman_operator_dv returns correct shape."""
    param = {
        'theta_ec': -1.0,
        'theta_rn': -0.5,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 0.5,
    }
    model = Model(n_players=2, n_demand=2, param=param, verbose=False)

    # Get a test value function
    v = np.random.randn(model.n_players, model.K)

    # Compute derivative
    dTv_dv = model.dbellman_operator_dv(v)

    # Check shape: should be sparse matrix (n_players*K, n_players*K)
    expected_shape = (model.n_players * model.K, model.n_players * model.K)
    assert dTv_dv.shape == expected_shape

    # Check it's a sparse matrix
    from scipy.sparse import issparse
    assert issparse(dTv_dv)


def test_dbellman_operator_dv_sparsity():
    """Test that dbellman_operator_dv has correct sparsity pattern."""
    param = {
        'theta_ec': -1.0,
        'theta_rn': -0.5,
        'theta_d': 1.0,
        'lambda': 2.0,
        'gamma': 1.0,
    }
    model = Model(n_players=3, n_demand=3, param=param, verbose=False)

    v = np.zeros((model.n_players, model.K))
    dTv_dv = model.dbellman_operator_dv(v)

    # Check that the format is CSC
    assert dTv_dv.format == 'csc'

    # Count non-zero elements using sparse matrix properties
    non_zero_count = dTv_dv.nnz
    total_elements = np.prod(dTv_dv.shape)

    # Should be sparse (most elements zero)
    sparsity = non_zero_count / total_elements
    assert sparsity < 0.2  # Less than 20% non-zero


def test_dbellman_operator_dv_diagonal():
    """Test diagonal elements of dbellman_operator_dv."""
    param = {
        'theta_ec': -1.0,
        'theta_rn': -0.5,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 0.5,
    }
    model = Model(n_players=3, n_demand=2, param=param, rho=0.1, verbose=False)

    v = np.random.randn(model.n_players, model.K)
    dTv_dv = model.dbellman_operator_dv(v)

    # Extract diagonal elements from sparse matrix
    diagonal = dTv_dv.diagonal()

    # All diagonal elements should be positive
    assert np.all(diagonal > 0)


def test_solve_implicit_derivatives_convergence():
    """Test that implicit derivatives can be computed."""
    param = {
        'theta_ec': -1.5,
        'theta_rn': -0.5,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 0.5,
    }
    model = Model(n_players=2, n_demand=2, param=param, verbose=False)

    # Get converged value function
    v, _ = model.value_function(vf_max_iter=100, vf_tol=1e-10)

    # Solve implicit derivatives
    dv = model.solve_implicit_derivatives(v)

    # Check that we get derivatives for all parameters
    assert set(dv.keys()) == set(model.param_keys)

    # Check shapes
    for key in dv:
        assert dv[key].shape == v.shape


def test_update_parameters_basic():
    """Test basic parameter update."""
    param = {
        'theta_ec': -1.0,
        'theta_rn': -0.5,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 0.5,
    }
    model = Model(n_players=2, n_demand=2, param=param, verbose=False)

    # New parameters
    new_theta = np.array([-2.0, -1.0, 2.0, 2.0, 1.0])

    # Update
    model.update_parameters(new_theta)

    # Check updates
    assert model.param['theta_ec'] == -2.0
    assert model.param['theta_rn'] == -1.0
    assert model.param['theta_d'] == 2.0
    assert model.param['lambda'] == 2.0
    assert model.param['gamma'] == 1.0


def test_update_parameters_array_size():
    """Test parameter update with wrong array size."""
    param = {
        'theta_ec': -1.0,
        'theta_rn': -0.5,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 0.5,
    }
    model = Model(n_players=2, n_demand=2, param=param, verbose=False)

    # Test with shorter array - should throw ValueError
    short_theta = np.array([-2.0, -1.0, 2.0])  # Only 3 elements
    with pytest.raises(ValueError):
        model.update_parameters(short_theta)

    # Check that no parameters were updated
    assert model.param['theta_ec'] == -1.0
    assert model.param['theta_rn'] == -0.5
    assert model.param['theta_d'] == 1.0
    assert model.param['lambda'] == 1.0
    assert model.param['gamma'] == 0.5


def test_discrete_time_dgp_basic():
    """Test basic data generation."""
    param = {
        'theta_ec': -1.0,
        'theta_rn': -0.5,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 0.5,
    }
    model = Model(n_players=3, n_demand=2, param=param, verbose=False)

    # Generate sample data
    n_obs = 20
    sample = model.discrete_time_dgp(n_obs=n_obs, Delta=1.0, seed=42)

    # Check that data has the correct type and length
    assert isinstance(sample, list)
    assert len(sample) == n_obs

    # Check that all are integers and valid states
    for obs in sample:
        assert isinstance(obs, (int, np.integer))
        assert 0 <= obs < model.K

    # Check that states can transition
    unique_states = set(sample)
    assert len(unique_states) > 1  # Should see multiple states


def test_discrete_time_dgp_seed_reproducibility():
    """Test that seed makes data generation reproducible."""
    param = {
        'theta_ec': -1.0,
        'theta_rn': -0.5,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 0.5,
    }
    model = Model(n_players=2, n_demand=2, param=param, verbose=False)

    # Generate with same seed twice
    sample1 = model.discrete_time_dgp(n_obs=50, Delta=1.0, seed=789)
    sample2 = model.discrete_time_dgp(n_obs=50, Delta=1.0, seed=789)

    # Should be identical
    for i in range(50):
        assert sample1[i] == sample2[i]

    # Different seed should give different data
    sample3 = model.discrete_time_dgp(n_obs=50, Delta=1.0, seed=790)

    # Should not be identical (with very high probability)
    differences = sum(1 for i in range(50) if sample1[i] != sample3[i])
    assert differences > 0


def test_preprocess_data():
    """Test basic data preprocessing."""
    param = {
        'theta_ec': -1.0,
        'theta_rn': -0.5,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 0.5,
    }
    model = Model(n_players=3, n_demand=2, param=param, verbose=False)

    # Check that D is not yet defined
    assert hasattr(model, 'D')
    assert model.D is None

    # Generate sample data
    sample = model.discrete_time_dgp(n_obs=100, Delta=1.0, seed=42)

    # Preprocess (populates model.D)
    result = model.preprocess_data(sample)

    # Method returns None but populates model.D
    assert result is None
    assert hasattr(model, 'D')

    # Check that D is a sparse matrix
    from scipy.sparse import issparse
    assert issparse(model.D)

    # D should be K x K
    assert model.D.shape == (model.K, model.K)

    # Check matrix has non-zero entries (data was processed)
    assert model.D.nnz > 20


def test_estimate_parameters_close():
    """Test that estimates are near true parameters."""
    true_param = {
        'theta_ec': -1.5,
        'theta_rn': -0.5,
        'theta_d': 1.0,
        'lambda': 2.0,
        'gamma': 1.0,
    }
    true_theta = np.array([
        true_param['theta_ec'],
        true_param['theta_rn'],
        true_param['theta_d'],
        true_param['lambda'],
        true_param['gamma']
    ])
    Delta = 1.0
    max_iter = 100
    model = Model(n_players=2, n_demand=2, param=true_param, verbose=False)

    # Generate a very large sample using true parameters with fixed seed
    np.random.seed(42)
    sample = model.discrete_time_dgp(n_obs=100000, Delta=Delta, seed=42)

    # Starting values (slightly perturbed from truth)
    start = true_theta + np.ones(5) * 0.1

    # Save initial log likelihood
    initial_ll = model.log_likelihood(start, sample, Delta=Delta, grad=False)

    # Estimate with limited iterations to save time
    result = model.estimate_parameters(
        sample, Delta=Delta, max_iter=max_iter, start=start, use_grad=True
    )

    # Compute final log likelihood
    final_ll = model.log_likelihood(result['x'], sample, Delta=Delta, grad=False)

    # Check that we get same result with same inputs
    result2 = model.estimate_parameters(
        sample, Delta=Delta, max_iter=max_iter, start=start, use_grad=True
    )

    # Results from two estimation runs with same inputs should be same
    np.testing.assert_allclose(result['x'], result2['x'], rtol=1e-10, err_msg='Should get same estimates with same inputs')

    # Check that objective improved or stayed same
    assert final_ll >= initial_ll, 'Final log likelihood should be greater than or equal to initial log likelihood'

    # Results should be close to true parameters
    np.testing.assert_allclose(result['x'], true_theta, rtol=0.2, err_msg='Estimated parameters should be close to true parameters')


def test_estimate_parameters_without_gradient():
    """Test estimation without analytical gradient."""
    param = {
        'theta_ec': -1.0,
        'theta_rn': -0.5,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 0.5,
    }
    model = Model(n_players=2, n_demand=2, param=param, verbose=False)

    # Generate small dataset
    sample = model.discrete_time_dgp(n_obs=50, Delta=1.0, seed=42)

    # Estimate without gradient
    result = model.estimate_parameters(
        sample, Delta=1.0, max_iter=10, use_grad=False
    )

    # Should return a result
    assert 'x' in result
    assert 'fun' in result
    assert result['x'].shape == (5,)


def test_count_intensity_nonzeros():
    """Test intensity matrix non-zero counting."""
    param = {
        'theta_ec': -1.0,
        'theta_rn': -0.5,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 0.5,
    }
    model = Model(n_players=3, n_demand=3, param=param, verbose=False)

    # Count non-zeros
    nnz = model._count_intensity_nonzeros()

    # Should be positive
    assert nnz > 0

    # Should match actual intensity matrix
    Q, _ = model.intensity_matrix()
    assert nnz == Q.nnz


def test_is_cython_enabled():
    """Test Cython availability check."""
    param = {
        'theta_ec': -1.0,
        'theta_rn': -0.5,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 0.5,
    }
    config = {
        'cython_threshold': 100,
    }
    # Test with small K (should not use Cython)
    model_small = Model(n_players=2, n_demand=2, param=param, config=config, verbose=False)
    cython_enabled_small = model_small.is_cython_enabled
    assert isinstance(cython_enabled_small, bool)
    assert not cython_enabled_small

    # Test with larger K above cython_threshold
    model_large = Model(n_players=5, n_demand=5, param=param, config=config, verbose=False)
    cython_enabled_large = model_large.is_cython_enabled
    assert isinstance(cython_enabled_large, bool)
    assert cython_enabled_large


def test_str_representation():
    """Test string representation of model."""
    param = {
        'theta_ec': -1.5,
        'theta_rn': -0.7,
        'theta_d': 1.2,
        'lambda': 2.0,
        'gamma': 1.5,
    }
    model = Model(n_players=3, n_demand=4, param=param, rho=0.1, verbose=False)

    # Test __str__
    str_repr = str(model)
    assert "n_players: 3" in str_repr
    assert "n_demand: 4" in str_repr
    assert "K: 32" in str_repr  # 2^3 * 4 = 32

    # Test __repr__
    repr_str = repr(model)
    assert "Model(n_players=3, n_demand=4)" == repr_str


def test_parameter_bounds():
    """Test that parameter bounds are correctly specified."""
    param = {
        'theta_ec': -1.0,
        'theta_rn': -1.0,
        'theta_d': 1.0,
        'lambda': 1.0,
        'gamma': 1.0,
    }
    model = Model(n_players=2, n_demand=2, param=param, rho=0.1, verbose=False)

    # Test bounds structure
    assert len(model.theta_bounds) == 5

    # Test specific bounds
    assert model.theta_bounds[0][1] <= 0.0    # theta_ec <= 0
    assert model.theta_bounds[1][1] <= 0.0    # theta_rn <= 0
    assert model.theta_bounds[2][0] >= 0.0    # theta_d >= 0
    assert model.theta_bounds[3][0] >= 0.0    # lambda > 0
    assert model.theta_bounds[4][0] >= 0.0    # gamma > 0


def test_parameter_validation():
    """Test parameter validation in Model.__init__."""
    valid_param = {
        'theta_ec': -1.0,
        'theta_rn': -0.5,
        'theta_d': 1.5,
        'lambda': 2.0,
        'gamma': 0.5,
    }

    # Test invalid n_players
    with pytest.raises(ValueError, match="n_players must be a positive integer"):
        Model(0, 2, valid_param)

    with pytest.raises(ValueError, match="n_players must be a positive integer"):
        Model(-1, 2, valid_param)

    with pytest.raises(ValueError, match="n_players must be a positive integer"):
        Model(2.5, 2, valid_param)

    # Test invalid n_demand
    with pytest.raises(ValueError, match="n_demand must be a positive integer"):
        Model(2, 0, valid_param)

    with pytest.raises(ValueError, match="n_demand must be a positive integer"):
        Model(2, -1, valid_param)

    # Test invalid param type
    with pytest.raises(TypeError, match="param must be a dictionary"):
        Model(2, 2, [1, 2, 3])

    # Test missing parameters
    incomplete_param = {'theta_ec': -1.0, 'theta_rn': -0.5}
    with pytest.raises(ValueError, match="Missing required parameters"):
        Model(2, 2, incomplete_param)

    # Test invalid parameter values
    invalid_param = valid_param.copy()
    invalid_param['lambda'] = -1.0
    with pytest.raises(ValueError, match="lambda must be positive"):
        Model(2, 2, invalid_param)

    invalid_param = valid_param.copy()
    invalid_param['gamma'] = 0.0
    with pytest.raises(ValueError, match="gamma must be positive"):
        Model(2, 2, invalid_param)

    invalid_param = valid_param.copy()
    invalid_param['theta_ec'] = np.inf
    with pytest.raises(ValueError, match="Parameter theta_ec must be a finite float"):
        Model(2, 2, invalid_param)

    invalid_param = valid_param.copy()
    invalid_param['theta_rn'] = np.nan
    with pytest.raises(ValueError, match="Parameter theta_rn must be a finite float"):
        Model(2, 2, invalid_param)

    # Test invalid rho
    with pytest.raises(ValueError, match="rho must be a non-negative finite float"):
        Model(2, 2, valid_param, rho=-0.1)

    with pytest.raises(ValueError, match="rho must be a non-negative finite float"):
        Model(2, 2, valid_param, rho=np.inf)


def test_update_parameters_validation():
    """Test parameter validation in update_parameters."""
    param = {
        'theta_ec': -1.0,
        'theta_rn': -0.5,
        'theta_d': 1.5,
        'lambda': 2.0,
        'gamma': 0.5,
    }
    model = Model(2, 2, param)

    # Test wrong number of parameters
    with pytest.raises(ValueError, match="Expected 5 parameters, got 3"):
        model.update_parameters([1, 2, 3])

    # Test invalid parameter values
    with pytest.raises(ValueError, match="lambda must be positive"):
        model.update_parameters([-1.0, -0.5, 1.5, -1.0, 0.5])

    with pytest.raises(ValueError, match="gamma must be positive"):
        model.update_parameters([-1.0, -0.5, 1.5, 2.0, 0.0])

    with pytest.raises(ValueError, match="Parameter theta_d.*must be a finite number"):
        model.update_parameters([-1.0, -0.5, np.inf, 2.0, 0.5])

    with pytest.raises(ValueError, match="Parameter lambda.*must be a finite number"):
        model.update_parameters([-1.0, -0.5, 1.5, np.nan, 0.5])


def test_config_validation():
    """Test configuration validation in Model.__init__."""
    valid_param = {
        'theta_ec': -1.0,
        'theta_rn': -0.5,
        'theta_d': 1.5,
        'lambda': 2.0,
        'gamma': 0.5,
    }

    # Test valid config
    valid_config = {
        'opt_max_iter': 200,
        'vf_max_iter': 1000,
        'vf_tol': 1e-10,
        'vf_algorithm': 'polyalgorithm',
        'vf_rtol': 0.2,
        'vf_max_newton_iter': 5,
        'vf_newton_solver': 'gmres',
        'use_cython': True,
    }
    model = Model(2, 2, valid_param, config=valid_config)
    assert model.config['opt_max_iter'] == 200
    assert model.config['vf_algorithm'] == 'polyalgorithm'

    # Test invalid config type
    with pytest.raises(TypeError, match="config must be a dictionary"):
        Model(2, 2, valid_param, config=[1, 2, 3])


def test_cython_threshold_configuration():
    """Test configurable Cython threshold functionality."""
    valid_param = {
        'theta_ec': -1.0,
        'theta_rn': -0.5,
        'theta_d': 1.5,
        'lambda': 2.0,
        'gamma': 0.5,
    }

    # Small model (2 players, 2 demand = 2^2 * 2 = 8 states)
    # Should not use Cython with default threshold of 200
    model_small = Model(2, 2, valid_param)
    assert model_small.K == 8
    if CYTHON_AVAILABLE:
        assert not model_small.is_cython_enabled  # K=8 < 200

    # Same small model but with low threshold should use Cython
    model_small_low_threshold = Model(2, 2, valid_param, config={'cython_threshold': 5})
    if CYTHON_AVAILABLE:
        assert model_small_low_threshold.is_cython_enabled  # K=8 > 5

    # Same small model but with high threshold should not use Cython
    model_small_high_threshold = Model(2, 2, valid_param, config={'cython_threshold': 1000})
    if CYTHON_AVAILABLE:
        assert not model_small_high_threshold.is_cython_enabled  # K=8 < 1000

    # Test with forced Cython (should override threshold)
    model_forced = Model(2, 2, valid_param, config={'use_cython': True, 'cython_threshold': 1000})
    if CYTHON_AVAILABLE:
        assert model_forced.is_cython_enabled  # Forced on regardless of threshold

    # Test with disabled Cython (should override threshold)
    model_disabled = Model(2, 2, valid_param, config={'use_cython': False, 'cython_threshold': 5})
    assert not model_disabled.is_cython_enabled  # Forced off regardless of threshold

    # Test zero threshold (Cython enabled for any K > 0)
    model_zero_threshold = Model(2, 2, valid_param, config={'cython_threshold': 0})
    if CYTHON_AVAILABLE:
        assert model_zero_threshold.is_cython_enabled  # K=8 > 0
