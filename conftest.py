"""
Shared pytest fixtures and configuration for continuous-time entry model tests.
"""

import pytest
import os
import json
from model import Model


@pytest.fixture(scope='session')
def baselines():
    """Load baseline results."""
    # Get the directory where this test file is located
    test_dir = os.path.dirname(os.path.abspath(__file__))
    baseline_path = os.path.join(test_dir, 'tests', 'test_baselines.json')
    try:
        with open(baseline_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        pytest.skip("test_baselines.json not found. Run generate_test_baselines.py first.")


@pytest.fixture(scope='session')
def tolerances():
    """Define tolerance levels for different tests."""
    return {
        'value_function': 1e-12,
        'derivative': 1e-12,
        'choice_prob': 1e-12,
        'intensity': 1e-12,
        'log_likelihood': 1e-12,
        'gradient': 1e-12,
        'algorithm_consistency': 1e-12,
    }


def create_model(config, param_set_name=None):
    """Create a model instance from configuration."""
    model_config = {
        k: config[k] for k in [
            'opt_max_iter',
            'vf_max_iter',
            'vf_tol',
            'vf_algorithm',
            'vf_rtol',
            'vf_max_newton_iter',
            'vf_newton_solver'
        ] if k in config
    }

    return Model(
        n_players=config['n_players'],
        n_demand=config['n_demand'],
        param=config['params'],
        rho=config['rho'],
        verbose=False,
        config=model_config,
    )
