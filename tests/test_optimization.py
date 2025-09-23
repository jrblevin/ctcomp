"""
Unit tests for OptimizationConfig and optimization sequence.
"""

import pytest  # noqa: F401
import numpy as np
from optimization_config import OptimizationConfig
from model_configurable import ConfigurableModel


def test_default_config():
    """Test default configuration has all optimizations enabled."""
    config = OptimizationConfig()
    assert config.vectorize is True
    assert config.sparse is True
    assert config.polyalgorithm is True
    assert config.cython is True
    assert config.derivatives is True


def test_full_config():
    """Test full() class method returns all optimizations enabled."""
    config = OptimizationConfig.full()
    assert config.vectorize is True
    assert config.sparse is True
    assert config.polyalgorithm is True
    assert config.cython is True
    assert config.derivatives is True


def test_baseline_config():
    """Test baseline() class method returns all optimizations disabled."""
    config = OptimizationConfig.baseline()
    assert config.vectorize is False
    assert config.sparse is False
    assert config.polyalgorithm is False
    assert config.cython is False
    assert config.derivatives is False


def test_copy_method():
    """Test copy() creates independent copy."""
    config1 = OptimizationConfig(vectorize=True, sparse=True)
    config2 = config1.copy()

    # Verify copy has same values
    assert config2.vectorize == config1.vectorize
    assert config2.sparse == config1.sparse

    # Verify they are independent
    config2.sparse = False
    assert config1.sparse is True
    assert config2.sparse is False


def test_sequential():
    """Test sequential() generates correct sequence."""
    configs = OptimizationConfig.sequential()

    # Should have 8 configurations: baseline + 7 optimizations
    assert len(configs) == 6

    # Check names
    expected_names = [
        "baseline",
        "vectorize",
        "polyalgorithm",
        "cython",
        "sparse",
        "derivatives",
    ]
    names = [name for name, _ in configs]
    assert names == expected_names

    # Check configurations are cumulative
    name, config = configs[0]
    assert name == "baseline"
    assert all(not v for v in config.to_dict().values())

    name, config = configs[1]
    assert config.vectorize is True

    name, config = configs[2]
    assert config.vectorize is True
    assert config.polyalgorithm is True
    assert config.sparse is False

    # Final config
    name, config = configs[-1]
    assert config.vectorize is True
    assert config.polyalgorithm is True
    assert config.cython is True
    assert config.sparse is True


def test_to_dict():
    """Test to_dict() method."""
    config = OptimizationConfig(
        vectorize=True,
        sparse=True,
        polyalgorithm=True,
        cython=True,
        derivatives=False
    )
    d = config.to_dict()
    assert d == {
        'vectorize': True,
        'sparse': True,
        'polyalgorithm': True,
        'cython': True,
        'derivatives': False
    }


def test_from_dict():
    """Test from_dict() class method."""
    d = {
        'vectorize': True,
        'sparse': False,
        'cython': True,
        'derivatives': True
    }
    config = OptimizationConfig.from_dict(d)
    assert config.vectorize is True
    assert config.sparse is False
    assert config.cython is True
    assert config.derivatives is True


def test_str_method():
    """Test __str__() method."""
    # Test with no optimizations
    config = OptimizationConfig.baseline()
    assert str(config) == "OptimizationConfig(no optimizations)"

    # Test with some optimizations
    config = OptimizationConfig(
        vectorize=True,
        sparse=False,
        polyalgorithm=True,
        cython=False,
        derivatives=False
    )
    assert str(config) == "OptimizationConfig(vectorize, polyalgorithm)"

    # Test with all optimizations
    config = OptimizationConfig.full()
    expected = "OptimizationConfig(vectorize, sparse, polyalgorithm, cython, derivatives)"
    assert str(config) == expected


def test_summary_method():
    """Test summary() method."""
    config = OptimizationConfig(
        vectorize=False,
        sparse=True,
        polyalgorithm=False,
        cython=False,
        derivatives=False
    )
    summary = config.summary()
    assert "Optimization Configuration:" in summary
    assert "Vectorized operations: False" in summary
    assert "Sparse matrix operations: True" in summary
    assert "Polyalgorithm value function: False" in summary
    assert "Cython acceleration: False" in summary
    assert "Analytical derivatives: False" in summary


def test_dataclass_features():
    """Test standard dataclass features work correctly."""
    # Test equality
    config1 = OptimizationConfig(vectorize=True, sparse=False)
    config2 = OptimizationConfig(vectorize=True, sparse=False)
    config3 = OptimizationConfig(vectorize=False, sparse=False)

    assert config1 == config2
    assert config1 != config3

    # Test field access
    config = OptimizationConfig()
    assert hasattr(config, 'vectorize')
    assert hasattr(config, 'sparse')
    assert hasattr(config, 'polyalgorithm')
    assert hasattr(config, 'cython')
    assert hasattr(config, 'derivatives')


def test_optimization_sequence():
    """Test that each optimization in the sequence produces valid results."""
    # Test parameters
    params = {
        'theta_ec': -1.0,
        'theta_rn': -0.5,
        'theta_d': 0.8,
        'lambda': 0.2,
        'gamma': 0.1,
    }
    n_players, n_demand = 2, 2

    # Get the sequence
    configs = OptimizationConfig.sequential()

    # Verify we have the right sequence
    expected_names = [
        "baseline",
        "vectorize",
        "polyalgorithm",
        "cython",
        "sparse",
        "derivatives",
    ]
    names = [name for name, _ in configs]
    assert names == expected_names

    # Store results for comparison
    results = []

    for name, config in configs:
        print(f"\nTesting configuration: {name}")
        print(f"Config: {config}")

        # Create model
        model = ConfigurableModel(n_players, n_demand, params, config=config)

        # Check implementation type
        if config.vectorize or config.polyalgorithm or config.cython or config.sparse or config.derivatives:
            assert model._impl is not None, f"{name} should use Model implementation"
        else:
            assert model._impl is None, f"{name} should use ConfigurableModel implementation"

        # Compute value function
        v, dv = model.value_function(vf_max_iter=1000, vf_tol=1e-10)

        # Check results are valid
        assert v.shape == (n_players, model.K)
        assert np.all(np.isfinite(v))

        # Store for comparison
        results.append((name, v))

        # Print some diagnostics
        print(f"  Implementation: {model}")
        print(f"  Value function range: [{v.min():.6f}, {v.max():.6f}]")

    # Check consistency across optimizations
    # Results should be very close (within numerical tolerance)
    baseline_v = results[0][1]
    for i, (name, v) in enumerate(results[1:], 1):
        diff = np.max(np.abs(v - baseline_v))
        print(f"\nDifference between baseline and {name}: {diff:.2e}")
        assert diff < 1e-8, f"{name} differs too much from baseline"


def test_implementation_transitions():
    """Test that implementation transitions happen at the right points."""
    params = {
        'theta_ec': -1.0,
        'theta_rn': -0.5,
        'theta_d': 0.8,
        'lambda': 0.2,
        'gamma': 0.1,
    }
    n_players, n_demand = 2, 2

    # Test early optimizations use ConfigurableModel implementation
    early_configs = [
        (OptimizationConfig.baseline(), False),
        (OptimizationConfig(vectorize=False, sparse=False,
                            polyalgorithm=False, cython=False,
                            derivatives=False), False),
    ]

    for config, should_use_model in early_configs:
        model = ConfigurableModel(n_players, n_demand, params, config=config)
        assert (model._impl is not None) == should_use_model, f"Expected {should_use_model} for {config}"

    # Test later optimizations use Model implementation
    later_configs = [
        (OptimizationConfig(vectorize=True, sparse=False,
                            polyalgorithm=False, cython=False,
                            derivatives=False), True),
        (OptimizationConfig(vectorize=False, sparse=False,
                            polyalgorithm=True, cython=False,
                            derivatives=False), True),
        (OptimizationConfig(vectorize=False, sparse=False,
                            polyalgorithm=False, cython=True,
                            derivatives=False), True),
        (OptimizationConfig(vectorize=False, sparse=False,
                            polyalgorithm=False, cython=False,
                            derivatives=True), True),
    ]

    for config, should_use_model in later_configs:
        model = ConfigurableModel(n_players, n_demand, params, config=config)
        assert (model._impl is not None) == should_use_model, f"Expected {should_use_model} for {config}"
