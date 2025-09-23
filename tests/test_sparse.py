#!/usr/bin/env python3
"""
Unit tests for sparse matrix utility functions.
"""

import pytest
import numpy as np
from scipy.sparse import csr_matrix, eye, diags
from scipy.linalg import expm
from sparse import vexpm, vexpm_deriv, spsolve_multiple_rhs


def test_vexpm_identity():
    """Test vexpm with identity matrix."""
    n = 5
    Q = eye(n, format='csr')
    v = np.ones(n)
    Delta = 1.0

    # exp(I*Delta)*v = exp(Delta)*v
    result = vexpm(Q, Delta, v)
    expected = np.exp(Delta) * v

    np.testing.assert_allclose(result, expected, rtol=1e-8)


def test_vexpm_zero():
    """Test vexpm with zero matrix."""
    n = 5
    Q = csr_matrix((n, n))
    v = np.random.randn(n)
    Delta = 1.0

    # exp(0*Delta)*v = I*v = v
    result = vexpm(Q, Delta, v)
    expected = v.copy()

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_vexpm_diagonal():
    """Test vexpm with diagonal matrix."""
    n = 4
    diag_values = np.array([-1.0, -2.0, -3.0, -4.0])
    Q = diags(diag_values, format='csr')
    v = np.ones(n)
    Delta = 0.5

    # For diagonal matrix, exp(Q*Delta)*v = diag(exp(diag_values*Delta))*v
    result = vexpm(Q, Delta, v)
    expected = np.exp(diag_values * Delta) * v

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_vexpm_2x2_intensity_matrix():
    """Test vexpm with a simple 2x2 intensity matrix."""
    lambda_rate = 2.0
    mu_rate = 3.0
    Q = csr_matrix([[-lambda_rate, lambda_rate],
                    [mu_rate, -mu_rate]])
    v = np.array([0.6, 0.4])
    Delta = 0.1

    # Compare with scipy's dense matrix exponential
    Q_dense = Q.toarray()
    exp_Q_dense = expm(Q_dense * Delta)
    expected = exp_Q_dense @ v

    result = vexpm(Q, Delta, v)

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_vexpm_3x3_intensity_matrix():
    """Test vexpm with a 3x3 intensity matrix."""
    Q = csr_matrix([[-3.0, 2.0, 1.0],
                    [1.0, -4.0, 3.0],
                    [2.0, 1.0, -3.0]])
    v = np.array([0.5, 0.3, 0.2])
    Delta = 0.2

    # Compare with scipy's dense matrix exponential
    Q_dense = Q.toarray()
    exp_Q_dense = expm(Q_dense * Delta)
    expected = exp_Q_dense @ v

    result = vexpm(Q, Delta, v)

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_vexpm_large_sparse_matrix():
    """Test vexpm with a larger sparse matrix."""
    n = 100
    # Create a tridiagonal intensity matrix
    main_diag = -2 * np.ones(n)
    upper_diag = np.ones(n-1)
    lower_diag = np.ones(n-1)

    Q = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)  # Normalize
    Delta = 0.01

    result = vexpm(Q, Delta, v)

    # Check basic properties
    assert result.shape == v.shape
    assert np.all(np.isfinite(result))
    # For small Delta, result should be close to v
    assert np.linalg.norm(result - v) < 0.1

    # Compare with scipy's dense matrix exponential
    Q_dense = Q.toarray()
    exp_Q_dense = expm(Q_dense * Delta)
    expected = exp_Q_dense @ v

    np.testing.assert_allclose(result, expected, rtol=1e-10)


@pytest.mark.parametrize("Delta", [0.01, 0.1, 1.0, 5.0])
def test_vexpm_various_delta(Delta):
    """Test vexpm for various time intervals."""
    n = 10
    # Create a valid continuous-time Markov chain generator
    np.random.seed(42)
    # Start with random non-negative off-diagonal elements
    off_diag = np.random.exponential(1.0, size=(n, n))
    np.fill_diagonal(off_diag, 0)
    # Set diagonal to make row sums zero
    Q_dense = off_diag - np.diag(off_diag.sum(axis=1))
    Q = csr_matrix(Q_dense)
    v = np.random.rand(n)
    v = v / v.sum()  # Normalize to probability distribution

    # Compare with scipy
    exp_Q_dense = expm(Q_dense * Delta)
    expected = exp_Q_dense @ v
    result = vexpm(Q, Delta, v)

    np.testing.assert_allclose(result, expected, rtol=1e-11)


def test_vexpm_error_handling():
    """Test error handling in vexpm."""
    # Non-square matrix
    Q = csr_matrix((3, 4))
    v = np.ones(4)
    with pytest.raises(ValueError, match="Q is not square"):
        vexpm(Q, 1.0, v)

    # Incompatible dimensions
    Q = csr_matrix((3, 3))
    v = np.ones(4)
    with pytest.raises(ValueError, match="v has 4 rows, but Q has 3 columns"):
        vexpm(Q, 1.0, v)


def test_vexpm_deriv_identity_matrix():
    """Test vexpm_deriv with identity matrix."""
    n = 3
    Q = eye(n, format='csr')
    dQ = {'alpha': csr_matrix((n, n))}  # Zero derivative
    v = np.ones(n)
    Delta = 1.0

    mu, dmu = vexpm_deriv(Q, dQ, Delta, v)

    # Check base result
    expected_mu = np.exp(Delta) * v
    np.testing.assert_allclose(mu, expected_mu, rtol=1e-8)

    # Derivative should be zero
    np.testing.assert_allclose(dmu['alpha'], np.zeros(n), rtol=1e-8)


def test_vexpm_deriv_diagonal_matrix():
    """Test vexpm_deriv with diagonal matrix and its derivative."""
    n = 3
    diag_values = np.array([-1.0, -2.0, -3.0])
    Q = diags(diag_values, format='csr')

    # Derivative with respect to first diagonal element
    dQ_alpha = csr_matrix((n, n))
    dQ_alpha[0, 0] = 1.0
    dQ = {'alpha': dQ_alpha}

    v = np.array([1.0, 0.0, 0.0])
    Delta = 0.5

    mu, dmu = vexpm_deriv(Q, dQ, Delta, v)

    # Analytical result for diagonal case
    expected_mu = np.exp(diag_values * Delta) * v
    # Derivative: d/d(alpha) exp((-1+alpha)*Delta) = Delta * exp((-1+alpha)*Delta)
    expected_dmu_alpha = np.array([Delta * np.exp(diag_values[0] * Delta), 0.0, 0.0])

    np.testing.assert_allclose(mu, expected_mu, rtol=1e-12)
    np.testing.assert_allclose(dmu['alpha'], expected_dmu_alpha, rtol=1e-12)


def test_vexpm_deriv_finite_differences():
    """Test vexpm_deriv using finite differences."""
    n = 4
    # Create a test intensity matrix
    Q_dense = np.array([[-3.0, 1.0, 1.0, 1.0],
                        [2.0, -5.0, 2.0, 1.0],
                        [1.0, 3.0, -6.0, 2.0],
                        [0.5, 0.5, 1.0, -2.0]])
    Q = csr_matrix(Q_dense)

    # Create derivative matrix (derivative with respect to Q[0,1])
    dQ_param = csr_matrix((n, n))
    dQ_param[0, 1] = 1.0
    dQ_param[0, 0] = -1.0  # Maintain row sum = 0
    dQ = {'param': dQ_param}

    v = np.array([0.25, 0.25, 0.25, 0.25])
    Delta = 0.1

    # Analytical derivative
    mu, dmu = vexpm_deriv(Q, dQ, Delta, v)

    # Finite difference approximation
    h = 1e-8
    Q_plus = Q.copy()
    Q_plus[0, 1] += h
    Q_plus[0, 0] -= h  # Maintain row sum
    mu_plus = vexpm(Q_plus, Delta, v)

    Q_minus = Q.copy()
    Q_minus[0, 1] -= h
    Q_minus[0, 0] += h  # Maintain row sum
    mu_minus = vexpm(Q_minus, Delta, v)

    dmu_numerical = (mu_plus - mu_minus) / (2 * h)

    np.testing.assert_allclose(dmu['param'], dmu_numerical, rtol=1e-6, atol=1e-10)


def test_vexpm_deriv_multiple_parameters():
    """Test vexpm_deriv with multiple parameter derivatives."""
    n = 3
    # Create a valid continuous-time Markov chain generator
    Q = csr_matrix([[-2.0, 1.0, 1.0],
                    [1.5, -3.0, 1.5],
                    [0.5, 2.0, -2.5]])

    # Verify it's a valid generator
    Q_dense = Q.toarray()
    assert np.allclose(Q_dense.sum(axis=1), 0)  # Row sums = 0
    assert np.all(Q_dense[~np.eye(n, dtype=bool)] >= 0)  # Off-diagonal >= 0

    # Multiple derivatives that maintain generator structure
    dQ1 = csr_matrix((n, n))
    dQ1[0, 1] = 1.0
    dQ1[0, 0] = -1.0

    dQ2 = csr_matrix((n, n))
    dQ2[1, 2] = 1.0
    dQ2[1, 1] = -1.0

    dQ3 = csr_matrix((n, n))
    dQ3[2, 0] = 1.0
    dQ3[2, 2] = -1.0

    dQ = {'param1': dQ1, 'param2': dQ2, 'param3': dQ3}

    # Use probability distribution
    v = np.array([0.5, 0.3, 0.2])
    assert np.abs(v.sum() - 1.0) < 1e-10
    Delta = 0.2

    mu, dmu = vexpm_deriv(Q, dQ, Delta, v)

    # Check all derivatives exist
    assert 'param1' in dmu
    assert 'param2' in dmu
    assert 'param3' in dmu

    # Check that all derivatives have correct shapes and are finite
    for param in dmu:
        assert dmu[param].shape == mu.shape
        assert np.all(np.isfinite(dmu[param]))


def test_vexpm_deriv_error_handling():
    """Test error handling in vexpm_deriv."""
    # Non-square Q
    Q = csr_matrix((3, 4))
    dQ = {'alpha': csr_matrix((3, 4))}
    v = np.ones(4)
    with pytest.raises(ValueError, match="Q is not square"):
        vexpm_deriv(Q, dQ, 1.0, v)

    # Incompatible v dimension
    Q = csr_matrix((3, 3))
    dQ = {'alpha': csr_matrix((3, 3))}
    v = np.ones(4)
    with pytest.raises(ValueError, match="v has 4 rows, but Q has 3 columns"):
        vexpm_deriv(Q, dQ, 1.0, v)

    # Mismatched dQ shape
    Q = csr_matrix((3, 3))
    dQ = {'alpha': csr_matrix((4, 4))}
    v = np.ones(3)
    with pytest.raises(ValueError, match="dQ\\[alpha\\] has shape"):
        vexpm_deriv(Q, dQ, 1.0, v)


def test_spsolve_identity_matrix():
    """Test solving with identity matrix."""
    n = 5
    A = eye(n, format='csc')

    # Multiple RHS vectors
    rhs_all = {
        'rhs1': np.ones(n),
        'rhs2': np.arange(n, dtype=float),
        'rhs3': np.random.randn(n)
    }
    solutions = spsolve_multiple_rhs(A, rhs_all)

    # For identity matrix, solution equals RHS
    for key in rhs_all:
        np.testing.assert_allclose(solutions[key], rhs_all[key], rtol=1e-10)


def test_spsolve_diagonal_matrix():
    """Test solving with diagonal matrix."""
    n = 4
    diag_values = np.array([2.0, 3.0, 4.0, 5.0])
    A = diags(diag_values, format='csc')

    rhs_all = {
        'rhs1': np.ones(n),
        'rhs2': np.array([10.0, 15.0, 20.0, 25.0])
    }
    solutions = spsolve_multiple_rhs(A, rhs_all)

    # Check solutions
    expected1 = 1.0 / diag_values
    expected2 = np.array([5.0, 5.0, 5.0, 5.0])

    np.testing.assert_allclose(solutions['rhs1'], expected1, rtol=1e-10)
    np.testing.assert_allclose(solutions['rhs2'], expected2, rtol=1e-10)


def test_spsolve_tridiagonal_system():
    """Test solving tridiagonal system."""
    n = 10
    # Create symmetric positive definite tridiagonal matrix
    main_diag = 4.0 * np.ones(n)
    off_diag = -1.0 * np.ones(n-1)
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csc')

    # Multiple RHS
    rhs_all = {}
    for i in range(3):
        rhs_all[f'rhs{i}'] = np.zeros(n)
        rhs_all[f'rhs{i}'][i] = 1.0  # Unit vectors
    solutions = spsolve_multiple_rhs(A, rhs_all)

    # Verify solutions
    for key, rhs in rhs_all.items():
        # Check A * x = rhs
        residual = A.dot(solutions[key]) - rhs
        assert np.linalg.norm(residual) < 1e-10


def test_spsolve_general_sparse_matrix():
    """Test solving with general sparse matrix."""
    n = 6
    # Create a sparse matrix with known inverse
    data = [5.0, 1.0, 1.0, 4.0, 1.0, 1.0, 6.0, 1.0, 3.0, 1.0, 5.0, 1.0, 4.0]
    row = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]
    col = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 5, 5]
    A = csr_matrix((data, (row, col)), shape=(n, n)).tocsc()

    # Random RHS vectors
    np.random.seed(42)
    rhs_all = {
        f'param{i}': np.random.randn(n) for i in range(5)
    }

    solutions = spsolve_multiple_rhs(A, rhs_all)

    # Verify all solutions
    for key, rhs in rhs_all.items():
        residual = A.dot(solutions[key]) - rhs
        assert np.linalg.norm(residual) < 1e-10


def test_spsolve_csr_format_conversion():
    """Test that CSR matrices are converted to CSC."""
    n = 5
    A_csr = diags([2.0] * n, format='csr')  # Create in CSR format

    # Should handle CSR format with warning
    rhs_all = {'rhs': np.ones(n)}
    solutions = spsolve_multiple_rhs(A_csr, rhs_all)

    # Check solution
    expected = 0.5 * np.ones(n)
    np.testing.assert_allclose(solutions['rhs'], expected, rtol=1e-10)


def test_spsolve_2d_rhs_handling():
    """Test that 2D RHS arrays are properly flattened."""
    n = 4
    A = diags([2.0] * n, format='csc')

    # Create 2D RHS (column vector)
    rhs_2d = np.ones((n, 1))
    rhs_all = {'rhs': rhs_2d}
    solutions = spsolve_multiple_rhs(A, rhs_all)

    # Should still work
    expected = 0.5 * np.ones(n)
    np.testing.assert_allclose(solutions['rhs'], expected, rtol=1e-10)


def test_spsolve_numerical_stability():
    """Test solver with ill-conditioned matrix."""
    n = 2
    # Create an ill-conditioned matrix
    A = csr_matrix([[1e10, 1.0],
                    [-1.0, 1e-10]])

    # Check the condition number
    cond_num = np.linalg.cond(A.toarray())
    assert cond_num > 1e19

    rhs_all = {'rhs': np.ones(n)}
    solutions = spsolve_multiple_rhs(A, rhs_all)

    # Should still get reasonable solution
    assert solutions  # Non-empty result
    assert 'rhs' in solutions
    assert np.all(np.isfinite(solutions['rhs']))

    # Residual should be small
    residual = A.dot(solutions['rhs']) - rhs_all['rhs']
    assert np.linalg.norm(residual) < 1e-10


def test_spsolve_empty_rhs():
    """Test solver with empty RHS dictionary."""
    n = 3
    A = eye(n, format='csc')
    rhs_all = {}

    solutions = spsolve_multiple_rhs(A, rhs_all)

    # Should return empty dictionary
    assert solutions == {}


def test_exp_zero_delta():
    """Test exp(Q*0) = I."""
    n = 5
    Q = csr_matrix(np.random.randn(n, n))
    v = np.random.randn(n)

    result = vexpm(Q, 0.0, v)
    np.testing.assert_allclose(result, v, rtol=1e-10)


def test_exp_scaling_property():
    """Test exp(Q*2t) = exp(Q*t) * exp(Q*t)."""
    n = 4
    # Create a stable intensity matrix
    A = np.random.randn(n, n)
    Q_dense = A - np.diag(A.sum(axis=1))
    Q = csr_matrix(Q_dense)
    v = np.random.randn(n)
    t = 0.1

    # exp(Q*2t)*v
    result_2t = vexpm(Q, 2*t, v)

    # exp(Q*t) * exp(Q*t) * v = exp(Q*t) * (exp(Q*t) * v)
    result_t = vexpm(Q, t, v)
    result_t_squared = vexpm(Q, t, result_t)

    np.testing.assert_allclose(result_2t, result_t_squared, rtol=1e-8)


def test_derivative_consistency():
    """Test that derivatives are consistent with finite differences."""
    Q = csr_matrix([[-2.0, 1.5, 0.5],
                    [1.0, -3.0, 2.0],
                    [0.5, 0.5, -1.0]])

    # Parameter that scales entire matrix
    dQ = {'scale': Q.copy()}
    v = np.array([0.4, 0.4, 0.2])
    Delta = 0.1

    # Analytical derivative
    mu, dmu = vexpm_deriv(Q, dQ, Delta, v)

    # Finite difference
    h = 1e-8
    mu_plus = vexpm(Q * (1 + h), Delta, v)
    mu_minus = vexpm(Q * (1 - h), Delta, v)
    dmu_numerical = (mu_plus - mu_minus) / (2 * h)

    np.testing.assert_allclose(dmu['scale'], dmu_numerical, rtol=1e-6)
