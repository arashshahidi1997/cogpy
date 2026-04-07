"""Tests for cogpy.regression.ols — OLS fit/predict/residual."""

import numpy as np
import pytest

from cogpy.regression.ols import ols_fit, ols_predict, ols_residual


@pytest.fixture
def rng():
    return np.random.default_rng(99)


class TestOlsFit:
    def test_perfect_fit(self):
        """Y = X @ beta exactly → recovered perfectly."""
        X = np.array([[1, 2], [1, 3], [1, 5], [1, 7]])
        beta_true = np.array([1.0, 0.5])
        Y = X @ beta_true
        beta = ols_fit(X, Y)
        np.testing.assert_allclose(beta, beta_true, atol=1e-10)

    def test_multichannel(self, rng):
        """Y is (n_time, n_channels) → beta is (n_predictors, n_channels)."""
        n, p, c = 100, 3, 5
        X = rng.normal(size=(n, p))
        beta_true = rng.normal(size=(p, c))
        Y = X @ beta_true + rng.normal(0, 0.01, size=(n, c))
        beta = ols_fit(X, Y)
        np.testing.assert_allclose(beta, beta_true, atol=0.1)

    def test_underdetermined(self):
        """More predictors than samples → still returns (minimum norm)."""
        X = np.eye(2, 5)
        Y = np.array([1.0, 2.0])
        beta = ols_fit(X, Y)
        assert beta.shape == (5,)
        # Verify prediction is exact
        np.testing.assert_allclose(X @ beta, Y, atol=1e-10)


class TestOlsPredict:
    def test_prediction(self):
        X = np.array([[1, 0], [0, 1], [1, 1]])
        beta = np.array([2.0, 3.0])
        y_hat = ols_predict(X, beta)
        np.testing.assert_allclose(y_hat, [2.0, 3.0, 5.0])


class TestOlsResidual:
    def test_residual_orthogonal_to_X(self, rng):
        """OLS residuals should be orthogonal to design matrix columns."""
        n, p = 50, 3
        X = rng.normal(size=(n, p))
        Y = rng.normal(size=n)
        beta = ols_fit(X, Y)
        resid = ols_residual(X, Y, beta)
        # X^T @ resid ≈ 0
        np.testing.assert_allclose(X.T @ resid, 0.0, atol=1e-10)

    def test_residual_reduces_variance(self, rng):
        """Residual variance <= original variance."""
        n = 200
        X = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
        Y = 3.0 + 0.5 * np.arange(n) + rng.normal(0, 1, size=n)
        beta = ols_fit(X, Y)
        resid = ols_residual(X, Y, beta)
        assert np.var(resid) < np.var(Y)

    def test_zero_residual_perfect_fit(self):
        X = np.eye(3)
        Y = np.array([1.0, 2.0, 3.0])
        beta = ols_fit(X, Y)
        resid = ols_residual(X, Y, beta)
        np.testing.assert_allclose(resid, 0.0, atol=1e-10)

    def test_multichannel_residual(self, rng):
        n, p, c = 80, 4, 3
        X = rng.normal(size=(n, p))
        beta_true = rng.normal(size=(p, c))
        Y = X @ beta_true
        beta = ols_fit(X, Y)
        resid = ols_residual(X, Y, beta)
        np.testing.assert_allclose(resid, 0.0, atol=1e-10)
