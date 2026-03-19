"""Tests for cogpy.regression.design — design matrix construction."""

import numpy as np
import pytest

from cogpy.regression.design import lagged_design_matrix, event_design_matrix


# ---------------------------------------------------------------------------
# lagged_design_matrix
# ---------------------------------------------------------------------------

class TestLaggedDesignMatrix:
    def test_shape_with_intercept(self):
        ref = np.arange(100.0)
        X = lagged_design_matrix(ref, range(5), intercept=True)
        assert X.shape == (100, 6)  # 1 intercept + 5 lags
        np.testing.assert_allclose(X[:, 0], 1.0)

    def test_shape_no_intercept(self):
        ref = np.arange(100.0)
        X = lagged_design_matrix(ref, range(5), intercept=False)
        assert X.shape == (100, 5)

    def test_lag_zero_is_original(self):
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X = lagged_design_matrix(ref, [0], intercept=False)
        np.testing.assert_array_equal(X[:, 0], ref)

    def test_lag_positive_shifts_right(self):
        """Lag k: predictor at time t = reference[t-k], so shifted right."""
        ref = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        X = lagged_design_matrix(ref, [2], intercept=False)
        # X[0:2, 0] should be zero (padded), X[2:, 0] = ref[0:3]
        np.testing.assert_array_equal(X[:2, 0], [0.0, 0.0])
        np.testing.assert_array_equal(X[2:, 0], [10.0, 20.0, 30.0])

    def test_lag_negative_shifts_left(self):
        """Negative lag: predictor at time t = reference[t+|k|]."""
        ref = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        X = lagged_design_matrix(ref, [-1], intercept=False)
        # X[:4, 0] = ref[1:5], X[4, 0] = 0 (padded)
        np.testing.assert_array_equal(X[:4, 0], [20.0, 30.0, 40.0, 50.0])
        assert X[4, 0] == 0.0

    def test_impulse_response(self):
        """Impulse in reference → one non-zero per column at offset = lag."""
        n = 50
        ref = np.zeros(n)
        ref[10] = 1.0
        lags = [0, 1, 2, 3]
        X = lagged_design_matrix(ref, lags, intercept=False)
        for i, k in enumerate(lags):
            col = X[:, i]
            nonzero = np.where(col != 0)[0]
            assert len(nonzero) == 1
            assert nonzero[0] == 10 + k  # impulse appears at 10+k

    def test_regression_recovers_weights(self):
        """Full round-trip: build X, generate y, recover coefficients."""
        rng = np.random.default_rng(42)
        n = 500
        ref = rng.normal(size=n)
        true_weights = np.array([1.0, -0.5, 0.3])
        X = lagged_design_matrix(ref, range(3), intercept=False)
        y = X @ true_weights + rng.normal(0, 0.01, size=n)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        np.testing.assert_allclose(beta, true_weights, atol=0.05)


# ---------------------------------------------------------------------------
# event_design_matrix
# ---------------------------------------------------------------------------

class TestEventDesignMatrix:
    def test_shape(self):
        tmpl = np.ones(10)
        events = np.array([5, 20, 50])
        X = event_design_matrix(100, events, tmpl, intercept=True)
        assert X.shape == (100, 4)  # 1 intercept + 3 events

    def test_template_placement(self):
        tmpl = np.array([1.0, 2.0, 3.0])
        events = np.array([10])
        X = event_design_matrix(50, events, tmpl, intercept=False)
        np.testing.assert_array_equal(X[10:13, 0], [1.0, 2.0, 3.0])
        # Everything else is zero
        assert X[:10, 0].sum() == 0.0
        assert X[13:, 0].sum() == 0.0

    def test_boundary_end(self):
        """Event near end: template is clipped."""
        tmpl = np.ones(10)
        events = np.array([95])
        X = event_design_matrix(100, events, tmpl, intercept=False)
        assert X[95:100, 0].sum() == 5.0  # only 5 samples fit
        assert X[:95, 0].sum() == 0.0

    def test_boundary_start(self):
        """Event before start (negative index): template is clipped."""
        tmpl = np.ones(10)
        events = np.array([-3])
        X = event_design_matrix(100, events, tmpl, intercept=False)
        # Samples 0..6 should have the tail of the template
        assert X[0:7, 0].sum() == 7.0
        assert X[7:, 0].sum() == 0.0

    def test_no_overlap(self):
        """Event completely outside range."""
        tmpl = np.ones(5)
        events = np.array([-10, 200])
        X = event_design_matrix(100, events, tmpl, intercept=False)
        assert X.sum() == 0.0

    def test_regression_round_trip(self):
        """Build event design, generate signal, recover scaling."""
        rng = np.random.default_rng(7)
        n_time = 500
        tmpl = np.exp(-np.linspace(0, 3, 15))
        events = np.array([50, 150, 250, 350])
        true_scales = np.array([1.0, 2.0, 0.5, 1.5])

        X = event_design_matrix(n_time, events, tmpl, intercept=True)
        beta_true = np.concatenate([[0.0], true_scales])  # intercept=0
        y = X @ beta_true + rng.normal(0, 0.001, size=n_time)

        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        np.testing.assert_allclose(beta[1:], true_scales, atol=0.05)
