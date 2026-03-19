"""Tests for cogpy.events.match — event matching & lag estimation."""

import numpy as np
import pytest

from cogpy.events.match import (
    match_nearest,
    match_nearest_symmetric,
    event_lag_histogram,
    estimate_lag,
    estimate_drift,
)


# ---------------------------------------------------------------------------
# match_nearest
# ---------------------------------------------------------------------------

class TestMatchNearest:
    def test_perfect_alignment(self):
        """Identical event trains → zero lags, all matched."""
        t = np.array([1.0, 2.0, 3.0, 4.0])
        idx_a, idx_b, lags = match_nearest(t, t, max_lag=0.1)
        np.testing.assert_array_equal(idx_a, [0, 1, 2, 3])
        np.testing.assert_array_equal(idx_b, [0, 1, 2, 3])
        np.testing.assert_allclose(lags, 0.0, atol=1e-15)

    def test_constant_lag(self):
        """B is shifted by a constant lag from A."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lag = 0.05
        b = a + lag
        idx_a, idx_b, lags = match_nearest(a, b, max_lag=0.1)
        assert len(idx_a) == 5
        np.testing.assert_allclose(lags, lag, atol=1e-12)

    def test_max_lag_excludes_distant(self):
        """Events outside max_lag are not matched."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.01, 2.5, 3.01])  # middle event 0.5 away
        idx_a, idx_b, lags = match_nearest(a, b, max_lag=0.1)
        assert len(idx_a) == 2
        np.testing.assert_array_equal(idx_a, [0, 2])

    def test_many_to_one(self):
        """Multiple A events can match the same B event."""
        a = np.array([1.0, 1.01, 1.02])
        b = np.array([1.015])
        idx_a, idx_b, lags = match_nearest(a, b, max_lag=0.05)
        assert len(idx_a) == 3
        np.testing.assert_array_equal(idx_b, [0, 0, 0])

    def test_empty_inputs(self):
        """Empty arrays return empty results."""
        empty = np.array([])
        b = np.array([1.0, 2.0])
        idx_a, idx_b, lags = match_nearest(empty, b, max_lag=1.0)
        assert idx_a.size == 0
        idx_a, idx_b, lags = match_nearest(b, empty, max_lag=1.0)
        assert idx_a.size == 0

    def test_negative_max_lag_raises(self):
        with pytest.raises(ValueError, match="max_lag must be positive"):
            match_nearest(np.array([1.0]), np.array([1.0]), max_lag=-1.0)

    def test_signed_lags(self):
        """Lag sign: positive means B follows A."""
        a = np.array([1.0])
        b = np.array([0.95])  # B before A
        _, _, lags = match_nearest(a, b, max_lag=0.1)
        assert lags[0] < 0  # B is earlier → negative lag

    def test_single_element(self):
        """Single-element arrays."""
        a = np.array([5.0])
        b = np.array([5.002])
        idx_a, idx_b, lags = match_nearest(a, b, max_lag=0.01)
        assert len(idx_a) == 1
        np.testing.assert_allclose(lags[0], 0.002, atol=1e-12)

    def test_large_scale(self):
        """Thousands of events — regression / perf sanity check."""
        rng = np.random.default_rng(42)
        a = np.sort(rng.uniform(0, 100, size=5000))
        b = np.sort(a + rng.normal(0, 0.001, size=5000))
        idx_a, idx_b, lags = match_nearest(a, b, max_lag=0.01)
        assert idx_a.size > 4000  # most should match
        assert np.all(np.abs(lags) <= 0.01)


# ---------------------------------------------------------------------------
# match_nearest_symmetric
# ---------------------------------------------------------------------------

class TestMatchNearestSymmetric:
    def test_bijective(self):
        """Each B event appears at most once in output."""
        a = np.array([1.0, 1.01, 1.02, 2.0])
        b = np.array([1.015, 2.01])
        idx_a, idx_b, lags = match_nearest_symmetric(a, b, max_lag=0.05)
        assert len(set(idx_b.tolist())) == len(idx_b)  # unique B indices

    def test_keeps_closest(self):
        """When multiple A map to same B, keeps the closest."""
        a = np.array([1.0, 1.01, 1.005])
        a.sort()
        b = np.array([1.006])
        idx_a, idx_b, lags = match_nearest_symmetric(a, b, max_lag=0.01)
        assert len(idx_a) == 1
        # 1.005 is closest to 1.006
        assert a[idx_a[0]] == pytest.approx(1.005, abs=1e-10)

    def test_perfect_alignment(self):
        t = np.arange(10.0)
        idx_a, idx_b, lags = match_nearest_symmetric(t, t, max_lag=0.5)
        assert len(idx_a) == 10
        np.testing.assert_allclose(lags, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# event_lag_histogram
# ---------------------------------------------------------------------------

class TestEventLagHistogram:
    def test_symmetric_identical(self):
        """Identical trains → all counts at lag 0."""
        t = np.arange(0, 10, 0.1)
        counts, edges = event_lag_histogram(t, t, max_lag=0.5, bin_width=0.05)
        # Peak should be at center (zero lag)
        center_bin = len(counts) // 2
        assert counts[center_bin] == counts.max()

    def test_empty(self):
        counts, edges = event_lag_histogram(
            np.array([]), np.array([1.0]), max_lag=1.0, bin_width=0.1
        )
        assert counts.sum() == 0

    def test_known_lag(self):
        """Constant offset → histogram peak at that lag."""
        a = np.arange(0, 100, 2.0)  # spacing >> max_lag to avoid cross-pairs
        b = a + 0.1
        counts, edges = event_lag_histogram(a, b, max_lag=0.5, bin_width=0.05)
        centers = 0.5 * (edges[:-1] + edges[1:])
        peak_lag = centers[np.argmax(counts)]
        assert abs(peak_lag - 0.1) < 0.05  # within one bin width


# ---------------------------------------------------------------------------
# estimate_lag
# ---------------------------------------------------------------------------

class TestEstimateLag:
    def test_constant_offset_median(self):
        a = np.arange(0, 100, 1.0)
        b = a + 0.05
        lag = estimate_lag(a, b, max_lag=0.1, method="median")
        assert lag == pytest.approx(0.05, abs=1e-10)

    def test_constant_offset_mean(self):
        a = np.arange(0, 100, 1.0)
        b = a + 0.03
        lag = estimate_lag(a, b, max_lag=0.1, method="mean")
        assert lag == pytest.approx(0.03, abs=1e-10)

    def test_constant_offset_mode(self):
        a = np.arange(0, 100, 1.0)
        b = a + 0.02
        lag = estimate_lag(a, b, max_lag=0.1, method="mode")
        assert abs(lag - 0.02) < 0.01  # mode has bin-width resolution

    def test_no_matches_returns_nan(self):
        a = np.array([1.0])
        b = np.array([100.0])
        lag = estimate_lag(a, b, max_lag=0.01)
        assert np.isnan(lag)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            estimate_lag(np.array([1.0]), np.array([1.0]), max_lag=1.0, method="bad")


# ---------------------------------------------------------------------------
# estimate_drift
# ---------------------------------------------------------------------------

class TestEstimateDrift:
    def test_linear_drift(self):
        """B drifts linearly from A → polyfit recovers slope."""
        a = np.arange(0, 100, 1.0)
        slope = 0.001  # 1 ms per second of drift
        b = a + 0.01 + slope * a
        coeffs = estimate_drift(a, b, max_lag=0.2, degree=1)
        assert len(coeffs) == 2
        assert coeffs[0] == pytest.approx(slope, abs=1e-4)
        assert coeffs[1] == pytest.approx(0.01, abs=0.02)

    def test_constant_drift_degree0(self):
        a = np.arange(0, 50, 1.0)
        b = a + 0.05
        coeffs = estimate_drift(a, b, max_lag=0.1, degree=0)
        assert len(coeffs) == 1
        assert coeffs[0] == pytest.approx(0.05, abs=1e-10)

    def test_insufficient_matches(self):
        a = np.array([1.0])
        b = np.array([100.0])
        coeffs = estimate_drift(a, b, max_lag=0.01, degree=1)
        assert np.all(np.isnan(coeffs))
