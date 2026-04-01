"""Tests for cogpy.measures.temporal — temporal signal measures."""

import numpy as np
import pytest

from cogpy.measures.temporal import temporal_stability


# ---------------------------------------------------------------------------
# temporal_stability
# ---------------------------------------------------------------------------


class TestTemporalStability:
    def test_constant_signal(self):
        """Constant values → CV near 0."""
        arr = np.full(100, 5.0)
        cv = temporal_stability(arr)
        assert cv == pytest.approx(0.0, abs=1e-10)

    def test_high_variability(self):
        """Large fluctuations → high CV."""
        arr = np.array([1.0, 100.0, 1.0, 100.0, 1.0])
        cv = temporal_stability(arr)
        assert cv > 0.5

    def test_batch_shape(self):
        """Batch dims preserved: (3, 4, time) → (3, 4)."""
        arr = np.ones((3, 4, 50))
        cv = temporal_stability(arr)
        assert cv.shape == (3, 4)

    def test_custom_axis(self):
        """Reduce over axis=0."""
        arr = np.ones((10, 5))
        cv = temporal_stability(arr, axis=0)
        assert cv.shape == (5,)

    def test_nan_handling(self):
        """NaN values are ignored."""
        arr = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        cv = temporal_stability(arr)
        assert np.isfinite(cv)

    def test_zero_mean(self):
        """Near-zero mean → large CV (not inf due to EPS)."""
        arr = np.array([-1.0, 1.0, -1.0, 1.0])
        cv = temporal_stability(arr)
        assert np.isfinite(cv)
        assert cv > 1.0  # std >> |mean|
