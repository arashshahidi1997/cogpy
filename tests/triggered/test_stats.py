"""Tests for cogpy.triggered.stats — triggered statistics."""

import numpy as np
import pytest
import xarray as xr

from cogpy.triggered.stats import (
    triggered_average,
    triggered_std,
    triggered_median,
    triggered_snr,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def epochs_np(rng):
    """(50 events, 100 lags) with a deterministic signal + noise."""
    signal = np.sin(np.linspace(0, 2 * np.pi, 100))
    noise = rng.normal(0, 0.5, size=(50, 100))
    return signal[np.newaxis, :] + noise


@pytest.fixture
def epochs_xr(epochs_np):
    return xr.DataArray(
        epochs_np,
        dims=("event", "lag"),
        coords={
            "event": np.arange(50),
            "lag": np.linspace(-0.05, 0.05, 100),
        },
    )


# ---------------------------------------------------------------------------
# triggered_average
# ---------------------------------------------------------------------------


class TestTriggeredAverage:
    def test_ndarray(self, epochs_np):
        avg = triggered_average(epochs_np)
        assert avg.shape == (100,)
        # Signal is ~1 amplitude, noise averages out → avg should be close to sin
        expected = np.sin(np.linspace(0, 2 * np.pi, 100))
        np.testing.assert_allclose(avg, expected, atol=0.2)

    def test_xarray(self, epochs_xr):
        avg = triggered_average(epochs_xr, event_dim="event")
        assert isinstance(avg, xr.DataArray)
        assert "event" not in avg.dims
        assert "lag" in avg.dims

    def test_multichannel(self, rng):
        """(events, channels, lags) shape."""
        epochs = rng.normal(size=(30, 4, 50))
        avg = triggered_average(epochs)
        assert avg.shape == (4, 50)

    def test_single_event(self):
        epochs = np.ones((1, 20))
        avg = triggered_average(epochs)
        np.testing.assert_allclose(avg, 1.0)


# ---------------------------------------------------------------------------
# triggered_std
# ---------------------------------------------------------------------------


class TestTriggeredStd:
    def test_ndarray(self, epochs_np):
        std = triggered_std(epochs_np)
        assert std.shape == (100,)
        # Noise std ~0.5
        assert np.median(std) == pytest.approx(0.5, abs=0.15)

    def test_xarray(self, epochs_xr):
        std = triggered_std(epochs_xr, event_dim="event")
        assert isinstance(std, xr.DataArray)
        assert "event" not in std.dims

    def test_ddof(self, epochs_np):
        std0 = triggered_std(epochs_np, ddof=0)
        std1 = triggered_std(epochs_np, ddof=1)
        # ddof=1 should be slightly larger
        assert np.all(std1 >= std0)


# ---------------------------------------------------------------------------
# triggered_median
# ---------------------------------------------------------------------------


class TestTriggeredMedian:
    def test_ndarray(self, epochs_np):
        med = triggered_median(epochs_np)
        assert med.shape == (100,)

    def test_xarray(self, epochs_xr):
        med = triggered_median(epochs_xr, event_dim="event")
        assert isinstance(med, xr.DataArray)

    def test_outlier_robust(self, rng):
        """Median should be robust to a single extreme outlier."""
        epochs = np.zeros((11, 50))
        epochs[0, :] = 1000  # huge outlier
        med = triggered_median(epochs)
        np.testing.assert_allclose(med, 0.0)


# ---------------------------------------------------------------------------
# triggered_snr
# ---------------------------------------------------------------------------


class TestTriggeredSnr:
    def test_high_snr_consistent_signal(self):
        """Nearly identical epochs → very high SNR."""
        rng = np.random.default_rng(42)
        base = np.sin(np.linspace(0, 2 * np.pi, 50))
        epochs = base[np.newaxis, :] + rng.normal(0, 1e-6, size=(20, 50))
        snr = triggered_snr(epochs)
        # avg ≈ base (~1 amplitude), SE ≈ 1e-6/sqrt(20) → SNR >> 1
        assert np.median(np.abs(snr)) > 100

    def test_pure_noise_low_snr(self, rng):
        """Pure noise → SNR should be small."""
        epochs = rng.normal(size=(100, 50))
        snr = triggered_snr(epochs)
        assert np.median(np.abs(snr)) < 5  # not significantly different from 0

    def test_xarray(self, epochs_xr):
        snr = triggered_snr(epochs_xr, event_dim="event")
        assert isinstance(snr, xr.DataArray)
        assert "event" not in snr.dims

    def test_shape_multichannel(self, rng):
        epochs = rng.normal(size=(20, 4, 50))
        snr = triggered_snr(epochs)
        assert snr.shape == (4, 50)
