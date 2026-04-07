"""Tests for cogpy.core.triggered.template — template estimation & subtraction."""

import numpy as np
import pytest
import xarray as xr

from cogpy.core.triggered.template import (
    estimate_template,
    fit_scaling,
    subtract_template,
)


@pytest.fixture
def rng():
    return np.random.default_rng(1)


@pytest.fixture
def template_waveform():
    """A simple 20-sample template pulse."""
    t = np.linspace(0, 1, 20)
    return np.exp(-5 * t) * np.sin(2 * np.pi * 3 * t)


# ---------------------------------------------------------------------------
# estimate_template
# ---------------------------------------------------------------------------

class TestEstimateTemplate:
    def test_mean_ndarray(self, rng, template_waveform):
        noise = rng.normal(0, 0.1, size=(40, 20))
        epochs = template_waveform + noise
        tmpl = estimate_template(epochs, method="mean")
        np.testing.assert_allclose(tmpl, template_waveform, atol=0.05)

    def test_median_ndarray(self, rng, template_waveform):
        epochs = np.tile(template_waveform, (40, 1))
        epochs[0, :] = 1000  # outlier
        tmpl = estimate_template(epochs, method="median")
        np.testing.assert_allclose(tmpl, template_waveform, atol=1e-10)

    def test_trimmean_ndarray(self, rng, template_waveform):
        noise = rng.normal(0, 0.1, size=(40, 20))
        epochs = template_waveform + noise
        tmpl = estimate_template(epochs, method="trimmean")
        np.testing.assert_allclose(tmpl, template_waveform, atol=0.05)

    def test_mean_xarray(self, rng, template_waveform):
        noise = rng.normal(0, 0.1, size=(40, 20))
        epochs = xr.DataArray(
            template_waveform + noise,
            dims=("event", "lag"),
        )
        tmpl = estimate_template(epochs, method="mean")
        assert isinstance(tmpl, xr.DataArray)
        assert "event" not in tmpl.dims
        np.testing.assert_allclose(tmpl.values, template_waveform, atol=0.05)

    def test_trimmean_xarray(self, rng, template_waveform):
        noise = rng.normal(0, 0.1, size=(40, 20))
        epochs = xr.DataArray(
            template_waveform + noise,
            dims=("event", "lag"),
        )
        tmpl = estimate_template(epochs, method="trimmean")
        assert isinstance(tmpl, xr.DataArray)
        np.testing.assert_allclose(tmpl.values, template_waveform, atol=0.05)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            estimate_template(np.ones((5, 10)), method="bad")

    def test_multichannel(self, rng):
        """(events, channels, lags)."""
        epochs = rng.normal(size=(20, 3, 50))
        tmpl = estimate_template(epochs, method="mean")
        assert tmpl.shape == (3, 50)


# ---------------------------------------------------------------------------
# fit_scaling
# ---------------------------------------------------------------------------

class TestFitScaling:
    def test_uniform_scaling(self, template_waveform):
        """Epochs that are exactly 2x template → alpha = 2."""
        epochs = np.tile(2.0 * template_waveform, (10, 1))
        alpha = fit_scaling(epochs, template_waveform)
        np.testing.assert_allclose(alpha, 2.0, atol=1e-10)

    def test_varying_scaling(self, template_waveform):
        """Known per-event scaling coefficients."""
        scales = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        epochs = scales[:, np.newaxis] * template_waveform[np.newaxis, :]
        alpha = fit_scaling(epochs, template_waveform)
        np.testing.assert_allclose(alpha, scales, atol=1e-10)

    def test_noisy_scaling(self, rng, template_waveform):
        """With noise, recovered scaling should be close."""
        true_alpha = 1.5
        noise = rng.normal(0, 0.01, size=(30, 20))
        epochs = true_alpha * template_waveform + noise
        alpha = fit_scaling(epochs, template_waveform)
        np.testing.assert_allclose(alpha, true_alpha, atol=0.05)

    def test_zero_template(self):
        """Zero template → alpha = 0 (no crash)."""
        alpha = fit_scaling(np.ones((5, 10)), np.zeros(10))
        np.testing.assert_allclose(alpha, 0.0)

    def test_multichannel(self, rng):
        """(events, channels, lags) epochs."""
        tmpl = rng.normal(size=(3, 20))
        epochs = 2.0 * np.tile(tmpl, (5, 1, 1))
        alpha = fit_scaling(epochs, tmpl)
        np.testing.assert_allclose(alpha, 2.0, atol=1e-10)


# ---------------------------------------------------------------------------
# subtract_template
# ---------------------------------------------------------------------------

class TestSubtractTemplate:
    def test_perfect_subtraction(self, template_waveform):
        """Place template at known locations, subtract, verify clean."""
        n_time = 200
        signal = np.zeros(n_time)
        events = np.array([10, 50, 100, 150])
        for s0 in events:
            signal[s0 : s0 + len(template_waveform)] += template_waveform

        cleaned = subtract_template(signal, events, template_waveform)
        np.testing.assert_allclose(cleaned, 0.0, atol=1e-12)

    def test_with_scaling(self, template_waveform):
        n_time = 200
        signal = np.zeros(n_time)
        events = np.array([10, 60])
        scales = np.array([2.0, 0.5])
        for s0, sc in zip(events, scales):
            signal[s0 : s0 + len(template_waveform)] += sc * template_waveform

        cleaned = subtract_template(signal, events, template_waveform, scaling=scales)
        np.testing.assert_allclose(cleaned, 0.0, atol=1e-12)

    def test_out_of_bounds_skipped(self, template_waveform):
        """Events near boundaries are silently skipped."""
        signal = np.zeros(100)
        events = np.array([-5, 50, 95])  # first and last are OOB
        signal[50 : 50 + len(template_waveform)] += template_waveform
        cleaned = subtract_template(signal, events, template_waveform)
        # Only event at 50 should be subtracted
        np.testing.assert_allclose(cleaned[50 : 50 + 20], 0.0, atol=1e-12)

    def test_xarray_preserves_metadata(self, template_waveform):
        n_time = 100
        sig = xr.DataArray(
            np.zeros(n_time),
            dims=("time",),
            coords={"time": np.arange(n_time) / 1000.0},
            attrs={"fs": 1000},
            name="lfp",
        )
        events = np.array([20])
        cleaned = subtract_template(sig, events, template_waveform)
        assert isinstance(cleaned, xr.DataArray)
        assert cleaned.attrs["fs"] == 1000
        assert cleaned.name == "lfp"
        assert cleaned.shape == sig.shape

    def test_multichannel(self, template_waveform):
        """2D signal (channels, time)."""
        n_ch, n_time = 4, 200
        tmpl_2d = np.tile(template_waveform, (n_ch, 1))
        signal = np.zeros((n_ch, n_time))
        events = np.array([30, 80])
        for s0 in events:
            signal[:, s0 : s0 + 20] += tmpl_2d
        cleaned = subtract_template(signal, events, tmpl_2d)
        np.testing.assert_allclose(cleaned, 0.0, atol=1e-12)

    def test_does_not_modify_input(self, template_waveform):
        signal = np.ones(100)
        signal_copy = signal.copy()
        _ = subtract_template(signal, np.array([10]), template_waveform)
        np.testing.assert_array_equal(signal, signal_copy)
