"""Tests for cogpy.measures.comparison — before/after signal metrics."""

import numpy as np
import pytest

from cogpy.measures.comparison import (
    snr_improvement,
    residual_energy_ratio,
    bandpower_change,
    waveform_residual_rms,
)


@pytest.fixture
def freqs():
    return np.linspace(0, 500, 501)


@pytest.fixture
def flat_psd(freqs):
    """Flat PSD at 1.0 across all frequencies."""
    return np.ones_like(freqs)


# ---------------------------------------------------------------------------
# residual_energy_ratio
# ---------------------------------------------------------------------------

class TestResidualEnergyRatio:
    def test_no_change(self):
        """Same signal → ratio = 0."""
        x = np.random.default_rng(0).normal(size=100)
        ratio = residual_energy_ratio(x, x)
        np.testing.assert_allclose(ratio, 0.0, atol=1e-10)

    def test_full_removal(self):
        """Cleaned = 0 → ratio = 1."""
        x = np.ones(50) * 5.0
        ratio = residual_energy_ratio(x, np.zeros(50))
        np.testing.assert_allclose(ratio, 1.0, atol=1e-10)

    def test_partial_removal(self):
        x = np.array([1.0, 2.0, 3.0])
        c = np.array([0.5, 1.0, 1.5])  # half removed
        ratio = residual_energy_ratio(x, c)
        expected = np.sum((x - c) ** 2) / np.sum(x ** 2)
        np.testing.assert_allclose(ratio, expected, atol=1e-10)

    def test_multichannel(self):
        """(channels, time) → per-channel ratio."""
        x = np.ones((3, 100))
        c = np.zeros((3, 100))
        ratio = residual_energy_ratio(x, c, axis=-1)
        np.testing.assert_allclose(ratio, 1.0)
        assert ratio.shape == (3,)


# ---------------------------------------------------------------------------
# bandpower_change
# ---------------------------------------------------------------------------

class TestBandpowerChange:
    def test_no_change(self, flat_psd, freqs):
        change = bandpower_change(flat_psd, flat_psd, freqs, band=(10, 50))
        np.testing.assert_allclose(change, 0.0, atol=1e-10)

    def test_doubled_power(self, flat_psd, freqs):
        psd_after = flat_psd * 2.0
        change = bandpower_change(flat_psd, psd_after, freqs, band=(10, 50))
        np.testing.assert_allclose(change, 1.0, atol=0.01)  # 100% increase

    def test_halved_power(self, flat_psd, freqs):
        psd_after = flat_psd * 0.5
        change = bandpower_change(flat_psd, psd_after, freqs, band=(10, 50))
        np.testing.assert_allclose(change, -0.5, atol=0.01)  # 50% decrease

    def test_narrowband_removal(self, freqs):
        """Remove a peak → negative change in that band."""
        psd_before = np.ones_like(freqs)
        psd_before[100:140] = 10.0  # peak at 100-140 Hz
        psd_after = np.ones_like(freqs)  # peak removed
        change = bandpower_change(psd_before, psd_after, freqs, band=(100, 140))
        assert change < -0.5  # big decrease


# ---------------------------------------------------------------------------
# snr_improvement
# ---------------------------------------------------------------------------

class TestSnrImprovement:
    def test_no_change(self, flat_psd, freqs):
        imp = snr_improvement(
            flat_psd, flat_psd, freqs,
            signal_band=(1, 30), noise_band=(100, 200),
        )
        assert imp == pytest.approx(0.0, abs=1e-10)

    def test_noise_reduction(self, freqs):
        """Reduce noise band → positive improvement."""
        psd_before = np.ones_like(freqs)
        psd_after = np.ones_like(freqs)
        psd_after[100:200] = 0.1  # reduce noise band
        imp = snr_improvement(
            psd_before, psd_after, freqs,
            signal_band=(1, 30), noise_band=(100, 200),
        )
        assert imp > 0

    def test_signal_reduction_negative(self, freqs):
        """Reduce signal band → negative improvement."""
        psd_before = np.ones_like(freqs)
        psd_after = np.ones_like(freqs)
        psd_after[1:30] = 0.1  # reduce signal
        imp = snr_improvement(
            psd_before, psd_after, freqs,
            signal_band=(1, 30), noise_band=(100, 200),
        )
        assert imp < 0


# ---------------------------------------------------------------------------
# waveform_residual_rms
# ---------------------------------------------------------------------------

class TestWaveformResidualRms:
    def test_identical(self):
        w = np.sin(np.linspace(0, 2 * np.pi, 50))
        rms = waveform_residual_rms(w, w)
        assert rms == pytest.approx(0.0, abs=1e-15)

    def test_known_difference(self):
        a = np.ones(100)
        b = np.zeros(100)
        rms = waveform_residual_rms(a, b)
        assert rms == pytest.approx(1.0, abs=1e-10)

    def test_symmetric(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=50)
        b = rng.normal(size=50)
        assert waveform_residual_rms(a, b) == pytest.approx(
            waveform_residual_rms(b, a), abs=1e-15
        )
