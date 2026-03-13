"""Tests for cogpy.core.spectral.features — spectral scalar/vector features."""

import numpy as np
import pytest

from cogpy.core.spectral.features import (
    narrowband_ratio,
    ftest_line_scan,
    spectral_flatness,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _white_psd(nf=256, seed=42):
    """Flat PSD (white noise)."""
    rng = np.random.default_rng(seed)
    freqs = np.linspace(0, 500, nf)
    psd = np.ones(nf) + rng.normal(0, 0.01, nf)
    psd = np.abs(psd)
    return psd, freqs


def _line_psd(nf=256, f_line=60.0, line_power=100.0, seed=42):
    """Flat PSD with a narrow peak at f_line."""
    psd, freqs = _white_psd(nf, seed)
    idx = np.argmin(np.abs(freqs - f_line))
    psd[idx] = line_power
    return psd, freqs


def _synth_signal(fs=1000.0, duration=2.0, f_line=60.0, snr_db=20.0, seed=42):
    """White noise + sinusoidal line at f_line."""
    rng = np.random.default_rng(seed)
    N = int(fs * duration)
    t = np.arange(N) / fs
    noise = rng.normal(0, 1.0, N)
    amp = 10 ** (snr_db / 20.0)
    signal = noise + amp * np.sin(2 * np.pi * f_line * t)
    return signal, fs


# ---------------------------------------------------------------------------
# spectral_flatness
# ---------------------------------------------------------------------------

class TestSpectralFlatness:
    def test_white_noise(self):
        """Flat PSD → flatness near 1."""
        psd = np.ones(100)
        freqs = np.linspace(0, 500, 100)
        sf = spectral_flatness(psd, freqs)
        assert sf == pytest.approx(1.0, abs=0.01)

    def test_pure_tone(self):
        """Single peak → flatness near 0."""
        psd = np.zeros(100) + 1e-10
        psd[50] = 1000.0
        freqs = np.linspace(0, 500, 100)
        sf = spectral_flatness(psd, freqs)
        assert sf < 0.1

    def test_batch_shape(self):
        """Batch dims preserved."""
        psd = np.ones((3, 4, 100))
        freqs = np.linspace(0, 500, 100)
        sf = spectral_flatness(psd, freqs)
        assert sf.shape == (3, 4)


# ---------------------------------------------------------------------------
# narrowband_ratio
# ---------------------------------------------------------------------------

class TestNarrowbandRatio:
    def test_flat_psd_ratio_near_one(self):
        """Flat PSD → all ratios near 1."""
        psd, freqs = _white_psd()
        ratio = narrowband_ratio(psd, freqs, flank_hz=10.0)
        # Exclude edge bins (may be NaN)
        interior = ratio[~np.isnan(ratio)]
        assert np.all(interior < 2.0)
        assert np.all(interior > 0.5)

    def test_line_detected(self):
        """Narrow peak → ratio >> 1 at peak frequency."""
        psd, freqs = _line_psd(f_line=60.0, line_power=100.0)
        ratio = narrowband_ratio(psd, freqs, flank_hz=10.0)
        idx_60 = np.argmin(np.abs(freqs - 60.0))
        assert ratio[idx_60] > 10.0

    def test_output_shape(self):
        """Output shape matches input PSD shape."""
        psd, freqs = _white_psd()
        ratio = narrowband_ratio(psd, freqs)
        assert ratio.shape == psd.shape

    def test_batch_dims(self):
        """Batch dims preserved."""
        psd, freqs = _white_psd()
        psd_batch = np.stack([psd, psd * 2], axis=0)  # (2, nf)
        ratio = narrowband_ratio(psd_batch, freqs)
        assert ratio.shape == psd_batch.shape

    def test_edge_bins_nan(self):
        """Very narrow freq range → edge bins are NaN."""
        freqs = np.linspace(0, 10, 5)  # only 5 bins, 2.5 Hz spacing
        psd = np.ones(5)
        ratio = narrowband_ratio(psd, freqs, flank_hz=1.0)
        # With 2.5 Hz spacing and 1 Hz flank, most bins have <2 flanks
        assert np.any(np.isnan(ratio))


# ---------------------------------------------------------------------------
# ftest_line_scan
# ---------------------------------------------------------------------------

class TestFtestLineScan:
    def test_detects_injected_line(self):
        """Sinusoid in noise → significant F-stat at injected freq."""
        signal, fs = _synth_signal(f_line=60.0, snr_db=20.0)
        fstat, freqs, sig_mask = ftest_line_scan(signal, fs, NW=4.0, p_threshold=0.05)

        # Find the bin nearest 60 Hz
        idx_60 = np.argmin(np.abs(freqs - 60.0))
        assert sig_mask[idx_60], "60 Hz line not detected"
        assert fstat[idx_60] > 10.0

    def test_clean_signal_few_detections(self):
        """Pure noise → few or no significant bins (at most ~5%)."""
        rng = np.random.default_rng(123)
        signal = rng.normal(0, 1, 2000)
        fstat, freqs, sig_mask = ftest_line_scan(signal, 1000.0, p_threshold=0.05)
        # Under H0, expect ~5% false positives
        false_pos_rate = np.mean(sig_mask)
        assert false_pos_rate < 0.15  # generous bound

    def test_output_shapes(self):
        """Shapes are consistent."""
        signal, fs = _synth_signal()
        fstat, freqs, sig_mask = ftest_line_scan(signal, fs)
        assert fstat.shape == freqs.shape
        assert sig_mask.shape == freqs.shape
        assert sig_mask.dtype == bool

    def test_rejects_multidim(self):
        """Non-1D input raises ValueError."""
        with pytest.raises(ValueError, match="1D"):
            ftest_line_scan(np.ones((10, 10)), 1000.0)

    def test_multiple_lines(self):
        """Two injected lines → both detected."""
        rng = np.random.default_rng(42)
        fs = 1000.0
        N = 4000
        t = np.arange(N) / fs
        signal = rng.normal(0, 1, N)
        signal += 50.0 * np.sin(2 * np.pi * 60 * t)
        signal += 30.0 * np.sin(2 * np.pi * 120 * t)
        fstat, freqs, sig_mask = ftest_line_scan(signal, fs, p_threshold=0.01)
        idx_60 = np.argmin(np.abs(freqs - 60.0))
        idx_120 = np.argmin(np.abs(freqs - 120.0))
        assert sig_mask[idx_60]
        assert sig_mask[idx_120]
