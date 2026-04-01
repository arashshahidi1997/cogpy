"""Tests for cogpy.spectral.features — spectral scalar/vector features."""

import numpy as np
import pytest

from cogpy.spectral.features import (
    narrowband_ratio,
    spectral_peak_freqs,
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
# spectral_peak_freqs
# ---------------------------------------------------------------------------


class TestSpectralPeakFreqs:
    def test_single_peak(self):
        """Single narrow peak detected."""
        psd, freqs = _line_psd(f_line=60.0, line_power=100.0)
        peaks = spectral_peak_freqs(psd, freqs, prominence=5.0)
        assert isinstance(peaks, np.ndarray)
        assert len(peaks) >= 1
        assert np.any(np.abs(peaks - 60.0) < 5.0)

    def test_two_peaks(self):
        """Two peaks at different frequencies."""
        psd, freqs = _line_psd(f_line=60.0, line_power=100.0)
        idx_120 = np.argmin(np.abs(freqs - 120.0))
        psd[idx_120] = 80.0
        peaks = spectral_peak_freqs(psd, freqs, prominence=5.0, min_distance_hz=10.0)
        near_60 = np.any(np.abs(peaks - 60.0) < 5.0)
        near_120 = np.any(np.abs(peaks - 120.0) < 5.0)
        assert near_60 and near_120

    def test_flat_psd_no_peaks(self):
        """Flat PSD → no peaks above prominence."""
        psd = np.ones(256)
        freqs = np.linspace(0, 500, 256)
        peaks = spectral_peak_freqs(psd, freqs, prominence=2.0)
        assert len(peaks) == 0

    def test_batch_returns_list(self):
        """Batched PSD → list of arrays."""
        psd, freqs = _line_psd(f_line=60.0, line_power=100.0)
        psd_batch = np.stack([psd, np.ones_like(psd)], axis=0)  # (2, nf)
        result = spectral_peak_freqs(psd_batch, freqs, prominence=5.0)
        assert isinstance(result, list)
        assert len(result) == 2
        # First element has a peak, second (flat) does not
        assert len(result[0]) >= 1
        assert len(result[1]) == 0

    def test_min_distance(self):
        """Peaks closer than min_distance_hz are suppressed."""
        freqs = np.linspace(0, 500, 1000)
        psd = np.ones(1000)
        # Two peaks 3 Hz apart
        idx_a = np.argmin(np.abs(freqs - 100.0))
        idx_b = np.argmin(np.abs(freqs - 103.0))
        psd[idx_a] = 50.0
        psd[idx_b] = 40.0
        # With 5 Hz min distance, only the larger peak survives
        peaks = spectral_peak_freqs(psd, freqs, prominence=5.0, min_distance_hz=5.0)
        assert len(peaks) == 1
        assert np.abs(peaks[0] - 100.0) < 2.0


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

    def test_batch_dims(self):
        """Batch dims are supported: (..., time) → (..., freq)."""
        signal, fs = _synth_signal()
        # Stack two copies as a (2, time) batch
        batch = np.stack([signal, signal])
        fstat, freqs, sig_mask = ftest_line_scan(batch, fs)
        assert fstat.shape == (2, freqs.shape[0])
        assert sig_mask.shape == (2, freqs.shape[0])
        # Results should be identical for identical signals
        np.testing.assert_allclose(fstat[0], fstat[1], rtol=1e-10)

    def test_batch_3d(self):
        """3D batch: (AP, ML, time) → (AP, ML, freq)."""
        rng = np.random.default_rng(99)
        fs = 1000.0
        N = 512
        batch = rng.standard_normal((3, 4, N))
        fstat, freqs, sig_mask = ftest_line_scan(batch, fs)
        assert fstat.shape == (3, 4, freqs.shape[0])
        assert sig_mask.shape == (3, 4, freqs.shape[0])

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


# ---------------------------------------------------------------------------
# reduce_tf_bands
# ---------------------------------------------------------------------------


class TestReduceTfBands:
    @staticmethod
    def _score_da():
        """(time_win, freq) score array."""
        import xarray as xr

        return xr.DataArray(
            np.ones((10, 100)),
            dims=("time_win", "freq"),
            coords={"time_win": np.arange(10), "freq": np.linspace(1, 200, 100)},
        )

    def test_basic_mean(self):
        from cogpy.spectral.features import reduce_tf_bands

        da = self._score_da()
        bands = {"alpha": (8, 13), "beta": (13, 30)}
        ds = reduce_tf_bands(da, bands)
        assert "alpha" in ds
        assert "beta" in ds
        assert "freq" not in ds.dims
        assert ds["alpha"].dims == ("time_win",)

    def test_median_method(self):
        from cogpy.spectral.features import reduce_tf_bands

        da = self._score_da()
        ds = reduce_tf_bands(da, {"low": (1, 50)}, method="median")
        assert ds["low"].dims == ("time_win",)

    def test_no_overlap_raises(self):
        from cogpy.spectral.features import reduce_tf_bands

        da = self._score_da()
        with pytest.raises(ValueError, match="no overlap"):
            reduce_tf_bands(da, {"oob": (500, 600)})

    def test_unknown_method_raises(self):
        from cogpy.spectral.features import reduce_tf_bands

        da = self._score_da()
        with pytest.raises(ValueError, match="Unknown method"):
            reduce_tf_bands(da, {"a": (1, 50)}, method="bad")

    def test_preserves_batch_dims(self):
        """Batch dims beyond (time_win, freq) are preserved."""
        import xarray as xr
        from cogpy.spectral.features import reduce_tf_bands

        da = xr.DataArray(
            np.ones((5, 10, 100)),
            dims=("batch", "time_win", "freq"),
            coords={"freq": np.linspace(1, 200, 100)},
        )
        ds = reduce_tf_bands(da, {"gamma": (30, 100)})
        assert ds["gamma"].dims == ("batch", "time_win")


# ---------------------------------------------------------------------------
# normalize_spectrogram
# ---------------------------------------------------------------------------


class TestNormalizeSpectrogram:
    @staticmethod
    def _spec_da():
        import xarray as xr

        rng = np.random.default_rng(42)
        return xr.DataArray(
            rng.random((4, 8, 64, 10)) + 0.01,
            dims=("AP", "ML", "freq", "time_win"),
            coords={
                "AP": np.arange(4),
                "ML": np.arange(8),
                "freq": np.linspace(1, 200, 64),
                "time_win": np.linspace(0, 9, 10),
            },
        )

    def test_robust_zscore(self):
        from cogpy.spectral.specx import normalize_spectrogram

        spec = self._spec_da()
        out = normalize_spectrogram(spec, method="robust_zscore", dim="freq")
        assert out.shape == spec.shape
        assert out.attrs["normalization"] == "robust_zscore"

    def test_db(self):
        from cogpy.spectral.specx import normalize_spectrogram

        spec = self._spec_da()
        out = normalize_spectrogram(spec, method="db")
        assert out.shape == spec.shape
        assert out.attrs["normalization"] == "db"
        assert out.attrs["units"] == "dB"
        # dB of values in (0,1) should be negative
        assert float(out.max()) < 0 or float(out.min()) < 0

    def test_unknown_method_raises(self):
        from cogpy.spectral.specx import normalize_spectrogram

        with pytest.raises(ValueError, match="Unknown normalization"):
            normalize_spectrogram(self._spec_da(), method="bad")
