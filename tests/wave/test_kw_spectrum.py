"""Tests for kw_spectrum module."""

import numpy as np
import pytest

from cogpy.wave._types import Geometry
from cogpy.wave.synthetic import plane_wave
from cogpy.wave.kw_spectrum import kw_spectrum_3d, kw_peaks


@pytest.fixture
def geom():
    return Geometry.regular(1.0, 1.0)


class TestKWSpectrum:
    def test_output_dims(self, geom):
        sig = plane_wave(
            shape=(256, 8, 8),
            geometry=geom,
            direction=0.0,
            speed=10.0,
            frequency=10.0,
            fs=256.0,
        )
        spec = kw_spectrum_3d(sig, geom)
        assert set(spec.dims) == {"freq", "kx", "ky"}
        assert spec.values.min() >= 0  # power is non-negative

    def test_peak_at_correct_frequency(self, geom):
        freq_true = 20.0
        sig = plane_wave(
            shape=(512, 8, 8),
            geometry=geom,
            direction=0.0,
            speed=10.0,
            frequency=freq_true,
            fs=256.0,
        )
        spec = kw_spectrum_3d(sig, geom)
        peaks = kw_peaks(spec, n_peaks=1)
        assert len(peaks) == 1
        # Peak frequency within 2 Hz.
        assert abs(peaks[0].frequency - freq_true) < 2.0

    def test_roundtrip_direction(self, geom):
        """Peak direction should roughly match input."""
        direction_true = np.pi / 4
        # speed=40, freq=10 -> k = 0.25 cycles/m, well below Nyquist (0.5).
        sig = plane_wave(
            shape=(512, 16, 16),
            geometry=geom,
            direction=direction_true,
            speed=40.0,
            frequency=10.0,
            fs=256.0,
        )
        spec = kw_spectrum_3d(sig, geom)
        peaks = kw_peaks(spec, n_peaks=1)
        # Cosine has power at both +k and -k; accept either direction.
        dir_err = abs(np.angle(np.exp(1j * (peaks[0].direction - direction_true))))
        dir_err = min(dir_err, np.pi - dir_err)
        assert dir_err < 0.8

    def test_segmented_spectrum(self, geom):
        """Welch-like averaging with nperseg."""
        sig = plane_wave(
            shape=(1024, 8, 8),
            geometry=geom,
            direction=0.0,
            speed=10.0,
            frequency=10.0,
            fs=256.0,
        )
        spec = kw_spectrum_3d(sig, geom, nperseg=256, noverlap=128)
        assert "freq" in spec.dims
        assert spec.sizes["freq"] == 129  # rfft of 256 -> 129

    def test_multiple_peaks(self, geom):
        from cogpy.wave.synthetic import multi_wave

        c1 = plane_wave(
            shape=(512, 8, 8),
            geometry=geom,
            direction=0.0,
            speed=10.0,
            frequency=10.0,
            fs=256.0,
        )
        c2 = plane_wave(
            shape=(512, 8, 8),
            geometry=geom,
            direction=np.pi / 2,
            speed=10.0,
            frequency=30.0,
            fs=256.0,
        )
        sig = multi_wave([c1, c2])
        spec = kw_spectrum_3d(sig, geom)
        peaks = kw_peaks(spec, n_peaks=2)
        assert len(peaks) == 2
        peak_freqs = sorted([p.frequency for p in peaks])
        assert abs(peak_freqs[0] - 10.0) < 3.0
        assert abs(peak_freqs[1] - 30.0) < 3.0
