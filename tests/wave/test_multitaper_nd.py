"""Tests for multitaper_nd module."""

import numpy as np
import pytest

from cogpy.wave._types import Geometry
from cogpy.wave.multitaper_nd import dpss_nd, multitaper_kw_spectrum
from cogpy.wave.synthetic import plane_wave


class TestDPSSND:
    def test_shape(self):
        tapers = dpss_nd((64, 8, 8), (4.0, 2.0, 2.0))
        for t in tapers:
            assert t.shape == (64, 8, 8)
        # With NW=4, K=7 per dim; NW=2, K=3 per dim → 7*3*3 = 63 tapers.
        assert len(tapers) == 7 * 3 * 3

    def test_single_taper(self):
        tapers = dpss_nd((32,), (1.0,))
        assert len(tapers) == 1
        assert tapers[0].shape == (32,)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            dpss_nd((64, 8), (4.0,))


class TestMultitaperKWSpectrum:
    def test_output_dims(self):
        geom = Geometry.regular(1.0)
        sig = plane_wave(
            shape=(64, 4, 4),
            geometry=geom,
            direction=0.0,
            speed=10.0,
            frequency=10.0,
            fs=128.0,
        )
        spec = multitaper_kw_spectrum(sig, geom, bw_time=2.0, bw_space=1.5)
        assert set(spec.dims) == {"freq", "kx", "ky"}
        assert spec.values.min() >= 0

    def test_peak_at_signal_frequency(self):
        geom = Geometry.regular(1.0)
        freq_true = 15.0
        sig = plane_wave(
            shape=(128, 4, 4),
            geometry=geom,
            direction=0.0,
            speed=10.0,
            frequency=freq_true,
            fs=128.0,
        )
        spec = multitaper_kw_spectrum(sig, geom, bw_time=3.0, bw_space=1.5)
        # Find peak frequency.
        power_vs_freq = spec.sum(dim=["kx", "ky"]).values
        freqs = spec.coords["freq"].values
        peak_freq = freqs[np.argmax(power_vs_freq[1:])] if len(freqs) > 1 else freqs[0]
        # Tolerance: within a few bins.
        assert abs(peak_freq - freq_true) < 5.0
