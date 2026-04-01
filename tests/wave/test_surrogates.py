"""Tests for surrogates module."""

import numpy as np
import pytest

from cogpy.wave._types import Geometry
from cogpy.wave.synthetic import plane_wave
from cogpy.wave.surrogates import phase_randomize, spatial_shuffle, surrogate_test


@pytest.fixture
def geom():
    return Geometry.regular(1.0, 1.0)


@pytest.fixture
def signal(geom):
    return plane_wave(
        shape=(256, 4, 4),
        geometry=geom,
        direction=0.0,
        speed=10.0,
        frequency=10.0,
        fs=256.0,
    )


class TestPhaseRandomize:
    def test_shape_preserved(self, signal):
        surr = phase_randomize(signal, rng=0)
        assert surr.shape == signal.shape

    def test_power_spectrum_preserved(self, signal):
        surr = phase_randomize(signal, rng=0)
        # Compare power spectra at one spatial location.
        orig_psd = np.abs(np.fft.rfft(signal.values[:, 2, 2])) ** 2
        surr_psd = np.abs(np.fft.rfft(surr.values[:, 2, 2])) ** 2
        # Use atol for near-zero bins and rtol for significant bins.
        np.testing.assert_allclose(orig_psd, surr_psd, rtol=1e-6, atol=1e-20)

    def test_different_from_original(self, signal):
        surr = phase_randomize(signal, rng=0)
        assert not np.allclose(surr.values, signal.values)


class TestSpatialShuffle:
    def test_shape_preserved(self, signal):
        surr = spatial_shuffle(signal, rng=0)
        assert surr.shape == signal.shape

    def test_values_conserved(self, signal):
        """All values should be present, just rearranged spatially."""
        surr = spatial_shuffle(signal, rng=0)
        for t in range(min(5, signal.shape[0])):
            orig_sorted = np.sort(signal.values[t].ravel())
            surr_sorted = np.sort(surr.values[t].ravel())
            np.testing.assert_allclose(orig_sorted, surr_sorted)


class TestSurrogateTest:
    def test_significant_for_real_wave(self, signal):
        """A real plane wave should have higher PGD than surrogates."""
        from cogpy.wave.phase_gradient import hilbert_phase, pgd

        geom = Geometry.regular(1.0)

        def estimator(data):
            phase = hilbert_phase(data)
            return float(pgd(phase, geom).mean())

        p_val, observed, null_dist = surrogate_test(
            signal,
            estimator,
            n_surrogates=50,
            seed=42,
            surrogate_type="spatial",
        )
        # PGD of the real wave should be significantly higher.
        assert p_val < 0.1
        assert observed > np.median(null_dist)

    def test_phase_surrogate_type(self, signal):
        def estimator(data):
            return float(np.std(data.values))

        p_val, observed, null_dist = surrogate_test(
            signal,
            estimator,
            n_surrogates=20,
            seed=0,
            surrogate_type="phase",
        )
        assert null_dist.shape == (20,)
