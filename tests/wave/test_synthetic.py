"""Tests for synthetic wave generators."""

import numpy as np
import pytest

from cogpy.wave._types import Geometry
from cogpy.wave.synthetic import plane_wave, spiral_wave, wave_packet, multi_wave


@pytest.fixture
def geom():
    return Geometry.regular(1.0, 1.0)


class TestPlaneWave:
    def test_shape_and_coords(self, geom):
        sig = plane_wave(
            shape=(100, 4, 4),
            geometry=geom,
            direction=0.0,
            speed=10.0,
            frequency=5.0,
            fs=500.0,
        )
        assert sig.shape == (100, 4, 4)
        assert set(sig.dims) == {"time", "AP", "ML"}
        assert float(sig.coords["fs"]) == 500.0

    def test_deterministic(self, geom):
        a = plane_wave(
            shape=(50, 4, 4),
            geometry=geom,
            direction=0.0,
            speed=10.0,
            frequency=5.0,
        )
        b = plane_wave(
            shape=(50, 4, 4),
            geometry=geom,
            direction=0.0,
            speed=10.0,
            frequency=5.0,
        )
        np.testing.assert_array_equal(a.values, b.values)

    def test_noise(self, geom):
        sig = plane_wave(
            shape=(100, 4, 4),
            geometry=geom,
            direction=0.0,
            speed=10.0,
            frequency=5.0,
            noise_std=0.1,
            rng=42,
        )
        # Should exceed [-1, 1] occasionally with noise.
        assert np.max(np.abs(sig.values)) > 0.5


class TestSpiralWave:
    def test_shape(self, geom):
        sig = spiral_wave(
            shape=(100, 8, 8),
            geometry=geom,
            center=(3.5, 3.5),
            angular_freq=2 * np.pi * 5,
            fs=500.0,
        )
        assert sig.shape == (100, 8, 8)


class TestWavePacket:
    def test_envelope(self, geom):
        sig = wave_packet(
            shape=(500, 8, 8),
            geometry=geom,
            direction=0.0,
            speed=10.0,
            frequency=10.0,
            sigma_t=0.05,
            sigma_x=2.0,
            fs=1000.0,
        )
        # Envelope should peak near the center time.
        center_trace = sig.values[:, 4, 4]
        peak_idx = np.argmax(np.abs(center_trace))
        assert 150 < peak_idx < 350


class TestMultiWave:
    def test_superposition(self, geom):
        c1 = plane_wave(
            shape=(100, 4, 4),
            geometry=geom,
            direction=0.0,
            speed=10.0,
            frequency=5.0,
        )
        c2 = plane_wave(
            shape=(100, 4, 4),
            geometry=geom,
            direction=np.pi / 2,
            speed=10.0,
            frequency=10.0,
        )
        result = multi_wave([c1, c2])
        np.testing.assert_allclose(result.values, c1.values + c2.values)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            multi_wave([])
