"""Tests for generalized_phase module."""

import numpy as np
import xarray as xr

from cogpy.wave.generalized_phase import generalized_phase
from cogpy.wave._types import Geometry
from cogpy.wave.synthetic import plane_wave


class TestGeneralizedPhase:
    def test_shape_preserved(self):
        geom = Geometry.regular(1.0)
        sig = plane_wave(
            shape=(256, 4, 4),
            geometry=geom,
            direction=0.0,
            speed=10.0,
            frequency=10.0,
            fs=256.0,
        )
        phase = generalized_phase(sig)
        assert phase.shape == sig.shape

    def test_monotonic_for_pure_tone(self):
        """Phase of a single-frequency signal should increase monotonically."""
        geom = Geometry.regular(1.0)
        sig = plane_wave(
            shape=(256, 4, 4),
            geometry=geom,
            direction=0.0,
            speed=10.0,
            frequency=10.0,
            fs=256.0,
        )
        phase = generalized_phase(sig)
        # Check at one spatial location (skip edge samples).
        diff = np.diff(phase.values[10:-10, 2, 2])
        assert np.all(diff > 0)

    def test_broadband_stability(self):
        """Phase should still be well-defined for broadband input."""
        rng = np.random.default_rng(42)
        # Broadband: sum of several frequencies.
        t = np.arange(512) / 256.0
        signal = np.zeros((512, 4, 4))
        for f in [5, 10, 20, 40]:
            signal += np.cos(2 * np.pi * f * t)[:, None, None]
        signal += rng.normal(0, 0.1, signal.shape)

        sig = xr.DataArray(
            signal,
            dims=("time", "AP", "ML"),
            coords={
                "time": t,
                "AP": np.arange(4),
                "ML": np.arange(4),
                "fs": 256.0,
            },
        )
        phase = generalized_phase(sig)
        # Phase should be finite and not all identical.
        assert np.all(np.isfinite(phase.values))
        assert np.std(phase.values) > 0

    def test_1d_signal(self):
        """Should work on (time, ch) signals too."""
        t = np.arange(256) / 256.0
        data = np.cos(2 * np.pi * 10 * t[:, None] + np.array([0, 1, 2])[None, :])
        sig = xr.DataArray(
            data,
            dims=("time", "ch"),
            coords={"time": t, "ch": [0, 1, 2], "fs": 256.0},
        )
        phase = generalized_phase(sig)
        assert phase.shape == sig.shape
