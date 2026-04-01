"""Tests for optical_flow module."""

import numpy as np
import pytest
import xarray as xr

from cogpy.wave._types import Geometry
from cogpy.wave.synthetic import plane_wave

try:
    import skimage  # noqa: F401

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

pytestmark = pytest.mark.skipif(not HAS_SKIMAGE, reason="scikit-image not installed")


@pytest.fixture
def geom():
    return Geometry.regular(1.0, 1.0)


class TestComputeFlow:
    def test_output_shape(self, geom):
        from cogpy.wave.optical_flow import compute_flow

        # Use a slow wave with large spatial features for visible motion.
        sig = plane_wave(
            shape=(10, 32, 32),
            geometry=geom,
            direction=0.0,
            speed=50.0,
            frequency=2.0,
            fs=100.0,
        )
        u, v = compute_flow(sig, method="ilk")
        assert u.shape == (9, 32, 32)
        assert v.shape == (9, 32, 32)
        assert set(u.dims) == {"time", "AP", "ML"}

    def test_dominant_direction(self, geom):
        """Flow for a moving Gaussian blob should be detectable."""
        from cogpy.wave.optical_flow import compute_flow

        # Create a moving bright blob — easier for optical flow than a cosine.
        n_t, n_ap, n_ml = 10, 32, 32
        ap = np.arange(n_ap, dtype=float)
        ml = np.arange(n_ml, dtype=float)
        X, Y = np.meshgrid(ap, ml, indexing="ij")
        frames = []
        for i in range(n_t):
            cx = 10.0 + i * 2.0  # move along AP
            cy = 16.0
            blob = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * 3.0**2))
            frames.append(blob)
        data = np.stack(frames)
        sig = xr.DataArray(
            data,
            dims=("time", "AP", "ML"),
            coords={"time": np.arange(n_t) / 100.0, "AP": ap, "ML": ml, "fs": 100.0},
        )
        u, v = compute_flow(sig, method="ilk")
        # Flow should be non-trivial.
        total_flow = np.mean(np.abs(u.values)) + np.mean(np.abs(v.values))
        assert total_flow > 0, "Flow should be non-zero for a moving blob"


class TestFlowToSpeedDirection:
    def test_speed_positive(self):
        from cogpy.wave.optical_flow import flow_to_speed_direction

        u = xr.DataArray(np.ones((5, 4, 4)), dims=("time", "AP", "ML"))
        v = xr.DataArray(np.ones((5, 4, 4)), dims=("time", "AP", "ML"))
        speed, direction = flow_to_speed_direction(u, v)
        np.testing.assert_allclose(speed.values, np.sqrt(2.0))
        np.testing.assert_allclose(direction.values, np.pi / 4)
