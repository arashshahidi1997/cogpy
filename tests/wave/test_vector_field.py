"""Tests for vector_field module."""

import numpy as np
import pytest
import xarray as xr

from cogpy.wave._types import Geometry, PatternType
from cogpy.wave.vector_field import divergence, curl, critical_points, classify_pattern


@pytest.fixture
def geom():
    return Geometry.regular(1.0, 1.0)


def _uniform_field(ux, uy, n=8):
    """Create a uniform velocity field."""
    u = xr.DataArray(np.full((n, n), ux), dims=("AP", "ML"))
    v = xr.DataArray(np.full((n, n), uy), dims=("AP", "ML"))
    return u, v


class TestDivergence:
    def test_zero_for_uniform(self, geom):
        u, v = _uniform_field(1.0, 2.0)
        d = divergence(u, v, geom)
        np.testing.assert_allclose(d.values, 0.0, atol=1e-10)

    def test_positive_for_source(self, geom):
        """Radial outward field -> positive divergence."""
        n = 16
        x = np.arange(n, dtype=float) - n / 2
        X, Y = np.meshgrid(x, x, indexing="ij")
        u = xr.DataArray(X, dims=("AP", "ML"))
        v = xr.DataArray(Y, dims=("AP", "ML"))
        d = divergence(u, v, geom)
        # Interior points should have divergence ≈ 2.
        np.testing.assert_allclose(d.values[4:-4, 4:-4], 2.0, atol=0.1)


class TestCurl:
    def test_zero_for_uniform(self, geom):
        u, v = _uniform_field(1.0, 2.0)
        c = curl(u, v, geom)
        np.testing.assert_allclose(c.values, 0.0, atol=1e-10)

    def test_nonzero_for_rotating(self, geom):
        """Rotating field u=-y, v=x -> curl = 2."""
        n = 16
        x = np.arange(n, dtype=float) - n / 2
        X, Y = np.meshgrid(x, x, indexing="ij")
        u = xr.DataArray(-Y, dims=("AP", "ML"))
        v = xr.DataArray(X, dims=("AP", "ML"))
        c = curl(u, v, geom)
        np.testing.assert_allclose(c.values[4:-4, 4:-4], 2.0, atol=0.1)


class TestCriticalPoints:
    def test_source_detected(self, geom):
        n = 16
        x = np.arange(n, dtype=float) - n / 2
        X, Y = np.meshgrid(x, x, indexing="ij")
        u = xr.DataArray(X, dims=("AP", "ML"))
        v = xr.DataArray(Y, dims=("AP", "ML"))
        pts = critical_points(u, v, geom)
        types = [p.type for p in pts]
        assert "source" in types

    def test_sink_detected(self, geom):
        n = 16
        x = np.arange(n, dtype=float) - n / 2
        X, Y = np.meshgrid(x, x, indexing="ij")
        u = xr.DataArray(-X, dims=("AP", "ML"))
        v = xr.DataArray(-Y, dims=("AP", "ML"))
        pts = critical_points(u, v, geom)
        types = [p.type for p in pts]
        assert "sink" in types


class TestClassifyPattern:
    def test_planar(self, geom):
        u, v = _uniform_field(1.0, 0.0, n=16)
        assert classify_pattern(u, v, geom) == PatternType.planar

    def test_rotating(self, geom):
        n = 16
        x = np.arange(n, dtype=float) - n / 2
        X, Y = np.meshgrid(x, x, indexing="ij")
        u = xr.DataArray(-Y, dims=("AP", "ML"))
        v = xr.DataArray(X, dims=("AP", "ML"))
        result = classify_pattern(u, v, geom)
        assert result in (PatternType.rotating, PatternType.spiral)
