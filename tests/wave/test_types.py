"""Tests for _types module."""

import numpy as np
import pytest

from cogpy.wave._types import Geometry, PatternType, WaveEstimate


class TestGeometry:
    def test_regular(self):
        g = Geometry.regular(0.4, 0.4)
        assert g.is_regular
        assert g.dx == 0.4
        assert g.dy == 0.4
        assert g.coords is None

    def test_regular_default_dy(self):
        g = Geometry.regular(0.5)
        assert g.dy == 0.5

    def test_irregular(self):
        coords = np.array([[0, 0], [1, 0], [0.5, 0.8]])
        g = Geometry.irregular(coords)
        assert not g.is_regular
        assert g.coords.shape == (3, 2)

    def test_irregular_bad_shape(self):
        with pytest.raises(ValueError):
            Geometry.irregular(np.array([1, 2, 3]))


class TestWaveEstimate:
    def test_auto_wavelength(self):
        est = WaveEstimate(
            direction=0.0,
            speed=10.0,
            frequency=5.0,
            wavenumber=0.5,
        )
        assert est.wavelength == pytest.approx(2.0)

    def test_auto_wavenumber(self):
        est = WaveEstimate(
            direction=0.0,
            speed=10.0,
            frequency=5.0,
            wavelength=4.0,
        )
        assert est.wavenumber == pytest.approx(0.25)


class TestPatternType:
    def test_values(self):
        assert PatternType.planar.value == "planar"
        assert PatternType.rotating.value == "rotating"
