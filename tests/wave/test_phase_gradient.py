"""Tests for phase_gradient module — synthetic round-trip verification."""

import numpy as np
import pytest

from cogpy.wave._types import Geometry, PatternType
from cogpy.wave.synthetic import plane_wave
from cogpy.wave.phase_gradient import hilbert_phase, phase_gradient, pgd, plane_wave_fit


@pytest.fixture
def geom():
    return Geometry.regular(1.0, 1.0)


@pytest.fixture
def clean_plane(geom):
    """Noiseless plane wave at 5 Hz travelling along AP axis (direction=0).

    wavelength = speed / freq = 80 / 5 = 16 grid points — well-sampled.
    """
    return plane_wave(
        shape=(500, 8, 8),
        geometry=geom,
        direction=0.0,
        speed=80.0,
        frequency=5.0,
        fs=1000.0,
    )


class TestHilbertPhase:
    def test_shape_preserved(self, clean_plane):
        phase = hilbert_phase(clean_plane)
        assert phase.shape == clean_plane.shape
        assert set(phase.dims) == {"time", "AP", "ML"}

    def test_monotonic_in_time(self, clean_plane):
        phase = hilbert_phase(clean_plane)
        # Phase should increase along time for positive frequency.
        diff = np.diff(phase.values[:, 4, 4])
        assert np.all(diff > 0)


class TestPhaseGradient:
    def test_gradient_shape(self, clean_plane, geom):
        phase = hilbert_phase(clean_plane)
        dphi_dx, dphi_dy = phase_gradient(phase, geom)
        assert dphi_dx.shape == phase.shape
        assert dphi_dy.shape == phase.shape

    def test_gradient_direction(self, geom):
        """For a wave travelling along AP (direction=0), gradient should be
        primarily along AP (dphi_dx dominant)."""
        sig = plane_wave(
            shape=(500, 8, 8),
            geometry=geom,
            direction=0.0,
            speed=80.0,
            frequency=5.0,
            fs=1000.0,
        )
        phase = hilbert_phase(sig)
        dphi_dx, dphi_dy = phase_gradient(phase, geom)
        # Interior points to avoid edge effects.
        interior = slice(2, -2)
        mean_gx = np.mean(np.abs(dphi_dx.values[100:400, interior, interior]))
        mean_gy = np.mean(np.abs(dphi_dy.values[100:400, interior, interior]))
        assert mean_gx > 3 * mean_gy

    def test_irregular_geometry(self):
        """Phase gradient on irregular arrays via linear regression."""
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 5, size=(16, 2))
        geom = Geometry.irregular(coords)

        # Synthetic linear phase field: φ = 2*x + 3*y + t*ω
        import xarray as xr

        n_t = 100
        t = np.arange(n_t) / 1000.0
        phi = (
            np.outer(t * 2 * np.pi * 10, np.ones(16))
            + (2 * coords[:, 0] + 3 * coords[:, 1])[None, :]
        )
        phase = xr.DataArray(
            phi,
            dims=("time", "ch"),
            coords={"time": t, "ch": np.arange(16)},
        )
        dphi_dx, dphi_dy = phase_gradient(phase, geom)
        # Should recover gradients ≈ 2 and ≈ 3.
        np.testing.assert_allclose(dphi_dx.values, 2.0, atol=0.01)
        np.testing.assert_allclose(dphi_dy.values, 3.0, atol=0.01)


class TestPGD:
    def test_high_pgd_for_plane_wave(self, geom):
        """PGD on a known linear phase field should be ~1."""
        import xarray as xr

        n_t, n_ap, n_ml = 100, 16, 16
        t = np.arange(n_t) / 1000.0
        ap = np.arange(n_ap, dtype=float)
        ml = np.arange(n_ml, dtype=float)
        # Linear phase ramp along AP: φ = 2π*5*t - 0.3*x
        T = t[:, None, None]
        X = ap[None, :, None]
        Y = ml[None, None, :]
        phase_vals = 2 * np.pi * 5 * T - 0.3 * X + 0 * Y
        phase = xr.DataArray(
            phase_vals,
            dims=("time", "AP", "ML"),
            coords={"time": t, "AP": ap, "ML": ml, "fs": 1000.0},
        )
        scores = pgd(phase, geom)
        assert np.mean(scores.values) > 0.85

    def test_low_pgd_for_noise(self):
        rng = np.random.default_rng(0)
        import xarray as xr

        geom_big = Geometry.regular(1.0)
        noise = xr.DataArray(
            rng.normal(size=(200, 16, 16)),
            dims=("time", "AP", "ML"),
            coords={
                "time": np.arange(200) / 1000.0,
                "AP": np.arange(16),
                "ML": np.arange(16),
                "fs": 1000.0,
            },
        )
        phase = hilbert_phase(noise)
        scores = pgd(phase, geom_big)
        assert np.mean(scores.values) < 0.5


class TestPlaneWaveFit:
    def test_roundtrip_direction(self, geom):
        """Fit should recover direction and speed from a known phase field."""
        import xarray as xr

        direction_true = np.pi / 4
        freq_true = 5.0
        speed_true = 80.0
        # kx = cos(dir)/speed, ky = sin(dir)/speed
        kx = np.cos(direction_true) / speed_true
        ky = np.sin(direction_true) / speed_true

        n_t, n_ap, n_ml = 200, 16, 16
        t = np.arange(n_t) / 1000.0
        ap = np.arange(n_ap, dtype=float)
        ml = np.arange(n_ml, dtype=float)
        T = t[:, None, None]
        X = ap[None, :, None]
        Y = ml[None, None, :]
        phase_vals = 2 * np.pi * freq_true * (T - kx * X - ky * Y)
        phase = xr.DataArray(
            phase_vals,
            dims=("time", "AP", "ML"),
            coords={"time": t, "AP": ap, "ML": ml, "fs": 1000.0},
        )
        estimates = plane_wave_fit(phase, geom, freq=freq_true)
        est = estimates[100]
        dir_err = abs(np.angle(np.exp(1j * (est.direction - direction_true))))
        assert dir_err < 0.2, f"Direction error {dir_err:.3f}"
        assert abs(est.speed - speed_true) / speed_true < 0.3
        assert est.pattern_type == PatternType.planar
        assert est.fit_quality > 0.8

    def test_fit_quality_decreases_with_noise(self, geom):
        sig_clean = plane_wave(
            shape=(500, 8, 8),
            geometry=geom,
            direction=0.0,
            speed=80.0,
            frequency=5.0,
            fs=1000.0,
        )
        sig_noisy = plane_wave(
            shape=(500, 8, 8),
            geometry=geom,
            direction=0.0,
            speed=80.0,
            frequency=5.0,
            fs=1000.0,
            noise_std=2.0,
            rng=42,
        )
        phase_clean = hilbert_phase(sig_clean)
        phase_noisy = hilbert_phase(sig_noisy)
        est_clean = plane_wave_fit(phase_clean, geom, freq=5.0)
        est_noisy = plane_wave_fit(phase_noisy, geom, freq=5.0)
        fq_clean = np.mean([e.fit_quality for e in est_clean[100:400]])
        fq_noisy = np.mean([e.fit_quality for e in est_noisy[100:400]])
        assert fq_clean > fq_noisy
