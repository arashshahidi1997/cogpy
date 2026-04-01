"""Tests for beamforming module."""

import numpy as np
import pytest

from cogpy.wave.beamforming import fk_spectrum, capon_beamformer


@pytest.fixture
def linear_array():
    """8-element linear array with 1m spacing."""
    coords = np.column_stack([np.arange(8, dtype=float), np.zeros(8)])
    return coords


@pytest.fixture
def slowness_grid():
    sx = np.linspace(-0.15, 0.15, 31)
    sy = np.linspace(-0.15, 0.15, 31)
    SX, SY = np.meshgrid(sx, sy, indexing="ij")
    return np.stack([SX, SY], axis=-1)


def _plane_wave_data(coords, freq, speed, direction, fs=200.0, n_time=512):
    """Generate multi-channel plane wave for beamforming tests."""
    t = np.arange(n_time) / fs
    kx = np.cos(direction) / speed
    ky = np.sin(direction) / speed
    delays = kx * coords[:, 0] + ky * coords[:, 1]
    data = np.cos(2 * np.pi * freq * (t[:, None] - delays[None, :]))
    return data


class TestFKSpectrum:
    def test_output_shape(self, linear_array, slowness_grid):
        data = _plane_wave_data(linear_array, freq=10.0, speed=15.0, direction=0.0)
        freqs = np.array([10.0])
        beam = fk_spectrum(data, linear_array, freqs, slowness_grid, fs=200.0)
        assert beam.shape == (1, 31, 31)
        assert np.all(beam >= 0)

    def test_peak_at_correct_slowness(self, linear_array):
        """Peak should be at slowness ≈ 1/speed along the wave direction."""
        # speed=30 -> slowness ≈ 0.033. Restrict slowness grid to unambiguous
        # region: |s| < 1/(2*freq*dx) = 0.05 to avoid grating lobes.
        speed = 30.0
        data = _plane_wave_data(
            linear_array, freq=10.0, speed=speed, direction=0.0, fs=200.0, n_time=200
        )
        freqs = np.array([10.0])
        sx = np.linspace(-0.05, 0.05, 21)
        sy = np.linspace(-0.05, 0.05, 21)
        SX, SY = np.meshgrid(sx, sy, indexing="ij")
        sg = np.stack([SX, SY], axis=-1)
        beam = fk_spectrum(data, linear_array, freqs, sg, fs=200.0)
        idx = np.unravel_index(np.argmax(beam[0]), beam[0].shape)
        peak_sx = sx[idx[0]]
        expected_sx = 1.0 / speed  # ≈ 0.033
        assert (
            abs(peak_sx - expected_sx) < 0.01
        ), f"peak_sx={peak_sx}, expected={expected_sx}"


class TestCaponBeamformer:
    def test_output_shape(self, linear_array, slowness_grid):
        n_ch = linear_array.shape[0]
        csd = np.eye(n_ch, dtype=complex)[None, :, :]  # (1, N, N)
        freqs = np.array([10.0])
        beam = capon_beamformer(csd, linear_array, freqs, slowness_grid)
        assert beam.shape == (1, 31, 31)
        assert np.all(beam >= 0)

    def test_sharper_than_conventional(self, linear_array, slowness_grid):
        """Capon beamformer should have narrower peak than conventional."""
        speed = 15.0
        data = _plane_wave_data(
            linear_array, freq=10.0, speed=speed, direction=0.0, fs=200.0, n_time=200
        )
        freqs = np.array([10.0])
        beam_conv = fk_spectrum(data, linear_array, freqs, slowness_grid, fs=200.0)

        # Build CSD from data.
        from numpy.fft import rfft, rfftfreq

        F = rfft(data, axis=0)
        rf = rfftfreq(data.shape[0], 1.0 / 200.0)
        fi = np.argmin(np.abs(rf - 10.0))
        x = F[fi][:, None]  # (N, 1)
        csd = (x @ x.conj().T)[None]  # (1, N, N)
        beam_capon = capon_beamformer(csd, linear_array, freqs, slowness_grid)

        # Capon peak should be more concentrated.
        conv_norm = beam_conv[0] / beam_conv[0].max()
        capon_norm = beam_capon[0] / beam_capon[0].max()
        # Count cells above half-max.
        assert np.sum(capon_norm > 0.5) <= np.sum(conv_norm > 0.5)
