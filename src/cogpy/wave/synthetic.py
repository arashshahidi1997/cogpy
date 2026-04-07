"""Synthetic travelling-wave generation for testing."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import xarray as xr

from ._types import Geometry

__all__ = ["plane_wave", "spiral_wave", "wave_packet", "multi_wave"]


def _make_coords(
    n_time: int, n_ap: int, n_ml: int, fs: float, geometry: Geometry
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (time, ap_pos, ml_pos) coordinate arrays."""
    t = np.arange(n_time) / fs
    dx = geometry.dx if geometry.dx is not None else 1.0
    dy = geometry.dy if geometry.dy is not None else 1.0
    ap = np.arange(n_ap) * dx
    ml = np.arange(n_ml) * dy
    return t, ap, ml


def _wrap_da(
    data: np.ndarray,
    t: np.ndarray,
    ap: np.ndarray,
    ml: np.ndarray,
    fs: float,
) -> xr.DataArray:
    """Wrap a 3-D array as ``(time, AP, ML)`` DataArray with ``fs`` coord."""
    return xr.DataArray(
        data,
        dims=("time", "AP", "ML"),
        coords={"time": t, "AP": ap, "ML": ml, "fs": fs},
    )


def plane_wave(
    shape: tuple[int, int, int],
    geometry: Geometry,
    direction: float,
    speed: float,
    frequency: float,
    fs: float = 1000.0,
    noise_std: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> xr.DataArray:
    """Generate a plane travelling wave.

    Parameters
    ----------
    shape : (n_time, n_ap, n_ml)
        Output array shape.
    geometry : Geometry
        Spatial layout (regular grid).
    direction : float
        Propagation direction in radians (0 = positive-AP axis).
    speed : float
        Propagation speed in spatial-units / second.
    frequency : float
        Temporal frequency in Hz.
    fs : float
        Sampling rate in Hz.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    rng : Generator or int or None
        Random state for reproducibility.

    Returns
    -------
    xr.DataArray
        Shape ``(time, AP, ML)`` with ``fs`` coordinate.
    """
    n_time, n_ap, n_ml = shape
    t, ap, ml = _make_coords(n_time, n_ap, n_ml, fs, geometry)

    # Wave-vector components.
    kx = np.cos(direction) / speed  # s / spatial-unit
    ky = np.sin(direction) / speed

    # Phase at each (t, ap, ml).
    T = t[:, None, None]
    X = ap[None, :, None]
    Y = ml[None, None, :]
    phase = 2 * np.pi * frequency * (T - kx * X - ky * Y)
    data = np.cos(phase)

    if noise_std > 0:
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        data = data + rng.normal(0, noise_std, data.shape)

    return _wrap_da(data, t, ap, ml, fs)


def spiral_wave(
    shape: tuple[int, int, int],
    geometry: Geometry,
    center: tuple[float, float],
    angular_freq: float,
    fs: float = 1000.0,
    noise_std: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> xr.DataArray:
    """Generate a spiral (rotating) wave.

    Parameters
    ----------
    shape : (n_time, n_ap, n_ml)
        Output array shape.
    geometry : Geometry
        Spatial layout (regular grid).
    center : (float, float)
        Spiral center in (AP, ML) spatial units.
    angular_freq : float
        Angular frequency in rad / s  (temporal rotation rate).
    fs : float
        Sampling rate in Hz.
    noise_std : float
        Additive Gaussian noise std.
    rng : Generator or int or None
        Random state.

    Returns
    -------
    xr.DataArray
    """
    n_time, n_ap, n_ml = shape
    t, ap, ml = _make_coords(n_time, n_ap, n_ml, fs, geometry)

    T = t[:, None, None]
    X = ap[None, :, None] - center[0]
    Y = ml[None, None, :] - center[1]
    theta = np.arctan2(Y, X)
    phase = angular_freq * T + theta
    data = np.cos(phase)

    if noise_std > 0:
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        data = data + rng.normal(0, noise_std, data.shape)

    return _wrap_da(data, t, ap, ml, fs)


def wave_packet(
    shape: tuple[int, int, int],
    geometry: Geometry,
    direction: float,
    speed: float,
    frequency: float,
    sigma_t: float,
    sigma_x: float,
    fs: float = 1000.0,
    noise_std: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> xr.DataArray:
    """Generate a Gaussian-envelope wave packet.

    Parameters
    ----------
    shape : (n_time, n_ap, n_ml)
        Output array shape.
    geometry : Geometry
        Spatial layout.
    direction : float
        Propagation direction (radians).
    speed : float
        Group speed (spatial-units / s).
    frequency : float
        Carrier frequency (Hz).
    sigma_t : float
        Temporal envelope width (seconds).
    sigma_x : float
        Spatial envelope width (spatial units).
    fs : float
        Sampling rate.
    noise_std : float
        Additive noise.
    rng : Generator or int or None
        Random state.

    Returns
    -------
    xr.DataArray
    """
    n_time, n_ap, n_ml = shape
    t, ap, ml = _make_coords(n_time, n_ap, n_ml, fs, geometry)

    kx = np.cos(direction) / speed
    ky = np.sin(direction) / speed

    T = t[:, None, None]
    X = ap[None, :, None]
    Y = ml[None, None, :]

    # Center of the grid.
    cx = ap[len(ap) // 2]
    cy = ml[len(ml) // 2]
    ct = t[len(t) // 2]

    # Propagation coordinate.
    prop = np.cos(direction) * (X - cx) + np.sin(direction) * (Y - cy)
    t_delay = prop / speed

    env_t = np.exp(-((T - ct - t_delay) ** 2) / (2 * sigma_t**2))
    env_x = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma_x**2))

    phase = 2 * np.pi * frequency * (T - kx * X - ky * Y)
    data = env_t * env_x * np.cos(phase)

    if noise_std > 0:
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        data = data + rng.normal(0, noise_std, data.shape)

    return _wrap_da(data, t, ap, ml, fs)


def multi_wave(
    components: Sequence[xr.DataArray],
    noise_std: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> xr.DataArray:
    """Superpose multiple wave components.

    Parameters
    ----------
    components : sequence of DataArray
        Each must share the same coordinates.
    noise_std : float
        Additional noise on the superposition.
    rng : Generator or int or None
        Random state.

    Returns
    -------
    xr.DataArray
    """
    if not components:
        raise ValueError("Need at least one component")
    result = sum(components[1:], components[0])
    if noise_std > 0:
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        result = result + rng.normal(0, noise_std, result.shape)
    return result
