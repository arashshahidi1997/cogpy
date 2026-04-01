"""Phase-gradient analysis and plane-wave fitting.

Implements the phase-gradient directionality (PGD) framework from
Zhang et al. (2018) for characterizing travelling waves via spatial
gradients of instantaneous phase.

References
----------
.. [1] Zhang et al., "Theta and Alpha Oscillations Are Traveling Waves
   in the Human Neocortex", Neuron, 2018.
   DOI: 10.1016/j.neuron.2018.05.019

Ported from sayak66/wm_travelingwaves_code (phase gradient, PGD,
plane-wave fit).
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.signal import hilbert

from ._types import Geometry, PatternType, WaveEstimate

__all__ = ["hilbert_phase", "phase_gradient", "plane_wave_fit", "pgd"]


# ---------------------------------------------------------------------------
# Analytic phase
# ---------------------------------------------------------------------------


def hilbert_phase(data: xr.DataArray, axis: str = "time") -> xr.DataArray:
    """Compute unwrapped instantaneous phase via the Hilbert transform.

    Parameters
    ----------
    data : DataArray
        Real-valued signal with a *time* dimension.
    axis : str
        Dimension along which to apply the Hilbert transform.

    Returns
    -------
    DataArray
        Unwrapped phase in radians, same shape as *data*.
    """
    ax = data.dims.index(axis)
    analytic = hilbert(data.values, axis=ax)
    phase = np.unwrap(np.angle(analytic), axis=ax)
    return xr.DataArray(phase, dims=data.dims, coords=data.coords)


# ---------------------------------------------------------------------------
# Spatial gradient
# ---------------------------------------------------------------------------


def phase_gradient(
    phase: xr.DataArray,
    geometry: Geometry,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute the spatial phase gradient at each time step.

    For regular grids the gradient is computed with ``numpy.gradient``.
    For irregular arrays a per-time-step linear regression is used.

    Parameters
    ----------
    phase : DataArray
        Instantaneous phase with dims ``(time, AP, ML)`` (grid) or
        ``(time, ch)`` (irregular, requires ``geometry.coords``).
    geometry : Geometry
        Spatial layout.

    Returns
    -------
    dphi_dx, dphi_dy : DataArray
        Spatial gradient components along AP and ML axes.
    """
    if geometry.is_regular and {"AP", "ML"}.issubset(phase.dims):
        # Regular grid: finite differences.
        vals = phase.values  # (time, AP, ML)
        gx = np.gradient(vals, geometry.dx, axis=1)
        gy = np.gradient(vals, geometry.dy, axis=2)
        dphi_dx = xr.DataArray(gx, dims=phase.dims, coords=phase.coords)
        dphi_dy = xr.DataArray(gy, dims=phase.dims, coords=phase.coords)
        return dphi_dx, dphi_dy

    if geometry.coords is not None and "ch" in phase.dims:
        # Irregular array: per-frame linear regression  φ ≈ a*x + b*y + c
        coords = geometry.coords  # (N, 2)
        n_ch = coords.shape[0]
        A = np.column_stack([coords, np.ones(n_ch)])  # (N, 3)
        pinv = np.linalg.pinv(A)  # (3, N)
        vals = phase.values  # (time, ch)
        coeffs = vals @ pinv.T  # (time, 3)
        dphi_dx = xr.DataArray(
            coeffs[:, 0], dims=("time",), coords={"time": phase.coords["time"]}
        )
        dphi_dy = xr.DataArray(
            coeffs[:, 1], dims=("time",), coords={"time": phase.coords["time"]}
        )
        return dphi_dx, dphi_dy

    raise ValueError(
        "phase must have dims (time, AP, ML) with regular geometry or "
        "(time, ch) with irregular geometry"
    )


# ---------------------------------------------------------------------------
# Phase gradient directionality (PGD)
# ---------------------------------------------------------------------------


def pgd(phase: xr.DataArray, geometry: Geometry) -> xr.DataArray:
    """Phase-gradient directionality score per time step.

    Measures how consistently the spatial phase gradient points in a
    single direction, following Zhang et al. (2018) [1]_.  PGD is the
    resultant length of the unit gradient vectors across the grid.

    Parameters
    ----------
    phase : DataArray
        Instantaneous phase, dims ``(time, AP, ML)``.
    geometry : Geometry
        Regular grid geometry.

    Returns
    -------
    DataArray
        PGD values in [0, 1] for each time step.

    References
    ----------
    .. [1] Zhang et al., "Theta and Alpha Oscillations Are Traveling
       Waves in the Human Neocortex", Neuron, 2018.
       DOI: 10.1016/j.neuron.2018.05.019
    """
    dphi_dx, dphi_dy = phase_gradient(phase, geometry)

    gx = dphi_dx.values  # (time, AP, ML)
    gy = dphi_dy.values

    mag = np.sqrt(gx**2 + gy**2)
    mag = np.where(mag == 0, 1.0, mag)
    ux = gx / mag
    uy = gy / mag

    # Resultant length across spatial dims.
    spatial_axes = tuple(range(1, ux.ndim))
    n = np.prod([ux.shape[a] for a in spatial_axes])
    mean_x = np.sum(ux, axis=spatial_axes) / n
    mean_y = np.sum(uy, axis=spatial_axes) / n
    pgd_vals = np.sqrt(mean_x**2 + mean_y**2)

    return xr.DataArray(pgd_vals, dims=("time",), coords={"time": phase.coords["time"]})


# ---------------------------------------------------------------------------
# Plane-wave fit
# ---------------------------------------------------------------------------


def plane_wave_fit(
    phase: xr.DataArray,
    geometry: Geometry,
    freq: float | None = None,
) -> list[WaveEstimate]:
    """Fit a plane wave to each time frame of a phase map.

    For each frame the phase surface is fit as
    ``φ(x, y) ≈ k_x * x + k_y * y + c`` via least-squares.  Speed is
    ``ω / |k|`` where ``ω = 2π * freq`` (estimated from the temporal
    derivative of phase if *freq* is not given).

    Parameters
    ----------
    phase : DataArray
        Instantaneous phase, dims ``(time, AP, ML)``.
    geometry : Geometry
        Regular grid geometry.
    freq : float, optional
        Temporal frequency (Hz).  Estimated from the phase derivative
        if not supplied.

    Returns
    -------
    list of WaveEstimate
        One estimate per time step.

    References
    ----------
    .. [1] Zhang et al., "Theta and Alpha Oscillations Are Traveling
       Waves in the Human Neocortex", Neuron, 2018.
       DOI: 10.1016/j.neuron.2018.05.019
    """
    if not geometry.is_regular:
        raise ValueError("plane_wave_fit requires a regular grid geometry")

    vals = phase.values  # (time, AP, ML)
    n_t, n_ap, n_ml = vals.shape

    dx = geometry.dx
    dy = geometry.dy
    ap = np.arange(n_ap) * dx
    ml = np.arange(n_ml) * dy
    X, Y = np.meshgrid(ap, ml, indexing="ij")  # (AP, ML)
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    A = np.column_stack([X_flat, Y_flat, np.ones(X_flat.size)])
    pinv = np.linalg.pinv(A)

    # Estimate instantaneous frequency from phase derivative if needed.
    if freq is None and "fs" in phase.coords:
        fs = float(phase.coords["fs"])
        dphi_dt = np.gradient(vals, 1.0 / fs, axis=0)
        inst_freq = dphi_dt / (2 * np.pi)
    else:
        inst_freq = None

    results: list[WaveEstimate] = []
    for i in range(n_t):
        phi = vals[i].ravel()
        coeffs = pinv @ phi  # (dφ/dx, dφ/dy, c)
        # Phase gradient is -k for a wave φ = ωt - k·r, so negate to
        # recover the propagation direction.
        kx, ky = -coeffs[0], -coeffs[1]
        k_mag = np.sqrt(kx**2 + ky**2)
        direction = np.arctan2(ky, kx)

        # Frequency for this frame.
        f_i = (
            freq
            if freq is not None
            else (float(np.median(inst_freq[i])) if inst_freq is not None else 0.0)
        )

        omega = 2 * np.pi * abs(f_i)
        speed = omega / k_mag if k_mag > 1e-12 else 0.0
        wavelength = (2 * np.pi / k_mag) if k_mag > 1e-12 else None
        wavenumber = (k_mag / (2 * np.pi)) if k_mag > 1e-12 else None

        # Fit quality: fraction of variance explained by the plane.
        phi_hat = A @ coeffs
        ss_res = np.sum((phi - phi_hat) ** 2)
        ss_tot = np.sum((phi - phi.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
        fit_quality = float(np.clip(r2, 0, 1))

        results.append(
            WaveEstimate(
                direction=float(direction),
                speed=float(speed),
                frequency=float(f_i),
                wavenumber=float(wavenumber) if wavenumber is not None else None,
                wavelength=float(wavelength) if wavelength is not None else None,
                pattern_type=PatternType.planar,
                confidence=fit_quality,
                fit_quality=fit_quality,
            )
        )

    return results
