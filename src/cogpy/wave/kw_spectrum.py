"""Wavenumber–frequency (k–ω) spectral analysis.

Estimates the 3-D power spectrum S(kx, ky, f) of gridded spatiotemporal
signals using windowed FFTs.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.signal.windows import hann

from ._types import Geometry, PatternType, WaveEstimate

__all__ = ["kw_spectrum_3d", "kw_peaks"]


def kw_spectrum_3d(
    data: xr.DataArray,
    geometry: Geometry,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> xr.DataArray:
    """Compute the 3-D wavenumber–frequency power spectrum.

    Parameters
    ----------
    data : DataArray
        Signal with dims ``(time, AP, ML)`` and ``fs`` coordinate.
    geometry : Geometry
        Regular grid geometry.
    nperseg : int, optional
        Temporal segment length (samples).  Defaults to the full time axis.
    noverlap : int, optional
        Overlap between temporal segments.  Defaults to ``nperseg // 2``.

    Returns
    -------
    DataArray
        Power spectrum with dims ``(freq, kx, ky)``.
    """
    if not geometry.is_regular:
        raise ValueError("kw_spectrum_3d requires a regular grid")

    fs = float(data.coords["fs"])
    vals = data.values  # (time, AP, ML)
    n_t, n_ap, n_ml = vals.shape

    if nperseg is None:
        nperseg = n_t
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap

    # Temporal windowing only — spatial axes are not windowed because the
    # grid typically covers the full array.  For sub-sampled arrays this
    # may introduce spatial spectral leakage.
    win_t = hann(nperseg, sym=False)

    # Accumulate Welch-like average.
    spectrum_sum = None
    n_seg = 0
    start = 0
    while start + nperseg <= n_t:
        seg = vals[start : start + nperseg]  # (nperseg, AP, ML)
        # Apply Hann window along time.
        seg = seg * win_t[:, None, None]
        # 3-D FFT.
        S = np.fft.fftn(seg, axes=(0, 1, 2))
        power = np.abs(S) ** 2
        if spectrum_sum is None:
            spectrum_sum = power
        else:
            spectrum_sum += power
        n_seg += 1
        start += step

    if spectrum_sum is None:
        raise ValueError("Signal too short for the given nperseg")

    spectrum = spectrum_sum / n_seg

    # Shift spatial axes to center zero-frequency.
    spectrum = np.fft.fftshift(spectrum, axes=(1, 2))
    # Keep only positive temporal frequencies.
    n_pos = nperseg // 2 + 1
    spectrum = spectrum[:n_pos]

    # Coordinate arrays.
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
    kx = np.fft.fftshift(np.fft.fftfreq(n_ap, geometry.dx))
    ky = np.fft.fftshift(np.fft.fftfreq(n_ml, geometry.dy))

    return xr.DataArray(
        spectrum,
        dims=("freq", "kx", "ky"),
        coords={"freq": freqs, "kx": kx, "ky": ky},
    )


def kw_peaks(
    spectrum: xr.DataArray,
    n_peaks: int = 1,
) -> list[WaveEstimate]:
    """Extract dominant peaks from a k–ω spectrum.

    Parameters
    ----------
    spectrum : DataArray
        Power spectrum with dims ``(freq, kx, ky)``.
    n_peaks : int
        Number of peaks to return.

    Returns
    -------
    list of WaveEstimate
    """
    vals = spectrum.values
    freqs = spectrum.coords["freq"].values
    kx_vals = spectrum.coords["kx"].values
    ky_vals = spectrum.coords["ky"].values

    # Exclude the DC bin (freq == 0).
    vals_work = vals.copy()
    vals_work[0, :, :] = 0.0

    results: list[WaveEstimate] = []
    for _ in range(n_peaks):
        idx = np.unravel_index(np.argmax(vals_work), vals_work.shape)
        f = freqs[idx[0]]
        kx = kx_vals[idx[1]]
        ky = ky_vals[idx[2]]
        k_mag = np.sqrt(kx**2 + ky**2)
        direction = np.arctan2(ky, kx)
        # fftfreq returns spatial frequency (cycles/unit), so phase
        # velocity = f / |k_spatial| (both in cycles-per-unit).
        speed = abs(f) / k_mag if k_mag > 1e-12 else 0.0
        wavelength = 1.0 / k_mag if k_mag > 1e-12 else None
        wavenumber = k_mag if k_mag > 1e-12 else None

        # Confidence from spectral peak prominence.
        peak_power = vals_work[idx]
        total_power = vals_work.sum()
        confidence = float(peak_power / total_power) if total_power > 0 else 0.0

        results.append(
            WaveEstimate(
                direction=float(direction),
                speed=float(speed),
                frequency=float(f),
                wavenumber=float(wavenumber) if wavenumber is not None else None,
                wavelength=float(wavelength) if wavelength is not None else None,
                pattern_type=PatternType.planar,
                confidence=confidence,
                fit_quality=confidence,
            )
        )
        # Zero-out neighbourhood to find next peak.
        vals_work[idx] = 0.0

    return results
