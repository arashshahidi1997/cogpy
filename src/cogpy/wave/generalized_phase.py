"""Broadband generalized-phase estimation.

Computes a stabilised instantaneous phase for wideband signals by
centering the analytic representation and correcting negative-frequency
contamination.

Adapted from mullerlab/generalized-phase [1]_.  Our implementation
uses a simplified frequency-domain approach (centre + re-zero negatives)
rather than the time-domain IF-epoch detection with PCHIP interpolation
used in the original MATLAB code.  This is adequate for well-sampled
signals; for noisy recordings with strong negative-frequency epochs,
the full MATLAB pipeline may be more robust.

References
----------
.. [1] Davis et al., "Spontaneous travelling cortical waves gate
   perception in behaving primates", Nature, 2020.
   DOI: 10.1038/s41586-020-2802-y
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.signal import hilbert

__all__ = ["generalized_phase"]


def generalized_phase(
    data: xr.DataArray,
    axis: str = "time",
) -> xr.DataArray:
    """Compute broadband generalized phase.

    Unlike narrowband analytic-signal phase, this method centres the
    complex representation to reduce bias from broadband spectral
    content, yielding stable phase estimates even for wideband signals.

    Parameters
    ----------
    data : DataArray
        Real-valued signal with a *time* dimension.
    axis : str
        Dimension along which to compute phase.

    Returns
    -------
    DataArray
        Instantaneous phase in radians (unwrapped), same shape as *data*.

    References
    ----------
    .. [1] Davis et al., "Spontaneous travelling cortical waves gate
       perception in behaving primates", Nature, 2020.
       DOI: 10.1038/s41586-020-2802-y
    """
    ax = data.dims.index(axis)
    vals = data.values.astype(np.float64)

    # Step 1: analytic signal.
    analytic = hilbert(vals, axis=ax)

    # Step 2: centre the complex representation per channel.
    # This removes the DC offset in both real and imaginary parts,
    # stabilising phase estimates for broadband signals.
    mean_real = np.mean(analytic.real, axis=ax, keepdims=True)
    mean_imag = np.mean(analytic.imag, axis=ax, keepdims=True)
    analytic = analytic - mean_real - 1j * mean_imag

    # Step 3: re-enforce one-sided spectrum after centering.
    # Centering (Step 2) only modifies the DC bin, so negative frequencies
    # should still be ≈0 from the initial hilbert().  We re-zero them as a
    # guard but do NOT re-double positive frequencies — they were already
    # doubled by scipy.signal.hilbert.
    #
    # Note: the MATLAB mullerlab reference uses a different approach here —
    # it computes instantaneous frequency (IF) via
    #   angle(z(t+1) * conj(z(t))) / (2π·dt),
    # detects epochs where IF < 0, and fills them with PCHIP interpolation.
    # Our frequency-domain zeroing is a simpler approximation that is
    # adequate for well-sampled signals but does not correct individual
    # negative-IF epochs the way the MATLAB code does.
    F = np.fft.fft(analytic, axis=ax)
    n = vals.shape[ax]
    slices_neg = [slice(None)] * vals.ndim
    slices_neg[ax] = slice(n // 2 + 1, None)
    F[tuple(slices_neg)] = 0.0
    analytic = np.fft.ifft(F, axis=ax)

    phase = np.unwrap(np.angle(analytic), axis=ax)
    return xr.DataArray(phase.real, dims=data.dims, coords=data.coords)
