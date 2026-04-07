"""Frequency–wavenumber beamforming for irregular electrode arrays.

Implements conventional (delay-and-sum) and adaptive Capon (MVDR)
beamformers on cross-spectral density matrices.

References
----------
.. [1] Capon, "High-resolution frequency-wavenumber spectrum analysis",
   Proc. IEEE, 1969. DOI: 10.1109/PROC.1969.7278
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["fk_spectrum", "capon_beamformer"]


def _steering_vectors(
    coords: NDArray,
    freqs: NDArray,
    slowness_grid: NDArray,
) -> NDArray:
    """Build steering vectors for each (freq, slowness) pair.

    Parameters
    ----------
    coords : (N, 2)
        Sensor positions.
    freqs : (n_freq,)
        Frequencies in Hz.
    slowness_grid : (n_sx, n_sy, 2)
        Grid of (sx, sy) slowness values in s/m.

    Returns
    -------
    ndarray, shape (n_freq, n_sx, n_sy, N)
        Complex steering vectors.
    """
    # Steering vector: a_i(f,s) = exp(-j 2π f (sx·x_i + sy·y_i))
    # following Capon (1969).  coords (N,2), slowness (n_sx, n_sy, 2).
    # dot product -> (n_sx, n_sy, N)
    delay = np.einsum("ijk,lk->ijl", slowness_grid, coords)  # (n_sx, n_sy, N)
    # (n_freq, n_sx, n_sy, N)
    phase = -2 * np.pi * freqs[:, None, None, None] * delay[None, :, :, :]
    return np.exp(1j * phase)


def fk_spectrum(
    data: NDArray,
    coords: NDArray,
    freqs: NDArray,
    slowness_grid: NDArray,
    fs: float,
) -> NDArray:
    """Conventional (delay-and-sum) f–k beamformer.

    Parameters
    ----------
    data : (n_time, N)
        Multi-channel time series.
    coords : (N, 2)
        Sensor positions.
    freqs : (n_freq,)
        Frequencies of interest in Hz.
    slowness_grid : (n_sx, n_sy, 2)
        Slowness search grid.
    fs : float
        Sampling rate.

    Returns
    -------
    ndarray, shape (n_freq, n_sx, n_sy)
        Beam power.

    References
    ----------
    .. [1] Capon (1969), DOI: 10.1109/PROC.1969.7278
    """
    n_time, n_ch = data.shape
    # FFT of data.
    F = np.fft.rfft(data, axis=0)
    rfft_freqs = np.fft.rfftfreq(n_time, 1.0 / fs)

    # Find nearest FFT bin for each requested frequency.
    freq_idx = np.array([np.argmin(np.abs(rfft_freqs - f)) for f in freqs])
    F_sel = F[freq_idx]  # (n_freq, N)

    steer = _steering_vectors(coords, freqs, slowness_grid)  # (n_freq, n_sx, n_sy, N)
    # Beam power = |a^H * x|^2  (conventional)
    beam = np.abs(np.einsum("fijn,fn->fij", np.conj(steer), F_sel)) ** 2
    return beam / n_ch**2


def capon_beamformer(
    csd: NDArray,
    coords: NDArray,
    freqs: NDArray,
    slowness_grid: NDArray,
) -> NDArray:
    """Capon (MVDR) beamformer on a cross-spectral density matrix.

    Parameters
    ----------
    csd : (n_freq, N, N)
        Cross-spectral density matrices.
    coords : (N, 2)
        Sensor positions.
    freqs : (n_freq,)
        Frequencies matching the first axis of *csd*.
    slowness_grid : (n_sx, n_sy, 2)
        Slowness search grid.

    Returns
    -------
    ndarray, shape (n_freq, n_sx, n_sy)
        MVDR beam power.

    References
    ----------
    .. [1] Capon, "High-resolution frequency-wavenumber spectrum
       analysis", Proc. IEEE, 1969. DOI: 10.1109/PROC.1969.7278
    """
    n_freq, n_ch, _ = csd.shape
    steer = _steering_vectors(coords, freqs, slowness_grid)
    n_sx, n_sy = slowness_grid.shape[0], slowness_grid.shape[1]

    beam = np.empty((n_freq, n_sx, n_sy))
    for fi in range(n_freq):
        # Regularise CSD for inversion.
        R = csd[fi]
        R_reg = R + np.eye(n_ch) * np.trace(R).real * 1e-6
        R_inv = np.linalg.inv(R_reg)
        for si in range(n_sx):
            for sj in range(n_sy):
                a = steer[fi, si, sj]  # (N,)
                denom = np.real(a.conj() @ R_inv @ a)
                beam[fi, si, sj] = 1.0 / denom if denom > 1e-30 else 0.0

    return beam
