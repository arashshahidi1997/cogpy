"""
Module: bivariate_spectral
Status: REVIEW
Last Updated: 2025-08-26

Summary:
    Implements multitaper spectral estimation methods for cross-spectral and power spectral density analysis using DPSS tapers and FFTW.

Functions:
    multitaper_csd: Computes multitaper cross-spectral density (CSD) estimate for multichannel signals.
    multitaper_chd: Computes multitaper power spectral density (PSD) estimate for single-channel signals.

Classes:
    None

Constants:
    None

Example:

Status
------
STATUS: DEPRECATED
Reason: multitaper_csd and multitaper_chd both compute PSD not cross-spectrum; superseded by psd_multitaper in spectral/psd.py (forthcoming). pyfftw dependency adds overhead for no benefit.
Superseded by: cogpy.core.spectral.psd (forthcoming)
Safe to remove: yes
"""

from scipy import signal
import numpy as np
import pyfftw
from multiprocessing import cpu_count


def multitaper_csd(
    y,
    axis=-1,
    NW=2,
    nfft=None,
    K_max=None,
    fs=1,
    detrend=False,
    n_fft_threads=cpu_count(),
):
    """
    Parameters
    ----------
    y: array (..., ch, time)
    axis: sample axis along which Fourier transform is operated
    NW: half-bandwidth in units of ?
    K_max: maximum number of slepian tapers.
        if None, K_max is set equal to 2*len(y)*W-1

    Returns
    -------
    csd: same shape as y (with sample dim replaced by freq dim of the same size)
        cross-spectral density estimate using multitaper method
    """
    y = np.swapaxes(y, axis, -1)  # swap axes so the samples axis is the last dimension
    N = y.shape[-1]  # number of samples
    if K_max is None:
        K_max = int(2 * NW - 1)
    assert K_max > 0 and K_max < N / 2, print(
        "increase resolution `len(y)` or decrease  `NW`"
    )

    if detrend:
        y = signal.detrend(y, axis=-1)

    win = signal.windows.dpss(N, NW, Kmax=K_max)
    tapered_y = np.expand_dims(y, axis=-1) * win.T  # (..., sample_dim, taper_dim)

    if nfft is None:
        nfft = N
    if nfft > N:
        tapered_y = np.pad(tapered_y, (0, nfft - N))

    # psd, f = periodogram(tapered_y, fs=fs, freq_range=freq_range, axis=-2, detrend=detrend)
    f = np.fft.rfftfreq(nfft, d=1 / fs)
    tapered_y = np.moveaxis(tapered_y, -2, 0)
    shape_origin = tapered_y.shape[1:]
    tapered_y = tapered_y.reshape(tapered_y.shape[0], -1)
    psd_shape = (len(f), tapered_y.shape[-1])
    # psd_shape = tapered_y.shape[:-2] + f.shape + (tapered_y.shape[-1],)
    psd_ = pyfftw.zeros_aligned(psd_shape, dtype="complex128")
    pyfftw.FFTW(tapered_y, psd_, axes=(0,), threads=n_fft_threads).execute()
    # f, psd = sciperiodogram(tapered_y, fs=fs, axis=-2, detrend=False, scaling='spectrum')
    psd_ = np.abs(psd_) ** 2 / N
    psd_ = psd_.reshape(-1, *shape_origin)
    psd_ = np.mean(psd_, axis=-1)  # average across tapers # (nfreq, nch, nwin, ntaper)
    psd_ = np.moveaxis(psd_, 0, -1)  # nch, nwin, nfreq
    psd_ = np.swapaxes(psd_, axis, -1)  # swap axes to original shape of input
    return psd_, f


def multitaper_chd(
    y,
    axis=-1,
    NW=2,
    nfft=None,
    K_max=None,
    fs=1,
    detrend=False,
    n_fft_threads=cpu_count(),
):
    """
    Parameters
    ----------
    y: array
    axis: sample axis along which Fourier transform is operated
    NW: half-bandwidth in units of ?
    K_max: maximum number of slepian tapers.
        if None, K_max is set equal to 2*len(y)*W-1

    Returns
    -------
    psd: same shape as y (with sample dim replaced by freq dim of the same size)
        multitaper estimate of signal y
    """
    y = np.swapaxes(y, axis, -1)  # swap axes so the samples axis is the last dimension
    N = y.shape[-1]  # number of samples
    if K_max is None:
        K_max = int(2 * NW - 1)
    assert K_max > 0 and K_max < N / 2, print(
        "increase resolution `len(y)` or decrease  `NW`"
    )

    if detrend:
        y = signal.detrend(y, axis=-1)

    win = signal.windows.dpss(N, NW, Kmax=K_max)
    tapered_y = np.expand_dims(y, axis=-1) * win.T  # (..., sample_dim, taper_dim)

    if nfft is None:
        nfft = N
    if nfft > N:
        tapered_y = np.pad(tapered_y, (0, nfft - N))

    # psd, f = periodogram(tapered_y, fs=fs, freq_range=freq_range, axis=-2, detrend=detrend)
    f = np.fft.rfftfreq(nfft, d=1 / fs)
    tapered_y = np.moveaxis(tapered_y, -2, 0)
    shape_origin = tapered_y.shape[1:]
    tapered_y = tapered_y.reshape(tapered_y.shape[0], -1)
    psd_shape = (len(f), tapered_y.shape[-1])
    # psd_shape = tapered_y.shape[:-2] + f.shape + (tapered_y.shape[-1],)
    psd_ = pyfftw.zeros_aligned(psd_shape, dtype="complex128")
    pyfftw.FFTW(tapered_y, psd_, axes=(0,), threads=n_fft_threads).execute()
    # f, psd = sciperiodogram(tapered_y, fs=fs, axis=-2, detrend=False, scaling='spectrum')
    psd_ = np.abs(psd_) ** 2 / N
    psd_ = psd_.reshape(-1, *shape_origin)
    psd_ = np.mean(psd_, axis=-1)  # average across tapers # (nfreq, nch, nwin, ntaper)
    psd_ = np.moveaxis(psd_, 0, -1)  # nch, nwin, nfreq
    psd_ = np.swapaxes(psd_, axis, -1)  # swap axes to original shape of input
    return psd_, f
