"""
PSD estimators with unified return convention.

Status
------
STATUS: ACTIVE
Reason: Unified (psd, freqs) interface over Welch and multitaper paths.
Superseded by: n/a
Safe to remove: no

All functions:
    arr:   (..., time)  input signal
    psd:   (..., freq)  float64, one-sided power spectral density
    freqs: (freq,)      float64, Hz, strictly increasing
"""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from .multitaper import multitaper_fft

EPS = 1e-12

__all__ = [
    "psd_welch",
    "psd_multitaper",
    "psd_from_mtfft",
]


def _clip_freq(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float | None):
    if fmax is None:
        fmax = float(freqs[-1])
    fmin = float(fmin)
    fmax = float(fmax)
    if fmin >= fmax:
        raise ValueError(f"Invalid frequency clip range: fmin={fmin} >= fmax={fmax}")
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        raise ValueError(
            f"Frequency clip removed all bins: fmin={fmin}, fmax={fmax}, "
            f"freqs=[{float(freqs[0])}, {float(freqs[-1])}]"
        )
    return psd[..., mask], freqs[mask]


def psd_welch(
    arr,
    fs,
    *,
    fmin=0.0,
    fmax=None,
    nperseg=256,
    noverlap=None,
    axis=-1,
    detrend="constant",
):
    """
    Welch PSD estimate.

    Parameters
    ----------
    arr : (..., time)
    fs : float — sampling rate in Hz
    fmin, fmax : float — frequency clip range in Hz; fmax=None uses Nyquist
    nperseg : int — segment length
    noverlap : int — overlap samples; defaults to nperseg // 2
    axis : int — time axis
    detrend : str — passed to scipy.signal.welch

    Returns
    -------
    psd : (..., freq)
    freqs : (freq,)

    Examples
    --------
    >>> arr = np.random.randn(4, 2000)
    >>> psd, freqs = psd_welch(arr, fs=1000)
    >>> psd.shape  # (4, freq)
    """
    if noverlap is None:
        noverlap = int(nperseg) // 2
    freqs, pxx = scipy_signal.welch(
        arr,
        fs=float(fs),
        nperseg=int(nperseg),
        noverlap=int(noverlap),
        axis=int(axis),
        detrend=detrend,
        return_onesided=True,
        scaling="density",
    )
    psd = np.asarray(pxx, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    return _clip_freq(psd, freqs, fmin, fmax)


def psd_multitaper(
    arr,
    fs,
    *,
    NW=4,
    K=None,
    fmin=0.0,
    fmax=None,
    detrend=True,
    axis=-1,
):
    """
    Multitaper PSD estimate using DPSS tapers.

    Calls multitaper_fft() internally. Averages |FFT|^2 across tapers
    and normalizes by (fs * N).

    Parameters
    ----------
    arr : (..., time)
    fs : float — sampling rate in Hz
    NW : float — time-bandwidth product; default 4
    K : int — number of tapers; defaults to int(2 * NW - 1)
    fmin, fmax : float — frequency clip range; fmax=None uses Nyquist
    detrend : bool — remove linear trend before tapering
    axis : int — time axis

    Returns
    -------
    psd : (..., freq)
    freqs : (freq,)

    Examples
    --------
    >>> arr = np.random.randn(4, 2000)
    >>> psd, freqs = psd_multitaper(arr, fs=1000, NW=4)
    >>> psd.shape  # (4, freq)
    """
    arr = np.asarray(arr)
    axis = int(axis)
    N = int(arr.shape[axis])
    nfft = N

    if K is None:
        K = int(2 * float(NW) - 1)
    mtfft = multitaper_fft(
        arr,
        axis=axis,
        NW=float(NW),
        nfft=int(nfft),
        K_max=int(K),
        detrend=bool(detrend),
    )
    freqs = np.fft.rfftfreq(int(nfft), d=1.0 / float(fs)).astype(np.float64, copy=False)
    psd = (np.abs(mtfft) ** 2).mean(axis=-2) / (float(fs) * float(N))
    psd = np.asarray(psd, dtype=np.float64)
    return _clip_freq(psd, freqs, fmin, fmax)


def psd_from_mtfft(mtfft, freqs, fs, N, *, fmin=0.0, fmax=None):
    """
    Derive averaged PSD from pre-computed tapered FFTs.

    Use when mtfft is already computed (e.g. for cross-spectrum)
    to avoid redundant FFT computation.

    Parameters
    ----------
    mtfft : (..., taper, freq) complex — from multitaper_fft()
    freqs : (freq,) — frequency axis from multitaper_fft()
    fs : float — sampling rate in Hz
    N : int — number of time samples in original signal
    fmin, fmax : float — frequency clip range

    Returns
    -------
    psd : (..., freq)
    freqs_clipped : (freq,)

    Examples
    --------
    >>> mtfft, freqs = multitaper_fft(arr, NW=4)
    >>> psd, freqs = psd_from_mtfft(mtfft, freqs, fs=1000, N=arr.shape[-1])
    """
    mtfft = np.asarray(mtfft)
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = (np.abs(mtfft) ** 2).mean(axis=-2) / (float(fs) * float(N))
    psd = np.asarray(psd, dtype=np.float64)
    return _clip_freq(psd, freqs, fmin, fmax)
