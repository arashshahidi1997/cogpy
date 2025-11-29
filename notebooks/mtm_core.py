"""Simple single-channel multitaper FFT-gram primitive and high-level wrappers.

This module splits the original ghostipy mtm_spectrogram logic into a
low-level mtm_fftgram that computes tapered, windowed FFTs for a single
time series, and higher-level functions that build on it:

- mtm_fftgram: returns tapered FFTs (per taper, per segment, per freq)
- mtm_spectrogram: single-channel multitaper spectrogram (PSD) using mtm_fftgram
- mtm_cross_spectrum: weighted multitaper cross-spectrum between two signals
- mtm_coherence: magnitude-squared coherence (and complex coherency if requested)

The implementation uses sliding_window_view (fall back to as_strided),
pyfftw for batched FFTs (aligned arrays and multi-threading), and
get_tapers from ghostipy.spectral.mtm to compute DPSS tapers and lambdas.

API notes
- Input signals are 1-D arrays (time).
- Time axis handling for timestamps is minimal: if timestamps is None, timestamps are
  inferred as np.arange(len(x))/fs, and segment timestamps are midpoints of each window.
- The module intentionally focuses on single-channel core primitives; multi-channel
  handling can be built on top by looping or batching calls to mtm_fftgram.

Author: ChatGPT (as requested)
"""
from typing import Optional, Tuple
import numpy as np
from numpy.lib.stride_tricks import as_strided
from multiprocessing import cpu_count
import pyfftw

# Import the taper helper from ghostipy as requested
from ghostipy.spectral.mtm import get_tapers

# sliding_window_view if available
try:
    from numpy.lib.stride_tricks import sliding_window_view  # type: ignore
    _HAS_SLIDING = True
except Exception:
    sliding_window_view = None
    _HAS_SLIDING = False


def _sliding_windows_1d(x: np.ndarray, window_length: int, step: int) -> np.ndarray:
    """
    Return an overlapping-window view of 1D array x with windows of length
    `window_length` and step `step` between window starts.

    The returned array has shape (n_segments, window_length). No copy if possible.
    """
    L = x.shape[0]
    if window_length > L:
        raise ValueError("window_length cannot be larger than the signal length")
    if step < 1:
        raise ValueError("step must be >= 1")
    # number of segments according to original mtm convention:
    n_segments = (L - (window_length - step)) // step
    if n_segments < 1:
        raise ValueError("Not enough data points for requested window/overlap")
    if _HAS_SLIDING:
        all_windows = sliding_window_view(x, window_length)  # shape (L - window_length + 1, window_length)
        starts = np.arange(0, n_segments * step, step)
        return all_windows[starts, :]
    else:
        shape = (n_segments, window_length)
        stride = x.strides[0]
        strides = (step * stride, stride)
        return as_strided(x, shape=shape, strides=strides, writeable=False)


def mtm_fftgram(
    data: np.ndarray,
    bandwidth: float,
    *,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    n_tapers: Optional[int] = None,
    min_lambda: float = 0.95,
    remove_mean: bool = False,
    nfft: Optional[int] = None,
    n_fft_threads: int = cpu_count()
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute tapered FFTs for a single-channel time series (the "fft-gram").

    Returns:
      X : complex ndarray, shape (n_tapers, n_segments, n_freqs)
          The tapered FFTs for each taper (k), segment (t), and frequency (f).
      freqs : ndarray, shape (n_freqs,)
          Frequency vector corresponding to FFT output.
      times : ndarray, shape (n_segments,)
          Midpoint timestamps for each segment (in seconds).
      lambdas : ndarray, shape (n_tapers,)
          The energy concentrations for each taper (useful for weighting).

    Parameters mirror the previous mtm_spectrogram but focus on single-channel.
    """
    if data.ndim != 1:
        raise ValueError("mtm_fftgram expects a 1-D time series.")

    N = data.shape[0]
    if nperseg is None:
        nperseg = 256
    if noverlap is None:
        noverlap = nperseg // 8
    if noverlap >= nperseg:
        raise ValueError("noverlap must be less than nperseg")
    step = nperseg - noverlap
    if step <= 0:
        raise ValueError("step must be positive")

    if nperseg > N:
        raise ValueError("'nperseg' cannot be larger than the data size")

    if remove_mean:
        data = data - data.mean()

    # compute tapers and lambdas using ghostipy helper
    tapers, lambdas = get_tapers(nperseg, bandwidth, fs=fs, n_tapers=n_tapers, min_lambda=min_lambda)
    n_tapers = tapers.shape[0]

    if nfft is None:
        nfft = nperseg
    if nfft < nperseg:
        raise ValueError("'nfft' must be at least nperseg")

    # get windows view: (n_segments, nperseg)
    windows = _sliding_windows_1d(data, nperseg, step)  # (n_segments, nperseg)
    n_segments = windows.shape[0]

    # timestamps: midpoint of each window
    times = (np.arange(n_segments) * step + (nperseg / 2.0)) / fs

    # prepare FFT output shape
    is_real = np.isrealobj(data)
    if is_real:
        nfreqs = nfft // 2 + 1
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    else:
        nfreqs = nfft
        freqs = np.fft.fftfreq(nfft, d=1.0 / fs)

    # prepare pyFFTW arrays and perform batched FFTs:
    # xtd shape (n_tapers, n_segments, nfft) for real input -> rfft output shape (n_tapers, n_segments, nfreqs)
    if is_real:
        xtd = pyfftw.zeros_aligned((n_tapers, n_segments, nfft), dtype='float64')
        xfd = pyfftw.zeros_aligned((n_tapers, n_segments, nfreqs), dtype='complex128')
        fft_obj = pyfftw.FFTW(
            xtd, xfd,
            axes=(2,),
            direction='FFTW_FORWARD',
            flags=['FFTW_ESTIMATE'],
            threads=n_fft_threads,
            planning_timelimit=0
        )
        # apply tapers and copy into aligned buffer
        # shapes: tapers (n_tapers, nperseg), windows (n_segments, nperseg)
        # want (n_tapers, n_segments, nperseg) via broadcasting:
        xtd[:, :, :nperseg] = tapers[:, None, :] * windows[None, :, :]
        if nfft > nperseg:
            xtd[:, :, nperseg:] = 0.0
        fft_obj(normalise_idft=True)
        X = xfd.copy()
    else:
        x = pyfftw.zeros_aligned((n_tapers, n_segments, nfft), dtype='complex128')
        fft_obj = pyfftw.FFTW(
            x, x,
            axes=(2,),
            direction='FFTW_FORWARD',
            flags=['FFTW_ESTIMATE'],
            threads=n_fft_threads,
            planning_timelimit=0
        )
        x[:, :, :nperseg] = tapers[:, None, :] * windows[None, :, :]
        if nfft > nperseg:
            x[:, :, nperseg:] = 0
        fft_obj(normalise_idft=True)
        X = x.copy()

    # X shape: (n_tapers, n_segments, nfreqs)
    return X, freqs, times, lambdas


def mtm_spectrogram(
    data: np.ndarray,
    bandwidth: float,
    *,
    fs: float = 1.0,
    timestamps: Optional[np.ndarray] = None,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    n_tapers: Optional[int] = None,
    min_lambda: float = 0.95,
    remove_mean: bool = False,
    nfft: Optional[int] = None,
    n_fft_threads: int = cpu_count()
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute single-channel multitaper spectrogram (PSD) using mtm_fftgram.

    Returns:
      S : ndarray, shape (n_freqs, n_timepoints)
          Multitaper spectrogram (PSD)
      freqs : ndarray
      times : ndarray

    This preserves the previous mtm_spectrogram output shape and scaling.
    """
    N = data.shape[0]
    if timestamps is None:
        timestamps = np.arange(N) / fs
    if timestamps.shape[0] != N:
        raise ValueError("timestamps length must match data length")

    # call mtm_fftgram to compute tapered FFTs
    X, freqs, times, lambdas = mtm_fftgram(
        data,
        bandwidth,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        n_tapers=n_tapers,
        min_lambda=min_lambda,
        remove_mean=remove_mean,
        nfft=nfft,
        n_fft_threads=n_fft_threads
    )

    # compute PSD per taper: shape (n_tapers, n_segments, nfreqs)
    sdfs = (X.real ** 2 + X.imag ** 2) / fs

    # frequency scaling for real input: double non-DC and non-Nyquist bins
    if np.isrealobj(data):
        if (nfft or (nperseg if nperseg is not None else 256)) % 2 == 0:
            # even nfft: bins 1..-2 doubled
            sdfs[:, :, 1:-1] *= 2.0
        else:
            sdfs[:, :, 1:] *= 2.0

    # combine tapers using lambdas weights
    w = lambdas[:, None, None]
    S = np.sum(w * sdfs, axis=0) / np.sum(lambdas)  # shape (n_segments, nfreqs)

    # transpose to (n_freqs, n_timepoints) like original
    S = S.T

    return S, freqs, times


def mtm_cross_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float,
    *,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    n_tapers: Optional[int] = None,
    min_lambda: float = 0.95,
    remove_mean: bool = False,
    nfft: Optional[int] = None,
    n_fft_threads: int = cpu_count()
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the weighted multitaper cross-spectrum Sxy between two real/complex signals x and y.

    Returns:
      Sxy : complex ndarray, shape (n_freqs, n_timepoints)
      freqs, times
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    # compute tapered FFTs for both
    X, freqs, times, lambdas = mtm_fftgram(
        x, bandwidth, fs=fs, nperseg=nperseg, noverlap=noverlap,
        n_tapers=n_tapers, min_lambda=min_lambda, remove_mean=remove_mean,
        nfft=nfft, n_fft_threads=n_fft_threads
    )
    Y, _, _, _ = mtm_fftgram(
        y, bandwidth, fs=fs, nperseg=nperseg, noverlap=noverlap,
        n_tapers=n_tapers, min_lambda=min_lambda, remove_mean=remove_mean,
        nfft=nfft, n_fft_threads=n_fft_threads
    )

    # X, Y shape: (n_tapers, n_segments, nfreqs)
    # compute weighted cross-spectrum: sum_k lambda_k * conj(X_k) * Y_k / sum(lambda)
    w = lambdas[:, None, None]
    Sxy = np.sum(w * (np.conjugate(X) * Y), axis=0) / np.sum(lambdas)  # shape (n_segments, nfreqs)
    Sxy = Sxy.T  # shape (nfreqs, n_timepoints)

    return Sxy, freqs, times


def mtm_coherence(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float,
    *,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    n_tapers: Optional[int] = None,
    min_lambda: float = 0.95,
    remove_mean: bool = False,
    nfft: Optional[int] = None,
    n_fft_threads: int = cpu_count(),
    return_complex: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute multitaper coherence between x and y.

    Returns:
      C2 : ndarray, shape (n_freqs, n_timepoints)
          Magnitude-squared coherence |Sxy|^2 / (Sxx*Syy)
      freqs, times

    If return_complex=True returns the complex coherency (Sxy / sqrt(Sxx*Syy)).
    """
    # compute tapered FFTs
    X, freqs, times, lambdas = mtm_fftgram(
        x, bandwidth, fs=fs, nperseg=nperseg, noverlap=noverlap,
        n_tapers=n_tapers, min_lambda=min_lambda, remove_mean=remove_mean,
        nfft=nfft, n_fft_threads=n_fft_threads
    )
    Y, _, _, _ = mtm_fftgram(
        y, bandwidth, fs=fs, nperseg=nperseg, noverlap=noverlap,
        n_tapers=n_tapers, min_lambda=min_lambda, remove_mean=remove_mean,
        nfft=nfft, n_fft_threads=n_fft_threads
    )

    # autospectra and cross-spectrum (weighted)
    w = lambdas[:, None, None]
    Sxx = np.sum(w * (np.conjugate(X) * X), axis=0) / np.sum(lambdas)  # shape (n_segments, nfreqs)
    Syy = np.sum(w * (np.conjugate(Y) * Y), axis=0) / np.sum(lambdas)
    Sxy = np.sum(w * (np.conjugate(X) * Y), axis=0) / np.sum(lambdas)

    # transpose to (nfreqs, n_timepoints)
    Sxx = Sxx.T
    Syy = Syy.T
    Sxy = Sxy.T

    denom = np.sqrt(Sxx * Syy)
    # avoid division by zero
    denom_safe = denom.copy()
    denom_safe[denom_safe == 0] = np.finfo(float).eps

    coherency = Sxy / denom_safe
    C2 = np.abs(coherency) ** 2

    if return_complex:
        return coherency, freqs, times
    else:
        return C2, freqs, times