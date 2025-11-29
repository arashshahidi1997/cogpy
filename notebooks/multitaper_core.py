"""
Standalone compute_tapered_ffts primitive and mtm_spectrogram refactor.

Depends on:
- numpy
- pyfftw
- ghostipy.spectral.mtm.get_tapers (to compute tapers and lambdas when needed)

Design:
- Accepts data with arbitrary leading dimensions; time axis defaults to -1.
- Uses sliding_window_view (fallback to as_strided) to create overlapping windows without full copies.
- Produces batched FFTs using pyFFTW aligned arrays; uses rfft for real data.
- Returns tapered FFTs shaped (n_tapers, *channel_shape, n_segments, n_freqs).

Example:
    from standalone_tapered_fft import compute_tapered_ffts, mtm_spectrogram
    tapers, lambdas = get_tapers(256, bandwidth=4.0, fs=1000.0)
    X, freqs, times = compute_tapered_ffts(data, tapers, nfft=512, noverlap=128, fs=1000.0)
    S, freqs, times = mtm_spectrogram(data, bandwidth=4.0, fs=1000.0, nperseg=256, noverlap=128)
"""
from typing import Tuple, Optional
import numpy as np
import pyfftw
from numpy.lib.stride_tricks import as_strided

# Import helper from ghostipy (as requested)
from ghostipy.spectral.mtm import get_tapers

# Prefer sliding_window_view when available
try:
    from numpy.lib.stride_tricks import sliding_window_view
    _HAS_SLIDING = True
except Exception:
    sliding_window_view = None
    _HAS_SLIDING = False


def _sliding_windows_subsampled(x: np.ndarray, window_length: int, step: int):
    """
    Produce windowed view of x along last axis with given window_length and step between windows.
    Returns view with shape (..., n_segments, window_length).
    Uses sliding_window_view when available (then subsamples), otherwise falls back to as_strided.
    """
    L = x.shape[-1]
    if window_length > L:
        raise ValueError("window_length cannot be larger than the length of the last axis")
    if step < 1:
        raise ValueError("step must be >= 1")

    # number of segments following the original mtm code convention:
    n_segments = (L - (window_length - step)) // step  # equivalent to (L - noverlap)//step where step = window_length - noverlap

    if _HAS_SLIDING:
        # sliding_window_view produces all windows with step=1, then take every 'step' window
        all_w = sliding_window_view(x, window_shape=window_length, axis=-1)  # shape (..., L - window_length + 1, window_length)
        starts = np.arange(0, n_segments * step, step)
        # take subsampled windows along the penultimate axis
        return all_w[..., starts, :]
    else:
        # fallback: construct as_strided view matching step
        # compute shape and strides consistent with the original mtm implementation
        shape = x.shape[:-1] + (n_segments, window_length)
        stride = x.strides[-1]
        strides = x.strides[:-1] + (step * stride, stride)
        return as_strided(x, shape=shape, strides=strides, writeable=False)


def compute_tapered_ffts(
    data: np.ndarray,
    tapers: np.ndarray,
    *,
    nfft: Optional[int] = None,
    noverlap: Optional[int] = None,
    fs: float = 1.0,
    time_axis: int = -1,
    n_fft_threads: int = 1,
    chunk_channels: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute tapered, windowed FFTs for data with arbitrary leading channel dims.

    Parameters
    ----------
    data : ndarray
        Input array. Time is by default the last axis. Leading axes are treated as channels
        (they are preserved in the output shape).
    tapers : ndarray, shape (n_tapers, nperseg)
        Tapers to apply to each window.
    nfft : int, optional
        Number of FFT points. Default is nperseg.
    noverlap : int, optional
        Number of points overlapped between windows. Default is nperseg // 8.
    fs : float
        Sampling frequency for frequency axis calculation.
    time_axis : int
        Axis index for the time axis in `data`. Defaults to -1.
    n_fft_threads : int
        Number of threads to use for pyFFTW.
    chunk_channels : int
        If > 0, process leading-channel dims in chunks of this many flattened channels to limit peak memory usage.

    Returns
    -------
    X : complex ndarray, shape (n_tapers, *channel_shape, n_segments, n_freqs)
        Tapered FFTs for each taper, channel, segment and frequency.
    freqs : ndarray, shape (n_freqs,)
        Frequency vector corresponding to FFT output.
    starts : ndarray, shape (n_segments,)
        Start sample indices for each segment (useful for constructing timestamps).
    """
    # move time axis to last position
    data = np.moveaxis(data, time_axis, -1)
    orig_shape = data.shape  # (*channel_shape, L)
    L = orig_shape[-1]
    channel_shape = orig_shape[:-1]
    n_channels_flat = int(np.prod(channel_shape)) if channel_shape else 1
    if n_channels_flat == 1:
        # ensure we have a leading dim for uniform handling
        data_flat = data.reshape((1, L))
    else:
        data_flat = data.reshape((n_channels_flat, L))

    n_tapers, nperseg = tapers.shape

    if noverlap is None:
        noverlap = nperseg // 8
    if noverlap >= nperseg:
        raise ValueError("noverlap must be less than nperseg")

    step = nperseg - noverlap
    if step <= 0:
        raise ValueError("step (nperseg - noverlap) must be positive")

    if nfft is None:
        nfft = nperseg

    # compute number of segments consistent with original mtm_spectrogram
    n_segments = (L - noverlap) // step
    if n_segments < 1:
        raise ValueError("Not enough data points for even one segment with the given nperseg/noverlap")

    # compute starts array (indices of window starts)
    starts = (np.arange(n_segments) * step).astype(int)

    # window the flattened data: result shape (n_channels_flat, n_segments, nperseg)
    windows = _sliding_windows_subsampled(data_flat, nperseg, step)  # full windows with step already subsampled
    # select only first n_segments to be safe (sliding_window may produce slightly more)
    windows = windows[..., :n_segments, :]

    # Determine FFT sizes and output freq axis
    is_real = np.isrealobj(data_flat)
    if is_real:
        M = nfft // 2 + 1
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    else:
        M = nfft
        freqs = np.fft.fftfreq(nfft, d=1.0 / fs)

    # We'll reshape channels in chunks to limit memory if requested
    def _process_channel_chunk(chunk_slice):
        cstart, cend = chunk_slice
        w_chunk = windows[cstart:cend]  # shape (n_chunk, n_segments, nperseg)
        n_chunk = w_chunk.shape[0]

        if is_real:
            # prepare aligned real->complex rfft buffers
            xtd = pyfftw.zeros_aligned((n_tapers, n_chunk, n_segments, nfft), dtype='float64')
            xfd = pyfftw.zeros_aligned((n_tapers, n_chunk, n_segments, M), dtype='complex128')
            fft_obj = pyfftw.FFTW(
                xtd, xfd,
                axes=(3,),
                direction='FFTW_FORWARD',
                flags=['FFTW_ESTIMATE'],
                threads=n_fft_threads,
                planning_timelimit=0)
            # fill tapered windows
            # (n_tapers, n_chunk, n_segments, nperseg) = (n_tapers, 1, 1, nperseg) * (1, n_chunk, n_segments, nperseg)
            xtd[:, :, :, :nperseg] = (tapers[:, None, None, :] * w_chunk[None, :, :, :])
            if nfft > nperseg:
                xtd[:, :, :, nperseg:] = 0.0
            fft_obj(normalise_idft=True)
            return xfd.copy()  # shape (n_tapers, n_chunk, n_segments, M)
        else:
            x = pyfftw.zeros_aligned((n_tapers, n_chunk, n_segments, nfft), dtype='complex128')
            fft_obj = pyfftw.FFTW(
                x, x,
                axes=(3,),
                direction='FFTW_FORWARD',
                flags=['FFTW_ESTIMATE'],
                threads=n_fft_threads,
                planning_timelimit=0)
            # multiply and place
            x[:, :, :, :nperseg] = (tapers[:, None, None, :] * w_chunk[None, :, :, :])
            if nfft > nperseg:
                x[:, :, :, nperseg:] = 0
            fft_obj(normalise_idft=True)
            return x.copy()  # shape (n_tapers, n_chunk, n_segments, M)

    # Prepare channel chunks
    if chunk_channels and chunk_channels > 0:
        chunk_size = int(chunk_channels)
    else:
        chunk_size = n_channels_flat

    X_chunks = []
    channel_slices = []
    for cstart in range(0, n_channels_flat, chunk_size):
        cend = min(cstart + chunk_size, n_channels_flat)
        Xc = _process_channel_chunk((cstart, cend))  # shape (n_tapers, n_chunk, n_segments, M)
        X_chunks.append(Xc)
        channel_slices.append((cstart, cend))

    # concatenate back along channel axis
    X_concat = np.concatenate(X_chunks, axis=1)  # shape (n_tapers, n_channels_flat, n_segments, M)

    # reshape channel dims back to original channel_shape
    if channel_shape:
        out_shape = (n_tapers,) + channel_shape + (n_segments, M)
        X = X_concat.reshape(out_shape)
    else:
        # single-channel original
        X = X_concat.reshape((n_tapers, 1, n_segments, M))
        X = X[:, 0:1, :, :]  # keep a singleton channel dimension for consistency

    # If original had no channel dims, remove the singleton channel axis to mirror input 1D behavior where appropriate.
    if not channel_shape:
        # return shape (n_tapers, n_segments, M) for single-channel input (but keep channels last in general functions expect channels preserved)
        X = X.reshape((n_tapers, n_segments, M))

    return X, freqs, starts


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
    n_fft_threads: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multitaper spectrogram that accepts data with leading channel dimensions.

    Parameters largely mirror the original mtm_spectrogram, but `data` may have
    leading dimensions (channels). The time axis is assumed to be the last axis.

    Returns
    -------
    S : ndarray
        Spectrogram with shape (*channel_shape, n_freqs, n_timepoints)
    freqs : ndarray
    out_timestamps : ndarray
    """
    # copy of the mtm_spectrogram's input checking and defaults
    N = data.shape[-1]
    if timestamps is None:
        timestamps = np.arange(N) / fs
    if timestamps.shape[0] != N:
        raise ValueError(f"Expected timestamps to contain {N} elements but got {timestamps.shape[0]}")

    estimated_fs = 1.0 / np.median(np.diff(timestamps))
    if np.abs((estimated_fs - fs) / fs) > 0.01:
        print("Warning: estimated fs and provided fs differ by more than 1%")

    if nperseg is None:
        nperseg = 256
    if noverlap is None:
        noverlap = nperseg // 8

    if nfft is None:
        nfft = nperseg
    if nfft < nperseg:
        raise ValueError(f"'nfft' must be at least {nperseg}")

    if nperseg > N:
        raise ValueError(f"'nperseg' cannot be larger than the data size {N}")
    if not N > noverlap:
        raise ValueError(f"'noverlap' cannot be larger than {N-1}")

    if remove_mean:
        # mean per channel over time axis
        data = data - data.mean(axis=-1, keepdims=True)

    # compute tapers and lambdas via ghostipy helper
    tapers, lambdas = get_tapers(nperseg, bandwidth, fs=fs, n_tapers=n_tapers, min_lambda=min_lambda)
    n_tapers = tapers.shape[0]

    # compute tapered FFTs for all channels / leading dims
    X, freqs, starts = compute_tapered_ffts(
        data,
        tapers,
        nfft=nfft,
        noverlap=noverlap,
        fs=fs,
        time_axis=-1,
        n_fft_threads=n_fft_threads,
    )

    # X may have shape (n_tapers, n_segments, M) for single-channel 1D input,
    # or (n_tapers, *channel_shape, n_segments, M)
    # normalize power and build spectrograms
    if X.ndim == 3:
        # single-channel: shape (n_tapers, n_segments, M) -> convert to (n_tapers, 1, n_segments, M)
        X = X.reshape((n_tapers, 1, X.shape[1], X.shape[2]))

    # compute power spectral densities per taper
    spectrograms = (X.real ** 2 + X.imag ** 2) / fs  # shape (n_tapers, *channel_shape, n_segments, M)

    # frequency-domain scaling (double non-DC and non-Nyquist bins for real signals)
    if np.isrealobj(data):
        if nfft % 2 == 0:
            # even nfft: bins 1..-2 doubled
            spectrograms[..., 1:-1] *= 2.0
        else:
            spectrograms[..., 1:] *= 2.0

    # Combine tapers weighted by lambdas
    # broadcast lambdas over remaining axes
    w = lambdas.reshape((n_tapers,) + (1,) * (spectrograms.ndim - 1))
    combined = np.sum(w * spectrograms, axis=0) / np.sum(lambdas)  # shape (*channel_shape, n_segments, M)

    # Reorder axes to (*channel_shape, n_freqs, n_timepoints) to match requested output
    # combined currently (*channel_shape, n_segments, M) -> transpose last two axes
    combined = np.swapaxes(combined, -1, -2)  # shape (*channel_shape, M, n_segments)

    # compute output timestamps as mean of each window (preserve original mtm behavior)
    ts = timestamps
    # sliding window on timestamps (1D)
    if _HAS_SLIDING:
        ts_windows = sliding_window_view(ts, window_shape=nperseg)
    else:
        ts_windows = as_strided(ts, shape=(ts.shape[0] - nperseg + 1, nperseg), strides=(ts.strides[0], ts.strides[0]), writeable=False)
    # subsample with same step to get starts used above
    step = nperseg - noverlap
    n_segments = (N - noverlap) // step
    starts_idx = np.arange(n_segments) * step
    ts_selected = ts_windows[starts_idx, :nperseg]
    out_timestamps = ts_selected.mean(axis=1)

    return combined, freqs, out_timestamps