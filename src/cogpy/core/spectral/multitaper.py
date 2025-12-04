import numpy as np
import xarray as xr
import dask.array as da
import ghostipy as gsp
from scipy import signal
from typing import Callable, Dict, List, Any
from ..utils.convert import closest_power_of_two
from ..utils import sliding as sl


def nperseg_from_ncycle(fm, fs=1, ncycle=7, power_of_two=True):
    """
    rel_nperseg: number of cycles per segment

    Parameters
    ----------
    fm: center frequency
    fs: sampling frequency

    Returns
    -------
    nperseg: number of samples per segment
    """
    nperseg = int(fs * ncycle / fm)
    if power_of_two:
        nperseg = closest_power_of_two(nperseg)
    return nperseg


def dpss_tapers(N: int, NW: float = 2, K_max: int = None) -> np.ndarray:
    """
    Generate Discrete Prolate Spheroidal Sequences (DPSS) tapers.

    Parameters
    ----------
    N : int
        Length of the tapers.
    NW : float
        Time-bandwidth product.
    K_max : int, optional
        Maximum number of tapers to return. If None, K_max is set to ``2 * NW-1``.

    Returns
    -------
    np.ndarray
        DPSS tapers of shape (K_max, N).

    See Also
    --------
    scipy.signal.windows.dpss
    """
    if K_max is None:
        K_max = int(2 * NW - 1)
    assert K_max > 0 and K_max < N / 2, print(
        "increase resolution `N` or decrease time-frequency-half-bandwidth `NW`"
    )
    return signal.windows.dpss(N, NW, Kmax=K_max)


# %% ghostipy
def mtm_kwarg_from_gsp(bandwidth, fs, nperseg, noverlap):
    """
    convert ghostipy mtm_spectrogram kwargs to NW, window_size, window_step
    """
    NW = bandwidth * nperseg / fs
    window_size = nperseg
    window_step = nperseg - noverlap
    return {"NW": NW, "fs": fs, "window_size": window_size, "window_step": window_step}


def mtm_kwarg_to_gsp(NW, fs, window_size, window_step):
    """
    convert NW, window_size, window_step to ghostipy mtm_spectrogram kwargs
    """
    bandwidth = NW * fs / window_size
    nperseg = window_size
    noverlap = window_size - window_step
    return {"bandwidth": bandwidth, "fs": fs, "nperseg": nperseg, "noverlap": noverlap}


def mtm_spectrogram(x, bandwidth, axis=-1, **kwargs):
    """
    Compute a multitaper spectrogram (Thomson method) on a NumPy or 
    Dask array by applying ``ghostipy.mtm_spectrogram`` independently 
    along a chosen time axis.

    This function acts as a wrapper around ``ghostipy.mtm_spectrogram``:
    - Ghostipy performs the *inner*, highly optimized 1-D multitaper 
      spectrogram computation (as_strided windowing, vectorized tapering,
      batched FFTW).
    - Dask provides *outer* parallelism across channels / trials /
      other non-time dimensions.

    It is therefore suitable for multichannel or batched signals, where
    each 1-D time series is processed independently.

    Notes on scaling and chunking
    -----------------------------
    - The time axis **must not be chunked**. Each Dask block must contain 
      the full time series along ``axis``; otherwise the underlying 1-D 
      multitaper algorithm cannot be applied. This function will rechunk
      as needed.

    - Parallelism occurs **across slices of all other axes**. For example,
      with an array shaped ``(n_channels, n_time)``, chunk over
      ``n_channels`` and keep ``n_time`` as a single chunk.

    - If using Dask with multiple workers or threads, it is usually best
      to set ``n_fft_threads=1`` in ``kwargs`` to avoid oversubscribing
      CPU cores, since FFTW itself also uses multithreading.

    - When applied to a NumPy array, this function behaves like a regular
      vectorized loop and returns a NumPy array.

    Parameters
    ----------
    x : np.ndarray or dask.array.Array
        Input array containing one or more time series. The spectrogram
        is computed independently along the specified ``axis`` for each
        slice of the remaining dimensions.
    bandwidth : float
        Taper bandwidth in Hz.
    axis : int, optional
        Axis corresponding to time (default: -1).

    Keyword Arguments
    -----------------
    fs : float, optional
        Sampling rate in Hz. Default is 1.
    timestamps : np.ndarray, optional
        Timestamps for the samples. If omitted, they are inferred as 
        ``np.arange(n_time) / fs``.
    nperseg : int, optional
        Window length in samples. Default: 256.
    noverlap : int, optional
        Number of samples of overlap. Default: ``nperseg // 8``.
    min_lambda : float, optional
        Minimum energy concentration for DPSS tapers (default: 0.95).
    n_tapers : int, optional
        Number of DPSS tapers. Default: all tapers satisfying
        ``min_lambda``.
    remove_mean : bool, optional
        Whether to remove the window mean before tapering. Default: False.
    nfft : int, optional
        FFT length. Default: ``nperseg``.
    n_fft_threads : int, optional
        Number of threads used internally by FFTW. Default: 
        the number of available CPUs, but using 1 is recommended when
        Dask itself is providing parallelism.

    Returns
    -------
    mtspec : dask.array.Array or np.ndarray
        Multitaper spectrogram with shape ``(..., n_freq, n_time_segments)``.
        The output corresponds to applying the 1-D spectrogram along 
        ``axis`` for each slice of the other axes.
    f : np.ndarray
        Frequency vector.
    t : np.ndarray
        Time vector (center of each window).

    Examples
    --------
    >>> import numpy as np
    >>> import dask.array as da
    >>> from cogpy.core.spectral import multitaper

    >>> fs = 1000
    >>> t = np.arange(0, 10, 1/fs)
    >>> signal = np.array([np.sin(2*np.pi*50*t + np.random.rand()*2*np.pi)
    ...                    for _ in range(8)])

    >>> # Parallelize over channels (time axis unchunked)
    >>> x_da = da.from_array(signal, chunks=(2, signal.shape[-1]))

    >>> mtspec, f, tt = multitaper.mtm_spectrogram(x_da, bandwidth=4, fs=fs,
    ...                                 nperseg=256, noverlap=128)
    >>> assert mtspec.shape == (8, 129, 77)  # (n_channels, n_freq, n_time_segments)
    """
    x = np.moveaxis(x, axis, -1)
    x_fiber = take_first_fiber_along_axis(x, axis=-1)
    if isinstance(x_fiber, da.Array):
        x_fiber: da.Array
        x_fiber = x_fiber.compute()
    S, f, t = gsp.mtm_spectrogram(x_fiber, bandwidth, **kwargs)

    def _mtspec_func(x_):
        S, *_ = gsp.mtm_spectrogram(x_, bandwidth, **kwargs)
        return S

    mtspec = da.apply_along_axis(_mtspec_func, -1, x, shape=S.shape, dtype=S.dtype)
    return mtspec, f, t


def mtm_spectrogramx(xsig, dim="time", **kwargs):
    """
    multitaper spectrogram using ghostipy backend, wrapped as xarray DataArray

    Parameters
    ----------
    xsig : xr.DataArray
            Signal with time dimension
    dim : str
            Name of time dimension
    **kwargs : keyword arguments for mtm_spectrogram function
            - bandwidth : float
                    Bandwidth of taper, in Hz

                    Note:
                        NW = bandwidth * N / fs
                        K = int(np.ceil(2*NW)) - 1

            - fs : float, optional
                    Sampling rate, in Hz.
            - timestamps : np.ndarray, with shape (T, ), optional
                    Timestamps for the data. If not provided, they will be
                    inferred using np.arange(len(data)) / fs
            - nperseg : int, optional
                    Number of samples to use for each segment/window.
            - noverlap : int, optional
                    Number of points to overlap between segments.
            - min_lambda : float, optional
                    Minimum energy concentration that each taper must satisfy.
            - n_tapers : int, optional
                    Number of tapers to compute
            - remove_mean : boolean, optional
                    Whether to remove the mean of the data before computing the
                    MTM spectrum.
            - nfft : int, optional
                    How many FFT points to use for each segment.
            - n_fft_threads : int, optional
                    Number of threads to use for the FFT.

    Returns
    -------
    mtspec_xr : xr.DataArray
            multitaper spectrogram with `freq` and `time` dimensions added
    """
    dim = xsig.get_axis_num(dim)
    mtspec, f, t = mtm_spectrogram(xsig.data, axis=dim, **kwargs)
    coords = {dim: xsig.coords[dim] for dim in xsig.dims if dim != "time"}
    coords["freq"] = f
    coords["time"] = t
    dims = list(xsig.dims)
    dims.remove("time")
    dims += ["freq", "time"]
    mtspec_xr = xr.DataArray(mtspec, coords=coords, dims=dims)
    return mtspec_xr


def take_first_fiber_along_axis(x, axis=-1):
    """
    Return the fiber of `x` along the given axis,
    fixing all other dimensions at index 0.

    Parameters
    ----------
    x : np.ndarray or dask.array.Array
        Input N-dimensional array.
    axis : int, optional
        Axis along which to extract the fiber. Default is -1 (last axis).

    Returns
    -------
    fiber : 1-D array (NumPy or Dask, same type as input)
        The slice along `axis` with all other indices fixed at 0.
    """
    # Normalize axis
    axis = axis % x.ndim

    # Build index tuple: all zeros except slice(None) for target axis
    idx = tuple(0 if ax != axis else slice(None) for ax in range(x.ndim))
    return x[idx]


# %% multitaper
def multitaper_fft(y, axis=-1, NW=2, nfft=None, K_max=None, detrend=True, tapers=None):
    """
    multitaper  density estimate

    Parameters
    ----------
    y: ndarray (..., time)
    axis: sample axis along which Fourier transform is operated
    NW: time frequency half-bandwidth
    K_max: maximum number of slepian tapers.
            if None, K_max is set equal to 2NW-1 (note N=y.shape[-1] # number of samples)

    Returns
    -------
    fft: ndarray (..., ntaper, nfreq)
            multitaper fft estimate of y
    """
    y = np.swapaxes(y, axis, -1)  # swap axes so the samples axis is the last dimension
    N = y.shape[-1]  # number of samples

    if tapers is None:
        tapers = dpss_tapers(N, NW, K_max)  # shape (ntaper, sample_dim)
    else:
        assert tapers.shape[1] == N, "tapers should have shape (ntaper, sample_dim)"
        assert tapers.ndim == 2, "tapers should have shape (ntaper, sample_dim)"

    if detrend:
        y = signal.detrend(y, axis=-1)

    tapered_y = np.expand_dims(y, axis=-1) * tapers.T  # (*dof, sample_dim, taper_dim)

    if nfft is None:
        nfft = N
    if nfft > N:
        pad_width = [(0, 0)] * tapered_y.ndim
        pad_width[-2] = (0, nfft - N)
        tapered_y = np.pad(tapered_y, pad_width, mode="constant")

    tapered_y = np.moveaxis(tapered_y, -2, 0)  # shape (sample_dim, *dof, taper_dim)
    shape_origin = tapered_y.shape[
        1:
    ]  # original shape without sample_dim (*dof, taper_dim)
    tapered_y = tapered_y.reshape(
        tapered_y.shape[0], -1
    )  # shape (sample_dim, *dof * taper_dim)
    fft_ = np.fft.rfft(tapered_y, n=nfft, axis=0)  # shape (nfreq, *dof * taper_dim)
    fft_ = fft_.reshape(-1, *shape_origin)  # (nfreq, *dof, taper_dim)
    fft_ = np.moveaxis(fft_, 0, -1)  # (*dof, ntaper, nfreq)
    return fft_


multitaper_fft.output_dtype = np.complex128


def multitaper_psd(y, axis=-1, NW=2, nfft=None, K_max=None, detrend=True):
    """
    multitaper power spectral density estimate

    Parameters
    ----------
    y: ndarray
    axis: sample axis along which Fourier transform is operated
    NW: time frequency half-bandwidth
    K_max: maximum number of slepian tapers.
            if None, K_max is set equal to 2NW-1 (note N=y.shape[-1] # number of samples)

    Returns
    -------
    psd_: same shape as y (with sample dim replaced by freq dim of the same size)
            multitaper estimate of signal y
    f: freq array
    """
    mtfft_ = multitaper_fft(
        y, axis=axis, NW=NW, nfft=nfft, K_max=K_max, detrend=detrend
    )
    if nfft is None:
        nfft = y.shape[axis]
    psd_ = np.abs(mtfft_) ** 2 / nfft
    psd_ = np.mean(psd_, axis=-2)  # average across tapers # (*dof, nfreq)
    return psd_


multitaper_psd.output_dtype = np.float64


# %% spectrogram functions (not fundamental, since all rolling window operations are handled using rolling_win)
def assign_freqs(freqx: xr.DataArray, fs, freq_dim="freq"):
    nfft = freqx.sizes[freq_dim]
    freqs_ = np.fft.rfftfreq(nfft, 1 / fs)
    freqx = freqx.assign_coords({freq_dim: freqs_})
    return freqx


def specx_coords(specx, nfft, fs, freq_dim="freq", time_dim="time"):
    reorder_dims = [d for d in specx.dims if d not in (time_dim, freq_dim)] + [
        freq_dim,
        time_dim,
    ]
    specx = specx.transpose(*reorder_dims)
    if freq_dim not in specx.coords:
        specx = specx.assign_coords({freq_dim: np.fft.rfftfreq(nfft, 1 / fs)})
    return specx


def multitaper_spectrogram(
    xsig: xr.DataArray,
    NW=2,
    window_size: int = 256,
    window_step: int = 64,
    fs=None,
    nfft=None,
    detrend=True,
    time_dim="time",
    window_dim="window",
    freq_dim="freq",
):
    mtx = running_spectral(
        multitaper_psd,
        xsig,
        measure_kwargs=dict(NW=NW, detrend=detrend),
        window_size=window_size,
        window_step=window_step,
        fs=fs,
        nfft=nfft,
        time_dim=time_dim,
        window_dim=window_dim,
        freq_dim=freq_dim,
        extra_output_dims=[],
        extra_output_sizes={},
        output_dtype=float,
    )
    return mtx


def multitaper_fftgram(
    xsig: xr.DataArray,
    NW=2,
    window_size: int = 256,
    window_step: int = 64,
    fs=None,
    nfft=None,
    detrend=True,
    time_dim="time",
    window_dim="window",
    freq_dim="freq",
    taper_dim="taper",
):
    ntapers = 2 * NW - 1
    mtx_fft = running_spectral(
        multitaper_fft,
        xsig,
        measure_kwargs=dict(NW=NW, detrend=detrend),
        window_size=window_size,
        window_step=window_step,
        fs=fs,
        nfft=nfft,
        time_dim=time_dim,
        window_dim=window_dim,
        freq_dim=freq_dim,
        extra_output_dims=[taper_dim],
        extra_output_sizes={taper_dim: ntapers},
        output_dtype=np.complex128,
    )
    mtx_fft = mtx_fft.assign_coords({taper_dim: np.arange(ntapers) + 1})
    return mtx_fft.transpose(
        *[d for d in mtx_fft.dims if d not in (time_dim, freq_dim, taper_dim)],
        time_dim,
        taper_dim,
        freq_dim,
    )


def running_spectral(
    spectral_measure: Callable,
    xsig: xr.DataArray,
    measure_kwargs: Dict[str, Any],
    window_size: int = 256,
    window_step: int = 64,
    fs=None,
    nfft=None,
    time_dim="time",
    window_dim="window",
    freq_dim="freq",
    extra_output_dims: List[str] = None,
    extra_output_sizes: Dict[str, int] = None,
    output_dtype=float,
) -> xr.DataArray:
    """
    spectral_func: function handle for spectral function
    """
    assert isinstance(xsig, xr.DataArray), "xsig should be an xarray DataArray"
    assert (
        time_dim in xsig.dims
    ), f"{time_dim} not in xsig.dims, please pass the name of your time dimension to time_dim in kwargs"
    assert (
        window_dim not in xsig.dims
    ), f"{window_dim} already in xsig.dims, please pass a different name for window dimension to window_dim in kwargs"

    # sampling frequency
    fs = getattr(xsig, "fs", fs)

    # nfft
    if nfft is None:
        nfft = window_size

    # output dims
    output_dims = extra_output_dims or [freq_dim]
    assert isinstance(
        extra_output_dims, list
    ), "output_dims should be a list of strings with names of output dimensions other than frequency dimension"
    extra_output_dims = extra_output_dims or []
    output_dims = extra_output_dims + [freq_dim]

    # output sizes
    output_sizes = {freq_dim: nfft // 2 + 1}
    extra_output_sizes = extra_output_sizes or dict()
    assert isinstance(
        extra_output_sizes, dict
    ), "output_sizes should be a dictionary with keys as output dimension names and values as their sizes"
    output_sizes.update(extra_output_sizes)

    # check output sizes and dims
    for output_dim in output_dims:
        assert (
            output_dim not in xsig.dims
        ), f"{output_dim} already in xsig.dims, please pass a different name for frequency dimension to output_dim in kwargs"
        assert (
            output_dim in output_sizes
        ), f"except for frequency size, please pass the size of output dimension {output_dim} in output_sizes kwargs"

    # Normalize to DataArray and ensure a time coord exists
    if time_dim not in xsig.coords:
        xsig = xsig.assign_coords({time_dim: np.arange(xsig.sizes[time_dim]) / fs})

    # get output dtype
    out_dtype = getattr(spectral_measure, "output_dtype", output_dtype)
    dof_dims = [dim_ for dim_ in xsig.dims if dim_ not in [time_dim]]

    # update measure kwargs
    measure_kwargs.update(dict(nfft=nfft))

    # running measure apply kwargs
    apply_kwargs = dict(
        slider_kwargs=dict(window_size=window_size, window_step=window_step),
        measure_kwargs=measure_kwargs,
        measure_input_core_dims=[dof_dims + [window_dim]],
        measure_output_core_dims=[dof_dims + [*output_dims]],
        measure_output_sizes=output_sizes,
        run_dim=time_dim,
        window_dim=window_dim,
        output_dtype=out_dtype,
    )

    # apply running measure
    mtx = sl.running_measure(spectral_measure, xsig, fs=fs, **apply_kwargs)

    # wrap as xarray and add freq coords
    mtx = specx_coords(mtx, nfft, fs, freq_dim=freq_dim, time_dim=time_dim)
    return mtx
