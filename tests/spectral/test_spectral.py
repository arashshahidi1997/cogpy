import numpy as np
import xarray as xr
from scipy.signal import spectrogram as scispec
from cogpy.spectral import multitaper as sp
import ghostipy as gsp
from cogpy.datasets import load as ld
from cogpy.utils.sliding import rolling_win


# %% test ghostipy
def test_mtm_spectrogramx():
    xsig = ld.load_sample()
    mtspec_kwargs = {
        "bandwidth": 2,
        "nperseg": 128,
        "noverlap": 64,
        "remove_mean": True,
        "fs": xsig.fs,
    }
    Sx = sp.mtm_spectrogramx(xsig, dim="time", **mtspec_kwargs).compute()
    S, f, t = gsp.mtm_spectrogram(xsig[0, 0].data.compute(), **mtspec_kwargs)
    assert np.allclose(S, Sx.data[0, 0])


# %% tests
def test_multitaper_spectrogram():
    # Test parameters
    NW = 2
    window_size = 256
    window_step = 64
    fs = 1000
    nfft = None
    detrend = False

    # Random signal generation
    np.random.seed(42)
    xsig = xr.DataArray(
        np.random.randn(2500), dims=("time"), coords={"time": np.arange(2500) / fs}
    )

    # Calculate spectrogram using MultiTaperSlidingWindow
    # mt = myspec.MultiTaperSlidingWindow(NW=NW, window_size=window_size, window_step=window_step, fs=fs,
    #                              nfft=nfft, detrend=detrend)
    # mt.transform(signal[np.newaxis])

    xwin = rolling_win(
        xsig,
        window_size=window_size,
        window_step=window_step,
        dim="time",
        window_dim="window",
    )
    mtx = xr.apply_ufunc(
        sp.multitaper_psd,
        xwin,
        input_core_dims=[["window"]],
        output_core_dims=[["freq"]],
        output_dtypes=[float],
    )

    # Calculate spectrogram using scipy.signal package
    freqs, times, scispec_spectrogram = scispec(
        xsig, fs=fs, nperseg=window_size, noverlap=window_size - window_step
    )
    # Assert the spectrogram results are close
    np.allclose(mtx, scispec_spectrogram.T, atol=5e-2)

    print(
        "Spectrogram comparison passed, the results are consistent with scipy.signal spectrogram."
    )


def test_multitaper_rolling_window():
    xsig = ld.load_sample()
    xwin = rolling_win(xsig, window_size=128, window_step=64, dim="time").chunk(
        {"AP": -1, "ML": -1, "window": -1}
    )
    # to get a numpy array
    # xspec_numpy = sp.multitaper_psd(xwin)

    # to keep as a dask array
    xspec_dask_ufunc = xr.apply_ufunc(
        sp.multitaper_psd,
        xwin,  # do not chunk here!
        input_core_dims=[["AP", "ML", "window"]],
        output_core_dims=[["AP", "ML", "freq"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[np.float64],
        dask_gufunc_kwargs=dict(output_sizes={"freq": 65}),
        kwargs=dict(NW=2),
    )
    xspec_dask_ufunc = sp.specx_coords(xspec_dask_ufunc, nfft=128, fs=xsig.fs)

    # to keep it as an xarray
    xspec = xr.apply_ufunc(
        sp.multitaper_psd,
        xwin.compute(),
        input_core_dims=[["AP", "ML", "window"]],
        output_core_dims=[["AP", "ML", "freq"]],
        output_dtypes=[np.float64],
        kwargs=dict(NW=2),
    )
    xspec = sp.specx_coords(xspec, nfft=128, fs=xsig.fs)

    # using helper
    xspec_direct = sp.multitaper_spectrogram(xsig, window_size=128, window_step=64)

    assert np.allclose(xspec_dask_ufunc.compute().data, xspec.data)
    assert np.allclose(
        xspec_dask_ufunc.chunk({"freq": -1, "time": 3}).compute(),
        xspec_dask_ufunc.compute(),
    )
    assert np.allclose(xspec_direct.compute().data, xspec.data)


def test_multitaper_fft():
    xsig = ld.load_sample()
    xwin = rolling_win(xsig, window_size=128, window_step=64, dim="time").chunk(
        {"AP": -1, "ML": -1, "window": -1}
    )
    xmtfft = xr.apply_ufunc(
        sp.multitaper_fft,
        xwin,
        input_core_dims=[["time", "AP", "ML", "window"]],
        output_core_dims=[["time", "AP", "ML", "tapers", "freq"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[np.complex128],
        dask_gufunc_kwargs=dict(output_sizes={"tapers": 3, "freq": 65}),
        kwargs=dict(K_max=3),
    )
    xmtfft = xmtfft.assign_coords(tapers=["taper1", "taper2", "taper3"])
    psd_rec = ((np.abs(xmtfft.compute()) ** 2) / 128).mean(dim=["tapers"])
    psd_rec = sp.specx_coords(
        psd_rec, nfft=128, fs=xsig.fs, freq_dim="freq", time_dim="time"
    )
    xspec = sp.multitaper_spectrogram(xsig, window_size=128, window_step=64).compute()
    assert [
        np.allclose(
            ((np.abs(sp.multitaper_fft(xwin.isel(time=iwin))) ** 2) / 128).mean(
                axis=-2
            ),
            sp.multitaper_psd(xwin.isel(time=iwin)),
        )
        for iwin in range(xwin.sizes["time"])
    ]
    assert np.allclose(xspec, psd_rec), print(xspec[0, 0] == psd_rec[0, 0])


def test_multitaper_fftgram():
    N = 512
    NW = 3
    xsig = ld.load_raw_sample()
    mt_fft_running = sp.multitaper_fftgram(xsig, NW=NW, window_size=N, window_step=256)
    mt_specgram = sp.multitaper_spectrogram(xsig, NW=NW, window_size=N, window_step=256)
    assert (
        (np.abs(mt_fft_running.compute()) ** 2 / N).mean("taper")
        == mt_specgram.compute()
    ).all()
