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
def test_multitaper_psd_vs_scipy():
    NW = 2
    window_size = 256
    window_step = 64
    fs = 1000

    np.random.seed(42)
    xsig = xr.DataArray(
        np.random.randn(2500), dims=("time"), coords={"time": np.arange(2500) / fs}
    )

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

    freqs, times, scispec_spectrogram = scispec(
        xsig, fs=fs, nperseg=window_size, noverlap=window_size - window_step
    )
    np.allclose(mtx, scispec_spectrogram.T, atol=5e-2)
