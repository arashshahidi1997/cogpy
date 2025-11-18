import numpy as np
import xarray as xr
from cogpy.preprocess.linenoise import drop_linenoise_harmonics, get_linenoise_freqs


def test_drop_linenoise_harmonics():
    freqs = np.array([1, 48, 49, 50.3, 51, 60, 102, 120, 133, 146, 151, 193])
    mtx = xr.DataArray(
        np.random.randn(len(freqs), 100), dims=["freq", "time"], coords={"freq": freqs}
    )
    mtx = drop_linenoise_harmonics(mtx)
    assert (mtx.freq.values == np.array([1, 60, 120, 133, 193])).all()

    freqs = np.array([1, 2, 5, 10])
    mtx = xr.DataArray(
        np.random.randn(len(freqs), 100), dims=["freq", "time"], coords={"freq": freqs}
    )
    mtx = drop_linenoise_harmonics(mtx)
    assert (mtx.freq.values == np.array([1, 2, 5, 10])).all()


def test_get_linenoise_freqs():
    freqs = np.array([1, 48, 49, 50.3, 51, 60, 102, 120, 133, 146, 151, 193])
    ln_closefreqs = get_linenoise_freqs(freqs)
    expected_output = np.array([48, 49, 50.3, 51, 102, 146, 151])
    assert (ln_closefreqs == expected_output).all()

    freqs = np.array([1, 2, 5, 10])
    ln_closefreqs = get_linenoise_freqs(freqs)
    expected_output = np.array([])
    assert (ln_closefreqs == expected_output).all()
