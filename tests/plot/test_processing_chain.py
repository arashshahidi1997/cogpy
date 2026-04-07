"""Tests for ProcessingChain notch filter and external PSD analysis."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("panel")
pytest.importorskip("param")

from cogpy.plot.hv.processing_chain import ProcessingChain


def _example_grid(*, fs: float = 1000.0, n_time: int = 3000) -> xr.DataArray:
    rng = np.random.default_rng(42)
    t = np.arange(n_time) / fs
    data = rng.normal(size=(n_time, 3, 4))
    return xr.DataArray(
        data,
        dims=("time", "AP", "ML"),
        coords={"time": t, "AP": np.arange(3), "ML": np.arange(4)},
        attrs={"fs": fs},
    )


def test_notch_filter_list_mode_describe():
    data = _example_grid()

    chain = ProcessingChain(data)
    chain.notch_on = True
    chain.notch_use_harmonics = False
    chain.notch_freqs = [60.0, 120.0]

    win = chain.get_window(0.0, 1.0)

    assert win.dims == ("time", "AP", "ML")
    assert "Notch(60.0,120.0Hz)" in chain.describe()


def test_notch_filter_harmonic_mode_describe():
    data = _example_grid()

    chain = ProcessingChain(data)
    chain.notch_on = True
    chain.notch_use_harmonics = True
    chain.notch_fundamental = 60.0
    chain.notch_harmonics = 3

    _ = chain.get_window(0.0, 1.0)

    assert "Notch(60.0Hz×3)" in chain.describe()


def test_compute_psd_dims_and_processing_attr():
    data = _example_grid(n_time=4096)

    chain = ProcessingChain(data)
    chain.zscore_on = False

    from cogpy.spectral.specx import psdx

    win = chain.get_window(0.0, 2.0)
    psd = psdx(win, method="multitaper", bandwidth=4.0)

    assert set(psd.dims) == {"freq", "AP", "ML"}
    assert len(psd.coords["freq"]) > 0


def test_psd_with_processing_pipeline_labels():
    data = _example_grid(n_time=4096)

    chain = ProcessingChain(data)
    chain.zscore_on = False
    chain.bandpass_on = True
    chain.bandpass_lo = 1.0
    chain.bandpass_hi = 100.0
    chain.notch_on = True
    chain.notch_use_harmonics = True
    chain.notch_fundamental = 60.0
    chain.notch_harmonics = 2

    from cogpy.spectral.specx import psdx

    win = chain.get_window(0.0, 2.0)
    psd = psdx(win, method="welch", nperseg=512, bandwidth=4.0)

    assert set(psd.dims) == {"freq", "AP", "ML"}
    assert "BP(" in chain.describe()
    assert "Notch(" in chain.describe()
