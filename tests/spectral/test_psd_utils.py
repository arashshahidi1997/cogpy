"""Tests for PSD utilities (v2.8.0)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cogpy.spectral.psd_utils import compute_psd_window, psd_to_db, stack_spatial_dims


@pytest.fixture
def signal_data():
    fs = 1000.0
    t = np.linspace(0.0, 10.0, 10000, dtype=float)
    y = np.sin(2 * np.pi * 10.0 * t) + 0.1 * np.random.RandomState(0).randn(t.size)
    return xr.DataArray(y, dims=["time"], coords={"time": t}, attrs={"fs": fs})


@pytest.fixture
def grid_data():
    fs = 100.0
    t = np.arange(0.0, 10.0, 1.0 / fs, dtype=float)
    y = np.random.RandomState(0).randn(t.size, 8, 8)
    return xr.DataArray(
        y,
        dims=("time", "AP", "ML"),
        coords={"time": t, "AP": np.arange(8), "ML": np.arange(8)},
        attrs={"fs": fs},
    )


def test_compute_psd_window(signal_data):
    psd = compute_psd_window(signal_data, t_center=5.0, window_size=2.0, nperseg=256, method="welch")
    assert "freq" in psd.dims
    assert "time" not in psd.dims
    assert psd.sizes["freq"] > 0


def test_psd_to_db(signal_data):
    psd = compute_psd_window(signal_data, t_center=5.0, window_size=2.0, nperseg=256, method="welch")
    psd_db = psd_to_db(psd)
    assert psd_db.attrs.get("units") == "dB"
    assert np.isfinite(np.asarray(psd_db.values)).all()


def test_stack_spatial_dims_grid(grid_data):
    st = stack_spatial_dims(grid_data)
    assert "channel" in st.dims
    assert "AP" not in st.dims
    assert "ML" not in st.dims
    assert st.sizes["channel"] == 64


def test_stack_spatial_dims_already_channel():
    da = xr.DataArray(np.random.RandomState(0).randn(100, 10), dims=("time", "channel"), attrs={"fs": 1.0})
    st = stack_spatial_dims(da)
    assert st.dims == ("time", "channel")
    assert st.sizes["channel"] == 10

