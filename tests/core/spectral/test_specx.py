"""Tests for specx (xarray spectral wrappers)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cogpy.spectral.specx import coherencex, psdx, spectrogramx


def test_psdx_multitaper_dims_and_attrs():
    rng = np.random.default_rng(42)
    data = xr.DataArray(
        rng.normal(size=(2000, 4, 3)),
        dims=("time", "AP", "ML"),
        coords={"time": np.arange(2000) / 1000.0, "AP": np.arange(4), "ML": np.arange(3)},
        attrs={"fs": 1000.0},
    )

    psd = psdx(data, axis="time", method="multitaper", bandwidth=4.0)

    assert psd.dims == ("AP", "ML", "freq")
    assert psd.attrs["method"] == "multitaper"
    assert psd.attrs["units"] == "power/Hz"
    assert np.isfinite(psd.data).all()
    assert len(psd.coords["freq"]) > 0


def test_psdx_welch_dims():
    rng = np.random.default_rng(42)
    data = xr.DataArray(
        rng.normal(size=(4096, 8)),
        dims=("time", "channel"),
        coords={"time": np.arange(4096) / 1000.0, "channel": np.arange(8)},
        attrs={"fs": 1000.0},
    )

    psd = psdx(data, axis="time", method="welch", nperseg=512, noverlap=256)

    assert psd.dims == ("channel", "freq")
    assert psd.attrs["method"] == "welch"
    assert len(psd.coords["freq"]) > 0


def test_psdx_preserves_non_time_coords():
    rng = np.random.default_rng(42)
    data = xr.DataArray(
        rng.normal(size=(1000, 4, 4)),
        dims=("time", "AP", "ML"),
        coords={
            "time": np.arange(1000) / 100.0,
            "AP": np.arange(4),
            "ML": np.arange(4),
            "AP_mm": ("AP", [0.0, 1.0, 2.0, 3.0]),
            "ML_mm": ("ML", [0.0, 1.0, 2.0, 3.0]),
        },
        attrs={"fs": 100.0},
    )

    psd = psdx(data, axis="time", method="welch", nperseg=256)

    assert "AP_mm" in psd.coords
    assert "ML_mm" in psd.coords
    assert "time" not in psd.coords
    assert "freq" in psd.coords


def test_spectrogramx_dims():
    pytest.importorskip("ghostipy")
    pytest.importorskip("dask")

    rng = np.random.default_rng(42)
    data = xr.DataArray(
        rng.normal(size=(2048, 3, 2)),
        dims=("time", "AP", "ML"),
        coords={"time": np.arange(2048) / 1000.0, "AP": np.arange(3), "ML": np.arange(2)},
        attrs={"fs": 1000.0},
    )

    spec = spectrogramx(data, axis="time", bandwidth=4.0, nperseg=256, noverlap=128)

    assert spec.dims == ("AP", "ML", "freq", "time")
    assert len(spec.coords["freq"]) > 0
    assert len(spec.coords["time"]) > 0


def test_coherencex_pure_sinusoid_has_high_coherence_at_f0():
    fs = 1000.0
    t = np.arange(4000) / fs
    f0 = 10.0

    rng = np.random.default_rng(42)
    x_vals = np.sin(2 * np.pi * f0 * t) + 0.01 * rng.normal(size=t.shape)
    y_vals = np.sin(2 * np.pi * f0 * t + 0.1) + 0.01 * rng.normal(size=t.shape)

    x = xr.DataArray(x_vals, dims=("time",), coords={"time": t}, attrs={"fs": fs})
    y = xr.DataArray(y_vals, dims=("time",), coords={"time": t}, attrs={"fs": fs})

    coh = coherencex(x, y, axis="time", NW=4.0)

    assert coh.dims == ("freq",)
    assert np.all((coh.data >= 0) & (coh.data <= 1))

    idx = int(np.argmin(np.abs(coh.coords["freq"].values - f0)))
    assert coh.data[idx] > 0.7  # high SNR → strong coherence near f0
