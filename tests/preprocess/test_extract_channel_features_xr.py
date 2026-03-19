import numpy as np
import pytest
import xarray as xr

from cogpy.preprocess.badchannel.channel_features import extract_channel_features_xr
from cogpy.datasets import schemas as sch


def _grid_xsig(*, time: int = 200, ml: int = 4, ap: int = 4, fs: float = 100.0) -> xr.DataArray:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((time, ml, ap)).astype(np.float64)
    t = np.arange(time, dtype=float) / float(fs)
    xsig = xr.DataArray(
        data,
        dims=("time", "ML", "AP"),
        coords={"time": t, "ML": np.arange(ml), "AP": np.arange(ap)},
        name="sig",
        attrs={"fs": float(fs)},
    )
    return sch.validate_ieeg_grid(xsig)


def _multichannel_xsig(*, time: int = 200, ch: int = 6, fs: float = 100.0) -> xr.DataArray:
    rng = np.random.default_rng(1)
    data = rng.standard_normal((ch, time)).astype(np.float64)
    t = np.arange(time, dtype=float) / float(fs)
    xsig = xr.DataArray(
        data,
        dims=("channel", "time"),
        coords={"channel": np.arange(ch), "time": t},
        name="sig",
        attrs={"fs": float(fs)},
    )
    return sch.validate_multichannel(xsig)


def _time_channel_xsig(*, time: int = 200, ch: int = 6, fs: float = 100.0) -> xr.DataArray:
    rng = np.random.default_rng(2)
    data = rng.standard_normal((time, ch)).astype(np.float64)
    t = np.arange(time, dtype=float) / float(fs)
    xsig = xr.DataArray(
        data,
        dims=("time", "channel"),
        coords={"channel": np.arange(ch), "time": t},
        name="sig",
        attrs={"fs": float(fs)},
    )
    return sch.validate_ieeg_time_channel(xsig)


def test_grid_windowed_schema_and_coord_attrs():
    xsig = _grid_xsig(time=200, ml=4, ap=4, fs=100.0)
    ds = extract_channel_features_xr(
        xsig,
        features=("mean", "variance"),
        window_size=50,
        window_step=25,
    )
    for v in ds.data_vars.values():
        sch.validate_ieeg_grid_windowed(v, win_dim="time_win")
    assert "time_win" in ds.coords
    assert np.all(np.diff(np.asarray(ds["time_win"].values)) > 0)
    assert ds["time_win"].attrs.get("long_name") == "window center time"


def test_grid_tml_windowed_blockwise_schema():
    xsig = _grid_xsig(time=200, ml=4, ap=4, fs=100.0)
    ds = extract_channel_features_xr(
        xsig,
        features=("temporal_mean_laplacian",),
        window_size=50,
        window_step=25,
    )
    v = ds["temporal_mean_laplacian"]
    assert tuple(v.dims) == ("time_win", "ML", "AP")
    sch.validate_ieeg_grid_windowed(v, win_dim="time_win")


def test_multichannel_windowed_schema_reorders_dims():
    xsig = _multichannel_xsig(time=200, ch=6, fs=100.0)
    ds = extract_channel_features_xr(
        xsig,
        features=("mean",),
        window_size=50,
        window_step=25,
    )
    v = ds["mean"]
    assert tuple(v.dims) == ("time_win", "channel")
    sch.validate_multichannel_windowed(v, win_dim="time_win")


def test_time_channel_windowed_schema():
    xsig = _time_channel_xsig(time=200, ch=6, fs=100.0)
    ds = extract_channel_features_xr(
        xsig,
        features=("mean",),
        window_size=50,
        window_step=25,
    )
    v = ds["mean"]
    assert tuple(v.dims) == ("time_win", "channel")
    sch.validate_multichannel_windowed(v, win_dim="time_win")


def test_out_dim_overridden_warns_and_uses_time_win():
    xsig = _multichannel_xsig(time=200, ch=6, fs=100.0)
    with pytest.warns(FutureWarning):
        ds = extract_channel_features_xr(
            xsig,
            features=("mean",),
            window_size=50,
            window_step=25,
            out_dim="window",
        )
    assert "time_win" in ds.coords
    assert "window" not in ds.coords
    assert tuple(ds["mean"].dims) == ("time_win", "channel")


def test_no_window_mode_unchanged():
    xsig = _multichannel_xsig(time=200, ch=6, fs=100.0)
    ds = extract_channel_features_xr(xsig, features=("mean", "variance"))
    assert "time_win" not in ds.dims
    for v in ds.data_vars.values():
        assert "time_win" not in v.dims


def test_spectral_feature_requires_fs():
    xsig = _multichannel_xsig(time=600, ch=4, fs=100.0).copy()
    xsig.attrs.pop("fs", None)
    with pytest.raises(ValueError, match="Sampling rate required"):
        extract_channel_features_xr(
            xsig,
            features=("snr",),
            window_size=256,
            window_step=128,
        )
