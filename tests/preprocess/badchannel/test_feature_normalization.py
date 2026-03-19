import numpy as np
import pytest
import xarray as xr

from cogpy.preprocess.badchannel.feature_normalization import (
    normalize_windowed_features,
    smooth_windowed_features,
    summarize_windowed_features,
)


@pytest.fixture
def windowed_ds():
    """Small synthetic windowed feature dataset."""
    rng = np.random.default_rng(0)
    n_win, n_ml, n_ap = 20, 4, 4
    variance = xr.DataArray(
        rng.standard_normal((n_win, n_ml, n_ap)) ** 2,
        dims=("time_win", "ML", "AP"),
        coords={
            "time_win": np.arange(n_win) * 0.25,
            "ML": np.arange(n_ml),
            "AP": np.arange(n_ap),
        },
    )
    kurtosis = xr.DataArray(
        rng.standard_normal((n_win, n_ml, n_ap)),
        dims=("time_win", "ML", "AP"),
        coords=variance.coords,
    )
    return xr.Dataset(
        {"variance": variance, "kurtosis": kurtosis},
        attrs={
            "fs": 1000.0,
            "fs_win": 4.0,  # 1000 / 250
            "window_step_s": 0.25,
            "window_size_s": 0.5,
            "window_size": 500,
            "window_step": 250,
        },
    )


def test_normalize_robust_zero_mean(windowed_ds):
    out = normalize_windowed_features(windowed_ds, robust=True)
    for name, da in out.data_vars.items():
        med = da.median(dim="time_win")
        assert float(np.abs(med).max()) < 0.1, f"{name}: median after robust normalization should be near 0"


def test_normalize_standard(windowed_ds):
    out = normalize_windowed_features(windowed_ds, robust=False)
    for name, da in out.data_vars.items():
        std = da.std(dim="time_win")
        assert float(np.abs(std - 1.0).max()) < 0.1, f"{name}: std after normalization should be near 1"


def test_normalize_attrs_preserved(windowed_ds):
    out = normalize_windowed_features(windowed_ds)
    assert out.attrs["fs_win"] == windowed_ds.attrs["fs_win"]
    assert out.attrs.get("normalization_applied") is True
    for _, da in out.data_vars.items():
        assert da.attrs.get("normalized") is True


def test_normalize_passthrough_no_dim(windowed_ds):
    static = xr.DataArray(
        np.ones((4, 4)),
        dims=("ML", "AP"),
        coords={"ML": np.arange(4), "AP": np.arange(4)},
    )
    ds2 = windowed_ds.assign(static_feat=static)
    out = normalize_windowed_features(ds2)
    assert float((out["static_feat"] - static).max()) == 0.0


def test_smooth_mean(windowed_ds):
    out = smooth_windowed_features(windowed_ds, window_s=1.0, method="mean")
    for name in windowed_ds.data_vars:
        assert out[name].dims == windowed_ds[name].dims
    assert out.attrs["smoothing_window_s"] == 1.0
    assert out.attrs["smoothing_method"] == "mean"


def test_smooth_requires_fs_win():
    ds = xr.Dataset({"x": xr.DataArray(np.ones((5, 3)), dims=("time_win", "channel"))})
    with pytest.raises(ValueError, match="fs_win"):
        smooth_windowed_features(ds, window_s=1.0)


def test_smooth_invalid_method(windowed_ds):
    with pytest.raises(ValueError, match="method"):
        smooth_windowed_features(windowed_ds, window_s=0.5, method="sum")


def test_summarize_output_shape(windowed_ds):
    out = summarize_windowed_features(windowed_ds, stats=("median", "mad", "max"))
    assert "time_win" not in out.dims
    expected_vars = {
        "variance_median",
        "variance_mad",
        "variance_max",
        "kurtosis_median",
        "kurtosis_mad",
        "kurtosis_max",
    }
    assert set(out.data_vars) == expected_vars


def test_summarize_mad_nonnegative(windowed_ds):
    out = summarize_windowed_features(windowed_ds, stats=("mad",))
    for name, da in out.data_vars.items():
        assert float(da.min()) >= 0.0, f"{name} MAD should be non-negative"

