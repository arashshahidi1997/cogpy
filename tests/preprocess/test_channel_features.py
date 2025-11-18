# tests/test_channel_features.py
import pytest
import numpy as np
import cogpy.utils.grid_neighborhood as gn
from cogpy.preprocess import channel_feature_functions as ft
from cogpy.datasets import load as ld
from cogpy.utils.sliding import rolling_win
import dask.array as da
from cogpy.preprocess.channel_feature_functions import (
    local_robust_zscore,
    local_robust_zscore_dask,
    make_footprint,
)

# from cogpy.preprocess.channel_features import fp_inc, loc_exclude_df, gradient_ndi, gradient
# # ---- TEST harness ----
# def test_gradient_equivalence(H=16, W=16, T=200, seed=0, tol=1e-10):
#     rng = np.random.default_rng(seed)
#     a = rng.normal(size=(T, H * W))  # (time, channel)

#     g_old = gradient(a, (H, W))
#     g_new = gradient_ndi(a, (H, W), mode="mirror")

#     # Exclude the outer 2 pixels (since footprint radius=2)
#     margin = 3
#     interior_old = g_old[margin:-margin, margin:-margin]
#     interior_new = g_new[margin:-margin, margin:-margin]

#     # Assert shapes match
#     assert interior_old.shape == interior_new.shape

#     # Assert values agree up to tolerance
#     assert np.allclose(interior_old, interior_new, atol=tol), \
#         f"Max diff in interior = {np.abs(interior_old - interior_new).max()}"


def test_noise_to_signal():
    a = np.random.randn(10000)
    fs = 650
    nsr = ft.noise_to_signal(a, fs)
    assert np.isclose(nsr, 1, atol=0.3), print(nsr)


@pytest.fixture(scope="module")
def xsig():
    return ld.load_sample()


@pytest.fixture(scope="module")
def xwin(xsig):
    # keep the same chunking you rely on
    return rolling_win(xsig, window_size=512, window_step=64, dim="time").chunk(
        {"AP": -1, "ML": -1, "window": -1, "time": 3}
    )


@pytest.fixture(scope="module")
def gneigh():
    footprint = gn.make_footprint(2, 1, 2)
    return gn.GridNeighborhood(AP=16, ML=16, footprint=footprint)


# ---------- temporal features ----------
TEMP_FEATURES = [
    (
        "noise_to_signal",
        lambda x, **k: ft.noise_to_signal(x.data, fs=k["fs"]),
        (16, 16, 7),
    ),
    ("relative_variance", lambda x, **k: ft.relative_variance(x.data), (16, 16, 7)),
    ("deviation", lambda x, **k: ft.deviation(x.data), (16, 16, 7)),
    ("standard_deviation", lambda x, **k: ft.standard_deviation(x.data), (16, 16, 7)),
    ("amplitude", lambda x, **k: ft.amplitude(x.data), (16, 16, 7)),
    ("kurtosis", lambda x, **k: ft.kurtosis(x.data), (16, 16, 7)),
    ("time_derivative", lambda x, **k: ft.time_derivative(x.data), (16, 16, 7)),
    ("hurst_exponent", lambda x, **k: ft.hurst_exponent(x.data), (16, 16, 7)),
    # ("is_dead",              lambda x, **k: ft.is_dead(x.data),                     (16,16,7)),
]


@pytest.mark.parametrize(
    "name,fn,expected", TEMP_FEATURES, ids=[n for n, _, _ in TEMP_FEATURES]
)
def test_temporal_shapes(xwin, xsig, name, fn, expected):
    out = fn(xwin, fs=xsig.fs)
    assert (
        getattr(out, "shape", None) == expected
    ), f"{name}: got {getattr(out,'shape',None)}, expected {expected}"


# ---------- spatial features ----------
SPATIAL_FEATURES = [
    (
        "anticorrelation",
        lambda x, n: ft.anticorrelation(x.isel(time=0).data, n.adj),
        (16, 16),
    ),
    # ("laplacian",                  lambda x, n: ft.laplacian(x.isel(time=0), n.grid_shape),                  (16,16,512)),
    (
        "temporal_mean_laplacian",
        lambda x, n: ft.temporal_mean_laplacian(x.isel(time=0)),
        (16, 16),
    ),
    # ("spatial_gradient",           lambda x, n: ft.spatial_gradient(x.isel(time=0)),                         (16,16)),
    (
        "gradient_rms_fast",
        lambda x, n: ft.gradient_rms_fast(x.isel(time=0).data, n.adj_src, n.adj_dst),
        (16, 16),
    ),
]


@pytest.mark.parametrize(
    "name,fn,expected", SPATIAL_FEATURES, ids=[n for n, _, _ in SPATIAL_FEATURES]
)
def test_spatial_shapes(xwin, gneigh, name, fn, expected):
    out = fn(xwin, gneigh)
    assert (
        getattr(out, "shape", None) == expected
    ), f"{name}: got {getattr(out,'shape',None)}, expected {expected}"


def test_random_array_matches_numpy_and_dask():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 16))
    footprint = make_footprint(rank=2, connectivity=1, niter=2)

    z_np = local_robust_zscore(x, footprint=footprint)
    z_da = local_robust_zscore_dask(x, footprint=footprint).compute()

    assert np.allclose(z_np, z_da, equal_nan=True)


def test_uniform_array_is_all_nan():
    x = np.ones((10, 10))
    footprint = make_footprint(rank=2, connectivity=1, niter=2)

    z_np = local_robust_zscore(x, footprint=footprint)
    z_da = local_robust_zscore_dask(x, footprint=footprint).compute()

    # Both should return all NaN
    assert np.all(np.isnan(z_np))
    assert np.all(np.isnan(z_da))
