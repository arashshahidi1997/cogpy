import numpy as np
import pytest

from cogpy.preprocess.interpolate import (
    interpolate_bads,
    interpolate_bads_coords,
    interpolate_bads_xarray,
)


# %% test
def test_interp():
    data = np.ones((3, 3, 1)) * np.array([1, 10, 3, 96]).reshape(1, 1, -1)
    nan_mask_ = np.zeros((3, 3), dtype=bool)
    nan_mask_[1, 1] = True
    nan_mask_[0, 0] = True
    idata = interpolate_bads(data, skip=nan_mask_, method="linear", gridshape=(3, 3))
    assert np.all(idata == data), print(idata)
    print("interpolation test passed")


# ── interpolate_bads_coords ─────────────────────────────────────────────


def test_coords_no_bad_returns_copy():
    """When no channels are bad, return a copy of the input unchanged."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    arr = np.array([[1.0], [2.0], [3.0], [4.0]])
    bad = np.zeros(4, dtype=bool)

    out = interpolate_bads_coords(arr, coords, bad)

    assert np.array_equal(out, arr)
    assert out is not arr  # copy, not the same object


def test_coords_linear_field_exact_reconstruction():
    """Linear interpolation reproduces a linear field exactly."""
    # 4x4 uniform grid
    coords = np.array(
        [[float(i), float(j)] for i in range(4) for j in range(4)]
    )
    # Linear field: f(x, y) = 2x + 3y
    values = 2 * coords[:, 0] + 3 * coords[:, 1]
    # Add a trailing dimension (e.g. time)
    arr = np.tile(values[:, None], (1, 5))

    # Interior bad point so it falls inside the convex hull
    bad = np.zeros(16, dtype=bool)
    bad[5] = True  # (1, 1) → value 2+3 = 5

    out = interpolate_bads_coords(arr.copy(), coords, bad)

    np.testing.assert_allclose(out[5], values[5], atol=1e-10)
    # Unchanged channels remain unchanged
    np.testing.assert_array_equal(out[~bad], arr[~bad])


def test_coords_preserves_trailing_dims():
    """Multi-dimensional trailing axes are preserved (vectorized over time)."""
    coords = np.array(
        [[float(i), float(j)] for i in range(4) for j in range(4)]
    )
    # Time-varying linear field: f(x, y, t) = x + y*t
    n_t = 7
    t = np.arange(n_t, dtype=float)
    arr = coords[:, 0:1] + coords[:, 1:2] * t[None, :]

    bad = np.zeros(16, dtype=bool)
    bad[5] = True  # interior
    bad[10] = True

    out = interpolate_bads_coords(arr.copy(), coords, bad)

    # At each time t, the field is still linear and should reconstruct exactly
    for ti in range(n_t):
        np.testing.assert_allclose(out[5, ti], arr[5, ti], atol=1e-10)
        np.testing.assert_allclose(out[10, ti], arr[10, ti], atol=1e-10)


def test_coords_nonuniform_geometry():
    """Non-grid coordinates (e.g. hemispheric gap) work without error."""
    # Two clusters with a gap, one interior point per cluster
    left = np.array(
        [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0],
         [1.0, 0.0], [1.0, 1.0], [1.0, 2.0],
         [2.0, 0.0], [2.0, 1.0], [2.0, 2.0]]
    )
    right = left.copy()
    right[:, 0] += 10.0  # shift right cluster by 10 units (large gap)
    coords = np.vstack([left, right])

    # Linear field on x
    values = coords[:, 0].copy()
    arr = values[:, None]  # (18, 1)

    # Interior point of left cluster
    bad = np.zeros(18, dtype=bool)
    bad[4] = True  # left cluster (1, 1) → value 1

    out = interpolate_bads_coords(arr, coords, bad)

    # Should interpolate to 1 (from left cluster neighbors),
    # NOT to 6 (which would be halfway across the gap to the right cluster)
    np.testing.assert_allclose(out[4, 0], 1.0, atol=1e-10)


def test_coords_edge_fallback_to_nearest():
    """Bad channels outside the convex hull get filled via 'nearest' fallback."""
    # Triangle + one bad point outside its convex hull
    coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]
    )
    values = np.array([1.0, 2.0, 3.0, 99.0])
    arr = values[:, None]

    bad = np.zeros(4, dtype=bool)
    bad[3] = True  # (5, 5) — far outside the triangle

    # With default fill_method='nearest', NaN should be filled
    out = interpolate_bads_coords(arr, coords, bad)
    assert not np.isnan(out[3, 0])

    # With fill_method=None, NaN should remain
    out_nan = interpolate_bads_coords(arr, coords, bad, fill_method=None)
    assert np.isnan(out_nan[3, 0])


def test_coords_shape_validation():
    """Mismatched shapes raise ValueError."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    arr = np.array([[1.0], [2.0], [3.0]])  # 3 channels but coords only has 2
    bad = np.array([True, False, False])

    with pytest.raises(ValueError, match="n_channels"):
        interpolate_bads_coords(arr, coords, bad)

    # 3D coords (should be 2D with shape (n, 2))
    coords_3d = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    arr2 = np.array([[1.0], [2.0]])
    bad2 = np.array([True, False])
    with pytest.raises(ValueError, match="shape"):
        interpolate_bads_coords(arr2, coords_3d, bad2)


# ── interpolate_bads_xarray ─────────────────────────────────────────────


def test_xarray_reads_coords_from_dataarray():
    """The xarray wrapper reads (x, y) from non-dimension coordinates."""
    xr = pytest.importorskip("xarray")

    # 4x4 grid flattened to a 'ch' dimension
    coords = np.array(
        [[float(i), float(j)] for i in range(4) for j in range(4)]
    )
    values = 2 * coords[:, 0] + 3 * coords[:, 1]
    arr = np.tile(values[:, None], (1, 5))

    sig = xr.DataArray(
        arr,
        dims=("ch", "time"),
        coords={
            "x": ("ch", coords[:, 0]),
            "y": ("ch", coords[:, 1]),
            "time": np.arange(5),
        },
    )

    bad = np.zeros(16, dtype=bool)
    bad[5] = True

    out = interpolate_bads_xarray(sig, bad)

    assert out.dims == sig.dims
    assert out.shape == sig.shape
    np.testing.assert_allclose(out.values[5], values[5], atol=1e-10)


def test_xarray_transpose_robustness():
    """Works when ch is not the first dimension."""
    xr = pytest.importorskip("xarray")

    coords = np.array(
        [[float(i), float(j)] for i in range(4) for j in range(4)]
    )
    values = coords[:, 0] + coords[:, 1]
    arr = np.tile(values[None, :], (5, 1))  # (time, ch) instead of (ch, time)

    sig = xr.DataArray(
        arr,
        dims=("time", "ch"),
        coords={
            "x": ("ch", coords[:, 0]),
            "y": ("ch", coords[:, 1]),
            "time": np.arange(5),
        },
    )

    bad = np.zeros(16, dtype=bool)
    bad[5] = True

    out = interpolate_bads_xarray(sig, bad)

    # Original dim order preserved
    assert out.dims == ("time", "ch")
    np.testing.assert_allclose(out.values[:, 5], values[5], atol=1e-10)


def test_xarray_missing_coords_raises():
    """Missing x/y coordinates raise a clear error."""
    xr = pytest.importorskip("xarray")

    sig = xr.DataArray(
        np.zeros((4, 3)),
        dims=("ch", "time"),
        coords={"time": np.arange(3)},
    )
    with pytest.raises(ValueError, match="'x' coordinate"):
        interpolate_bads_xarray(sig, np.zeros(4, dtype=bool))
