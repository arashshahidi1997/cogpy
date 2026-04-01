import numpy as np
import xarray as xr


def test_as_ieeg_time_channel_from_grid_ml_major():
    from cogpy.datasets.schemas import validate_ieeg_time_channel
    from cogpy.io.converters._xarray_views import as_ieeg_time_channel

    da = xr.DataArray(
        np.zeros((2, 2, 3)),
        dims=("time", "ML", "AP"),
        coords={"time": [0.0, 0.001], "ML": [10, 20], "AP": [1, 2, 3]},
    ).assign_attrs(fs=1000.0)

    out = as_ieeg_time_channel(da)
    validate_ieeg_time_channel(out)

    # ML-major ordering: channel = ml*n_ap + ap (AP varies fastest).
    assert tuple(out.dims) == ("time", "channel")
    assert np.array_equal(out["ML"].values, np.array([10, 10, 10, 20, 20, 20]))
    assert np.array_equal(out["AP"].values, np.array([1, 2, 3, 1, 2, 3]))


def test_as_ieeg_time_channel_accepts_time_ch():
    from cogpy.datasets.schemas import validate_ieeg_time_channel
    from cogpy.io.converters._xarray_views import as_ieeg_time_channel

    da = xr.DataArray(
        np.zeros((3, 4)),
        dims=("time", "ch"),
        coords={"time": [0.0, 0.001, 0.002], "ch": [0, 1, 2, 3]},
    ).assign_attrs(fs=1000.0)

    out = as_ieeg_time_channel(da)
    assert tuple(out.dims) == ("time", "channel")
    validate_ieeg_time_channel(out)
