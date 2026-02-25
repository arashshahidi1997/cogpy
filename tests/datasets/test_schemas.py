import numpy as np
import pandas as pd
import pytest
import xarray as xr


def test_validate_ieeg_grid_hint_on_wrong_dim_order():
    from cogpy.datasets.schemas import validate_ieeg_grid

    da = xr.DataArray(
        np.zeros((3, 4, 5)),
        dims=("time", "ML", "AP"),
        coords={"time": [0, 1, 2], "ML": range(4), "AP": range(5)},
    ).assign_attrs(fs=1000.0)

    bad = da.transpose("AP", "time", "ML")
    with pytest.raises(ValueError) as e:
        validate_ieeg_grid(bad)
    assert "Hint:" in str(e.value)
    assert "transpose('time', 'ML', 'AP')" in str(e.value)


def test_coerce_ieeg_grid_transposes_and_injects_fs():
    from cogpy.datasets.schemas import coerce_ieeg_grid

    da = xr.DataArray(
        np.zeros((5, 3, 4)),
        dims=("AP", "time", "ML"),
        coords={"time": [0, 1, 2], "ML": range(4), "AP": range(5)},
    )
    out = coerce_ieeg_grid(da, fs=1000.0)
    assert tuple(out.dims) == ("time", "ML", "AP")
    assert out.attrs["fs"] == 1000.0


def test_validate_burst_peaks_columns():
    from cogpy.datasets.schemas import validate_burst_peaks

    df = pd.DataFrame({"burst_id": [0], "x": [0.0], "y": [0.0], "t": [0.0], "z": [1.0], "value": [2.0]})
    validate_burst_peaks(df)

    with pytest.raises(ValueError):
        validate_burst_peaks(pd.DataFrame({"burst_id": [0]}))


def test_validate_ieeg_time_channel_accepts_reset_index_form():
    from cogpy.datasets.schemas import validate_ieeg_time_channel

    base = xr.DataArray(
        np.random.randn(3, 2, 2),
        dims=("time", "AP", "ML"),
        coords={"time": [0, 1, 2], "AP": [0, 1], "ML": [0, 1]},
    ).assign_attrs(fs=1000.0)

    tc = base.stack(channel=("AP", "ML")).reset_index("channel")
    tc = tc.transpose("time", "channel")
    validate_ieeg_time_channel(tc)

