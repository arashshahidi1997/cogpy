import numpy as np
import pytest
import xarray as xr


def test_to_mne_constructs_rawarray():
    pytest.importorskip("mne")

    from cogpy.io.converters import to_mne

    da = xr.DataArray(
        np.random.randn(10, 3),
        dims=("time", "channel"),
        coords={"time": np.arange(10, dtype=float) / 1000.0, "channel": [0, 1, 2]},
    ).assign_attrs(fs=1000.0)

    raw = to_mne(da, unit="V")
    assert raw.n_times == 10
    assert len(raw.ch_names) == 3
    assert raw.info["sfreq"] == 1000.0

