import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cogpy.datasets.schemas import EventCatalog, Events, Intervals, coerce_event_catalog, validate_event_catalog


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


# -----------------------------------------------------------------------------
# EventCatalog


def _make_table(n=3):
    rng = np.random.default_rng(42)
    t0 = np.sort(rng.uniform(0, 10, n))
    dur = rng.uniform(0.1, 0.5, n)
    t1 = t0 + dur
    t = t0 + dur / 2
    return pd.DataFrame(
        {
            "event_id": [f"e{i:03d}" for i in range(n)],
            "t": t,
            "t0": t0,
            "t1": t1,
            "duration": dur,
            "label": ["event"] * n,
            "score": rng.uniform(0.5, 2.0, n),
        }
    )


def _make_meta(n=3):
    return {
        "detector": "test_detector",
        "params": {},
        "fs": 1000.0,
        "n_events": n,
        "cogpy_version": "0.0.0",
    }


def test_event_catalog_construct():
    cat = EventCatalog(
        family="burst",
        table=_make_table(),
        meta=_make_meta(),
    )
    assert cat.family == "burst"
    assert len(cat.table) == 3


def test_event_catalog_required_columns():
    validate_event_catalog(
        EventCatalog(
            family="burst",
            table=_make_table(),
            meta=_make_meta(),
        )
    )  # must not raise


@pytest.mark.parametrize("col", ["event_id", "t", "t0", "t1", "duration", "label", "score"])
def test_event_catalog_missing_column(col):
    df = _make_table()
    df = df.drop(columns=[col])
    cat = EventCatalog(family="burst", table=df, meta=_make_meta())
    with pytest.raises(ValueError):
        validate_event_catalog(cat)


def test_event_catalog_t_outside_window():
    df = _make_table()
    df.loc[0, "t"] = df.loc[0, "t0"] - 0.1  # t < t0
    cat = EventCatalog(family="burst", table=df, meta=_make_meta())
    with pytest.raises(ValueError):
        validate_event_catalog(cat)


def test_event_catalog_t1_not_gt_t0():
    df = _make_table()
    df.loc[0, "t1"] = df.loc[0, "t0"]  # t1 == t0
    cat = EventCatalog(family="burst", table=df, meta=_make_meta())
    with pytest.raises(ValueError):
        validate_event_catalog(cat)


def test_event_catalog_duration_inconsistent():
    df = _make_table()
    df.loc[0, "duration"] = 999.0  # wrong
    cat = EventCatalog(family="burst", table=df, meta=_make_meta())
    with pytest.raises(ValueError):
        validate_event_catalog(cat)


def test_event_catalog_meta_n_events_mismatch():
    meta = _make_meta(n=3)
    meta["n_events"] = 99  # wrong
    cat = EventCatalog(family="burst", table=_make_table(3), meta=meta)
    with pytest.raises(ValueError):
        validate_event_catalog(cat)


def test_event_catalog_invalid_family():
    cat = EventCatalog(
        family="unknown_family",
        table=_make_table(),
        meta=_make_meta(),
    )
    with pytest.raises(ValueError):
        validate_event_catalog(cat)


def test_event_catalog_memberships_valid():
    df = _make_table(3)
    mem = pd.DataFrame(
        {
            "event_id": ["e000", "e001"],
            "channel": [0, 1],
        }
    )
    cat = EventCatalog(
        family="burst",
        table=df,
        meta=_make_meta(3),
        memberships=mem,
    )
    validate_event_catalog(cat)  # must not raise


def test_event_catalog_memberships_bad_event_id():
    df = _make_table(3)
    mem = pd.DataFrame(
        {
            "event_id": ["e000", "NONEXISTENT"],
            "channel": [0, 1],
        }
    )
    cat = EventCatalog(
        family="burst",
        table=df,
        meta=_make_meta(3),
        memberships=mem,
    )
    with pytest.raises(ValueError):
        validate_event_catalog(cat)


def test_event_catalog_to_events():
    cat = EventCatalog(
        family="burst",
        table=_make_table(),
        meta=_make_meta(),
    )
    evs = cat.to_events()
    assert isinstance(evs, Events)
    assert len(evs.times) == 3
    np.testing.assert_array_equal(evs.times, cat.table["t"].to_numpy())


def test_event_catalog_to_intervals():
    cat = EventCatalog(
        family="burst",
        table=_make_table(),
        meta=_make_meta(),
    )
    ivs = cat.to_intervals()
    assert isinstance(ivs, Intervals)
    np.testing.assert_array_equal(ivs.starts, cat.table["t0"].to_numpy())
    np.testing.assert_array_equal(ivs.ends, cat.table["t1"].to_numpy())


def test_event_catalog_to_array():
    cat = EventCatalog(
        family="burst",
        table=_make_table(),
        meta=_make_meta(),
    )
    arr = cat.to_array()
    assert arr.shape == (3, 2)
    np.testing.assert_array_equal(arr[:, 0], cat.table["t0"])
    np.testing.assert_array_equal(arr[:, 1], cat.table["t1"])


def test_event_catalog_to_event_stream():
    cat = EventCatalog(
        family="burst",
        table=_make_table(),
        meta=_make_meta(),
    )
    es = cat.to_event_stream()
    # EventStream must have event_id and t columns accessible
    assert "event_id" in es.df.columns or hasattr(es, "id_col")
    assert es.name == "burst"


def test_coerce_from_event_catalog():
    cat = EventCatalog(
        family="burst",
        table=_make_table(),
        meta=_make_meta(),
    )
    result = coerce_event_catalog(cat)
    assert result is cat or isinstance(result, EventCatalog)


def test_coerce_from_dataframe():
    df = _make_table()
    meta = _make_meta()
    result = coerce_event_catalog(df, family="ripple", meta=meta)
    assert isinstance(result, EventCatalog)
    assert result.family == "ripple"


def test_coerce_from_dataframe_missing_duration():
    df = _make_table()
    df = df.drop(columns=["duration"])
    meta = _make_meta()
    result = coerce_event_catalog(df, family="burst", meta=meta)
    assert "duration" in result.table.columns
    np.testing.assert_allclose(
        result.table["duration"],
        result.table["t1"] - result.table["t0"],
        atol=1e-9,
    )


def test_coerce_from_burst_dict():
    # minimal burst dict matching aggregate_bursts output format
    burst_dicts = [
        {
            "burst_id": "b000001",
            "t0_s": 1.0,
            "t1_s": 1.5,
            "t_peak_s": 1.25,
            "f0_hz": 30.0,
            "f1_hz": 80.0,
            "f_peak_hz": 55.0,
            "n_channels": 3,
            "ch_min": 0,
            "ch_max": 2,
        }
    ]
    meta = {
        "detector": "aggregate_bursts",
        "params": {},
        "fs": 1000.0,
        "n_events": 1,
        "cogpy_version": "0.0.0",
    }
    result = coerce_event_catalog(burst_dicts, meta=meta)
    assert isinstance(result, EventCatalog)
    assert result.family == "burst"
    assert result.table.loc[0, "event_id"] == "b000001"
    assert result.table.loc[0, "t0"] == 1.0
    assert result.table.loc[0, "t1"] == 1.5
    assert result.table.loc[0, "t"] == 1.25
    assert result.table.loc[0, "f0"] == 30.0

