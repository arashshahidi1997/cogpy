"""Tests for `cogpy.core.events.EventCatalog`."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cogpy.core.events import EventCatalog


def test_eventcatalog_creation():
    df = pd.DataFrame({"event_id": [0, 1, 2], "t": [1.0, 2.0, 3.0]})
    catalog = EventCatalog(df=df, name="test")

    assert len(catalog) == 3
    assert catalog.name == "test"
    assert catalog.is_point_events
    assert not catalog.is_interval_events


def test_eventcatalog_requires_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        EventCatalog(df=pd.DataFrame({"t": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="Missing required columns"):
        EventCatalog(df=pd.DataFrame({"event_id": [0, 1]}))


def test_eventcatalog_sorts_by_time():
    df = pd.DataFrame({"event_id": [0, 1, 2], "t": [3.0, 1.0, 2.0]})
    catalog = EventCatalog(df=df)
    assert list(catalog.df["t"]) == [1.0, 2.0, 3.0]


def test_eventcatalog_validates_intervals():
    df = pd.DataFrame(
        {
            "event_id": [0, 1],
            "t": [1.5, 2.5],
            "t0": [1.0, 3.0],
            "t1": [2.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="t1 > t0"):
        EventCatalog(df=df)


def test_eventcatalog_auto_duration():
    df = pd.DataFrame(
        {
            "event_id": [0, 1],
            "t": [1.5, 2.5],
            "t0": [1.0, 2.0],
            "t1": [2.0, 3.0],
        }
    )
    catalog = EventCatalog(df=df)
    assert "duration" in catalog.df.columns
    assert list(catalog.df["duration"]) == [1.0, 1.0]


def test_from_hmaxima():
    peaks_df = pd.DataFrame({"t": [1.0, 2.0, 3.0], "AP": [1, 2, 3], "ML": [4, 5, 6], "amp": [10.0, 20.0, 15.0]})
    catalog = EventCatalog.from_hmaxima(peaks_df, label="burst_peak", detector="detect_hmaxima")

    assert len(catalog) == 3
    assert "event_id" in catalog.df.columns
    assert catalog.df["label"].iloc[0] == "burst_peak"
    assert "value" in catalog.df.columns
    assert catalog.metadata["detector"] == "detect_hmaxima"


def test_from_blob_candidates():
    blob_dict = {
        "t_peak_s": np.array([1.5, 2.5, 3.5]),
        "t0_s": np.array([1.0, 2.0, 3.0]),
        "t1_s": np.array([2.0, 3.0, 4.0]),
        "f_peak_hz": np.array([40.0, 45.0, 50.0]),
        "f0_hz": np.array([35.0, 40.0, 45.0]),
        "f1_hz": np.array([45.0, 50.0, 55.0]),
        "channel": np.array([0, 1, 2]),
        "score": np.array([0.9, 0.85, 0.95]),
    }
    catalog = EventCatalog.from_blob_candidates(blob_dict)

    assert len(catalog) == 3
    assert catalog.is_interval_events
    assert "duration" in catalog.df.columns
    assert "bandwidth" in catalog.df.columns


def test_from_burst_dict():
    burst_list = [
        {
            "burst_id": "b000001",
            "t_peak_s": 1.5,
            "t0_s": 1.0,
            "t1_s": 2.0,
            "f_peak_hz": 40.0,
            "f0_hz": 35.0,
            "f1_hz": 45.0,
            "n_channels": 3,
        }
    ]
    catalog = EventCatalog.from_burst_dict(burst_list, label="burst")
    assert len(catalog) == 1
    assert catalog.df["event_id"].iloc[0] == "b000001"
    assert "duration" in catalog.df.columns
    assert "bandwidth" in catalog.df.columns


def test_to_events():
    df = pd.DataFrame({"event_id": [0, 1, 2], "t": [1.0, 2.0, 3.0], "label": ["a", "b", "c"]})
    catalog = EventCatalog(df=df)
    events = catalog.to_events()

    assert len(events.times) == 3
    assert list(events.times) == [1.0, 2.0, 3.0]
    assert list(events.labels) == ["a", "b", "c"]


def test_to_intervals():
    df = pd.DataFrame({"event_id": [0, 1], "t": [1.5, 2.5], "t0": [1.0, 2.0], "t1": [2.0, 3.0]})
    catalog = EventCatalog(df=df)
    intervals = catalog.to_intervals()

    assert len(intervals.starts) == 2
    assert list(intervals.starts) == [1.0, 2.0]
    assert list(intervals.ends) == [2.0, 3.0]


def test_to_intervals_requires_t0_t1():
    df = pd.DataFrame({"event_id": [0, 1], "t": [1.0, 2.0]})
    catalog = EventCatalog(df=df)
    with pytest.raises(ValueError, match="lack t0/t1 columns"):
        catalog.to_intervals()


def test_to_point_intervals():
    df = pd.DataFrame({"event_id": [0, 1], "t": [1.0, 2.0]})
    catalog = EventCatalog(df=df)
    intervals = catalog.to_point_intervals(half_window=0.5)

    assert list(intervals.starts) == [0.5, 1.5]
    assert list(intervals.ends) == [1.5, 2.5]


def test_to_event_stream():
    df = pd.DataFrame({"event_id": [0, 1], "t": [1.0, 2.0]})
    catalog = EventCatalog(df=df, name="test_events")
    stream = catalog.to_event_stream()

    assert stream.name == "test_events"
    assert len(stream.df) == 2


def test_filter_by_time():
    df = pd.DataFrame({"event_id": [0, 1, 2, 3], "t": [1.0, 2.0, 3.0, 4.0]})
    catalog = EventCatalog(df=df)
    filtered = catalog.filter_by_time(1.5, 3.5)

    assert len(filtered) == 2
    assert list(filtered.df["t"]) == [2.0, 3.0]


def test_filter_by_channel():
    df = pd.DataFrame({"event_id": [0, 1, 2], "t": [1.0, 2.0, 3.0], "channel": [0, 1, 0]})
    catalog = EventCatalog(df=df)

    filtered = catalog.filter_by_channel(0)
    assert len(filtered) == 2

    filtered = catalog.filter_by_channel([0, 1])
    assert len(filtered) == 3


def test_filter_by_spatial_radius():
    df = pd.DataFrame({"event_id": [0, 1, 2], "t": [1.0, 2.0, 3.0], "AP": [0.0, 1.0, 5.0], "ML": [0.0, 1.0, 5.0]})
    catalog = EventCatalog(df=df)

    filtered = catalog.filter_by_spatial(AP=0.5, ML=0.5, radius=1.0)
    assert len(filtered) == 2

