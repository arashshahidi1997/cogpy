"""Tests for ThresholdDetector (v2.6.4)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cogpy.detect.threshold import ThresholdDetector
from cogpy.events import EventCatalog


@pytest.fixture
def signal_data():
    t = np.linspace(0, 10, 1000, dtype=float)
    y = np.zeros_like(t)
    y[200:250] = 10.0
    y[500:550] = 8.0
    y[800:850] = 12.0
    return xr.DataArray(y, dims=["time"], coords={"time": t})


def test_threshold_detector_creation():
    det = ThresholdDetector(threshold=5.0)
    assert det.threshold == 5.0
    assert det.direction == "both"


def test_threshold_detector_detect_intervals(signal_data):
    det = ThresholdDetector(threshold=5.0, direction="positive", min_duration=0.01)
    catalog = det.detect(signal_data)
    assert isinstance(catalog, EventCatalog)
    assert len(catalog) >= 3
    assert catalog.is_interval_events
    assert (catalog.df["t1"] >= catalog.df["t0"]).all()


def test_threshold_detector_serialization_roundtrip():
    det = ThresholdDetector(threshold=3.0, direction="negative", merge_gap=0.05)
    cfg = det.to_dict()
    det2 = ThresholdDetector.from_dict(cfg)
    assert det2.threshold == det.threshold
    assert det2.direction == det.direction
    assert det2.merge_gap == det.merge_gap
