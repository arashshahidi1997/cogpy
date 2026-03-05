"""Tests for RippleDetector / SpindleDetector (v2.6.4)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cogpy.core.detect.ripple import RippleDetector, SpindleDetector
from cogpy.core.events import EventCatalog


@pytest.fixture
def lfp_data():
    fs = 1000.0
    t = np.arange(0.0, 10.0, 1.0 / fs, dtype=float)
    y = np.zeros_like(t)

    # Add a few high-frequency bursts (ripple-like).
    f0 = 150.0
    for t0 in (2.0, 5.0, 8.0):
        mask = (t >= t0) & (t < t0 + 0.08)
        y[mask] += 3.0 * np.sin(2 * np.pi * f0 * (t[mask] - t0))

    # Small noise for stable z-score.
    y += 0.05 * np.random.RandomState(0).randn(t.size)

    return xr.DataArray(y, dims=["time"], coords={"time": t}, attrs={"fs": fs})


def test_ripple_detector_creation():
    det = RippleDetector()
    assert det.freq_range == (100.0, 250.0)
    assert det.threshold_low == 2.0


def test_ripple_detector_detect_returns_intervals(lfp_data):
    det = RippleDetector(freq_range=(100, 250), threshold_low=1.5, threshold_high=2.5)
    catalog = det.detect(lfp_data)
    assert isinstance(catalog, EventCatalog)
    assert catalog.is_interval_events or len(catalog) == 0
    if len(catalog):
        assert (catalog.df["duration"] > 0).all()


def test_spindle_detector_smoke(lfp_data):
    # Not a true spindle signal, but the detector should run and return an EventCatalog.
    det = SpindleDetector(threshold_low=2.0, threshold_high=3.0)
    catalog = det.detect(lfp_data)
    assert isinstance(catalog, EventCatalog)

