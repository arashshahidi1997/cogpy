"""Tests for `cogpy.detect.BurstDetector`."""

from __future__ import annotations

import numpy as np
import pytest

from cogpy.detect import BurstDetector
from cogpy.events import EventCatalog


@pytest.fixture
def grid_data():
    xr = pytest.importorskip("xarray")
    data = np.random.RandomState(0).randn(1024, 4, 4)
    return xr.DataArray(
        data,
        dims=("time", "AP", "ML"),
        coords={"time": np.arange(1024) / 1000.0, "AP": np.arange(4), "ML": np.arange(4)},
        attrs={"fs": 1000.0},
    )


def test_burst_detector_creation():
    det = BurstDetector(h_quantile=0.9, nperseg=256)
    assert det.name == "BurstDetector"
    assert det.h_quantile == 0.9
    assert det.nperseg == 256


def test_burst_detector_can_accept_raw(grid_data):
    det = BurstDetector()
    assert det.can_accept(grid_data) is True
    assert det.needs_transform(grid_data) is True


def test_burst_detector_can_accept_spectrogram(grid_data):
    pytest.importorskip("ghostipy")
    xr = pytest.importorskip("xarray")
    from cogpy.spectral.specx import spectrogramx

    spec = spectrogramx(grid_data, nperseg=256, noverlap=128, bandwidth=4.0, axis="time")
    assert isinstance(spec, xr.DataArray)

    det = BurstDetector()
    assert det.can_accept(spec) is True
    assert det.needs_transform(spec) is False


def test_burst_detector_detect_implicit(grid_data):
    pytest.importorskip("ghostipy")
    det = BurstDetector(h_quantile=0.9, nperseg=256, noverlap=128, bandwidth=4.0)
    cat = det.detect(grid_data)

    assert isinstance(cat, EventCatalog)
    assert "event_id" in cat.df.columns
    assert "t" in cat.df.columns
    assert cat.metadata.get("computed_spectrogram") is True


def test_burst_detector_detect_explicit(grid_data):
    pytest.importorskip("ghostipy")
    from cogpy.spectral.specx import spectrogramx

    spec = spectrogramx(grid_data, nperseg=256, noverlap=128, bandwidth=4.0, axis="time")
    det = BurstDetector(h_quantile=0.9)
    cat = det.detect(spec)

    assert isinstance(cat, EventCatalog)
    assert "event_id" in cat.df.columns
    assert "t" in cat.df.columns
    assert cat.metadata.get("computed_spectrogram") is False


def test_burst_detector_serialization():
    det = BurstDetector(h_quantile=0.95, nperseg=512, noverlap=256, bandwidth=2.0)
    cfg = det.to_dict()
    assert cfg["detector"] == "BurstDetector"
    det2 = BurstDetector.from_dict(cfg)
    assert det2.h_quantile == 0.95
    assert det2.nperseg == 512


def test_burst_detector_transform_info():
    det = BurstDetector(nperseg=256, bandwidth=4.0)
    info = det.get_transform_info()
    assert info["required"] is True
    assert info["implicit"] is True
    assert info["explicit"] is True

