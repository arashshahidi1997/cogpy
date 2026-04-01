"""Tests for DetectionPipeline (v2.6.5)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cogpy.detect import DetectionPipeline, ThresholdDetector
from cogpy.detect.transforms import BandpassTransform, ZScoreTransform
from cogpy.events import EventCatalog


@pytest.fixture
def signal_data():
    fs = 1000.0
    t = np.arange(0.0, 2.0, 1.0 / fs, dtype=float)
    y = 0.05 * np.random.RandomState(0).randn(t.size)
    y[500:520] += 2.5
    return xr.DataArray(y, dims=["time"], coords={"time": t}, attrs={"fs": fs})


def test_pipeline_roundtrip_serialization():
    pipe = DetectionPipeline(
        transforms=[BandpassTransform(low=1.0, high=100.0, order=3), ZScoreTransform()],
        detector=ThresholdDetector(
            threshold=2.0, direction="positive", min_duration=0.005
        ),
        name="x",
    )
    cfg = pipe.to_dict()
    pipe2 = DetectionPipeline.from_dict(cfg)
    assert pipe2.name == "x"
    assert len(pipe2.transforms) == 2
    assert pipe2.detector is not None


def test_pipeline_run_returns_catalog(signal_data):
    pipe = DetectionPipeline(
        transforms=[ZScoreTransform()],
        detector=ThresholdDetector(
            threshold=2.0, direction="positive", min_duration=0.005
        ),
        name="thr",
    )
    cat = pipe.run(signal_data)
    assert isinstance(cat, EventCatalog)
    assert "pipeline" in cat.metadata
    assert cat.metadata["pipeline"] == "thr"
