"""Tests for `cogpy.core.detect.EventDetector`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cogpy.core.detect.base import EventDetector
from cogpy.core.events import EventCatalog


class DummyDetector(EventDetector):
    def __init__(self, threshold: float = 0.5):
        super().__init__("DummyDetector")
        self.threshold = float(threshold)
        self.params = {"threshold": self.threshold}

    def detect(self, data, **_kwargs):
        df = pd.DataFrame({"event_id": [0, 1], "t": [1.0, 2.0]})
        return EventCatalog(df=df, name="dummy_events")

    def get_event_dims(self):
        return ["time"]


def test_eventdetector_abstract():
    with pytest.raises(TypeError):
        EventDetector()  # type: ignore[abstract]


def test_eventdetector_subclass():
    det = DummyDetector(threshold=0.7)
    assert det.name == "DummyDetector"
    assert det.threshold == 0.7
    assert det.params == {"threshold": 0.7}


def test_eventdetector_detect_returns_catalog():
    det = DummyDetector()
    data = np.random.randn(10)
    out = det.detect(data)
    assert isinstance(out, EventCatalog)
    assert len(out) == 2


def test_eventdetector_serialization_roundtrip():
    det = DummyDetector(threshold=0.7)
    cfg = det.to_dict()
    assert cfg == {"detector": "DummyDetector", "params": {"threshold": 0.7}}

    det2 = DummyDetector.from_dict(cfg)
    assert det2.threshold == 0.7


def test_eventdetector_defaults():
    det = DummyDetector()
    assert det.can_accept(None) is True
    assert det.needs_transform(None) is False
    info = det.get_transform_info()
    assert info["required"] is False
    assert info["transform_type"] is None

