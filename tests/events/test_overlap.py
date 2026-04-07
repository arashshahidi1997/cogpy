"""Tests for interval overlap detection (v2.6.3)."""

from __future__ import annotations

import pandas as pd

from cogpy.events import EventCatalog
from cogpy.events.overlap import detect_overlaps


def test_detect_overlaps_finds_pairs():
    df = pd.DataFrame(
        {
            "event_id": ["e0", "e1", "e2"],
            "t": [1.0, 1.4, 3.0],
            "t0": [0.8, 1.2, 2.8],
            "t1": [1.6, 1.9, 3.3],
        }
    )
    catalog = EventCatalog(df=df, name="x")

    overlaps = detect_overlaps(catalog)
    assert len(overlaps) == 1
    assert overlaps.iloc[0]["event_1"] == "e0"
    assert overlaps.iloc[0]["event_2"] == "e1"
    assert float(overlaps.iloc[0]["overlap_duration"]) > 0.0


def test_detect_overlaps_empty_for_non_interval():
    df = pd.DataFrame({"event_id": ["a", "b"], "t": [1.0, 2.0]})
    catalog = EventCatalog(df=df, name="p")
    overlaps = detect_overlaps(catalog)
    assert len(overlaps) == 0
