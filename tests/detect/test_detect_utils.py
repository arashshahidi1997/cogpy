"""Tests for cogpy.detect.utils — detection utilities."""

import numpy as np
import pytest

from cogpy.detect.utils import (
    dual_threshold_events_1d,
    merge_intervals,
    score_to_bouts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ramp_score(n=100, fs=10.0):
    """Score that ramps up, plateaus, ramps down — one clear bout."""
    times = np.arange(n) / fs
    score = np.zeros(n)
    # Bout from sample 20..60
    score[20:60] = np.concatenate([
        np.linspace(0, 5, 20),   # ramp up
        np.linspace(5, 0, 20),   # ramp down
    ])
    return score, times


def _two_bouts_score(n=200, fs=10.0, gap_samples=10):
    """Two bouts separated by a controllable gap."""
    times = np.arange(n) / fs
    score = np.zeros(n)
    # Bout 1: samples 20..40
    score[20:40] = 3.0
    score[28:32] = 6.0  # peak above high threshold
    # Bout 2: samples (40+gap)..(60+gap)
    start2 = 40 + gap_samples
    end2 = start2 + 20
    if end2 <= n:
        score[start2:end2] = 3.0
        mid2 = start2 + 8
        score[mid2:mid2 + 4] = 6.0
    return score, times


# ---------------------------------------------------------------------------
# score_to_bouts
# ---------------------------------------------------------------------------

class TestScoreToBouts:
    def test_single_bout(self):
        """Single bout detected from ramp score."""
        score, times = _ramp_score()
        events = score_to_bouts(score, times, low=1.0, high=3.0)
        assert len(events) >= 1
        e = events[0]
        assert "t0" in e and "t1" in e and "t" in e and "value" in e and "duration" in e
        assert e["value"] >= 3.0
        assert e["duration"] > 0

    def test_no_events_below_threshold(self):
        """Score never exceeds high → no events."""
        score = np.ones(100) * 0.5
        times = np.arange(100) / 10.0
        events = score_to_bouts(score, times, low=1.0, high=2.0)
        assert len(events) == 0

    def test_min_duration_filter(self):
        """Short bouts filtered by min_duration."""
        score, times = _ramp_score()
        # Get unfiltered events
        events_all = score_to_bouts(score, times, low=1.0, high=3.0)
        assert len(events_all) >= 1
        # Set min_duration longer than the bout
        events_filtered = score_to_bouts(
            score, times, low=1.0, high=3.0, min_duration=100.0
        )
        assert len(events_filtered) == 0

    def test_merge_gap(self):
        """Two close bouts merged into one when merge_gap is large enough."""
        # Gap of 5 samples at fs=10 → 0.5 sec gap
        score, times = _two_bouts_score(gap_samples=5)
        # Without merging: should find 2 events
        events_no_merge = score_to_bouts(score, times, low=2.0, high=5.0)
        # With merging at gap > 0.5s: should find 1 event
        events_merged = score_to_bouts(
            score, times, low=2.0, high=5.0, merge_gap=1.0
        )
        assert len(events_no_merge) >= 2
        assert len(events_merged) == 1

    def test_no_merge_when_gap_zero(self):
        """merge_gap=0 → no merging, same as dual_threshold_events_1d."""
        score, times = _two_bouts_score(gap_samples=5)
        events = score_to_bouts(score, times, low=2.0, high=5.0, merge_gap=0.0)
        events_raw = dual_threshold_events_1d(score, times, low=2.0, high=5.0)
        assert len(events) == len(events_raw)

    def test_empty_score(self):
        """Empty arrays → no events."""
        events = score_to_bouts(np.array([]), np.array([]), low=1.0, high=2.0)
        assert events == []

    def test_event_dict_keys(self):
        """Each event has the expected keys."""
        score, times = _ramp_score()
        events = score_to_bouts(score, times, low=1.0, high=3.0)
        if events:
            keys = set(events[0].keys())
            assert keys == {"t0", "t1", "t", "value", "duration"}


# ---------------------------------------------------------------------------
# merge_intervals (smoke tests for existing function)
# ---------------------------------------------------------------------------

class TestMergeIntervals:
    def test_no_merge(self):
        intervals = [(0, 5), (10, 15)]
        result = merge_intervals(intervals, gap=2)
        assert len(result) == 2

    def test_merge_close(self):
        intervals = [(0, 5), (7, 15)]
        result = merge_intervals(intervals, gap=2)
        assert len(result) == 1
        assert result[0] == (0, 15)

    def test_empty(self):
        assert merge_intervals([], gap=5) == []


# ---------------------------------------------------------------------------
# bout_occupancy / bout_duration_summary
# ---------------------------------------------------------------------------

class TestBoutOccupancy:
    def test_basic(self):
        from cogpy.detect.utils import bout_occupancy

        bouts = [{"duration": 1.0}, {"duration": 2.0}]
        assert bout_occupancy(bouts, 10.0) == pytest.approx(0.3)

    def test_empty(self):
        from cogpy.detect.utils import bout_occupancy

        assert bout_occupancy([], 10.0) == 0.0

    def test_full_occupancy_clamped(self):
        from cogpy.detect.utils import bout_occupancy

        bouts = [{"duration": 15.0}]
        assert bout_occupancy(bouts, 10.0) == pytest.approx(1.0)

    def test_zero_total_raises(self):
        from cogpy.detect.utils import bout_occupancy

        with pytest.raises(ValueError):
            bout_occupancy([], 0.0)


class TestBoutDurationSummary:
    def test_basic_stats(self):
        from cogpy.detect.utils import bout_duration_summary

        bouts = [{"duration": 1.0}, {"duration": 2.0}, {"duration": 3.0}]
        s = bout_duration_summary(bouts)
        assert s["count"] == 3
        assert s["mean"] == pytest.approx(2.0)
        assert s["median"] == pytest.approx(2.0)
        assert set(s.keys()) == {"count", "mean", "median", "std", "p5", "p95"}

    def test_empty(self):
        from cogpy.detect.utils import bout_duration_summary

        s = bout_duration_summary([])
        assert s["count"] == 0
        assert s["mean"] == 0.0

    def test_single_bout(self):
        from cogpy.detect.utils import bout_duration_summary

        s = bout_duration_summary([{"duration": 5.0}])
        assert s["count"] == 1
        assert s["mean"] == pytest.approx(5.0)
        assert s["std"] == 0.0
