"""Tests for RippleDetector / SpindleDetector (v2.6.4)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cogpy.detect.ripple import RippleDetector, SpindleDetector
from cogpy.events import EventCatalog


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


# ---------------------------------------------------------------------------
# Spindle signal fixture (13 Hz bursts, spindle-like)
# ---------------------------------------------------------------------------

@pytest.fixture
def spindle_data():
    """Synthetic signal with 13 Hz spindle bursts."""
    fs = 1000.0
    t = np.arange(0.0, 20.0, 1.0 / fs, dtype=float)
    y = np.zeros_like(t)

    # Three spindle bursts at 13 Hz, duration ~1s, well separated.
    f0 = 13.0
    for t0 in (3.0, 8.0, 14.0):
        dur = 1.0
        mask = (t >= t0) & (t < t0 + dur)
        # Gaussian-modulated sinusoid (waxing-waning)
        t_local = t[mask] - t0
        envelope = np.exp(-((t_local - dur / 2) ** 2) / (2 * (dur / 6) ** 2))
        y[mask] += 5.0 * envelope * np.sin(2 * np.pi * f0 * t_local)

    y += 0.05 * np.random.RandomState(42).randn(t.size)
    return xr.DataArray(y, dims=["time"], coords={"time": t}, attrs={"fs": fs})


# ---------------------------------------------------------------------------
# Enrichment tests
# ---------------------------------------------------------------------------

def test_spindle_no_enrichment_by_default(spindle_data):
    """Without enrichment flags, no extra columns are added."""
    det = SpindleDetector(threshold_low=1.5, threshold_high=2.5)
    cat = det.detect(spindle_data)
    assert isinstance(cat, EventCatalog)
    for col in ("frequency", "rel_power", "symmetry"):
        assert col not in cat.df.columns


def test_spindle_frequency_enrichment(spindle_data):
    det = SpindleDetector(
        threshold_low=1.5, threshold_high=2.5, compute_frequency=True,
    )
    cat = det.detect(spindle_data)
    assert isinstance(cat, EventCatalog)
    if len(cat) > 0:
        assert "frequency" in cat.df.columns
        # Frequencies should be in a plausible spindle range
        freqs = cat.df["frequency"].dropna()
        assert len(freqs) > 0
        assert (freqs > 5).all() and (freqs < 25).all()


def test_spindle_rel_power_enrichment(spindle_data):
    det = SpindleDetector(
        threshold_low=1.5, threshold_high=2.5, compute_rel_power=True,
    )
    cat = det.detect(spindle_data)
    if len(cat) > 0:
        assert "rel_power" in cat.df.columns
        rp = cat.df["rel_power"].dropna()
        assert len(rp) > 0
        # Relative power should be between 0 and some reasonable upper bound
        assert (rp > 0).all()


def test_spindle_rel_power_threshold(spindle_data):
    """rel_power_min should reject events below the threshold."""
    det_no_thresh = SpindleDetector(
        threshold_low=1.5, threshold_high=2.5, compute_rel_power=True,
    )
    cat_all = det_no_thresh.detect(spindle_data)

    det_high_thresh = SpindleDetector(
        threshold_low=1.5, threshold_high=2.5,
        compute_rel_power=True, rel_power_min=0.99,
    )
    cat_strict = det_high_thresh.detect(spindle_data)
    assert len(cat_strict) <= len(cat_all)


def test_spindle_symmetry_enrichment(spindle_data):
    det = SpindleDetector(
        threshold_low=1.5, threshold_high=2.5, compute_symmetry=True,
    )
    cat = det.detect(spindle_data)
    if len(cat) > 0:
        assert "symmetry" in cat.df.columns
        sym = cat.df["symmetry"].dropna()
        assert len(sym) > 0
        # Symmetry is in [0, 1]
        assert (sym >= 0).all() and (sym <= 1).all()


def test_spindle_isolation(spindle_data):
    """min_isolation should drop events that are too close together."""
    det_no_iso = SpindleDetector(threshold_low=1.5, threshold_high=2.5)
    cat_all = det_no_iso.detect(spindle_data)

    det_iso = SpindleDetector(
        threshold_low=1.5, threshold_high=2.5, min_isolation=100.0,
    )
    cat_iso = det_iso.detect(spindle_data)
    # With a very large isolation, at most 1 event should survive
    assert len(cat_iso) <= 1


def test_spindle_all_enrichments(spindle_data):
    """All enrichments enabled simultaneously."""
    det = SpindleDetector(
        threshold_low=1.5, threshold_high=2.5,
        compute_frequency=True, compute_rel_power=True,
        compute_symmetry=True, min_isolation=0.5,
    )
    cat = det.detect(spindle_data)
    assert isinstance(cat, EventCatalog)
    if len(cat) > 0:
        for col in ("frequency", "rel_power", "symmetry"):
            assert col in cat.df.columns


def test_spindle_params_serialization():
    """Enrichment params should appear in detector params dict."""
    det = SpindleDetector(
        compute_frequency=True, compute_symmetry=True, min_isolation=1.0,
    )
    assert det.params["compute_frequency"] is True
    assert det.params["compute_symmetry"] is True
    assert det.params["min_isolation"] == 1.0
    assert det.params["compute_rel_power"] is False

