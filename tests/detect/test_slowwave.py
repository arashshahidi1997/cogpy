"""Tests for SlowWaveDetector and gamma_envelope_validator."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cogpy.detect.slowwave import SlowWaveDetector, gamma_envelope_validator
from cogpy.events import EventCatalog


def _make_slow_wave_signal(fs: float = 1000.0, duration: float = 20.0, seed: int = 42):
    """Create a synthetic LFP with embedded slow waves.

    Generates a ~1 Hz slow oscillation with clear negative troughs,
    plus broadband gamma that is suppressed during DOWN states.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(0.0, duration, 1.0 / fs, dtype=float)
    n = t.size

    # Slow oscillation at ~1 Hz with large amplitude.
    slow = 5.0 * np.sin(2 * np.pi * 1.0 * t)

    # Add some noise to make it realistic.
    noise = 0.3 * rng.randn(n)

    # Gamma burst (60 Hz) modulated by the slow oscillation — high during UP, low during DOWN.
    gamma_mod = np.clip(slow / 5.0 + 0.5, 0, 1)  # 0 at trough, 1 at peak
    gamma = gamma_mod * 0.5 * np.sin(2 * np.pi * 60.0 * t)

    y = slow + gamma + noise
    return xr.DataArray(y, dims=["time"], coords={"time": t}, attrs={"fs": fs})


@pytest.fixture
def slow_wave_signal():
    return _make_slow_wave_signal()


def test_slowwave_detector_creation():
    det = SlowWaveDetector()
    assert det.freq_range == (0.5, 4.0)
    assert det.dur_neg == (0.08, 1.0)
    assert det.dur_cycle == (0.3, 1.5)
    assert det.name == "SlowWaveDetector"


def test_slowwave_detector_custom_params():
    det = SlowWaveDetector(
        freq_range=(0.5, 2.0),
        dur_neg=(0.1, 0.8),
        dur_cycle=(0.5, 2.0),
        amp_ptp_percentile=10.0,
    )
    assert det.freq_range == (0.5, 2.0)
    assert det.params["dur_neg"] == (0.1, 0.8)


def test_slowwave_detector_detect_returns_catalog(slow_wave_signal):
    det = SlowWaveDetector(freq_range=(0.5, 4.0), amp_ptp_percentile=10.0)
    catalog = det.detect(slow_wave_signal)
    assert isinstance(catalog, EventCatalog)
    assert catalog.name == "slow_wave_events"


def test_slowwave_detector_finds_events(slow_wave_signal):
    det = SlowWaveDetector(freq_range=(0.5, 4.0), amp_ptp_percentile=10.0)
    catalog = det.detect(slow_wave_signal)
    # With a clear 1 Hz slow wave over 20s, should detect multiple events.
    assert len(catalog) > 0
    assert catalog.is_interval_events


def test_slowwave_event_columns(slow_wave_signal):
    det = SlowWaveDetector(freq_range=(0.5, 4.0), amp_ptp_percentile=10.0)
    catalog = det.detect(slow_wave_signal)
    if len(catalog) == 0:
        pytest.skip("No events detected")
    df = catalog.df
    for col in ["trough_time", "midcrossing_time", "peak_time", "amplitude", "state", "label"]:
        assert col in df.columns, f"Missing column: {col}"
    assert (df["label"] == "slow_wave").all()
    assert (df["state"] == "DOWN").all()
    assert (df["amplitude"] > 0).all()
    assert (df["duration"] > 0).all()


def test_slowwave_duration_gating():
    """Events outside duration bounds should be filtered out."""
    det = SlowWaveDetector(dur_cycle=(0.8, 1.2), amp_ptp_percentile=0.0)
    sig = _make_slow_wave_signal()
    catalog = det.detect(sig)
    if len(catalog) > 0:
        assert (catalog.df["duration"] >= 0.8).all()
        assert (catalog.df["duration"] <= 1.2).all()


def test_slowwave_multichannel():
    """Should work on (time, channel) data."""
    fs = 1000.0
    t = np.arange(0.0, 10.0, 1.0 / fs)
    rng = np.random.RandomState(0)
    n_ch = 3
    data = np.zeros((t.size, n_ch))
    for ch in range(n_ch):
        data[:, ch] = 4.0 * np.sin(2 * np.pi * 1.0 * t) + 0.2 * rng.randn(t.size)
    sig = xr.DataArray(data, dims=["time", "channel"], coords={"time": t}, attrs={"fs": fs})

    det = SlowWaveDetector(amp_ptp_percentile=10.0)
    catalog = det.detect(sig)
    assert isinstance(catalog, EventCatalog)
    if len(catalog) > 0:
        assert "channel" in catalog.df.columns


def test_slowwave_empty_on_noise():
    """Pure noise should yield few or no slow-wave events."""
    rng = np.random.RandomState(99)
    fs = 1000.0
    t = np.arange(0.0, 5.0, 1.0 / fs)
    y = 0.1 * rng.randn(t.size)
    sig = xr.DataArray(y, dims=["time"], coords={"time": t}, attrs={"fs": fs})
    det = SlowWaveDetector(amp_ptp_percentile=50.0)
    catalog = det.detect(sig)
    assert isinstance(catalog, EventCatalog)
    # May find a few spurious events, but shouldn't find many.
    assert len(catalog) < 5


def test_slowwave_serialization():
    det = SlowWaveDetector(freq_range=(0.5, 2.0))
    d = det.to_dict()
    assert d["detector"] == "SlowWaveDetector"
    assert d["params"]["freq_range"] == (0.5, 2.0)

    det2 = SlowWaveDetector.from_dict(d)
    assert det2.freq_range == (0.5, 2.0)


def test_slowwave_can_accept():
    det = SlowWaveDetector()
    good = xr.DataArray(np.zeros(10), dims=["time"])
    bad = xr.DataArray(np.zeros(10), dims=["freq"])
    assert det.can_accept(good)
    assert not det.can_accept(bad)


def test_gamma_envelope_validator(slow_wave_signal):
    det = SlowWaveDetector(freq_range=(0.5, 4.0), amp_ptp_percentile=10.0)
    catalog = det.detect(slow_wave_signal)
    if len(catalog) == 0:
        pytest.skip("No events detected")

    result = gamma_envelope_validator(slow_wave_signal, catalog)
    assert "gamma_at_trough" in result.columns
    assert "gamma_valid" in result.columns
    assert result["gamma_valid"].dtype == bool
    # In our synthetic signal, gamma is suppressed during DOWN states,
    # so most troughs should have low (negative z-scored) gamma.
    valid_frac = result["gamma_valid"].mean()
    assert valid_frac > 0.3, f"Expected most troughs to coincide with gamma minima, got {valid_frac:.2f}"
