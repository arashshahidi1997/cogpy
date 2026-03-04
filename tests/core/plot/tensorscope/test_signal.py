"""Tests for signal objects and registry."""

from __future__ import annotations

import numpy as np
import xarray as xr

from cogpy.core.plot.tensorscope.signal import SignalObject, SignalRegistry


def test_signal_object_creation(small_ieeg):
    """Test SignalObject creation."""
    signal = SignalObject(small_ieeg, "Test Signal")

    assert signal.name == "Test Signal"
    assert signal.data is small_ieeg
    assert signal.processing is not None
    assert len(signal.id) == 8


def test_signal_duplicate(small_ieeg):
    """Test signal duplication."""
    signal = SignalObject(small_ieeg, "Original")
    signal.processing.bandpass_on = True
    signal.processing.bandpass_lo = 10.0

    dup = signal.duplicate("Copy")

    assert dup.name == "Copy"
    assert dup.id != signal.id
    assert dup.data is signal.data
    assert dup.processing is not signal.processing
    assert dup.processing.bandpass_on is True
    assert float(dup.processing.bandpass_lo) == 10.0


def test_signal_get_window(small_ieeg):
    """Test signal.get_window()."""
    signal = SignalObject(small_ieeg, "Test")

    win = signal.get_window(1.0, 3.0)

    assert "time" in win.dims
    assert float(win.time.values[0]) >= 1.0
    assert float(win.time.values[-1]) <= 3.0


def test_signal_with_spectral_analysis(small_ieeg):
    """Test using signal with cogpy.core.spectral functions."""
    from cogpy.core.spectral.specx import psdx

    signal = SignalObject(small_ieeg, "Test")

    win = signal.get_window(1.0, 3.0)
    psd = psdx(win, method="welch", nperseg=256)

    assert "freq" in psd.dims
    assert len(psd.freq) > 0


def test_signal_registry():
    """Test SignalRegistry."""
    data = xr.DataArray(
        np.random.randn(1000, 8, 8),
        dims=("time", "AP", "ML"),
        coords={
            "time": np.arange(1000) / 1000.0,
            "AP": np.arange(8),
            "ML": np.arange(8),
        },
        attrs={"fs": 1000.0},
    )

    registry = SignalRegistry()

    sig1 = SignalObject(data, "Signal 1")
    id1 = registry.register(sig1)

    assert id1 in registry.list()
    assert registry.get_active() is sig1
    assert registry.get(id1) is sig1


def test_signal_registry_duplicate():
    """Test registry.duplicate()."""
    data = xr.DataArray(
        np.random.randn(1000, 8, 8),
        dims=("time", "AP", "ML"),
        coords={
            "time": np.arange(1000) / 1000.0,
            "AP": np.arange(8),
            "ML": np.arange(8),
        },
        attrs={"fs": 1000.0},
    )

    registry = SignalRegistry()

    sig1 = SignalObject(data, "Original")
    id1 = registry.register(sig1)

    id2 = registry.duplicate(id1, "Copy")
    sig2 = registry.get(id2)

    assert sig2 is not None
    assert sig2.name == "Copy"
    assert id2 != id1
    assert len(registry.list()) == 2


def test_signal_registry_remove():
    """Test registry.remove()."""
    data = xr.DataArray(
        np.random.randn(1000, 8, 8),
        dims=("time", "AP", "ML"),
        coords={
            "time": np.arange(1000) / 1000.0,
            "AP": np.arange(8),
            "ML": np.arange(8),
        },
        attrs={"fs": 1000.0},
    )

    registry = SignalRegistry()

    sig1 = SignalObject(data, "Signal 1")
    sig2 = SignalObject(data, "Signal 2")
    id1 = registry.register(sig1)
    _id2 = registry.register(sig2)

    assert len(registry.list()) == 2

    registry.remove(id1)

    assert len(registry.list()) == 1
    assert registry.get(id1) is None
    assert registry.get_active() is sig2
