"""Tests for multi-modal data support (Phase 5)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cogpy.core.plot.tensorscope import TensorScopeState
from cogpy.core.plot.tensorscope.data.modalities import (
    FlatLFPModality,
    GridLFPModality,
    SpectrogramModality,
)
from cogpy.core.plot.tensorscope.data.registry import DataRegistry


def test_grid_lfp_modality():
    data = xr.DataArray(
        np.random.randn(1000, 8, 8),
        dims=("time", "AP", "ML"),
        coords={
            "time": np.arange(1000) / 1000.0,
            "AP": np.arange(8),
            "ML": np.arange(8),
        },
    )

    modality = GridLFPModality(data)

    assert modality.modality_type == "grid_lfp"
    assert modality.time_bounds() == (0.0, 0.999)
    assert modality.sampling_rate == pytest.approx(1000.0, rel=0.1)


def test_grid_lfp_window():
    data = xr.DataArray(
        np.random.randn(1000, 8, 8),
        dims=("time", "AP", "ML"),
        coords={"time": np.arange(1000) / 1000.0, "AP": np.arange(8), "ML": np.arange(8)},
    )

    modality = GridLFPModality(data)
    window = modality.get_window(0.2, 0.5)

    assert window.dims == ("time", "AP", "ML")
    assert len(window.time) > 0
    assert float(window.time.values[0]) >= 0.2
    assert float(window.time.values[-1]) <= 0.5


def test_flat_lfp_modality(small_ieeg):
    from cogpy.core.plot.tensorscope.schema import flatten_grid_to_channels, validate_and_normalize_grid

    # Fixtures may not be in canonical (time, AP, ML) order; normalize first.
    flat = flatten_grid_to_channels(validate_and_normalize_grid(small_ieeg))
    modality = FlatLFPModality(flat)

    assert modality.modality_type == "flat_lfp"
    assert modality.time_bounds()[0] >= 0
    assert modality.sampling_rate > 0


def test_spectrogram_modality():
    data = xr.DataArray(
        np.random.randn(100, 50, 8, 8),
        dims=("time", "freq", "AP", "ML"),
        coords={
            "time": np.arange(100) / 10.0,
            "freq": np.linspace(1, 100, 50),
            "AP": np.arange(8),
            "ML": np.arange(8),
        },
    )

    modality = SpectrogramModality(data)
    assert modality.modality_type == "spectrogram"
    assert modality.freq_bounds() == (1.0, 100.0)


def test_data_registry():
    data1 = xr.DataArray(
        np.random.randn(1000, 8, 8),
        dims=("time", "AP", "ML"),
        coords={"time": np.arange(1000) / 1000.0, "AP": np.arange(8), "ML": np.arange(8)},
    )
    data2 = xr.DataArray(
        np.random.randn(100, 50, 8, 8),
        dims=("time", "freq", "AP", "ML"),
        coords={
            "time": np.arange(100) / 10.0,
            "freq": np.linspace(1, 100, 50),
            "AP": np.arange(8),
            "ML": np.arange(8),
        },
    )

    registry = DataRegistry()
    registry.register("lfp", GridLFPModality(data1))
    registry.register("spectrogram", SpectrogramModality(data2))

    assert registry.list() == ["lfp", "spectrogram"]
    assert registry.get_active_name() == "lfp"

    registry.set_active("spectrogram")
    assert registry.get_active_name() == "spectrogram"
    assert registry.get_active().modality_type == "spectrogram"


def test_state_modality_registration(small_ieeg):
    state = TensorScopeState(small_ieeg)
    assert "grid_lfp" in state.data_registry.list()

    spec_data = xr.DataArray(
        np.random.randn(100, 50, 8, 8),
        dims=("time", "freq", "AP", "ML"),
        coords={
            "time": np.arange(100) / 10.0,
            "freq": np.linspace(1, 100, 50),
            "AP": np.arange(8),
            "ML": np.arange(8),
        },
    )

    state.register_modality("spectrogram", SpectrogramModality(spec_data))
    assert len(state.data_registry.list()) == 2

    state.set_active_modality("spectrogram")
    assert state.active_modality == "spectrogram"
    assert state.get_active_modality().modality_type == "spectrogram"


def test_state_active_modality_param_syncs_registry(small_ieeg):
    """Setting `state.active_modality` directly also updates the registry active name."""
    state = TensorScopeState(small_ieeg)

    spec_data = xr.DataArray(
        np.random.randn(100, 50, 8, 8),
        dims=("time", "freq", "AP", "ML"),
        coords={
            "time": np.arange(100) / 10.0,
            "freq": np.linspace(1, 100, 50),
            "AP": np.arange(8),
            "ML": np.arange(8),
        },
    )
    state.register_modality("spectrogram", SpectrogramModality(spec_data))

    # Simulate a UI binding that updates the param but doesn't call
    # `state.set_active_modality()`.
    state.active_modality = "spectrogram"

    assert state.data_registry.get_active_name() == "spectrogram"
    assert state.get_active_modality().modality_type == "spectrogram"
