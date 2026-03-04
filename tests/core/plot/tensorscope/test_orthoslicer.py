"""Tests for Orthoslicer module (v2.4)."""

from __future__ import annotations

import numpy as np
import pytest

from cogpy.core.plot.tensorscope import TensorScopeState


@pytest.fixture
def grid_data():
    xr = pytest.importorskip("xarray")

    rng = np.random.RandomState(0)
    data = rng.randn(256, 4, 4)
    return xr.DataArray(
        data,
        dims=("time", "AP", "ML"),
        coords={"time": np.arange(256) / 1000.0, "AP": np.arange(4), "ML": np.arange(4)},
        attrs={"fs": 1000.0},
    )


def test_orthoslicer_module_creation():
    from cogpy.core.plot.tensorscope.modules.orthoslicer import create_orthoslicer_module

    module = create_orthoslicer_module(nperseg=256, noverlap=128, bandwidth=4.0, max_seconds=0.5)
    assert module.name == "orthoslicer"


def test_orthoslicer_activation_returns_object(grid_data):
    hv = pytest.importorskip("holoviews")
    hv.extension("bokeh")

    from cogpy.core.plot.tensorscope.modules.orthoslicer import create_orthoslicer_module

    state = TensorScopeState(grid_data)
    module = create_orthoslicer_module(nperseg=256, noverlap=128, bandwidth=4.0, max_seconds=0.5)

    layout = module.activate(state)
    assert layout is not None
    assert isinstance(layout, (hv.Layout, hv.Div))


def test_orthoslicer_spectrogram_dims_if_available(grid_data):
    # spectrogramx depends on optional ghostipy; skip if not installed.
    from cogpy.core.plot.tensorscope.modules.orthoslicer import _OrthoParams, _compute_spectrogram

    try:
        spec = _compute_spectrogram(
            grid_data,
            _OrthoParams(nperseg=256, noverlap=128, bandwidth=4.0, max_seconds=0.5),
        )
    except ImportError:
        pytest.skip("spectrogramx requires optional ghostipy dependency")

    assert "time" in spec.dims
    assert "freq" in spec.dims
    assert "AP" in spec.dims
    assert "ML" in spec.dims
