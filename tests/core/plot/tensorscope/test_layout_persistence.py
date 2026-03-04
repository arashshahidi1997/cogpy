"""Tests for TensorScope v2.3 layout persistence."""

from __future__ import annotations

import tempfile
from pathlib import Path

from cogpy.core.plot.tensorscope.layout_persistence import load_layout, save_layout
from cogpy.core.plot.tensorscope.modules import ViewPresetModule
from cogpy.core.plot.tensorscope.view_spec import ViewSpec


def test_save_and_load_layout():
    module = ViewPresetModule(
        name="test_module",
        description="Test module for persistence",
        specs=[
            ViewSpec(kdims=["AP", "ML"], controls=["time"]),
            ViewSpec(kdims=["time"], controls=["AP", "ML"]),
        ],
        layout="grid",
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = Path(f.name)

    try:
        save_layout(temp_path, module)
        loaded = load_layout(temp_path)

        assert loaded.name == module.name
        assert loaded.description == module.description
        assert len(loaded.specs) == len(module.specs)
        assert loaded.layout == module.layout
    finally:
        temp_path.unlink()


def test_load_preserves_viewspec_details():
    spec = ViewSpec(
        kdims=["AP", "ML"],
        controls=["time"],
        colormap="RdBu_r",
        symmetric_clim=True,
        title="Test View",
    )

    module = ViewPresetModule(
        name="detail_test",
        description="Testing details",
        specs=[spec],
        layout="grid",
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = Path(f.name)

    try:
        save_layout(temp_path, module)
        loaded = load_layout(temp_path)

        loaded_spec = loaded.specs[0]
        assert loaded_spec.colormap == "RdBu_r"
        assert loaded_spec.symmetric_clim is True
        assert loaded_spec.title == "Test View"
    finally:
        temp_path.unlink()

