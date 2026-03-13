"""
Layout persistence helpers (v2.3).

Save/load a `ViewPresetModule` to a JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path

from .modules import ViewPresetModule
from .view_spec import ViewSpec

__all__ = ["save_layout", "load_layout"]


def save_layout(filepath: str | Path, module: ViewPresetModule) -> None:
    """
    Save a module configuration to JSON.

    Notes
    -----
    Only serializes standard `ViewPresetModule` fields and `ViewSpec.to_dict()`.
    Custom modules that rely on `activate_fn` are not reconstructible from JSON.
    """
    config = {
        "name": module.name,
        "description": module.description,
        "specs": [spec.to_dict() for spec in (module.specs or [])],
        "layout": module.layout,
        "version": "2.3.0",
    }
    Path(filepath).write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_layout(filepath: str | Path) -> ViewPresetModule:
    """Load a module configuration from JSON."""
    config = json.loads(Path(filepath).read_text(encoding="utf-8"))
    specs = [ViewSpec.from_dict(spec_dict) for spec_dict in (config.get("specs") or [])]
    return ViewPresetModule(
        name=str(config["name"]),
        description=str(config.get("description") or ""),
        specs=specs,
        layout=str(config.get("layout") or "grid"),
    )

