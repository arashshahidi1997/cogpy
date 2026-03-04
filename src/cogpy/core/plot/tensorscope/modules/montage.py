"""Montage module: multiple spatial views in a grid (v2.3)."""

from __future__ import annotations

from ..view_spec import ViewSpec
from .base import ViewPresetModule

__all__ = ["MODULE", "create_montage_module"]


def create_montage_module(times: list[float] | None = None) -> ViewPresetModule:
    """
    Create a montage module.

    Notes
    -----
    v2.3 uses the shared cursor time control for all views. A future extension
    may allow fixed per-view control values (true time snapshots).
    """
    if times is None:
        times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    specs: list[ViewSpec] = []
    for t in times:
        specs.append(
            ViewSpec(
                kdims=["AP", "ML"],
                controls=["time"],
                colormap="RdBu_r",
                symmetric_clim=True,
                title=f"t≈{t:.1f}s",
            )
        )

    return ViewPresetModule(
        name=f"montage_{len(times)}",
        description=f"Spatial montage ({len(times)} panels)",
        specs=specs,
        layout="grid",
    )


MODULE = create_montage_module()

