"""Built-in module: compare multiple signals side-by-side."""

from __future__ import annotations

from ..view_spec import ViewSpec
from .base import ViewPresetModule

MODULE = ViewPresetModule(
    name="comparison",
    description="Compare multiple signals (configure signal_id per view)",
    specs=[
        ViewSpec(
            kdims=["AP", "ML"],
            controls=["time"],
            signal_id=None,
            colormap="RdBu_r",
            symmetric_clim=True,
            title="Signal A",
        ),
        ViewSpec(
            kdims=["AP", "ML"],
            controls=["time"],
            signal_id=None,
            colormap="RdBu_r",
            symmetric_clim=True,
            title="Signal B",
        ),
        ViewSpec(
            kdims=["AP", "ML"],
            controls=["time"],
            signal_id=None,
            colormap="RdBu_r",
            symmetric_clim=True,
            title="Signal C",
        ),
    ],
    layout="grid",
)

