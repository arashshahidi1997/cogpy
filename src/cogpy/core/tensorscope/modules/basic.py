"""Built-in module: basic spatial + temporal views."""

from __future__ import annotations

from ..view_spec import ViewSpec
from .base import ViewPresetModule

MODULE = ViewPresetModule(
    name="basic",
    description="Spatial map and linked timeseries",
    specs=[
        ViewSpec(
            kdims=["AP", "ML"],
            controls=["time"],
            colormap="RdBu_r",
            symmetric_clim=True,
            title="Spatial LFP",
        ),
        ViewSpec(
            kdims=["time"],
            controls=["AP", "ML"],
            title="Timeseries",
        ),
    ],
    layout="stack",
)

