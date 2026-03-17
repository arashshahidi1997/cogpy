from __future__ import annotations

from typing import Literal

import holoviews as hv
import panel as pn

from cogpy.datasets.gui_bundles import spectrogram_bursts_bundle

from .orthoslicer_bursts import OrthoSlicerRangerBursts

__all__ = ["spectrogram_bursts_app"]


def spectrogram_bursts_app(
    *,
    mode: Literal["small", "large"] = "small",
    seed: int = 0,
    kind: Literal["toy", "ar_grid"] = "toy",
) -> pn.viewable.Viewable:
    """
    Servable Panel app for a 4D spectrogram + burst table navigation.
    """
    pn.extension()
    hv.extension("bokeh")

    bundle = spectrogram_bursts_bundle(mode=mode, seed=seed, kind=kind)
    da = bundle.spec
    bursts = bundle.bursts

    dx = ("ml", hv.Dimension("x", label="Medial-Lateral", unit="mm"))
    dy = ("ap", hv.Dimension("y", label="Anterior-Posterior", unit="mm"))
    dt = ("time", hv.Dimension("t", label="Time", unit="s"))
    dz = ("freq", hv.Dimension("z", label="Frequency", unit="Hz"))

    slicer = OrthoSlicerRangerBursts(da, bursts=bursts, dt=dt, dz=dz, dy=dy, dx=dx)
    slicer.tz_logy = True
    return slicer.panel_app()

