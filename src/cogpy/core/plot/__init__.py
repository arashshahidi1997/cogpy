"""Visualization components for neural signal exploration."""
from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(
    __name__,
    submodules=[
        "grid_frame_element",
        "grid_indexing",
        "frame_plot",
        "ieeg_toolkit_app",
        "linked_views",
        "processing_chain",
        "multichannel_timeseries",
        "theme",
        "orthoslicer_rangercopy",
        "orthoslicer_bursts",
        "orthoslicer_bursts_timeseries",
        "specgram_plot",
        "spectrogram_bursts_app",
        "topomap",
        "time_player",
        "time_plot",
        "xarray_hv",
        "xarr_plot",
        "decomposition",
    ],
)

if TYPE_CHECKING:
    from . import (
        grid_frame_element,
        frame_plot,
        multichannel_timeseries,
        processing_chain,
        orthoslicer_rangercopy,
        orthoslicer_bursts,
        orthoslicer_bursts_timeseries,
        specgram_plot,
        time_player,
        time_plot,
        xarray_hv,
        xarr_plot,
    )
