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
        "orthoslicer",
        "orthoslicer_facet",
        "orthoslicer_ranger",
        "orthoslicer_rangercopy",
        "orthoslicer_bursts",
        "orthoslicer_bursts_timeseries",
        "orthoslicer_zoom",
        "orthoslicer.py",  # keep "orthoslicer" twice? No—see note below.
        "specgram_plot",
        "spectrogram_bursts_app",
        "topomap",
        "time_player",
        "time_plot",
        "xarray_hv",
        "xarr_plot",
    ],
)

# NOTE: You have both a folder "orthoslicer" (with base.py) and files
# "orthoslicer.py", "orthoslicer_ranger.py", etc. Keep either the folder
# package or the single file to avoid name collision. If both must exist,
# rename the single-file variant (e.g., "orthoslicer_core.py") and update here.

if TYPE_CHECKING:
    from . import (
        grid_frame_element,
        frame_plot,
        multichannel_timeseries,
        processing_chain,
        orthoslicer,
        orthoslicer_facet,
        orthoslicer_ranger,
        orthoslicer_rangercopy,
        orthoslicer_bursts,
        orthoslicer_bursts_timeseries,
        orthoslicer_zoom,
        specgram_plot,
        time_player,
        time_plot,
        xarray_hv,
        xarr_plot,
    )
