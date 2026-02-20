from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(
    __name__,
    submodules=[
        "frame_plot",
        "multichannel_timeseries",
        "orthoslicer",
        "orthoslicer_facet",
        "orthoslicer_ranger",
        "orthoslicer_rangercopy",
        "orthoslicer_bursts",
        "orthoslicer_bursts_timeseries",
        "orthoslicer_zoom",
        "orthoslicer.py",  # keep "orthoslicer" twice? No—see note below.
        "specgram_plot",
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
        frame_plot,
        multichannel_timeseries,
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
