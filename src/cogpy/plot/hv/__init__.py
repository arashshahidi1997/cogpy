"""Interactive HoloViews/Panel visualization components for ECoG data.

This subpackage provides reusable, composable building blocks for interactive
neural signal exploration built on HoloViews and Panel.

Quick-start imports::

    from cogpy.plot.hv import grid_movie, multichannel_view, add_time_hair
"""

from lazy_loader import attach as _attach

__getattr__, __dir__, __all__ = _attach(
    __name__,
    submodules=[
        "channel_grid",
        "channel_grid_widget",
        "ecog_viewer",
        "grid_frame_element",
        "grid_indexing",
        "ieeg_toolkit",
        "ieeg_viewer",
        "linked_views",
        "multichannel_timeseries",
        "multichannel_viewer",
        "orthoslicer",
        "processing_chain",
        "selection_policy",
        "signals",
        "theme",
        "time_player",
        "topomap",
        "xarray_hv",
    ],
)
