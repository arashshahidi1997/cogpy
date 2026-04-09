"""Visualization components for neural signal exploration.

Submodules
----------
hv : Interactive HoloViews/Panel components (grid_movie, multichannel_view,
    TopoMap, OrthoSlicerRanger, ChannelGridSelector).
decomposition : Static matplotlib plots for PCA / ICA decompositions.
specgram_plot : Spectrogram and time–frequency image helpers.
time_plot : Time-series trace plotting utilities.
"""

from lazy_loader import attach as _attach

__getattr__, __dir__, __all__ = _attach(
    __name__,
    submodules=[
        "hv",
        "decomposition",
        "specgram_plot",
        "time_plot",
    ],
)
