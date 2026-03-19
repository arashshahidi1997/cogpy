"""Visualization components for neural signal exploration.

Interactive HoloViews/Panel components live under :mod:`cogpy.plot.hv`.
Static matplotlib/plotly helpers remain in this top-level package.
Deprecated modules are in :mod:`cogpy.plot._legacy`.
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
