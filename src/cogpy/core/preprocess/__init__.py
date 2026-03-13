"""Signal preprocessing: filtering, bad-channel detection, resampling, and interpolation."""
from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, _submod_all = attach(
    __name__,
    submodules=[
        "filtering",
        "filtx",
        "interpolate",
        "linenoise",
        "resample",
        "badchannel",
        # Legacy — importable but not in __all__:
        "channel_feature_functions",
        "channel_feature",
        "detect_bads",
    ],
)

# Curated public API
__all__ = [
    "filtering",
    "filtx",
    "interpolate",
    "linenoise",
    "resample",
    "badchannel",
]

if TYPE_CHECKING:
    from . import (
        filtering,
        filtx,
        interpolate,
        linenoise,
        resample,
        badchannel,
        channel_feature_functions,
        channel_feature,
        detect_bads,
    )
