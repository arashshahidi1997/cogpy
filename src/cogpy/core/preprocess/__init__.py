from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(__name__, submodules=[
    "channel_feature_functions", "channel_feature",
    "detect_bads", "filt", "filtx", "interpolate",
    "linenoise", "resample",
])

if TYPE_CHECKING:
    from . import (
        channel_feature_functions, channel_feature, detect_bads,
        filt, filtx, interpolate, linenoise, resample
    )
    # If you want deep symbol resolution for this specific import:
    from .detect_bads import OutlierDetector as OutlierDetector
