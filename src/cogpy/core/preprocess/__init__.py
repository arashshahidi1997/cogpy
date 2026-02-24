from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(
    __name__,
    submodules=[
        # Legacy compatibility surfaces (prefer `badchannel` for new code):
        "channel_feature_functions",
        "channel_feature",
        "detect_bads",
        "filt",
        "filtx",
        "interpolate",
        "linenoise",
        "linenoise_io",
        "resample",
        "badchannel",
    ],
)

if TYPE_CHECKING:
    from . import (
        channel_feature_functions,
        channel_feature,
        detect_bads,
        filt,
        filtx,
        interpolate,
        linenoise,
        linenoise_io,
        resample,
        badchannel,
    )
