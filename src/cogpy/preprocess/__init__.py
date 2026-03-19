"""Signal preprocessing: filtering, bad-channel detection, resampling, and interpolation."""
from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(
    __name__,
    submodules=[
        "filtering",
        "interpolate",
        "linenoise",
        "resample",
        "badchannel",
    ],
)

if TYPE_CHECKING:
    from . import (
        filtering,
        interpolate,
        linenoise,
        resample,
        badchannel,
    )
