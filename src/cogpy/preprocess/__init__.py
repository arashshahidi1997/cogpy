"""Signal preprocessing: filtering, bad-channel detection, resampling, and interpolation.

Submodules
----------
filtering : Temporal (IIR), spatial (grid), reference (CMR), and normalization filters.
badchannel : Feature extraction, spatial normalization, and DBSCAN-based bad-channel detection.
resample : Decimation and resampling utilities.
interpolate : Spatial interpolation for missing / bad channels.
linenoise : Line-noise (50/60 Hz) removal via adaptive notch filtering.
"""

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
