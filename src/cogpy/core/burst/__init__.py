from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(__name__, submodules=[
    "blob_detection", "burst_merge", "burst_phase", "burst_wave", "utils",
])

if TYPE_CHECKING:
    from . import blob_detection, burst_merge, burst_phase, burst_wave, utils
