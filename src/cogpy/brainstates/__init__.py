"""Brain-state detection and classification."""

from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(
    __name__,
    submodules=[
        "brainstates",
        "EMG",
        "intervals",
    ],
)

if TYPE_CHECKING:
    from . import brainstates, EMG, intervals
