"""Temporal and spatial signal measures."""
from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(
    __name__,
    submodules=[
        "temporal",
        "spatial",
    ],
)

if TYPE_CHECKING:
    from . import temporal
    from . import spatial
