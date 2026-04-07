"""Temporal, spatial, and coupling signal measures."""

from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(
    __name__,
    submodules=[
        "temporal",
        "spatial",
        "comparison",
        "coupling",
        "pac",
    ],
)

if TYPE_CHECKING:
    from . import temporal
    from . import spatial
    from . import comparison
    from . import coupling
    from . import pac
