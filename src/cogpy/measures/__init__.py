"""Temporal, spatial, and coupling signal measures.

Submodules
----------
temporal : Variance, RMS, Hjorth parameters, and other time-domain statistics.
spatial : Moran's I, gradient anisotropy, and grid-based spatial metrics.
comparison : Signal comparison utilities (correlation, distance).
coupling : Cross-frequency and inter-regional coupling measures.
pac : Phase–amplitude coupling estimators (MI, MVL, GLM).
"""

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
