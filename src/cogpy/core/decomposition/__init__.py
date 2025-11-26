# """Decomposition tools.

# This module :mod:`cogpy.decomposition` provides tools for analyzing linear probes and depth electrodes

#     from cogpy.decomposition import embed

# Subpackages
# -----------
# .. autosummary::
#    :toctree: generated

#     cogpy.decomposition.decomposition
#     cogpy.decomposition.embed
#     cogpy.decomposition.erpPCA
#     cogpy.decomposition.manifold
#     cogpy.core.decomposition.erpppca
# """

from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(
    __name__,
    submodules=[
        "decomposition",
        "embed",
        "erpPCA",
        "manifold",  # files
        "erppca",  # nested package
    ],
)

if TYPE_CHECKING:
    from . import decomposition, embed, erpPCA, manifold, erppca
