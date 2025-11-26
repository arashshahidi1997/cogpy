# """ERP-PCA decomposition of neural signals.

# Event-related potential principal component analysis (ERP-PCA) is a technique used to decompose neural signals into their underlying components based on temporal and spatial patterns. This module provides tools for performing ERP-PCA decomposition, allowing researchers to analyze and interpret complex neural data more effectively.
# This module :mod:`cogpy.core.decomposition.erppca` provides tools for ERP-PCA decomposition of neural signals

#     from cogpy.core.decomposition import erppca

# Subpackages
# -----------
# .. autosummary::
#    :toctree: generated

#     cogpy.core.decomposition.erppca.base
#     cogpy.core.decomposition.erppca.burst
#     cogpy.core.decomposition.erppca.erpPCA
#     cogpy.core.decomposition.erppca.match
#     cogpy.core.decomposition.erppca.plot
#     cogpy.core.decomposition.erppca.scores
#     cogpy.core.decomposition.erppca.spatspec   
# """
from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(
    __name__,
    submodules=[
        "base",
        "burst",
        "erpPCA",
        "match",
        "plot",
        "scores",
        "spatspec",
    ],
)

if TYPE_CHECKING:
    from . import base, burst, erpPCA, match, plot, scores, spatspec
