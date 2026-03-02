"""Spatial measures.

This module :mod:`cogpy.measures.spatial` re-exports the canonical
implementations from :mod:`cogpy.core.measures.spatial`.

"""

from cogpy.core.measures.spatial import *

__all__ = [
    "moran_i",
    "csd_power",
    "spatial_coherence_profile",
]

