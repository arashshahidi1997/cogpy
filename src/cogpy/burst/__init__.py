"""Burst detection and analysis.

This module :mod:`cogpy.burst` provides tools for detecting and analyzing neural spectral burst patterns

    from cogpy.burst.detect import blob_detection

Subpackages
-----------
.. autosummary::
   :toctree: generated

    cogpy.burst.blob_detection
    cogpy.burst.burst_merge
    cogpy.burst.burst_phase
    cogpy.burst.burst_wave
"""

# Auto-generated shim: exposes cogpy.core.burst as cogpy.burst
from cogpy.core import burst as _impl
from cogpy.core.burst import *

__all__ = getattr(_impl, "__all__", [])
