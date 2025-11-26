"""Wave detection and analysis.

This module :mod:`cogpy.wave` provides tools for detecting and analyzing neural wave patterns

    from cogpy.wave.detect import detect_extrema

Subpackages
-----------
.. autosummary::
   :toctree: generated

    cogpy.wave.detect
    cogpy.wave.features
    cogpy.wave.plot
    cogpy.wave.process
    cogpy.wave.utils
"""

# Auto-generated shim: exposes cogpy.core.wave as cogpy.wave
from cogpy.core import wave as _impl
from cogpy.core.wave import *

__all__ = getattr(_impl, "__all__", [])
