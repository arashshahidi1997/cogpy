"""Brain state detection and analysis.

This module :mod:`cogpy.brainstates` provides tools for detecting and analyzing neural brain state patterns

    from cogpy.brainstates import EMG

Subpackages
-----------
.. autosummary::
   :toctree: generated

    cogpy.brainstates.brainstates
    cogpy.brainstates.EMG
    cogpy.brainstates.intervals
"""

# Auto-generated shim: exposes cogpy.core.brainstates as cogpy.brainstates
from cogpy.core import brainstates as _impl
from cogpy.core.brainstates import *

__all__ = getattr(_impl, "__all__", [])
