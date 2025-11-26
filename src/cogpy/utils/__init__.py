"""Utils subpackage

Utility functions for various tasks.

    from cogpy.utils import some_utility_function

Subpackages
-----------
.. autosummary::
   :toctree: generated

    cogpy.utils.convert
    cogpy.utils.curve
    cogpy.utils.grid_neighborhood
    cogpy.utils.reshape
    cogpy.utils.sliding
    cogpy.utils.stats
    cogpy.utils.subgrid
    cogpy.utils.time_series
    cogpy.utils.wrappers
    cogpy.utils.xarr
"""

# Auto-generated shim: exposes cogpy.core.utils as cogpy.utils
from cogpy.core import utils as _impl
from cogpy.core.utils import *

__all__ = getattr(_impl, "__all__", [])
