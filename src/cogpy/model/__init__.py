"""Simulating local field potentials.

This module :mod:`cogpy.model` provides tools for simulating local field potentials

    from cogpy.model import gaussian_cover

Subpackages
-----------
.. autosummary::
   :toctree: generated

    cogpy.model.base
    cogpy.model.data_generator
    cogpy.model.envelopes
    cogpy.model.gaussian_cover
    cogpy.model.plot
    cogpy.model.poisson_process
"""

# Auto-generated shim: exposes cogpy.core.model as cogpy.model
from cogpy.core import model as _impl
from cogpy.core.model import *

__all__ = getattr(_impl, "__all__", [])
