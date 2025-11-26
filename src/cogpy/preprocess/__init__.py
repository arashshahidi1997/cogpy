"""Preprocessing of neural signals.

This module :mod:`cogpy.preprocess` provides tools for preprocessing of neural signals

    from cogpy.preprocess import interpolate

Subpackages
-----------
.. autosummary::
   :toctree: generated

    cogpy.preprocess.channel_feature_functions
    cogpy.preprocess.channel_feature
    cogpy.preprocess.detect_bads
    cogpy.preprocess.filt
    cogpy.preprocess.filtx
    cogpy.preprocess.interpolate
    cogpy.preprocess.linenoise
    cogpy.preprocess.resample
"""
# Auto-generated shim: exposes cogpy.core.preprocess as cogpy.preprocess
from cogpy.core import preprocess as _impl
from cogpy.core.preprocess import *

__all__ = getattr(_impl, "__all__", [])
