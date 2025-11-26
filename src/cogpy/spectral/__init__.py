"""Spectral analysis and processing.

This module :mod:`cogpy.spectral` provides tools for spectral analysis and processing of neural signals

    from cogpy.spectral import mtm_spectrogram

Subpackages
-----------
.. autosummary::
   :toctree: generated

    cogpy.spectral.bivariate_spectral
    cogpy.spectral.gsp_multichannel
    cogpy.spectral.multitaper
    cogpy.spectral.oscillations
    cogpy.spectral.process_spectrogram
    cogpy.spectral.ssa
    cogpy.spectral.superlet
    cogpy.spectral.whitening
"""
# Auto-generated shim: exposes cogpy.core.spectral as cogpy.spectral
from cogpy.core import spectral as _impl
from cogpy.core.spectral import *

__all__ = getattr(_impl, "__all__", [])
