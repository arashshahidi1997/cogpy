"""Spectral analysis: PSD, spectrograms, coherence, and multitaper methods.

Submodules
----------
psd : Power spectral density estimation (Welch, periodogram).
multitaper : Multitaper spectral estimation and F-tests.
specx : Short-time spectrograms and time–frequency representations.
bivariate : Coherence, phase-locking value, and cross-spectral measures.
features : Band-power extraction, spectral edge, and peak frequency.
whitening : Spectral flattening and 1/f removal.
process_spectrogram : Post-processing utilities for spectrogram arrays.
"""

from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(
    __name__,
    submodules=[
        "bivariate",
        "features",
        "multitaper",
        "psd",
        "psd_utils",
        "specx",
        "process_spectrogram",
        "whitening",
    ],
)

if TYPE_CHECKING:
    from . import (
        bivariate,
        features,
        multitaper,
        psd,
        psd_utils,
        specx,
        process_spectrogram,
        whitening,
    )
