"""Spectral analysis: PSD, spectrograms, coherence, and multitaper methods."""

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
