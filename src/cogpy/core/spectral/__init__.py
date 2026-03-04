from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(
    __name__,
    submodules=[
        "bivariate",
        "bivariate_spectral",
        "features",
        "multitaper",
        "psd",
        "specx",
        "process_spectrogram",
        "ssa",
        "superlet",
        "whitening",
    ],
)

if TYPE_CHECKING:
    from . import (
        bivariate,
        bivariate_spectral,
        features,
        multitaper,
        psd,
        specx,
        process_spectrogram,
        ssa,
        superlet,
        whitening,
    )
