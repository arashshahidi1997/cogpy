from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(
    __name__,
    submodules=[
        "bivariate_spectral",
        "multitaper",
        "process_spectrogram",
        "ssa",
        "superlet",
        "whitening",
    ],
)

if TYPE_CHECKING:
    from . import (
        bivariate_spectral,
        multitaper,
        process_spectrogram,
        ssa,
        superlet,
        whitening,
    )
