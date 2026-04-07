"""Dimensionality reduction and signal decomposition."""

from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(
    __name__,
    submodules=[
        "pca",
        "spatspec",
        "scores",
        "match",
        "embed",
    ],
)

if TYPE_CHECKING:
    from . import pca, spatspec, scores, match, embed
