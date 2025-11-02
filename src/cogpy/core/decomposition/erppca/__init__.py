from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(__name__, submodules=[
    "base", "burst", "erpPCA", "match", "plot", "scores", "spatspec",
])

if TYPE_CHECKING:
    from . import base, burst, erpPCA, match, plot, scores, spatspec
