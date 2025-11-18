from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(
    __name__,
    submodules=[
        "decomposition",
        "embed",
        "erpPCA",
        "manifold",  # files
        "erppca",  # nested package
    ],
)

if TYPE_CHECKING:
    from . import decomposition, embed, erpPCA, manifold, erppca
