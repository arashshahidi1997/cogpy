from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(
    __name__,
    submodules=[
        "detect",
        "features",
        "plot",
        "process",
        "utils",
    ],
)

if TYPE_CHECKING:
    from . import detect, features, plot, process, utils
