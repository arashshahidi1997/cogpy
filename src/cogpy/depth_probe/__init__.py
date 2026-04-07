from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(
    __name__,
    submodules=[
        "csd",
        "linear_signal",
    ],
)

if TYPE_CHECKING:
    from . import csd, linear_signal
