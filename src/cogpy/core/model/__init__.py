from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(
    __name__,
    submodules=[
        "base",
        "data_generator",
        "envelopes",
        "gaussian_cover",
        "plot",
        "poisson_process",
    ],
)

if TYPE_CHECKING:
    from . import base, data_generator, envelopes, gaussian_cover, plot, poisson_process
