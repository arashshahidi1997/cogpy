"""Top-level API for cogpy.

This module :mod:`cogpy.core` provides core functionalities for the cogpy package.

"""

from lazy_loader import attach

__getattr__, __dir__, __all__ = attach(
    __name__,
    submodules=[
        "events",
    ],
)
