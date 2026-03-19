from lazy_loader import attach
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = attach(
    __name__,
    submodules=[
        "_email",
        "_functools",
        "convert",
        "curve",
        "grid_neighborhood",
        "imports",
        "_jupyter",
        "reshape",
        "sliding",
        "stats",
        "subgrid",
        "time_series",
        "wrappers",
        "xarr",
        "manifold",
    ],
)

if TYPE_CHECKING:
    from . import (
        _email,
        _functools,
        convert,
        curve,
        grid_neighborhood,
        imports,
        _jupyter,
        reshape,
        sliding,
        stats,
        subgrid,
        time_series,
        wrappers,
        xarr,
    )
