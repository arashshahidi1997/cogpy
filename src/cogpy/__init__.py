# cogpy/__init__.py
from typing import TYPE_CHECKING
import importlib as _il

# All top-level subpackages you want accessible as `cogpy.<name>`
_SUBPKGS = (
    "brainstates", "burst", "decomposition", "depth_probe",
    "model", "plot", "preprocess", "spectral", "utils", "wave",
)

def __getattr__(name: str):
    if name in _SUBPKGS:
        # Return the actual module object from cogpy.core.<name>
        return _il.import_module(f".core.{name}", __name__)
    raise AttributeError(name)

def __dir__():
    # Make them show up in tab-completion
    return sorted(list(globals().keys()) + list(_SUBPKGS))

# ---- Editor-only breadcrumbs (static for Pylance) ----
if TYPE_CHECKING:
    from .core import (
        brainstates, burst, decomposition, depth_probe,
        model, plot, preprocess, spectral, utils, wave,
    )
