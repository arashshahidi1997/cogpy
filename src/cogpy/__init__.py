"""Top-level API for cogpy.

    cogpy is a Python package for ecog data analysis. Submodules such as
    :mod:`cogpy.brainstates`, :mod:`cogpy.preprocess`, and :mod:`cogpy.spectral`
    expose the primary functionality and are documented in the API reference.
"""

from typing import TYPE_CHECKING
import sys, importlib

_SUBPKGS = (
    "brainstates",
    "burst",
    "decomposition",
    "depth_probe",
    "model",
    "plot",
    "preprocess",
    "spectral",
    "utils",
    "wave",
)

# Make `import cogpy.<name>` resolve to `cogpy.core.<name>`
for _name in _SUBPKGS:
    sys.modules[__name__ + "." + _name] = importlib.import_module(
        f".core.{_name}", __name__
    )

if TYPE_CHECKING:
    from .core import (
        brainstates,
        burst,
        decomposition,
        depth_probe,
        model,
        plot,
        preprocess,
        spectral,
        utils,
        wave,
    )
