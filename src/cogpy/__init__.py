"""Top-level API for cogpy.

    cogpy is a Python package for ecog data analysis.

Subpackages
-----------
.. autosummary::
   :toctree: generated

    cogpy.brainstates
    cogpy.burst
    cogpy.cli    
    cogpy.core
    cogpy.datasets
    cogpy.decomposition    
    cogpy.depth_probe
    cogpy.io
    cogpy.model
    cogpy.plot
    cogpy.preprocess
    cogpy.spectral
    cogpy.utils
    cogpy.wave
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
