import sklearn

"""
Package for ECoG data analysis.

Submodules:
    plot - plotting functions
    io - input/output functions
    model - model fitting and evaluation
    decomposition - signal decomposition
    wave - wave analysis
    stats - statistical analysis
    preprocessing - data preprocessing
    utils - general helpers that don't fit elsewhere
"""

# Re-export selected public API
# curated top-level API (example)
from .io.ecog_io import from_file, to_zarr
__all__ = ["from_file", "to_zarr"]

# module shims so old `import cogpy.decomposition` still works
import sys, importlib
for _name in ("brainstates","burst","decomposition","depth_probe",
              "model","preprocess","spectral","utils","wave"):
    sys.modules[__name__ + "." + _name] = importlib.import_module(f".core.{_name}", __name__)
