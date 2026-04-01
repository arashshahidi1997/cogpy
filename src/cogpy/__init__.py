"""Top-level API for cogpy.

cogpy is a Python package for ecog data analysis. Submodules such as
:mod:`cogpy.brainstates`, :mod:`cogpy.preprocess`, and :mod:`cogpy.spectral`
expose the primary functionality and are documented in the API reference.
"""

from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version("cogpy")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
