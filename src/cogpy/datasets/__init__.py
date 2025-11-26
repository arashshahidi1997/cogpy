"""
The :mod:`cogpy.datasets` module tools for making, loading and fetching ECoG datasets.

Subpackages
-----------
.. autosummary::
   :toctree: generated

    cogpy.datasets.load
    cogpy.datasets.tensor
"""

from .load import load_sample, load_raw_sample

# from .make import

__all__ = ["load_sample", "load_raw_sample"]
