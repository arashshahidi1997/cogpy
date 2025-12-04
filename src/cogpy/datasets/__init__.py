"""
The :mod:`cogpy.datasets` module tools for making, loading and fetching ECoG datasets.

"""

from .load import load_sample, load_raw_sample

# from .make import

__all__ = ["load_sample", "load_raw_sample"]
