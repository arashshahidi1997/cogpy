"""
The :mod:`src.datasets` module tools for making, loading and fetching ECoG datasets.
"""

from .load import load_sample, load_sample_batch
# from .make import 

__all__ = ['load_sample', 'load_sample_batch']
