"""
Event catalog infrastructure (v2.6).

Provides `EventCatalog`, a lightweight bridge between:
- analysis containers (`cogpy.datasets.schemas.Events` / `Intervals`)
- visualization (`cogpy.tensorscope.events.EventStream`)
"""

from .catalog import EventCatalog
from . import match

__all__ = ["EventCatalog", "match"]

