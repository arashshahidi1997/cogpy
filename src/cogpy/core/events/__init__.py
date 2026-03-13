"""
Event catalog infrastructure (v2.6).

Provides `EventCatalog`, a lightweight bridge between:
- analysis containers (`cogpy.datasets.schemas.Events` / `Intervals`)
- visualization (`cogpy.core.tensorscope.events.EventStream`)
"""

from .catalog import EventCatalog

__all__ = ["EventCatalog"]

