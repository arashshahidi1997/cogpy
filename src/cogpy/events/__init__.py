"""
Event catalog infrastructure (v2.6).

Provides `EventCatalog`, a lightweight bridge between:
- analysis containers (`cogpy.datasets.schemas.Events` / `Intervals`)
- visualization (`cogpy.events.EventStream`)
"""

from .catalog import EventCatalog
from .stream import EventStream, EventStyle
from .registry import EventRegistry
from . import match

__all__ = ["EventCatalog", "EventStream", "EventStyle", "EventRegistry", "match"]
