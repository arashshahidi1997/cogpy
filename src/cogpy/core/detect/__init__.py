"""
Event detection framework (v2.6.1).

Provides a unified detector interface and concrete detectors that return
`cogpy.core.events.EventCatalog`.
"""

from .base import EventDetector
from .burst import BurstDetector

__all__ = ["BurstDetector", "EventDetector"]

