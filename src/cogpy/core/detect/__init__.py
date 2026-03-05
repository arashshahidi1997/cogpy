"""
Event detection framework (v2.6.1).

Provides a unified detector interface and concrete detectors that return
`cogpy.core.events.EventCatalog`.
"""

from .base import EventDetector
from .burst import BurstDetector
from .ripple import RippleDetector, SpindleDetector
from .threshold import ThresholdDetector

__all__ = ["BurstDetector", "EventDetector", "RippleDetector", "SpindleDetector", "ThresholdDetector"]
