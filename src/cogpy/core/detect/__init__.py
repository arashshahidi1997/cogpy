"""
Event detection framework (v2.6.1).

Provides a unified detector interface and concrete detectors that return
`cogpy.core.events.EventCatalog`.
"""

from .base import EventDetector
from .burst import BurstDetector
from .pipeline import DetectionPipeline
from .pipelines import BURST_PIPELINE, FAST_RIPPLE_PIPELINE, GAMMA_BURST_PIPELINE, RIPPLE_PIPELINE
from .ripple import RippleDetector, SpindleDetector
from .threshold import ThresholdDetector
from . import transforms

__all__ = [
    "BURST_PIPELINE",
    "BurstDetector",
    "DetectionPipeline",
    "EventDetector",
    "FAST_RIPPLE_PIPELINE",
    "GAMMA_BURST_PIPELINE",
    "RIPPLE_PIPELINE",
    "RippleDetector",
    "SpindleDetector",
    "ThresholdDetector",
    "transforms",
    "get_detector_class",
]


def get_detector_class(name: str):
    """Get a detector class by its serialized name."""
    table = {
        "BurstDetector": BurstDetector,
        "ThresholdDetector": ThresholdDetector,
        "RippleDetector": RippleDetector,
        "SpindleDetector": SpindleDetector,
    }
    key = str(name)
    if key not in table:
        raise ValueError(f"Unknown detector: {key}")
    return table[key]
