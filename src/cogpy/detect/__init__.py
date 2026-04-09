"""Event detection framework.

Provides a unified ``EventDetector`` interface and composable
``DetectionPipeline`` that return :class:`cogpy.events.EventCatalog`.

Detectors: ``BurstDetector``, ``RippleDetector``, ``SpindleDetector``,
``SlowWaveDetector``, ``ThresholdDetector``.
Built-in pipelines: ``BURST_PIPELINE``, ``RIPPLE_PIPELINE``,
``FAST_RIPPLE_PIPELINE``, ``GAMMA_BURST_PIPELINE``.
"""

from .base import EventDetector
from .burst import BurstDetector
from .pipeline import DetectionPipeline
from .pipelines import (
    BURST_PIPELINE,
    FAST_RIPPLE_PIPELINE,
    GAMMA_BURST_PIPELINE,
    RIPPLE_PIPELINE,
)
from .ripple import RippleDetector, SpindleDetector
from .slowwave import SlowWaveDetector, gamma_envelope_validator
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
    "SlowWaveDetector",
    "SpindleDetector",
    "ThresholdDetector",
    "gamma_envelope_validator",
    "transforms",
    "get_detector_class",
]


def get_detector_class(name: str):
    """Get a detector class by its serialized name."""
    table = {
        "BurstDetector": BurstDetector,
        "ThresholdDetector": ThresholdDetector,
        "RippleDetector": RippleDetector,
        "SlowWaveDetector": SlowWaveDetector,
        "SpindleDetector": SpindleDetector,
    }
    key = str(name)
    if key not in table:
        raise ValueError(f"Unknown detector: {key}")
    return table[key]
