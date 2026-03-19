"""Pre-built detection pipelines (v2.6.5)."""

from __future__ import annotations

from .burst import BurstDetector
from .pipeline import DetectionPipeline
from .threshold import ThresholdDetector
from .transforms import BandpassTransform, HilbertTransform, SpectrogramTransform, ZScoreTransform

__all__ = [
    "BURST_PIPELINE",
    "FAST_RIPPLE_PIPELINE",
    "GAMMA_BURST_PIPELINE",
    "RIPPLE_PIPELINE",
]


BURST_PIPELINE = DetectionPipeline(
    transforms=[SpectrogramTransform(nperseg=256, noverlap=128, bandwidth=4.0)],
    detector=BurstDetector(h_quantile=0.9),
    name="burst_detection",
)

RIPPLE_PIPELINE = DetectionPipeline(
    transforms=[
        BandpassTransform(low=100, high=250, order=4),
        HilbertTransform(),
        ZScoreTransform(),
    ],
    detector=ThresholdDetector(
        threshold=3.0,
        direction="positive",
        min_duration=0.02,
    ),
    name="ripple_detection",
)

FAST_RIPPLE_PIPELINE = DetectionPipeline(
    transforms=[
        BandpassTransform(low=250, high=500, order=4),
        HilbertTransform(),
        ZScoreTransform(),
    ],
    detector=ThresholdDetector(
        threshold=3.0,
        direction="positive",
        min_duration=0.01,
    ),
    name="fast_ripple_detection",
)

GAMMA_BURST_PIPELINE = DetectionPipeline(
    transforms=[
        BandpassTransform(low=30, high=80, order=4),
        HilbertTransform(),
        ZScoreTransform(),
    ],
    detector=ThresholdDetector(
        threshold=2.5,
        direction="positive",
        min_duration=0.05,
    ),
    name="gamma_burst_detection",
)

