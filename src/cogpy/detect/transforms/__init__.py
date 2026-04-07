"""Detection transforms for `DetectionPipeline` (v2.6.5)."""

from __future__ import annotations

from .base import Transform
from .envelope import HilbertTransform, ZScoreTransform
from .filtering import BandpassTransform, HighpassTransform, LowpassTransform
from .spectral import SpectrogramTransform

__all__ = [
    "BandpassTransform",
    "HighpassTransform",
    "HilbertTransform",
    "LowpassTransform",
    "SpectrogramTransform",
    "Transform",
    "ZScoreTransform",
    "get_transform_class",
]


def get_transform_class(name: str):
    """Get a transform class by its serialized name."""
    table = {
        "BandpassTransform": BandpassTransform,
        "HighpassTransform": HighpassTransform,
        "LowpassTransform": LowpassTransform,
        "HilbertTransform": HilbertTransform,
        "ZScoreTransform": ZScoreTransform,
        "SpectrogramTransform": SpectrogramTransform,
    }
    key = str(name)
    if key not in table:
        raise ValueError(f"Unknown transform: {key}")
    return table[key]
