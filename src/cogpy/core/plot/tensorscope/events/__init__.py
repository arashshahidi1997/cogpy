"""TensorScope events package (Phase 0 scaffolding)."""

from __future__ import annotations

from .model import EventStream, EventStyle
from .registry import EventRegistry

__all__ = ["EventRegistry", "EventStream", "EventStyle"]
