"""TensorScope - Signal-centric neurophysiology visualization."""

from __future__ import annotations

from .app import TensorScopeApp
from .signal import SignalObject, SignalRegistry
from .state import TensorScopeState

__all__ = [
    "TensorScopeState",
    "TensorScopeApp",
    "SignalObject",
    "SignalRegistry",
]

__version__ = "0.6.0"  # Signal-centric refactor
