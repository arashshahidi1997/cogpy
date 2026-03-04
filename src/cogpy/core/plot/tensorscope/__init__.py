"""TensorScope - Signal-centric neurophysiology visualization."""

from __future__ import annotations

from .app import TensorScopeApp
from .modules import ModuleRegistry, ViewPresetModule
from .signal import SignalObject, SignalRegistry
from .state import TensorScopeState
from .view_factory import ViewFactory
from .view_spec import ViewSpec

__all__ = [
    "TensorScopeState",
    "TensorScopeApp",
    "SignalObject",
    "SignalRegistry",
    "ViewSpec",
    "ViewFactory",
    "ModuleRegistry",
    "ViewPresetModule",
]

__version__ = "0.6.0"  # Signal-centric refactor
