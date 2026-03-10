"""TensorScope - Tensor-centric neurophysiology visualization (v3.0)."""

from __future__ import annotations

from .app import TensorScopeApp
from .state import SelectionState, TensorNode, TensorRegistry, TensorScopeState

__all__ = [
    "TensorNode",
    "TensorRegistry",
    "SelectionState",
    "TensorScopeState",
    "TensorScopeApp",
]

__version__ = "3.0.0"
