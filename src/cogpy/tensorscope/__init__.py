"""TensorScope - Tensor-centric neurophysiology visualization (v3.0).

Requires the ``viz`` extra: ``pip install "cogpy[viz]"``
"""

from __future__ import annotations

from cogpy.utils.imports import import_optional

import_optional("panel")
import_optional("param")

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
