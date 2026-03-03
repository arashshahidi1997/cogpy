"""
TensorScope - Neurophysiology visualization application.

TensorScope is a Panel-based multi-view visualization system for exploring
neurophysiology data (LFP, spikes, events) with linked interactions.

Architecture (see the design docs in ``code/lib/cogpy/docs/source/explanation/plot``):
- Single source of truth (``TensorScopeState``)
- Layer-based views (``TensorLayer`` wrappers)
- Event-driven updates (param watchers / reactive bindings)
- Composition over inheritance

Phase 0 (Foundation) provides package structure + API contracts only.
Implementation is delivered incrementally in later phases.

Quick Start
-----------
>>> from cogpy.datasets.entities import example_ieeg_grid
>>> from cogpy.core.plot.tensorscope import TensorScopeState
>>> data = example_ieeg_grid(mode="small")
>>> state = TensorScopeState(data)  # Phase 0: minimal, no real controllers yet

See ``code/lib/cogpy/examples/tensorscope/hello_tensorscope.py`` for a runnable demo.
"""

from __future__ import annotations

from .app import TensorScopeApp
from .layers.base import TensorLayer
from .state import TensorScopeState

__all__ = [
    "TensorLayer",
    "TensorScopeApp",
    "TensorScopeState",
]

__version__ = "0.0.1"  # Phase 0

