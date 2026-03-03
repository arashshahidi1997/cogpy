"""
TensorScope layer interface.

A TensorLayer is a thin wrapper around an existing component (Panel/HoloViews/Bokeh)
that binds to ``TensorScopeState`` and renders a view.

Phase 0: Interface only.
Phase 2+: Concrete layer implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from ..state import TensorScopeState

if TYPE_CHECKING:
    import panel as pn


class TensorLayer(ABC):
    """
    Base interface for TensorScope view layers.

    Layers must:
    - Accept a state object (dependency injection; no globals)
    - Expose a Panel view via :meth:`panel`
    - Provide a :meth:`dispose` hook to avoid Bokeh document ownership issues

    Notes
    -----
    This is an abstract base class rather than a ``param.Parameterized`` in Phase 0
    to keep the interface minimal. Phase 2 may introduce shared param patterns.
    """

    def __init__(self, state: TensorScopeState, *, name: str | None = None) -> None:
        self.state = state
        self.name = name or self.__class__.__name__

    @abstractmethod
    def panel(self) -> "pn.viewable.Viewable":
        """Return the Panel view for this layer."""

    def dispose(self) -> None:
        """
        Release resources owned by the layer.

        Implementations should stop watchers/streams and detach from Bokeh docs
        where applicable.
        """

        return None

