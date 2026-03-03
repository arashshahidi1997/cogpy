"""
TensorScope state management.

This module implements the centralized state object that owns controllers and
serves as the single source of truth for the application.

Phase 0: Minimal scaffold (no real controller wiring yet).
Phase 1+: Full controller ownership + serialization + validation.
"""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

import param

if TYPE_CHECKING:
    import xarray as xr


class TensorScopeState(param.Parameterized):
    """
    Central authoritative state for the TensorScope application.

    Owns controllers (e.g. TimeHair, ChannelGrid, ProcessingChain) and provides
    a stable surface area for layers to bind to.

    Design principles (from the TensorScope design docs):
    - Single source of truth
    - Delegates to controllers, doesn't duplicate
    - State is serializable

    Parameters
    ----------
    data
        Primary dataset. In Phase 1 this will be validated/normalized on init.

    Attributes
    ----------
    time_hair
        Placeholder for a time cursor controller (Phase 1).
    channel_grid
        Placeholder for a channel selection controller (Phase 1).
    processing
        Placeholder for a processing controller (Phase 1).
    """

    data = param.Parameter(default=None, doc="Primary dataset for the session.")

    def __init__(self, data: "xr.DataArray", **params: Any) -> None:
        super().__init__(data=data, **params)

        # Phase 0 placeholders. Phase 1 will construct real controllers and
        # enforce invariants.
        self.time_hair: Any | None = None
        self.channel_grid: Any | None = None
        self.processing: Any | None = None

        # Phase 0 fallback state (until controllers are wired).
        self._current_time: float | None = None
        self._selected_channels: frozenset[Any] = frozenset()

    @property
    def current_time(self) -> float | None:
        """
        Current time cursor position, delegated to the time controller.

        Returns None in Phase 0 unless ``time_hair`` is provided externally.
        """

        if self.time_hair is None:
            return self._current_time
        return getattr(self.time_hair, "t", self._current_time)

    @current_time.setter
    def current_time(self, value: float) -> None:
        """Set the current time cursor position (delegated to the time controller)."""

        if self.time_hair is None:
            self._current_time = float(value)
            return
        setattr(self.time_hair, "t", float(value))

    @property
    def selected_channels(self) -> frozenset[Any]:
        """
        Selected channels (delegated to ChannelGrid).

        Returns an empty set in Phase 0 unless ``channel_grid`` is provided.
        """

        if self.channel_grid is None:
            return self._selected_channels
        selected = getattr(self.channel_grid, "selected", None)
        if selected is None:
            return self._selected_channels
        if isinstance(selected, frozenset):
            return selected
        return frozenset(selected)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize state to a JSON-serializable dictionary.

        Phase 0: best-effort minimal serialization (controller details omitted).
        Phase 1+: full serialization of controllers + layout + data references.
        """

        return {
            "current_time": self.current_time,
            "selected_channels": sorted(self.selected_channels),
        }

    @classmethod
    def from_dict(
        cls,
        state_dict: dict[str, Any],
        *,
        data_resolver: Callable[[], "xr.DataArray"],
    ) -> "TensorScopeState":
        """
        Restore state from a serialized dictionary.

        Parameters
        ----------
        state_dict
            Dictionary produced by :meth:`to_dict`.
        data_resolver
            Callable that returns the primary dataset for this state.
        """

        state = cls(data_resolver())
        if "current_time" in state_dict and state_dict["current_time"] is not None:
            state.current_time = float(state_dict["current_time"])
        if "selected_channels" in state_dict and state_dict["selected_channels"] is not None:
            state._selected_channels = frozenset(state_dict["selected_channels"])
        return state
