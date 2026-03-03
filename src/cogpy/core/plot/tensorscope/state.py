"""
TensorScope state management.

This module implements the centralized state object that owns controllers and
serves as the single source of truth for the application.

Phase 1: Full controller ownership + serialization + validation.
"""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

import param

if TYPE_CHECKING:
    import xarray as xr

from .data.modality import DataModality


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
        Primary dataset (must have dims: time, AP, ML or time, channel).
        Will be validated and normalized to canonical schema on init.
    """

    # Controllers (owned by state, not duplicated)
    time_hair = param.Parameter(default=None, doc="TimeHair controller")
    time_window = param.Parameter(default=None, doc="TimeWindowCtrl controller")
    channel_grid = param.Parameter(default=None, doc="ChannelGrid controller")
    processing = param.Parameter(default=None, doc="ProcessingChain controller")

    # Data registry
    data_registry = param.Parameter(default=None, doc="DataRegistry instance")
    event_registry = param.Parameter(default=None, doc="EventRegistry instance")

    # App-level state
    active_layout_preset = param.String(default="default")
    active_modality = param.String(default="grid_lfp")

    def __init__(self, data: "xr.DataArray", **params: Any) -> None:
        """
        Initialize state with validated data.

        Parameters
        ----------
        data : xr.DataArray
            Primary dataset (must have dims: time, AP, ML or time, channel)
            Will be validated and normalized to canonical schema.

        Raises
        ------
        ValueError
            If data doesn't conform to required schema
        """
        from cogpy.core.plot.channel_grid import ChannelGrid
        from cogpy.core.plot.processing_chain import ProcessingChain
        from cogpy.core.plot.time_player import TimeHair

        from .data.modalities import GridLFPModality
        from .data.registry import DataRegistry
        from .events.registry import EventRegistry
        from .schema import flatten_grid_to_channels
        from .schema import validate_and_normalize_grid
        from .time_window import TimeWindowCtrl

        normalized_data = validate_and_normalize_grid(data)

        super().__init__(**params)

        n_ap = int(normalized_data.sizes["AP"])
        n_ml = int(normalized_data.sizes["ML"])

        t_min = float(normalized_data.time.values[0])
        t_max = float(normalized_data.time.values[-1])

        self.time_hair = TimeHair(snap=True)
        self.time_window = TimeWindowCtrl(bounds=(t_min, t_max))
        self.channel_grid = ChannelGrid(n_ap=n_ap, n_ml=n_ml)
        # ProcessingChain is used by both the timeseries and spatial map layers.
        # Use the stacked-flat (time, channel) form so channel subsetting works
        # and spatial layers can reconstruct frames via AP/ML coords.
        self.processing = ProcessingChain(flatten_grid_to_channels(normalized_data))

        self.data_registry = DataRegistry()
        self.event_registry = EventRegistry()

        self.data_registry.register("grid_lfp", GridLFPModality(normalized_data))
        self.active_modality = str(self.data_registry.get_active_name() or "grid_lfp")

    @property
    def current_time(self) -> float | None:
        """
        Current time cursor position, delegated to the time controller.
        """
        if self.time_hair is None:
            return None
        return self.time_hair.t

    @current_time.setter
    def current_time(self, value: float) -> None:
        """Set the current time cursor position (delegated to the time controller)."""
        if self.time_hair is not None:
            self.time_hair.t = float(value)

    @property
    def selected_channels(self) -> frozenset[tuple[int, int]]:
        """
        Selected channels (delegated to ChannelGrid).
        """
        if self.channel_grid is None:
            return frozenset()
        return self.channel_grid.selected

    @property
    def selected_channels_flat(self) -> list[int]:
        """Selected channels as flat indices (row-major)."""
        if self.channel_grid is None:
            return []
        return self.channel_grid.flat_indices

    def register_modality(self, name: str, modality: DataModality) -> None:
        """Register a data modality by name."""
        if self.data_registry is None:
            raise RuntimeError("data_registry is not initialized")
        self.data_registry.register(name, modality)

    def set_active_modality(self, name: str) -> None:
        """
        Set active data modality.

        Switches which modality layers should use for visualization.
        """
        if self.data_registry is None:
            raise RuntimeError("data_registry is not initialized")
        self.data_registry.set_active(name)
        self.active_modality = str(name)

    def get_active_modality(self) -> DataModality | None:
        """Get currently active modality."""
        if self.data_registry is None:
            return None
        return self.data_registry.get_active()

    def register_events(self, name: str, stream: Any) -> None:
        """Register an event stream."""
        if self.event_registry is None:
            raise RuntimeError("event_registry is not initialized")
        # Normalize name and ensure the stream is registered under that name.
        try:
            stream.name = str(name)
        except Exception:  # noqa: BLE001
            pass
        self.event_registry.register(stream)

    def jump_to_event(self, stream_name: str, event_id) -> None:
        """Jump to specific event by ID."""
        if self.event_registry is None:
            raise RuntimeError("event_registry is not initialized")
        stream = self.event_registry.get(stream_name)
        if stream is None:
            raise ValueError(f"Event stream {stream_name!r} not found")
        ev = stream.get_event_by_id(event_id)
        if ev is not None:
            self.current_time = float(ev[stream.time_col])

    def next_event(self, stream_name: str) -> None:
        """Jump to next event in stream."""
        if self.event_registry is None or self.current_time is None:
            return
        stream = self.event_registry.get(stream_name)
        if stream is None:
            return
        ev = stream.get_next_event(float(self.current_time))
        if ev is not None:
            self.current_time = float(ev[stream.time_col])

    def prev_event(self, stream_name: str) -> None:
        """Jump to previous event in stream."""
        if self.event_registry is None or self.current_time is None:
            return
        stream = self.event_registry.get(stream_name)
        if stream is None:
            return
        ev = stream.get_prev_event(float(self.current_time))
        if ev is not None:
            self.current_time = float(ev[stream.time_col])

    def to_dict(self) -> dict:
        """
        Serialize state to a JSON-serializable dictionary.

        Returns dict with state values but NOT raw data arrays.
        References to data are stored as paths/identifiers.
        """
        return {
            "version": "1.0",
            "current_time": self.current_time,
            "time_window": tuple(self.time_window.window) if self.time_window else None,
            "selected_channels": [list(ch) for ch in self.selected_channels],
            "processing": self.processing.to_dict() if self.processing else {},
            "active_layout_preset": self.active_layout_preset,
            "active_modality": self.active_modality,
            "data_registry": self.data_registry.to_dict() if self.data_registry else {},
            "event_registry": self.event_registry.to_dict() if self.event_registry else {},
        }

    @classmethod
    def from_dict(
        cls,
        state_dict: dict,
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

        if state_dict.get("current_time") is not None:
            state.current_time = float(state_dict["current_time"])

        if state_dict.get("time_window") is not None and state.time_window is not None:
            t0, t1 = state_dict["time_window"]
            state.time_window.set_window(float(t0), float(t1))

        if state_dict.get("selected_channels") and state.channel_grid is not None:
            for ap, ml in state_dict["selected_channels"]:
                state.channel_grid.select_cell(int(ap), int(ml))

        state.active_layout_preset = str(state_dict.get("active_layout_preset", "default"))
        desired_modality = str(state_dict.get("active_modality", "grid_lfp"))
        if state.data_registry is not None and desired_modality in state.data_registry.list():
            state.set_active_modality(desired_modality)
        else:
            state.active_modality = desired_modality

        return state
