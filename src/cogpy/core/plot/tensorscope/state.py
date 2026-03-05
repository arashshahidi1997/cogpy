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
from .signal import SignalObject, SignalRegistry


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
    processing = param.Parameter(default=None, doc="ProcessingChain controller (legacy: active signal)")

    # Signal-centric state
    selected_time = param.Number(
        default=None,
        allow_None=True,
        doc="Selected time for analysis (PSD/spatial). Independent of cursor.",
    )
    signal_registry = param.Parameter(default=None, doc="SignalRegistry instance")
    spatial_space = param.Parameter(default=None, doc="CoordinateSpace for linked spatial selection")

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
        self.selected_time = t_min

        # Signal registry (Phase 6: signal-as-object).
        self.signal_registry = SignalRegistry()
        self._signal_registry_watch_owner = None
        self._signal_registry_watch_signal = None
        self._signal_registry_watch_active = None
        self._attach_signal_registry(self.signal_registry)

        base_signal = SignalObject(
            data=normalized_data,
            name="Raw LFP",
            processing=ProcessingChain(flatten_grid_to_channels(normalized_data)),
            metadata={"type": "grid_lfp", "is_base": True},
        )
        self.signal_registry.register(base_signal)

        # Legacy: ProcessingChain is used by both the timeseries and spatial map layers.
        # Keep `state.processing` as an alias to the active signal's chain.
        self._sync_processing_to_active_signal()

        # Spatial coordinate space (v2.1): shared AP/ML index selections across views.
        from .transforms.base import CoordinateSpace

        base = self.signal_registry.get_active()
        spatial_dims: set[str]
        if base is not None:
            d = base.data
            if ("AP" in d.dims) and ("ML" in d.dims):
                spatial_dims = {"AP", "ML"}
            elif "channel" in d.dims:
                spatial_dims = {"channel"}
            else:
                spatial_dims = set()
        else:
            spatial_dims = set()

        self.spatial_space = CoordinateSpace("spatial", dims=spatial_dims)
        if ("AP" in spatial_dims) and ("ML" in spatial_dims) and base is not None:
            n_ap0 = int(base.data.sizes.get("AP", 0))
            n_ml0 = int(base.data.sizes.get("ML", 0))
            if n_ap0 > 0 and n_ml0 > 0:
                self.spatial_space.set_selection("AP", n_ap0 // 2)
                self.spatial_space.set_selection("ML", n_ml0 // 2)

        self.data_registry = DataRegistry()
        self.event_registry = EventRegistry()

        self.data_registry.register("grid_lfp", GridLFPModality(normalized_data))
        self.active_modality = str(self.data_registry.get_active_name() or "grid_lfp")
        # Keep the registry's active modality in sync even if UI code sets
        # `state.active_modality = "..."` directly (without calling
        # `set_active_modality()`).
        self._active_modality_sync_watch = self.param.watch(
            self._on_active_modality_param_change, "active_modality"
        )

        # If someone replaces the registry object, re-wire watchers.
        self._signal_registry_replace_watch = self.param.watch(
            self._on_signal_registry_replaced, "signal_registry"
        )

    def _on_active_modality_param_change(self, event) -> None:
        """
        Sync DataRegistry active modality to the `active_modality` param.

        This is a safety net for UI bindings that update the param directly.
        """
        if self.data_registry is None:
            return
        new_name = str(getattr(event, "new", ""))
        if not new_name:
            return
        try:
            if self.data_registry.get_active_name() != new_name and new_name in self.data_registry.list():
                self.data_registry.set_active(new_name)
        except Exception:  # noqa: BLE001
            pass

    def _attach_signal_registry(self, registry: SignalRegistry | None) -> None:
        if registry is None:
            return

        # Detach existing.
        owner = getattr(self, "_signal_registry_watch_owner", None)
        if owner is not None:
            for handle in (self._signal_registry_watch_signal, self._signal_registry_watch_active):
                if handle is None:
                    continue
                try:
                    owner.param.unwatch(handle)
                except Exception:  # noqa: BLE001
                    pass

        self._signal_registry_watch_signal = registry.param.watch(
            lambda _e=None: self._sync_processing_to_active_signal(), "signals"
        )
        self._signal_registry_watch_active = registry.param.watch(
            lambda _e=None: self._sync_processing_to_active_signal(), "active_signal_id"
        )
        self._signal_registry_watch_owner = registry

    def _on_signal_registry_replaced(self, event) -> None:
        try:
            registry = event.new
        except Exception:  # noqa: BLE001
            registry = None
        if registry is None:
            return
        self._attach_signal_registry(registry)
        self._sync_processing_to_active_signal()

    def _sync_processing_to_active_signal(self) -> None:
        sig = self.signal_registry.get_active() if self.signal_registry is not None else None
        self.processing = sig.processing if sig is not None else None

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

    def create_derived_signal(
        self,
        source_id: str,
        name: str,
        processing_config: dict | None = None,
    ) -> str:
        """
        Create a new signal derived from an existing signal.

        Parameters
        ----------
        source_id
            ID of source signal to duplicate.
        name
            Name for new signal.
        processing_config
            Dict of ProcessingChain param values to apply.
        """
        if self.signal_registry is None:
            raise RuntimeError("signal_registry is not initialized")

        new_id = self.signal_registry.duplicate(source_id, name)
        if processing_config:
            sig = self.signal_registry.get(new_id)
            if sig is not None:
                for key, value in dict(processing_config).items():
                    if key in sig.processing.param:
                        setattr(sig.processing, key, value)
        return new_id

    def set_selected_time_from_cursor(self) -> None:
        """Lock current cursor position as selected time."""
        if self.time_hair is not None and self.time_hair.t is not None:
            self.selected_time = float(self.time_hair.t)

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

    def register_event_catalog(self, name: str, catalog: Any, *, style: Any | None = None) -> None:
        """
        Register an EventCatalog as a TensorScope EventStream.

        This is an additive convenience bridge for v2.6.x detector workflows.

        Parameters
        ----------
        name
            Name under which the event stream will be registered.
        catalog
            `cogpy.core.events.EventCatalog` instance.
        style
            Optional EventStyle or dict of EventStyle fields.
        """
        from cogpy.core.events import EventCatalog

        if not isinstance(catalog, EventCatalog):
            raise TypeError(f"catalog must be an EventCatalog, got {type(catalog)!r}")

        stream = catalog.to_event_stream(style=style)
        self.register_events(str(name), stream)

    def run_detector(
        self,
        detector: Any,
        *,
        signal_id: str | None = None,
        event_type: str = "events",
        transform_result: Any | None = None,
        style: Any | None = None,
    ):
        """
        Run an EventDetector and register its results as an event stream.

        Parameters
        ----------
        detector
            Detector instance (e.g. `cogpy.core.detect.BurstDetector`).
        signal_id
            Which signal to run detection on. If None, uses the active signal.
        event_type
            Name to register detected events under.
        transform_result
            Optional pre-computed transform to pass to the detector (e.g. spectrogram).
            If provided, `signal_id` is only used for provenance/selection.
        style
            Optional EventStyle or dict of EventStyle fields for visualization.

        Returns
        -------
        EventCatalog
            Detected events.
        """
        from cogpy.core.events import EventCatalog

        if self.signal_registry is None:
            raise RuntimeError("signal_registry is not initialized")

        signal = None
        if signal_id is None:
            signal = self.signal_registry.get_active()
        else:
            signal = self.signal_registry.get(str(signal_id))
            if signal is None:
                # Allow lookup by human-readable signal name as a convenience.
                try:
                    for _sid, sig in self.signal_registry.signals.items():
                        if getattr(sig, "name", None) == signal_id:
                            signal = sig
                            break
                except Exception:  # noqa: BLE001
                    signal = None

        if signal is None and transform_result is None:
            raise ValueError(f"Signal {signal_id!r} not found and no transform_result provided")

        data = transform_result if transform_result is not None else signal.data

        catalog = detector.detect(data)
        if not isinstance(catalog, EventCatalog):
            raise TypeError(
                "detector.detect(...) must return an EventCatalog for TensorScope integration; "
                f"got {type(catalog)!r}"
            )

        self.register_event_catalog(event_type, catalog, style=style)
        return catalog

    def run_pipeline(
        self,
        pipeline: Any,
        *,
        signal_id: str | None = None,
        event_type: str = "events",
        style: Any | None = None,
    ):
        """
        Run a DetectionPipeline and register its output as an event stream.

        Parameters
        ----------
        pipeline
            `cogpy.core.detect.pipeline.DetectionPipeline` instance.
        signal_id
            Which signal to run the pipeline on. If None, uses the active signal.
        event_type
            Name to register detected events under.
        style
            Optional EventStyle or dict of EventStyle fields for visualization.
        """
        from cogpy.core.events import EventCatalog

        if self.signal_registry is None:
            raise RuntimeError("signal_registry is not initialized")

        signal = self.signal_registry.get_active() if signal_id is None else self.signal_registry.get(str(signal_id))
        if signal is None:
            raise ValueError(f"Signal {signal_id!r} not found")

        catalog = pipeline.run(signal.data)
        if not isinstance(catalog, EventCatalog):
            raise TypeError(
                "pipeline.run(...) must return an EventCatalog for TensorScope integration; "
                f"got {type(catalog)!r}"
            )

        self.register_event_catalog(event_type, catalog, style=style)
        return catalog

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
            "selected_time": self.selected_time,
            "time_window": tuple(self.time_window.window) if self.time_window else None,
            "selected_channels": [list(ch) for ch in self.selected_channels],
            "processing": self.processing.to_dict() if self.processing else {},
            "active_layout_preset": self.active_layout_preset,
            "active_modality": self.active_modality,
            "signal_registry": self.signal_registry.to_dict() if self.signal_registry else {},
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
        data = data_resolver()
        state = cls(data)

        if state_dict.get("current_time") is not None:
            state.current_time = float(state_dict["current_time"])

        if state_dict.get("selected_time") is not None:
            state.selected_time = float(state_dict["selected_time"])

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

        # Restore signal registry (new format). If absent, fall back to restoring
        # the legacy processing config into the base signal.
        if state_dict.get("signal_registry") and state.signal_registry is not None:
            try:
                state.signal_registry = SignalRegistry.from_dict(
                    dict(state_dict.get("signal_registry") or {}), data
                )
            except Exception:  # noqa: BLE001
                pass

        # Restore processing settings to the active signal's chain (legacy key).
        proc = state_dict.get("processing") or {}
        try:
            if state.processing is not None:
                for key, value in dict(proc).items():
                    if key in state.processing.param:
                        setattr(state.processing, key, value)
        except Exception:  # noqa: BLE001
            pass

        # Restore event streams (Phase 6: session persistence).
        ev = state_dict.get("event_registry") or {}
        try:
            from .events.registry import EventRegistry

            state.event_registry = EventRegistry.from_dict(ev)
        except Exception:  # noqa: BLE001
            pass

        return state
