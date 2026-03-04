"""
TensorScope application shell.

Main composition root that owns state, layers, and layout.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import panel as pn

from cogpy.core.plot.theme import BG_PANEL, BLUE, TEAL

from .layout import LayoutManager
from .layers.manager import LayerManager, LayerSpec
from .state import TensorScopeState

if TYPE_CHECKING:
    import xarray as xr


class TensorScopeApp:
    """
    TensorScope application composition root.

    Owns:
    - TensorScopeState (central state)
    - LayerManager (layer instances)
    - LayoutManager (UI layout)
    """

    def __init__(self, data: Any, title: str = "TensorScope", theme: str = "dark"):
        self.state = TensorScopeState(data)
        self.layer_manager = LayerManager(self.state)
        self.layout_manager = LayoutManager(title=title, theme=theme)
        self._register_default_layers()
        self._panels: dict[str, pn.viewable.Viewable] = {}

    def _register_default_layers(self) -> None:
        from .layers import (
            ChannelSelectorLayer,
            EventOverlayLayer,
            EventTableLayer,
            ProcessingControlsLayer,
            SignalManagerLayer,
            SpatialMapLayer,
            SpectrogramLayer,
            TimeseriesLayer,
            TimeNavigatorLayer,
        )

        self.layer_manager.register(
            LayerSpec(
                layer_id="timeseries",
                title="Timeseries",
                factory=lambda s: TimeseriesLayer(s),
                description="Stacked timeseries traces",
                layer_type="timeseries",
            )
        )
        self.layer_manager.register(
            LayerSpec(
                layer_id="spatial_map",
                title="Spatial Map",
                factory=lambda s: SpatialMapLayer(s),
                description="Time-linked spatial scalar map",
                layer_type="spatial",
            )
        )
        self.layer_manager.register(
            LayerSpec(
                layer_id="selector",
                title="Channel Selector",
                factory=lambda s: ChannelSelectorLayer(s),
                description="Interactive channel grid selection",
                layer_type="controls",
            )
        )
        self.layer_manager.register(
            LayerSpec(
                layer_id="processing",
                title="Processing",
                factory=lambda s: ProcessingControlsLayer(s),
                description="Transform controls",
                layer_type="controls",
            )
        )
        self.layer_manager.register(
            LayerSpec(
                layer_id="signal_manager",
                title="Signal Manager",
                factory=lambda s: SignalManagerLayer(s),
                description="Manage signal objects and processing pipelines",
                layer_type="controls",
            )
        )
        self.layer_manager.register(
            LayerSpec(
                layer_id="navigator",
                title="Time Navigator",
                factory=lambda s: TimeNavigatorLayer(s),
                description="Play/pause/step controls",
                layer_type="navigation",
            )
        )

        # Phase 5: spectrogram layer (only renders when active_modality == "spectrogram")
        self.layer_manager.register(
            LayerSpec(
                layer_id="spectrogram",
                title="Spectrogram",
                factory=lambda s: SpectrogramLayer(s),
                description="Spectrogram heatmap (active when modality is spectrogram)",
                layer_type="spectrogram",
            )
        )

        # Phase 4: event layers (default stream name: "bursts")
        self.layer_manager.register(
            LayerSpec(
                layer_id="event_table",
                title="Event Table",
                factory=lambda s: EventTableLayer(s, "bursts"),
                description="Event table with navigation",
                layer_type="events",
            )
        )
        self.layer_manager.register(
            LayerSpec(
                layer_id="event_overlay",
                title="Event Overlay",
                factory=lambda s: EventOverlayLayer(s, "bursts"),
                description="Event markers overlay (placeholder)",
                layer_type="events",
            )
        )

    def add_layer(self, layer_id: str, instance_id: str | None = None) -> "TensorScopeApp":
        # Default instance_id: use the layer_id itself (so layout presets can
        # refer to stable panel IDs like "timeseries", "spatial_map", etc.).
        if instance_id is None and (layer_id not in self._panels) and (self.layer_manager.get(layer_id) is None):
            instance_id = layer_id

        layer = self.layer_manager.add(layer_id, instance_id)
        iid = getattr(layer, "instance_id", instance_id or layer_id)

        header = BLUE if layer_id in {"timeseries", "spatial_map"} else TEAL
        panel = pn.Card(
            layer.panel(),
            title=getattr(layer, "title", layer_id),
            header_background=header,
            styles={"background": BG_PANEL},
            sizing_mode="stretch_both",
        )

        self._panels[str(iid)] = panel
        return self

    def remove_layer(self, instance_id: str) -> "TensorScopeApp":
        self.layer_manager.remove(instance_id)
        self._panels.pop(instance_id, None)
        return self

    def with_layout(self, preset_name: str) -> "TensorScopeApp":
        self.layout_manager._current_preset = str(preset_name)
        try:
            self.state.active_layout_preset = str(preset_name)
        except Exception:  # noqa: BLE001
            pass
        return self

    def build(self) -> pn.template.FastGridTemplate:
        preset = self.layout_manager.current_preset
        sidebar_ids = self.layout_manager.sidebar_panels_for(preset)
        sidebar_widgets = [self._panels[i] for i in sidebar_ids if i in self._panels]

        template = self.layout_manager.build_template(sidebar=sidebar_widgets)
        self.layout_manager.apply_preset(preset, self._panels)
        return template

    def servable(self) -> pn.template.FastGridTemplate:
        template = self.build()
        template.servable()
        return template

    def shutdown(self) -> None:
        self.layer_manager.dispose_all()
        self._panels.clear()

    def to_session(self) -> dict[str, Any]:
        return {
            "version": "1.0",
            "state": self.state.to_dict(),
            "layout": self.layout_manager.to_dict(),
            # Store layer *types* in order. (Instance IDs are implementation detail.)
            "layers": list(self.layer_manager.list_instance_types()),
        }

    @classmethod
    def from_session(cls, session: dict[str, Any], *, data_resolver) -> "TensorScopeApp":
        data = data_resolver()
        layout_config = session.get("layout", {}) or {}

        app = cls(
            data,
            title=layout_config.get("title", "TensorScope"),
            theme=layout_config.get("theme", "dark"),
        )

        # Restore state (and rebuild managers bound to that restored state).
        state_dict = session.get("state", {}) or {}
        app.state = TensorScopeState.from_dict(state_dict, data_resolver=lambda: data)
        app.layer_manager = LayerManager(app.state)
        app.layout_manager = LayoutManager.from_dict(layout_config)
        app._register_default_layers()
        app._panels = {}

        preset = layout_config.get("current_preset", "default")
        app.with_layout(preset)

        for layer_type in session.get("layers", []) or []:
            if not layer_type:
                continue
            app.add_layer(str(layer_type))

        return app
