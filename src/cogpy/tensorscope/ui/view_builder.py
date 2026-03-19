"""
View Builder UI for interactive view creation (v2.3).

Allows users to build ViewSpecs through a small Panel GUI and preview them.
"""

from __future__ import annotations

import panel as pn

from ..view_factory import ViewFactory
from ..view_spec import ViewSpec

__all__ = ["ViewBuilderLayer"]


class ViewBuilderLayer:
    """Interactive UI for building custom views."""

    def __init__(self, state):
        self.state = state

        sig = None
        reg = getattr(state, "signal_registry", None)
        if reg is not None:
            sig = reg.get_active()

        if sig is not None:
            available_dims = list(sig.data.dims)
        else:
            available_dims = ["time", "AP", "ML"]
        self.available_dims = available_dims

        self._build_ui()

    def _default_kdims(self) -> list[str]:
        if ("AP" in self.available_dims) and ("ML" in self.available_dims):
            return ["AP", "ML"]
        if "time" in self.available_dims and len(self.available_dims) >= 2:
            return [d for d in self.available_dims if d != "time"][:2]
        return self.available_dims[:2]

    def _default_controls(self) -> list[str]:
        return ["time"] if "time" in self.available_dims else []

    def _build_ui(self) -> None:
        reg = getattr(self.state, "signal_registry", None)
        signal_names = reg.list_names() if reg is not None else []
        active_id = getattr(reg, "active_signal_id", None) if reg is not None else None

        self.signal_selector = pn.widgets.Select(
            name="Signal",
            options={name: sid for sid, name in signal_names},
            value=active_id if active_id in {sid for sid, _name in signal_names} else None,
        )

        self.kdims_selector = pn.widgets.CheckBoxGroup(
            name="Display Dimensions (kdims)",
            options=self.available_dims,
            value=self._default_kdims(),
            inline=False,
        )

        self.controls_selector = pn.widgets.CheckBoxGroup(
            name="Control Dimensions (controls)",
            options=self.available_dims,
            value=self._default_controls(),
            inline=False,
        )

        self.view_type = pn.widgets.RadioButtonGroup(
            name="View Type",
            options=["auto", "Image", "Curve", "Overlay"],
            value="auto",
        )

        self.colormap = pn.widgets.Select(
            name="Colormap",
            options=["RdBu_r", "viridis", "plasma", "coolwarm", "seismic"],
            value="RdBu_r",
        )

        self.symmetric_clim = pn.widgets.Checkbox(
            name="Symmetric color limits (center at 0)",
            value=True,
        )

        self.title = pn.widgets.TextInput(
            name="Title (optional)",
            placeholder="View title...",
        )

        self.preview_btn = pn.widgets.Button(name="Preview", button_type="primary")
        self.preview_btn.on_click(self._on_preview)

        self.add_btn = pn.widgets.Button(name="Add to Layout", button_type="success")
        self.save_btn = pn.widgets.Button(name="Save as Module")

        self.preview_pane = pn.Column(
            pn.pane.Markdown("*Click Preview to see view*"),
            sizing_mode="stretch_both",
        )

        self._ui = pn.Column(
            pn.pane.Markdown("## View Builder"),
            self.signal_selector,
            pn.layout.Divider(),
            self.kdims_selector,
            self.controls_selector,
            pn.layout.Divider(),
            self.view_type,
            self.colormap,
            self.symmetric_clim,
            self.title,
            pn.layout.Divider(),
            pn.Row(self.preview_btn, self.add_btn, self.save_btn),
            pn.layout.Divider(),
            pn.pane.Markdown("### Preview"),
            self.preview_pane,
            sizing_mode="stretch_width",
        )

    def build_spec(self) -> ViewSpec:
        return ViewSpec(
            kdims=list(self.kdims_selector.value),
            controls=list(self.controls_selector.value),
            signal_id=self.signal_selector.value,
            view_type=str(self.view_type.value),
            colormap=str(self.colormap.value),
            symmetric_clim=bool(self.symmetric_clim.value),
            title=str(self.title.value) if self.title.value else None,
        )

    def _on_preview(self, _event=None) -> None:
        try:
            spec = self.build_spec()
            view = ViewFactory.create(spec, self.state)
            self.preview_pane.objects = [pn.pane.HoloViews(view, sizing_mode="stretch_both")]
        except Exception as exc:  # noqa: BLE001
            self.preview_pane.objects = [pn.pane.Alert(f"Error creating view: {exc}", alert_type="danger")]

    def panel(self):
        return self._ui

