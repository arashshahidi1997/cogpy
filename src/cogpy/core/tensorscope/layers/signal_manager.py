"""Signal manager UI layer."""

from __future__ import annotations

import datetime

import panel as pn

from .base import TensorLayer

__all__ = ["SignalManagerLayer"]


class SignalManagerLayer(TensorLayer):
    """
    UI for managing signal objects.

    Allows users to:
    - View all signals
    - Duplicate signals
    - Configure processing per signal
    - Delete derived signals (not base signals)
    - Switch active signal
    """

    def __init__(self, state):
        super().__init__(state)
        self.layer_id = "signal_manager"
        self.title = "Signal Manager"

        self._signal_list = None
        self._processing_pane = pn.Column(sizing_mode="stretch_width")
        self._build_ui()

        if getattr(state, "signal_registry", None) is not None:
            self._watch(state.signal_registry, self._refresh_signal_list, "signals")
            self._watch(state.signal_registry, self._on_active_signal_change, "active_signal_id")

    def _signal_options(self) -> dict[str, str]:
        reg = getattr(self.state, "signal_registry", None)
        if reg is None:
            return {}
        return {sig.name: sid for sid, sig in (reg.signals or {}).items()}

    def _build_ui(self) -> None:
        options = self._signal_options()
        active = getattr(getattr(self.state, "signal_registry", None), "active_signal_id", None)

        self._signal_list = pn.widgets.Select(
            name="Signals",
            options=options,
            value=active if active in options.values() else None,
            size=8,
        )
        self._signal_list.param.watch(self._update_processing_controls, "value")

        duplicate_btn = pn.widgets.Button(name="Duplicate", button_type="primary", width=110)
        delete_btn = pn.widgets.Button(name="Delete", button_type="danger", width=90)
        set_active_btn = pn.widgets.Button(name="Set Active", button_type="success", width=110)

        duplicate_btn.on_click(self._duplicate_signal)
        delete_btn.on_click(self._delete_signal)
        set_active_btn.on_click(self._set_active_signal)

        self._ui = pn.Column(
            pn.pane.Markdown("## Signals"),
            self._signal_list,
            pn.Row(duplicate_btn, delete_btn, set_active_btn, sizing_mode="stretch_width"),
            pn.layout.Divider(),
            self._processing_pane,
            sizing_mode="stretch_width",
        )

        self._update_processing_controls()

    def _selected_signal_id(self) -> str | None:
        return getattr(self._signal_list, "value", None)

    def _duplicate_signal(self, _event=None) -> None:
        reg = getattr(self.state, "signal_registry", None)
        sid = self._selected_signal_id()
        if reg is None or not sid:
            return

        src = reg.get(sid)
        if src is None:
            return

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        new_name = f"{src.name} ({timestamp})"
        new_id = reg.duplicate(sid, new_name)

        self._refresh_signal_list()
        self._signal_list.value = new_id

    def _delete_signal(self, _event=None) -> None:
        reg = getattr(self.state, "signal_registry", None)
        sid = self._selected_signal_id()
        if reg is None or not sid:
            return

        sig = reg.get(sid)
        if sig is None:
            return

        if bool(sig.metadata.get("is_base")):
            self._processing_pane.objects = [
                pn.pane.Alert("Cannot delete base signals", alert_type="warning")
            ]
            return

        reg.remove(sid)
        self._refresh_signal_list()

    def _set_active_signal(self, _event=None) -> None:
        reg = getattr(self.state, "signal_registry", None)
        sid = self._selected_signal_id()
        if reg is None or not sid:
            return
        reg.set_active(sid)

    def _on_active_signal_change(self, event=None) -> None:
        if self._signal_list is None:
            return
        active = getattr(event, "new", None)
        opts = set((self._signal_list.options or {}).values())
        if active in opts and self._signal_list.value != active:
            self._signal_list.value = active
        self._update_processing_controls()

    def _refresh_signal_list(self, *_args) -> None:
        if self._signal_list is None:
            return

        options = self._signal_options()
        self._signal_list.options = options

        current = self._signal_list.value
        if current not in options.values():
            self._signal_list.value = next(iter(options.values()), None)

        self._update_processing_controls()

    def _update_processing_controls(self, *_args) -> None:
        reg = getattr(self.state, "signal_registry", None)
        sid = self._selected_signal_id()
        if reg is None or not sid:
            self._processing_pane.objects = []
            return

        sig = reg.get(sid)
        if sig is None:
            self._processing_pane.objects = [pn.pane.Markdown("**Signal not found**")]
            return

        is_base = bool(sig.metadata.get("is_base", False))
        is_active = sid == getattr(reg, "active_signal_id", None)

        status_md = f"### {sig.name}\n\n"
        if is_base:
            status_md += "Base signal (cannot delete)\n\n"
        if is_active:
            status_md += "Active signal\n\n"

        self._processing_pane.objects = [pn.pane.Markdown(status_md), sig.processing.controls()]

    def panel(self):
        return self._ui

