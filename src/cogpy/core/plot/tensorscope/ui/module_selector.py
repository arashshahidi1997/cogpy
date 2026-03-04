"""Module Selector UI for switching between view presets (v2.3)."""

from __future__ import annotations

import panel as pn

__all__ = ["ModuleSelectorLayer"]


class ModuleSelectorLayer:
    """UI for selecting and loading registered `ViewPresetModule`s."""

    def __init__(self, state, registry):
        self.state = state
        self.registry = registry

        self.layout_container = pn.Column(
            pn.pane.Markdown("*No module loaded*"),
            sizing_mode="stretch_both",
        )

        self._build_ui()

    def _build_ui(self) -> None:
        module_names = list(self.registry.list())
        module_options = {}
        for name in module_names:
            mod = self.registry.get(name)
            desc = getattr(mod, "description", "") if mod is not None else ""
            module_options[f"{name} — {desc}"] = name

        default_value = module_names[0] if module_names else None

        self.module_selector = pn.widgets.Select(
            name="Select Module",
            options=module_options,
            value=default_value,
        )

        self.info_pane = pn.pane.Markdown(sizing_mode="stretch_width")

        self.load_btn = pn.widgets.Button(name="Load Module", button_type="primary")
        self.load_btn.on_click(self._on_load)

        self.module_selector.param.watch(self._update_info, "value")
        self._update_info()

        self._ui = pn.Column(
            pn.pane.Markdown("## Module Selector"),
            self.module_selector,
            self.info_pane,
            self.load_btn,
            sizing_mode="stretch_width",
        )

    def _update_info(self, *_args) -> None:
        name = self.module_selector.value
        if not name:
            self.info_pane.object = ""
            return
        mod = self.registry.get(name)
        if mod is None:
            self.info_pane.object = f"**{name}**\n\n*(module not found)*"
            return

        n_views = len(getattr(mod, "specs", []) or [])
        layout = getattr(mod, "layout", "grid")
        desc = getattr(mod, "description", "")
        self.info_pane.object = f"**{mod.name}**\n\n{desc}\n\nViews: {n_views}  \nLayout: {layout}"

    def _on_load(self, _event=None) -> None:
        name = self.module_selector.value
        if not name:
            return

        try:
            mod = self.registry.get(name)
            if mod is None:
                raise ValueError(f"Module {name!r} not found")
            layout = mod.activate(self.state)
            self.layout_container.objects = [pn.pane.HoloViews(layout, sizing_mode="stretch_both")]
        except Exception as exc:  # noqa: BLE001
            self.layout_container.objects = [pn.pane.Alert(f"Error loading module: {exc}", alert_type="danger")]

    def panel(self):
        return self._ui

    def get_layout_container(self):
        return self.layout_container

