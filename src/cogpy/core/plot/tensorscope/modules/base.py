"""
Module system for declarative view composition (v2.2).

Modules are named collections of `ViewSpec`s with a layout policy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..view_factory import ViewFactory
from ..view_spec import ViewSpec

__all__ = ["ViewPresetModule", "ModuleRegistry"]


@dataclass(frozen=True)
class ViewPresetModule:
    """
    A preset module: a collection of view specs with a layout policy.

    Parameters
    ----------
    name
        Module identifier.
    description
        Human-readable description.
    specs
        View specifications that will be instantiated via `ViewFactory`.
    layout
        Layout policy: "grid", "stack", or "tabs" (backend-dependent).
    """

    name: str
    description: str
    specs: list[ViewSpec]
    layout: str = "grid"
    activate_fn: Callable[[object], object] | None = None

    def activate(self, state):
        """Instantiate all views and return a HoloViews layout."""
        import holoviews as hv

        hv.extension("bokeh")

        if self.activate_fn is not None:
            return self.activate_fn(state)

        views = [ViewFactory.create(spec, state) for spec in self.specs]
        return self._arrange(views)

    def _arrange(self, views):
        import holoviews as hv

        views_list = list(views)
        if self.layout == "stack":
            return hv.Layout(views_list).cols(1)
        if self.layout == "tabs":
            # Tabs are backend-specific; provide a stable fallback.
            return hv.Layout(views_list).cols(max(len(views_list), 1))
        # Default: grid.
        return hv.Layout(views_list).cols(max(len(views_list), 1))


class ModuleRegistry:
    """Registry of available view modules."""

    def __init__(self) -> None:
        self._modules: dict[str, ViewPresetModule] = {}
        self._register_builtin()

    def _register_builtin(self) -> None:
        from . import basic, comparison, electrode_panel, event_explorer, montage, orthoslicer, psd_explorer

        self.register(basic.MODULE)
        self.register(comparison.MODULE)
        self.register(montage.MODULE)
        self.register(electrode_panel.MODULE)
        self.register(orthoslicer.MODULE)
        self.register(event_explorer.MODULE)
        self.register(psd_explorer.MODULE)

    def register(self, module: ViewPresetModule) -> None:
        self._modules[str(module.name)] = module

    def get(self, name: str) -> ViewPresetModule | None:
        return self._modules.get(str(name))

    def list(self) -> list[str]:
        return list(self._modules.keys())

    def list_with_descriptions(self) -> list[tuple[str, str]]:
        return [(name, mod.description) for name, mod in self._modules.items()]
