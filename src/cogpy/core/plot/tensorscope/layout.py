"""
Layout management for TensorScope.

Handles FastGridTemplate configuration and layout presets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import panel as pn


@dataclass(frozen=True, slots=True)
class LayoutPreset:
    """
    Layout preset configuration.

    grid_assignments maps panel_id -> (r0, r1, c0, c1)
    """

    name: str
    description: str
    grid_assignments: dict[str, tuple[int, int, int, int]]
    sidebar_panels: list[str]


class LayoutManager:
    """Manages FastGridTemplate layout and presets."""

    def __init__(self, title: str = "TensorScope", theme: str = "dark"):
        self.title = str(title)
        self.theme = str(theme)
        self._template: pn.template.FastGridTemplate | None = None
        self._current_preset = "default"

        self._presets: dict[str, LayoutPreset] = {
            "default": LayoutPreset(
                name="default",
                description="Spatial + navigator on top, timeseries bottom",
                grid_assignments={
                    "spatial_map": (0, 5, 0, 6),
                    "navigator": (0, 5, 6, 12),
                    "timeseries": (5, 10, 0, 12),
                },
                sidebar_panels=["selector", "processing"],
            ),
            "spatial_focus": LayoutPreset(
                name="spatial_focus",
                description="Large spatial map, small timeseries",
                grid_assignments={
                    "spatial_map": (0, 8, 0, 12),
                    "timeseries": (8, 10, 0, 12),
                },
                sidebar_panels=["selector", "processing", "navigator"],
            ),
            "timeseries_focus": LayoutPreset(
                name="timeseries_focus",
                description="Large timeseries, small spatial thumbnail",
                grid_assignments={
                    "spatial_map": (0, 3, 0, 4),
                    "navigator": (0, 3, 4, 12),
                    "timeseries": (3, 10, 0, 12),
                },
                sidebar_panels=["selector", "processing"],
            ),
            "psd_explorer": LayoutPreset(
                name="psd_explorer",
                description="Basic views + PSD explorer panel",
                grid_assignments={
                    "spatial_map": (0, 5, 0, 4),
                    "navigator": (0, 5, 4, 12),
                    "timeseries": (5, 10, 0, 12),
                    "psd_explorer": (10, 18, 0, 12),
                },
                sidebar_panels=["selector", "processing", "psd_settings"],
            ),
        }

    def preset_names(self) -> list[str]:
        return sorted(self._presets.keys())

    def get_preset(self, preset_name: str) -> LayoutPreset:
        if preset_name not in self._presets:
            raise ValueError(
                f"Preset {preset_name!r} not found. Available: {sorted(self._presets.keys())}"
            )
        return self._presets[preset_name]

    def build_template(
        self,
        *,
        sidebar: list[pn.viewable.Viewable] | None = None,
        sidebar_width: int = 400,
        row_height: int = 80,
    ) -> pn.template.FastGridTemplate:
        self._template = pn.template.FastGridTemplate(
            title=self.title,
            theme=self.theme,
            sidebar_width=int(sidebar_width),
            sidebar=list(sidebar or []),
            row_height=int(row_height),
        )
        return self._template

    def apply_preset(self, preset_name: str, panels: dict[str, pn.viewable.Viewable]) -> None:
        preset = self.get_preset(preset_name)
        if self._template is None:
            raise RuntimeError("Must call build_template() first")

        for panel_id, (r0, r1, c0, c1) in preset.grid_assignments.items():
            if panel_id in panels:
                self._template.main[r0:r1, c0:c1] = panels[panel_id]

        self._current_preset = preset_name

    def sidebar_panels_for(self, preset_name: str) -> list[str]:
        return list(self.get_preset(preset_name).sidebar_panels)

    @property
    def current_preset(self) -> str:
        return self._current_preset

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "theme": self.theme,
            "current_preset": self._current_preset,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "LayoutManager":
        manager = cls(
            title=config.get("title", "TensorScope"),
            theme=config.get("theme", "dark"),
        )
        manager._current_preset = config.get("current_preset", "default")
        return manager
