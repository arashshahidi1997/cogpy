"""Spatial map visualization layer wrapping GridFrameElement."""

from __future__ import annotations

from cogpy.core.plot.grid_frame_element import GridFrameElement

from .base import TensorLayer


class SpatialMapLayer(TensorLayer):
    """Spatial scalar map layer driven by TimeHair (wraps GridFrameElement)."""

    def __init__(
        self,
        state,
        mode: str = "instantaneous",
        window_s: float = 0.1,
        colormap: str = "rdbu_r",
    ):
        super().__init__(state)
        self.layer_id = "spatial_map"
        mode_s = str(mode)
        self.title = "Spatial LFP" if mode_s == "instantaneous" else f"Spatial {mode_s.upper()}"

        modality = state.data_registry.get("grid_lfp") if state.data_registry is not None else None
        if modality is None:
            raise ValueError("No grid_lfp modality registered")

        self._element = self._add_data_ref(
            GridFrameElement(
                sig_grid=modality.data,
                hair=state.time_hair,
                mode=mode,
                window_s=window_s,
                chain=state.processing,
                colormap=colormap,
                symmetric=(mode_s == "instantaneous"),
                title=self.title,
            )
        )

    def panel(self):
        if self._panel is None:
            self._panel = self._element.panel()
        return self._panel
