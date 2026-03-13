"""Spectrogram visualization layer (Phase 5)."""

from __future__ import annotations

from typing import Any

import numpy as np
import panel as pn

from .base import TensorLayer


class SpectrogramLayer(TensorLayer):
    """
    Simple spectrogram heatmap layer.

    Notes
    -----
    This is a lightweight reference implementation intended to validate the
    multi-modal switching mechanics. It displays a 2D (time, freq) slice by
    averaging any spatial/channel dimensions.
    """

    def __init__(
        self,
        state,
        *,
        channel_idx: int = 0,
        title: str = "Spectrogram",
        cmap: str = "viridis",
    ):
        super().__init__(state)
        self.layer_id = "spectrogram"
        self.title = str(title)
        self.channel_idx = int(channel_idx)
        self._cmap = str(cmap)

        self._hv_pane = pn.pane.HoloViews(None, sizing_mode="stretch_both")
        self._msg = pn.pane.Markdown("", sizing_mode="stretch_both")

        self._watch(state, self._on_modality_change, "active_modality")
        self._refresh()

    def _on_modality_change(self, event=None) -> None:
        try:
            print(f"SpectrogramLayer: modality changed to {getattr(event, 'new', None)}")
        except Exception:  # noqa: BLE001
            pass
        self._refresh()

    def _refresh(self) -> None:
        # Prefer the state parameter (what the UI controls), but fall back to
        # the registry if needed.
        active_name = getattr(self.state, "active_modality", None)
        if active_name is None and getattr(self.state, "data_registry", None) is not None:
            active_name = self.state.data_registry.get_active_name()
        active_name = str(active_name) if active_name is not None else None

        modality = None
        if getattr(self.state, "data_registry", None) is not None and active_name:
            modality = self.state.data_registry.get(active_name)
        if modality is None:
            modality = getattr(self.state, "get_active_modality", lambda: None)()

        if modality is None:
            self._msg.object = f"No active modality. (active_modality={active_name!r})"
            self._msg.styles = {"color": "#888", "padding": "40px", "text-align": "center"}
            self._hv_pane.object = None
            return

        if getattr(modality, "modality_type", None) != "spectrogram":
            shown = active_name or getattr(modality, "modality_type", "unknown")
            self._msg.object = (
                "### Spectrogram Layer\n\n"
                f"Currently showing: **{shown}** modality\n\n"
                "Switch to `spectrogram` in the dropdown to view time-frequency analysis.\n\n"
                "---\n\n"
                "*The spectrogram layer displays power spectral density over time.*"
            )
            self._msg.styles = {
                "color": "#888",
                "padding": "40px",
                "text-align": "center",
                "background": "#1a1a2e",
                "border-radius": "8px",
            }
            self._hv_pane.object = None
            return

        self._msg.object = ""
        self._msg.styles = {}
        self._hv_pane.object = self._spectrogram_view(modality.data)

    def _spectrogram_view(self, data) -> Any:
        import holoviews as hv

        hv.extension("bokeh")

        da = data
        # If the modality is flat spectrogram, allow selecting a channel.
        if "channel" in da.dims:
            try:
                da = da.isel(channel=self.channel_idx)
            except Exception:  # noqa: BLE001
                pass

        # Reduce to (time, freq) by averaging remaining axes.
        reduce_dims = [d for d in da.dims if d not in {"time", "freq"}]
        if reduce_dims:
            da = da.mean(dim=reduce_dims)

        if da.dims != ("time", "freq"):
            da = da.transpose("time", "freq")

        t = np.asarray(da["time"].values, dtype=float)
        f = np.asarray(da["freq"].values, dtype=float)
        z = np.asarray(da.values, dtype=float)

        # NOTE: hv.Image expects evenly-sampled axes; our demo uses log-spaced
        # frequency bins. For irregular sampling, prefer QuadMesh.
        def _is_evenly_sampled(vals: np.ndarray, *, rtol: float = 1e-3) -> bool:
            if vals.size < 3:
                return True
            d = np.diff(vals)
            d0 = float(d[0])
            if not np.isfinite(d0):
                return False
            return bool(np.allclose(d, d0, rtol=rtol, atol=0.0))

        if _is_evenly_sampled(f):
            # hv.Image expects a 2D array indexed by (x, y). We'll use (time, freq).
            el = hv.Image((t, f, z.T), kdims=["time", "freq"], vdims=[da.name or "power"])
        else:
            el = hv.QuadMesh((t, f, z.T), kdims=["time", "freq"], vdims=[da.name or "power"])

        base_opts = dict(
            cmap=self._cmap,
            colorbar=True,
            width=700,
            height=500,
            xlabel="Time (s)",
            ylabel="Frequency (Hz)",
            tools=["hover"],
            fontscale=1.2,
        )
        # Don't pass string-valued `aspect` here: some HoloViews versions expect
        # aspect to be numeric and will crash in padding calculations.
        return el.opts(**base_opts)

    def panel(self) -> pn.viewable.Viewable:
        if self._panel is None:
            self._panel = pn.Column(self._msg, self._hv_pane, sizing_mode="stretch_both")
        return self._panel
