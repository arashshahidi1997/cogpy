"""
Spatial LFP view: instantaneous voltage map.

Shows signal.data at selected time point as an (AP, ML) heatmap.
No temporal aggregation - pure spatial slice.
"""

from __future__ import annotations

import numpy as np
import panel as pn

from cogpy.core.tensorscope.data.alignment import find_nearest_time_index

from .base import View

__all__ = ["SpatialLFPView"]


class SpatialLFPView(View):
    """
    Spatial map of instantaneous LFP voltage.

    Parameters
    ----------
    state
        TensorScopeState
    signal_id
        Signal id to render
    selected_time_source
        One of: 'cursor', 'selected', 'independent'
    independent_time
        Initial time for independent mode
    colormap
        HoloViews colormap name
    symmetric_limits
        If True, centers clim at 0 using +/- max(|x|)
    view_id
        Optional view identifier (for duplication)
    """

    def __init__(
        self,
        state,
        signal_id: str,
        selected_time_source: str = "cursor",
        independent_time: float | None = None,
        colormap: str = "RdBu_r",
        symmetric_limits: bool = True,
        view_id: str | None = None,
    ):
        super().__init__(state, signal_id, view_id)

        self.selected_time_source = str(selected_time_source)
        self.independent_time = independent_time
        self.colormap = str(colormap)
        self.symmetric_limits = bool(symmetric_limits)

        self._container = pn.Column(sizing_mode="stretch_both")
        self.time_slider = None

        if self.selected_time_source == "cursor":
            self._watch(state.time_hair, self._update, "t")
        elif self.selected_time_source == "selected":
            self._watch(state, self._update, "selected_time")
        elif self.selected_time_source == "independent":
            sig = self.get_signal()
            if sig is not None and "time" in sig.data.dims:
                t_vals = np.asarray(sig.data["time"].values, dtype=float)
                t_min, t_max = float(t_vals[0]), float(t_vals[-1])
                fs = float(sig.data.attrs.get("fs", 1000.0))
                step = 1.0 / fs if fs > 0 else 0.001
                init = float(independent_time) if independent_time is not None else t_min
                self.time_slider = pn.widgets.FloatSlider(
                    name="Time (s)",
                    start=t_min,
                    end=t_max,
                    value=init,
                    step=step,
                )
                self.time_slider.param.watch(self._update, "value")
        else:
            raise ValueError(
                "selected_time_source must be one of "
                f"'cursor','selected','independent'; got {self.selected_time_source!r}"
            )

        # Link to shared spatial selection space if present.
        space = getattr(state, "spatial_space", None)
        self._space_watch_handle = None
        if space is not None:
            self._space_watch_handle = space.watch(self._update_spatial_marker)

        self._update()

    def _get_current_time(self) -> float | None:
        if self.selected_time_source == "cursor":
            return self.state.time_hair.t
        if self.selected_time_source == "selected":
            return getattr(self.state, "selected_time", None)
        if self.selected_time_source == "independent":
            return float(self.time_slider.value) if self.time_slider is not None else None
        return None

    def _update(self, *_args) -> None:
        sig = self.get_signal()
        if sig is None:
            self._container.objects = [pn.pane.Markdown("**No signal selected**")]
            return

        t = self._get_current_time()
        if t is None:
            self._container.objects = [pn.pane.Markdown("**No time selected**")]
            return

        data = sig.data
        if "time" not in data.dims:
            self._container.objects = [pn.pane.Markdown("**Signal has no time dimension**")]
            return
        if not (("AP" in data.dims) and ("ML" in data.dims)):
            self._container.objects = [pn.pane.Markdown("**Signal is not grid-shaped (AP×ML)**")]
            return

        t_vals = np.asarray(data["time"].values, dtype=float)
        time_idx = find_nearest_time_index(float(t), t_vals)
        actual_t = float(t_vals[time_idx])

        spatial_slice = data.isel(time=int(time_idx)).transpose("AP", "ML")
        plot = self._render_heatmap(spatial_slice, actual_t)

        panes = []
        if self.selected_time_source == "independent" and self.time_slider is not None:
            panes.append(self.time_slider)
        panes.append(pn.pane.HoloViews(plot, sizing_mode="stretch_both"))
        self._container.objects = panes

    def _render_heatmap(self, spatial_slice, t: float):
        import holoviews as hv
        from holoviews.streams import Tap

        hv.extension("bokeh")

        vals = np.asarray(spatial_slice.values, dtype=float)
        if self.symmetric_limits:
            vmax = float(np.nanmax(np.abs(vals))) if vals.size else 1.0
            vmin = -vmax
        else:
            vmin = float(np.nanmin(vals)) if vals.size else 0.0
            vmax = float(np.nanmax(vals)) if vals.size else 1.0

        n_ap = int(spatial_slice.sizes.get("AP", vals.shape[0] if vals.ndim > 0 else 0))
        n_ml = int(spatial_slice.sizes.get("ML", vals.shape[1] if vals.ndim > 1 else 0))

        # Use index coordinates for consistent selection semantics (AP/ML indices),
        # since ChannelGrid and other TensorScope components use indices.
        ap_idx = np.arange(n_ap)
        ml_idx = np.arange(n_ml)
        img = hv.Image(
            (ml_idx, ap_idx, vals),
            kdims=["ML", "AP"],
            vdims=["LFP (a.u.)"],
        ).opts(
            cmap=self.colormap,
            clim=(vmin, vmax),
            colorbar=True,
            tools=["hover", "tap"],
            xlabel="ML (index)",
            ylabel="AP (index)",
            title=f"LFP @ t={t:.3f}s",
            width=420,
            height=420,
            aspect="equal",
        )

        space = getattr(self.state, "spatial_space", None)
        if space is not None:
            ap_sel = space.get_selection("AP")
            ml_sel = space.get_selection("ML")
        else:
            ap_sel, ml_sel = None, None

        if (ap_sel is not None) and (ml_sel is not None):
            marker = hv.Points([(float(ml_sel), float(ap_sel))], kdims=["ML", "AP"]).opts(
                color="yellow",
                marker="x",
                size=15,
                line_width=3,
            )
            img = img * marker

        tap = self._add_stream(Tap(source=img))

        def _on_tap(x, y):
            if x is None or y is None:
                return
            if space is None:
                return
            ml_i = int(np.clip(int(np.round(float(x))), 0, max(n_ml - 1, 0)))
            ap_i = int(np.clip(int(np.round(float(y))), 0, max(n_ap - 1, 0)))
            space.set_selection("ML", ml_i)
            space.set_selection("AP", ap_i)

        # Use HoloViews' subscriber API (avoids Param watcher signature pitfalls).
        tap.add_subscriber(_on_tap)
        return img

    def _update_spatial_marker(self, _dim: str, _value) -> None:
        self._update()

    def panel(self):
        return self._container

    def get_config(self) -> dict:
        return {
            "selected_time_source": self.selected_time_source,
            "independent_time": self._get_current_time(),
            "colormap": self.colormap,
            "symmetric_limits": self.symmetric_limits,
        }

    def dispose(self) -> None:
        try:
            space = getattr(self.state, "spatial_space", None)
            if space is not None and self._space_watch_handle is not None:
                space.unwatch(self._space_watch_handle)
        except Exception:  # noqa: BLE001
            pass
        super().dispose()
