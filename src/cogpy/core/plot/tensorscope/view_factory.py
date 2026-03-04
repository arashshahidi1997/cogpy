"""
ViewFactory: convert ViewSpec to HoloViews objects.

This is the v2.2 declarative layer that sits on top of v2.1 controllers:
- `TensorScopeState.time_hair` for cursor time
- `TensorScopeState.spatial_space` for linked AP/ML selection
- `TensorScopeState.signal_registry` for signal selection
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from cogpy.core.plot.tensorscope.data.alignment import find_nearest_time_index

from .view_spec import ViewSpec

__all__ = ["ViewFactory", "infer_view_type"]


def infer_view_type(kdims: list[str]) -> str:
    """
    Infer a HoloViews element type based on the display dims.

    v2.2.0 uses minimal rules:
    - 2 dims -> Image
    - 1 dim  -> Curve
    - else   -> Generic (placeholder)
    """
    if len(kdims) == 2:
        return "Image"
    if len(kdims) == 1:
        return "Curve"
    return "Generic"


class ViewFactory:
    """Create HoloViews objects from `ViewSpec`."""

    @staticmethod
    def create(spec: ViewSpec, state):
        import holoviews as hv

        hv.extension("bokeh")

        reg = getattr(state, "signal_registry", None)
        sig = reg.get(spec.signal_id) if (reg is not None and spec.signal_id) else (reg.get_active() if reg else None)
        if sig is None:
            return hv.Div(f"<b>No signal found</b><br>signal_id={spec.signal_id!r}")

        view_type = infer_view_type(spec.kdims) if spec.view_type == "auto" else str(spec.view_type)

        kdims_set = {str(d) for d in spec.kdims}
        if view_type == "Image" and kdims_set == {"AP", "ML"}:
            return ViewFactory._create_spatial_image(spec, sig.data, state)
        if view_type == "Curve" and kdims_set == {"time"}:
            return ViewFactory._create_temporal_curve(spec, sig.data, state)

        return hv.Div(
            "<b>Generic ViewFactory fallback</b><br>"
            f"view_type={view_type!r}<br>kdims={spec.kdims!r}<br>controls={spec.controls!r}"
        )

    @staticmethod
    def _create_time_stream(state):
        import holoviews as hv

        time0 = None
        hair = getattr(state, "time_hair", None)
        if hair is not None:
            time0 = getattr(hair, "t", None)
        if time0 is None:
            time0 = getattr(state, "selected_time", None)

        stream = hv.streams.Stream.define("Time", time=float(time0) if time0 is not None else None)()

        if hair is not None and hasattr(hair, "param") and "t" in getattr(hair, "param", {}):
            hair.param.watch(lambda e: stream.event(time=float(e.new) if e.new is not None else None), "t")
        return stream

    @staticmethod
    def _create_spatial_stream(state, data):
        import holoviews as hv

        space = getattr(state, "spatial_space", None)

        n_ap = int(data.sizes.get("AP", 0))
        n_ml = int(data.sizes.get("ML", 0))

        ap0 = None
        ml0 = None
        if space is not None:
            ap0 = space.get_selection("AP")
            ml0 = space.get_selection("ML")

        if ap0 is None:
            ap0 = n_ap // 2 if n_ap > 0 else 0
        if ml0 is None:
            ml0 = n_ml // 2 if n_ml > 0 else 0

        stream = hv.streams.Stream.define("Spatial", AP=int(ap0), ML=int(ml0))()

        if space is not None:

            def _on_space(dim: str, value) -> None:
                if dim == "AP":
                    stream.event(AP=int(value))
                elif dim == "ML":
                    stream.event(ML=int(value))

            space.watch(_on_space)

        return stream

    @staticmethod
    def _apply_operation(values: np.ndarray, operation: Callable | None) -> np.ndarray:
        if operation is None:
            return values
        try:
            out = operation(values)
            return np.asarray(out)
        except Exception:  # noqa: BLE001
            return values

    @staticmethod
    def _create_spatial_image(spec: ViewSpec, data, state):
        import holoviews as hv

        if "time" not in data.dims:
            return hv.Div("<b>Signal has no 'time' dimension</b>")
        if not (("AP" in data.dims) and ("ML" in data.dims)):
            return hv.Div("<b>Signal is not grid-shaped (AP×ML)</b>")

        time_stream = ViewFactory._create_time_stream(state) if "time" in spec.controls else None

        tap_stream = hv.streams.Tap(x=None, y=None)

        n_ap = int(data.sizes["AP"])
        n_ml = int(data.sizes["ML"])
        ap_idx = np.arange(n_ap)
        ml_idx = np.arange(n_ml)

        space = getattr(state, "spatial_space", None)

        def _render(time=None, x=None, y=None):
            # Click -> update linked selection (indices).
            if (space is not None) and (x is not None) and (y is not None):
                ml_i = int(np.clip(int(np.round(float(x))), 0, max(n_ml - 1, 0)))
                ap_i = int(np.clip(int(np.round(float(y))), 0, max(n_ap - 1, 0)))
                try:
                    space.set_selection("ML", ml_i)
                    space.set_selection("AP", ap_i)
                except Exception:  # noqa: BLE001
                    pass

            fixed_t = None
            try:
                fixed_t = spec.fixed_values.get("time") if getattr(spec, "fixed_values", None) else None
            except Exception:  # noqa: BLE001
                fixed_t = None

            t = float(time) if time is not None else (float(fixed_t) if fixed_t is not None else None)
            if t is None:
                t = getattr(getattr(state, "time_hair", None), "t", None)
            if t is None:
                t = float(getattr(state, "selected_time", 0.0) or 0.0)

            t_vals = np.asarray(data["time"].values, dtype=float)
            time_idx = find_nearest_time_index(float(t), t_vals)
            actual_t = float(t_vals[int(time_idx)])

            frame = data.isel(time=int(time_idx)).transpose("AP", "ML")
            vals = np.asarray(frame.values, dtype=float)
            vals = ViewFactory._apply_operation(vals, spec.operation)

            if spec.clim is not None:
                vmin, vmax = float(spec.clim[0]), float(spec.clim[1])
            elif spec.symmetric_clim:
                vmax = float(np.nanmax(np.abs(vals))) if vals.size else 1.0
                vmin = -vmax
            else:
                vmin = float(np.nanmin(vals)) if vals.size else 0.0
                vmax = float(np.nanmax(vals)) if vals.size else 1.0

            title = spec.title or f"LFP @ t={actual_t:.3f}s"

            img = hv.Image(
                (ml_idx, ap_idx, vals),
                kdims=["ML", "AP"],
                vdims=["LFP (a.u.)"],
            ).opts(
                cmap=spec.colormap,
                clim=(vmin, vmax),
                colorbar=True,
                tools=["hover", "tap"],
                xlabel="ML (index)",
                ylabel="AP (index)",
                title=title,
                width=420,
                height=420,
                aspect="equal",
            )

            ap_sel = space.get_selection("AP") if space is not None else None
            ml_sel = space.get_selection("ML") if space is not None else None
            if (ap_sel is not None) and (ml_sel is not None):
                marker = hv.Points([(float(ml_sel), float(ap_sel))], kdims=["ML", "AP"]).opts(
                    color="yellow",
                    marker="x",
                    size=15,
                    line_width=3,
                )
                return img * marker

            return img

        streams = [tap_stream]
        if time_stream is not None:
            streams.insert(0, time_stream)
        return hv.DynamicMap(_render, streams=streams)

    @staticmethod
    def _create_temporal_curve(spec: ViewSpec, data, state):
        import holoviews as hv

        if "time" not in data.dims:
            return hv.Div("<b>Signal has no 'time' dimension</b>")
        if not (("AP" in data.dims) and ("ML" in data.dims)):
            return hv.Div("<b>Signal is not grid-shaped (AP×ML)</b>")

        spatial_stream = ViewFactory._create_spatial_stream(state, data)

        n_ap = int(data.sizes["AP"])
        n_ml = int(data.sizes["ML"])

        def _render(AP=0, ML=0):
            ap_i = int(np.clip(int(AP), 0, max(n_ap - 1, 0)))
            ml_i = int(np.clip(int(ML), 0, max(n_ml - 1, 0)))

            trace = data.isel(AP=ap_i, ML=ml_i)
            t_vals = np.asarray(trace["time"].values, dtype=float)
            y_vals = np.asarray(trace.values, dtype=float)
            y_vals = ViewFactory._apply_operation(y_vals, spec.operation)

            title = spec.title or f"Trace @ (AP={ap_i}, ML={ml_i})"
            return hv.Curve((t_vals, y_vals), kdims=["time"], vdims=["LFP (a.u.)"]).opts(
                tools=["hover"],
                xlabel="Time (s)",
                ylabel="LFP (a.u.)",
                title=title,
                width=600,
                height=300,
            )

        return hv.DynamicMap(_render, streams=[spatial_stream])
