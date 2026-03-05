"""
PSD Explorer module (v2.8.0).

HoloViews-native layout for exploring PSD in a window around the current time.
Links:
- time: via `state.time_hair` (cursor) or `state.selected_time`
- spatial: via `state.spatial_space` (AP/ML selection)
- frequency: local stream controlled by tapping the average PSD curve
"""

from __future__ import annotations

import numpy as np

from cogpy.core.plot.tensorscope.data.alignment import find_nearest_time_index

from ..view_spec import ViewSpec
from .base import ViewPresetModule

__all__ = ["PSDExplorerModule", "MODULE", "create_psd_explorer_module"]


def _get_time_center(state) -> float:
    t = getattr(getattr(state, "time_hair", None), "t", None)
    if t is None:
        t = getattr(state, "selected_time", None)
    return float(t) if t is not None else 0.0


def create_psd_explorer_module(
    *,
    filter_type: str = "bandpass",
    filter_low: float = 1.0,
    filter_high: float = 100.0,
    signal_mode: str = "raw",
    window_size: float = 1.0,
    nperseg: int = 256,
    method: str = "welch",
) -> ViewPresetModule:
    """
    Create a PSD explorer module preset.

    Notes
    -----
    The v2.2 module system returns HoloViews objects. This module therefore
    uses tap/stream interactions instead of Panel widgets.
    """

    filter_type = str(filter_type).lower()
    signal_mode = str(signal_mode).lower()
    method = str(method).lower()

    def _activate(state):
        import holoviews as hv

        hv.extension("bokeh")

        # Frequency selection stream.
        FreqStream = hv.streams.Stream.define("Freq", freq=float(40.0))
        freq_stream = FreqStream()

        # Time stream from state.
        TimeStream = hv.streams.Stream.define("TimeSel", time=float(_get_time_center(state)))
        time_stream = TimeStream()

        hair = getattr(state, "time_hair", None)
        if hair is not None and hasattr(hair, "param") and "t" in getattr(hair, "param", {}):
            hair.param.watch(lambda e: time_stream.event(time=float(e.new)), "t")

        # Spatial stream from CoordinateSpace.
        try:
            spatial_stream = state.spatial_space.create_stream()
        except Exception:  # noqa: BLE001
            SpatialStream = hv.streams.Stream.define("SpatialSel", AP=0.0, ML=0.0)
            spatial_stream = SpatialStream()

        # Prepare raw/filtered views of the active signal.
        sig = state.signal_registry.get_active() if getattr(state, "signal_registry", None) is not None else None
        if sig is None:
            return hv.Div("<b>No active signal</b>")
        data = sig.data

        def _apply_filter(x):
            if filter_type == "none":
                return x
            from cogpy.core.preprocess.filtx import bandpassx, highpassx, lowpassx

            if filter_type == "bandpass":
                return bandpassx(x, float(filter_low), float(filter_high), 4, axis="time")
            if filter_type == "highpass":
                return highpassx(x, float(filter_low), 4, axis="time")
            if filter_type == "lowpass":
                return lowpassx(x, float(filter_high), 4, axis="time")
            return x

        raw = data
        filt = _apply_filter(data)

        def _choose(mode: str):
            if mode == "filtered":
                return filt
            return raw

        # -------- Trace stack (window around time) --------
        def _trace_stack(time=None):
            import holoviews as hv

            t_center = float(time) if time is not None else _get_time_center(state)
            half = float(window_size) / 2.0
            win = _choose(signal_mode).sel(time=slice(t_center - half, t_center + half))

            from cogpy.core.spectral.psd_utils import stack_spatial_dims

            stacked = stack_spatial_dims(win)
            if "channel" not in stacked.dims:
                return hv.Div("<b>No channels</b>")

            # Use at most 32 traces for responsiveness.
            n_ch = int(stacked.sizes["channel"])
            n_show = min(n_ch, 32)

            # Convert to (time, channel) for iteration.
            st = stacked.transpose("time", "channel")
            t_vals = np.asarray(st["time"].values, dtype=float)
            vals = np.asarray(st.values, dtype=float)

            # Normalize offsets.
            if vals.size == 0:
                return hv.Overlay([])
            scale = float(np.nanstd(vals)) if np.isfinite(np.nanstd(vals)) else 1.0
            scale = scale if scale > 0 else 1.0
            spacing = 4.0 * scale

            curves = []
            for i in range(n_show):
                y = vals[:, i] + i * spacing
                curves.append(hv.Curve((t_vals, y), kdims=["time"], vdims=["amp"]).opts(color="#2a6fdb", alpha=0.7))

            vline = hv.VLine(t_center).opts(color="#ff0000", alpha=0.7, line_dash="dashed", line_width=2)
            return (hv.Overlay(curves) * vline).opts(
                width=520,
                height=420,
                xlabel="Time (s)",
                ylabel="Channels (stacked)",
                title=f"Traces ({signal_mode})",
                tools=["hover"],
            )

        trace_view = hv.DynamicMap(_trace_stack, streams=[time_stream])

        # -------- PSD computation (returns psd with freq dim) --------
        def _psd(time=None):
            t_center = float(time) if time is not None else _get_time_center(state)
            from cogpy.core.spectral.psd_utils import compute_psd_window

            return compute_psd_window(
                _choose(signal_mode),
                t_center=t_center,
                window_size=float(window_size),
                nperseg=int(nperseg),
                method=str(method),
                axis="time",
            )

        # -------- PSD heatmap: freq × channel --------
        def _psd_heatmap(time=None):
            import holoviews as hv

            psd = _psd(time=time)
            from cogpy.core.spectral.psd_utils import stack_spatial_dims

            psd_ch = stack_spatial_dims(psd)
            if "channel" not in psd_ch.dims:
                return hv.Div("<b>PSD has no channel/AP/ML dims</b>")

            psd_ch = psd_ch.transpose("channel", "freq")
            z = np.asarray(psd_ch.values, dtype=float)
            ch = np.arange(int(psd_ch.sizes["channel"]))
            f = np.asarray(psd_ch["freq"].values, dtype=float)

            # Use QuadMesh for non-uniform freq bins.
            el = hv.QuadMesh((ch, f, z.T), kdims=["channel", "freq"], vdims=["power"])
            return el.opts(
                width=520,
                height=420,
                cmap="viridis",
                colorbar=True,
                xlabel="Channel",
                ylabel="Frequency (Hz)",
                title="PSD heatmap",
                tools=["hover"],
            )

        heatmap_view = hv.DynamicMap(_psd_heatmap, streams=[time_stream])

        # -------- Average PSD: mean ± std over channels --------
        def _psd_avg(time=None, freq=None):
            import holoviews as hv

            psd = _psd(time=time)
            # Average over spatial dims.
            dims = [d for d in psd.dims if d != "freq"]
            if dims:
                mu = psd.mean(dim=dims)
                sd = psd.std(dim=dims)
            else:
                mu = psd
                sd = psd * 0.0

            f = np.asarray(mu["freq"].values, dtype=float)
            y = np.asarray(mu.values, dtype=float)
            s = np.asarray(sd.values, dtype=float)
            curve = hv.Curve((f, y), kdims=["freq"], vdims=["power"]).opts(
                width=520,
                height=260,
                tools=["hover", "tap"],
                xlabel="Frequency (Hz)",
                ylabel="Power",
                title="Average PSD",
                color="#2a6fdb",
                line_width=2,
            )
            band = hv.Area((f, y - s, y + s), kdims=["freq"], vdims=["lower", "upper"]).opts(
                color="#2a6fdb",
                alpha=0.2,
                line_width=0,
            )

            # Tap on curve to set freq selection.
            tap = hv.streams.Tap(source=curve)

            def _on_tap(e):
                x = getattr(e, "x", None)
                if x is None:
                    return
                freq_stream.event(freq=float(x))

            tap.param.watch(_on_tap, "x")

            # Show selected frequency line.
            f_sel = float(freq) if freq is not None else float(freq_stream.freq)
            hline = hv.VLine(f_sel).opts(color="#ffcc00", alpha=0.9, line_width=2)
            return band * curve * hline

        avg_view = hv.DynamicMap(_psd_avg, streams=[time_stream, freq_stream])

        # -------- Spatial PSD map at selected frequency --------
        def _spatial_map(time=None, freq=None, AP=None, ML=None):
            import holoviews as hv

            psd = _psd(time=time)
            f_sel = float(freq) if freq is not None else float(getattr(freq_stream, "freq", 40.0))

            if ("AP" not in psd.dims) or ("ML" not in psd.dims):
                return hv.Div("<b>No spatial dims (AP/ML) in PSD</b>")

            # Select nearest frequency.
            f_vals = np.asarray(psd["freq"].values, dtype=float)
            fi = find_nearest_time_index(f_sel, f_vals)
            f_actual = float(f_vals[int(fi)])
            sl = psd.isel(freq=int(fi)).transpose("AP", "ML")
            z = np.asarray(sl.values, dtype=float)
            ap = np.arange(int(sl.sizes["AP"]))
            ml = np.arange(int(sl.sizes["ML"]))

            img = hv.Image((ml, ap, z), kdims=["ML", "AP"], vdims=["power"]).opts(
                width=420,
                height=420,
                cmap="hot",
                colorbar=True,
                xlabel="ML (index)",
                ylabel="AP (index)",
                title=f"Spatial power @ {f_actual:.1f} Hz",
                tools=["hover"],
                aspect="equal",
            )

            # Current spatial selection marker.
            ap_sel = AP if AP is not None else getattr(state.spatial_space, "get_selection", lambda _d: None)("AP")
            ml_sel = ML if ML is not None else getattr(state.spatial_space, "get_selection", lambda _d: None)("ML")
            if ap_sel is not None and ml_sel is not None:
                marker = hv.Points([(float(ml_sel), float(ap_sel))], kdims=["ML", "AP"]).opts(
                    color="#00ffff", marker="x", size=14, line_width=3
                )
                return img * marker
            return img

        spatial_view = hv.DynamicMap(_spatial_map, streams=[time_stream, freq_stream, spatial_stream])

        header = hv.Div(
            "<b>PSD Explorer</b><br>"
            f"filter={filter_type}({filter_low:g},{filter_high:g}) window={window_size:g}s nperseg={int(nperseg)}"
        )

        top = hv.Layout([header, avg_view]).cols(1)
        grid = hv.Layout([trace_view, heatmap_view, spatial_view]).cols(2)
        return (top + grid).cols(1)

    return ViewPresetModule(
        name="psd_explorer",
        description="PSD explorer (traces + PSD heatmap + average + spatial map)",
        specs=[],
        layout="custom",
        activate_fn=_activate,
    )


class PSDExplorerModule:
    """Convenience wrapper mirroring the module API (explicit construction)."""

    def __init__(self, **kwargs) -> None:
        self._kwargs = dict(kwargs)

    def as_preset(self) -> ViewPresetModule:
        return create_psd_explorer_module(**self._kwargs)

    def activate(self, state):
        return self.as_preset().activate(state)


MODULE = create_psd_explorer_module()
