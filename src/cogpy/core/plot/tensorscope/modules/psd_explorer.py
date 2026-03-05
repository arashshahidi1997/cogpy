"""
PSD Explorer module (v2.8.0).

Panel layout for exploring PSD in a window around the current time.
Links:
- time: via `state.time_hair` (cursor) or `state.selected_time`
- spatial: via `state.spatial_space` (AP/ML selection)
- frequency: selected via a Panel control
"""

from __future__ import annotations

import numpy as np

from cogpy.core.plot.tensorscope.data.alignment import find_nearest_time_index

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
        import panel as pn

        hv.extension("bokeh")

        # Frequency selection stream.
        FreqStream = hv.streams.Stream.define("Freq", freq=float(40.0))
        freq_stream = FreqStream()

        # Refresh stream for widget-driven recomputation.
        RefreshStream = hv.streams.Stream.define("Refresh", tick=int(0))
        refresh_stream = RefreshStream()

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

        # ---------------- Controls ----------------
        signal_mode_w = pn.widgets.Select(
            name="Signal",
            options=["raw", "filtered", "comparison"],
            value=signal_mode if signal_mode in {"raw", "filtered", "comparison"} else "raw",
            width=220,
        )
        filter_type_w = pn.widgets.Select(
            name="Filter",
            options=["none", "bandpass", "highpass", "lowpass", "notch", "multi_notch", "spatial_median"],
            value=filter_type
            if filter_type in {"none", "bandpass", "highpass", "lowpass", "notch", "multi_notch", "spatial_median"}
            else "bandpass",
            width=220,
        )
        filter_low_w = pn.widgets.FloatInput(name="Low (Hz)", value=float(filter_low), start=0.0, width=220)
        filter_high_w = pn.widgets.FloatInput(name="High (Hz)", value=float(filter_high), start=0.0, width=220)
        notch_freqs_w = pn.widgets.TextInput(name="Notch freqs (Hz)", value="50,100,150", width=220, visible=False)

        time_min = float(np.asarray(data["time"].values, dtype=float).min())
        time_max = float(np.asarray(data["time"].values, dtype=float).max())
        t_default = _get_time_center(state)
        if not (time_min <= t_default <= time_max):
            t_default = float(np.asarray(data["time"].values, dtype=float).mean())

        # Infer a reasonable slider step (fallback 0.01s).
        t_vals = np.asarray(data["time"].values, dtype=float)
        dt = float(np.nanmedian(np.diff(t_vals))) if t_vals.size > 2 else 0.01
        step = 0.01 if not np.isfinite(dt) else max(dt, 0.001)
        time_slider = pn.widgets.FloatSlider(
            name="Time (s)",
            start=time_min,
            end=time_max,
            value=float(t_default),
            step=float(step),
            width=220,
        )

        window_size_w = pn.widgets.FloatInput(name="Window (s)", value=float(window_size), start=0.01, width=220)
        nperseg_w = pn.widgets.IntInput(name="nperseg", value=int(nperseg), start=16, step=16, width=220)
        method_w = pn.widgets.Select(
            name="PSD method",
            options=["welch", "multitaper"],
            value=method if method in {"welch", "multitaper"} else "welch",
            width=220,
        )
        db_w = pn.widgets.Checkbox(name="dB scale", value=False, width=220)
        freq_w = pn.widgets.FloatInput(name="Freq (Hz)", value=float(freq_stream.freq), start=0.0, width=220)

        chain_pane = pn.pane.Markdown(sizing_mode="stretch_width")

        filter_cache: dict[str, object] = {"key": None, "data": None}

        def _update_filter_visibility(event=None):
            ft = str(filter_type_w.value)
            notch_freqs_w.visible = ft in {"notch", "multi_notch"}

            if ft == "spatial_median":
                filter_low_w.visible = False
                filter_high_w.visible = False
                return

            if ft == "notch":
                filter_low_w.name = "Notch freq (Hz)"
                filter_low_w.visible = True
                filter_high_w.visible = False
                return

            if ft == "bandpass":
                filter_low_w.name = "Low (Hz)"
                filter_low_w.visible = True
                filter_high_w.name = "High (Hz)"
                filter_high_w.visible = True
                return

            if ft == "highpass":
                filter_low_w.name = "Cutoff (Hz)"
                filter_low_w.visible = True
                filter_high_w.visible = False
                return

            if ft == "lowpass":
                filter_high_w.name = "Cutoff (Hz)"
                filter_low_w.visible = False
                filter_high_w.visible = True
                return

            # none / multi_notch: hide low/high (multi uses notch_freqs)
            filter_low_w.visible = False
            filter_high_w.visible = False

        def _apply_filter(x):
            ft = str(filter_type_w.value)
            if ft == "none":
                return x
            from cogpy.core.preprocess.filtx import bandpassx, highpassx, lowpassx, median_spatialx, notchx, notchesx

            low = float(filter_low_w.value)
            high = float(filter_high_w.value)
            if ft == "bandpass":
                return bandpassx(x, low, high, 4, axis="time")
            if ft == "highpass":
                return highpassx(x, low, 4, axis="time")
            if ft == "lowpass":
                return lowpassx(x, high, 4, axis="time")
            if ft == "notch":
                return notchx(x, w0=low, Q=30.0, time_dim="time")
            if ft == "multi_notch":
                freqs_str = str(notch_freqs_w.value or "")
                freqs = []
                for part in freqs_str.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    freqs.append(float(part))
                return notchesx(x, freqs=freqs, Q=30.0, time_dim="time") if freqs else x
            if ft == "spatial_median":
                if ("AP" in x.dims) and ("ML" in x.dims):
                    return median_spatialx(x, size=3, ap_dim="AP", ml_dim="ML")
                return x
            return x

        def _filtered():
            key = (
                str(filter_type_w.value),
                float(filter_low_w.value),
                float(filter_high_w.value),
                str(notch_freqs_w.value or ""),
            )
            if filter_cache["key"] == key and filter_cache["data"] is not None:
                return filter_cache["data"]
            out = _apply_filter(data)
            filter_cache["key"] = key
            filter_cache["data"] = out
            return out

        def _choose():
            mode = str(signal_mode_w.value)
            if mode == "filtered":
                return _filtered()
            if mode == "comparison":
                return data, _filtered()
            return data

        def _update_chain():
            mode = str(signal_mode_w.value)
            ft = str(filter_type_w.value)
            chain = f"**Processing chain**  \n`{mode}`"
            if ft != "none":
                if ft == "bandpass":
                    chain += f" → `bandpass({float(filter_low_w.value):g},{float(filter_high_w.value):g})`"
                elif ft == "highpass":
                    chain += f" → `highpass({float(filter_low_w.value):g})`"
                elif ft == "lowpass":
                    chain += f" → `lowpass({float(filter_high_w.value):g})`"
                elif ft == "notch":
                    chain += f" → `notch({float(filter_low_w.value):g})`"
                elif ft == "multi_notch":
                    chain += f" → `multi_notch({str(notch_freqs_w.value)})`"
                elif ft == "spatial_median":
                    chain += " → `spatial_median(size=3)`"
                else:
                    chain += f" → `{ft}`"

            chain += f" → `PSD({str(method_w.value)}, nperseg={int(nperseg_w.value)})`"
            if bool(db_w.value):
                chain += " → `dB`"
            chain += f"\n\n**Selected freq:** `{float(freq_w.value):.2f} Hz`"
            chain_pane.object = chain

        def _bump_refresh(*_):
            # Any control change should trigger recomputation. Clear cached filter.
            filter_cache["key"] = None
            filter_cache["data"] = None
            refresh_stream.event(tick=int(refresh_stream.tick) + 1)
            _update_chain()

        for w in [
            time_slider,
            signal_mode_w,
            filter_type_w,
            filter_low_w,
            filter_high_w,
            notch_freqs_w,
            window_size_w,
            nperseg_w,
            method_w,
            db_w,
        ]:
            w.param.watch(_bump_refresh, "value")

        filter_type_w.param.watch(_update_filter_visibility, "value")
        _update_filter_visibility()

        def _on_freq_widget(event):
            try:
                freq_stream.event(freq=float(event.new))
            finally:
                _bump_refresh()

        freq_w.param.watch(_on_freq_widget, "value")

        def _on_time_slider(event):
            t_new = float(event.new)
            # Prefer driving the global cursor if present.
            hair2 = getattr(state, "time_hair", None)
            if hair2 is not None and hasattr(hair2, "param") and "t" in getattr(hair2, "param", {}):
                hair2.t = t_new
            else:
                time_stream.event(time=t_new)

        time_slider.param.watch(_on_time_slider, "value")
        _update_chain()

        # -------- Trace stack (window around time) --------
        def _trace_stack(time=None, tick=None):
            import holoviews as hv

            t_center = float(time) if time is not None else float(time_slider.value)
            half = float(window_size_w.value) / 2.0

            chosen = _choose()

            from cogpy.core.spectral.psd_utils import stack_spatial_dims

            def _stack_and_draw(x, color):
                win = x.sel(time=slice(t_center - half, t_center + half))
                stacked = stack_spatial_dims(win)
                if "channel" not in stacked.dims:
                    return []

                n_ch = int(stacked.sizes["channel"])
                n_show = min(n_ch, 32)

                st = stacked.transpose("time", "channel")
                t_vals = np.asarray(st["time"].values, dtype=float)
                vals = np.asarray(st.values, dtype=float)
                if vals.size == 0:
                    return []
                scale = float(np.nanstd(vals)) if np.isfinite(np.nanstd(vals)) else 1.0
                scale = scale if scale > 0 else 1.0
                spacing = float(4.0 * scale)
                curves = []
                for i in range(n_show):
                    y = vals[:, i] + i * spacing
                    curves.append(hv.Curve((t_vals, y), kdims=["time"], vdims=["amp"]).opts(color=color, alpha=0.7))
                return curves

            # Use at most 32 traces for responsiveness.
            curves = []
            if isinstance(chosen, tuple):
                curves.extend(_stack_and_draw(chosen[0], "#2a6fdb"))
                curves.extend(_stack_and_draw(chosen[1], "#db2a2a"))
                title = "Traces (raw vs filtered)"
            else:
                curves.extend(_stack_and_draw(chosen, "#2a6fdb"))
                title = f"Traces ({str(signal_mode_w.value)})"

            vline = hv.VLine(t_center).opts(color="#ff0000", alpha=0.7, line_dash="dashed", line_width=2)
            return (hv.Overlay(curves) * vline).opts(
                width=520,
                height=420,
                xlabel="Time (s)",
                ylabel="Channels (stacked)",
                title=title,
                tools=["hover"],
            )

        trace_view = hv.DynamicMap(_trace_stack, streams=[time_stream, refresh_stream])

        # -------- PSD computation (returns psd with freq dim) --------
        def _psd(time=None, *, use_filtered: bool = False):
            t_center = float(time) if time is not None else float(time_slider.value)
            from cogpy.core.spectral.psd_utils import compute_psd_window

            return compute_psd_window(
                _filtered() if use_filtered else data,
                t_center=t_center,
                window_size=float(window_size_w.value),
                nperseg=int(nperseg_w.value),
                method=str(method_w.value),
                axis="time",
            )

        # -------- PSD heatmap: freq × channel --------
        def _psd_heatmap(time=None, tick=None):
            import holoviews as hv

            # In comparison mode, show heatmap for filtered to highlight band-limited changes.
            use_filtered = str(signal_mode_w.value) in {"filtered", "comparison"} and str(filter_type_w.value) != "none"
            psd = _psd(time=time, use_filtered=use_filtered)
            if bool(db_w.value):
                from cogpy.core.spectral.psd_utils import psd_to_db

                psd = psd_to_db(psd)
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

        heatmap_view = hv.DynamicMap(_psd_heatmap, streams=[time_stream, refresh_stream])

        # -------- Average PSD: mean ± std over channels --------
        def _psd_avg(time=None, freq=None, tick=None):
            import holoviews as hv

            def _mean_std(psd):
                dims = [d for d in psd.dims if d != "freq"]
                if dims:
                    return psd.mean(dim=dims), psd.std(dim=dims)
                return psd, psd * 0.0

            if str(signal_mode_w.value) == "comparison":
                psd_raw = _psd(time=time, use_filtered=False)
                psd_f = _psd(time=time, use_filtered=True)
                if bool(db_w.value):
                    from cogpy.core.spectral.psd_utils import psd_to_db

                    psd_raw = psd_to_db(psd_raw)
                    psd_f = psd_to_db(psd_f)

                mu0, sd0 = _mean_std(psd_raw)
                mu1, sd1 = _mean_std(psd_f)
                f = np.asarray(mu0["freq"].values, dtype=float)
                y0 = np.asarray(mu0.values, dtype=float)
                s0 = np.asarray(sd0.values, dtype=float)
                y1 = np.asarray(mu1.values, dtype=float)
                s1 = np.asarray(sd1.values, dtype=float)

                curve0 = hv.Curve((f, y0), kdims=["freq"], vdims=["power"]).opts(color="#2a6fdb", line_width=2)
                band0 = hv.Area((f, y0 - s0, y0 + s0), kdims=["freq"], vdims=["lower", "upper"]).opts(
                    color="#2a6fdb", alpha=0.15, line_width=0
                )
                curve1 = hv.Curve((f, y1), kdims=["freq"], vdims=["power"]).opts(color="#db2a2a", line_width=2)
                band1 = hv.Area((f, y1 - s1, y1 + s1), kdims=["freq"], vdims=["lower", "upper"]).opts(
                    color="#db2a2a", alpha=0.15, line_width=0
                )
                base = (band0 * curve0 * band1 * curve1).opts(
                    width=520,
                    height=260,
                    tools=["hover"],
                    xlabel="Frequency (Hz)",
                    ylabel="Power (dB)" if bool(db_w.value) else "Power",
                    title="Average PSD (raw vs filtered)",
                )
            else:
                use_filtered = str(signal_mode_w.value) == "filtered" and str(filter_type_w.value) != "none"
                psd = _psd(time=time, use_filtered=use_filtered)
                if bool(db_w.value):
                    from cogpy.core.spectral.psd_utils import psd_to_db

                    psd = psd_to_db(psd)

                mu, sd = _mean_std(psd)
                f = np.asarray(mu["freq"].values, dtype=float)
                y = np.asarray(mu.values, dtype=float)
                s = np.asarray(sd.values, dtype=float)
                curve = hv.Curve((f, y), kdims=["freq"], vdims=["power"])
                band = hv.Area((f, y - s, y + s), kdims=["freq"], vdims=["lower", "upper"])
                base = (band.opts(color="#2a6fdb", alpha=0.2, line_width=0) * curve.opts(color="#2a6fdb", line_width=2)).opts(
                    width=520,
                    height=260,
                    tools=["hover"],
                    xlabel="Frequency (Hz)",
                    ylabel="Power (dB)" if bool(db_w.value) else "Power",
                    title="Average PSD",
                )

            f_sel = float(freq) if freq is not None else float(freq_stream.freq)
            hline = hv.VLine(f_sel).opts(color="#ffcc00", alpha=0.9, line_width=2)
            return base * hline

        avg_view = hv.DynamicMap(_psd_avg, streams=[time_stream, freq_stream, refresh_stream])

        # -------- Spatial PSD map at selected frequency --------
        def _spatial_map(time=None, freq=None, AP=None, ML=None, tick=None):
            import holoviews as hv

            use_filtered = str(signal_mode_w.value) in {"filtered", "comparison"} and str(filter_type_w.value) != "none"
            psd = _psd(time=time, use_filtered=use_filtered)
            if bool(db_w.value):
                from cogpy.core.spectral.psd_utils import psd_to_db

                psd = psd_to_db(psd)
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

        spatial_view = hv.DynamicMap(_spatial_map, streams=[time_stream, freq_stream, spatial_stream, refresh_stream])

        # -------- Spatial LFP at current time --------
        def _spatial_lfp(time=None, tick=None):
            import holoviews as hv

            t_center = float(time) if time is not None else float(time_slider.value)
            chosen = _choose()
            x = chosen[1] if isinstance(chosen, tuple) else chosen

            if ("AP" not in x.dims) or ("ML" not in x.dims):
                return hv.Div("<b>No spatial dims (AP/ML)</b>")

            t_vals2 = np.asarray(x["time"].values, dtype=float)
            ti = find_nearest_time_index(t_center, t_vals2)
            t_actual = float(t_vals2[int(ti)])

            sl = x.isel(time=int(ti)).transpose("AP", "ML")
            z = np.asarray(sl.values, dtype=float)
            ap = np.arange(int(sl.sizes["AP"]))
            ml = np.arange(int(sl.sizes["ML"]))

            img = hv.Image((ml, ap, z), kdims=["ML", "AP"], vdims=["amplitude"]).opts(
                width=420,
                height=420,
                cmap="RdBu_r",
                colorbar=True,
                xlabel="ML (index)",
                ylabel="AP (index)",
                title=f"LFP @ {t_actual:.3f} s",
                tools=["hover"],
                aspect="equal",
            )
            return img

        spatial_lfp_view = hv.DynamicMap(_spatial_lfp, streams=[time_stream, refresh_stream])

        def _sync_freq_to_stream(event):
            if event.new is None:
                return
            try:
                freq_w.value = float(event.new)
            except Exception:  # noqa: BLE001
                pass

        freq_stream.param.watch(_sync_freq_to_stream, "freq")

        controls = pn.Column(
            pn.pane.Markdown("## PSD Explorer"),
            chain_pane,
            time_slider,
            signal_mode_w,
            filter_type_w,
            filter_low_w,
            filter_high_w,
            notch_freqs_w,
            window_size_w,
            nperseg_w,
            method_w,
            db_w,
            freq_w,
            sizing_mode="stretch_height",
            width=260,
        )

        views = pn.Column(
            pn.Row(
                pn.pane.HoloViews(trace_view, width=520, height=420, sizing_mode="fixed"),
                pn.pane.HoloViews(heatmap_view, width=520, height=420, sizing_mode="fixed"),
                sizing_mode="fixed",
            ),
            pn.Row(
                pn.pane.HoloViews(avg_view, width=520, height=280, sizing_mode="fixed"),
                pn.pane.HoloViews(spatial_view, width=420, height=420, sizing_mode="fixed"),
                pn.pane.HoloViews(spatial_lfp_view, width=420, height=420, sizing_mode="fixed"),
                sizing_mode="fixed",
            ),
            sizing_mode="fixed",
        )

        return pn.Row(controls, views, sizing_mode="fixed")

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
