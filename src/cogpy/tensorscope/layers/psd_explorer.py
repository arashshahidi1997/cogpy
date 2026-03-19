"""
PSD Explorer layers for TensorScope.

Design
------
PSD Explorer is an *add-on* to the existing TensorScope app:
- It reuses shared state/controllers (time_hair, spatial_space, processing).
- It does not re-implement basic views (timeseries/spatial map).
- It adds PSD-specific controls (window/FFT/method/frequency) and views.
"""

from __future__ import annotations

import numpy as np
import panel as pn

from cogpy.tensorscope.data.alignment import find_nearest_time_index

from .base import TensorLayer
from .psd_settings import PSDSettings, _ensure_psd_settings

class PSDExplorerLayer(TensorLayer):
    """PSD views (heatmap + average curve + spatial PSD map)."""

    def __init__(self, state):
        super().__init__(state)
        self.layer_id = "psd_explorer"
        self.title = "PSD Explorer"
        self.settings = _ensure_psd_settings(state)

    def panel(self) -> pn.viewable.Viewable:
        if self._panel is not None:
            return self._panel

        import holoviews as hv

        hv.extension("bokeh")

        def _apply_logf_axis(plot, _element) -> None:
            """
            Ensure the Bokeh y-scale updates when toggling log-frequency.

            HoloViews' `logy=` option may not reinitialize the axis scale on
            DynamicMap updates (axis type can get "stuck"). This hook forces
            the scale to match current settings on every render/update.
            """
            try:
                from bokeh.models import LinearScale, LogScale
            except Exception:  # noqa: BLE001
                return

            fig = getattr(plot, "state", None)
            if fig is None:
                return

            want_log = bool(getattr(self.settings, "freq_log", False))
            try:
                fig.y_scale = LogScale() if want_log else LinearScale()
            except Exception:  # noqa: BLE001
                pass

        # Streams.
        FreqStream = hv.streams.Stream.define("Freq", freq=float(self.settings.freq))
        freq_stream = self._add_stream(FreqStream())

        RefreshStream = hv.streams.Stream.define("Refresh", tick=int(0))
        refresh_stream = self._add_stream(RefreshStream())

        t0 = getattr(getattr(self.state, "time_hair", None), "t", None)
        if t0 is None:
            t0 = getattr(self.state, "selected_time", 0.0)
        TimeStream = hv.streams.Stream.define("TimeSel", time=float(t0 or 0.0))
        time_stream = self._add_stream(TimeStream())

        hair = getattr(self.state, "time_hair", None)
        if hair is not None and hasattr(hair, "param") and "t" in getattr(hair, "param", {}):
            self._watch(hair, lambda e: time_stream.event(time=float(e.new)), "t")

        try:
            spatial_stream = self._add_stream(self.state.spatial_space.create_stream())
        except Exception:  # noqa: BLE001
            SpatialStream = hv.streams.Stream.define("SpatialSel", AP=0.0, ML=0.0)
            spatial_stream = self._add_stream(SpatialStream())

        # Settings -> streams / refresh.
        self._watch(self.settings, lambda e: freq_stream.event(freq=float(e.new)), "freq")

        def _bump_refresh(_e=None) -> None:
            refresh_stream.event(tick=int(refresh_stream.tick) + 1)

        self._watch(
            self.settings,
            lambda _e: _bump_refresh(),
            ["window_size", "method", "db", "nperseg", "noverlap", "bandwidth", "freq_min", "freq_max", "freq_log"],
        )

        chain = getattr(self.state, "processing", None)
        if chain is not None and hasattr(chain, "param"):
            try:
                names = [n for n in list(chain.param) if n not in {"name"}]
                if names:
                    self._watch(chain, lambda _e: _bump_refresh(), names)
            except Exception:  # noqa: BLE001
                pass

        sig = (
            self.state.signal_registry.get_active()
            if getattr(self.state, "signal_registry", None) is not None
            else None
        )
        if sig is None:
            self._panel = pn.pane.Alert("No active signal", alert_type="warning")
            return self._panel

        data = sig.data

        psd_cache: dict[str, object] = {"key": None, "psd": None}

        def _compute_psd(time: float, tick: int):
            key = (float(time), int(tick))
            if psd_cache["key"] == key and psd_cache["psd"] is not None:
                return psd_cache["psd"]

            from cogpy.spectral.specx import psdx

            half = float(self.settings.window_size) / 2.0
            chain2 = getattr(self.state, "processing", None)
            if chain2 is not None:
                win = chain2.get_window(float(time) - half, float(time) + half)
            else:
                win = data.sel(time=slice(float(time) - half, float(time) + half)).compute()

            psd = psdx(
                win,
                axis="time",
                method=str(self.settings.method),  # type: ignore[arg-type]
                nperseg=int(self.settings.nperseg),
                noverlap=int(self.settings.noverlap) if str(self.settings.method) == "welch" else None,
                bandwidth=float(self.settings.bandwidth),
                fmin=float(max(float(self.settings.freq_min), 1e-6)) if bool(self.settings.freq_log) else float(self.settings.freq_min),
                fmax=float(self.settings.freq_max),
            )

            if bool(self.settings.db):
                from cogpy.spectral.psd_utils import psd_to_db

                psd = psd_to_db(psd)

            psd_cache["key"] = key
            psd_cache["psd"] = psd
            return psd

        def _psd_as_channel_freq(psd):
            from cogpy.spectral.psd_utils import stack_spatial_dims

            psd = stack_spatial_dims(psd)
            if "channel" not in psd.dims:
                raise ValueError(f"PSD missing channel dim; dims={psd.dims}")
            return psd.transpose("channel", "freq")

        def _heatmap(time=None, tick=None):
            psd = _compute_psd(float(time), int(tick))
            psd_ch = _psd_as_channel_freq(psd)

            ch = np.arange(int(psd_ch.sizes["channel"]))
            f = np.asarray(psd_ch["freq"].values, dtype=float)
            z = np.asarray(psd_ch.values, dtype=float)

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
                ylim=(
                    float(max(float(self.settings.freq_min), 1e-6)) if bool(self.settings.freq_log) else float(self.settings.freq_min),
                    float(self.settings.freq_max),
                ),
                logy=bool(self.settings.freq_log),
                hooks=[_apply_logf_axis],
            )

        def _avg(time=None, freq=None, tick=None):
            psd = _compute_psd(float(time), int(tick))
            psd_ch = _psd_as_channel_freq(psd)

            mu = psd_ch.mean(dim="channel")
            sd = psd_ch.std(dim="channel")
            f = np.asarray(mu["freq"].values, dtype=float)
            y = np.asarray(mu.values, dtype=float)
            s = np.asarray(sd.values, dtype=float)

            curve = hv.Curve((y, f), kdims=["power"], vdims=["freq"]).opts(color="#2a6fdb", line_width=2)
            xs = np.concatenate([y - s, (y + s)[::-1]])
            ys = np.concatenate([f, f[::-1]])
            band = hv.Polygons([{"power": xs, "freq": ys}], kdims=["power", "freq"]).opts(
                fill_color="#2a6fdb",
                fill_alpha=0.2,
                line_width=0,
            )
            base = (band * curve).opts(
                width=280,
                height=420,
                tools=["hover"],
                xlabel="Power (dB)" if bool(self.settings.db) else "Power",
                ylabel="Frequency (Hz)",
                title="Average PSD",
                ylim=(
                    float(max(float(self.settings.freq_min), 1e-6)) if bool(self.settings.freq_log) else float(self.settings.freq_min),
                    float(self.settings.freq_max),
                ),
                framewise=True,
                logy=bool(self.settings.freq_log),
                hooks=[_apply_logf_axis],
            )
            f_sel = float(freq) if freq is not None else float(self.settings.freq)
            if bool(self.settings.freq_log):
                f_sel = float(max(f_sel, 1e-6))
            hline = hv.HLine(f_sel).opts(color="#ffcc00", alpha=0.9, line_width=2, line_dash="dashed")
            return (base * hline).opts(framewise=True, hooks=[_apply_logf_axis])

        def _spatial_psd(time=None, freq=None, AP=None, ML=None, tick=None):
            psd = _compute_psd(float(time), int(tick))
            psd_ch = _psd_as_channel_freq(psd)

            f_vals = np.asarray(psd_ch["freq"].values, dtype=float)
            f_sel = float(freq) if freq is not None else float(self.settings.freq)
            fi = int(find_nearest_time_index(f_sel, f_vals))
            f_actual = float(f_vals[fi])
            sl = psd_ch.isel(freq=fi)

            # Reconstruct grid from AP/ML coords if available (ProcessingChain output uses these).
            if ("AP" not in sl.coords) or ("ML" not in sl.coords):
                return hv.Div("<b>PSD missing AP/ML coords for spatial map</b>")

            ap_vals = np.asarray(sl.coords["AP"].values)
            ml_vals = np.asarray(sl.coords["ML"].values)
            try:
                grid0 = self.state.data_registry.get("grid_lfp").data  # type: ignore[union-attr]
                ap_u = np.asarray(grid0.coords["AP"].values)
                ml_u = np.asarray(grid0.coords["ML"].values)
            except Exception:  # noqa: BLE001
                ap_u = np.unique(ap_vals)
                ml_u = np.unique(ml_vals)

            ap_to_i = {v: i for i, v in enumerate(ap_u)}
            ml_to_i = {v: i for i, v in enumerate(ml_u)}
            z = np.full((len(ap_u), len(ml_u)), np.nan, dtype=float)
            vals = np.asarray(sl.values, dtype=float)
            for ch_i in range(vals.size):
                ai = ap_to_i.get(ap_vals[ch_i])
                mi = ml_to_i.get(ml_vals[ch_i])
                if ai is not None and mi is not None:
                    z[int(ai), int(mi)] = float(vals[ch_i])

            ap_i = np.arange(len(ap_u), dtype=int)
            ml_i = np.arange(len(ml_u), dtype=int)

            img = hv.Image((ml_i, ap_i, z), kdims=["ML", "AP"], vdims=["power"]).opts(
                width=400,
                height=400,
                cmap="hot",
                colorbar=True,
                xlabel="ML (index)",
                ylabel="AP (index)",
                title=f"Spatial power @ {f_actual:.1f} Hz",
                tools=["hover"],
                aspect="equal",
                data_aspect=1,
                invert_yaxis=True,
            )
            return img

        heatmap_view = hv.DynamicMap(_heatmap, streams=[time_stream, refresh_stream])
        avg_view = hv.DynamicMap(_avg, streams=[time_stream, freq_stream, refresh_stream])
        spatial_view = hv.DynamicMap(_spatial_psd, streams=[time_stream, freq_stream, spatial_stream, refresh_stream])

        heatmap_card = pn.Card(
            pn.pane.HoloViews(heatmap_view, sizing_mode="fixed", width=520, height=420),
            title="PSD Heatmap (freq × channel)",
            sizing_mode="fixed",
            min_width=560,
            min_height=480,
        )
        avg_card = pn.Card(
            pn.pane.HoloViews(avg_view, sizing_mode="fixed", width=280, height=420),
            title="Average PSD (power × freq)",
            sizing_mode="fixed",
            min_width=320,
            min_height=480,
        )
        spatial_card = pn.Card(
            pn.pane.HoloViews(spatial_view, sizing_mode="fixed", width=400, height=400),
            title="Spatial PSD @ freq",
            sizing_mode="fixed",
            min_width=440,
            min_height=480,
        )

        quick_db = pn.widgets.Checkbox.from_param(self.settings.param.db, name="dB", width=60)
        quick_logf = pn.widgets.Checkbox.from_param(self.settings.param.freq_log, name="log f", width=70)
        quick_freq = pn.widgets.FloatSlider.from_param(
            self.settings.param.freq,
            name="Freq (Hz)",
            width=260,
            step=0.5,
        )

        header = pn.Row(
            pn.pane.Markdown("## PSD Explorer"),
            pn.Spacer(),
            quick_db,
            quick_logf,
            quick_freq,
            sizing_mode="stretch_width",
        )

        self._panel = pn.Column(
            header,
            pn.Row(heatmap_card, avg_card, spatial_card, sizing_mode="fixed"),
            sizing_mode="fixed",
        )
        return self._panel
