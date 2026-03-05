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

from dataclasses import dataclass

import numpy as np
import panel as pn
import param

from cogpy.core.plot.tensorscope.data.alignment import find_nearest_time_index

from .base import TensorLayer


class PSDSettings(param.Parameterized):
    """PSD parameters shared between the sidebar controls and PSD views."""

    window_size = param.Number(default=1.0, bounds=(0.01, 60.0), doc="Window size (s)")
    nperseg = param.Integer(default=256, bounds=(16, 8192), doc="FFT size (Welch)")
    method = param.ObjectSelector(default="welch", objects=["welch", "multitaper"])
    db = param.Boolean(default=False, doc="Display PSD in dB")
    freq_min = param.Number(default=0.0, bounds=(0.0, 500.0), doc="Min frequency (Hz)")
    freq_max = param.Number(default=150.0, bounds=(0.0, 500.0), doc="Max frequency (Hz)")
    freq = param.Number(default=40.0, bounds=(0.0, 500.0), doc="Selected frequency (Hz)")


def _ensure_psd_settings(state) -> PSDSettings:
    settings = getattr(state, "psd_settings", None)
    if isinstance(settings, PSDSettings):
        return settings

    settings = PSDSettings(name="psd_settings")
    try:
        setattr(state, "psd_settings", settings)
    except Exception:  # noqa: BLE001
        pass
    return settings


def _fs_from_signal(sig) -> float:
    try:
        fs = float(getattr(sig, "attrs", {}).get("fs", 1.0) or 1.0)
    except Exception:  # noqa: BLE001
        fs = 1.0
    return fs if np.isfinite(fs) and fs > 0 else 1.0


@dataclass(frozen=True, slots=True)
class _PsdCacheKey:
    time: float
    tick: int


class PSDSettingsLayer(TensorLayer):
    """Sidebar controls for PSD settings."""

    def __init__(self, state):
        super().__init__(state)
        self.layer_id = "psd_settings"
        self.title = "PSD Settings"
        self.settings = _ensure_psd_settings(state)

    def panel(self) -> pn.viewable.Viewable:
        if self._panel is not None:
            return self._panel

        sig = (
            self.state.signal_registry.get_active()
            if getattr(self.state, "signal_registry", None) is not None
            else None
        )
        fs = _fs_from_signal(getattr(sig, "data", None) if sig is not None else None)
        fmax = max(1.0, fs / 2.0)
        fmax_default = float(min(150.0, fmax))
        for pname in ("freq", "freq_min", "freq_max"):
            try:
                self.settings.param[pname].bounds = (0.0, float(fmax))
            except Exception:  # noqa: BLE001
                pass

        if float(self.settings.freq_max) > fmax_default:
            self.settings.freq_max = fmax_default
        if float(self.settings.freq) > float(self.settings.freq_max):
            self.settings.freq = float(self.settings.freq_max)

        window_size_w = pn.widgets.FloatInput.from_param(
            self.settings.param.window_size, name="Window (s)", width=220
        )
        nperseg_w = pn.widgets.IntInput.from_param(self.settings.param.nperseg, name="FFT", width=220)
        method_w = pn.widgets.Select.from_param(self.settings.param.method, name="Method", width=220)
        db_w = pn.widgets.Checkbox.from_param(self.settings.param.db, name="dB scale", width=220)

        freq_min_w = pn.widgets.FloatInput.from_param(
            self.settings.param.freq_min, name="Freq min (Hz)", width=105
        )
        freq_max_w = pn.widgets.FloatInput.from_param(
            self.settings.param.freq_max, name="Freq max (Hz)", width=105
        )

        freq_slider = pn.widgets.FloatSlider.from_param(
            self.settings.param.freq,
            name="Freq (Hz)",
            width=220,
            step=0.5,
        )
        freq_input = pn.widgets.FloatInput.from_param(
            self.settings.param.freq,
            name="Freq (Hz)",
            width=220,
        )

        def _clamp_range(_event=None) -> None:
            lo = float(self.settings.freq_min)
            hi = float(self.settings.freq_max)
            if not np.isfinite(lo):
                lo = 0.0
            if not np.isfinite(hi):
                hi = float(fmax_default)
            if hi < lo:
                hi = lo
            self.settings.freq_min = lo
            self.settings.freq_max = hi
            try:
                self.settings.param["freq"].bounds = (float(lo), float(hi))
            except Exception:  # noqa: BLE001
                pass
            if float(self.settings.freq) < lo:
                self.settings.freq = lo
            if float(self.settings.freq) > hi:
                self.settings.freq = hi

        self._watch(self.settings, lambda _e: _clamp_range(), ["freq_min", "freq_max"])
        _clamp_range()

        self._panel = pn.Column(
            pn.pane.Markdown("### PSD Settings"),
            window_size_w,
            nperseg_w,
            method_w,
            db_w,
            pn.layout.Divider(),
            pn.Row(freq_min_w, freq_max_w),
            freq_slider,
            freq_input,
            sizing_mode="stretch_width",
        )
        return self._panel


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

        self._watch(self.settings, lambda _e: _bump_refresh(), ["window_size", "nperseg", "method", "db"])

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
            key = _PsdCacheKey(time=float(time), tick=int(tick))
            if psd_cache["key"] == key and psd_cache["psd"] is not None:
                return psd_cache["psd"]

            from cogpy.core.spectral.specx import psdx

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
                fmin=float(self.settings.freq_min),
                fmax=float(self.settings.freq_max),
            )

            if bool(self.settings.db):
                from cogpy.core.spectral.psd_utils import psd_to_db

                psd = psd_to_db(psd)

            psd_cache["key"] = key
            psd_cache["psd"] = psd
            return psd

        def _psd_as_channel_freq(psd):
            from cogpy.core.spectral.psd_utils import stack_spatial_dims

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
                ylim=(float(self.settings.freq_min), float(self.settings.freq_max)),
            )

        def _avg(time=None, freq=None, tick=None):
            psd = _compute_psd(float(time), int(tick))
            psd_ch = _psd_as_channel_freq(psd)

            mu = psd_ch.mean(dim="channel")
            sd = psd_ch.std(dim="channel")
            f = np.asarray(mu["freq"].values, dtype=float)
            y = np.asarray(mu.values, dtype=float)
            s = np.asarray(sd.values, dtype=float)

            curve = hv.Curve((f, y), kdims=["freq"], vdims=["power"]).opts(color="#2a6fdb", line_width=2)
            band = hv.Area((f, y - s, y + s), kdims=["freq"], vdims=["lower", "upper"]).opts(
                color="#2a6fdb", alpha=0.2, line_width=0
            )
            f_sel = float(freq) if freq is not None else float(self.settings.freq)
            vline = hv.VLine(f_sel).opts(color="#ffcc00", alpha=0.9, line_width=2)
            return (band * curve * vline).opts(
                width=520,
                height=280,
                tools=["hover"],
                xlabel="Frequency (Hz)",
                ylabel="Power (dB)" if bool(self.settings.db) else "Power",
                title="Average PSD",
                xlim=(float(self.settings.freq_min), float(self.settings.freq_max)),
            )

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

            ap_sel = AP if AP is not None else getattr(self.state.spatial_space, "get_selection", lambda _d: None)("AP")
            ml_sel = ML if ML is not None else getattr(self.state.spatial_space, "get_selection", lambda _d: None)("ML")
            if ap_sel is not None and ml_sel is not None:
                marker = hv.Points([(float(ml_sel), float(ap_sel))], kdims=["ML", "AP"]).opts(
                    color="#00ffff", marker="x", size=14, line_width=3
                )
                return img * marker
            return img

        heatmap_view = hv.DynamicMap(_heatmap, streams=[time_stream, refresh_stream])
        avg_view = hv.DynamicMap(_avg, streams=[time_stream, freq_stream, refresh_stream])
        spatial_view = hv.DynamicMap(_spatial_psd, streams=[time_stream, freq_stream, spatial_stream, refresh_stream])

        heatmap_card = pn.Card(
            pn.pane.HoloViews(heatmap_view, sizing_mode="fixed", width=520, height=420),
            title="PSD Heatmap",
            sizing_mode="fixed",
            min_width=540,
            min_height=480,
        )
        spatial_card = pn.Card(
            pn.pane.HoloViews(spatial_view, sizing_mode="fixed", width=420, height=420),
            title="Spatial PSD",
            sizing_mode="fixed",
            min_width=440,
            min_height=480,
        )
        avg_card = pn.Card(
            pn.pane.HoloViews(avg_view, sizing_mode="fixed", width=520, height=280),
            title="Average PSD",
            sizing_mode="fixed",
            min_width=540,
            min_height=340,
        )

        self._panel = pn.Column(
            pn.pane.Markdown("## PSD Explorer"),
            pn.Row(heatmap_card, spatial_card, sizing_mode="fixed"),
            avg_card,
            sizing_mode="fixed",
        )
        return self._panel
