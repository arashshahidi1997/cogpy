"""PSD settings layer for TensorScope."""

from __future__ import annotations

import numpy as np
import panel as pn
import param

from .base import TensorLayer

__all__ = ["PSDSettings", "PSDSettingsLayer", "_ensure_psd_settings"]


class PSDSettings(param.Parameterized):
    """PSD parameters shared between the sidebar controls and PSD views."""

    # Common
    window_size = param.Number(default=1.0, bounds=(0.01, 60.0), doc="Window size (s)")
    method = param.ObjectSelector(default="welch", objects=["welch", "multitaper"])
    db = param.Boolean(default=False, doc="Display PSD in dB")

    # Welch
    nperseg = param.Integer(default=256, bounds=(16, 8192), doc="FFT size (Welch)")
    noverlap = param.Integer(default=128, bounds=(0, 8192), doc="Overlap (Welch)")

    # Multitaper (psdx uses bandwidth -> NW conversion internally)
    bandwidth = param.Number(default=4.0, bounds=(0.1, 500.0), doc="Bandwidth (Hz, multitaper)")

    # Display range + selection
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
        method_w = pn.widgets.Select.from_param(self.settings.param.method, name="Method", width=220)
        db_w = pn.widgets.Checkbox.from_param(self.settings.param.db, name="dB scale", width=220)

        nperseg_w = pn.widgets.IntInput.from_param(self.settings.param.nperseg, name="FFT", width=220)
        noverlap_w = pn.widgets.IntInput.from_param(self.settings.param.noverlap, name="Overlap", width=220)
        bandwidth_w = pn.widgets.FloatInput.from_param(
            self.settings.param.bandwidth, name="Bandwidth (Hz)", width=220
        )

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

        def _sync_method_widgets(_event=None) -> None:
            m = str(self.settings.method)
            is_welch = m == "welch"
            nperseg_w.visible = is_welch
            noverlap_w.visible = is_welch
            bandwidth_w.visible = not is_welch

            # Keep overlap sane when FFT changes.
            if is_welch:
                try:
                    self.settings.param.noverlap.bounds = (0, int(self.settings.nperseg))
                except Exception:  # noqa: BLE001
                    pass
                if int(self.settings.noverlap) > int(self.settings.nperseg):
                    self.settings.noverlap = int(self.settings.nperseg) // 2

        self._watch(self.settings, lambda _e: _clamp_range(), ["freq_min", "freq_max"])
        self._watch(self.settings, lambda _e: _sync_method_widgets(), ["method", "nperseg"])
        _clamp_range()
        _sync_method_widgets()

        self._panel = pn.Column(
            pn.pane.Markdown("### PSD Settings"),
            window_size_w,
            method_w,
            nperseg_w,
            noverlap_w,
            bandwidth_w,
            db_w,
            pn.layout.Divider(),
            pn.Row(freq_min_w, freq_max_w),
            freq_slider,
            freq_input,
            sizing_mode="stretch_width",
        )
        return self._panel

