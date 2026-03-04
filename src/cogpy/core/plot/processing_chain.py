"""
processing_chain.py
===================
Param-based processing chain for on-demand visualization.

This module intentionally mirrors NeuroScope2's "slice → process → return"
pattern: processing is applied imperatively at render time (not reactively).
"""

from __future__ import annotations

import numpy as np
import param
import panel as pn
import xarray as xr

from cogpy.core.preprocess.filtx import bandpassx, cmrx, median_spatialx, notchesx, zscorex

__all__ = ["ProcessingChain"]


def _fmt_num(x: float) -> str:
    s = f"{float(x):.6f}".rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return s


class ProcessingChain(param.Parameterized):
    """
    Applies optional processing to a time window of xr.DataArray on demand.

    Call get_window(t0, t1) from Element render callbacks to materialize
    and process a time slice. Processing params are Panel-controllable via
    controls().

    Processing order (fixed, matches NeuroScope2):
        1. CMR (common median reference)
        2. Bandpass filter
        3. Notch filter (line noise removal)
        4. Spatial median (grid input only)
        5. Z-score (display normalization, last)

    Features
    --------
    - Notch filter:
        - List mode: notch exact frequencies, e.g. [60.0, 120.0]
        - Harmonic mode: notch fundamental + N harmonics, e.g. 60Hz × 3
    - PSD computation:
        - Multitaper or Welch PSD via compute_psd(), using the same processing chain

    Parameters
    ----------
    data : xr.DataArray
        Full lazy or eager signal array. Not materialized until get_window()
        is called. Must conform to IEEGGridTimeSeries or IEEGTimeChannel schema.
    time_dim : str
        Name of the time dimension in ``data``. Default ``"time"``.
    """

    # CMR
    cmr_on = param.Boolean(default=False, doc="Common median reference")

    # Bandpass
    bandpass_on = param.Boolean(default=False)
    bandpass_lo = param.Number(default=1.0, bounds=(0.01, 499.0))
    bandpass_hi = param.Number(default=40.0, bounds=(0.1, 500.0))
    bandpass_order = param.Integer(default=3, bounds=(1, 8))

    # Notch filter
    notch_on = param.Boolean(default=False, doc="Notch filter (line noise removal)")
    notch_freqs = param.List(
        default=[60.0],
        doc="Frequencies to notch (Hz). E.g., [60.0] or [50.0, 100.0, 150.0]",
    )
    notch_fundamental = param.Number(
        default=60.0,
        bounds=(1.0, 500.0),
        doc="Fundamental frequency for harmonic notching",
    )
    notch_harmonics = param.Integer(
        default=3,
        bounds=(1, 10),
        doc="Number of harmonics to notch (including fundamental)",
    )
    notch_use_harmonics = param.Boolean(
        default=False,
        doc="Use harmonic mode (notch fundamental + N harmonics)",
    )
    notch_q = param.Number(
        default=30.0,
        bounds=(1.0, 100.0),
        doc="Quality factor (higher = narrower notch)",
    )

    # Spatial median (grid only)
    spatial_median_on = param.Boolean(default=False)
    spatial_median_size = param.Integer(default=3, bounds=(1, 15))

    # Z-score
    zscore_on = param.Boolean(default=True)
    zscore_robust = param.Boolean(
        default=False, doc="Use median/MAD instead of mean/std"
    )

    def __init__(self, data: xr.DataArray, *, time_dim: str = "time", **params):
        super().__init__(**params)
        self._data = data
        self._time_dim = str(time_dim)
        self._fs = float(data.attrs.get("fs", 1.0))
        self._is_grid = self._detect_grid_capability(data)
        if self._time_dim not in data.dims:
            raise ValueError(
                f"ProcessingChain expected time_dim={self._time_dim!r} in data.dims={tuple(data.dims)}"
            )

        self._controls_view = None

    @staticmethod
    def _detect_grid_capability(data: xr.DataArray) -> bool:
        if "AP" in data.dims and "ML" in data.dims:
            return True

        if "channel" not in data.dims:
            return False

        # Stacked-flat form (recommended): AP/ML are per-channel coords.
        try:
            if (
                "AP" in data.coords
                and "ML" in data.coords
                and data.coords["AP"].dims == ("channel",)
                and data.coords["ML"].dims == ("channel",)
            ):
                return True
        except Exception:  # noqa: BLE001
            pass

        # Fallback: MultiIndex channel with AP/ML levels.
        try:
            if "channel" in data.coords:
                idx = data["channel"].to_index()
                names = set(getattr(idx, "names", []) or [])
                if {"AP", "ML"}.issubset(names):
                    return True
        except Exception:  # noqa: BLE001
            pass

        return False

    @staticmethod
    def _extract_ap_ml(sig: xr.DataArray) -> tuple[object, object]:
        if "AP" in sig.coords and "ML" in sig.coords and sig.coords["AP"].dims == ("channel",) and sig.coords["ML"].dims == ("channel",):
            return sig.coords["AP"].values, sig.coords["ML"].values

        try:
            idx = sig["channel"].to_index()
            names = set(getattr(idx, "names", []) or [])
            if {"AP", "ML"}.issubset(names):
                return idx.get_level_values("AP").to_numpy(), idx.get_level_values("ML").to_numpy()
        except Exception:  # noqa: BLE001
            pass

        raise ValueError(
            "Grid-capable flat data must provide AP/ML per-channel coords "
            "or a MultiIndex channel with AP/ML levels."
        )

    @staticmethod
    def _median_spatial_flat(sig: xr.DataArray, *, size: int, time_dim: str) -> xr.DataArray:
        """
        Apply spatial median to a flat (time, channel) DataArray with AP/ML coords.

        Preserves channel order/coords by mapping each channel to (AP, ML), filtering
        on a dense grid, then sampling back at the same (AP, ML) locations.
        """
        time_dim = str(time_dim)
        if time_dim not in sig.dims or "channel" not in sig.dims:
            raise ValueError(
                f"Expected dims ({time_dim!r},'channel') (order flexible), got dims={tuple(sig.dims)}"
            )

        sig_tc = sig.transpose(time_dim, "channel")
        ap_vals, ml_vals = ProcessingChain._extract_ap_ml(sig_tc)
        ap_u = np.unique(np.asarray(ap_vals))
        ml_u = np.unique(np.asarray(ml_vals))

        ap_to_i = {v: i for i, v in enumerate(ap_u)}
        ml_to_i = {v: i for i, v in enumerate(ml_u)}

        t_len = int(sig_tc.sizes[time_dim])
        n_ap = len(ap_u)
        n_ml = len(ml_u)

        grid = np.full((t_len, n_ap, n_ml), np.nan, dtype=np.float64)
        vals = np.asarray(sig_tc.values, dtype=np.float64)
        for ch in range(int(sig_tc.sizes["channel"])):
            ai = ap_to_i[ap_vals[ch]]
            mi = ml_to_i[ml_vals[ch]]
            grid[:, ai, mi] = vals[:, ch]

        grid_x = xr.DataArray(
            grid,
            dims=(time_dim, "AP", "ML"),
            coords={time_dim: sig_tc[time_dim].values, "AP": ap_u, "ML": ml_u},
            attrs=dict(sig_tc.attrs),
            name=sig_tc.name,
        )
        grid_f = median_spatialx(grid_x, size=int(size))

        out_vals = np.empty_like(vals)
        for ch in range(int(sig_tc.sizes["channel"])):
            ai = ap_to_i[ap_vals[ch]]
            mi = ml_to_i[ml_vals[ch]]
            out_vals[:, ch] = np.asarray(grid_f.values)[:, ai, mi]

        out = xr.DataArray(
            out_vals,
            dims=(time_dim, "channel"),
            coords=sig_tc.coords,
            attrs=dict(sig_tc.attrs),
            name=sig_tc.name,
        )
        # Restore original dim order if needed.
        return out.transpose(*sig.dims)

    def get_window(
        self,
        t0: float,
        t1: float,
        *,
        channels: list[int] | None = None,
    ) -> xr.DataArray:
        """
        Materialize and process a time window. Hot path — called on every render.

        Steps (in order):
            1. Slice time axis to [t0, t1]
            2. Optionally select channels by integer index
            3. .compute() to materialize (handles Dask)
            4. Apply enabled processing steps in canonical order
            5. Return processed numpy-backed DataArray

        Parameters
        ----------
        t0, t1 : float
            Time bounds in seconds (same units as data[time_dim] coordinate).
        channels : list[int] | None
            Integer indices into the channel axis. None = all channels.
            Only used for non-grid (time×channel) input.

        Notes
        -----
        Bandpass with low cutoff (< 1 Hz) can require long windows to avoid
        edge artifacts; as a rule of thumb use at least ~10 * (1 / bandpass_lo)
        seconds of data when possible.
        """
        t0f = float(t0)
        t1f = float(t1)
        if t1f < t0f:
            t0f, t1f = t1f, t0f

        win = self._data.sel({self._time_dim: slice(t0f, t1f)})

        apply_channels_late = False
        if (channels is not None) and ("channel" in win.dims):
            # For grid-capable flat data, CMR and spatial filters should see the full grid
            # (median across channels / spatial neighborhood). Subset after processing.
            if bool(self._is_grid) and (bool(self.cmr_on) or bool(self.spatial_median_on)):
                apply_channels_late = True
            else:
                win = win.isel(channel=list(channels))

        # materialize (Dask -> NumPy)
        win = win.compute()

        if bool(self.cmr_on):
            win = cmrx(win)

        if bool(self.bandpass_on):
            lo = float(self.bandpass_lo)
            hi = float(self.bandpass_hi)
            if not (hi > lo):
                raise ValueError(f"bandpass_hi must be > bandpass_lo; got lo={lo}, hi={hi}")
            win = bandpassx(win, lo, hi, int(self.bandpass_order), axis=self._time_dim)

        if bool(self.notch_on):
            if bool(self.notch_use_harmonics):
                fund = float(self.notch_fundamental)
                n_harm = int(self.notch_harmonics)
                freqs_to_notch = [fund * i for i in range(1, n_harm + 1)]
            else:
                freqs_to_notch = [float(f) for f in (self.notch_freqs or [])]

            if freqs_to_notch:
                win = notchesx(win, freqs=freqs_to_notch, Q=float(self.notch_q), time_dim=self._time_dim)

        if bool(self.spatial_median_on) and bool(self._is_grid):
            if "AP" in win.dims and "ML" in win.dims:
                win = median_spatialx(win, size=int(self.spatial_median_size))
            elif "channel" in win.dims:
                win = self._median_spatial_flat(
                    win, size=int(self.spatial_median_size), time_dim=self._time_dim
                )

        if bool(self.zscore_on):
            win = zscorex(win, dim=self._time_dim, robust=bool(self.zscore_robust))

        if apply_channels_late and (channels is not None) and ("channel" in win.dims):
            win = win.isel(channel=list(channels))

        return win

    def compute_psd(
        self,
        t0: float,
        t1: float,
        *,
        channels: list[int] | None = None,
        method: str = "multitaper",
        bandwidth: float = 4.0,
        nperseg: int = 256,
    ) -> xr.DataArray:
        """
        Compute power spectral density for a time window.

        Applies the same processing pipeline as get_window(), then transforms
        to frequency domain using multitaper or Welch PSD.
        """
        from cogpy.core.spectral.specx import psdx

        win = self.get_window(t0, t1, channels=channels)
        psd = psdx(
            win,
            axis=self._time_dim,
            method=method,  # type: ignore[arg-type]
            bandwidth=float(bandwidth),
            nperseg=int(nperseg),
        )
        if "freq" in psd.dims:
            psd = psd.transpose("freq", *[d for d in psd.dims if d != "freq"])
        psd.attrs["processing"] = self.describe()
        return psd

    def to_dict(self) -> dict[str, object]:
        """Serialize processing settings (not data)."""
        return {
            "cmr_on": bool(self.cmr_on),
            "bandpass_on": bool(self.bandpass_on),
            "bandpass_lo": float(self.bandpass_lo),
            "bandpass_hi": float(self.bandpass_hi),
            "bandpass_order": int(self.bandpass_order),
            "notch_on": bool(self.notch_on),
            "notch_freqs": list(self.notch_freqs),
            "notch_fundamental": float(self.notch_fundamental),
            "notch_harmonics": int(self.notch_harmonics),
            "notch_use_harmonics": bool(self.notch_use_harmonics),
            "notch_q": float(self.notch_q),
            "spatial_median_on": bool(self.spatial_median_on),
            "spatial_median_size": int(self.spatial_median_size),
            "zscore_on": bool(self.zscore_on),
            "zscore_robust": bool(self.zscore_robust),
        }

    def controls(self) -> pn.Column:
        if self._controls_view is not None:
            return self._controls_view

        cmr_chk = pn.widgets.Checkbox.from_param(
            self.param.cmr_on, name="Common median reference"
        )

        bp_chk = pn.widgets.Checkbox.from_param(self.param.bandpass_on, name="Bandpass")
        bp_lo = pn.widgets.FloatInput.from_param(self.param.bandpass_lo, name="Lo (Hz)", width=110)
        bp_hi = pn.widgets.FloatInput.from_param(self.param.bandpass_hi, name="Hi (Hz)", width=110)
        bp_ord = pn.widgets.IntInput.from_param(self.param.bandpass_order, name="Order", width=110)
        self._bp_group = pn.Column(
            pn.GridBox(bp_lo, bp_hi, ncols=2),
            bp_ord,
        )
        self._bp_group.visible = bool(self.bandpass_on)

        notch_chk = pn.widgets.Checkbox.from_param(
            self.param.notch_on, name="Notch filter (line noise)"
        )
        notch_mode = pn.widgets.RadioButtonGroup(
            name="Mode",
            options=["List", "Harmonics"],
            value="Harmonics" if bool(self.notch_use_harmonics) else "List",
            button_type="default",
        )

        def _update_notch_mode(event):
            self.notch_use_harmonics = event.new == "Harmonics"

        notch_mode.param.watch(_update_notch_mode, "value")

        notch_freqs_input = pn.widgets.TextInput(
            name="Frequencies (Hz, comma-separated)",
            value=",".join(str(float(f)) for f in (self.notch_freqs or [])),
            width=250,
            placeholder="60.0, 120.0, 180.0",
        )

        def _update_notch_freqs(event):
            try:
                freqs = [float(f.strip()) for f in str(event.new).split(",") if f.strip()]
            except ValueError:
                return
            if freqs:
                self.notch_freqs = freqs

        notch_freqs_input.param.watch(_update_notch_freqs, "value")

        notch_fund = pn.widgets.FloatInput.from_param(
            self.param.notch_fundamental, name="Fundamental (Hz)", width=150
        )
        notch_harm = pn.widgets.IntSlider.from_param(
            self.param.notch_harmonics, name="N harmonics", width=200
        )
        notch_q_slider = pn.widgets.FloatSlider.from_param(
            self.param.notch_q,
            name="Q factor (narrowness)",
            width=200,
            start=1.0,
            end=100.0,
            step=1.0,
        )

        self._notch_list_group = pn.Column(notch_freqs_input)
        self._notch_harm_group = pn.Column(notch_fund, notch_harm)
        def _apply_notch_mode(use_harmonics: bool) -> None:
            self._notch_list_group.visible = not bool(use_harmonics)
            self._notch_harm_group.visible = bool(use_harmonics)
            desired = "Harmonics" if bool(use_harmonics) else "List"
            if notch_mode.value != desired:
                notch_mode.value = desired

        _apply_notch_mode(bool(self.notch_use_harmonics))
        self.param.watch(lambda e: _apply_notch_mode(bool(e.new)), "notch_use_harmonics")

        self._notch_group = pn.Column(
            notch_mode,
            self._notch_list_group,
            self._notch_harm_group,
            notch_q_slider,
        )
        self._notch_group.visible = bool(self.notch_on)

        spat_chk = pn.widgets.Checkbox.from_param(
            self.param.spatial_median_on, name="Spatial median"
        )
        spat_size = pn.widgets.IntSlider.from_param(
            self.param.spatial_median_size, name="Kernel size"
        )
        if not self._is_grid:
            spat_chk.disabled = True
            spat_size.disabled = True
        self._spat_group = pn.Column(spat_size)
        self._spat_group.visible = bool(self.spatial_median_on) and bool(self._is_grid)

        z_chk = pn.widgets.Checkbox.from_param(
            self.param.zscore_on, name="Z-score per window"
        )
        z_rob = pn.widgets.Checkbox.from_param(
            self.param.zscore_robust, name="Robust (median/MAD)"
        )
        self._zrob_group = pn.Column(z_rob)
        self._zrob_group.visible = bool(self.zscore_on)

        def _toggle_visible(obj, on: bool):
            obj.visible = bool(on)
            return obj

        def _toggle_spatial(on: bool):
            self._spat_group.visible = bool(on) and bool(self._is_grid)
            return self._spat_group

        def _toggle_notch(on: bool):
            self._notch_group.visible = bool(on)
            return self._notch_group

        bp_visible = pn.bind(_toggle_visible, self._bp_group, self.param.bandpass_on)
        notch_visible = pn.bind(_toggle_notch, self.param.notch_on)
        spat_visible = pn.bind(_toggle_spatial, self.param.spatial_median_on)
        zrob_visible = pn.bind(_toggle_visible, self._zrob_group, self.param.zscore_on)

        self._controls_view = pn.Column(
            pn.pane.Markdown("**Reference**"),
            cmr_chk,
            pn.pane.Markdown("**Temporal filter**"),
            bp_chk,
            bp_visible,
            pn.pane.Markdown("**Notch filter**"),
            notch_chk,
            notch_visible,
            pn.pane.Markdown("**Spatial filter (grid only)**"),
            spat_chk,
            spat_visible,
            pn.pane.Markdown("**Normalization**"),
            z_chk,
            zrob_visible,
        )
        return self._controls_view

    def describe(self) -> str:
        """
        Return a short human-readable string of active processing steps.
        Used for plot titles and logging.
        Example: "CMR → BP(1–40Hz,ord=3) → SpatMed(3) → Zscore"
        """
        # Snapshot params for consistency if callers mutate during a render.
        cmr_on = bool(self.cmr_on)
        bandpass_on = bool(self.bandpass_on)
        bandpass_lo = float(self.bandpass_lo)
        bandpass_hi = float(self.bandpass_hi)
        bandpass_order = int(self.bandpass_order)
        notch_on = bool(self.notch_on)
        notch_use_harmonics = bool(self.notch_use_harmonics)
        notch_freqs = list(self.notch_freqs)
        notch_fundamental = float(self.notch_fundamental)
        notch_harmonics = int(self.notch_harmonics)
        spatial_median_on = bool(self.spatial_median_on)
        spatial_median_size = int(self.spatial_median_size)
        zscore_on = bool(self.zscore_on)
        zscore_robust = bool(self.zscore_robust)

        parts: list[str] = []

        if cmr_on:
            parts.append("CMR")

        if bandpass_on:
            lo = _fmt_num(bandpass_lo)
            hi = _fmt_num(bandpass_hi)
            parts.append(f"BP({lo}–{hi}Hz,ord={bandpass_order})")

        if notch_on:
            if notch_use_harmonics:
                fund = _fmt_num(notch_fundamental)
                parts.append(f"Notch({fund}Hz×{notch_harmonics})")
            else:
                freqs_str = ",".join(_fmt_num(f) for f in notch_freqs[:3])
                if len(notch_freqs) > 3:
                    freqs_str += ",…"
                parts.append(f"Notch({freqs_str}Hz)")

        if spatial_median_on and bool(self._is_grid):
            parts.append(f"SpatMed({spatial_median_size})")

        if zscore_on:
            parts.append("Zscore(robust)" if zscore_robust else "Zscore")

        return " → ".join(parts) if parts else "Raw"
