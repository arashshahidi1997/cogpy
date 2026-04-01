"""Ripple detection (v2.6.4): bandpass + envelope + dual threshold."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .base import EventDetector
from .utils import (
    bandpass_filter,
    dual_threshold_events_1d,
    hilbert_envelope,
    zscore_1d,
)

__all__ = ["RippleDetector", "SpindleDetector"]


class RippleDetector(EventDetector):
    """
    Ripple detector using bandpass → Hilbert envelope → z-score → dual threshold.

    Produces **interval events** with (t0, t, t1, duration, value).
    """

    def __init__(
        self,
        *,
        freq_range: tuple[float, float] = (100.0, 250.0),
        threshold_low: float = 2.0,
        threshold_high: float = 3.0,
        min_duration: float = 0.02,
        max_duration: float = 0.2,
        filter_order: int = 4,
        direction: str = "positive",
    ) -> None:
        super().__init__("RippleDetector")

        self.freq_range = (float(freq_range[0]), float(freq_range[1]))
        self.threshold_low = float(threshold_low)
        self.threshold_high = float(threshold_high)
        self.min_duration = float(min_duration)
        self.max_duration = float(max_duration)
        self.filter_order = int(filter_order)
        self.direction = str(direction)

        self.params = {
            "freq_range": self.freq_range,
            "threshold_low": self.threshold_low,
            "threshold_high": self.threshold_high,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "filter_order": self.filter_order,
            "direction": self.direction,
        }

    def can_accept(self, data: xr.DataArray) -> bool:
        return "time" in data.dims

    def detect(self, data: xr.DataArray, **_kwargs: Any):
        from cogpy.events import EventCatalog

        # (1) Bandpass + (2) envelope.
        x_f = bandpass_filter(
            data, self.freq_range[0], self.freq_range[1], order=self.filter_order
        )
        env = hilbert_envelope(x_f)

        events: list[dict[str, Any]] = []
        if ("AP" in env.dims) and ("ML" in env.dims):
            n_ap = int(env.sizes["AP"])
            n_ml = int(env.sizes["ML"])
            for ap_i in range(n_ap):
                for ml_i in range(n_ml):
                    ts = env.isel(AP=ap_i, ML=ml_i)
                    events.extend(self._detect_1d(ts, AP=int(ap_i), ML=int(ml_i)))
        elif "channel" in env.dims:
            n_ch = int(env.sizes["channel"])
            for ch_i in range(n_ch):
                ts = env.isel(channel=ch_i)
                events.extend(self._detect_1d(ts, channel=int(ch_i)))
        else:
            events.extend(self._detect_1d(env))

        df = (
            pd.DataFrame.from_records(events)
            if events
            else pd.DataFrame(columns=["event_id", "t", "t0", "t1"])
        )
        if not df.empty:
            df = df.sort_values("t").reset_index(drop=True)
            df["event_id"] = [f"ripple_{i:06d}" for i in range(len(df))]
            df["label"] = "ripple"
        return EventCatalog(
            df=df, name="ripple_events", metadata={"detector": self.name, **self.params}
        )

    def _detect_1d(self, ts: xr.DataArray, **loc: Any) -> list[dict[str, Any]]:
        if "time" not in ts.dims:
            return []

        t = np.asarray(ts["time"].values, dtype=float)
        y = ts.data
        try:
            y = y.compute()
        except Exception:  # noqa: BLE001
            pass
        y = np.asarray(y, dtype=float).reshape(-1)

        if t.size != y.size or t.size < 2:
            return []

        z = zscore_1d(y)
        evs = dual_threshold_events_1d(
            z,
            t,
            low=self.threshold_low,
            high=self.threshold_high,
            direction=self.direction,
        )

        out: list[dict[str, Any]] = []
        for ev in evs:
            dur = float(ev["t1"] - ev["t0"])
            if (dur < self.min_duration) or (dur > self.max_duration):
                continue
            out.append({**ev, **loc})
        return out

    def get_event_dims(self) -> list[str]:
        return ["time"]

    def get_transform_info(self) -> dict:
        return {
            "required": True,
            "transform_type": "BandpassEnvelopeZScore",
            "params": {
                "freq_range": self.freq_range,
                "filter_order": self.filter_order,
                "threshold_low": self.threshold_low,
                "threshold_high": self.threshold_high,
            },
            "implicit": True,
            "explicit": False,
        }


class SpindleDetector(RippleDetector):
    """
    Spindle detector with optional yasa-style enrichment features.

    Extends RippleDetector (bandpass → envelope → z-score → dual threshold)
    with per-event metrics:

    - **frequency**: peak oscillation frequency via zero-crossing count
    - **rel_power**: sigma-band power / broadband power within each event
    - **symmetry**: position of peak amplitude (0–1; 0.5 = symmetric)
    - **isolation**: reject events closer than *min_isolation* seconds

    All enrichment features are off by default to keep the detector lightweight.

    Typical spindle band: 11–16 Hz with duration 0.5–3.0 s.
    """

    def __init__(
        self,
        *,
        freq_range: tuple[float, float] = (11.0, 16.0),
        threshold_low: float = 2.0,
        threshold_high: float = 3.0,
        min_duration: float = 0.5,
        max_duration: float = 3.0,
        filter_order: int = 4,
        direction: str = "positive",
        # enrichment options (off by default)
        compute_frequency: bool = False,
        compute_rel_power: bool = False,
        rel_power_broadband: tuple[float, float] = (1.0, 40.0),
        rel_power_min: float | None = None,
        compute_symmetry: bool = False,
        min_isolation: float | None = None,
    ) -> None:
        super().__init__(
            freq_range=freq_range,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            min_duration=min_duration,
            max_duration=max_duration,
            filter_order=filter_order,
            direction=direction,
        )
        self.name = "SpindleDetector"
        self.compute_frequency = bool(compute_frequency)
        self.compute_rel_power = bool(compute_rel_power)
        self.rel_power_broadband = (
            float(rel_power_broadband[0]),
            float(rel_power_broadband[1]),
        )
        self.rel_power_min = float(rel_power_min) if rel_power_min is not None else None
        self.compute_symmetry = bool(compute_symmetry)
        self.min_isolation = float(min_isolation) if min_isolation is not None else None
        # ensure serialization includes the right detector name
        self.params = {
            **self.params,
            "compute_frequency": self.compute_frequency,
            "compute_rel_power": self.compute_rel_power,
            "rel_power_broadband": self.rel_power_broadband,
            "rel_power_min": self.rel_power_min,
            "compute_symmetry": self.compute_symmetry,
            "min_isolation": self.min_isolation,
        }

    def detect(self, data: xr.DataArray, **kwargs: Any):
        from cogpy.events import EventCatalog

        cat = super().detect(data, **kwargs)
        df = cat.df.copy()
        if "label" in df.columns and len(df):
            df["label"] = "spindle"

        if not df.empty:
            df = self._enrich(df, data)

        return EventCatalog(
            df=df,
            name="spindle_events",
            metadata={"detector": self.name, **self.params},
        )

    def _enrich(self, df: pd.DataFrame, data: xr.DataArray) -> pd.DataFrame:
        """Apply optional enrichment columns to the detected events."""
        if not (
            self.compute_frequency
            or self.compute_rel_power
            or self.compute_symmetry
            or self.min_isolation is not None
        ):
            return df

        t_arr = np.asarray(data["time"].values, dtype=float)

        # Precompute filtered signal for frequency estimation
        if self.compute_frequency:
            x_f = bandpass_filter(
                data, self.freq_range[0], self.freq_range[1], order=self.filter_order
            )

        # Precompute broadband power for rel_power
        if self.compute_rel_power:
            x_broad = bandpass_filter(
                data,
                self.rel_power_broadband[0],
                self.rel_power_broadband[1],
                order=self.filter_order,
            )
            x_sigma = bandpass_filter(
                data, self.freq_range[0], self.freq_range[1], order=self.filter_order
            )

        freq_vals: list[float] = []
        rel_power_vals: list[float] = []
        symmetry_vals: list[float] = []

        for _, row in df.iterrows():
            t0, t1 = float(row["t0"]), float(row["t1"])
            mask = (t_arr >= t0) & (t_arr <= t1)

            # --- Per-event signal extraction (1D or spatial) ---
            loc_kw: dict[str, Any] = {}
            if "AP" in df.columns and "AP" in data.dims:
                loc_kw["AP"] = int(row["AP"])
            if "ML" in df.columns and "ML" in data.dims:
                loc_kw["ML"] = int(row["ML"])
            if "channel" in df.columns and "channel" in data.dims:
                loc_kw["channel"] = int(row["channel"])

            def _sel(arr: xr.DataArray) -> np.ndarray:
                s = arr.isel(**loc_kw) if loc_kw else arr
                v = np.asarray(s.values, dtype=float)
                if v.ndim > 1:
                    v = v.reshape(-1)
                return v[mask]

            # --- Frequency (zero-crossing count) ---
            if self.compute_frequency:
                seg = _sel(x_f)
                if seg.size > 2:
                    crossings = np.sum(np.diff(np.sign(seg)) != 0)
                    dur = t1 - t0
                    freq_vals.append(
                        float(crossings / (2.0 * dur)) if dur > 0 else np.nan
                    )
                else:
                    freq_vals.append(np.nan)

            # --- Relative sigma power ---
            if self.compute_rel_power:
                seg_sigma = _sel(x_sigma)
                seg_broad = _sel(x_broad)
                sigma_pow = float(np.mean(seg_sigma**2)) if seg_sigma.size > 0 else 0.0
                broad_pow = float(np.mean(seg_broad**2)) if seg_broad.size > 0 else 0.0
                rel_power_vals.append(
                    sigma_pow / broad_pow if broad_pow > 0 else np.nan
                )

            # --- Symmetry index ---
            if self.compute_symmetry:
                seg_raw = _sel(data)
                env_seg = np.abs(seg_raw)
                if env_seg.size > 1:
                    peak_pos = int(np.argmax(env_seg))
                    symmetry_vals.append(float(peak_pos) / float(env_seg.size - 1))
                else:
                    symmetry_vals.append(np.nan)

        if self.compute_frequency:
            df["frequency"] = freq_vals
        if self.compute_rel_power:
            df["rel_power"] = rel_power_vals
        if self.compute_symmetry:
            df["symmetry"] = symmetry_vals

        # --- Isolation criterion: reject events that are too close ---
        if self.min_isolation is not None and len(df) > 1:
            df = self._apply_isolation(df)

        # --- Relative power threshold ---
        if self.compute_rel_power and self.rel_power_min is not None:
            df = df[df["rel_power"] >= self.rel_power_min].reset_index(drop=True)

        return df

    def _apply_isolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove events that are too close, keeping the stronger one."""
        min_iso = self.min_isolation
        if min_iso is None or len(df) < 2:
            return df

        df = df.sort_values("t").reset_index(drop=True)
        keep = np.ones(len(df), dtype=bool)
        t_vals = df["t"].values
        v_vals = df["value"].values if "value" in df.columns else np.zeros(len(df))

        for i in range(len(df) - 1):
            if not keep[i]:
                continue
            for j in range(i + 1, len(df)):
                if not keep[j]:
                    continue
                if t_vals[j] - t_vals[i] >= min_iso:
                    break
                # Too close — drop the weaker one
                if v_vals[j] > v_vals[i]:
                    keep[i] = False
                    break
                else:
                    keep[j] = False

        return df[keep].reset_index(drop=True)
