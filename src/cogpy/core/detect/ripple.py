"""Ripple detection (v2.6.4): bandpass + envelope + dual threshold."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .base import EventDetector
from .utils import bandpass_filter, dual_threshold_events_1d, hilbert_envelope, zscore_1d

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
        from cogpy.core.events import EventCatalog

        # (1) Bandpass + (2) envelope.
        x_f = bandpass_filter(data, self.freq_range[0], self.freq_range[1], order=self.filter_order)
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

        df = pd.DataFrame.from_records(events) if events else pd.DataFrame(columns=["event_id", "t", "t0", "t1"])
        if not df.empty:
            df = df.sort_values("t").reset_index(drop=True)
            df["event_id"] = [f"ripple_{i:06d}" for i in range(len(df))]
            df["label"] = "ripple"
        return EventCatalog(df=df, name="ripple_events", metadata={"detector": self.name, **self.params})

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
    Spindle detector (optional) implemented as RippleDetector with different defaults.

    Typical spindle band: 11–16 Hz with longer duration constraints.
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
        # ensure serialization includes the right detector name
        self.params = dict(self.params)

    def detect(self, data: xr.DataArray, **kwargs: Any):
        from cogpy.core.events import EventCatalog

        cat = super().detect(data, **kwargs)
        df = cat.df.copy()
        if "label" in df.columns and len(df):
            df["label"] = "spindle"
        return EventCatalog(df=df, name="spindle_events", metadata={"detector": self.name, **self.params})
