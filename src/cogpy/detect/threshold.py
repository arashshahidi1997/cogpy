"""Threshold-based event detection (v2.6.4)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .base import EventDetector
from .utils import find_true_runs, merge_intervals

__all__ = ["ThresholdDetector"]


class ThresholdDetector(EventDetector):
    """
    Generic threshold crossing detector producing interval events.

    Supports:
    - positive threshold: x >= threshold
    - negative threshold: x <= -abs(threshold) (or <= threshold if threshold is negative)
    - both: abs(x) >= abs(threshold)

    Parameters
    ----------
    threshold
        Threshold in data units.
    direction
        'positive', 'negative', or 'both'.
    bandpass
        Optional (low, high) bandpass in Hz.
    use_envelope
        If True, compute Hilbert envelope before thresholding.
    min_duration
        Minimum event duration in seconds.
    merge_gap
        Merge events separated by <= merge_gap seconds.
    filter_order
        Bandpass filter order (only used if bandpass is set).
    """

    def __init__(
        self,
        threshold: float,
        *,
        direction: str = "both",
        bandpass: tuple[float, float] | None = None,
        use_envelope: bool = False,
        min_duration: float = 0.0,
        merge_gap: float = 0.0,
        filter_order: int = 4,
    ) -> None:
        super().__init__("ThresholdDetector")

        self.threshold = float(threshold)
        self.direction = str(direction)
        self.bandpass = tuple(bandpass) if bandpass is not None else None
        self.use_envelope = bool(use_envelope)
        self.min_duration = float(min_duration)
        self.merge_gap = float(merge_gap)
        self.filter_order = int(filter_order)

        self.params = {
            "threshold": self.threshold,
            "direction": self.direction,
            "bandpass": self.bandpass,
            "use_envelope": self.use_envelope,
            "min_duration": self.min_duration,
            "merge_gap": self.merge_gap,
            "filter_order": self.filter_order,
        }

    def can_accept(self, data: xr.DataArray) -> bool:
        return "time" in data.dims

    def detect(self, data: xr.DataArray, **_kwargs: Any):
        from cogpy.events import EventCatalog

        x = data
        if self.bandpass is not None:
            from .utils import bandpass_filter

            x = bandpass_filter(x, self.bandpass[0], self.bandpass[1], order=self.filter_order)

        if self.use_envelope:
            from .utils import hilbert_envelope

            x = hilbert_envelope(x)

        events: list[dict[str, Any]] = []

        if ("AP" in x.dims) and ("ML" in x.dims):
            n_ap = int(x.sizes["AP"])
            n_ml = int(x.sizes["ML"])
            for ap_i in range(n_ap):
                for ml_i in range(n_ml):
                    ts = x.isel(AP=ap_i, ML=ml_i)
                    events.extend(self._detect_1d(ts, AP=int(ap_i), ML=int(ml_i)))
        elif "channel" in x.dims:
            n_ch = int(x.sizes["channel"])
            for ch_i in range(n_ch):
                ts = x.isel(channel=ch_i)
                events.extend(self._detect_1d(ts, channel=int(ch_i)))
        else:
            events.extend(self._detect_1d(x))

        df = pd.DataFrame.from_records(events) if events else pd.DataFrame(columns=["event_id", "t"])
        if not df.empty:
            df["event_id"] = [f"thresh_{i:06d}" for i in range(len(df))]
            # Ensure canonical columns exist.
            for col in ("t", "t0", "t1"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            if ("t0" in df.columns) and ("t1" in df.columns) and ("duration" not in df.columns):
                df["duration"] = df["t1"] - df["t0"]

        return EventCatalog(df=df, name="threshold_events", metadata={"detector": self.name, **self.params})

    def _detect_1d(self, ts: xr.DataArray, **loc: Any) -> list[dict[str, Any]]:
        if "time" not in ts.dims:
            return []

        t = np.asarray(ts["time"].values, dtype=float)
        y = np.asarray(ts.data, dtype=float)
        try:
            y = y.compute()
        except Exception:  # noqa: BLE001
            pass
        y = np.asarray(y, dtype=float).reshape(-1)

        if t.size != y.size or t.size < 2:
            return []

        thr = float(self.threshold)
        direction = str(self.direction)
        if direction not in {"positive", "negative", "both"}:
            raise ValueError("direction must be 'positive', 'negative', or 'both'")

        if direction == "positive":
            mask = y >= abs(thr)
            score = y
        elif direction == "negative":
            neg_thr = thr if thr < 0 else -abs(thr)
            mask = y <= float(neg_thr)
            score = -y  # larger is "more extreme"
        else:
            mask = np.abs(y) >= abs(thr)
            score = np.abs(y)

        intervals = find_true_runs(mask)
        if not intervals:
            return []

        # Merge by sample gap.
        if self.merge_gap > 0:
            dt = float(np.median(np.diff(t)))
            if np.isfinite(dt) and dt > 0:
                gap_samp = int(round(self.merge_gap / dt))
                intervals = merge_intervals(intervals, gap=gap_samp)

        out: list[dict[str, Any]] = []
        for i0, i1 in intervals:
            if i1 <= i0:
                continue

            t0 = float(t[i0])
            t1 = float(t[i1])
            dur = float(t1 - t0)
            if self.min_duration and dur < float(self.min_duration):
                continue

            seg = score[i0 : i1 + 1]
            ip = i0 + int(np.argmax(seg))
            out.append(
                {
                    "t0": t0,
                    "t": float(t[ip]),
                    "t1": t1,
                    "duration": dur,
                    "value": float(y[ip]),
                    "label": "threshold",
                    **loc,
                }
            )
        return out

    def get_event_dims(self) -> list[str]:
        return ["time"]

