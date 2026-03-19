"""
EventCatalog: Unified event data structure (v2.6).

This is a lightweight DataFrame wrapper intended to bridge analysis workflows
and TensorScope visualization. It supports both point events and interval
events (t0/t1 optional).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

__all__ = ["EventCatalog"]


def _as_1d(a) -> np.ndarray:
    arr = np.asarray(a)
    return arr.reshape(-1)


@dataclass
class EventCatalog:
    """
    Unified event catalog with a standardized table schema.

    Required columns:
      - event_id : str | int
      - t        : float (seconds)

    Optional interval columns:
      - t0, t1, duration
    """

    df: pd.DataFrame
    name: str = "events"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError(f"EventCatalog.df must be a pandas DataFrame, got {type(self.df).__name__}")

        required = {"event_id", "t"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        t = pd.to_numeric(self.df["t"], errors="coerce").to_numpy(dtype=float, copy=False)
        if not np.all(np.isfinite(t)):
            raise ValueError("Column 't' contains non-finite values")

        # Interval validation (only when both endpoints are present).
        if ("t0" in self.df.columns) and ("t1" in self.df.columns):
            t0 = pd.to_numeric(self.df["t0"], errors="coerce").to_numpy(dtype=float, copy=False)
            t1 = pd.to_numeric(self.df["t1"], errors="coerce").to_numpy(dtype=float, copy=False)

            if not (np.isfinite(t0).all() and np.isfinite(t1).all()):
                raise ValueError("Interval columns 't0'/'t1' must be finite when present")
            if not np.all(t1 > t0):
                raise ValueError("Interval events must satisfy t1 > t0")

            if "duration" not in self.df.columns:
                self.df = self.df.copy()
                self.df["duration"] = t1 - t0

        # Sort by time (stable baseline for UI + analysis).
        self.df = self.df.sort_values("t").reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        kind = "interval" if self.is_interval_events else "point"
        return f"EventCatalog(name={self.name!r}, n_events={len(self)}, type={kind})"

    @property
    def is_point_events(self) -> bool:
        return not self.is_interval_events

    @property
    def is_interval_events(self) -> bool:
        return ("t0" in self.df.columns) and ("t1" in self.df.columns)

    # -------------------- Converters --------------------

    def to_events(self):
        """Convert to `cogpy.datasets.schemas.Events` (always possible)."""
        from cogpy.datasets.schemas import Events

        times = pd.to_numeric(self.df["t"], errors="coerce").to_numpy(dtype=float, copy=False)
        labels = self.df["label"].to_numpy(dtype=str, copy=False) if "label" in self.df.columns else None
        return Events(times=times, labels=labels, name=str(self.name))

    def to_intervals(self):
        """Convert to `cogpy.datasets.schemas.Intervals` (requires t0/t1)."""
        from cogpy.datasets.schemas import Intervals

        if not self.is_interval_events:
            raise ValueError(
                "Cannot convert to Intervals: events lack t0/t1 columns. "
                "Use to_point_intervals(half_window=...) to create windows around point events."
            )

        starts = pd.to_numeric(self.df["t0"], errors="coerce").to_numpy(dtype=float, copy=False)
        ends = pd.to_numeric(self.df["t1"], errors="coerce").to_numpy(dtype=float, copy=False)
        return Intervals(starts=starts, ends=ends, name=str(self.name))

    def to_point_intervals(self, half_window: float):
        """Convert point events to symmetric windows around `t`."""
        from cogpy.datasets.schemas import Intervals

        hw = float(half_window)
        times = pd.to_numeric(self.df["t"], errors="coerce").to_numpy(dtype=float, copy=False)
        return Intervals(starts=times - hw, ends=times + hw, name=str(self.name))

    def to_event_stream(self, style: Any | None = None):
        """Convert to `EventStream`."""
        from cogpy.events.stream import EventStyle, EventStream

        style_obj = None
        if style is None:
            style_obj = None
        elif isinstance(style, EventStyle):
            style_obj = style
        elif isinstance(style, dict):
            style_obj = EventStyle(**style)
        else:
            raise TypeError("style must be None, an EventStyle, or a dict of EventStyle fields")

        return EventStream(
            name=str(self.name),
            df=self.df.copy(),
            time_col="t",
            id_col="event_id",
            style=style_obj,
        )

    # -------------------- Factories --------------------

    @classmethod
    def from_hmaxima(
        cls,
        peaks_df: pd.DataFrame,
        *,
        time_col: str | None = None,
        label: str = "peak",
        id_prefix: str = "peak",
        value_col: str | None = None,
        **metadata: Any,
    ) -> "EventCatalog":
        """
        Create a point-event catalog from an h-maxima peak table.

        Supports both:
        - `detect_hmaxima`-style columns (`time`, `freq`, `ap`, `ml`, `amp`, ...)
        - orthoslicer-friendly columns (`t`, `z`, `x`, `y`, `value`)
        """
        if not isinstance(peaks_df, pd.DataFrame):
            raise TypeError("peaks_df must be a pandas DataFrame")

        df = peaks_df.copy()

        if time_col is None:
            if "t" in df.columns:
                time_col = "t"
            elif "time" in df.columns:
                time_col = "time"
            else:
                raise ValueError("Could not infer time column; provide time_col=...")
        time_col = str(time_col)
        if time_col not in df.columns:
            raise ValueError(f"time_col {time_col!r} not found in peaks_df columns")

        if value_col is None:
            if "value" in df.columns:
                value_col = "value"
            elif "amp" in df.columns:
                value_col = "amp"
            else:
                value_col = None
        if value_col is not None and value_col not in df.columns:
            raise ValueError(f"value_col {value_col!r} not found in peaks_df columns")

        n = len(df)
        df["event_id"] = [f"{id_prefix}_{i:06d}" for i in range(n)]
        df["t"] = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float, copy=False)
        df["label"] = str(label)
        if value_col is not None:
            df["value"] = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float, copy=False)

        return cls(df=df, name=f"{label}_events", metadata=dict(metadata))

    @classmethod
    def from_blob_candidates(
        cls,
        blob_dict: dict[str, Any],
        *,
        label: str = "burst_candidate",
        id_prefix: str = "blob",
        **metadata: Any,
    ) -> "EventCatalog":
        need = {"t0_s", "t1_s", "t_peak_s", "f0_hz", "f1_hz", "f_peak_hz"}
        missing = sorted(need - set(blob_dict.keys()))
        if missing:
            raise ValueError(f"blob_dict missing required keys: {missing}")

        t0 = _as_1d(blob_dict["t0_s"]).astype(float)
        t1 = _as_1d(blob_dict["t1_s"]).astype(float)
        t = _as_1d(blob_dict["t_peak_s"]).astype(float)
        f0 = _as_1d(blob_dict["f0_hz"]).astype(float)
        f1 = _as_1d(blob_dict["f1_hz"]).astype(float)
        f_peak = _as_1d(blob_dict["f_peak_hz"]).astype(float)

        n = len(t)
        df = pd.DataFrame(
            {
                "event_id": [f"{id_prefix}_{i:06d}" for i in range(n)],
                "t": t,
                "t0": t0,
                "t1": t1,
                "duration": t1 - t0,
                "freq": f_peak,
                "f0": f0,
                "f1": f1,
                "bandwidth": f1 - f0,
                "label": str(label),
            }
        )

        for opt_key, out_col in (
            ("channel", "channel"),
            ("score", "score"),
        ):
            if opt_key in blob_dict:
                df[out_col] = _as_1d(blob_dict[opt_key])

        return cls(df=df, name=f"{label}_events", metadata=dict(metadata))

    @classmethod
    def from_burst_dict(
        cls,
        burst_list: list[dict[str, Any]],
        *,
        label: str = "burst",
        **metadata: Any,
    ) -> "EventCatalog":
        df = pd.DataFrame(list(burst_list or []))
        if len(df) == 0:
            df = pd.DataFrame({"event_id": [], "t": []})
            return cls(df=df, name=f"{label}_events", metadata=dict(metadata))

        rename = {
            "burst_id": "event_id",
            "t_peak_s": "t",
            "t0_s": "t0",
            "t1_s": "t1",
            "f_peak_hz": "freq",
            "f0_hz": "f0",
            "f1_hz": "f1",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        if "label" not in df.columns:
            df["label"] = str(label)
        if ("t0" in df.columns) and ("t1" in df.columns) and ("duration" not in df.columns):
            df["duration"] = pd.to_numeric(df["t1"], errors="coerce") - pd.to_numeric(df["t0"], errors="coerce")
        if ("f0" in df.columns) and ("f1" in df.columns) and ("bandwidth" not in df.columns):
            df["bandwidth"] = pd.to_numeric(df["f1"], errors="coerce") - pd.to_numeric(df["f0"], errors="coerce")

        return cls(df=df, name=f"{label}_events", metadata=dict(metadata))

    @classmethod
    def from_spwr_mat(
        cls,
        mat_data: dict[str, Any],
        *,
        label: str = "ripple",
        id_prefix: str = "ripple",
        **metadata: Any,
    ) -> "EventCatalog":
        for k in ("start", "end", "peak"):
            if k not in mat_data:
                raise ValueError(f"mat_data missing required field: {k!r}")

        t0 = _as_1d(mat_data["start"]).astype(float)
        t1 = _as_1d(mat_data["end"]).astype(float)
        t = _as_1d(mat_data["peak"]).astype(float)
        n = len(t)

        df = pd.DataFrame(
            {
                "event_id": [f"{id_prefix}_{i:06d}" for i in range(n)],
                "t": t,
                "t0": t0,
                "t1": t1,
                "duration": t1 - t0,
                "label": str(label),
            }
        )
        if "peak_amplitude" in mat_data:
            df["value"] = _as_1d(mat_data["peak_amplitude"]).astype(float)

        return cls(df=df, name=f"{label}_events", metadata=dict(metadata))

    # -------------------- Queries --------------------

    def filter_by_time(self, t_min: float, t_max: float) -> "EventCatalog":
        lo = float(min(t_min, t_max))
        hi = float(max(t_min, t_max))
        mask = (self.df["t"] >= lo) & (self.df["t"] <= hi)
        return EventCatalog(df=self.df.loc[mask].copy(), name=str(self.name), metadata=dict(self.metadata))

    def filter_by_channel(self, channels: int | Iterable[int]) -> "EventCatalog":
        if "channel" not in self.df.columns:
            raise ValueError("Events don't have 'channel' column")
        if isinstance(channels, int):
            channels = [channels]
        mask = self.df["channel"].isin(list(channels))
        return EventCatalog(df=self.df.loc[mask].copy(), name=str(self.name), metadata=dict(self.metadata))

    def filter_by_spatial(self, *, AP: float | None = None, ML: float | None = None, radius: float | None = None) -> "EventCatalog":
        df = self.df
        if (AP is None) and (ML is None):
            return EventCatalog(df=df.copy(), name=str(self.name), metadata=dict(self.metadata))

        if (AP is not None) and (ML is not None) and (radius is not None):
            if ("AP" not in df.columns) or ("ML" not in df.columns):
                raise ValueError("Events don't have spatial columns (AP, ML)")
            ap = pd.to_numeric(df["AP"], errors="coerce").to_numpy(dtype=float, copy=False)
            ml = pd.to_numeric(df["ML"], errors="coerce").to_numpy(dtype=float, copy=False)
            dist = np.sqrt((ap - float(AP)) ** 2 + (ml - float(ML)) ** 2)
            mask = dist <= float(radius)
            return EventCatalog(df=df.loc[mask].copy(), name=str(self.name), metadata=dict(self.metadata))

        if AP is not None:
            if "AP" not in df.columns:
                raise ValueError("Events don't have 'AP' column")
            mask = df["AP"] == AP
            return EventCatalog(df=df.loc[mask].copy(), name=str(self.name), metadata=dict(self.metadata))

        if ML is not None:
            if "ML" not in df.columns:
                raise ValueError("Events don't have 'ML' column")
            mask = df["ML"] == ML
            return EventCatalog(df=df.loc[mask].copy(), name=str(self.name), metadata=dict(self.metadata))

        return EventCatalog(df=df.copy(), name=str(self.name), metadata=dict(self.metadata))

