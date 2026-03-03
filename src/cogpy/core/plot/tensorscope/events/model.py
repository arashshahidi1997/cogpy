"""
Event stream data model for TensorScope.

Events represent discrete occurrences in time (bursts, ripples, spikes, annotations).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class EventStyle:
    """
    Visual style for event display.

    Attributes
    ----------
    color
        Marker/line color (hex or named).
    marker
        Marker shape ('circle', 'square', 'triangle').
    line_width
        Line width for markers.
    alpha
        Transparency (0-1).
    """

    color: str = "#ff0000"
    marker: str = "circle"
    line_width: float = 2.0
    alpha: float = 0.8


class EventStream:
    """
    Container for event data.

    Events are stored in a pandas DataFrame with required and optional columns.

    Required columns:
    - event_id (int or str): Unique identifier
    - t (float): Event time in seconds

    Optional columns:
    - channel (int): Flat channel index
    - AP (int): Grid row
    - ML (int): Grid column
    - label (str): Event type/category
    - value (float): Amplitude/score
    - duration (float): Duration in seconds
    - ... (any other metadata)
    """

    def __init__(
        self,
        name: str,
        df: pd.DataFrame,
        *,
        time_col: str = "t",
        id_col: str = "event_id",
        style: EventStyle | None = None,
    ):
        if time_col not in df.columns:
            raise ValueError(f"DataFrame must have {time_col!r} column")
        if id_col not in df.columns:
            raise ValueError(f"DataFrame must have {id_col!r} column")

        self.name = str(name)
        self.time_col = str(time_col)
        self.id_col = str(id_col)
        self.style = style or EventStyle()

        self.df = df.copy()
        self.df = self.df.sort_values(self.time_col).reset_index(drop=True)

    def get_events_in_window(self, t0: float, t1: float) -> pd.DataFrame:
        t0f = float(t0)
        t1f = float(t1)
        if t1f < t0f:
            t0f, t1f = t1f, t0f
        mask = (self.df[self.time_col] >= t0f) & (self.df[self.time_col] <= t1f)
        return self.df[mask]

    def get_event_by_id(self, event_id) -> pd.Series | None:
        mask = self.df[self.id_col] == event_id
        events = self.df[mask]
        if len(events) > 0:
            return events.iloc[0]
        return None

    def get_next_event(self, current_time: float) -> pd.Series | None:
        mask = self.df[self.time_col] > float(current_time)
        events = self.df[mask]
        if len(events) > 0:
            return events.iloc[0]
        return None

    def get_prev_event(self, current_time: float) -> pd.Series | None:
        mask = self.df[self.time_col] < float(current_time)
        events = self.df[mask]
        if len(events) > 0:
            return events.iloc[-1]
        return None

    def __len__(self) -> int:
        return len(self.df)

    def to_dict(self) -> dict:
        t_min = float(self.df[self.time_col].min()) if len(self.df) else None
        t_max = float(self.df[self.time_col].max()) if len(self.df) else None
        return {
            "name": self.name,
            "time_col": self.time_col,
            "id_col": self.id_col,
            "n_events": len(self.df),
            "time_range": (t_min, t_max),
            "style": {
                "color": self.style.color,
                "marker": self.style.marker,
                "alpha": self.style.alpha,
                "line_width": self.style.line_width,
            },
        }

