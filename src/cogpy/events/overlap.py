"""Overlap detection utilities for interval EventCatalogs."""

from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = ["detect_overlaps"]


def detect_overlaps(catalog: Any) -> pd.DataFrame:
    """
    Detect overlapping interval events in an EventCatalog.

    Parameters
    ----------
    catalog
        `cogpy.events.EventCatalog` with interval columns `t0`, `t1`.

    Returns
    -------
    pd.DataFrame
        Overlap pairs with columns:
        - event_1, event_2
        - overlap_start, overlap_end, overlap_duration
    """
    from cogpy.events import EventCatalog

    if not isinstance(catalog, EventCatalog):
        raise TypeError(f"catalog must be an EventCatalog, got {type(catalog)!r}")

    cols = ["event_1", "event_2", "overlap_start", "overlap_end", "overlap_duration"]
    if not catalog.is_interval_events:
        return pd.DataFrame(columns=cols)

    df = catalog.df.sort_values("t0").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=cols)

    overlaps: list[dict[str, object]] = []
    n = int(len(df))
    for i in range(n - 1):
        t1_i = float(df.iloc[i]["t1"])
        for j in range(i + 1, n):
            t0_j = float(df.iloc[j]["t0"])
            if t1_i <= t0_j:
                break

            overlap_start = t0_j
            overlap_end = min(t1_i, float(df.iloc[j]["t1"]))
            overlaps.append(
                {
                    "event_1": df.iloc[i]["event_id"],
                    "event_2": df.iloc[j]["event_id"],
                    "overlap_start": overlap_start,
                    "overlap_end": overlap_end,
                    "overlap_duration": float(overlap_end - overlap_start),
                }
            )

    return pd.DataFrame(overlaps, columns=cols)

