"""
EventCatalog Demo (v2.6).

Run with:
    conda run -n cogpy python code/lib/cogpy/examples/events/event_catalog_demo.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cogpy.core.events import EventCatalog

print("=" * 60)
print("EventCatalog Demo")
print("=" * 60)

# Example 1: point events (h-maxima style)
peaks_df = pd.DataFrame(
    {
        "t": [1.2, 2.5, 3.8, 5.1, 6.7],
        "AP": [2, 3, 4, 3, 2],
        "ML": [5, 6, 5, 4, 5],
        "freq": [40.0, 45.0, 38.5, 42.0, 41.5],
        "amp": [12.3, 15.1, 10.2, 14.5, 13.8],
    }
)

catalog = EventCatalog.from_hmaxima(peaks_df, label="burst_peak", detector="detect_hmaxima", h_quantile=0.9)
print("\nPoint catalog:", catalog)
print("Columns:", list(catalog.df.columns))
print("Time span:", float(catalog.df["t"].min()), "→", float(catalog.df["t"].max()))

events = catalog.to_events()
intervals = catalog.to_point_intervals(0.05)
stream = catalog.to_event_stream()
print("As Events:", len(events))
print("As point Intervals:", len(intervals))
print("As EventStream:", stream.name)

# Example 2: interval events (blob candidates)
blob_dict = {
    "t_peak_s": np.array([1.5, 3.2, 5.8]),
    "t0_s": np.array([1.2, 2.9, 5.5]),
    "t1_s": np.array([1.8, 3.5, 6.1]),
    "f_peak_hz": np.array([40.0, 45.0, 42.5]),
    "f0_hz": np.array([35.0, 40.0, 38.0]),
    "f1_hz": np.array([45.0, 50.0, 47.0]),
    "channel": np.array([0, 1, 0]),
    "score": np.array([0.92, 0.88, 0.95]),
}

catalog2 = EventCatalog.from_blob_candidates(blob_dict, detector="detect_blob_candidates")
print("\nInterval catalog:", catalog2)
print("Has intervals:", catalog2.is_interval_events)
print("Mean duration:", float(catalog2.df["duration"].mean()))
print("As Intervals:", len(catalog2.to_intervals()))

# Example 3: filtering
df = pd.DataFrame(
    {
        "event_id": range(20),
        "t": np.linspace(0, 10, 20),
        "channel": np.tile([0, 1, 2, 3], 5),
        "AP": np.random.uniform(0, 5, 20),
        "ML": np.random.uniform(0, 5, 20),
        "value": np.random.uniform(5, 15, 20),
        "label": "event",
    }
)
catalog3 = EventCatalog(df=df, name="test_events")
filtered = catalog3.filter_by_time(2.0, 8.0).filter_by_channel([0, 1])
print("\nFiltered:", filtered)

