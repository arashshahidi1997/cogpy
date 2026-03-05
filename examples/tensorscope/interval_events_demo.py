"""
TensorScope Interval Events Demo (v2.6.3).

Demonstrates:
- interval-aware temporal overlays (VSpan t0→t1 + dashed VLine at t)
- event-triggered average view in `event_explorer`
- overlap detection for interval EventCatalogs

Run with:
    /storage/share/python/environments/Anaconda3/envs/cogpy/bin/python \\
        code/lib/cogpy/examples/tensorscope/interval_events_demo.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cogpy.core.events import EventCatalog
from cogpy.core.events.overlap import detect_overlaps
from cogpy.core.plot.tensorscope.app import TensorScopeApp
from cogpy.datasets.entities import example_ieeg_grid


def main():
    data = example_ieeg_grid(mode="small")
    app = TensorScopeApp(data, title="TensorScope v2.6.3: Interval Events", theme="dark")

    # Simulated interval events (peak time t, interval t0..t1)
    t_min = float(data.time.values[0])
    t_max = float(data.time.values[-1])
    rng = np.random.RandomState(0)
    n = 30
    t0 = np.linspace(t_min + 0.5, t_max - 0.5, n)
    dur = rng.uniform(0.05, 0.25, size=n)
    t1 = t0 + dur
    t = t0 + 0.5 * dur

    df = pd.DataFrame(
        {
            "event_id": [f"iv_{i:04d}" for i in range(n)],
            "t": t,
            "t0": t0,
            "t1": t1,
            "duration": dur,
            "AP": rng.randint(0, int(data.sizes["AP"]), size=n),
            "ML": rng.randint(0, int(data.sizes["ML"]), size=n),
            "freq": rng.uniform(20.0, 80.0, size=n),
            "value": rng.uniform(0.0, 1.0, size=n),
            "label": "interval_demo",
        }
    )

    catalog = EventCatalog(df=df, name="intervals")
    overlaps = detect_overlaps(catalog)
    print(f"Interval events: n={len(catalog)}, overlap_pairs={len(overlaps)}")

    # Register under the default stream name used by TensorScopeApp event layers.
    app.state.register_event_catalog("bursts", catalog, style={"color": "#ff4444", "alpha": 0.8})

    # Build default app layout (includes EventTableLayer + EventOverlayLayer panels).
    template = app.build()
    return template


if __name__ == "__main__":
    main().show()

