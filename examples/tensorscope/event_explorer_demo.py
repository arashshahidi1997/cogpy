"""
TensorScope Event Explorer Demo (v2.6.2).

Demonstrates:
- creating a TensorScopeState
- running a detector (if available) or using a simulated catalog
- exploring results with the `event_explorer` module

Run with:
    /storage/share/python/environments/Anaconda3/envs/cogpy/bin/python \\
        code/lib/cogpy/examples/tensorscope/event_explorer_demo.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import holoviews as hv
import panel as pn

from cogpy.events import EventCatalog
from cogpy.plot.tensorscope.modules import ModuleRegistry
from cogpy.plot.tensorscope.state import TensorScopeState
from cogpy.datasets.entities import example_ieeg_grid

pn.extension()
hv.extension("bokeh")


def main():
    data = example_ieeg_grid(mode="small")
    state = TensorScopeState(data)

    # Try to run a real detector if available (ghostipy-dependent in some envs).
    catalog = None
    try:
        from cogpy.detect import BurstDetector

        detector = BurstDetector(h_quantile=0.9, nperseg=128, noverlap=64)
        catalog = state.run_detector(detector, event_type="bursts")
        print(f"Detected {len(catalog)} events via BurstDetector")
    except Exception as e:  # noqa: BLE001
        print(f"Falling back to simulated events (BurstDetector unavailable): {e}")
        df = pd.DataFrame(
            {
                "event_id": [f"e{i:03d}" for i in range(50)],
                "t": np.linspace(float(data.time.values[0]), float(data.time.values[-1]), 50),
                "AP": np.random.RandomState(0).randint(0, int(data.sizes["AP"]), size=50),
                "ML": np.random.RandomState(1).randint(0, int(data.sizes["ML"]), size=50),
                "freq": np.random.RandomState(2).uniform(20.0, 80.0, size=50),
                "value": np.random.RandomState(3).uniform(0.0, 1.0, size=50),
                "label": "sim",
            }
        )
        catalog = EventCatalog(df=df, name="sim")
        state.register_event_catalog("bursts", catalog)

    # Activate event explorer module.
    registry = ModuleRegistry()
    mod = registry.get("event_explorer")
    layout = mod.activate(state) if mod is not None else hv.Div("<b>event_explorer not found</b>")

    app = pn.template.FastListTemplate(
        title="TensorScope v2.6.2: Event Explorer Demo",
        main=[pn.pane.HoloViews(layout, sizing_mode="stretch_both")],
    )
    return app


if __name__ == "__main__":
    main().show()

