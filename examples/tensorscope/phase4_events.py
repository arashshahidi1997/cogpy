"""
Phase 4 Interactive Events Demo

Shows event system in action:
- Event table with synthetic data
- Click row to jump to event
- Prev/Next navigation

Run with:
    conda run -n cogpy panel serve code/lib/cogpy/examples/tensorscope/phase4_events.py --show
"""

import numpy as np
import pandas as pd
import panel as pn

from cogpy.core.plot.tensorscope import TensorScopeApp
from cogpy.core.plot.tensorscope.events import EventStream, EventStyle
from cogpy.core.plot.tensorscope.layers.events import EventTableLayer
from cogpy.datasets.entities import example_ieeg_grid

pn.extension("tabulator")

print("Loading data...")
data = example_ieeg_grid(mode="small")

print("Creating synthetic events...")
np.random.seed(42)
n_events = 30
event_times = np.sort(np.random.uniform(0, float(data.time.values[-1]), n_events))
event_df = pd.DataFrame(
    {
        "event_id": range(n_events),
        "t": event_times,
        "label": np.random.choice(["burst", "ripple", "spindle"], n_events),
        "channel": np.random.randint(0, 64, n_events),
        "amplitude": np.random.uniform(2.0, 5.0, n_events),
        "duration": np.random.uniform(0.05, 0.3, n_events),
    }
)

print("Creating TensorScope app...")
app = (
    TensorScopeApp(data, title="TensorScope Phase 4: Events")
    .with_layout("default")
    .add_layer("timeseries")
    .add_layer("spatial_map")
    .add_layer("selector")
    .add_layer("processing")
    .add_layer("navigator")
)

bursts = EventStream("bursts", event_df, style=EventStyle(color="#ff5555", marker="circle"))
app.state.register_events("bursts", bursts)

# Add event table panel to sidebar by directly constructing the layer
event_table = EventTableLayer(app.state, "bursts")

template = app.build()
template.sidebar.append(event_table.panel())
template.servable()

print(f"✅ App created with {len(event_df)} events!")
print(f"   Events registered: {app.state.event_registry.list()}")
print("✅ Phase 4 events demo ready! Click an event row to jump to its time.")

