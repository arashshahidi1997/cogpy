"""
Phase 5 Multi-Modal Interactive Demo

Shows multiple data modalities:
- LFP grid data
- Synthetic spectrogram
- Switch between modalities

Run with:
    conda run -n cogpy panel serve code/lib/cogpy/examples/tensorscope/phase5_multimodal.py --show
"""

from __future__ import annotations

import numpy as np
import panel as pn
import xarray as xr

from cogpy.core.plot.theme import TEXT
from cogpy.core.plot.tensorscope import TensorScopeApp
from cogpy.core.plot.tensorscope.data.modalities import SpectrogramModality
from cogpy.datasets.entities import example_ieeg_grid


pn.extension("tabulator")

print("Loading LFP data...")
lfp_data = example_ieeg_grid(mode="small")

print("Creating synthetic spectrogram...")
n_time = 200
n_freq = 60
spec_data = xr.DataArray(
    np.random.randn(n_time, n_freq, 8, 8),
    dims=("time", "freq", "AP", "ML"),
    coords={
        "time": np.linspace(0, float(lfp_data.time.values[-1]), n_time),
        "freq": np.logspace(0, 2, n_freq),
        "AP": np.arange(8),
        "ML": np.arange(8),
    },
)

print("Creating TensorScope app...")
app = (
    TensorScopeApp(lfp_data, title="TensorScope Phase 5: Multi-Modal")
    .with_layout("default")
    .add_layer("timeseries")
    .add_layer("spatial_map")
    .add_layer("selector")
    .add_layer("processing")
)

spec_modality = SpectrogramModality(spec_data)
app.state.register_modality("spectrogram", spec_modality)

modality_selector = pn.widgets.Select(
    name="Active Modality",
    options=app.state.data_registry.list(),
    value=app.state.data_registry.get_active_name(),
)


def update_modality(event):
    app.state.set_active_modality(event.new)
    print(f"Switched to modality: {event.new}")


modality_selector.param.watch(update_modality, "value")

info = pn.pane.Markdown(
    "## Multi-Modal Demo\n\n"
    "**Available modalities:**\n"
    f"- grid_lfp: {lfp_data.shape}\n"
    f"- spectrogram: {spec_data.shape}\n\n"
    "**Try:**\n"
    "- Switch modality in dropdown\n"
    "- Navigate time with the player\n",
    styles={"color": TEXT},
)

template = app.build()
template.sidebar.insert(0, pn.Column(info, modality_selector))

print(f"✅ App created with {len(app.state.data_registry.list())} modalities!")
print(f"   Modalities: {app.state.data_registry.list()}")

template.servable()

