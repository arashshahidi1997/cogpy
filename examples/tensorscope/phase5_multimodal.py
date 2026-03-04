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
    .add_layer("spectrogram")
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
    print("\n=== Modality Switch ===")
    print(f"From: {app.state.data_registry.get_active_name()}")
    print(f"To: {event.new}")

    app.state.set_active_modality(event.new)

    print(f"State active_modality: {app.state.active_modality}")
    print(f"Registry active: {app.state.data_registry.get_active_name()}")
    print("=" * 25 + "\n")


modality_selector.param.watch(update_modality, "value")

def make_info(active_mod: str) -> str:
    modalities_list = app.state.data_registry.list()

    info_text = "## Multi-Modal Demo\n\n"
    info_text += "**Available modalities:**\n"
    for mod_name in modalities_list:
        if mod_name == active_mod:
            info_text += f"- **{mod_name}** (ACTIVE)\n"
        else:
            info_text += f"- {mod_name}\n"

    info_text += "\n**Current views:**\n"
    info_text += "- Spatial: Grid RMS\n"
    info_text += "- Timeseries: LFP traces\n"
    if active_mod == "spectrogram":
        info_text += "- Spectrogram: Time-frequency power\n"
    else:
        info_text += "- Spectrogram: Inactive\n"

    info_text += "\n**Try:**\n"
    info_text += "- Switch modality in dropdown\n"
    info_text += "- Navigate time with the player\n"

    return info_text


info_pane = pn.bind(
    lambda mod: pn.pane.Markdown(make_info(str(mod)), styles={"color": TEXT}),
    app.state.param.active_modality,
)

# Build template with custom layout (avoid preset overlap).
template = pn.template.FastGridTemplate(
    title="TensorScope Phase 5: Multi-Modal",
    theme="dark",
    sidebar_width=320,
    row_height=100,
)

sidebar_objects = [
    info_pane,
    modality_selector,
    pn.layout.Divider(),
    app._panels.get("selector"),
    app._panels.get("processing"),
]
template.sidebar.append(pn.Column(*[o for o in sidebar_objects if o is not None]))

# Manually arrange panels in grid:
# Top: Spatial map (left) + Spectrogram (right)
# Bottom: Timeseries (full width)
spatial_panel = app._panels.get("spatial_map")
timeseries_panel = app._panels.get("timeseries")
spec_panel = app._panels.get("spectrogram")

if spatial_panel is not None:
    template.main[0:5, 0:6] = spatial_panel

if spec_panel is not None:
    template.main[0:5, 6:12] = spec_panel

if timeseries_panel is not None:
    template.main[5:10, 0:12] = timeseries_panel  # Full width at bottom

print(f"✅ App created with {len(app.state.data_registry.list())} modalities!")
print(f"   Modalities: {app.state.data_registry.list()}")

template.servable()
