# TensorScope User Guide

TensorScope is a neurophysiology visualization application built on Panel +
HoloViews. It provides a centralized, serializable state (`TensorScopeState`)
and a set of thin “layers” that wrap existing `cogpy.core.plot` components.

## Getting Started

### Installation

TensorScope is part of `cogpy`.

For visualization support, install the `viz` extra:

```bash
pip install "cogpy[viz]"
```

### Quick start (Python)

```python
from cogpy.datasets.entities import example_ieeg_grid
from cogpy.core.tensorscope import TensorScopeApp

data = example_ieeg_grid(mode="small")
app = (
    TensorScopeApp(data, title="TensorScope")
    .with_layout("default")
    .add_layer("timeseries")
    .add_layer("spatial_map")
    .add_layer("selector")
    .add_layer("processing")
    .add_layer("navigator")
)

app.servable()
```

### Quick start (CLI)

```bash
tensorscope presets
tensorscope modules
tensorscope serve data.nc --layout default --port 5008 --show
tensorscope serve data.nc --module psd_explorer --port 5008 --show
```

## Loading Data

TensorScope expects an `xarray.DataArray` with a time dimension.

### Grid data

Preferred schema for electrode grids:

- dims: `("time", "AP", "ML")`
- time: strictly increasing
- `AP`, `ML`: any numeric coordinates are accepted; TensorScope normalizes them
  to integer indices (preserving original coords as `AP_src`, `ML_src`).

### Flat data

Supported schema:

- dims: `("time", "channel")`
- requires per-channel `AP(channel)` and `ML(channel)` coords (or a MultiIndex
  channel with `AP`/`ML` levels) so the data can be reshaped into a dense grid.

TensorScope validates and normalizes inputs during state initialization.

## Navigation

### Time cursor

The authoritative cursor time lives in `TimeHair` (owned by state).

```python
state.current_time = 5.3
print(state.current_time)
```

### Time window

The visible time window is managed by `TimeWindowCtrl`:

```python
state.time_window.set_window(2.0, 8.0)
state.time_window.recenter(5.0, width_s=3.0)
print(state.time_window.window)
```

## Channel Selection

Selection is managed by `ChannelGrid` (owned by state) and exposed via
delegation properties:

```python
state.channel_grid.select_cell(2, 3)
print(state.selected_channels)       # frozenset[(AP, ML)]
print(state.selected_channels_flat)  # row-major indices
```

## Processing

`ProcessingChain` holds display-time transformations (e.g. filtering, z-score).
Use the UI controls layer, or set parameters programmatically:

```python
state.processing.cmr = True
window = state.processing.get_window(2.0, 5.0)
```

## Events

Events are discrete occurrences in time (ripples, bursts, annotations). They
are stored in an `EventStream` and managed by the state’s `EventRegistry`.

```python
import pandas as pd
from cogpy.core.tensorscope.events import EventStream

df = pd.DataFrame({"event_id": [0, 1], "t": [1.5, 3.2], "label": ["burst", "ripple"]})
stream = EventStream("bursts", df)
state.register_events("bursts", stream)

state.jump_to_event("bursts", event_id=1)
state.next_event("bursts")
state.prev_event("bursts")
```

To browse events interactively, add the event table layer:

```python
app.add_layer("event_table")
```

## Multi-Modal Data

TensorScope can manage multiple modalities (grid LFP, spectrograms, spikes…).

```python
from cogpy.core.tensorscope.data.modalities import SpectrogramModality

app = TensorScopeApp(lfp_data)
app.state.register_modality("spectrogram", SpectrogramModality(spec_data))
app.state.set_active_modality("spectrogram")
```

Notes:
- `state.active_modality` is a `param.String` suitable for UI binding.
- `state.set_active_modality(name)` updates both the registry and the param.

## Layouts

Built-in layout presets:
- `default`
- `spatial_focus`
- `timeseries_focus`

```python
app = TensorScopeApp(data).with_layout("spatial_focus")
```

## Session Management

Save:

```python
import json
session = app.to_session()
json.dump(session, open("session.json", "w"), indent=2)
```

Load:

```python
import json
from cogpy.core.tensorscope import TensorScopeApp

session = json.load(open("session.json"))
app2 = TensorScopeApp.from_session(session, data_resolver=lambda: data)
```

## Troubleshooting

See:
- `tensorscope-issues.md` (common pitfalls and fixes)
- examples in `code/lib/cogpy/examples/tensorscope/`
