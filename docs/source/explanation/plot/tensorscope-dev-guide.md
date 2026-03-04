# TensorScope Developer Guide

This guide is for developers extending TensorScope or contributing layers,
modalities, and tooling.

## Architecture (v1.0)

Core components:

- `TensorScopeState`: authoritative, serializable state (owns controllers)
- `TensorLayer`: base class for visualization layers (lifecycle + watchers)
- `LayerManager`: registers/instantiates layers and disposes them
- `LayoutManager`: layout presets for `FastGridTemplate`
- `TensorScopeApp`: composition root (state + managers + builder API)

Data flow:

```
User input → state param/controller update → param watchers → layer refresh → view render
```

## Implementing a Custom Layer

### 1) Subclass `TensorLayer`

Use `self._watch(...)` so watchers are tracked and cleaned up by `dispose()`.

```python
import panel as pn

from cogpy.core.plot.tensorscope.layers.base import TensorLayer


class MyLayer(TensorLayer):
    def __init__(self, state):
        super().__init__(state)
        self.layer_id = "my_layer"
        self.title = "My Layer"

        self._pane = pn.pane.Markdown("", sizing_mode="stretch_both")
        self._watch(state, self._on_time, "active_modality")
        self._refresh()

    def _on_time(self, _event=None):
        self._refresh()

    def _refresh(self):
        self._pane.object = f"Active modality: {self.state.active_modality}"

    def panel(self):
        return self._pane
```

### 2) Register the layer type

```python
from cogpy.core.plot.tensorscope.layers.manager import LayerSpec

app.layer_manager.register(
    LayerSpec(
        layer_id="my_layer",
        title="My Layer",
        factory=lambda s: MyLayer(s),
        description="Example custom layer",
        layer_type="generic",
    )
)
```

### 3) Use it

```python
app.add_layer("my_layer")
```

## Adding a New Modality

Implement `DataModality`:

```python
from cogpy.core.plot.tensorscope.data.modality import DataModality


class MyModality(DataModality):
    def __init__(self, data):
        self.data = data

    def time_bounds(self):
        return (float(self.data.time.values[0]), float(self.data.time.values[-1]))

    def get_window(self, t0, t1):
        return self.data.sel(time=slice(float(t0), float(t1)))

    @property
    def sampling_rate(self):
        return 1000.0

    @property
    def modality_type(self):
        return "my_modality"
```

Register + activate:

```python
state.register_modality("my_modality", MyModality(my_data))
state.set_active_modality("my_modality")
```

## Testing

Run TensorScope tests:

```bash
python -m pytest code/lib/cogpy/tests/core/plot/tensorscope/ -v
```

Run benchmarks (explicit):

```bash
python -m pytest code/lib/cogpy/tests/core/plot/tensorscope/benchmarks.py -m benchmark -v -s
```

## Performance Guidelines

Do:
- use windowed processing (avoid full-recording operations in callbacks)
- keep layers thin (wrap existing components instead of reimplementing)
- call `dispose()` for every layer instance removed

Avoid:
- import-time backend initialization in reusable modules
- large copies in interactive callbacks
- long blocking work on the UI thread

## Release Checklist (Phase 6)

1. Add/update docs (user + dev guides)
2. Ensure examples run
3. Run unit tests + optional benchmarks
4. Validate session serialization round-trip
5. Add CLI entry point

