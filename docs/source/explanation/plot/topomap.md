# TopoMap (AP×ML scalar heatmap)

`TopoMap` renders an **AP×ML topographic heatmap** of any per-electrode scalar.
Typical inputs include RMS, bad-channel scores, correlation-to-seed, or time-windowed band power.

## API

### From a numpy array

```python
import panel as pn
import numpy as np
from cogpy.core.plot.topomap import TopoMap

pn.extension("bokeh")

rms = np.random.randn(16, 16)  # (AP, ML)
t = TopoMap(
    rms,
    ap_coords=np.linspace(-4, 1, 16),
    ml_coords=np.linspace(-4, 4, 16),
    colormap="viridis",
    symmetric=False,
    title="RMS",
)
t.panel().servable()
```

### From an xarray DataArray

```python
import panel as pn
from cogpy.core.plot.topomap import TopoMap

pn.extension("bokeh")

# da must include AP and ML dims (either order).
t = TopoMap.from_dataarray(da, colormap="rdbu", symmetric=True, title="Z-score")
t.panel().servable()
```

### Update in place

```python
t.update(new_values)  # new_values must be (n_ap, n_ml)
```

### Tap callbacks

```python
def on_tap(info):
    print(info["ap_idx"], info["ml_idx"], info["ap"], info["ml"], info["value"])

t.on_tap(on_tap)
```

## Notes

- `TopoMap` draws electrodes in **physical coordinate space** when you pass `ap_coords` and `ml_coords`. This is intended to compose cleanly with future atlas overlays.
- Use `symmetric=True` for diverging colormaps (e.g. z-scored quantities).

