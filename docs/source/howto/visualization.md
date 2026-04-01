# How to visualize ECoG data

All interactive components live in `cogpy.plot.hv` and require a Bokeh backend.
Call `hv.extension("bokeh")` once at the top of your notebook.

```python
import holoviews as hv
import panel as pn
hv.extension("bokeh")
pn.extension()
```

---

## Stacked multichannel traces

`multichannel_view` accepts `(time, ch)` or `(time, AP, ML)` signals and
renders stacked traces with a linked minimap.

```python
from cogpy.plot.hv import multichannel_view

# Works with flat (time, ch) or grid (time, AP, ML) signals
view = multichannel_view(sig)
view
```

Control the initial visible window with `boundsx`:

```python
view = multichannel_view(sig, boundsx=(0.0, 2.0), title="LFP")
```

The minimap below the traces is a z-scored image; drag the range box to pan,
resize it to zoom.

---

## Spatial grid movie

`grid_movie` animates an `(AP, ML)` image across remaining dimensions (e.g.,
`time` or `freq`). HoloViews generates a slider for each extra dimension.

```python
from cogpy.plot.hv import grid_movie

# sig_grid has dims (time, AP, ML)
movie = grid_movie(sig_grid, cmap="RdBu_r", symmetric=True)
movie
```

For a spectrogram `(freq, AP, ML)`:

```python
movie = grid_movie(spectrogram, x_dim="ML", y_dim="AP", cmap="viridis", symmetric=False)
```

---

## Grid movie with linked time cursor

`grid_movie_with_time_curve` combines a spatial frame (top) with a 1D summary
curve (bottom). Clicking the curve moves the time cursor and updates the image.

```python
from cogpy.plot.hv import grid_movie_with_time_curve

layout = grid_movie_with_time_curve(sig_grid, time_dim="time")
layout
```

To react to the selected time programmatically:

```python
layout, ctrl = grid_movie_with_time_curve(sig_grid, return_controller=True)
ctrl.param.watch(lambda e: print("t =", e.new), "t")
layout
```

For a 4D spectrogram `(time, freq, AP, ML)`, fix the frequency slice with
`indexers`:

```python
layout = grid_movie_with_time_curve(
    spec4d, time_dim="time", indexers={"freq": 80.0}
)
```

---

## Per-electrode scalar heatmap (TopoMap)

`TopoMap` renders a single scalar per electrode as an AP×ML heatmap using
Bokeh directly (no HoloViews required). Typical inputs: RMS power, z-score,
bad-channel flag.

```python
from cogpy.plot.hv.topomap import TopoMap
import numpy as np

# values: shape (n_ap, n_ml)
values = sig.std(dim="time").values  # e.g. per-electrode RMS
tmap = TopoMap(
    values,
    ap_coords=sig.coords["AP"].values,
    ml_coords=sig.coords["ML"].values,
    colormap="viridis",
    title="RMS power",
)
pn.panel(tmap.figure).servable()
```

Use `symmetric=True` for diverging quantities (e.g. z-score):

```python
TopoMap(zscores, symmetric=True, colormap="coolwarm")
```

---

## Interactive orthoslicer (time × frequency × space)

`OrthoSlicerRanger` provides a linked 3-panel view for 4D data
`(time, freq, AP, ML)`: a time–frequency spectrogram (TZ), a spatial
AP×ML frame (XY), and an optional 1D time summary curve.

```python
from cogpy.plot.hv.orthoslicer import OrthoSlicerRanger
import holoviews as hv

# Map xarray dim names to labeled HoloViews Dimensions
dx = ("ML",   hv.Dimension("x", label="Medial-Lateral",    unit="mm"))
dy = ("AP",   hv.Dimension("y", label="Anterior-Posterior", unit="mm"))
dt = ("time", hv.Dimension("t", label="Time",               unit="s"))
dz = ("freq", hv.Dimension("z", label="Frequency",          unit="Hz"))

# Optional 1D signal drawn under the TZ view
summary = spec4d.mean(dim=("freq", "AP", "ML"))

slicer = OrthoSlicerRanger(
    spec4d,
    rangeslider_sig=summary,
    dt=dt, dz=dz, dy=dy, dx=dx,
)
slicer.panel_app().servable()
```

Click the TZ panel to select a time; click XY to select a spatial location.
The summary curve at the bottom controls the visible time window.

---

## Composing views with a shared controller

Use `add_time_hair` to add a clickable vertical cursor to any `hv.Curve`,
then share the returned controller across other views.

```python
from cogpy.plot.hv import add_time_hair, grid_movie_linked_to_controller

curve = hv.Curve(summary_da, kdims=["time"])
curve_with_hair, ctrl = add_time_hair(curve, time_kdim="time", return_controller=True)

# A grid movie whose frame follows the cursor
linked_movie = grid_movie_linked_to_controller(sig_grid, controller=ctrl)

(linked_movie + curve_with_hair).cols(1)
```

To add a crosshair on an AP×ML image that tracks a selected channel:

```python
from cogpy.plot.hv.xarray_hv import (
    standardize_time_channel_with_geometry,
    selected_channel_curve,
    apml_crosshair_from_channel,
    bind_apml_tap_to_channel_controller,
)
import param

sig_tc, ap_ch, ml_ch = standardize_time_channel_with_geometry(sig_grid)

ch_ctrl = param.Parameterized()
ch_ctrl.param._add_parameter("value", param.Integer(default=0))

topo_img = grid_movie(sig_grid.isel(time=0))
tap = bind_apml_tap_to_channel_controller(topo_img, ch_controller=ch_ctrl, ap_ch=ap_ch, ml_ch=ml_ch)
crosshair = apml_crosshair_from_channel(ch_controller=ch_ctrl, ap_ch=ap_ch, ml_ch=ml_ch)
ch_curve   = selected_channel_curve(sig_tc, ch_controller=ch_ctrl)

(topo_img * crosshair + ch_curve).cols(1)
```

---

## Serving as a Panel app

Any layout can be served as a standalone dashboard:

```python
pn.panel(layout).show()        # opens browser tab (blocking)
pn.panel(layout).servable()    # use inside `panel serve notebook.ipynb`
```

## See also

- {doc}`/explanation/architecture` — compute vs visualization boundary
- {doc}`/api/burst` — low-level burst detection for overlay data
