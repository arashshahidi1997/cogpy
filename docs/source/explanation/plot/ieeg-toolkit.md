# iEEG Visualization Toolkit

Four modules for interactive visualization of iEEG recorded from a 2D electrode grid.

---

## `channel_grid.py` — ChannelGrid

Pure selection logic. No display code. Knows the grid shape and computes which channels are selected based on the current mode.

**Modes**

- `row` — all channels in one AP row
- `column` — all channels in one ML column
- `sparse` — strided subgrid covering the full extent (`stride=2` on 16×16 → 64 evenly spaced channels)
- `neighborhood` — filled Chebyshev square around a center electrode (`radius=2` → 5×5)
- `manual` — individual cell toggling

**Key outputs**

- `grid.selected` — `frozenset` of `(ap, ml)` int tuples. Watch this downstream.
- `grid.flat_indices` — `list[int]`, row-major: `ap * n_ml + ml`
- `grid.as_array` — `(n_ap, n_ml)` bool mask

```python
grid = ChannelGrid(n_ap=16, n_ml=16)
grid.select_row(3)
grid.select_sparse(stride=2, offset=0)
grid.select_neighborhood(ap=8, ml=8, radius=2)
grid.toggle_manual(3, 5)

grid.param.watch(lambda e: print(e.new), "selected")
```

> **Coordinate convention**: all indices are integers 0..N-1. Normalize physical AP/ML coords first with `normalize_coords_to_index()`.

---

## `channel_grid_widget.py` — ChannelGridWidget

Bokeh heatmap of the electrode grid plus Panel mode controls. Wraps a `ChannelGrid` — clicking cells updates the grid, grid state changes redraw the widget.

Optionally accepts a `cell_values` array (e.g. per-channel RMS) to show signal amplitude as background brightness on unselected cells, and an `atlas_image` for anatomical context.

```python
rms = sig.std(dim="time").transpose("AP", "ML").values   # (n_ap, n_ml)
w   = ChannelGridWidget.from_grid(grid, cell_values=rms)
w.panel()   # returns pn.Column
w.grid      # the ChannelGrid inside
```

---

## `multichannel_viewer.py` — MultichannelViewer

Stacked trace viewer. Takes numpy directly — no xarray, no grid awareness. Call `show_channels()` to update which channels are displayed; the plot updates in place without rebuilding.

Uses a fixed-size `hv.NdOverlay` internally: always `max_channels` slots, inactive ones are empty with `alpha=0`. This prevents HoloViews from leaving ghost traces when the selection changes.

```python
viewer = MultichannelViewer(sig_z, t_vals, ch_labels, max_channels=32)
viewer.show_channels([0, 1, 2, 3])
viewer.panel().servable()
```

---

## `ieeg_viewer.py` — ieeg_viewer / IEEGViewer

Thin integration layer. Accepts an xarray DataArray, handles z-scoring and Dask materialisation, builds a `MultichannelViewer`, and optionally wires a `ChannelGrid` so selection changes drive the viewer. Includes an Apply button to batch rapid selection changes into a single render.

```python
# Standalone
viewer = ieeg_viewer(sig_tc)
viewer.panel().servable()

# Grid-wired
viewer = ieeg_viewer(sig_tc, channel_grid=grid, n_ml=16)
pn.Row(w.panel(), viewer.panel()).servable()
```

---

## Full example

```python
import panel as pn
from cogpy.core.plot.hv.xarray_hv import normalize_coords_to_index
from cogpy.datasets.tensor import example_ieeg
from cogpy.core.plot.hv.channel_grid import ChannelGrid
from cogpy.core.plot.hv.channel_grid_widget import ChannelGridWidget
from cogpy.core.plot.hv.ieeg_viewer import ieeg_viewer

pn.extension("bokeh")

sig      = example_ieeg()                                      # (time, ML, AP)
sig_norm = normalize_coords_to_index(sig, ("AP", "ML"))
sig_tc   = sig_norm.transpose("time", "AP", "ML").stack(channel=("AP", "ML"))
n_ap, n_ml = sig_norm.sizes["AP"], sig_norm.sizes["ML"]

rms  = sig_norm.std(dim="time").transpose("AP", "ML").values
grid = ChannelGrid(n_ap=n_ap, n_ml=n_ml)
w    = ChannelGridWidget.from_grid(grid, cell_values=rms)

viewer = ieeg_viewer(sig_tc, channel_grid=grid, n_ml=n_ml, initial_window_s=5)

pn.Row(w.panel(), viewer.panel()).servable()
```

---

## Demo apps (servable entrypoints)

These are thin “glue” apps intended for GUI development and manual testing.

### iEEG grid + traces

```python
import panel as pn
from cogpy.core.plot.hv.ieeg_toolkit import ieeg_toolkit_app

pn.extension("bokeh")
ieeg_toolkit_app(mode="small", seed=0).servable()
```

### 4D spectrogram + bursts navigator

```python
import panel as pn
from cogpy.core.plot._legacy.spectrogram_bursts_app import spectrogram_bursts_app

pn.extension()
spectrogram_bursts_app(mode="small", seed=0, kind="toy").servable()
```

---

## Future directions

- **Signal-driven selection** — top-N by variance in the current window, or most correlated with a seed channel
- **Sorting** — reorder selected channels by AP, ML, variance, or correlation
- **Atlas placement** — proper asymmetric AP extent for `atlas_mode="full"`
- **Bad channel detection** — flag high-kurtosis electrodes on the grid widget
- **Linked views** — shared time cursor between trace viewer, spectrogram, and AP×ML topomap
- **Lazy loading** — materialise only the visible time window for long recordings

---

## Atlas overlay (typed)

`ChannelGridWidget` supports passing an `AtlasImageOverlay` (image + extents) so the placement metadata travels with the image.

```python
import numpy as np
from PIL import Image

from cogpy.datasets.schemas import AtlasImageOverlay
from cogpy.core.plot.hv.channel_grid import ChannelGrid
from cogpy.core.plot.hv.channel_grid_widget import ChannelGridWidget

atlas = np.array(Image.open(\"docs/assets/atlas/dorsal-cortex.png\").convert(\"RGBA\"), dtype=np.uint8)
overlay = AtlasImageOverlay(
    image=atlas,
    ap_extent=(-4.0, 1.0),
    ml_extent=(-4.0, 4.0),
    bl_distance=7.5,
)

grid = ChannelGrid(n_ap=16, n_ml=16)
w = ChannelGridWidget.from_grid(
    grid,
    ap_coords=np.linspace(-4, 1, 16),
    ml_coords=np.linspace(-4, 4, 16),
    atlas_overlay=overlay,
)
w.panel().servable()
```
