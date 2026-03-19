---
title: Channel Features via sliding_core (experiment)
---

# Channel features via `cogpy.utils.sliding_core` (experiment)

This notebook prototypes the **channel feature** computations from
`cogpy.preprocess.channel_feature` using the **NumPy-only** sliding-window
helpers in `cogpy.utils.sliding_core`.

The goal is to validate that:

- sliding windows can be represented as **non-copying views** (`as_strided`)
- per-window feature extraction can be expressed as `running_blockwise(...)`
- we can compute a stacked feature tensor shaped like `(feature, AP, ML, window)`

This is **not** a refactor of the production code. It’s an in-memory prototype
to evaluate whether a reimplementation is worth pursuing.

## Key differences vs the current module

- `sliding_core` operates on **NumPy arrays only** (eager, in-memory).
- The current `channel_feature` module primarily targets **xarray/Dask** to keep
  computations lazy and chunked.
- Window alignment differs:
  - `sliding_core.sliding_window` makes **left-aligned windows** at starts
    `[0, step, 2*step, ...]`.
  - `cogpy.utils.sliding.rolling_win` uses **centered windows** and then
    trims to valid center positions.

## Imports and synthetic data

We’ll create a toy `(AP, ML, time)` signal and compute several channel features
per window.

```{code-cell} python
import numpy as np

from cogpy.preprocess.channel_feature_functions import (
    anticorrelation,
    relative_variance,
    deviation,
    amplitude,
    time_derivative,
    hurst_exponent,
    temporal_mean_laplacian,
    local_robust_zscore,
)
from cogpy.utils.grid_neighborhood import adjacency_matrix, make_footprint
from cogpy.utils.sliding_core import sliding_window, running_blockwise

rng = np.random.default_rng(0)

AP, ML, T = 8, 8, 4000
fs = 1_000.0
t = np.arange(T) / fs

# Toy ECoG-like: smooth oscillation + noise per channel
base = (np.sin(2 * np.pi * 10 * t) + 0.25 * np.sin(2 * np.pi * 3 * t)).astype(np.float32)
xsig = base[None, None, :] + 0.1 * rng.standard_normal((AP, ML, T)).astype(np.float32)
xsig.shape
```

## Sliding windows are views (no copy)

```{code-cell} python
WINDOW = 512
STEP = 64

w = sliding_window(xsig, window_size=WINDOW, window_step=STEP, axis=2)
w.shape, np.shares_memory(w, xsig)
```

`w` has shape `(AP, ML, n_windows, window_size)` and does not copy `xsig`.

## Defining the per-window feature function

`running_blockwise(...)` will call our function with a block shaped
`(AP, ML, window_size)` and stack the outputs across windows.

To mimic the “stacked feature” output in `ChannelFeatures.transform(...)`, we
return an array shaped `(n_features, AP, ML)` for each window.

```{code-cell} python
footprint = make_footprint(rank=2, connectivity=1, niter=2)
adj = adjacency_matrix((AP, ML), footprint=footprint, exclude=True)

feature_fns = [
    lambda a: anticorrelation(a, adj=adj),
    relative_variance,
    deviation,
    amplitude,
    time_derivative,
    hurst_exponent,
    temporal_mean_laplacian,
]
feature_names = [
    "anticorrelation",
    "relative_variance",
    "deviation",
    "amplitude",
    "time_derivative",
    "hurst_exponent",
    "temporal_mean_laplacian",
]

def feature_stack(block: np.ndarray, *, zscore: bool = True) -> np.ndarray:
  out = np.empty((len(feature_fns),) + block.shape[:2], dtype=np.float32)
  for i, fn in enumerate(feature_fns):
      val = fn(block).astype(np.float32, copy=False)
      if zscore:
          val = local_robust_zscore(val, footprint=footprint).astype(np.float32, copy=False)
      out[i] = val
  return out

# quick shape check on a single window
feature_stack(w[:, :, 0, :]).shape
```

## Running feature extraction across windows

```{code-cell} python
feat = running_blockwise(
    xsig,
    window_size=WINDOW,
    window_step=STEP,
    func=lambda block: feature_stack(block, zscore=True),
    axis=2,  # time axis
)
feat.shape
```

By design, `running_blockwise` stacks over windows first, so the shape is:
`(n_windows, n_features, AP, ML)`.

To match the typical `channel_feature` convention we can transpose to:
`(feature, AP, ML, window)`.

```{code-cell} python
feat_t = np.moveaxis(feat, 0, -1)  # (n_features, AP, ML, n_windows)
feat_t.shape
```

## Window time coordinates (left-aligned vs centered)

With `sliding_core`, windows start at indices `[0, STEP, 2*STEP, ...]`.
If you want to label each window by its **center sample**, use:

```{code-cell} python
n_windows = feat.shape[0]
start_idx = np.arange(n_windows) * STEP
center_idx = start_idx + (WINDOW // 2)
center_time = center_idx / fs
center_time[:5]
```

If we pursue a reimplementation, we need to decide whether the “centered window”
semantics of the current xarray path should be preserved (and how to treat edge
windows), or whether we switch to left-aligned windows for speed/simplicity.

## Feasibility notes for a future reimplementation

This prototype shows the core computation is expressible with `sliding_core`.
However, the production module currently targets xarray + Dask:

- `sliding_core` itself is NumPy-only, so integrating it with Dask would require
  `map_blocks`/`map_overlap` (to supply halo for windows) or an xarray wrapper
  that carefully aligns chunks and window boundaries.
- The current module supports lazy computation and chunked execution; a pure
  NumPy approach would require loading full arrays (or managing chunking
  manually).

If you like the shapes/semantics in this notebook, the next step would be a
small Dask-backed experiment using `map_overlap` to apply `feature_stack` on
overlapped time chunks and stitch the window outputs together.

