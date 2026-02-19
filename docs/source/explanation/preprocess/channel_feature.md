# Channel Features (current implementation)

This page documents the *current* behavior of `cogpy.core.preprocess.channel_feature` and highlights common performance pitfalls when computing rolling-window (“running”) channel features with Dask/Xarray.

## What this module provides

`src/cogpy/core/preprocess/channel_feature.py` exposes two parallel APIs:

- `ChannelFeatures` (class-based): constructs adjacency/footprint once and offers methods to compute features.
- Module-level functions: `transform_feature(...)` / `transform_dask(...)` for convenience (no persistent object).

The feature set is defined by `FEATURE_NAMES` and corresponds to functions in `channel_feature_functions`:

- `anticorrelation` (depends on neighborhood adjacency)
- `relative_variance`
- `deviation`
- `amplitude`
- `time_derivative`
- `hurst_exponent`
- `temporal_mean_laplacian`

Optional post-processing:

- local robust z-scoring via `local_robust_zscore` using a 2D boolean `footprint`.

All rolling-window logic is delegated to utilities in `src/cogpy/core/utils/sliding.py`, primarily:

- `rolling_win(...)` (construct strided centered rolling windows)
- `running_measure(...)` / `running_measure_sane(...)` (apply a per-window function via `xr.apply_ufunc`)

## Data model and dimensions

The code assumes an input `xarray.DataArray` shaped like:

- spatial dims: `AP` (rows) and `ML` (cols) by default
- time dim: `time` by default

Most “window measures” are written to accept an array shaped `(AP, ML, window)` for each window.

## How features are computed (two patterns)

### Pattern A: one feature at a time (repeats rolling work)

`ChannelFeatures.transform_dask(...)` loops through features and calls `ChannelFeatures.transform_feature(...)`.
That ultimately calls `running_measure(...)` separately for each feature.

Similarly, the module-level `transform_dask(...)` loops features and calls the module-level `transform_feature(...)`, which also calls `running_measure(...)` per feature.

**Implication:** when using Dask, this creates *one rolling-window/apply_ufunc graph per feature*, even though all features use the same rolling windows.

### Pattern B: compute all features in one pass (single rolling graph)

`ChannelFeatures.transform(...)` calls `running_measure(self.measure, ...)` once, where `self.measure(...)` computes all features for a window and returns a stacked array shaped `(feature, AP, ML)`.

**Implication:** this builds a *single* rolling-window/apply_ufunc graph for all features, which typically reduces graph size and scheduler overhead.

## Performance pitfalls to be aware of

### 1) “Per-feature Dask graphs” multiply overhead

When you loop features and call `running_measure(...)` per feature, you usually pay:

- repeated rolling-window construction in the task graph
- repeated scheduling overhead (N graphs instead of 1)
- repeated overlap handling along the time axis (rolling needs halo/overlap)

This can look like “Dask loads the data again”, especially if you compute each feature separately.

**Symptom:** computing a Dataset feature-by-feature is much slower than stacking features in one call, even if the math per window is the same.

### 2) Writing/Computing feature-by-feature can re-trigger reads

If you call `.compute()` or `.to_zarr()` on each feature variable separately (or build a loop that triggers compute per variable), Dask may re-read inputs and recompute upstream tasks each time, unless you explicitly persist/cache shared intermediates.

The helper `save_features(...)` writes variables sequentially; whether this recomputes upstream depends on how the input Dataset was constructed (single shared graph vs separate graphs).

### 3) Chunking along `time` dominates rolling performance

Rolling windows are expensive when time chunks are too small relative to `window_size`, or when chunk boundaries force lots of overlap reads.

`running_measure_sane(...)` in `src/cogpy/core/utils/sliding.py` exists specifically to rechunk the rolling axis in a “window-aware” way before constructing windows.

**Symptom:** extremely large task graphs, high scheduler overhead, or very slow progress even on modest data.

### 4) Module-level convenience functions redo setup work

The module-level `transform_feature(...)` currently recreates several objects each call:

- `adjacency_matrix((nrows, ncols))` is rebuilt for `anticorrelation`
- `footprint` is (re)derived via `ensure_footprint(...)`

This overhead is small compared to full Dask computations, but can matter when:

- you compute many times on small arrays (interactive use)
- you loop features repeatedly (it compounds)

The class-based API (`ChannelFeatures.__init__`) computes `adj` and `footprint` once and reuses them.

### 5) Hidden dtype/precision and memory implications

Some paths cast to `float32` (e.g., `compute_features(...)` in the class). Others rely on `output_dtype` passed into `running_measure*`.

For large grids/windows, intermediate arrays (especially rolling windows) can be memory-heavy; chunk sizes and dtype materially change memory pressure.

## Practical guidance (without changing code)

- Prefer `ChannelFeatures.transform(...)` (single-pass stacked features) when you need *all* features.
- Use the class-based API (`ChannelFeatures`) when computing repeatedly on similarly shaped signals; it avoids repeated adjacency/footprint construction.
- If you must compute a subset of features, expect `transform_feature(...)` / `transform_dask(...)` to be slower under Dask due to repeated rolling graphs.
- Consider using `sane=True` on the module-level `transform_feature(...)` / `transform_dask(...)` to route through `running_measure_sane(...)` (better rolling-axis chunking), especially for large time series.

## Where to look in the code

- Class API: `src/cogpy/core/preprocess/channel_feature.py`
  - `ChannelFeatures.transform(...)` (single-pass, stacked features)
  - `ChannelFeatures.transform_dask(...)` + `ChannelFeatures.transform_feature(...)` (per-feature loop)
- Convenience functions: `src/cogpy/core/preprocess/channel_feature.py`
  - `transform_dask(...)` and `transform_feature(...)` (per-feature loop)
- Rolling utilities: `src/cogpy/core/utils/sliding.py`
  - `running_measure(...)` and `running_measure_sane(...)`
  - `rolling_win(...)`

