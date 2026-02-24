# Channel Features

## Status

The channel-feature domain currently has both canonical and legacy surfaces.

- Canonical (new code):
  - `src/cogpy/core/preprocess/badchannel/channel_features.py`
  - `src/cogpy/core/preprocess/badchannel/pipeline.py`
  - `src/cogpy/core/preprocess/badchannel/spatial.py`
  - `src/cogpy/core/preprocess/badchannel/badlabel.py`
- Legacy (compatibility only):
  - `src/cogpy/core/preprocess/channel_feature_functions.py` (deprecated)
  - `src/cogpy/core/preprocess/channel_feature.py` (legacy)
  - `src/cogpy/core/preprocess/detect_bads.py` (legacy)

## Canonical stack

### 1) Raw per-channel features

`badchannel/channel_features.py` defines raw temporal features over arrays shaped like `(..., time)`, e.g.:

- `relative_variance`
- `deviation`
- `standard_deviation`
- `amplitude`
- `time_derivative`
- `hurst_exponent`
- `kurtosis`
- `noise_to_signal` (with `snr` alias)

For xarray workflows, use `extract_channel_features_xr(...)`.

### 2) Sliding-window utilities

Canonical sliding logic is in `src/cogpy/core/utils/sliding_core.py`.

`badchannel/channel_features.py` uses:

- `running_reduce_xr(...)`
- `running_blockwise_xr(...)`

`badchannel/pipeline.py` uses shared window index helpers:

- `window_onsets(...)`
- `window_centers_idx(...)`

This avoids re-implementing rolling/window indexing in preprocess modules.

### 3) Spatial normalization and feature maps

`badchannel/pipeline.py` orchestrates per-window feature map construction:

- `compute_feature_maps_for_window(...)` for normalized spatial features
- `compute_raw_feature_maps_for_window(...)` for raw maps
- `compute_features_sliding(...)` for stacked `(feature, AP, ML, time_win)` output

Normalization primitives are in `badchannel/spatial.py`:

- `normalize_ratio`
- `normalize_difference`
- `normalize_robust_z`
- neighborhood statistics from adjacency

### 4) Outlier labeling

`badchannel/badlabel.py` is the canonical DBSCAN outlier path:

- `dbscan_outliers(...)`
- `grouped_dbscan_outliers(...)`

## Notes on legacy modules

Legacy modules remain importable for compatibility, but should not be used for new implementations.

- `channel_feature_functions.py` contains deprecated duplicates and older helpers.
- `channel_feature.py` / `detect_bads.py` retain older orchestration patterns and APIs.

When migrating code:

1. Move raw feature logic to `badchannel/channel_features.py`.
2. Keep spatial normalization in `badchannel/spatial.py` + `badchannel/pipeline.py`.
3. Use `sliding_core.py` for all rolling/window primitives.
