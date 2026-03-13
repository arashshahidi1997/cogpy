# PSD Explorer Module Specification v2.8.0

## Version
- Version: 2.8.0
- Date: 2026-03-05
- Depends on: v2.2 (ViewSpec/ViewFactory modules), v2.6.x (signal + linking primitives)

## Overview

PSD Explorer adds a HoloViews-native module for power spectral density exploration:

- time-windowed trace stack (context around current cursor time)
- PSD heatmap (freq × channel)
- average PSD curve (mean ± std across channels)
- spatial PSD map (AP×ML) at selected frequency

Filtering is supported via simple bandpass/highpass/lowpass transforms applied prior to PSD computation.

## Implementation

### PSD utilities

**File:** `cogpy/core/spectral/psd_utils.py`

- `compute_psd_window(...)` wraps `cogpy.core.spectral.specx.psdx` on a time slice.
- `psd_to_db(...)` helper.
- `stack_spatial_dims(...)` stacks `(AP, ML)` → `channel`.

### PSDExplorer module

**File:** `cogpy/core/plot/tensorscope/modules/psd_explorer.py`

`psd_explorer` is registered as a built-in module and returns a HoloViews layout.

Linking:
- time is driven by `state.time_hair` (cursor)
- frequency selection is a local stream updated by tapping the average PSD curve
- spatial marker is driven by `state.spatial_space` (AP/ML selection)

## Success Criteria

- Module appears in `ModuleRegistry.list()` as `psd_explorer`.
- Module activation returns a valid HoloViews layout.
- PSD utilities work on 1D and grid signals (with `fs` set).

