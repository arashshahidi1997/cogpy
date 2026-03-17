# TensorScope v2.6.2 Specification: Event Detection Integration

## Version
- Version: 2.6.2
- Date: 2026-03-05
- Status: Implementation
- Depends on: v2.6.0 (`cogpy.core.events.EventCatalog`), v2.6.1 (`cogpy.core.detect.EventDetector`)

## Overview

TensorScope v2.6.2 integrates CogPy’s event detection stack (EventCatalog + EventDetector) into the TensorScope visualization framework. It adds:

- A functional `EventOverlayLayer` that renders events as HoloViews overlays.
- An `event_explorer` preset module for lightweight exploration with overlays + summary stats.
- `TensorScopeState.run_detector()` and `TensorScopeState.register_event_catalog()` convenience bridges.

This is **additive and non-breaking**: existing TensorScope apps that use `EventStream` directly remain valid.

## Existing Infrastructure (as implemented)

TensorScope already includes:

- `EventStream` / `EventStyle`: `cogpy.core.tensorscope.events.model`
- `EventRegistry`: `cogpy.core.tensorscope.events.registry`
- State navigation helpers:
  - `TensorScopeState.register_events(name, stream)`
  - `TensorScopeState.jump_to_event(stream_name, event_id)`
  - `TensorScopeState.next_event(stream_name)` / `prev_event(stream_name)`
- Visible time window controller:
  - `TensorScopeState.time_window.window` is authoritative.

## Additions in v2.6.2

### 1) EventOverlayLayer (complete)

**File:** `cogpy/core/plot/tensorscope/layers/events.py`

`EventOverlayLayer` now provides three overlay constructors:

- `create_spatial_overlay()` → `hv.Points` overlay at `(ML, AP)` (requires `AP`/`ML` columns)
- `create_temporal_overlay()` → `hv.VLine` overlay at event time (uses stream `time_col`)
- `create_spectrogram_overlay()` → `hv.Points` overlay at `(time, freq)` (requires `freq`)

Filtering:

- Always filters by the current visible time window: `state.time_window.window`.
- For temporal + spectrogram overlays, additionally filters by spatial selection when event columns include `AP`/`ML`.

Performance:

- Marker count is limited by `max_markers` (default: 200).

### 2) TensorScopeState detector helpers

**File:** `cogpy/core/plot/tensorscope/state.py`

Added:

- `register_event_catalog(name, catalog, style=None)`:
  - Converts `EventCatalog` → `EventStream` via `catalog.to_event_stream(style=...)`
  - Registers the resulting stream under `name`.

- `run_detector(detector, signal_id=None, event_type="events", transform_result=None, style=None)`:
  - Selects the signal (active or by `signal_id`), unless `transform_result` is provided.
  - Runs `detector.detect(data)` and requires an `EventCatalog` result.
  - Registers it via `register_event_catalog(event_type, catalog, ...)`.
  - Returns the `EventCatalog`.

### 3) EventExplorer preset module

**File:** `cogpy/core/plot/tensorscope/modules/event_explorer.py`

Adds the built-in module:

- `event_explorer` (`MODULE`)

Activation returns a HoloViews layout with:

- Spatial view + temporal view (via `ViewFactory`) with event overlays.
- Summary statistics:
  - Event rate curve
  - Spatial count heatmap (if `AP`/`ML`)
  - Frequency histogram (if `freq`)

## Success Criteria

v2.6.2 is complete when:

- `EventOverlayLayer` produces valid HoloViews overlays for spatial/temporal/spectrogram cases.
- `TensorScopeState.run_detector()` registers detector output as an event stream.
- The `event_explorer` module is registered in `ModuleRegistry`.
- Unit tests cover the new integration points.

