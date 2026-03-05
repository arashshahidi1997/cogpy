# EventCatalog Specification v2.6.0

## Version
- Version: 2.6.0
- Date: 2026-03-05
- Status: Implementation
- Depends on: `cogpy.datasets.schemas` (`Events`, `Intervals`), `cogpy.core.plot.tensorscope.events` (`EventStream`)

## Overview

`EventCatalog` provides a unified, lightweight event representation that bridges:

- **Analysis workflows:** typed containers (`Events`, `Intervals`) for integration with pynapple/MNE-style pipelines.
- **Visualization workflows:** `EventStream` for TensorScope UI (tables, navigation, overlays).

It is a wrapper around a pandas DataFrame with:
- a **minimal required schema** (`event_id`, `t`)
- optional standardized columns (intervals/spatial/spectral/metadata/provenance)
- conversion and factory helpers for common detector outputs

This is intentionally **non-breaking and additive**. It does not replace the existing
strict detector contract `cogpy.datasets.schemas.EventCatalog` (which assumes interval
fields are present); instead it provides a more general “bridge” catalog suitable for
point events and interval events.

## Schema

### Required Columns
- `event_id` : `str | int` — unique identifier
- `t` : `float` — event time in seconds (peak/center)

### Optional Interval Columns
- `t0` : `float` — start time (seconds)
- `t1` : `float` — end time (seconds)
- `duration` : `float` — computed as `t1 - t0` when missing (requires `t0`, `t1`)

### Optional Spatial Columns
- `channel` : `int | str`
- `AP` : `float`
- `ML` : `float`

### Optional Spectral Columns
- `freq` : `float`
- `f0`, `f1` : `float`
- `bandwidth` : `float` (computed as `f1 - f0` when missing)

### Optional Metadata / Provenance Columns
- `label` : `str`
- `score` : `float`
- `value` : `float`
- `family` : `str`
- `detector` : `str`
- `source_signal` : `str`
- `pipeline` : `str`

## Core API

Converters:
- `to_events()` → `Events`
- `to_intervals()` → `Intervals` (requires `t0`, `t1`)
- `to_point_intervals(half_window)` → `Intervals`
- `to_event_stream(style=...)` → TensorScope `EventStream`

Factories:
- `from_hmaxima(...)`
- `from_blob_candidates(...)`
- `from_burst_dict(...)`
- `from_spwr_mat(...)`

Queries:
- `filter_by_time(t_min, t_max)`
- `filter_by_channel(channels)`
- `filter_by_spatial(AP, ML, radius)`

## Success Criteria

- `EventCatalog` implemented in `cogpy.core.events`
- Factories cover common existing detector outputs
- Converters to `Events`, `Intervals`, and TensorScope `EventStream` work
- Validation and basic filters are tested
- Example demonstrates end-to-end usage

