# TensorScope v2.6.3: Interval Event Enhancements

## Version
- Version: 2.6.3
- Date: 2026-03-05
- Depends on: v2.6.2

## Overview

TensorScope v2.6.3 improves support for **interval events** (events with start/end times `t0 → t1`):

- Temporal overlays render **spans** for interval events (plus a dashed peak marker at `t`).
- Event Explorer adds **interval-specific statistics** (duration + IEI + onset rate).
- Adds an **event-triggered average** (ETA) view.
- Adds **overlap detection** utilities for interval catalogs.

## Implemented Components

### 1) Temporal overlay: point vs interval

**File:** `cogpy/core/plot/tensorscope/layers/events.py`

- If an event stream includes `t0` and `t1`, temporal overlays render:
  - `hv.VSpan(t0, t1)` for the interval
  - `hv.VLine(t)` dashed for the peak/center time
- If `t0/t1` are absent, overlays fall back to point-event `hv.VLine(t)`.

### 2) Event-triggered average layer

**File:** `cogpy/core/plot/tensorscope/layers/event_triggered_average.py`

`EventTriggeredAverageLayer` computes the average (±1 std) of the currently-selected trace around event times.

### 3) Interval statistics in `event_explorer`

**File:** `cogpy/core/plot/tensorscope/modules/event_explorer.py`

Adds:

- Duration histogram (`duration` or `t1 - t0`)
- IEI histogram (diffs of sorted `t0`)
- Onset-rate curve (histogram of `t0`)
- ETA view (via `EventTriggeredAverageLayer`)

### 4) Overlap detection

**File:** `cogpy/core/events/overlap.py`

`detect_overlaps(EventCatalog)` returns a DataFrame of overlapping event pairs with overlap duration.

## Success Criteria

- Interval events render as spans on temporal overlays.
- Interval stats appear in `event_explorer` when `t0/t1` are present.
- ETA view renders without requiring a server.
- Overlap detection works for interval catalogs and is empty for point catalogs.

