# TensorScope v2.4 Specification: Orthoslicer & Multi-Stream Linking

## Version
- Version: 2.4.0
- Date: 2026-03-04
- Status: Implementation
- Builds on: v2.3 (View Builder UI)

## Overview

TensorScope v2.4 introduces:
- An **Orthoslicer** module for exploring time-frequency-spatial data (spectrograms).
- **Multi-stream linking** via multiple `CoordinateSpace` instances (spatial, temporal, spectral).
- An enhanced `ViewSpec` that supports **fixed dimension values** (needed for true montages).

This release builds on the v2.2+ declarative view system without special-casing orthoslicers at the app level: the orthoslicer is “just another module”, implemented using HoloViews + shared coordinate selections.

## Core Features

### 1. Orthoslicer Module (Spectrogram)

Orthoslicer explores 4D spectrogram output produced by `spectrogramx()`:

- dims: `(AP, ML, freq, time)` for grid LFP inputs

It displays four linked views:

```text
┌──────────────┬─────────────────────────┐
│ Time Profile │ Freq Profile            │
│ time @       │ freq @                  │
│ (freq, AP,ML)│ (time, AP,ML)           │
├──────────────┼─────────────────────────┤
│ TF Heatmap   │ Spatial Map             │
│ (time×freq)  │ (AP×ML)                 │
│ @ (AP,ML)    │ @ (time,freq)           │
└──────────────┴─────────────────────────┘
```

Interactions:
- Click TF heatmap → updates selected `time` and `freq`.
- Click spatial map → updates selected `AP` and `ML` (indices).
- All views update through shared coordinate spaces.

### 2. Multi-Stream Coordinate Spaces

v2.4 extends `CoordinateSpace` with an optional HoloViews `Stream` bridge:
- `create_stream()` creates a stream with parameters matching the space dims.
- `set_selection()` updates the stream (when present).
- Stream updates can sync back into the coordinate space (guarded against recursion).

### 3. Enhanced ViewSpec

Adds:
- `fixed_values: dict[str, object]` — dimension values fixed for the view (not controls)
- `coord_spaces: list[str]` — declarative hint for which coordinate spaces to link (future use)

## Success Criteria

v2.4 is complete when:
- `CoordinateSpace` can create/sync a HoloViews stream
- `ViewSpec` supports `fixed_values` + `coord_spaces` with validation and serialization
- Orthoslicer module computes `spectrogramx()` and renders 4 linked views
- Tests cover the new behavior
- Example demonstrates orthoslicer interactivity

