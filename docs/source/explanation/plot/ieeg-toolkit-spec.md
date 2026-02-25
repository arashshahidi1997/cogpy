# iEEG Toolkit: Goals & Specifications (Pre-Implementation)

This document turns the “Future directions” in `ieeg-toolkit.md` into a concrete, testable set of goals and specifications for evolving the iEEG GUI toolkit.

## Scope

Interactive exploration of iEEG recorded from a **2D electrode grid**, using:

- Panel for app composition
- Bokeh/HoloViews for plotting
- `cogpy.core.plot.*` as the implementation home

The core UX is: *select electrodes on a grid, inspect time series, and link into richer views (spectrogram/topomap) without losing interactivity on large recordings*.

## Goals

- **Deterministic dev fixtures**: every viewer has a “known-good” example dataset for quick debugging and for responsiveness checks on large inputs.
- **Explicit schemas**: GUIs consume a small number of named entities with stable dims/coords/attrs (no hidden transposes).
- **Composable widgets**: selection, transforms, and viewers remain modular and are linkable via shared state (e.g. time cursor).
- **Performance by construction**:
  - render budgets (downsampling),
  - avoid full materialization where possible (windowed compute),
  - predictable latency during selection changes.
- **Minimal IO assumptions**: GUIs operate on in-memory `xarray`/`numpy` objects; IO belongs elsewhere.

## Non-goals

- Full Neuroscope2 parity (spike sorting, complete annotation systems, file browsing).
- A universal viewer for all electrode layouts (start with 2D grids + multichannel).
- Making decisions about canonical storage formats (this is GUI/data-model oriented).

## Inputs: expected entities (summary)

This toolkit should standardize on a small set of entities (schemas defined in the datasets docs):

- `IEEGGridTimeSeries`: `xr.DataArray` with dims `("time","AP","ML")` (canonical target).
- `MultichannelTimeSeries`: `xr.DataArray` with dims `("channel","time")`.
- Optional: `GridSpectrogram4D` + `BurstPeaksTable` for linked time–frequency views.

For GUI development, these are expected to come from bundle constructors (see `explanation/datasets/gui-bundles.md`).

## Component responsibilities (current + planned)

### Current components (implemented)

- `ChannelGrid`: selection state + modes; *no rendering*.
- `ChannelGridWidget`: electrode grid visualization + interaction; optional atlas image and background scalar.
- `MultichannelViewer`: stacked time series viewer with overview strip and downsampling.
- `IEEGViewer`: wiring layer: xarray → numpy, z-score, optional grid-driven selection + apply button.

### Planned cross-cutting components (spec only)

- `TimeCursor` (shared state):
  - single “current time” and/or `(t0, t1)` window shared by traces/spectrogram/topomap.
- `SelectionPolicy` (pure compute):
  - converts raw metrics and user intents (“top-N variance”) into a `ChannelGrid.selected` set.
- `TransformPipeline` (view-time transforms):
  - optional reref/filter/standardization applied as a *view* (not necessarily written back).

## Feature specifications derived from “Future directions”

Each feature below includes a minimal contract and acceptance criteria.

### 1) Signal-driven selection

**User story**
- “Given the current visible time window, select the top-N channels by variance (or by correlation to a seed channel).”

**Inputs**
- `sig_tc` as `xr.DataArray` or numpy view of shape `(channel, time)` (canonical for compute).
- Current time window `(t0, t1)` from viewer state.
- Strategy parameters:
  - `metric`: `"variance"` | `"correlation"`
  - `n`: int
  - `seed_channel`: optional (for correlation)

**Outputs**
- A set of `(ap, ml)` pairs (or flat channel indices) that can be applied to `ChannelGrid`.

**Acceptance criteria**
- Deterministic on fixed input + seed (if randomness is used at all).
- Uses only the current window (no full-recording scan unless explicitly requested).
- If `n` exceeds available channels, clamps gracefully.

### 2) Sorting (reordering displayed channels)

**User story**
- “Reorder the stacked traces by AP, ML, variance, or correlation.”

**Inputs**
- Current selected channel set.
- Sorting key:
  - `"AP"`, `"ML"`, `"variance"`, `"correlation_to_seed"`

**Outputs**
- Ordered list of flat channel indices passed to `MultichannelViewer.show_channels(...)`.

**Acceptance criteria**
- Sorting does not change which channels are selected, only their display order.
- Sorting is stable (ties deterministic).

### 3) Atlas placement improvements (`atlas_mode="full"`)

**User story**
- “Show the full atlas image and place the electrode grid in correct physical location, including asymmetric AP extent.”

**Inputs**
- `ap_coords`, `ml_coords` (physical coords) and atlas placement metadata.

**Outputs**
- Correct placement parameters (image origin + extent) used by `ChannelGridWidget`.

**Acceptance criteria**
- The same physical coordinate at two places (grid tick labels and hover labels) maps consistently.
- Documented coordinate conventions: which direction is positive AP/ML, and image orientation assumptions.

### 4) Bad channel detection overlay (grid widget)

**User story**
- “Flag likely bad channels on the grid (e.g., high kurtosis) and allow quick exclusion.”

**Inputs**
- Metric computed per channel from the current window or full data:
  - e.g. kurtosis, RMS, line noise power proxy.

**Outputs**
- A per-cell flag and/or scalar displayed on the grid:
  - color outline, hatch, or icon-like overlay (implementation-dependent).

**Acceptance criteria**
- Bad-channel flags do not break selection interaction.
- “Exclude bads” is reversible and does not destroy the underlying selection state.

### 5) Linked views (shared time cursor)

**User story**
- “Move a cursor in the trace view and see the same time reflected in spectrogram and topomap.”

**Inputs**
- Shared `TimeCursor` state.

**Outputs**
- Linked updates across all views.

**Acceptance criteria**
- Cursor updates do not trigger full recompute of unrelated views.
- With large mode data, cursor movement remains responsive (bounded redraw cost).

### 6) Lazy loading / windowed materialization

**User story**
- “Long recordings should not require full `.compute()` to be viewable.”

**Inputs**
- Potentially Dask-backed xarray arrays.
- Current visible time window and rendering budgets (`detail_px`, `overview_px`).

**Outputs**
- View-time window extraction and downsampling.

**Acceptance criteria**
- Large mode can run without materializing the full array in memory.
- Window changes only compute what is needed for the view.
- Explicitly document any “overview strip” assumptions (e.g., precomputed mean trace vs streaming).

## Performance targets (guidance)

- “Small mode”: viewer construction + first render should be near-instant.
- “Large mode”: interactions should remain bounded:
  - selection changes should not rebuild the entire HoloViews object graph,
  - time window changes should not allocate full-resolution traces.

Bundle sizing guidance is specified in `explanation/datasets/modes.md`.

## Compatibility notes / known tensions

- Current examples and some code paths use `("time","ML","AP")` in places. The **spec target** is `("time","AP","ML")` for grid-aware signals.
- The flattening convention (AP-major vs ML-major) must be defined once and tested because it affects:
  - `ChannelGrid.flat_indices`
  - `stack(channel=("AP","ML"))` ordering
  - wiring between grid selection and trace display

## Proposed roadmap (phased)

1. **Schema hardening**: choose canonical grid dims/flattening; add validation helpers.
2. **Selection improvements**: signal-driven selection + sorting + presets.
3. **QC overlays**: bad-channel metrics and grid overlays.
4. **Linked views**: shared time cursor + spectrogram + topomap.
5. **Scalability**: windowed compute + optional Dask-friendly paths.

