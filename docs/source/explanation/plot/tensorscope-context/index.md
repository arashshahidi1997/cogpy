# TensorScope Context Snapshot

This section is a **fast on-ramp** for new agent/dev sessions.
It is designed to stay useful even as TensorScope evolves (layers vs modules, new presets, etc.).

Latest snapshot: [2026-03-06](snapshots/2026-03-06.md)

---

## 1) What TensorScope Is (Stable Concepts)

TensorScope is a Panel/HoloViews app for interactive exploration of neurophysiology tensors (typically LFP on an AP×ML grid over time).

Stable concepts:

- **Authoritative state**: a single `TensorScopeState` owns or references controllers for:
  - time cursor + time window (navigation / analysis windows)
  - channel selection (grid or linear)
  - processing chain (filters / transforms)
  - active modality / signal selection (raw vs processed vs derived)
- **View composition**: the UI is built by composing view components around the state.
  - This may be implemented as “layers” (thin wrappers) or older “modules/presets” (bundled views).
- **Data model**: core views assume an `xarray.DataArray` with a time axis, and either:
  - grid-shaped: `(time, AP, ML)` (preferred), or
  - linear: `(time, channel)`

The key invariant: **state is the single source of truth**; views subscribe/react.

---

## 2) How to Locate the Current Architecture (No version coupling)

Don’t start by memorizing file paths. Start by locating entrypoints by **semantics**.

### Find the composition root
Search for the app shell:

```bash
rg -n "class TensorScopeApp\\b" code/lib/cogpy/src
rg -n "def _register_default_layers\\b|register\\(" code/lib/cogpy/src/cogpy/core/plot/tensorscope
```

### Find what is persisted / restorable
Search for state serialization:

```bash
rg -n "class TensorScopeState\\b" code/lib/cogpy/src
rg -n "def to_dict\\b|def from_dict\\b" code/lib/cogpy/src/cogpy/core/plot/tensorscope
```

### Find layout/presets (panel placement)

```bash
rg -n "class LayoutManager\\b|LayoutPreset\\b|grid_assignments" code/lib/cogpy/src/cogpy/core/plot/tensorscope
```

### Find layer/module composition (what gets rendered)

```bash
rg -n "add_layer\\(|with_layout\\(|build\\(" code/lib/cogpy/src/cogpy/core/plot/tensorscope
rg -n "layers/|modules/" code/lib/cogpy/src/cogpy/core/plot/tensorscope -S
```

---

## 3) Current Working Set (Dated)

**As of 2026-03-06**:

- The current “canonical” running application is the `TensorScopeApp` builder that composes **layers** (thin wrappers around reusable plotting components).
- Some older “module/preset” implementations may still exist in-tree; treat them as **historical** unless the current app routes through them.

See: [snapshot 2026-03-06](snapshots/2026-03-06.md).

---

## 4) Known Issues + Root Causes (Living)

This section tracks *mechanisms* (why bugs happen), not version numbers.

### Welch → multitaper: average-PSD scale doesn’t readapt
Pattern:
- Plot autoscaling can be lost when composing overlays (e.g., `curve * area * line`) if the autoscale option is applied to only one element.
- In HoloViews, apply `framewise=True` (or range options) to the **final composed element** returned by a `DynamicMap`, not only to a base curve.

Where to inspect:
- “Average PSD” view construction + any overlay composition in the PSD explorer layer/module.

### Timeseries overview: what it is + why Y zoom can couple
What the overview is:
- A **downsampled summary signal** (commonly: mean across channels) used as a navigation strip.

How Y-coupling can happen:
- shared Bokeh `Range1d` objects (intentional or accidental)
- HoloViews `shared_axes=True` in a composed layout
- linking utilities that reuse range models

Where to inspect:
- multichannel viewer overview creation
- any link/hook that connects overview and detail plots (often only X should be linked)

### Spatial LFP: processed vs raw (instantaneous mode)
Pattern:
- “Instantaneous” spatial maps often index into the raw tensor at a selected time.
- Processing chains are often applied only through *windowed extraction* helpers; if instantaneous mode bypasses those helpers, it may show raw data.

Where to inspect:
- spatial layer → underlying “frame element” implementation → how (or whether) the processing chain is applied

---

## 5) Repro / Operator Commands

List available presets:
```bash
tensorscope presets
```

List available modules (if supported by the CLI):
```bash
tensorscope modules
```

Serve a dataset with a chosen layout:
```bash
tensorscope serve recording.nc --layout default --port 5008 --show
```

Serve with a PSD-oriented preset/module if available:
```bash
tensorscope serve recording.nc --module psd_explorer --port 5008 --show
```

Tip: if behavior differs across versions, capture it in a dated snapshot under `snapshots/`.

