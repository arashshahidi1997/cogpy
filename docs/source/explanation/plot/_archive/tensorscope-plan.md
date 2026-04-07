# TensorScope Implementation Plan (v1.0)

This document is a **phased, implementation-oriented plan** for building **TensorScope**, aligned with the design principles in `tensorscope-spec.md` and grounded in the existing `cogpy.plot` infrastructure (ChannelGrid/Widget, ProcessingChain, MultichannelViewer, TopoMap, TimeHair, etc.).

> Note: the principles doc is named `tensorscope-spec.md`. All references in this plan match.

---

## 1. Executive Summary

### Vision
TensorScope is a **Panel-based neurophysiology visualization app** that composes reusable visualization components (“layers”) around a **single authoritative state**, enabling fast, interactive exploration of multi-modal session data (LFP/LFP-grid, events, spectrograms, spikes) at scale.

### Core Goals
- **Single source of truth** for time, selection, transforms, layout (per `tensorscope-spec.md` §2).
- **Wrap existing components** (thin layers) rather than rewriting proven viewers (per §3.2).
- **Fast by architecture**: bounded compute via windowed materialization + downsampling budgets (per §7).
- **Composable multi-view**: linked cursor/window and consistent schemas across layers.
- **Gradual migration**: existing `ieeg_viewer.py` and demo apps remain usable while TensorScope is adopted incrementally.

### Success Metrics
- **Interactive latency budgets** (target): cursor moves < 100 ms UI response; selection changes < 300 ms (small mode) and < 1–2 s (large mode) with bounded compute.
- **Memory safety**: opening “large mode” does not materialize full arrays; windowed compute only.
- **Reliability**: no Bokeh “single document ownership” crashes in common workflows; layers implement `dispose()` and pass leak checks.
- **Adoption**: at least 2 existing notebooks/apps migrated to TensorScope shell without losing capabilities (e.g., iEEG toolkit + bursts).

### Timeline (conservative)
- **Phase 0 (Weeks 1–2)**: Foundation + repo structure + minimal app boot.
- **Phase 1 (Weeks 3–4)**: State architecture + serialization + schema validation.
- **Phase 2 (Weeks 5–7)**: Core layers wrapping existing components.
- **Phase 3 (Weeks 8–9)**: Application shell + FastGridTemplate + presets.
- **Phase 4 (Weeks 10–12)**: Events system (table + overlays + navigation).
- **Phase 5 (Weeks 13–15)**: Multi-modal support + alignment.
- **Phase 6 (Weeks 16–18)**: Polish/optimization + orthoslicer consolidation path.

> **Note:** For Panel/HoloViews compatibility issues and workarounds, see [tensorscope-issues.md](tensorscope-issues.md).
> **Runtime:** For the recommended `conda run -n cogpy ...` commands (Panel/pytest), see [tensorscope-runtime.md](tensorscope-runtime.md).

---

## 1.5 Quick Reference: Component Status by Phase

This table provides an at-a-glance view of what gets built when.

| Component | Phase | Type | Location | Depends On |
|-----------|-------|------|----------|------------|
| Package structure | 0 | INFRA | `cogpy/core/plot/tensorscope/` | - |
| TensorScopeState | 1 | NEW | `tensorscope/state.py` | TimeHair, ChannelGrid, ProcessingChain |
| TimeWindowCtrl | 1 | NEW | `tensorscope/time_window.py` | param |
| Schema validators | 1 | NEW | `tensorscope/schema.py` | xarray |
| Session serialization | 1 | NEW | `tensorscope/state.py` | TensorScopeState |
| TensorLayer (base) | 2 | NEW | `tensorscope/layers/base.py` | param, Panel |
| TimeseriesLayer | 2 | WRAP | `tensorscope/layers/timeseries.py` | MultichannelViewer, TensorLayer |
| SpatialMapLayer | 2 | WRAP | `tensorscope/layers/spatial.py` | GridFrameElement, TensorLayer |
| ChannelSelectorLayer | 2 | WRAP | `tensorscope/layers/controls.py` | ChannelGridWidget, TensorLayer |
| ProcessingControlsLayer | 2 | WRAP | `tensorscope/layers/controls.py` | ProcessingChain, TensorLayer |
| TimeNavigatorLayer | 2 | WRAP | `tensorscope/layers/navigation.py` | PlayerWithRealTime, TensorLayer |
| LayerManager | 2 | NEW | `tensorscope/layers/manager.py` | TensorLayer |
| LayoutManager | 3 | NEW | `tensorscope/layout.py` | Panel FastGridTemplate |
| TensorScopeApp | 3 | NEW | `tensorscope/app.py` | State, LayerManager, LayoutManager |
| Layout presets | 3 | DATA | `tensorscope/presets/` | LayoutManager |
| EventStream | 4 | NEW | `tensorscope/events/model.py` | pandas |
| EventRegistry | 4 | NEW | `tensorscope/events/registry.py` | EventStream |
| EventTableLayer | 4 | NEW | `tensorscope/layers/events.py` | EventStream, Panel Tabulator |
| EventOverlayLayer | 4 | NEW | `tensorscope/layers/events.py` | EventStream, HoloViews |
| EventNavigationLayer | 4 | NEW | `tensorscope/layers/events.py` | EventStream |
| DataRegistry | 5 | NEW | `tensorscope/data/registry.py` | - |
| GridLFPModality | 5 | NEW | `tensorscope/data/modalities/lfp.py` | xarray, schema validators |
| FlatLFPModality | 5 | NEW | `tensorscope/data/modalities/lfp.py` | xarray, schema validators |
| SpectrogramModality | 5 | NEW | `tensorscope/data/modalities/spectrogram.py` | xarray |
| Time alignment utilities | 5 | NEW | `tensorscope/data/alignment.py` | numpy, xarray |
| Performance benchmarks | 6 | TEST | `tests/tensorscope/benchmarks/` | pytest-benchmark |
| CLI entrypoint | 6 | NEW | `tensorscope/cli.py` | click, TensorScopeApp |
| Orthoslicer consolidation | 6 | REFACTOR | `tensorscope/layers/slicer.py` | orthoslicer_rangercopy |

**Legend:**
- **NEW**: Build from scratch
- **WRAP**: Thin wrapper around existing component
- **REUSE**: Use existing component as-is (no changes)
- **INFRA**: Project structure, not code
- **DATA**: Configuration files
- **TEST**: Testing infrastructure
- **REFACTOR**: Consolidate existing code

## 2. Architecture Overview

### 2.1 Component Hierarchy

```text
TensorScopeApp (composition root)
└── TensorScopeState (authoritative state + controller ownership)
    ├── time_hair: TimeHair          (controller; owns current_time)
    ├── time_window: TimeWindowCtrl  (controller; owns (t0,t1))
    ├── channel_grid: ChannelGrid    (controller; owns selected (ap,ml))
    ├── processing: ProcessingChain  (controller; owns transform params)
    ├── data_registry: DataRegistry  (references to modalities; adapters)
    └── event_registry: EventRegistry (EventStream(s))
└── LayerManager
    ├── TensorLayer (base interface + dispose tracking)
    ├── ChannelSelectorLayer (wrap ChannelGridWidget)
    ├── TimeseriesLayer (wrap MultichannelViewer or multichannel_timeseries_view)
    ├── SpatialMapLayer (wrap GridFrameElement/TopoMap)
    ├── ProcessingControlsLayer (wrap ProcessingChain.controls)
    ├── TimeNavigatorLayer (wrap PlayerWithRealTime)
    └── Event layers (Phase 4)
└── LayoutManager (FastGridTemplate, presets, persistence)
```

### 2.2 Data Flow (disk → state → controllers → views)

```text
flowchart LR
  A[Disk / NWB / Zarr / NetCDF] -->|load (outside UI)| B[DataAdapters]
  B --> C[TensorScopeState.data_registry]
  C -->|state change| D[Controllers]
  D -->|windowed compute + downsample| E[Layer wrappers]
  E --> F[Views: Panel + HoloViews + Bokeh]
  F -->|user actions| C
```

**Design principle alignment**
- Unidirectional flow for *authoritative state* (spec §2.2): user action → state → listeners → view updates.
- Views may own *derived state* (zoom/pan) but can synchronize it to authoritative state when persistence is required (spec §2.5).

### 2.3 State Management (ownership map)

| State | Owner | Notes |
|---|---|---|
| Current time cursor | `TimeHair` owned by `TensorScopeState` | `TensorScopeState.current_time` delegates to `time_hair.t`. |
| Current visible time window | `TimeWindowCtrl` owned by `TensorScopeState` | Sourced from viewers (RangeTool/RangeX/BoundsX), commanded by navigation. |
| Channel selection (grid) | `ChannelGrid` owned by `TensorScopeState` | Authoritative selection as `(ap,ml)` set. |
| Channel selection (flat indices) | Derived in `TensorScopeState` | Computed from grid selection + flattening convention; cached. |
| Processing parameters | `ProcessingChain` owned by `TensorScopeState` | Used by windowed compute layers. |
| Layout preset + persistence | `LayoutManager` | Stored/loaded via session serialization. |
| Zoom/pan of Bokeh/HV views | View-owned (derived) by default | Persist only if required; then mirror into state. |

### 2.4 Extension Points (plugins/customization hooks)

The extension model is intentionally simple (spec §1.2, §3.3):
- **Layer plugins**: register a `LayerSpec` that constructs a `TensorLayer`.
- **Selection policies**: register pure functions (e.g., `top_n_variance`) that update `ChannelGrid`.
- **Event sources**: register `EventStream` objects under a name (e.g., `"bursts"`, `"spindles"`).
- **Data modalities**: register modality adapters (e.g., `GridLFP`, `FlatLFP`, `Spectrogram`, `Spikes`).

---

## 3. Component Inventory

This inventory is based on the existing `cogpy.plot` scan and the `ieeg-toolkit-spec.md`/`tensorscope-spec.md` principles.

### A. Reuse As-Is

| Component | Location | Purpose | Integration Notes |
|---|---|---|---|
| `TimeHair`, `AxisHair` | `cogpy/core/plot/time_player.py` | Click/tap cursors | `TensorScopeState` owns these controllers and delegates properties. |
| `ProcessingChain` | `cogpy/core/plot/processing_chain.py` | Windowed transforms | Wrap in a layer for controls; used by compute layers for window extraction. |
| `ChannelGrid` | `cogpy/core/plot/channel_grid.py` | Selection model | Authoritative selection model in state. |
| `ChannelGridWidget` | `cogpy/core/plot/channel_grid_widget.py` | Selection UI + atlas/QC | Channel selector layer wraps it; emits taps/selection changes. |
| `TopoMap` | `cogpy/core/plot/topomap.py` | Spatial scalar view | SpatialMapLayer can either embed `TopoMap` directly or via `GridFrameElement`. |
| `GridFrameElement` | `cogpy/core/plot/grid_frame_element.py` | Time-linked scalar frame | Use as the first SpatialMap layer in v1.0. |
| `selection_policy` functions | `cogpy/core/plot/selection_policy.py` | Pure compute | Register as “policies” callable from UI (Phase 2/4). |
| `theme` constants | `cogpy/core/plot/theme.py` | Consistent look | TensorScope bootstrap imports these; Panel template uses compatible colors. |

### B. Adapt/Wrap

| Component | Location | Issue | Adaptation Strategy |
|---|---|---|---|
| `MultichannelViewer` | `cogpy/core/plot/multichannel_viewer.py` | Exposes linking via private `_range_stream` | Add a public `range_stream` (read-only) and/or `get_time_window()` API; keep internals. |
| `linked_views.link_topomap_to_viewer` | `cogpy/core/plot/linked_views.py` | Private coupling (`viewer._range_stream`) | Rework to accept a public stream/controller; move to TensorScope “links” module. |
| `ieeg_viewer` | `cogpy/core/plot/ieeg_viewer.py` | App-specific glue | Refactor into a reusable modality adapter (`GridLFPAdapter`, `FlatLFPAdapter`) and a `TimeseriesLayer` wrapper. Keep old API for compatibility. |
| `xarray_hv` helpers | `cogpy/core/plot/xarray_hv.py` | Mixed responsibilities | Keep as-is initially; later split into `adapters.py` vs `hv_helpers.py` in TensorScope package. |
| Orthoslicer “best” variant | `cogpy/core/plot/orthoslicer_rangercopy.py` | Monolithic class | Wrap in a provisional `TensorSlicerLayer`; consolidate later (Phase 6). |
| Demo apps | `ieeg_toolkit_app.py`, `spectrogram_bursts_app.py` | Not a general shell | Keep for manual tests; add “run TensorScope demo” equivalents. |

### C. New Components

| Component | Purpose | Dependencies | Complexity |
|---|---|---|---|
| `TensorScopeState` | Central authoritative state | `param`, controllers above | Medium |
| `TimeWindowCtrl` | Owns `(t0,t1)` with validation | `param` | Low |
| `DataRegistry` + modality adapters | Register/access data + schema normalization | `xarray`, existing adapters | High |
| `EventStream` model | Unified event data for table/overlays | `pandas` or `xarray` | Medium |
| `TensorLayer` base + lifecycle | Standard layer interface + `dispose()` | `panel`, `param`, `holoviews` | Medium |
| `LayerManager` + `LayerSpec` | Add/remove layers dynamically | `TensorLayer` | Medium |
| `LayoutManager` | FastGridTemplate layout + presets + persistence | `panel` | Medium |
| Session serialization | Save/load state + layout + references | `json`, `pydantic` (optional) | Medium |
| Schema validation utilities | Enforce dims/coords/attrs conventions | `xarray` | Medium |
| Minimal perf harness | Regression detection + budgets | `pytest` + timers | Low–Medium |

### D. Deprecate (phased out / discouraged for TensorScope core)

| Component | Reason | Migration Path |
|---|---|---|
| `orthoslicer_ranger.py` | Incomplete | Prefer `orthoslicer_rangercopy.py`; later consolidate into TensorScope slicer layer. |
| `orthoslicer/base.py` | Stub only | Implement modular pieces inside TensorScope package instead. |
| `frame_plot.py`, `xarr_plot.py`, `specgram_plot.py`, `time_plot.py` | Legacy plotting stack (matplotlib/ipywidgets/plotly) | Keep as legacy utilities; do not build TensorScope on them. |

---

## 4. Implementation Phases

### Phase 0: Foundation (Weeks 1–2)

**Goals**
- Establish TensorScope package structure and conventions.
- Create a minimal end-to-end app using the existing demo (`notebooks/tensorscope/tensorscope_app.py`) as the UI reference.

**Deliverables**

1. **Package structure:**
```
cogpy/core/plot/tensorscope/
├── __init__.py              # Public exports
├── state.py                 # TensorScopeState (stub, completed in Phase 1)
├── app.py                   # TensorScopeApp (stub, completed in Phase 3)
├── layers/
│   ├── __init__.py
│   └── base.py             # TensorLayer interface (stub)
├── data/
│   ├── __init__.py
│   └── registry.py         # DataRegistry (stub)
└── events/
    ├── __init__.py
    └── model.py            # EventStream (stub)

tests/core/plot/tensorscope/
├── __init__.py
├── test_state.py           # Phase 1 tests (stubs)
├── test_layers.py          # Phase 2 tests (stubs)
└── conftest.py             # Shared fixtures
```

2. **"Hello TensorScope" demo** (`examples/tensorscope/hello_tensorscope.py`):
   - Loads `example_ieeg_grid()` from cogpy.datasets
   - Shows FastGridTemplate with 3 placeholder cards:
     - Spatial view (empty Card with title)
     - Timeseries view (empty Card with title)
     - Controls sidebar (minimal widget)
   - Demonstrates state object owns TimeHair + ChannelGrid (stub delegation)
   - Proves state delegation pattern compiles and runs
   - Servable via `panel serve examples/tensorscope/hello_tensorscope.py`

3. **Stub implementations** for all major classes (empty methods with docstrings)

4. **CI integration**: Add `pytest tests/core/plot/tensorscope/` to test suite (smoke tests only)

**Dependencies**
- None (pure scaffolding).

**Success criteria**
- `pytest -k tensorscope` passes (new tests added for basic construction).
- A servable app launches without errors and shows the grid layout.

**Risk factors**
- Panel/HoloViews version behaviors. Mitigation: pin versions in environment and add smoke tests (see §7).

**Definition of Done**
- [ ] All deliverables completed and committed
- [ ] All new tests passing (unit + integration for this phase)
- [ ] Code review completed (see §7.5 checklist)
- [ ] No regressions in existing cogpy.plot tests
- [ ] Documentation updated (API docs + relevant guides)
- [ ] `panel serve examples/tensorscope/hello_tensorscope.py` launches without errors
- [ ] Package imports successfully: `from cogpy.tensorscope import TensorScopeApp`

---

### Phase 1: State Architecture (Weeks 3–4)

**Goals**
- Implement authoritative state, delegation to controllers, schema validation entry points, and session serialization.
- Remove private coupling assumptions in new code (spec §3.3).

**Deliverables**
- `TensorScopeState` with:
  - controller ownership (`TimeHair`, `ChannelGrid`, `ProcessingChain`, `TimeWindowCtrl`)
  - derived properties (flat indices, selection summaries)
  - registry hooks (`register_modality`, `register_events`)
- `TimeWindowCtrl` (t0/t1 with bounds/ordering + “snap-to-data” hooks).
- Session serialization: `state.to_dict()` + `state.from_dict(...)` (references, not raw arrays; spec §2.4).
- Schema validators for the first modality (grid LFP): validate dims, coords, flattening convention.

**Dependencies**
- Existing controllers (`TimeHair`, `ChannelGrid`, `ProcessingChain`).

**Success criteria**
- Round-trip serialization test passes (state → dict → state).
- Schema validation catches known dimension-order issues.

**Risk factors**
- Persisting view-owned state (zoom) too early. Mitigation: keep zoom as derived state in Phase 1; persist later if needed (spec §2.5).

**Definition of Done**
- [ ] All deliverables completed and committed
- [ ] All new tests passing (unit + integration for this phase)
- [ ] Code review completed (see §7.5 checklist)
- [ ] No regressions in existing cogpy.plot tests
- [ ] Documentation updated (API docs + relevant guides)
- [ ] Round-trip serialization test passes with realistic state
- [ ] Schema validator catches `(time, ML, AP)` and rejects it
- [ ] Flattening convention test verifies row-major (`test_flattening_convention()`)
- [ ] State delegation tests pass (e.g., `state.current_time` ↔ `time_hair.t`)

---

### Phase 2: Core Layers (Weeks 5–7)

**Goals**
- Define the layer interface and implement wrappers for existing core components.
- Produce a usable “TensorScope v0.1” app that loads real iEEG bundle data and supports linked interactions.

**Deliverables**
- `TensorLayer` base class with `dispose()` tracking watchers/streams (spec §10).
- Layer wrappers:
  - `ChannelSelectorLayer` (wrap `ChannelGridWidget`)
  - `TimeseriesLayer` (wrap `MultichannelViewer`; expose public stream/controller)
  - `SpatialMapLayer` (wrap `GridFrameElement` or `TopoMap`)
  - `ProcessingControlsLayer` (wrap `ProcessingChain.controls()`)
  - `TimeNavigatorLayer` (wrap `PlayerWithRealTime`)
- “Link contracts”:
  - state `time_hair` drives timeseries hair and spatial frame
  - timeseries view emits current window → state `time_window`
- `ieeg_grid` demo: reuse `cogpy.datasets.gui_bundles.ieeg_grid_bundle` as Phase 2 fixture.

**Dependencies**
- Phase 1 state + validators.
- Small adaptation to `MultichannelViewer` (public range stream API) OR a TensorScope wrapper that owns the stream.

**Success criteria**
- Demo supports: grid selection → apply → traces update; cursor click → spatial map updates; processing toggle re-renders window.
- No memory leaks after add/remove layers in a test (dispose correctness).

**Risk factors**
- Bokeh document ownership errors in notebooks. Mitigation: standardize `.panel(fresh=...)` strategy at wrapper boundaries; add tests that construct twice.

**Definition of Done**
- [ ] All deliverables completed and committed
- [ ] All new tests passing (unit + integration for this phase)
- [ ] Code review completed (see §7.5 checklist)
- [ ] No regressions in existing cogpy.plot tests
- [ ] Documentation updated (API docs + relevant guides)
- [ ] Demo app with real iEEG data runs end-to-end
- [ ] Memory leak test passes (100 layer create/destroy cycles)
- [ ] Performance budget test passes (<100ms cursor update with 64 channels)
- [ ] All layers implement `dispose()` correctly (verified by tests)
- [ ] Public stream API implemented (no `._private` coupling)

---

### Phase 2.5: Integration Checkpoint (End of Week 7)

**Purpose**
Pause before major app shell work to verify core layers are production-ready.

**Goals**
- Validate that Phase 2 deliverables are robust and complete
- Gather early feedback from demo
- Identify any issues before building on this foundation

**Checkpoint criteria (all must pass):**
- [ ] Demo app with real iEEG data works end-to-end (no crashes)
- [ ] Memory leak test passes (100 layer create/destroy cycles, <10MB growth)
- [ ] Performance budget test passes (<100ms cursor update, <300ms selection change)
- [ ] All Phase 2 layers implement `dispose()` correctly (verified)
- [ ] Team demo/feedback session completed with stakeholders
- [ ] No critical bugs in issue tracker from Phase 2

**Decision point**
- **If all criteria pass**: Proceed to Phase 3
- **If any fail**: Extend Phase 2 by 1-2 weeks to address issues
- **Do NOT proceed to Phase 3 with failing criteria**

**Duration**: 2-3 days (including demo preparation)

**Rationale**: Building app shell (Phase 3) on unstable layers wastes time. Catch issues early.

---

### Phase 3: Application Shell (Weeks 8–9)

**Goals**
- Convert the demo layout into a real app shell with FastGridTemplate, presets, and layout persistence.

**Deliverables**
- `TensorScopeApp`:
  - owns `TensorScopeState`, `LayerManager`, `LayoutManager`
  - `add_layer`, `remove_layer`, `shutdown`
  - `servable()` and `show()` helpers
- `LayoutManager`:
  - default layout preset (“3-pane”: spatial / events / traces)
  - save/load layout in session file
- Add a “Controls” sidebar pattern consistent with `tensorscope_app.py`.

**Dependencies**
- Core layers implemented.

**Success criteria**
- Load/save session preserves: selected channels, processing params, cursor, layout preset.

**Risk factors**
- Template sizing/scroll interactions causing derived state mismatch. Mitigation: treat sizing/scroll as derived; persist only layout grid positions and high-level options.

**Definition of Done**
- [ ] All deliverables completed and committed
- [ ] All new tests passing (unit + integration for this phase)
- [ ] Code review completed (see §7.5 checklist)
- [ ] No regressions in existing cogpy.plot tests
- [ ] Documentation updated (API docs + relevant guides)
- [ ] Load/save session preserves state (cursor, selection, processing, layout)
- [ ] At least 2 layout presets work correctly
- [ ] FastGridTemplate integration stable (no sizing/scroll bugs)

---

### Phase 4: Events System (Weeks 10–12)

**Goals**
- Add first-class event model and UX: table, overlay markers, navigation, selection linking.

**Deliverables**
- `EventStream` data model and registry in `TensorScopeState`.
- `EventTableLayer` (Tabulator) + `EventNavigationLayer`:
  - clicking an event jumps `state.current_time`
  - optional “follow cursor” mode
- `EventOverlayLayer`:
  - overlays events on timeseries (vertical markers)
  - overlays events on spatial map (if events have spatial coords)
- Example event source:
  - reuse bursts tables from orthoslicer bundles (`spectrogram_bursts_bundle`) as initial integration.

**Dependencies**
- State serialization (events should be referenced by name/path, not duplicated).

**Success criteria**
- Jump-to-event is deterministic and works across layers.
- Overlays do not trigger full recomputation; only update overlay glyphs.

**Risk factors**
- Event schema disagreements (time-only vs time+channel vs time+AP/ML). Mitigation: define a minimal required schema + optional fields (see component specs).

**Definition of Done**
- [ ] All deliverables completed and committed
- [ ] All new tests passing (unit + integration for this phase)
- [ ] Code review completed (see §7.5 checklist)
- [ ] No regressions in existing cogpy.plot tests
- [ ] Documentation updated (API docs + relevant guides)
- [ ] Event navigation works: click table row → cursor jumps to event time
- [ ] Event overlays render without triggering full recomputation
- [ ] At least 1 real event dataset integrated (bursts from spectrogram bundle)

---

### Phase 5: Multi-Modal Support (Weeks 13–15)

**Goals**
- Introduce modality abstraction and time alignment across sampling rates.

**Deliverables**
- `DataModality` interface + adapters:
  - `GridLFPModality` (xarray `(time,AP,ML)`)
  - `FlatLFPModality` (xarray `(time,channel)`)
  - `SpectrogramModality` (xarray `(time,freq,AP,ML)` or `(time,freq,channel)`)
  - `SpikeTrainsModality` (events-like, with rate/rasters)
- Alignment utilities:
  - shared clock mapping functions (nearest sample, resample hints)
- Example app: LFP traces + bursts + spectrogram slice.

**Dependencies**
- Robust schema validation and controller-driven time.

**Success criteria**
- Switching active modality updates dependent layers without app restart.

**Risk factors**
- Over-engineering. Mitigation: ship only 2 modalities (GridLFP + Events) first; expand iteratively.

**Definition of Done**
- [ ] All deliverables completed and committed
- [ ] All new tests passing (unit + integration for this phase)
- [ ] Code review completed (see §7.5 checklist)
- [ ] No regressions in existing cogpy.plot tests
- [ ] Documentation updated (API docs + relevant guides)
- [ ] At least 2 modalities work: GridLFP + one other (events or spectrogram)
- [ ] Switching active modality updates views without crash
- [ ] Time alignment utilities tested with different sampling rates

---

### Phase 6: Polish & Optimization (Weeks 16–18)

**Goals**
- Performance regression control, orthoslicer consolidation, docs + ergonomics.

**Deliverables**
- Performance benchmarks (window size × channels × latency).
- Consolidation plan for `orthoslicer*.py`:
  - keep `orthoslicer_rangercopy.py` as the reference behavior
  - extract shared utilities into TensorScope slicer layer
  - deprecate duplicate variants or keep as experimental modules
- CLI/config entrypoint for launching TensorScope on a dataset path.
- Docs + tutorial notebooks.

**Success criteria**
- Stable v1.0 workflows (see §7 E2E tests) pass reliably.

**Risk factors**
- Scope creep in “polish” work. Mitigation: time-box Phase 6 workstreams; defer non-critical improvements to post-v1.0.
- Performance tuning without baselines. Mitigation: establish Phase 2 baselines early and treat Phase 6 as regression-controlled improvements.
- Orthoslicer consolidation complexity. Mitigation: consolidate via wrappers first; refactor only with passing benchmarks + tests.

**Definition of Done**
- [ ] All deliverables completed and committed
- [ ] All new tests passing (unit + integration for this phase)
- [ ] Code review completed (see §7.5 checklist)
- [ ] No regressions in existing cogpy.plot tests
- [ ] Documentation updated (API docs + relevant guides)
- [ ] Performance benchmarks established (latency vs window size vs channels)
- [ ] No performance regressions vs Phase 2 baseline
- [ ] CLI launches TensorScope on a file path
- [ ] At least 2 tutorial notebooks completed

---

## 5. Detailed Component Specifications (NEW components)

Below are **new components** required for TensorScope v1.0. Each spec aligns explicitly with `tensorscope-spec.md`:
- separation of concerns (§1.1),
- single source of truth (§2.1),
- thin wrappers (§3.2),
- public APIs (§3.3),
- lifecycle disposal (§10).

### 5.1 Component: `TensorScopeState`

**Purpose**
Central authoritative state for TensorScope; owns controllers and registries; exposes derived properties for layers.

**Location**
`cogpy/core/plot/tensorscope/state.py`

**Public API**
```python
import param
from dataclasses import dataclass
from typing import Any

class TensorScopeState(param.Parameterized):
    # Controllers (owned)
    time_hair = param.Parameter()        # TimeHair
    time_window = param.Parameter()      # TimeWindowCtrl
    channel_grid = param.Parameter()     # ChannelGrid
    processing = param.Parameter()       # ProcessingChain

    # App-level state (authoritative, serializable)
    active_layout_preset = param.String(default="default")
    active_modality = param.String(default="grid_lfp")

    # Registries
    data_registry = param.Parameter()    # DataRegistry
    event_registry = param.Parameter()   # EventRegistry

    # Derived (read-only) convenience
    @property
    def current_time(self) -> float | None: ...
    @current_time.setter
    def current_time(self, v: float | None) -> None: ...

    def register_modality(self, name: str, modality: Any) -> None: ...
    def register_events(self, name: str, events: Any) -> None: ...

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict, *, data_resolver: Any) -> "TensorScopeState": ...
```

**Dependencies**
- Delegates: `TimeHair` (`cogpy.plot.time_player`), `ChannelGrid`, `ProcessingChain`.
- Uses: `DataRegistry`, `EventRegistry`, `TimeWindowCtrl`.

**Design constraints**
- `param.Parameterized` for reactive updates (spec §2.2).
- Own controllers; do not duplicate controller state (spec §2.1).
- Serialization stores **references**, not raw arrays (spec §2.4).

**Testing strategy**
- Unit: delegation correctness (`current_time` ↔ `time_hair.t`).
- Unit: derived selection (grid → flat indices) stable across conventions.
- Unit: serialization round-trip with a mock resolver.

**Implementation priority**
Phase 1.

**Estimated effort**
4–6 days (includes tests + documentation).

---

### 5.2 Component: `TimeWindowCtrl`

**Purpose**
Owns and validates the current visible window `(t0, t1)` and bounds; mediates between view-derived window and state commands.

**Location**
`cogpy/core/plot/tensorscope/time_window.py`

**Public API**
```python
import param

class TimeWindowCtrl(param.Parameterized):
    bounds = param.Range(default=(0.0, 1.0))
    window = param.Range(default=(0.0, 1.0))  # (t0, t1)
    snap = param.Boolean(default=True)

    def set_bounds(self, t_min: float, t_max: float) -> None: ...
    def set_window(self, t0: float, t1: float) -> None: ...
    def recenter(self, t: float, width_s: float) -> None: ...
```

**Dependencies**
- None (pure param object).

**Design constraints**
- Always maintains `t0 < t1` and clamps to bounds.
- Can be driven by view-derived windows without loops (spec §2.5).

**Testing strategy**
- Unit: clipping, ordering, recenter behavior.

**Implementation priority**
Phase 1.

**Estimated effort**
1–2 days.

---

### 5.3 Component: `TensorLayer` (base) + lifecycle tracking

**Purpose**
Standard interface for layers and centralized cleanup via `dispose()` (spec §10).

**Location**
`cogpy/core/plot/tensorscope/layers/base.py`

**Public API**
```python
from abc import ABC, abstractmethod
import panel as pn

class TensorLayer(ABC):
    layer_id: str
    title: str

    def __init__(self):
        self._watchers = []
        self._streams = []
        self._data_refs = []

    @abstractmethod
    def bind(self, state) -> None: ...

    @abstractmethod
    def panel(self) -> pn.viewable.Viewable: ...

    def dispose(self) -> None: ...
```

**Dependencies**
- Panel, and optionally HoloViews streams.

**Design constraints**
- Wrappers only: no “business logic” beyond wiring state to an existing component (spec §3.2).

**Testing strategy**
- Unit: `dispose()` unregisters watchers and clears references.

**Implementation priority**
Phase 2.

**Estimated effort**
2–3 days.

---

### 5.4 Component: `LayerManager` + `LayerSpec`

**Purpose**
Add/remove layers dynamically, keep construction declarative, support plugin registration.

**Location**
`cogpy/core/plot/tensorscope/layers/manager.py`

**Public API**
```python
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class LayerSpec:
    layer_id: str
    title: str
    factory: Callable[[object], "TensorLayer"]  # (state) -> layer

class LayerManager:
    def __init__(self, state): ...
    def register(self, spec: LayerSpec) -> None: ...
    def add(self, layer_id: str) -> "TensorLayer": ...
    def remove(self, layer_id: str) -> None: ...
    def list(self) -> list[str]: ...
```

**Dependencies**
- `TensorLayer`, `TensorScopeState`.

**Design constraints**
- App manages lifecycle; removal must call `dispose()` (spec §10.2).

**Testing strategy**
- Unit: add/remove calls bind/dispose; registry errors are clear.

**Implementation priority**
Phase 2.

**Estimated effort**
2–4 days.

---

### 5.5 Component: `DataRegistry` + Modality adapters

**Purpose**
Store references to session data and provide standardized accessors for layers/controllers (schema normalization lives here, not in views).

**Location**
`cogpy/core/plot/tensorscope/data/registry.py` and `cogpy/core/plot/tensorscope/data/modalities/*.py`

**Public API**
```python
class DataRegistry:
    def register(self, name: str, modality) -> None: ...
    def get(self, name: str): ...
    def list(self) -> list[str]: ...

class GridLFPModality:
    \"\"\"Owns/refs an xr.DataArray with dims (time, AP, ML) after validation.\"\"\"
    def __init__(self, sig_grid): ...
    def time_bounds(self) -> tuple[float, float]: ...
    def to_time_channel(self): ...  # standardized flat view + geometry
```

**Dependencies**
- `xarray`, existing helpers in `cogpy.plot.xarray_hv` and `grid_indexing`.

**Design constraints**
- Do not bake UI dependencies into modality adapters (spec §1.1).
- Validate and normalize dimension conventions early (spec “Explicit over implicit”).

**Testing strategy**
- Unit: validate dims, coordinate normalization, flattening conventions.

**Implementation priority**
Phase 1–2 (start minimal with GridLFP).

**Estimated effort**
6–10 days (iterative).

---

### 5.6 Component: `EventStream` + `EventRegistry`

**Purpose**
Unified events model for table/overlays/navigation, independent of UI implementation.

**Location**
`cogpy/core/plot/tensorscope/events/model.py`

**Public API**
```python
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class EventStream:
    name: str
    df: pd.DataFrame
    time_col: str = "t"
    id_col: str = "event_id"
    # optional columns: channel, AP, ML, label, value, duration, etc.

class EventRegistry:
    def register(self, stream: EventStream) -> None: ...
    def get(self, name: str) -> EventStream: ...
    def list(self) -> list[str]: ...
```

**Dependencies**
- `pandas` (consistent with existing bursts tables).

**Design constraints**
- Minimal required schema: `event_id` + `t` (and optional spatial/channel context).
- Pure model; UI layers interpret it (spec §1.1).

**Testing strategy**
- Unit: schema validation; sorting; lookup by id; window filter helper.

**Implementation priority**
Phase 4.

**Estimated effort**
3–5 days.

---

### 5.7 Component: `LayoutManager` (FastGridTemplate)

**Purpose**
Owns layout presets and persistence; manages adding layer panes into `FastGridTemplate`.

**Location**
`cogpy/core/plot/tensorscope/layout.py`

**Public API**
```python
import panel as pn

class LayoutManager:
    def __init__(self, *, title: str, theme: str = "dark"): ...
    def build_template(self) -> pn.template.FastGridTemplate: ...
    def apply_preset(self, name: str, panes: dict[str, pn.viewable.Viewable]) -> None: ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "LayoutManager": ...
```

**Dependencies**
- Panel templates (`FastGridTemplate`).

**Design constraints**
- Layout persistence stores grid placements and layer IDs, not view internals.

**Testing strategy**
- Integration: apply preset produces expected placements; serialization round-trip.

**Implementation priority**
Phase 3.

**Estimated effort**
4–6 days.

---

### 5.8 Component: `TensorScopeApp` (composition root)

**Purpose**
Composition root managing state, layers, layout, and lifecycle (spec §1.2, §10.2).

**Location**
`cogpy/core/plot/tensorscope/app.py`

**Public API**
```python
import panel as pn

class TensorScopeApp:
    def __init__(self, state, *, config: dict | None = None): ...
    def add_layer(self, layer_id: str) -> "TensorScopeApp": ...
    def remove_layer(self, layer_id: str) -> "TensorScopeApp": ...
    def build(self) -> pn.viewable.Viewable: ...
    def servable(self) -> pn.viewable.Viewable: ...
    def shutdown(self) -> None: ...

    def to_session(self) -> dict: ...
    @classmethod
    def from_session(cls, session: dict, *, data_resolver) -> "TensorScopeApp": ...
```

**Dependencies**
- `TensorScopeState`, `LayerManager`, `LayoutManager`.

**Design constraints**
- App is the owner of layer lifecycle; layers do not find the app globally (spec §1.3).

**Testing strategy**
- Integration: construct app, add core layers, build template, shutdown cleanly.

**Implementation priority**
Phase 0–3 (incremental).

**Estimated effort**
5–8 days.

---

## 6. Migration Strategy

### 6.1 For existing code (notebooks/apps)

**Goal:** gradual migration (spec §3.1).

1. **Phase 2:** Provide `TensorScopeApp.from_ieeg_bundle(...)` convenience that produces an app equivalent to `ieeg_toolkit_app(...)` but using TensorScope layers.
2. Keep `ieeg_toolkit_app.py` and `spectrogram_bursts_app.py` intact as smoke tests and references.
3. Update notebooks to:
   - construct `TensorScopeState` + layers explicitly (no magic globals),
   - embed in FastGridTemplate (as in `notebooks/tensorscope/tensorscope_app.py`).

### 6.2 Backward compatibility
- Do **not** break `ieeg_viewer(...)` initially.
- Introduce TensorScope modules in a new namespace: `cogpy.tensorscope`.
- For any needed changes in existing components (e.g. `MultichannelViewer` public API), keep the old behavior and add new properties/methods.

### 6.3 Deprecation timeline
- Mark orthoslicer variants as:
  - `orthoslicer_rangercopy.py`: “reference implementation”
  - others: “experimental / legacy”
- Legacy plotting modules remain available but are not used by TensorScope core.

---

## 7. Testing Strategy

### 7.1 Per-phase testing

- **Phase 0**
  - Smoke test: app constructs, template builds, returns `pn.viewable.Viewable`.
- **Phase 1**
  - Unit tests: state delegation, time window ctrl, schema validators.
  - Serialization round-trip tests.
- **Phase 2**
  - Integration tests: layer bind/unbind; dispose cleanup; linking (cursor + window).
  - Regression tests for “no private coupling” (public APIs only).
- **Phase 3**
  - Layout preset tests + persistence tests.
- **Phase 4**
  - Event navigation + overlay tests (logic-level; minimal UI checks).
- **Phase 5**
  - Modality switching tests; alignment helpers.
- **Phase 6**
  - Performance benchmarks (non-flaky budgets; record timings, allow margins).

### 7.2 Testing infrastructure
- Prefer `pytest.importorskip` for optional GUI deps (pattern already used in `cogpy/tests/core/plot/test_apps.py`).
- Add a small set of deterministic fixtures via existing `cogpy.datasets.gui_bundles.*`:
  - “small” and “large” mode for iEEG and spectrogram bursts.

### 7.3 Coverage goals
- New TensorScope core modules: **80%+** unit coverage.
- Integration coverage: all state synchronization paths (cursor, window, selection, processing).
- E2E workflows (3–5):
  1. Load iEEG bundle → select grid channels → apply → traces update.
  2. Click timeseries → cursor updates → spatial map updates.
  3. Toggle processing → bounded re-render of current window.
  4. Select event in table → jump cursor/window → overlays update.
  5. Save session → reload session → state + layout restored.

### 7.4 Test Data Strategy

**Use existing cogpy.datasets fixtures:**

**Small datasets (for unit tests):**
- `example_ieeg_grid(mode="small")` - 8×8 grid, 10s, 1kHz (~50KB)
- Fast to load, fast to process
- Use for: schema validation, layer construction, state serialization

**Medium datasets (for integration tests):**
- `example_ieeg_grid(mode="large")` - 8×8 grid, 60s, 1kHz (~300KB)
- Use for: windowed processing, performance budgets, layer linking

**Large datasets (for performance tests, generated on-demand):**
- Synthetic: 8×8 grid, 600s (10min), 1kHz (~3MB)
- Generate programmatically in `conftest.py` fixtures
- Use for: memory leak tests, performance benchmarks
- **DO NOT commit to repo** (too large)

**Event datasets:**
- `spectrogram_bursts_bundle()` provides known-good burst events
- Use for: event navigation, overlay rendering, table interaction tests

**Multi-modal datasets (Phase 5):**
- Pair GridLFP with synthetic spike trains (generate in tests)
- Synthetic spectrogram (via processing chain on LFP)
- Use for: modality switching, time alignment

**Guidelines:**
- Keep committed test data < 1MB total
- Generate large fixtures in tests (deterministic seeds)
- Document expected properties (e.g., "burst at t=30s")
- Reuse existing `cogpy.datasets` bundles where possible

### 7.5 Code Review Checklist

Every pull request must verify these criteria before merge:

**Architecture compliance (per tensorscope-spec.md):**
- [ ] No component imports another component directly (only via state/dependency injection)
- [ ] No use of `._private` attributes from other modules (public APIs only)
- [ ] State updates flow unidirectionally (user → state → listeners → views)
- [ ] No circular dependencies between modules

**Performance (per spec §7):**
- [ ] No `.compute()` calls on unbounded data (only windowed slices)
- [ ] All processing respects `time_window` (bounded compute)
- [ ] Downsampling budget enforced where applicable (e.g., MultichannelViewer max_points)
- [ ] No operations with O(n_samples) complexity on full dataset

**Lifecycle (per spec §10):**
- [ ] Every `param.watch()` has corresponding `param.unwatch()` in `dispose()`
- [ ] Every HoloViews/Bokeh stream has `.clear()` or equivalent in `dispose()`
- [ ] Large data arrays (>1MB) are explicitly deleted/dereferenced in `dispose()`
- [ ] Layers track watchers/streams in `_watchers`/`_streams` lists for cleanup

**Testing (per spec §6):**
- [ ] New components have unit tests (target: 80%+ coverage)
- [ ] Integration tests for state bindings (param.watch, streams)
- [ ] No flaky tests (run `pytest -x 10` to verify stability)
- [ ] Test data fixtures are deterministic (fixed seeds)

**Documentation (per spec §8):**
- [ ] Public APIs have docstrings with parameter descriptions
- [ ] Docstrings include usage examples (runnable code)
- [ ] Design decisions documented (why this approach, not what the code does)
- [ ] Breaking changes noted in module docstring and CHANGELOG

**Code quality:**
- [ ] Type hints on all public functions/methods
- [ ] No TODO/FIXME/HACK comments without issue tracker links
- [ ] Follows existing cogpy code style (imports, naming, structure)
- [ ] No unused imports or variables

**PR description must include:**
- [ ] Which phase this belongs to
- [ ] What component(s) are added/modified
- [ ] How to test manually (if applicable)
- [ ] Screenshots/videos for UI changes

---

## 8. Documentation Plan

### User documentation
- “Getting Started”: launch TensorScope on a demo bundle.
- “Core workflows”: selection, cursor navigation, window navigation, processing toggles, event navigation.

### Developer documentation
- Architecture guide: state/controller/view split (link to `tensorscope-spec.md` sections).
- “How to write a layer”: interface + lifecycle + wiring examples.
- “How to add a modality” and “How to add an event source”.

### Documentation deliverables (phase-by-phase)
- Phase 0–1: add minimal docs + a “TensorScope Quickstart” snippet.
- Phase 2: document core layer wrappers and state APIs.
- Phase 3: document presets + session files.
- Phase 4: document event stream schema and overlay conventions.

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| Panel/HoloViews API changes | Medium | High | Pin versions; add smoke tests; validate upgrades in CI. |
| Performance regressions on large data | High | High | Windowed compute only; set explicit budgets; add perf tests early. |
| Private coupling between components | Medium | High | Add public APIs (e.g., `MultichannelViewer.range_stream`); lint/grep checks in review. |
| Bokeh “single document” errors | Medium | Medium | Enforce `fresh=True` patterns; avoid reusing Bokeh models across outputs. |
| Dim/coord convention drift (AP/ML order) | High | High | Central validators + adapters; tests for flattening and mapping. |
| Memory leaks from watchers/streams | Medium | High | Mandatory `dispose()`; tests that add/remove layers repeatedly. |
| Over-engineering plugin system too early | Medium | Medium | Start with simple `LayerSpec` registry; expand only if needed. |

---

## 10. Open Questions → RESOLVED DECISIONS

All critical design questions have been resolved for v1.0:

### 10.1 RESOLVED: Spec filename
**Decision**: The spec is correctly named `tensorscope-spec.md`. All references updated.

### 10.2 RESOLVED: Flattening convention (was Open Question #2)
**Decision**: Row-major (`channel = ap * n_ml + ml`) is MANDATORY for TensorScope.
- Document in `tensorscope/schema.py`
- Add explicit test in Phase 1 (`test_flattening_convention()`)
- All data adapters must enforce this convention
- Column-major is NOT supported in v1.0

**Rationale**: Single standard prevents bugs. Row-major matches Neuralynx convention (per ieeg-toolkit-spec.md).

### 10.3 RESOLVED: Events schema (was Open Question #3)
**Decision**: Minimal required schema for `EventStream`:

**REQUIRED fields:**
- `event_id` (int or str, unique identifier)
- `t` (float, timestamp in seconds)

**OPTIONAL fields:**
- `channel` (int, flat channel index)
- `AP` (int, grid row)
- `ML` (int, grid column)
- `label` (str, event type/category)
- `value` (float, amplitude/score)
- `duration` (float, seconds)

**Validation**: Phase 4 validators check REQUIRED fields only. Optional fields are pass-through.

**Rationale**: Flexibility for different event types (point events, intervals, spatial events) while ensuring minimal table/navigation works.

### 10.4 RESOLVED: Large dataset strategy (was Open Question #4)
**Decision**: v1.0 uses windowed xarray with `.compute()` only.
- NO Dask/Zarr integration in v1.0 (post-v1.0 enhancement)
- If dataset is too large for windowed compute, user must downsample/rechunk before loading
- Set explicit memory budgets in Phase 6 benchmarks

**Rationale**: Simplicity first. Dask adds complexity. Window-based approach works for 95% of use cases.

### 10.5 RESOLVED: Offline mode (was Open Question #5)
**Decision**: v1.0 is Panel server only (no static HTML export).
- Panel `.servable()` is the primary deployment
- Static HTML export is a post-v1.0 feature
- Don't constrain architecture for static export yet

**Rationale**: Panel server is the mature path. Static export has limitations (no interactivity). Ship server-based v1.0 first.

---

## 11. Future Enhancements (Post-v1.0)

- Streaming/real-time data ingestion.
- Collaborative annotation and multi-user sessions.
- Advanced event discovery tools (AI-assisted detectors).
- GPU-accelerated rendering paths for ultra-long recordings.
- Rich plugin ecosystem (entry points) once core APIs stabilize.

## 12. Pre-Development Checklist

Before starting Phase 0 implementation, verify:

**Plan finalized:**
- [ ] All open questions resolved (Section 10)
- [ ] All refinements from review incorporated
- [ ] Team sign-off on plan received

**Environment ready:**
- [ ] Panel 1.8.8+ installed and tested
- [ ] HoloViews + Bokeh compatible versions verified
- [ ] cogpy development environment working
- [ ] Can run existing `cogpy.plot` tests successfully

**Repository ready:**
- [ ] Create feature branch: `feature/tensorscope-foundation`
- [ ] Branch is up-to-date with main
- [ ] CI/CD pipeline runs on branch

**Team alignment:**
- [ ] Developers assigned to Phase 0
- [ ] Code review protocol established
- [ ] Communication channel set up (Slack/Discord/etc)
- [ ] Weekly sync meeting scheduled

**Documentation:**
- [ ] This plan accessible to all developers
- [ ] tensorscope-spec.md reviewed by team
- [ ] Contributing guidelines reviewed

**Ready to proceed when all boxes checked.**

---

**Next step: Begin Phase 0 (Foundation)**

See Phase 0 specification in Section 4 for detailed deliverables.
