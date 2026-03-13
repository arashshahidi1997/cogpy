# TensorScope v3.0 Specification: Tensor-Centric Foundation

## Version

- Version: 3.0.0
- Date: 2026-03-06
- Depends on: TensorScope v2.8 design lineage (breaking refactor)

## Overview

TensorScope v3 refactors from **tool-/module-centric** architecture to a **tensor-centric** foundation.

Core concepts:

- **Tensor**: Named `xarray.DataArray` plus lineage metadata.
- **State**: Single source of truth (tensor registry + global selection).
- **Views**: Pure projections (read-only), discovered by tensor dimension signature.
- **Tabs**: UI tabs correspond to registered tensors.

Out of scope for v3.0:

- Transform DAG (planned v4.0)
- Caching/memoization (planned v4.0)
- Dynamic view builder (planned v3.1)
- Real-time streaming (planned v5.0)

## Data Model

### TensorNode

Implemented in `code/lib/cogpy/src/cogpy/core/plot/tensorscope/state.py` as `TensorNode`:

- `name: str`
- `data: xr.DataArray`
- `source: str | None`
- `transform: str`
- `params: dict`

### TensorRegistry

Implemented in `code/lib/cogpy/src/cogpy/core/plot/tensorscope/state.py` as `TensorRegistry`:

- `add(node)`
- `get(name)`
- `list()`
- `lineage_str(name)`

## State Model

Implemented in `code/lib/cogpy/src/cogpy/core/plot/tensorscope/state.py` as `TensorScopeState`:

- `tensors: TensorRegistry`
- `selection: SelectionState`
- `active_tensor: str`

Selection is a single reactive object:

Implemented in `code/lib/cogpy/src/cogpy/core/plot/tensorscope/state.py` as `SelectionState(param.Parameterized)`:

- `time: float` (seconds)
- `freq: float` (Hz)
- `ap: int`
- `ml: int`
- `channel: int | None`

## View Model

Views are pure projections:

`view = project(tensor, selection) -> visualization`

Invariants:

- Views **never mutate tensors**
- Views **never mutate selection**

Implemented in `code/lib/cogpy/src/cogpy/core/plot/tensorscope/views/__init__.py`.

### View Registry

Dimension-based registry maps tensor dimension signatures to view classes.

Canonical signatures:

- `('time', 'AP', 'ML')` → signal views
- `('freq', 'AP', 'ML')` → PSD views
- `('time', 'freq', 'AP', 'ML')` → spectrogram views (deferred)

Note: some spectral utilities output dimensions in different orders (e.g. `('AP','ML','freq')`).
View discovery in v3.0 canonicalizes dimension *sets* into canonical signatures.

## UI Structure

Implemented in `code/lib/cogpy/src/cogpy/core/plot/tensorscope/app.py` as `TensorScopeApp`:

- Tensor tabs (one per registered tensor)
- Fixed v3.0 layout (first two registered views per tensor)
- Sidebar with selection controls (v3.0 minimal: time + freq sliders)

## Success Criteria

- [ ] Unified `TensorScopeState` (tensor registry + selection)
- [ ] Tensor tabs (Signal/PSD) work
- [ ] Selection changes propagate to views
- [ ] Views never mutate tensor/selection
- [ ] View discovery based on tensor dimensions
- [ ] Example app runs via `panel serve`
