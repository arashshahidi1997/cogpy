# Orthoslicer Consolidation Plan

TensorScope v1.0 provides a modern, state-driven alternative to the legacy
orthoslicer family. This document captures the **recommended** orthoslicer
variant for maintenance work and outlines a **non-breaking** consolidation plan.

## Current State

Multiple orthoslicer implementations exist with overlapping functionality:

- `orthoslicer.py` — core slicer (XY + TZ)
- `orthoslicer_zoom.py` — adds zoom persistence
- `orthoslicer_ranger.py` — adds RangeTool-style time window
- `orthoslicer_rangercopy.py` — linked time-window view (most stable)
- `orthoslicer_bursts.py` — adds event/burst overlays
- `orthoslicer_bursts_timeseries.py` — adds multichannel timeseries
- `orthoslicer_facet.py` — adds faceting/local sampling helpers
- `orthoslicer/base.py` — folder scaffold for refactor

## Issues

1. **Duplication**: substantial code overlap across variants.
2. **Maintenance tax**: bug fixes require updating multiple files.
3. **User confusion**: unclear which module to use for new work.
4. **Import-time side effects**: some variants initialize plotting backends at import time.

## Recommended Strategy

### Phase 6a (Now): Document best practice + deprecate variants

**Best practice for maintenance** (non-breaking):

- Use `orthoslicer_rangercopy.py` for existing orthoslicer-based workflows.
- Use TensorScope for **new** projects.

For variants other than `orthoslicer_rangercopy.py`, add `DeprecationWarning`
at import time so new code paths don’t start depending on them.

### Phase 6b (Post v1.0): TensorScope alternative

Add a dedicated TensorScope layer that wraps the best orthoslicer behavior:

- `TensorSlicerLayer` (future)
  - wraps `orthoslicer_rangercopy` behavior
  - integrates with `TensorScopeState`
  - uses TensorScope’s layer lifecycle (`dispose()`)
  - integrates event overlays via TensorScope’s event layers

### Phase 6c (Long-term): composable slicer layers

Move from “one big orthoslicer” to composable layers:

- `SlicerBaseLayer` (core XY/TZ/YZ views)
- `SlicerZoomLayer` (persistent zoom)
- `SlicerWindowLayer` (time window selection)
- `SlicerEventsLayer` (event overlays)

## Migration Guide

### Existing orthoslicer code (short-term)

```python
from cogpy.core.plot.orthoslicer_rangercopy import OrthoSlicerRanger
```

### New development (now)

```python
from cogpy.core.plot.tensorscope import TensorScopeApp
app = TensorScopeApp(data).add_layer("timeseries").add_layer("spatial_map")
```

### Future (v2+)

```python
# app.add_layer("tensor_slicer")  # planned
```

## Decision: Phase 6 Scope

**In scope**
- Deprecation warnings on older variants
- Documentation of recommended variant
- Encourage TensorScope usage in new examples

**Out of scope**
- Breaking consolidation / file removals
- Full orthoslicer rewrite as TensorScope layers

Rationale: ship TensorScope v1.0 first, then consolidate with user feedback.

