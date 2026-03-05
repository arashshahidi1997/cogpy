# Additional Detectors Specification v2.6.4

## Version
- Version: 2.6.4
- Date: 2026-03-05
- Depends on: v2.6.1 (`cogpy.core.detect.EventDetector`), v2.6.0 (`cogpy.core.events.EventCatalog`)

## Overview

v2.6.4 expands the detector library beyond `BurstDetector` by adding:

- `ThresholdDetector`: generic threshold-crossing detector (interval events)
- `RippleDetector`: bandpass + Hilbert envelope + z-score + dual threshold (interval events)
- `SpindleDetector` (optional): spindle-band wrapper with longer duration constraints

All detectors return `EventCatalog` and are compatible with TensorScope v2.6.2+ (`TensorScopeState.run_detector()` / overlays / event explorer).

## Components

### Detection utilities

**File:** `cogpy/core/detect/utils.py`

Shared helpers for:
- bandpass filter (`bandpassx`)
- Hilbert envelope (SciPy)
- safe z-scoring
- run/interval finding and merging
- dual-threshold interval event extraction

### ThresholdDetector

**File:** `cogpy/core/detect/threshold.py`

Produces interval events by finding contiguous runs above a threshold:
- `direction="positive"`: `x >= abs(threshold)`
- `direction="negative"`: `x <= -abs(threshold)` (or `<= threshold` if threshold is already negative)
- `direction="both"`: `abs(x) >= abs(threshold)`

Supports:
- optional bandpass filtering
- optional Hilbert envelope
- optional merging by `merge_gap`
- filtering by `min_duration`

### RippleDetector

**File:** `cogpy/core/detect/ripple.py`

Pipeline:
1. bandpass filter (`freq_range`)
2. Hilbert envelope
3. z-score
4. dual threshold (`threshold_low`, `threshold_high`)
5. duration constraints (`min_duration`, `max_duration`)

### SpindleDetector (optional)

Implemented as a wrapper over `RippleDetector` with default spindle-band parameters (11–16 Hz) and longer duration constraints (0.5–3.0 s).

## Success Criteria

- Detectors are importable from `cogpy.core.detect`.
- Each detector returns a valid `EventCatalog` (with intervals for ripple/spindle).
- Unit tests cover basic detection + serialization.
- Example script demonstrates running multiple detectors.

