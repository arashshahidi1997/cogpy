# Detector Architecture Specification v2.6.1

## Version
- Version: 2.6.1
- Date: 2026-03-05
- Status: Implementation
- Depends on: v2.6.0 (`cogpy.events.EventCatalog`)

## Overview

This specification defines a unified detector interface for event detection in CogPy. Detectors wrap existing, battle-tested detection functions (e.g., `detect_hmaxima`) behind a standardized interface that returns `cogpy.events.EventCatalog`.

Key goals:
- Wrap existing detectors instead of reimplementing them.
- Standardize outputs as `EventCatalog`.
- Support “smart transforms”: accept either precomputed transforms (explicit) or raw signals (implicit transform computed internally).
- Provide serializable, declarative configuration (`to_dict` / `from_dict`).

## Design Principles

### Wrap, Don’t Reimplement

Existing detection code already exists across:
- `cogpy.burst.blob_detection` (`detect_hmaxima`, `detect_blobs`, …)
- `cogpy.spectral.spectrogram_burst` (blob candidates + aggregation)

Detectors provide:
- `detect(data) -> EventCatalog`
- optional transform handling
- metadata/provenance fields

### Smart Transform Handling

Some detectors need transforms (e.g., spectrogram) before detection:

```text
Raw LFP -> Spectrogram -> Detect peaks
```

Detectors should support:
- **Explicit mode:** user provides precomputed spectrogram (no recomputation).
- **Implicit mode:** user provides raw signal; detector computes the needed transform internally.

### Declarative Configuration

Detectors must be serializable:

```python
cfg = detector.to_dict()
det2 = BurstDetector.from_dict(cfg)
```

## EventDetector Base Class

Required:
- `detect(data: xr.DataArray, **kwargs) -> EventCatalog`
- `get_event_dims() -> list[str]`

Optional:
- `can_accept(data) -> bool`
- `needs_transform(data) -> bool`
- `get_transform_info() -> dict`

## BurstDetector

`BurstDetector` is the first concrete detector.

Wraps:
- `cogpy.burst.blob_detection.detect_hmaxima`

Accepts:
- spectrogram-like input (`"freq"` in `data.dims`) — explicit mode
- raw time-series input (`"time"` in `data.dims` and `"freq"` not in dims) — implicit mode

Implicit transform:
- uses `cogpy.spectral.specx.spectrogramx` with detector parameters

Returns:
- `cogpy.events.EventCatalog` (point events) via `EventCatalog.from_hmaxima(...)`

## Success Criteria

- `cogpy.detect.EventDetector` base implemented
- `cogpy.detect.BurstDetector` implemented with explicit + implicit modes
- tests pass for base + burst detector
- demo example shows both modes

