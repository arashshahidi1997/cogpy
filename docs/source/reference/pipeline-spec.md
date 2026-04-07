# Detection Pipeline Specification v2.6.5

## Version
- Version: 2.6.5
- Date: 2026-03-05
- Depends on: v2.6.1 (EventDetector), v2.6.4 (multiple detectors)

## Overview

Detection pipelines formalize multi-stage detection workflows as composable objects. A pipeline chains **transforms** (spectrogram, filtering, envelope, normalization) with a **detector**, making workflows reproducible and serializable.

## Components

### Transform base class

**File:** `cogpy/core/detect/transforms/base.py`

- `Transform.compute(data) -> xr.DataArray`
- `Transform.to_dict()` / `Transform.from_dict(...)` for serialization

### Concrete transforms

**Files:**
- `cogpy/core/detect/transforms/spectral.py` (`SpectrogramTransform`)
- `cogpy/core/detect/transforms/filtering.py` (`BandpassTransform`, `HighpassTransform`, `LowpassTransform`)
- `cogpy/core/detect/transforms/envelope.py` (`HilbertTransform`, `ZScoreTransform`)

### DetectionPipeline

**File:** `cogpy/core/detect/pipeline.py`

`DetectionPipeline` applies transforms in order, then runs a detector:

- `run(data)` returns `EventCatalog` if a detector is configured, otherwise the transformed `xr.DataArray`.
- Adds pipeline provenance into `EventCatalog.metadata` (`pipeline`, `transforms`, `detector`).
- Supports `to_dict()` / `from_dict()` for JSON-friendly configuration.

### Pre-built pipelines

**File:** `cogpy/core/detect/pipelines.py`

Provided presets:
- `BURST_PIPELINE` (spectrogram → BurstDetector)
- `RIPPLE_PIPELINE` (bandpass → envelope → zscore → ThresholdDetector)
- `FAST_RIPPLE_PIPELINE`
- `GAMMA_BURST_PIPELINE`

## Success Criteria

- Pipelines are serializable and reloadable (`to_dict` / `from_dict`).
- Presets run (subject to their transform dependencies).
- Pipeline outputs (`EventCatalog`) are compatible with visualization frontends.

