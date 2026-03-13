# cogpy Design Specification

This document is the single authoritative reference for cogpy's design: the data
model, processing abstractions, and module contracts. It consolidates decisions
from earlier per-version specs into one living document.

## 1. Purpose

cogpy exists to provide **composable, file-agnostic compute primitives** for
ECoG / iEEG signal analysis, backed by a structured I/O layer and reproducible
pipelines. It also serves as the **backend API** for visualization frontends
(TensorScope, React + TypeScript).

## 2. Data Model

All core compute operates on `xarray.DataArray` (or `xarray.Dataset`) objects
with standardized dimension names and metadata. This section defines the
canonical schemas.

### 2.1 Signal Schemas (`cogpy.core.base.ECoGSchema`)

| Schema | Dims | Use case |
|--------|------|----------|
| Grid ECoG | `("time", "AP", "ML")` | Spatial analysis on 2D electrode arrays |
| Flat ECoG | `("time", "ch")` | Channel-indexed analysis without grid semantics |
| Multichannel | `("channel", "time")` | Generic multichannel (no grid) |

**Required metadata:**
- `fs` (float) — sampling rate in Hz. Accessible as a 0-D coordinate named
  `"fs"` or as `attrs["fs"]`. Validated by `cogpy.core.base.ensure_fs()`.
- `time` coordinate — seconds, 1-D, strictly increasing.

**Optional metadata:**
- `units` (str) — e.g. `"uV"`
- `AP`, `ML` coordinates — physical positions (mm), 1-D

### 2.2 Spectrogram Schema

| Entity | Dims | Notes |
|--------|------|-------|
| `GridSpectrogram4D` | `("ml", "ap", "time", "freq")` | For orthoslicer / spatial-spectral views |

Coordinates: `ml`, `ap` (physical), `time` (seconds), `freq` (Hz, increasing).

### 2.3 Event Representations

| Entity | Type | Required fields |
|--------|------|-----------------|
| `EventCatalog` | `pandas.DataFrame` wrapper | `event_id`, `t` (seconds) |
| `BurstPeaksTable` | `pandas.DataFrame` | `burst_id`, `x`, `y`, `t`, `z`, `value` |

**EventCatalog optional columns:**
- Interval: `t0`, `t1`, `duration`
- Spatial: `channel`, `AP`, `ML`
- Spectral: `freq`, `f0`, `f1`, `bandwidth`
- Provenance: `label`, `score`, `detector`, `pipeline`

**Converters:** `to_events()`, `to_intervals()`, `to_point_intervals()`,
`to_event_stream()` (for visualization).

### 2.4 Coercion and Validation

Boundary functions in `cogpy.datasets.schemas` enforce schemas at entry points:
- `validate_*()` — raise `ValueError` with hints on mismatch.
- `coerce_*()` — fix "almost-right" inputs (dim permutations, missing `fs`)
  before validation.

Core compute functions should **not** coerce internally; validation happens at
boundaries (I/O, CLI, GUI entry points).

## 3. Processing Framework

### 3.1 Layering

```
cogpy.io          Load / save / sidecar management
    ↓ xarray
cogpy.core        Pure compute (filtering, spectral, detection, …)
    ↓ xarray + EventCatalog
cogpy.cli / wf    Thin orchestration (Snakemake, argparse)
    ↓ API
Frontend          TensorScope (React + TS), notebooks
```

Functions in `cogpy.core` never touch the filesystem. Functions in `cogpy.io`
never do heavy compute. Pipelines compose both.

### 3.2 Preprocessing Stack

**Canonical modules** (under `cogpy.core.preprocess`):

| Module | Responsibility |
|--------|---------------|
| `filtering/` | Temporal/spatial filtering (xarray + Dask aware); `filtx` is a compat shim |
| `badchannel/channel_features` | Per-channel feature extraction |
| `badchannel/spatial` | Spatial normalization (neighborhood statistics) |
| `badchannel/pipeline` | Sliding-window feature-map orchestration |
| `badchannel/badlabel` | Outlier labeling (DBSCAN) |
| `linenoise` | Line-noise removal |
| `resample` | Downsampling / decimation |
| `interpolate` | Bad-channel interpolation |

Legacy modules (`channel_feature_functions`, `channel_feature`, `detect_bads`)
remain for backward compatibility but are not the target for new code.

### 3.3 Spectral Analysis

| Function | Module | Description |
|----------|--------|-------------|
| `psdx()` | `core.spectral.specx` | Power spectral density (Welch / multitaper) |
| `spectrogramx()` | `core.spectral.specx` | Time–frequency spectrogram |
| `coherencex()` | `core.spectral.specx` | Coherence between channels |

All accept `xarray.DataArray` and return `xarray.DataArray` with appropriate
frequency/time-frequency dimensions.

### 3.4 Detection Framework

Detection is built on three abstractions:

**EventDetector** (`cogpy.core.detect.base`):
- `detect(data) -> EventCatalog`
- `can_accept(data) -> bool`
- `needs_transform(data) -> bool` (smart transform: accept raw or precomputed)
- Serializable via `to_dict()` / `from_dict()`

**Concrete detectors:**

| Detector | Input | Output | Wraps |
|----------|-------|--------|-------|
| `BurstDetector` | spectrogram or raw signal | point events | `detect_hmaxima` |
| `ThresholdDetector` | 1-D signal | interval events | contiguous-run finder |
| `RippleDetector` | raw signal | interval events | bandpass → envelope → z-score → dual threshold |
| `SpindleDetector` | raw signal | interval events | `RippleDetector` with spindle-band defaults |

**Transform** (`cogpy.core.detect.transforms`):
- `compute(data) -> xr.DataArray`
- Concrete: `BandpassTransform`, `HighpassTransform`, `LowpassTransform`,
  `SpectrogramTransform`, `HilbertTransform`, `ZScoreTransform`

**DetectionPipeline** (`cogpy.core.detect.pipeline`):
- Chains transforms + detector into a single reproducible unit.
- `run(data) -> EventCatalog`
- Adds provenance to output metadata.
- Serializable via `to_dict()` / `from_dict()`.

**Pre-built pipelines** (`cogpy.core.detect.pipelines`):
`BURST_PIPELINE`, `RIPPLE_PIPELINE`, `FAST_RIPPLE_PIPELINE`, `GAMMA_BURST_PIPELINE`

## 4. I/O Layer

| Module | Formats | Key functions |
|--------|---------|---------------|
| `io.ecog_io` | Binary `.lfp` + XML metadata | `from_file()`, `from_arr()` |
| `io.ieeg_io` | BIDS-iEEG | `from_file()` |
| `io.ecephys_io` | NWB-style ecephys | `from_file()`, `load_ecephys_metadata()` |
| `io.converters` | Zarr, DAT | `bids_lfp_to_zarr()`, `zarr_to_dat()` |
| `io.sidecars` | JSON metadata | `propagate_sidecars()`, `update_sampling_frequency_json()` |

I/O is also responsible for constructing valid `xarray.DataArray` objects with
correct schemas from raw file data.

## 5. Datasets & Fixtures

`cogpy.datasets` provides deterministic synthetic data for testing and GUI
development:

- **Entity generators** (`datasets.entities`) — single arrays matching schemas above.
- **Bundles** (`datasets.gui_bundles`) — coordinated collections for GUI dev:
  - `ieeg_grid_bundle()` → grid signal + stacked view + RMS scalar + atlas hook
  - `spectrogram_bursts_bundle()` → 4D spectrogram + burst peaks
- **Modes** — `"small"` (fast debug) and `"large"` (stress-test rendering).
- All accept `seed` for reproducibility.

## 6. Open Design Questions

- Finalize canonical dim order: `("time", "AP", "ML")` vs `("time", "ML", "AP")`
  — current code has both; spec target is `("time", "AP", "ML")`.
- Standardize `fs` as coordinate vs attribute (currently both are accepted).
- Define when `from_file()` should return `DataArray` vs `Dataset`.
- Define a public API surface for TensorScope backend (which functions/schemas
  the React frontend should depend on).
