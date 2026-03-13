# Architecture And Design Principles

This document captures the design direction for `cogpy`: what the package is for,
how its pieces fit together, and what conventions new code should follow.

## Package Goals

`cogpy` is a Python toolkit for **ECoG / iEEG signal processing and analysis**.
Its purpose is to provide:

1. **Reusable compute primitives** — filtering, spectral analysis, event detection,
   bad-channel identification, brain-state classification — that operate on
   `xarray.DataArray` objects with standardized schemas.
2. **Structured I/O** — readers/writers for common electrophysiology formats
   (binary LFP, Zarr, BIDS-iEEG) with sidecar and metadata management.
3. **Reproducible preprocessing pipelines** — Snakemake workflows that compose
   I/O and compute into auditable, configurable processing chains.
4. **A backend API for visualization tools** — cogpy serves as the data and
   compute backend for frontends such as
   TensorScope (see note below).

### What cogpy is *not*

- A GUI application. Interactive visualization has migrated to a separate
  React + TypeScript project (TensorScope) that calls cogpy as its backend.
- A storage-format authority. Format decisions belong to the I/O layer; core
  algorithms are format-agnostic.

## Design Principles

- Keep core processing code file-agnostic and testable.
- Centralize IO and file-format concerns in `cogpy.io`.
- Avoid combinatorial growth of functions that mix IO and compute.
- Enable Snakemake pipelines to remain thin orchestrators.

## Layering

`cogpy` is organized into two primary layers:

- `cogpy.core`: Pure compute on in-memory structures.
- `cogpy.io`: File reading/writing and format translation.

Pipelines or scripts should assemble workflows by calling `cogpy.io` to load
data and then pass data to `cogpy.core` for transformation. Outputs are saved
through `cogpy.io`.

## Core Data Model

Core functions should operate on an internal signal representation, not files.
Current expectation is an `xarray.DataArray` (or `xarray.Dataset`) with:

- A time-like dimension.
- Metadata attributes for sampling rate and channel info.
- Optional coordinates for channels and time.

IO modules are responsible for constructing this representation from files and
for converting it back to external formats when needed.

## IO Responsibilities

IO modules are responsible for:

- Loading raw inputs into the internal representation.
- Managing sidecars and metadata updates.
- Resolving paths and file-type specifics.

Example: updating JSON sidecars belongs in `cogpy.io` (not `cogpy.core`), even if
the update happens alongside a compute step like resampling.

## Avoiding Combinatorial Functions

Functions that mix IO and compute (e.g., "downsample and update sidecar") scale
poorly as inputs/outputs grow. Prefer:

1. Core functions that accept the internal signal object.
2. IO helpers that load/save specific formats and update sidecars.
3. Lightweight pipeline glue that composes the two.

If a convenience wrapper is needed, put it in `cogpy.io` and keep it thin.

## Snakemake Guidance

Snakemake rules should:

- Use `cogpy.io` to load inputs and update sidecars.
- Use `cogpy.core` for transformations.
- Avoid custom per-pipeline logic when a shared IO helper exists.

## Preprocessing Structure

The preprocessing module is organized into focused subpackages:

### Filtering (`cogpy.core.preprocess.filtering/`)

xarray-native signal filters, split by domain:

- `temporal.py` — Butterworth bandpass/lowpass/highpass, notch, decimation
- `spatial.py` — Gaussian, median, median-highpass over (AP, ML) grids
- `reference.py` — Common median reference (CMR)
- `normalization.py` — Z-score normalization

The legacy `filtx.py` module is a backward-compatible shim that re-exports
everything from `filtering/`.

### Bad-Channel Detection (`cogpy.core.preprocess.badchannel/`)

For channel-feature workflows, use the canonical badchannel stack:

- `channel_features` — raw feature functions and xarray wrappers
- `pipeline` — sliding feature-map orchestration
- `spatial` — spatial normalization relative to grid neighbors
- `badlabel` — DBSCAN outlier labeling
- `grid` — grid adjacency construction

### Legacy Modules (deprecated)

These emit `DeprecationWarning` on import. Use `badchannel` instead:

- `channel_feature_functions` → `badchannel.channel_features`
- `channel_feature` → `badchannel.pipeline`
- `detect_bads` → `badchannel.badlabel`

## TensorScope: Legacy Subpackage

TensorScope was originally developed as a Panel/HoloViews interactive viewer
inside `cogpy.core.plot.tensorscope`. It has since **migrated to its own
standalone React + TypeScript project**, with cogpy serving as the data and
compute backend via a Python API.

The `cogpy.core.plot.tensorscope` subpackage and its associated specification
documents (v2.x, v3.x) are retained for historical reference but are no longer
the active development target. New visualization work should target the
standalone TensorScope frontend and use cogpy's public API for data access.

## Open Questions

- Standardize attribute names for sampling rate and channel labels.
- Confirm a minimal schema for the internal signal object.
- Decide when IO should return `DataArray` vs `Dataset`.
