# Architecture And Design Principles

This document captures the current design direction for `cogpy` as we refactor
preprocess pipelines and shared utilities. It focuses on keeping compute logic
clean and reusable while consolidating file-format and sidecar handling into IO.

## Goals

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

## Open Questions

- Standardize attribute names for sampling rate and channel labels.
- Confirm a minimal schema for the internal signal object.
- Decide when IO should return `DataArray` vs `Dataset`.
