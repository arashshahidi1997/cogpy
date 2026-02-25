# GUI Development Bundles: Intentions & Specs

This document specifies a **bundle layer**: small, deterministic collections of entities that are known to work together.
Bundles exist to accelerate GUI development by providing “one import” fixtures with consistent coordinates, shapes, and optional overlays.

## Bundle principles

- **Schema-first**: every field has a declared schema (see `schemas.md`).
- **Reproducible**: bundles accept `seed` and are stable across runs.
- **Two modes**: `mode="small"` and `mode="large"` (see `modes.md`).
- **GUI-ready**: precompute the small derived items GUIs commonly need:
  - grid index mapping,
  - per-channel scalars for heatmaps (e.g. RMS),
  - optional atlas image + placement metadata.

## Implemented public API

### `cogpy.datasets.gui_bundles.ieeg_grid_bundle(...) -> IEEGGridGuiBundle`

**Intent**
- Provide a grid-shaped time series plus derived forms needed by the current Panel viewers:
  - `ChannelGridWidget` wants `(AP, ML)` background values and optional atlas data.
  - `ieeg_viewer` typically wants a time×channel view (stacked) and/or a grid-wired selector.

**Signature (current)**
- `ieeg_grid_bundle(*, mode="small", seed=0, with_atlas=False, large_backend="numpy")`

**Return type**
- `cogpy.datasets.gui_bundles.IEEGGridGuiBundle` (dataclass)

**Return fields (current)**
- `sig_grid: xr.DataArray` — `IEEGGridTimeSeries` dims `("time","ML","AP")`
- `sig_apml: xr.DataArray` — derived view dims `("time","AP","ML")` used for consistent flattening/stacking
- `sig_tc: xr.DataArray` — stacked view dims `("time","channel")` via `sig_apml.stack(channel=("AP","ML"))`
- `rms_apml: np.ndarray` — `(n_ap, n_ml)` scalar for grid background (std over time)
- `n_ap, n_ml: int` — grid shape
- `fs: float | None` — sampling rate if present in attrs
- `ap_coords, ml_coords: np.ndarray` — physical coords copied from `sig_grid`
- `atlas_image: np.ndarray | None` — currently `None` (hook for future atlas bundling)
- `meta: dict` — includes `mode`, `seed`, `large_backend`

**Acceptance criteria**
- `sig_grid` and `sig_tc` represent the *same underlying signal* (no mismatch in indexing order).
- `rms_apml` matches `sig_apml.std(dim="time").transpose("AP","ML")`.
- Flattening/stacking convention is consistent with `ChannelGrid`:
  - `flat = ap * n_ml + ml` corresponds to `sig_apml.stack(channel=("AP","ML")).isel(channel=flat)`.

### `cogpy.datasets.gui_bundles.spectrogram_bursts_bundle(...) -> SpectrogramGuiBundle`

**Intent**
- Provide a 4D spectrogram tensor plus peak annotations for linked orthoslicer + peak overlay development.

**Signature (current)**
- `spectrogram_bursts_bundle(*, mode="small", seed=0, kind="toy")`

**Return fields (current)**
- `spec: xr.DataArray` — `GridSpectrogram4D` dims `("ml","ap","time","freq")`
- `bursts: pd.DataFrame` — `BurstPeaksTable`
- `meta: dict`

**Acceptance criteria**
- Every burst row lies within the coordinate extents of `spec`.
- Burst coordinates use the same units as `spec` coords.

## Relationship to existing generators

The current module `code/lib/cogpy/src/cogpy/datasets/tensor.py` already provides:

- 4D toy tensors: `make_dataset`, `make_flat_blob_dataset`
- peak table schema: `detect_bursts_hmaxima`
- a realistic-ish oscillatory grid simulator: `AROscillatorGrid.make(...)`
- iEEG-like multichannel signals: `example_smooth_multichannel_sigx`, `example_ieeg`

The intention is to *wrap* (or refactor) these into bundle constructors while keeping:
- deterministic defaults,
- explicit schema validation,
- mode-driven sizing.
