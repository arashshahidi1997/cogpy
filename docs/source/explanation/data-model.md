# Data Model

cogpy uses `xarray.DataArray` as its primary data structure. This page
explains the schema conventions and why they were chosen.

## Why xarray?

ECoG data has labeled dimensions (time, space, frequency) that are
meaningful to the analysis. Raw numpy arrays lose this context — you have
to track which axis is which separately. xarray solves this by attaching
named dimensions and coordinates to arrays.

cogpy chose xarray over custom classes because:
- It integrates with the scientific Python ecosystem (pandas, dask, zarr)
- It provides serialization for free (netCDF, Zarr)
- Dimension-aware operations (`sel`, `isel`, `groupby`) reduce indexing bugs
- It avoids the maintenance burden of a custom array class

## Signal schemas

All schemas are defined in `cogpy.base.ECoGSchema`:

### Grid ECoG: `(time, AP, ML)`

The primary schema for 2D electrode grids. Used by spatial measures,
CSD computation, and grid-aware preprocessing.

```
sig.dims  →  ("time", "AP", "ML")
sig.fs    →  1000.0  (Hz, scalar coordinate)
sig.time  →  [0.000, 0.001, 0.002, ...]  (seconds)
sig.AP    →  [0, 1, 2, ..., 15]  (grid row indices)
sig.ML    →  [0, 1, 2, ..., 15]  (grid column indices)
```

**AP** = anterior-posterior (row 0 = most posterior). **ML** = medial-lateral
(column 0 = most medial). This convention matches the physical electrode
layout.

### Flat ECoG: `(time, ch)`

For channel-indexed data without grid semantics (e.g., strip electrodes,
or after flattening a grid).

### Multichannel: `(channel, time)`

Generic multichannel format. Note the transposed axis order compared to
flat ECoG — this matches common neuroscience conventions (channels as rows).

## Sampling rate

`fs` can be stored in two ways:
- **Scalar coordinate:** `sig.coords["fs"]` — preferred
- **Attribute:** `sig.attrs["fs"]` — fallback

Use `cogpy.base.get_fs(sig)` to retrieve it regardless of storage
method, and `ensure_fs(sig, fs=1000.0)` to guarantee it is set.

## Spectrogram schemas

Two spectrogram schemas serve different roles:

### `GridWindowedSpectrum` — compute pipeline form

```
spec.dims  →  ("time_win", "AP", "ML", "freq")
```

Uppercase spatial dims match the `(..., AP, ML)` batch convention expected by
spatial measures. Use `coerce_grid_windowed_spectrum(da)` to convert
`spectrogramx()` output into this form (handles renames and transposes).

### `GridSpectrogram4D` — orthoslicer/GUI form

```
spec.dims  →  ("ml", "ap", "time", "freq")
```

Lowercase spatial dims, optimized for slice-based visualization.

### Flat form

```
spec.dims  →  ("ch", "time_win", "freq")
```

### Normalization

`normalize_spectrogram(spec, method=...)` produces a whitened or
dB-transformed spectrogram with the same dims:

- `"robust_zscore"` — `(x - median) / MAD` along `freq`
- `"db"` — `10 * log10(x)`

## Batch dimension convention

Compute functions in cogpy follow numpy's broadcasting convention with
**typed trailing axes**:

| Domain | Convention | Example |
|--------|-----------|---------|
| Temporal measures | `(..., time)` | `kurtosis(arr)` reduces last axis |
| Spectral features | `(..., freq)` | `band_power(psd, freqs, band)` |
| Spatial measures | `(..., AP, ML)` | `moran_i(grid)` reduces last two axes |

Leading `...` dimensions are batch dimensions (time windows, frequency bins,
channels, etc.). Functions broadcast over them automatically.

This design means you can apply a spatial measure to a full 4D spectrogram
`(time, freq, AP, ML)` in one call — no loops required.

## Event representations

Events are stored in `EventCatalog`, a thin pandas DataFrame wrapper with
standardized columns:

| Column | Required | Description |
|--------|----------|-------------|
| `event_id` | yes | Unique integer ID |
| `t` | yes | Event time (seconds) |
| `t0`, `t1` | no | Interval start/end |
| `duration` | no | `t1 - t0` |
| `freq` | no | Peak frequency (Hz) |
| `AP`, `ML` | no | Grid position |
| `channel` | no | Channel index |
| `label` | no | Event type label |
| `score` | no | Detection confidence |
| `detector` | no | Detector name |
| `pipeline` | no | Pipeline provenance |

## Validation boundaries

Core compute functions assume valid input — they do not coerce dimensions
or check schemas. Validation happens at **system boundaries**:

- `cogpy.io` — constructs valid DataArrays from raw files
- `cogpy.datasets.schemas` — `validate_*()` and `coerce_*()` functions
- `cogpy.cli` — argument parsing and input validation
- Frontend entry points — before passing data to the backend

This keeps core functions fast and simple.
