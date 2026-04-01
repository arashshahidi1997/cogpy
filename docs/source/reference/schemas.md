# Entity Collection: Expected Schemas

This document proposes a small collection of **named entities** with explicit schemas (dims/coords/attrs).
These entities are *not* file formats; they are in-memory objects used by `cogpy.plot.*` and future GUIs.

## Design goals

- **Predictable**: explicit dims + coordinate names (avoid silent transposes).
- **Composable**: entities can be bundled into GUI dev fixtures.
- **Deterministic**: generators accept a `seed` and are stable over time.
- **Small + large modes**: same schema, different sizes.
- **Self-documenting failures**: runtime validators raise clear `ValueError`s with hints (see `cogpy.datasets.schemas`).
- **Coercion at boundaries**: a `coerce_*` layer fixes “almost-right” inputs (dim permutations, missing `fs`) before validating.

## Non-goals (for this layer)

- No disk IO, no BIDS/BIDS-iEEG conventions here.
- No attempt to represent every dataset type; start with what GUIs need.

---

## Entity: `IEEGGridTimeSeries`

Grid-aware time series for a 2D electrode array.

**Type**
- `xarray.DataArray`

**Dims (canonical)**
- `("time", "ML", "AP")`

**Coords**
- `time`: seconds, 1D, strictly increasing
- `ML`: physical coordinate (e.g., mm), 1D, length `n_ml`
- `AP`: physical coordinate (e.g., mm), 1D, length `n_ap`

**Name**
- `name`: recommended `"ieeg"` or `"val"` (avoid long names in plot titles)

**Attrs (recommended)**
- `fs` (float): sampling rate in Hz (optional but strongly recommended)
- `units` (str): e.g. `"uV"` (optional)

**Notes**
- This matches the implemented example generator `cogpy.datasets.entities.example_ieeg_grid(...)`.
- When you need consistent **row-major flattening** aligned to `ChannelGrid` (AP rows, ML cols), use a derived view:
  - `sig_apml = sig.transpose("time","AP","ML")`
  - flattening convention: `flat = ap * n_ml + ml`
  - stacking: `sig_apml.stack(channel=("AP","ML"))`
- NeuroScope `.lfp` files are MATLAB-native and typically stored **column-major**; when interpreting a file-ordered `(time, channel)` matrix as a grid, the natural flattening is:
  - `flat = ml * n_ap + ap` (col-major over the grid plane)

---

## Entity: `MultichannelTimeSeries`

Channel-unaware multichannel time series (no grid semantics).

**Type**
- `xarray.DataArray`

**Dims**
- `("channel", "time")` (canonical)

**Coords**
- `channel`: strings or ints, 1D
- `time`: seconds, 1D, strictly increasing

**Attrs (recommended)**
- `fs` (float), `units` (str)

---

## Entity: `IEEGTimeChannel` (stacked view for viewers)

Time × channel representation derived from `IEEGGridTimeSeries` for stacked-trace viewers.

**Type**
- `xarray.DataArray`

**Dims (canonical for `cogpy.datasets.gui_bundles.ieeg_grid_bundle`)**
- `("time", "channel")`

**Coords**
- `time`: seconds, 1D
- **Canonical (recommended)**: `channel` is a flat integer coordinate and `AP` and `ML` are 1D coords aligned to `channel` (created via `stack(...).reset_index("channel")`).
- Allowed fallback: `channel` is a MultiIndex over `(AP, ML)` when created via `stack(channel=("AP","ML"))`.

**Runtime helpers**
- `validate_ieeg_time_channel(da)` validates either canonical form.
- `coerce_ieeg_time_channel(da)` converts MultiIndex to the canonical reset-index form.

---

## Entity: `GridSpectrogram4D`

Time–frequency representation for a 2D grid (orthoslicer-friendly).

**Type**
- `xarray.DataArray`

**Dims (canonical; matches existing toy tensors)**
- `("ml", "ap", "time", "freq")`

**Coords**
- `ml`, `ap`: physical coords (or normalized 0..1), 1D
- `time`: seconds, 1D
- `freq`: Hz, 1D, strictly increasing

**Name**
- `"val"` (recommended)

**Notes**
- This matches `make_dataset(...)` and `AROscillatorGrid.spectrogram` in `code/lib/cogpy/src/cogpy/datasets/tensor.py`.
- Mapping convention for orthoslicers / GUI overlays:
  - `ml -> x`, `ap -> y`, `time -> t`, `freq -> z`

---

## Entity: `GridWindowedSpectrum`

Time–frequency representation for a 2D grid (compute-oriented, uppercase spatial dims).

**Type**
- `xarray.DataArray`

**Dims (canonical)**
- `("time_win", "AP", "ML", "freq")`

**Coords**
- `time_win`: window-center seconds, 1D, strictly increasing
- `AP`: physical coordinate (e.g., mm), 1D
- `ML`: physical coordinate (e.g., mm), 1D
- `freq`: Hz, 1D, strictly increasing

**Relationship to `GridSpectrogram4D`**
- `GridSpectrogram4D` uses lowercase `("ml", "ap", "time", "freq")` for orthoslicer/GUI use.
- `GridWindowedSpectrum` uses uppercase `("time_win", "AP", "ML", "freq")` for compute pipelines, matching the spatial measure batch convention `(..., AP, ML)`.

**Runtime helpers**
- `validate_grid_windowed_spectrum(da)` validates dims, coord monotonicity.
- `coerce_grid_windowed_spectrum(da)` accepts `spectrogramx()` output `("ML", "AP", "freq", "time")`,
  lowercase `ml/ap` variants, or any permutation, and transposes/renames to canonical form.

---

## Entity: `BurstPeaksTable`

Tabular peak/burst annotations aligned to `GridSpectrogram4D` (or similar).

**Type**
- `pandas.DataFrame`

**Required columns**
- `burst_id` (int)
- `x` (float): ML coordinate (same units as `ml`)
- `y` (float): AP coordinate (same units as `ap`)
- `t` (float): time in seconds (same units as `time`)
- `z` (float): frequency in Hz (same units as `freq`)
- `value` (float): amplitude/score

**Optional columns**
- index-space helpers (useful for debugging): `i_ml`, `i_ap`, `i_time`, `i_freq`

**Notes**
- This matches `detect_bursts_hmaxima(...)` output schema in `code/lib/cogpy/src/cogpy/datasets/tensor.py`.

---

## Entity: `AtlasImageOverlay` (optional)

An anatomical background image intended to sit behind `ChannelGridWidget`.

**Type**
- `np.ndarray` of shape `(H, W, 3)` or `(H, W, 4)` `uint8`

**Associated metadata (bundle-level)**
- physical extent / scaling parameters required to place the image in AP/ML space.
