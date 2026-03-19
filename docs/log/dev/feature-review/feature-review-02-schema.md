# CogPy Data Schema Review

## Spectral outputs
dims:
- Low-level PSD functions in `src/cogpy/core/spectral/psd.py` are numpy-first: input `(..., time)` and return `(psd, freqs)` with `psd.shape == (..., freq)`.
- `psdx()` in `src/cogpy/core/spectral/specx.py` returns a single `xr.DataArray` with dims `[all input dims except time axis] + ["freq"]`.
- For canonical multichannel input `("channel", "time")`, output is `("channel", "freq")`.
- For canonical time-channel input `("time", "channel")`, output is `("channel", "freq")`.
- For canonical grid input `("time", "ML", "AP")`, output is `("ML", "AP", "freq")`.

coords:
- `freq` is attached as a strictly increasing 1D coord.
- All coords that do not depend on the reduced time axis are preserved.
- Grid PSD therefore preserves `ML` and `AP` coords if present.

attrs:
- `psdx()` copies input attrs, then adds `units="power/Hz"` and `fs`.
- Multitaper path also adds `method="multitaper"`, `bandwidth_hz`, and derived `NW`.
- Welch path adds `method="welch"`, `nperseg`, and `noverlap`.

return type:
- `psd_welch()` / `psd_multitaper()` / `psd_from_mtfft()`: `tuple[np.ndarray, np.ndarray]`.
- `psdx()`: `xr.DataArray`.

Notes:
- Actual grid PSD order is `("ML", "AP", "freq")`, not `DIMS_GRID_SPECTRUM = ("AP", "ML", "freq")`.
- Downstream spectral feature functions in `src/cogpy/core/spectral/features.py` still expect raw `(psd, freqs)` arrays, so `psdx()` composes by passing `psd_da.data` and `psd_da["freq"].values`.

## Spectrogram outputs
dims:
- `spectrogramx()` returns dims `[all input dims except time axis] + ["freq", "time"]`.
- For canonical grid input `("time", "ML", "AP")`, actual output is `("ML", "AP", "freq", "time")`.
- `compute_multitaper_spectrogram()` in `src/cogpy/core/spectral/spectrogram_burst.py` returns dataset variable `spec` with dims `("channel", "time", "freq")`.
- Toy tensor / orthoslicer helpers in `src/cogpy/datasets/tensor.py` use `("ml", "ap", "time", "freq")`.

coords:
- `spectrogramx()` attaches 1D `freq` and window-center `time` coords.
- Non-time coords are preserved from the input.
- `compute_multitaper_spectrogram()` attaches integer `channel`, float `time`, and float `freq`.

attrs:
- `spectrogramx()` copies input attrs, then adds `method="multitaper_spectrogram"`, `units="power/Hz"`, `fs`, `bandwidth_hz`, `nperseg`, `noverlap`.
- `compute_multitaper_spectrogram()` sets `spec.attrs["time_semantics"] = "window_center_seconds"` and stores most provenance on `ds.attrs`: `fs`, `engine`, `time_bandwidth`, `num_tapers`, `min_lambda`, `window_duration_s`, `step_duration_s`, `fmin_hz`, `fmax_hz`, `normalization`, `time_semantics`, `remove_mean`.

return type:
- `spectrogramx()`: `xr.DataArray`.
- `compute_multitaper_spectrogram()`: `xr.Dataset` with data var `spec`.

Notes:
- The actively used `spectrogramx()` schema does not match `DIMS_SPECTROGRAM4D = ("ml", "ap", "time", "freq")`.
- `coerce_spectrogram4d()` can normalize case/order for tensors that already use `ML/AP` or `ml/ap`, but `spectrogramx()` is not calling it.

## Channel feature maps
dims:
- `extract_channel_features_xr()` in `src/cogpy/core/preprocess/badchannel/channel_features.py` returns an `xr.Dataset`, one variable per feature.
- Static multichannel features reduce `time` and produce `("channel",)`.
- Static grid features reduce `time` and produce `("ML", "AP")`.
- Windowed multichannel features produce `("time_win", "channel")`.
- Windowed grid features produce `("time_win", "ML", "AP")`.

coords:
- Inputs are first coerced to canonical continuous schemas, so outputs inherit/infer `time`, `channel`, `ML`, `AP` coords.
- Windowed outputs get a `time_win` coord from `sliding_core.window_centers_time()` when a time coord exists, else sample indices.

attrs:
- Each output variable mostly inherits input attrs.
- Windowed variables also carry `window_size`, `window_step`, and `run_dim` from `running_reduce_xr()` / `running_blockwise_xr()`.
- Dataset attrs add `time_dim`, `features`, `schema_kind`.
- Windowed dataset attrs add `window_size`, `window_step`, `out_dim="time_win"`, `center_method`, and when `fs` exists also `window_size_s`, `window_step_s`, `fs_win`.

return type:
- `extract_channel_features_xr()`: `xr.Dataset`.
- `compute_features_sliding()` / `compute_features_sliding_legacy()`: raw `np.ndarray` plus `feature_names` and `centers`.

Notes:
- Actual static grid feature maps are `("ML", "AP")`, not `DIMS_GRID_FEATURE_MAP = ("AP", "ML")`.
- The newer design doc proposes per-variable `feature_name` attrs, but current implementation does not enforce or add them.

## Grid outputs (AP × ML)
dims:
- Continuous grid signal schema is `DIMS_IEEG_GRID = ("time", "ML", "AP")`.
- Windowed continuous grid schema is `DIMS_IEEG_GRID_WINDOW = ("time_win", "ML", "AP")`.
- Spatial measure functions in `src/cogpy/core/measures/spatial.py` are numpy-first and expect spatial axes ordered as `(..., AP, ML)` or `(AP, ML, time)`.
- `csd_power()` returns `(AP, ML, time)`.
- `spatial_coherence_profile()` returns coherence matrix `(distance_bin, freq)` plus separate distance/frequency vectors.
- `moran_i()` and `gradient_anisotropy()` reduce spatial dims to scalars.
- `marginal_energy_outlier()` returns row/column vectors and masks, not xarray.

coords:
- Spatial numpy functions do not attach coords.
- Xarray spatial filters in `src/cogpy/core/preprocess/filtering/spatial.py` preserve original dims/coords and add filter attrs.

attrs:
- Spatial filters preserve input attrs and add `filter_type` plus filter-specific parameters.
- Numpy spatial measures carry no attrs; provenance must be added by the caller.

return type:
- Mixed: `np.ndarray`, `float`, `dict[str, np.ndarray]`, or `tuple[np.ndarray, np.ndarray, np.ndarray]`.

Notes:
- There is a persistent mismatch between xarray grid tensors (`time, ML, AP`) and numpy spatial measure expectations (`..., AP, ML`).
- Several callers explicitly transpose to bridge this, for example `temporal_mean_laplacian` handling in `extract_channel_features_xr()`.

## Event outputs
table schema
- `validate_burst_peaks()` in `src/cogpy/datasets/schemas.py` defines a minimal burst-peak table with columns `burst_id, x, y, t, z, value`.
- `detect_bursts_hmaxima()` in `src/cogpy/datasets/tensor.py` emits exactly that orthoslicer-friendly table, mapping `x=ml`, `y=ap`, `t=time`, `z=freq`, `value=amp`.
- `cogpy.datasets.schemas.EventCatalog` is a strict interval-event contract.
- `cogpy.events.EventCatalog` is a looser detector/UI contract.

fields
- Strict `datasets.schemas.EventCatalog.table` required columns: `event_id`, `t`, `t0`, `t1`, `duration`, `label`, `score`.
- Strict `datasets.schemas.EventCatalog.meta` required keys: `detector`, `params`, `fs`, `n_events`, `cogpy_version`.
- Optional strict table columns include `channel`, `AP`, `ML`, `f0`, `f1`, `f_peak`, `n_channels`, `ch_min`, `ch_max`, `source`.
- Optional strict memberships table columns: `event_id`, `channel`.
- Looser `core.events.EventCatalog` only requires `event_id` and `t`; interval columns are optional and auto-derived where possible.
- `Events` dataclass schema: `times`, optional `labels`, `name`.
- `Intervals` dataclass schema: `starts`, `ends`, `name`.

## Dimension constants
Canonical constants currently defined in `src/cogpy/datasets/schemas.py`:

| Constant | Dims |
|---|---|
| `DIMS_EVENT_CATALOG` | `("event",)` |
| `DIMS_IEEG_GRID` | `("time", "ML", "AP")` |
| `DIMS_IEEG_GRID_WINDOW` | `("time_win", "ML", "AP")` |
| `DIMS_MULTICHANNEL` | `("channel", "time")` |
| `DIMS_MULTICHANNEL_WINDOW` | `("time_win", "channel")` |
| `DIMS_SPECTROGRAM4D` | `("ml", "ap", "time", "freq")` |
| `DIMS_IEEG_TIME_CHANNEL` | `("time", "channel")` |
| `DIMS_CHANNEL_FEATURE_MAP` | `("channel",)` |
| `DIMS_GRID_FEATURE_MAP` | `("AP", "ML")` |
| `DIMS_CHANNEL_SPECTRUM` | `("channel", "freq")` |
| `DIMS_GRID_SPECTRUM` | `("AP", "ML", "freq")` |
| `DIMS_CHANNEL_WINDOWED_SPECTRUM` | `("time_win", "channel", "freq")` |
| `DIMS_GRID_WINDOWED_SPECTRUM` | `("time_win", "AP", "ML", "freq")` |
| `DIMS_PAIRWISE_FEATURE_MATRIX` | `("channel_i", "channel_j")` |
| `DIMS_PAIRWISE_SPECTRUM` | `("channel_i", "channel_j", "freq")` |
| `DIMS_COMODULOGRAM` | `("channel", "freq_phase", "freq_amp")` |
| `DIMS_SPATIAL_COHERENCE_PROFILE` | `("distance_bin", "freq")` |
| `DIMS_EVENTS_TABLE` | `("time",)` |
| `DIMS_INTERVALS_TABLE` | `("t_start", "t_end")` |
| `DIMS_PERIEVENT` | `("event", "channel", "time")` |

Observed free-form / legacy dims outside the constant set:
- `ch` as a legacy alias for `channel`.
- Uppercase `AP/ML` and lowercase `ap/ml` both occur in real code.
- Legacy feature stacks use a `feature` dim.

## Validators
Existing validation helpers in `src/cogpy/datasets/schemas.py`:

- `validate_ieeg_grid()` / `coerce_ieeg_grid()`
  - Canonical dims `("time","ML","AP")`.
  - Require/infer 1D `time`, `ML`, `AP` coords.
  - Warn if `fs` missing.

- `validate_ieeg_grid_windowed()` / `coerce_ieeg_grid_windowed()`
  - Canonical dims `(win_dim,"ML","AP")`, default `win_dim="time_win"`.
  - Warn if `fs`, `window_size`, `window_step` missing.

- `validate_multichannel()` / `coerce_multichannel()`
  - Canonical dims `("channel","time")`.
  - Require increasing `time`, warn if `fs` missing.

- `validate_multichannel_windowed()` / `coerce_multichannel_windowed()`
  - Canonical dims `(win_dim,"channel")`.
  - Warn if `fs`, `window_size`, `window_step` missing.

- `validate_spectrogram4d()` / `coerce_spectrogram4d()`
  - Canonical dims `("ml","ap","time","freq")`.
  - Require increasing `time` and `freq`.
  - `coerce_spectrogram4d()` accepts uppercase `ML/AP` and renames them.

- `validate_ieeg_time_channel()` / `coerce_ieeg_time_channel()`
  - Canonical dims `("time","channel")`.
  - Accept flat `channel`, reset-index `AP/ML` per-channel coords, or MultiIndex channels.

- `validate_burst_peaks()`
  - Requires DataFrame columns `burst_id, x, y, t, z, value`.

- `validate_event_catalog()` / `coerce_event_catalog()`
  - Strict interval-event contract for the `datasets.schemas.EventCatalog` dataclass.

Gaps:
- No `validate_*` / `coerce_*` exist yet for `DIMS_CHANNEL_FEATURE_MAP`, `DIMS_GRID_FEATURE_MAP`, `DIMS_*SPECTRUM`, `DIMS_PAIRWISE_*`, `DIMS_COMODULOGRAM`, or `DIMS_SPATIAL_COHERENCE_PROFILE`.
- Those constants are currently design targets more than enforced runtime contracts.

## psdx compatibility
- `psdx()` is compatible with the PSD-first spectral feature stack because it emits a labeled `freq` axis while preserving non-time batch dims.
- For scalar/vector spectral features in `src/cogpy/core/spectral/features.py`, the practical composition pattern is:
  - compute `psd_da = psdx(...)`
  - pass `psd_da.data` and `psd_da["freq"].values` into numpy feature functions
  - re-wrap the output if labeled xarray output is needed
- `psdx()` is directly compatible with TensorScope PSD utilities (`src/cogpy/core/spectral/psd_utils.py`) because those utilities only require a `freq` dim and optionally stack `AP/ML` into `channel`.
- `psdx()` is not yet wired into schema validators for spectrum outputs; there is no `coerce_grid_spectrum()` or `validate_channel_spectrum()`.
- Grid `psdx()` output uses `("ML","AP","freq")`, so it does not directly satisfy the newer conceptual `("AP","ML","freq")` design.
- `psdx()` does not feed the current detector pipeline; burst detection uses `spectrogramx()`, not PSDs.

## Observed conventions
- Low-level compute is numpy-first; xarray wrappers are added at boundaries.
- `fs` is expected in `attrs`, not as a required coord.
- Xarray wrappers typically preserve all non-reduced coords and copy input attrs before appending method metadata.
- Windowed xarray outputs use `time_win` as the dimension name and store window provenance in attrs.
- Multi-feature outputs are usually `xr.Dataset` objects with one variable per feature, not a single `feature`-indexed `DataArray`.
- Continuous signal canonical schemas are comparatively mature and enforced.
- Feature-map / spectrum / pairwise schema constants exist, but most are not yet enforced at runtime.

## Potential friction points
- Spatial dim order is inconsistent across the codebase: continuous xarray signals use `("time","ML","AP")`, many spatial numpy functions expect `(...,"AP","ML")`, and new design constants prefer `("AP","ML",...)`.
- Spatial dim case is inconsistent: actively used tensors can be `ML/AP` or `ml/ap`.
- `spectrogramx()` emits `(...,"freq","time")`, while `DIMS_SPECTROGRAM4D` says `("ml","ap","time","freq")`.
- `psdx()` on grids emits `("ML","AP","freq")`, while `DIMS_GRID_SPECTRUM` says `("AP","ML","freq")`.
- Window-center semantics differ between modern xarray sliding helpers (`midpoint` time coordinates) and legacy bad-channel pipeline helpers (`upper` sample-index centers).
- There are two different `EventCatalog` implementations with different required fields and metadata contracts.
- Many schema constants for future feature outputs are present but currently unused by validators or by the active xarray transforms.
- Feature-specific attrs such as `feature_name`, `method`, or frequency-band descriptors are not applied consistently across returned variables.
- Return types are heterogeneous across adjacent APIs: `xr.DataArray`, `xr.Dataset`, `np.ndarray`, `tuple`, `dict`, `pd.DataFrame`, and two event-catalog classes all coexist.
