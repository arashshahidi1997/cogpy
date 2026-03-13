# CogPy TF-Space Schema Review

## Canonical signal dims observed

- The only enforced canonical grid signal schema is `DIMS_IEEG_GRID = ("time", "ML", "AP")` in `src/cogpy/datasets/schemas.py`.
- `validate_ieeg_grid()` and `coerce_ieeg_grid()` require or coerce exactly that uppercase grid order. They also require/infer 1D `time`, `ML`, and `AP` coords and recommend `fs` in attrs.
- The enforced windowed grid schema is `DIMS_IEEG_GRID_WINDOW = ("time_win", "ML", "AP")`.
- The enforced multichannel schema is `("channel", "time")`; a stacked viewer form `("time", "channel")` also exists via `validate_ieeg_time_channel()` / `coerce_ieeg_time_channel()`.
- In practice, older visualization / IO code still often uses `("time", "AP", "ML")` or explicitly transposes into `("AP", "ML", "time")` before NumPy compute. The validator layer does not normalize that globally.

## Spectral output dims observed

- `psdx()` in `src/cogpy/core/spectral/specx.py` returns `[all non-time dims] + ["freq"]`.
- For canonical grid input `("time", "ML", "AP")`, actual PSD output is `("ML", "AP", "freq")`.
- `spectrogramx()` returns `[all non-time dims] + ["freq", "time"]`.
- For canonical grid input `("time", "ML", "AP")`, actual spectrogram output is `("ML", "AP", "freq", "time")`.
- This does not match the only spectrogram validator target: `DIMS_SPECTROGRAM4D = ("ml", "ap", "time", "freq")`.
- `coerce_spectrogram4d()` can rename uppercase `ML/AP` to lowercase `ml/ap`, but it still expects the final order `("ml", "ap", "time", "freq")`, so `spectrogramx()` output needs both rename and transpose to pass.
- The pipeline-specific `compute_multitaper_spectrogram()` in `src/cogpy/core/spectral/spectrogram_burst.py` returns dataset var `spec(channel, time, freq)`, not a grid TF tensor.

## Grid output dims observed

- Spatial measure code in `src/cogpy/core/measures/spatial.py` is NumPy-first and assumes spatial axes are the last two axes in `(…, AP, ML)`.
- Scalar spatial measures such as `moran_i()`, `gradient_anisotropy()`, `spatial_kurtosis()`, and `spatial_noise_concentration()` reduce `(AP, ML)` to scalar and preserve all leading batch dims.
- `marginal_energy_outlier()` returns structured row/column summaries rather than xarray.
- `csd_power()` expects and returns `(AP, ML, time)`.
- Windowed grid feature extraction in `src/cogpy/core/preprocess/badchannel/channel_features.py` produces `("time_win", "ML", "AP")` and immediately coerces to that schema.

## Candidate TF-space tensor shapes

### signal: `(time, ML, AP)` or equivalent

- Feasible now: yes, for exactly `(time, ML, AP)` with uppercase spatial dims.
- Feasible with transpose/wrapping: yes, if a caller has `(time, AP, ML)` or another permutation of the same dim names.
- Blocked / unclear: lower-case `ml/ap` signal tensors are not part of the canonical signal validator path.

### spectrogram: `(time_win, freq, ML, AP)` or `(time_win, freq, AP, ML)`

- Feasible now: not as a canonical validated schema.
- Feasible with transpose/wrapping: yes. `spectrogramx()` already computes dense TF output, but actual order is `("ML", "AP", "freq", "time")`. To fit the target shape, the caller must transpose and usually rename the trailing `"time"` to `"time_win"` if window-center semantics matter.
- Blocked / unclear: there is no validator/coercer for `DIMS_GRID_WINDOWED_SPECTRUM = ("time_win", "AP", "ML", "freq")`, and no active function emits that shape directly.

### whitened spectrogram: same dims as spectrogram

- Feasible now: partially.
- Feasible with transpose/wrapping: yes, if whitening means a post-hoc normalization of an existing spectrogram, because `zscorex(dim="freq")`, `normalize_windowed_features()`, or custom arithmetic on `xr.DataArray` can preserve dims.
- Blocked / unclear: there is no canonical reusable `whitened_spectrogram` constructor or validator. `src/cogpy/core/spectral/whitening.py` whitens time series, not spectrogram tensors.

### derived spatial-summary tensor: `(time_win, freq)`

- Feasible now: yes, after an explicit transpose bridge into spatial order.
- Feasible with transpose/wrapping: yes. If a spectrogram is rearranged to `(time_win, freq, AP, ML)` or `(freq, time_win, AP, ML)`, the spatial measures can batch over `(time_win, freq)` and reduce the last two spatial dims.
- Blocked / unclear: no xarray wrapper exists that preserves coords and names for these reductions; current spatial measures return NumPy arrays or dict-like outputs.

### event/bout tables derived from 1D or 2D score reductions

- Feasible now: yes for 1D scores.
- Feasible with transpose/wrapping: yes for 2D reductions once the caller first collapses `(time_win, freq)` to a 1D score over time or extracts per-band 1D traces.
- Blocked / unclear: there is no built-in helper that converts a full 2D TF score map directly into a validated event catalog without a caller-defined reduction.

## Existing validation/coercion support

- Present and enforceable today:
  - `validate_ieeg_grid()` / `coerce_ieeg_grid()`
  - `validate_ieeg_grid_windowed()` / `coerce_ieeg_grid_windowed()`
  - `validate_multichannel()` / `coerce_multichannel()`
  - `validate_multichannel_windowed()` / `coerce_multichannel_windowed()`
  - `validate_ieeg_time_channel()` / `coerce_ieeg_time_channel()`
  - `validate_spectrogram4d()` / `coerce_spectrogram4d()`
  - `validate_event_catalog()` / `coerce_event_catalog()`
  - `validate_burst_peaks()`
- Important limitation: spectrum constants exist for channel/grid PSD and windowed spectrum outputs, but there are no `validate_*` / `coerce_*` functions for them.
- Important limitation: the spectrogram validator is orthoslicer-oriented and lower-case:
  - canonical dims are `("ml", "ap", "time", "freq")`
  - increasing `time` and `freq` are required
  - uppercase `ML/AP` are accepted only through `coerce_spectrogram4d()`
- This means the repo has a real validator for one 4D spectrogram form, but not for the actually emitted `spectrogramx()` form and not for the design constant `("time_win", "AP", "ML", "freq")`.

## Batch-dim compatibility of spatial measures

- `src/cogpy/core/measures/spatial.py` is batch-friendly by design:
  - `moran_i(grid)` expects `(…, AP, ML)` and returns `…`
  - `gradient_anisotropy(grid)` expects `(…, AP, ML)` and returns `…`
  - `spatial_kurtosis(grid)` expects `(…, AP, ML)` and returns `…`
  - `spatial_noise_concentration(grid)` expects `(…, AP, ML)` and returns `…`
- This is sufficient for TF-space reductions if the caller makes spatial axes last.
- `marginal_energy_outlier()` also supports batch-style use, but it returns row/column energies, z-scores, and masks rather than a single scalar summary tensor.
- The main caveat is xarray boundary friction: these functions take NumPy arrays, not labeled arrays, so coord preservation is the caller's responsibility.

## Event/bout schema compatibility

- Two event catalog layers exist.
- Strict schema layer:
  - `src/cogpy/datasets/schemas.py::EventCatalog`
  - required table columns: `event_id`, `t`, `t0`, `t1`, `duration`, `label`, `score`
  - required meta keys: `detector`, `params`, `fs`, `n_events`, `cogpy_version`
  - optional spatial/frequency columns already anticipated: `channel`, `AP`, `ML`, `f0`, `f1`, `f_peak`, `n_channels`, `ch_min`, `ch_max`, `source`
  - optional memberships table schema: `event_id`, `channel`
- Looser detector/UI layer:
  - `src/cogpy/core/events/catalog.py::EventCatalog`
  - only `event_id` and `t` are strictly required; interval columns are optional
- For 1D score traces, `src/cogpy/core/detect/utils.py::score_to_bouts()` already produces bout dicts with `t0`, `t`, `t1`, `value`, `duration`.
- To reach the strict schema from those bout dicts, the caller still needs to provide `meta` with `fs`, `detector`, and `params`, and likely map `value` to `score`.
- There is no canonical helper for turning a `(time_win, freq)` score map directly into a strict event catalog without first reducing across frequency.

## Friction points

- AP/ML order mismatches
  - Canonical continuous grid signal schema is `("time", "ML", "AP")`.
  - Spatial measures assume NumPy inputs with last axes `(AP, ML)`.
  - PSD from `psdx()` on grid signals is `("ML", "AP", "freq")`.
  - Windowed/grid spectrum design constant is `("time_win", "AP", "ML", "freq")`.
  - Several workflow and plotting callers explicitly transpose into `("AP", "ML", "time")` or `("AP", "ML")`.
- time vs time_win naming
  - Sliding-window schema validators use `"time_win"`.
  - `spectrogramx()` uses `"time"` for window centers.
  - `compute_multitaper_spectrogram()` also uses `"time"` for window centers and annotates `time_semantics = "window_center_seconds"`.
  - Practical consequence: TF outputs can be windowed in meaning but still not satisfy the windowed schema naming convention.
- numpy/xarray boundary issues
  - Spectral wrappers are xarray-first, but spatial measures are NumPy-first.
  - Batch compatibility is good numerically, but callers must manage transposes and xarray re-wrapping themselves.
  - `running_blockwise_xr()` warns implicitly through its design that feature-axis order follows input dims unless `feature_dims` is passed explicitly.
- functions that drop coords/attrs
  - Spatial measure functions return plain NumPy arrays, so all coords/attrs are lost at that boundary.
  - `marginal_energy_outlier()` similarly returns unlabeled summary structures.
  - `spectrogramx()` preserves non-time coords and copies attrs, but it replaces the original time coord with a new window-center `time` coord.
  - `running_blockwise_xr()` and `running_reduce_xr()` preserve attrs and inferred coords, but only for dims they can infer; more complex outputs rely on caller-supplied `feature_dims`.
  - Detector/event paths ultimately convert to pandas-backed catalogs, so xarray coords do not survive unless explicitly copied into columns.

Overall: CogPy can represent and process TF-space QC tensors, but not yet with one clean canonical schema across the whole path. The strongest current path is:

- signal in `("time", "ML", "AP")`
- spectrogram computed by `spectrogramx()` as `("ML", "AP", "freq", "time")`
- explicit transpose to `(time_win, freq, AP, ML)` or `(freq, time_win, AP, ML)`
- NumPy spatial reductions over trailing `(AP, ML)`
- optional 1D time-score reduction into `score_to_bouts()`
- explicit wrapping into the strict `datasets.schemas.EventCatalog` only at the output boundary

The main blockers are not missing compute primitives; they are schema mismatch, missing spectrum validators/coercers, and repeated manual transpose bridges at xarray/NumPy boundaries.
