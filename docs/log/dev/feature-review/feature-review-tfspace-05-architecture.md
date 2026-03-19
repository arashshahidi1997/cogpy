# CogPy TF-Space Architecture Assessment

This document assesses CogPy's readiness to support external pipelines that implement a time–frequency–space noise characterization workflow. It synthesizes the TF-space inventory (01), schema (02), composition (03), and DRY/vectorization (04) reviews into concrete architectural recommendations.

The target workflow operates on dense 4D tensors:

- `spec(time_win, freq, AP, ML)` — multitaper spectrogram over a grid
- `wspec(time_win, freq, AP, ML)` — whitened/normalized variant
- batched spatial measures reduced from each `(time_win, freq)` slice
- event/distribution summaries derived from those reduced score tensors

CogPy's role is to provide compute primitives and schema contracts, not to own the workflow orchestration.

---

## Readiness summary

| Capability | Status | Detail |
|-----------|--------|--------|
| Dense grid spectrogram | Ready | `spectrogramx()` computes `(ML, AP, freq, time)` from grid signal; rename + transpose needed to reach canonical form |
| Multitaper engine | Ready | `mtm_spectrogram` / `mtm_spectrogramx` provide the lower-level backend |
| Scalar spatial measures on `(..., AP, ML)` | Ready | `moran_i`, `gradient_anisotropy`, `spatial_kurtosis`, `spatial_noise_concentration` all accept leading batch dims |
| Structured spatial summaries | Ready | `marginal_energy_outlier` returns row/col profiles with batch support |
| Directional Moran's I | Ready | `moran_i(adjacency="ap_only")` and `"ml_only"` already implemented |
| 1D score → bout extraction | Ready | `score_to_bouts`, `dual_threshold_events_1d`, `merge_intervals` compose into event tables |
| Event catalog schema | Ready | `datasets.schemas.EventCatalog` with optional `f0`, `f1`, `f_peak` columns |
| Robust z-score along any dim | Ready | `zscorex(dim=..., robust=True)` preserves xarray labels |
| PSD spectral features | Ready | `narrowband_ratio`, `spectral_flatness`, `line_noise_ratio`, `band_power`, `broadband_snr` all vectorize over `(..., freq)` |
| Aperiodic/periodic decomposition | Ready | `aperiodic_exponent`, `fooof_periodic` via specparam on `(..., freq)` |
| Sliding window infrastructure | Ready | `running_blockwise_xr`, `running_reduce_xr` handle windowed transforms with coord propagation |
| Grid topology helpers | Ready | `grid_adjacency`, `make_footprint`, neighborhood median/MAD |
| Canonical TF schema coercion | **Implemented** | `coerce_grid_windowed_spectrum` / `validate_grid_windowed_spectrum` in `datasets/schemas.py` |
| Spectrogram normalization/whitening | **Implemented** | `normalize_spectrogram(method="robust_zscore"|"db")` in `spectral/specx.py` |
| xarray spatial-measure wrappers | **Implemented** | `spatial_summary_xr()` in `measures/spatial.py` — preserves coords, returns `xr.Dataset` |
| TF score → 1D reduction | **Implemented** | `reduce_tf_bands()` in `spectral/features.py` — per-band frequency collapse |
| Grid windowed spectrum validators | **Implemented** | `validate_grid_windowed_spectrum` / `coerce_grid_windowed_spectrum` in `datasets/schemas.py` |
| Occupancy/duration summaries | **Implemented** | `bout_occupancy()` and `bout_duration_summary()` in `detect/utils.py` |
| `ftest_line_scan` batch support | **Implemented** | Accepts `(..., time)` batched input, returns `(..., freq)` outputs |

---

## What already exists and is sufficient

These require no new code. They compose directly into a TF-space pipeline with at most standard transposition at boundaries.

### Spectrogram computation

`spectrogramx(sigx)` on a `("time", "ML", "AP")` grid signal produces `("ML", "AP", "freq", "time")` with correct frequency and time-center coordinates. The dense multitaper engine (`mtm_spectrogram` via Ghostipy) handles the heavy computation. Attrs and non-time coords are preserved.

For the pipeline: call `spectrogramx()` once per session, persist the result. The emitted dim order differs from the target `(time_win, freq, AP, ML)`, but this is a one-time transpose + rename, not a compute gap.

### Spatial measures with batch dims

All scalar spatial reducers already accept `(..., AP, ML)` and return `(...)`:

```python
# After transposing spec to (time_win, freq, AP, ML):
moran_vals = moran_i(spec_np)                    # (time_win, freq)
aniso_vals = gradient_anisotropy(spec_np)         # (time_win, freq)
kurt_vals  = spatial_kurtosis(spec_np)            # (time_win, freq)
conc_vals  = spatial_noise_concentration(spec_np) # (time_win, freq)
```

No Python loops. No new vectorization. The only requirement is that spatial axes are the trailing two dims.

Directional variants are also ready: `moran_i(spec_np, adjacency="ap_only")` and `"ml_only"` produce directional autocorrelation over batch dims.

### Event extraction from 1D scores

Once a `(time_win, freq)` score tensor is reduced to a 1D time trace (by collapsing across frequency), the existing `score_to_bouts(score, times, low=..., high=...)` produces bout dicts with `t0`, `t`, `t1`, `value`, `duration`. The strict `datasets.schemas.EventCatalog` can receive those bouts at the output boundary with optional `f0/f1/f_peak` columns.

### PSD-first spectral features

The spectral feature functions (`narrowband_ratio`, `spectral_flatness`, `band_power`, `line_noise_ratio`, `broadband_snr`, `aperiodic_exponent`, `fooof_periodic`) all operate on `(..., freq)` NumPy arrays. They broadcast correctly over `(time_win, AP, ML, freq)` or any other batch layout with `freq` last. No new wrappers needed for per-slice spectral characterization.

### Normalization primitives

`zscorex(da, dim="freq", robust=True)` already provides robust z-score normalization along any named xarray dimension. For TF-space work, calling `zscorex(spec_da, dim="freq", robust=True)` produces a frequency-normalized spectrogram while preserving all other dims, coords, and attrs.

Local spatial normalization via `local_robust_zscore_grid` and `neighborhood_median` / `neighborhood_mad` from `badchannel/spatial.py` can normalize feature maps against grid neighborhood baselines.

### Sliding window infrastructure

`running_blockwise_xr()` applies arbitrary functions to windowed data with coord propagation and `time_win` axis construction. This is the correct substrate for any time-resolved feature extraction that doesn't already exist as a dedicated function. It avoids reinventing window-center computation and label propagation.

---

## What needs thin wrappers/helpers — NOW IMPLEMENTED

All wrappers below have been implemented. This section documents the rationale and design decisions for each.

### 1. TF schema coercer: `coerce_grid_windowed_spectrum`

**Purpose**: Normalize `spectrogramx()` output to the canonical `DIMS_GRID_WINDOWED_SPECTRUM = ("time_win", "AP", "ML", "freq")` form in one call.

**What it does**: Accepts `("ML", "AP", "freq", "time")` or any permutation of those dims, renames `"time"` to `"time_win"` if spectrogram semantics apply, transposes to canonical order, validates that `time_win` and `freq` coords are monotonically increasing.

**Landing spot**: `src/cogpy/datasets/schemas.py`, alongside existing `coerce_spectrogram4d`.

**Justification**: The 01-inventory and 02-schema reviews document that every TF-space pipeline step begins with a manual rename + transpose from `spectrogramx()` output. The howto and tutorial docs both contain identical `spec.values.transpose(2, 3, 0, 1)` calls. A named coercer eliminates this and prevents axis-order bugs.

```python
def coerce_grid_windowed_spectrum(da: xr.DataArray) -> xr.DataArray:
    """Coerce to ("time_win", "AP", "ML", "freq")."""
```

### 2. Spectrogram normalization: `normalize_spectrogram`

**Purpose**: Provide a reusable `spec → wspec` boundary with explicit method selection.

**What it does**: Accepts a spectrogram `xr.DataArray` and a `method` argument. Supported methods:

| Method | Implementation |
|--------|---------------|
| `"robust_zscore"` | `zscorex(spec, dim="freq", robust=True)` |
| `"db"` | `10 * np.log10(spec)` preserving coords |
| `"aperiodic_subtract"` | Subtract `aperiodic_exponent`-fit background per `(time_win, AP, ML)` slice |
| `"flank_ratio"` | Per-bin `narrowband_ratio` recast as normalization |

Returns a spectrogram `xr.DataArray` with the same shape/dims plus `attrs["normalization"] = method`.

**Landing spot**: `src/cogpy/core/spectral/specx.py` or a new `src/cogpy/core/spectral/normalization.py`.

**Justification**: The 01-inventory identifies this as the single strongest missing piece. The 03-composition review confirms that `spec → wspec` is the biggest missing reusable composition boundary. The components exist, but no function assembles them into a labeled TF output. Without this, every pipeline script will re-implement its own normalization arithmetic and metadata tracking.

**Conservative scope**: Start with `"robust_zscore"` and `"db"` only. These compose from existing primitives with zero new algorithms. Add `"aperiodic_subtract"` later if pipeline demand justifies it.

```python
def normalize_spectrogram(
    spec: xr.DataArray,
    *,
    method: str = "robust_zscore",
    dim: str = "freq",
) -> xr.DataArray:
    """Normalize a spectrogram along a dimension."""
```

### 3. xarray wrappers for spatial reductions: `spatial_summary_xr`

**Purpose**: Apply spatial reducers to an xarray tensor and return labeled `xr.DataArray` or `xr.Dataset` with coords preserved.

**What it does**: Accepts a spectrogram or feature tensor with named `AP` and `ML` dims. Applies one or more spatial reducers (`moran_i`, `gradient_anisotropy`, `spatial_kurtosis`, `spatial_noise_concentration`) and returns the results as an `xr.Dataset` with all non-spatial dims (e.g., `time_win`, `freq`) preserved as coords.

**Landing spot**: `src/cogpy/core/measures/spatial.py`, alongside existing NumPy functions.

**Justification**: The 04-DRY review documents that tutorials and howtos teach `spec.values.transpose(...)` before spatial reduction, then discard all labels. The NumPy kernels are already batch-ready; only the label-preserving `xr.apply_ufunc` layer is missing. This is high-value, low-risk.

```python
def spatial_summary_xr(
    da: xr.DataArray,
    *,
    measures: Sequence[str] = ("moran_i", "gradient_anisotropy", "spatial_kurtosis"),
    ap_dim: str = "AP",
    ml_dim: str = "ML",
) -> xr.Dataset:
    """Compute scalar spatial summaries preserving non-spatial coords."""
```

### 4. TF score band reduction: `reduce_tf_bands`

**Purpose**: Collapse a `(time_win, freq)` score tensor to per-band or broadband time traces.

**What it does**: Accepts a `(time_win, freq)` xarray DataArray and a band specification (dict of `name → (fmin, fmax)` or `"broadband"`). Returns an `xr.Dataset` with one variable per band, each of shape `(time_win,)`.

**Landing spot**: `src/cogpy/core/spectral/features.py` or a new `src/cogpy/core/spectral/reduction.py`.

**Justification**: The 03-composition review identifies that step 4 → 5 (spatial summaries → event extraction) currently requires caller-defined frequency reductions. Without a shared helper, each pipeline script will choose its own collapsing strategy and lose metadata about which bands were used.

```python
def reduce_tf_bands(
    score: xr.DataArray,
    bands: dict[str, tuple[float, float]],
    *,
    freq_dim: str = "freq",
    method: str = "mean",
) -> xr.Dataset:
    """Reduce (time_win, freq) to per-band time traces."""
```

---

## What requires real new primitives — NOW IMPLEMENTED

All primitives below have been implemented. This section documents the rationale and design decisions for each.

### 1. `ftest_line_scan` batch support

**Current state**: Accepts only `(time,)` 1D signal. Returns `(fstat, freqs, sig_mask)` where each is 1D.

**Required**: Accept `(..., time)` batched input, return `(..., freq)` outputs. This is needed for producing spatially and temporally resolved line-noise maps: apply the F-test over `(AP, ML, time_win)` windows to get `(AP, ML, time_win, freq)` significance maps.

**Implementation path**: The DPSS tapers and eigenvalues depend only on `(N, NW, K)`, which are shared across batch elements. Compute tapered FFTs as a single batched matrix multiply `(..., K, freq) = (..., time) @ dpss.T`, then reduce to F-statistics per element. This is a medium-effort change to the existing function, not a new module.

**Landing spot**: `src/cogpy/core/spectral/features.py` — modify existing `ftest_line_scan`.

### 2. Grid windowed spectrum validators

**Current state**: `DIMS_GRID_WINDOWED_SPECTRUM = ("time_win", "AP", "ML", "freq")` exists as a constant. No `validate_grid_windowed_spectrum` or `coerce_grid_windowed_spectrum` is implemented.

**Required**: A validate/coerce pair following the existing pattern in `schemas.py`. This enforces the canonical dim order and validates coord monotonicity at pipeline boundaries.

**Implementation path**: Follow `validate_ieeg_grid` / `coerce_ieeg_grid` pattern exactly. Low effort.

**Landing spot**: `src/cogpy/datasets/schemas.py`.

### 3. Occupancy and duration summary helpers

**Current state**: `Intervals.total_duration()` and `score_to_bouts()` exist. No named helper computes occupancy fraction (time above threshold / total time) or duration distribution summaries (mean, median, percentiles of bout durations).

**Required**: Two small functions:

```python
def bout_occupancy(bouts: list[dict], total_duration: float) -> float:
    """Fraction of total_duration occupied by bouts."""

def bout_duration_summary(bouts: list[dict]) -> dict:
    """Summary statistics of bout durations: count, mean, median, std, p5, p95."""
```

**Implementation path**: Trivial arithmetic on the `"duration"` field of bout dicts. These become reusable across QC, detection, and noise-characterization pipelines.

**Landing spot**: `src/cogpy/core/detect/utils.py`, near `score_to_bouts`.

---

## What should remain pipeline-side logic

These belong in external Snakemake scripts, not in CogPy core. They are workflow decisions, not reusable primitives.

### Spectrogram persistence and caching

The decision of when to compute, persist (Zarr), and reuse a spectrogram tensor is orchestration logic. CogPy provides `spectrogramx()` and xarray serialization; the pipeline decides the DAG.

### Threshold and merge-gap selection

Which thresholds to use for bout extraction, what merge gap to apply, and what minimum duration to enforce are analysis decisions. CogPy provides `score_to_bouts()` with configurable parameters; the pipeline supplies the values.

### Cross-session aggregation

Aggregating noise profiles across sessions (ranking, clustering, reporting) is workflow-level logic that depends on the experiment structure. CogPy does not need a cross-session aggregation framework.

### Band definitions

The choice of frequency bands for reduction (e.g., delta/theta/alpha/beta/gamma) is domain- and experiment-specific. CogPy provides `band_power()` and `reduce_tf_bands()` with band arguments; the pipeline defines the bands.

### TTL epoch selection

Selecting recording epochs based on TTL markers or behavioral metadata is pipeline logic. CogPy provides `sigx.sel(time=slice(t0, t1))` via xarray; the pipeline knows the epoch boundaries.

### Plot/report generation

HTML, PNG, and other inspection outputs are leaf products of the pipeline. CogPy may provide plotting utilities but the pipeline decides what to plot, when, and where to save.

### Detection pipeline composition

The order in which spatial measures, spectral features, and event extraction are chained — and whether to use `spec` or `wspec` as input — is an analysis decision. CogPy provides composable primitives; the pipeline composes them.

---

## Proposed APIs / module landing spots

### Schema layer (`src/cogpy/datasets/schemas.py`)

```python
# New coercer alongside existing DIMS_GRID_WINDOWED_SPECTRUM constant
def coerce_grid_windowed_spectrum(da: xr.DataArray) -> xr.DataArray:
    """Coerce to ("time_win", "AP", "ML", "freq").

    Accepts spectrogramx() output ("ML", "AP", "freq", "time") and
    any other permutation. Renames "time" → "time_win" if present.
    Validates freq and time_win are monotonically increasing.
    """

def validate_grid_windowed_spectrum(da: xr.DataArray) -> xr.DataArray:
    """Validate dims match ("time_win", "AP", "ML", "freq") exactly."""
```

### Spectral transforms (`src/cogpy/core/spectral/specx.py`)

```python
def normalize_spectrogram(
    spec: xr.DataArray,
    *,
    method: str = "robust_zscore",
    dim: str = "freq",
) -> xr.DataArray:
    """Normalize a spectrogram along a dimension.

    Parameters
    ----------
    method : {"robust_zscore", "db"}
        "robust_zscore" — (x - median) / (MAD * 1.4826) along dim
        "db" — 10 * log10(x), no dim reduction
    dim : str
        Dimension to normalize along. Typically "freq".

    Returns
    -------
    xr.DataArray with same shape/dims; attrs updated with normalization metadata.
    """
```

### Spectral features (`src/cogpy/core/spectral/features.py`)

```python
# Modify existing function signature:
def ftest_line_scan(
    signal,  # (..., time) ndarray — was (time,) only
    fs: float,
    *,
    NW: float = 4.0,
    p_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thomson F-test line scan.

    Returns (fstat, freqs, sig_mask) where fstat and sig_mask
    are (..., freq) — matching leading batch dims of input.
    """
```

### Spatial measures (`src/cogpy/core/measures/spatial.py`)

```python
def spatial_summary_xr(
    da: xr.DataArray,
    *,
    measures: Sequence[str] = ("moran_i", "gradient_anisotropy", "spatial_kurtosis"),
    ap_dim: str = "AP",
    ml_dim: str = "ML",
    moran_adjacency: str = "queen",
) -> xr.Dataset:
    """Compute scalar spatial summaries preserving non-spatial coords.

    Applies each named measure to the (AP, ML) spatial dims of da,
    returning an xr.Dataset with one variable per measure. All non-spatial
    dims and coords are preserved.

    Parameters
    ----------
    da : xr.DataArray
        Input with named AP and ML dims (any position).
    measures : sequence of str
        Measure names. Supported: "moran_i", "gradient_anisotropy",
        "spatial_kurtosis", "spatial_noise_concentration",
        "moran_ap", "moran_ml".
    """
```

### Spectral reduction (`src/cogpy/core/spectral/features.py`)

```python
def reduce_tf_bands(
    score: xr.DataArray,
    bands: dict[str, tuple[float, float]],
    *,
    freq_dim: str = "freq",
    method: str = "mean",
) -> xr.Dataset:
    """Reduce (..., freq) to per-band scalars.

    Parameters
    ----------
    score : xr.DataArray
        Input with a freq dim.
    bands : dict
        Mapping of band_name → (fmin, fmax).
    method : {"mean", "median", "max", "sum"}
        Reduction method within each band.

    Returns
    -------
    xr.Dataset with one variable per band, freq dim removed.
    """
```

### Event helpers (`src/cogpy/core/detect/utils.py`)

```python
def bout_occupancy(bouts: list[dict], total_duration: float) -> float:
    """Fraction of total_duration occupied by bouts."""

def bout_duration_summary(bouts: list[dict]) -> dict:
    """Summary statistics of bout durations.

    Returns dict with keys: count, mean, median, std, p5, p95.
    """
```

---

## Schema/coercion recommendations

### Adopt `(time_win, AP, ML, freq)` as canonical windowed-spectrum order

This aligns with:
- `DIMS_GRID_WINDOWED_SPECTRUM` already defined as `("time_win", "AP", "ML", "freq")`
- Spatial measures expecting trailing `(AP, ML)`
- Time/frequency batch dims leading

The coercer should accept:
- `spectrogramx()` output `("ML", "AP", "freq", "time")` → rename `time` → `time_win`, transpose
- Lower-case `("ml", "ap", ...)` → uppercase
- Any permutation of the four dims

### Do not change `spectrogramx()` emission order

Changing the output order of `spectrogramx()` would break existing callers. The coercer absorbs the mismatch at composition boundaries.

### Rename `time` → `time_win` only in the coercer

`spectrogramx()` should continue emitting `"time"` for window centers. The rename to `"time_win"` happens in `coerce_grid_windowed_spectrum()` when the caller explicitly requests windowed-spectrum semantics. This avoids breaking downstream code that expects `"time"` as a spectrogram dim name.

### Validate coord monotonicity

`coerce_grid_windowed_spectrum()` should verify:
- `time_win` is strictly increasing
- `freq` is strictly increasing
- `AP` and `ML` are monotonic (not necessarily increasing, depending on grid orientation)

### Do not unify AP/ML case globally

The uppercase `AP/ML` vs lowercase `ml/ap` split is established: uppercase for compute-oriented schemas, lowercase for orthoslicer/visualization schemas. The TF-space workflow is compute-oriented, so it uses uppercase. The coercer accepts either and normalizes to uppercase.

---

## Implementation order — ALL PHASES COMPLETE

All six phases have been implemented and tested (125 tests passing).

| Phase | Deliverable | Location | Status |
|-------|------------|----------|--------|
| 0 | `coerce_grid_windowed_spectrum` | `src/cogpy/datasets/schemas.py` | Done |
| 1 | `normalize_spectrogram` | `src/cogpy/core/spectral/specx.py` | Done |
| 2 | `spatial_summary_xr` | `src/cogpy/core/measures/spatial.py` | Done |
| 3 | `reduce_tf_bands` | `src/cogpy/core/spectral/features.py` | Done |
| 4 | `bout_occupancy`, `bout_duration_summary` | `src/cogpy/core/detect/utils.py` | Done |
| 5 | `ftest_line_scan` batch | `src/cogpy/core/spectral/features.py` | Done |

### End-to-end composition example

With all phases implemented, the full TF-space workflow composes in ~20 lines of pipeline script:

```python
from cogpy.spectral.specx import spectrogramx, normalize_spectrogram
from cogpy.spectral.features import reduce_tf_bands, ftest_line_scan
from cogpy.measures.spatial import spatial_summary_xr
from cogpy.detect.utils import score_to_bouts, bout_occupancy, bout_duration_summary
from cogpy.datasets.schemas import coerce_grid_windowed_spectrum

# 1. Compute and coerce spectrogram
spec = spectrogramx(sigx, bandwidth=4.0, nperseg=512)
spec = coerce_grid_windowed_spectrum(spec)     # → (time_win, AP, ML, freq)

# 2. Normalize to whitened spectrogram
wspec = normalize_spectrogram(spec, method="robust_zscore", dim="freq")

# 3. Compute spatial summaries over each (time_win, freq) slice
ds = spatial_summary_xr(wspec, measures=("moran_i", "gradient_anisotropy", "spatial_kurtosis"))

# 4. Collapse to per-band time traces
bands = {"gamma": (30, 80), "high_gamma": (80, 200)}
band_scores = reduce_tf_bands(ds["moran_i"], bands)

# 5. Extract events and summaries
bouts = score_to_bouts(band_scores["gamma"].values, band_scores.time_win.values, low=2.0, high=3.0)
occ = bout_occupancy(bouts, total_duration=float(spec.time_win[-1] - spec.time_win[0]))
summary = bout_duration_summary(bouts)

# 6. Batched F-test for line-noise maps (optional)
fstat, freqs, sig_mask = ftest_line_scan(sigx_windows, fs)  # (..., time) → (..., freq)
```

---

## Risks / open questions

### Memory pressure from dense 4D tensors

A `spec(time_win, freq, AP, ML)` tensor for a typical ECoG grid (8×16 electrodes, 1000 time windows, 256 frequency bins) is `~32 GB` at float64 or `~16 GB` at float32. Pipelines will need to either:
- Work at reduced resolution (fewer windows, coarser frequency)
- Use Dask-backed xarray for out-of-core computation
- Persist to Zarr and process in chunks

CogPy's spatial measures are NumPy-first and assume the full `(AP, ML)` grid fits in memory per slice. This is fine for ECoG grids (small spatial dims) but the batch over `(time_win, freq)` could be large. `xr.apply_ufunc` with `dask="parallelized"` would handle this, but only if the spatial wrappers are written with that in mind.

**Recommendation**: Design `spatial_summary_xr` to work with both eager NumPy and lazy Dask arrays from the start. The per-slice computation is fast; the issue is materializing all slices simultaneously.

### `marginal_energy_outlier` returns dicts, not scalars

Unlike the other spatial measures, `marginal_energy_outlier` returns a structured dict of row/column profiles, z-scores, and masks. It cannot be wrapped into a single `xr.apply_ufunc` call that returns a scalar per slice.

**Options**:
1. Wrap it separately, returning an `xr.Dataset` with vector-valued variables (e.g., `row_zscore(time_win, freq, AP)`).
2. Derive a scalar summary from it (e.g., max row z-score) and include that in `spatial_summary_xr`.
3. Leave it outside `spatial_summary_xr` and document the manual composition pattern.

**Recommendation**: Option 2 for `spatial_summary_xr` (derive `"max_row_zscore"` and `"max_col_zscore"` scalars). Offer option 1 as a separate helper if full profiles are needed.

### `narrowband_ratio` loop over frequency bins

The 04-DRY review flags that `narrowband_ratio` iterates `for i in range(nf)` with Python-level flank mask recomputation. For dense TF-space use where it is applied to many `(time_win, AP, ML)` slices, this frequency loop becomes a bottleneck if the slices are large.

**Recommendation**: Defer optimization unless profiling shows it is actually the bottleneck. The spatial dims are small (128 electrodes), so the per-bin cost is dominated by the outer batch loop, not the frequency loop itself.

### Consolidation of robust z-score implementations

Three independent implementations of robust z-score exist (`zscorex`, `normalize_windowed_features`, inline `_zscore` in `marginal_energy_outlier`). The TF-space wrappers should delegate to `zscorex` as the single canonical implementation.

**Recommendation**: Do not consolidate the legacy implementations now — that is a separate refactoring task. New TF-space code should use `zscorex` exclusively.

### `DIMS_SPECTROGRAM4D` vs `DIMS_GRID_WINDOWED_SPECTRUM`

Two 4D spectrogram schemas exist:
- `DIMS_SPECTROGRAM4D = ("ml", "ap", "time", "freq")` — orthoslicer/visualization oriented, lowercase
- `DIMS_GRID_WINDOWED_SPECTRUM = ("time_win", "AP", "ML", "freq")` — compute oriented, uppercase

These serve different consumers. Do not unify them. The TF-space workflow uses `DIMS_GRID_WINDOWED_SPECTRUM`. TensorScope and orthoslicer views use `DIMS_SPECTROGRAM4D`. The existing `coerce_spectrogram4d` handles the visualization form; the new `coerce_grid_windowed_spectrum` handles the compute form.

### Scope creep into workflow orchestration

The strongest temptation will be to build a `TFSpacePipeline` class or a `run_tf_qc(sigx, config)` function that chains all the steps. This should be resisted. The evidence from the 03-composition review and the existing Snakemake pipeline practice is clear: orchestration belongs in external scripts, not in CogPy.

CogPy's additions should be:
- Pure functions with standard types
- No global state or configuration objects
- No execution ordering
- No file I/O embedded in compute functions

If a user can compose the full TF-space workflow in 20–30 lines of pipeline script using CogPy primitives, the architecture is doing its job.
