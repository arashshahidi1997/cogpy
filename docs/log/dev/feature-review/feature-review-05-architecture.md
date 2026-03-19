# CogPy Feature Architecture Assessment

## Context

This document assesses CogPy's readiness to support a cross-session ECoG noise-characterization demo. It synthesizes the inventory (01), schema (02), composition (03), and DRY/vectorization (04) reviews into concrete architectural recommendations.

The target demo composes spectral, spatial, joint spatial-spectral, and normalization measures into a reproducible noise-characterization workflow orchestrated externally (Snakemake). CogPy's role is to provide the compute primitives, not to own the workflow.

---

## Inventory evaluation

### Exists and ready

These measures have direct implementations in CogPy with compatible signatures. They can be composed into the demo with minimal or no new code.

| Measure | Module | Function | Input | Output |
|---------|--------|----------|-------|--------|
| narrowband ratio | `spectral/features.py` | `narrowband_ratio` | `(psd, freqs, *, flank_hz=5.0)` | `(..., freq)` |
| spectral flatness | `spectral/features.py` | `spectral_flatness` | `(psd, freqs)` | `(...,)` scalar |
| F-test line scan | `spectral/features.py` | `ftest_line_scan` | `(signal, fs, *, NW, p_threshold)` | `(fstat, freqs, sig_mask)` |
| Moran's I (incl. directional) | `measures/spatial.py` | `moran_i` | `(grid, *, adjacency)` | scalar or `(...,)` |
| gradient anisotropy | `measures/spatial.py` | `gradient_anisotropy` | `(grid)` | scalar or `(...,)` |
| row/column energy profiles | `measures/spatial.py` | `marginal_energy_outlier` | `(grid, *, robust, threshold)` | dict of vectors + masks |
| channel z-score | `filtering/normalization.py` | `zscorex` | `(sigx, *, dim, robust)` | `xr.DataArray` |
| robust z-score | `badchannel/spatial.py` | `normalize_robust_z` | `(x, neigh_med, neigh_mad)` | same shape |
| grid neighborhood normalization | `badchannel/spatial.py` | `local_robust_zscore_grid` | `(input_arr, *, footprint)` | `(AP, ML)` |
| neighborhood median/MAD | `badchannel/spatial.py` | `neighborhood_median`, `neighborhood_mad` | `(values, *, neighbors)` | same shape |
| PSD computation | `spectral/specx.py` | `psdx` | `(sigx, *, method, bandwidth, ...)` | `xr.DataArray(..., freq)` |
| multitaper F-stat (known f0) | `spectral/bivariate.py` | `mtm_fstat` | `(arr, fs, *, NW, K, f0)` | `(...,)` scalar |

### Exists but needs adaptation

| Measure | Gap | Adaptation needed |
|---------|-----|-------------------|
| narrowband-to-broadband ratio | `line_noise_ratio` targets a single known frequency | Generalize to accept a detected-peak frequency vector, or compose `narrowband_ratio` with a peak picker |
| spectral peak detection | `fooof_periodic` returns the periodic component, not discrete peak locations | Add a thin peak-picker that operates on `fooof_periodic` or `narrowband_ratio` output |
| temporal stability | No direct function; `compute_features_sliding` provides windowed features | Compute coefficient of variation across windows for any scalar feature; trivial with existing sliding helpers |
| per-frequency normalization | `zscorex` operates along one dim; no freq-axis-specific variant | Apply `zscorex(psd_da, dim="freq")` or a manual z-score along freq; no new function needed |

### Missing

| Measure | Description | Priority |
|---------|-------------|----------|
| sliding F-test line scanner | `ftest_line_scan` operates on a single window; needs a sliding-window wrapper to produce `(time_win, freq)` maps | High — core demo deliverable |
| spatial frequency ratio | Ratio of low to high spatial frequency energy on the grid | Medium — new spatial spectral primitive |
| spatial kurtosis | Kurtosis of the spatial amplitude distribution across AP x ML | Low — trivial given `scipy.stats.kurtosis` on flattened grid |
| spatially resolved line noise maps | `narrowband_ratio` or `ftest_line_scan` applied per grid electrode, producing `(AP, ML, freq)` | Medium — composition pattern, not a new primitive |
| spatial noise concentration | Fraction of total grid energy in the top-k electrodes | Low — one-liner on sorted energy |
| directional Moran's I | Already exists via `moran_i(adjacency="ap_only")` and `moran_i(adjacency="ml_only")` | None — already implemented |

---

## Schema compatibility

### Current schema strengths

The existing schema infrastructure supports the demo's primary data flow:

1. **Grid signal input**: `coerce_ieeg_grid()` enforces `("time", "ML", "AP")` with `fs` in attrs. All pipeline scripts already use this.
2. **PSD output**: `psdx()` returns `xr.DataArray` with `freq` coordinate and preserves non-time dims. Grid PSD output is `("ML", "AP", "freq")`.
3. **Windowed grid features**: `DIMS_IEEG_GRID_WINDOW = ("time_win", "ML", "AP")` exists as a design target.
4. **Dimension constants**: `DIMS_GRID_SPECTRUM`, `DIMS_GRID_FEATURE_MAP`, `DIMS_CHANNEL_FEATURE_MAP` are defined.

### Schema friction for the demo

| Issue | Impact | Severity |
|-------|--------|----------|
| Spatial dim order mismatch | `psdx()` emits `("ML", "AP", "freq")`; spatial measures expect `(..., AP, ML)` | Medium — requires transpose at composition boundary |
| No spectrum validator | `coerce_grid_spectrum()` does not exist; PSD outputs are not validated | Low — convention is stable enough for scripts |
| Feature map validators missing | `coerce_grid_feature_map()`, `coerce_channel_feature_map()` not implemented | Low — relevant if storing intermediate products |
| `ftest_line_scan` is 1D-only | Cannot directly accept grid PSD; must be mapped over spatial dims | Medium — needs `apply_ufunc` or explicit loop |
| Window axis naming | External pipeline uses `"time"` for window centers; schema says `"time_win"` | Low — cosmetic; scripts can rename |

### Compatibility with `psdx` and external scripts

The PSD-first spectral feature pattern composes well:

```
psd_da = psdx(sigx)                          # xr.DataArray (..., freq)
nb_ratio = narrowband_ratio(psd_da.data, psd_da["freq"].values)  # (..., freq)
flatness = spectral_flatness(psd_da.data, psd_da["freq"].values) # (...,)
```

This is the same pattern already used in the external pipeline. The main gap is that spectral feature functions return numpy, so re-wrapping into xarray is left to the caller. This is consistent with the two-layer design and should remain so.

For grid PSD, the composition boundary requires one transpose:

```
psd_da = psdx(sigx)                          # ("ML", "AP", "freq")
psd_np = psd_da.transpose("AP", "ML", "freq").data  # (..., AP, ML, freq)
```

This is a known friction point but not worth a schema change. A small helper could encapsulate it (see Composition Helpers below).

### Assessment

Schemas are sufficient for the demo. The most valuable schema additions would be `coerce_grid_spectrum()` and `coerce_grid_feature_map()`, but these are not blocking.

---

## Practical composition assessment

### How the external pipeline composes CogPy today

Based on `docs/reference/example-snakemake-pipeline`, the real composition model is:

1. **File-first DAG**: Each Snakemake rule produces one serialized artifact (Zarr, `.npy`, `.npz`, `.tsv`, `.png`).
2. **Script-local xarray**: Within each script, xarray objects exist only for the duration of computation. Scripts load, transpose to numpy-friendly order, compute, re-wrap, and save.
3. **Thin orchestration**: Scripts are 50-150 lines each. Logic is linear. No internal pipeline framework is used.
4. **Repeated manual bridging**: `sigx.transpose("AP", "ML", "time").compute().data` appears in `feature.py`, `interpolate.py`, and `rowcol_noise.py`. Quantile-feature stacking appears in `badlabel.py` and `plot_feature_maps.py`.

### How the noise-characterization demo would compose

The demo would follow the same pattern:

```
rule psd:
    input: zarr signal
    output: zarr PSD dataset
    script: compute psdx() per session, save as xr.Dataset

rule spectral_features:
    input: zarr PSD
    output: zarr spectral feature maps
    script: apply narrowband_ratio, spectral_flatness, etc.

rule spatial_features:
    input: zarr spectral feature maps
    output: zarr spatial feature maps
    script: apply moran_i, gradient_anisotropy, marginal_energy_outlier per freq band

rule normalization:
    input: feature maps
    output: normalized feature maps
    script: apply per-frequency z-score, neighborhood normalization

rule summary:
    input: all feature maps
    output: cross-session noise profile (table + figures)
    script: aggregate, rank, visualize
```

Each rule calls 1-3 CogPy functions. No internal pipeline framework is needed.

### Friction points for the demo

1. **PSD serialization**: The external pipeline currently computes PSD ad hoc inside leaf scripts and does not persist it. The demo needs a PSD Zarr intermediate. `psdx()` output is directly serializable via `xr.Dataset({"psd": psd_da}).to_zarr(...)`, so this is straightforward.

2. **Applying spectral features to grids**: Spectral feature functions accept `(..., freq)` numpy arrays. For grid PSD `(AP, ML, freq)`, they broadcast correctly. No loop needed — this already works.

3. **Applying spatial measures to per-frequency maps**: For a PSD grid `(AP, ML, freq)`, computing `moran_i` per frequency requires either:
   - a loop: `[moran_i(psd[:, :, fi]) for fi in range(nf)]`
   - or treating freq as a batch dim: `moran_i(psd.transpose(2, 0, 1))` → `(freq,)` since `moran_i` supports `(..., AP, ML)` batching

   The batched approach already works. This is a strength.

4. **Sliding F-test**: `ftest_line_scan` currently accepts only 1D input. For grid signals, the caller must map it over `(AP, ML)`. This is the most significant composition gap.

### Assessment

CogPy's existing primitives compose well for the demo. The external Snakemake pattern works as-is. The main gaps are:
- A small number of missing primitives (listed below)
- No persistent PSD product convention (trivial to establish)
- `ftest_line_scan` 1D limitation (needs vectorization or a grid wrapper)

---

## Internal pipeline stance

### Recommendation: avoid owning workflow orchestration

The evidence is clear:

1. The external pipeline is the real composition target. It uses Snakemake, not any CogPy internal abstraction.
2. `DetectionPipeline` and the prebuilt detector pipelines are not exercised by the external pipeline.
3. The internal workflow at `src/cogpy/workflows/preprocess/` uses deprecated modules and is not the canonical reference.
4. Every script is 50-150 lines of straightforward load-compute-save logic. There is no composition complexity that demands a framework.

CogPy should:
- **Not** introduce a workflow engine, pipeline builder, or orchestration layer.
- **Not** expand `DetectionPipeline` to cover noise characterization.
- **Not** add configuration-driven feature dispatch (the external pipeline already handles this via Snakemake params).

### What CogPy should provide

1. **Compute primitives**: Pure functions with `(..., spatial, freq)` batch conventions.
2. **Schema validators**: Lightweight `coerce_*` / `validate_*` at pipeline boundaries.
3. **Thin composition helpers**: Small, reusable functions that eliminate repeated script glue (see below).

### Boundary: what counts as a "helper" vs. a "framework"

A helper is a function that:
- Accepts and returns standard types (numpy, xarray, dict)
- Has no global state, registry, or configuration
- Does one thing
- Can be ignored without breaking anything else

A framework is anything that:
- Manages execution order
- Dispatches based on configuration
- Requires the caller to inherit from a base class
- Couples unrelated features through shared state

The noise-characterization demo needs helpers, not a framework.

---

## Minimal additions

### Required new primitives

| # | Name | Purpose | Effort |
|---|------|---------|--------|
| 1 | `narrowband_to_broadband_ratio` | Per-frequency narrowband-to-broadband ratio generalized beyond a single known line frequency | Low — compose from `band_power` and `narrowband_ratio` logic |
| 2 | `spectral_peak_freqs` | Detect discrete peak frequencies from a PSD or `narrowband_ratio` output | Low — thin wrapper around `scipy.signal.find_peaks` on `narrowband_ratio` |
| 3 | `temporal_stability` | Coefficient of variation of a scalar feature across windows | Low — `std / mean` along window axis |
| 4 | `spatial_kurtosis` | Kurtosis of the spatial amplitude distribution | Low — `scipy.stats.kurtosis` on flattened `(AP, ML)` |
| 5 | `spatial_frequency_ratio` | Low-to-high spatial frequency energy ratio on a grid | Medium — 2D FFT + band integration |
| 6 | `spatial_noise_concentration` | Fraction of total grid energy in top-k electrodes | Low — one-liner |

### Required vectorization extensions

| # | Name | Current limitation | Fix |
|---|------|--------------------|-----|
| 1 | `ftest_line_scan` batch support | 1D input only | Add `(..., time)` batch support via `np.apply_along_axis` or explicit reshape |

### Required composition helpers

| # | Name | Purpose |
|---|------|---------|
| 1 | `grid_psd_to_spatial_order` | Transpose `psdx()` output from `("ML", "AP", "freq")` to `("AP", "ML", "freq")` |
| 2 | `score_to_bouts` | Convert a 1D score time series to event intervals via dual threshold |
| 3 | `grid_feature_quantiles` | Compute quantile summaries from a feature dataset, stack to flat channels |

---

## Proposed APIs

### 1. `narrowband_to_broadband_ratio`

```
module:       src/cogpy/core/spectral/features.py
function:     narrowband_to_broadband_ratio(psd, freqs, *, flank_hz=5.0, broadband_range=None)
input:        psd: (..., freq) ndarray, freqs: (freq,) ndarray
output:       (..., freq) ndarray — ratio of narrowband peak power to local broadband floor
vectorization: fully vectorized; freq axis last, arbitrary batch dims
xarray compat: numpy-first; caller wraps output if needed (standard CogPy pattern)
orchestration: stateless pure function; no file I/O
```

Implementation: For each frequency bin, compute `psd[bin] / median(psd[broadband_range])`. If `broadband_range` is None, use the full frequency range excluding the flank region around the current bin. This generalizes `line_noise_ratio` (which targets a single known frequency) to produce a per-bin profile.

Difference from `narrowband_ratio`: `narrowband_ratio` uses local flanking bins as the denominator. `narrowband_to_broadband_ratio` uses the global broadband floor. The former detects local peaks; the latter measures peak prominence against the full spectrum.

### 2. `spectral_peak_freqs`

```
module:       src/cogpy/core/spectral/features.py
function:     spectral_peak_freqs(psd, freqs, *, prominence=2.0, min_distance_hz=2.0)
input:        psd: (..., freq) ndarray, freqs: (freq,) ndarray
output:       list[ndarray] — one array of peak frequencies per batch element
              (or for 1D input: single ndarray of peak frequencies)
vectorization: batch loop over leading dims (peaks are variable-count per spectrum)
xarray compat: numpy-first; returns list/array, not xarray
orchestration: stateless pure function
```

Implementation: Apply `scipy.signal.find_peaks` to each spectrum with `prominence` and `distance` (converted from Hz to bins). Return the frequency values at detected peaks. This is a thin wrapper, not a reimplementation.

### 3. `temporal_stability`

```
module:       src/cogpy/core/measures/temporal.py
function:     temporal_stability(x, *, axis=-1)
input:        x: (..., time) ndarray — windowed scalar feature values
output:       (...,) ndarray — coefficient of variation (std / |mean|)
vectorization: fully vectorized via numpy
xarray compat: numpy-first
orchestration: stateless pure function
```

Implementation: `np.nanstd(x, axis=axis) / (np.abs(np.nanmean(x, axis=axis)) + EPS)`. Lower values indicate more stable features.

### 4. `spatial_kurtosis`

```
module:       src/cogpy/core/measures/spatial.py
function:     spatial_kurtosis(grid)
input:        grid: (..., AP, ML) ndarray
output:       (...,) ndarray — excess kurtosis of flattened spatial distribution
vectorization: fully vectorized; reshape (..., AP, ML) to (..., AP*ML) then reduce
xarray compat: numpy-first
orchestration: stateless pure function
```

Implementation: Flatten last two dims, compute `scipy.stats.kurtosis(flat, axis=-1, nan_policy='omit')`. High kurtosis indicates spatially concentrated energy (few hot electrodes).

### 5. `spatial_frequency_ratio`

```
module:       src/cogpy/core/measures/spatial.py
function:     spatial_frequency_ratio(grid, *, cutoff_fraction=0.25)
input:        grid: (..., AP, ML) ndarray
output:       (...,) ndarray — log ratio of low to high spatial frequency energy
vectorization: fully vectorized; 2D FFT over last two dims
xarray compat: numpy-first
orchestration: stateless pure function
```

Implementation: Compute 2D FFT of the `(AP, ML)` grid. Split the spatial frequency domain at `cutoff_fraction` of the Nyquist frequency (radial). Return `log2(E_low / E_high)`. Positive values indicate spatially smooth (biological); negative values indicate spatially high-frequency (noise/artifact).

### 6. `spatial_noise_concentration`

```
module:       src/cogpy/core/measures/spatial.py
function:     spatial_noise_concentration(grid, *, k=3)
input:        grid: (..., AP, ML) ndarray
output:       (...,) ndarray — fraction of total energy in top-k electrodes
vectorization: fully vectorized; sort along flattened spatial dim
xarray compat: numpy-first
orchestration: stateless pure function
```

Implementation: Flatten `(AP, ML)`, sort descending, return `sum(top_k) / sum(all)`. Values near 1.0 indicate energy concentrated in few channels.

### 7. `ftest_line_scan` batch extension

```
module:       src/cogpy/core/spectral/features.py
function:     ftest_line_scan(signal, fs, *, NW=4.0, p_threshold=0.05)
              (modify existing function)
input:        signal: (..., time) ndarray  [currently: (time,) only]
output:       tuple (fstat, freqs, sig_mask) where fstat/sig_mask are (..., freq)
vectorization: reshape to (batch, time), apply per-row, reshape back
xarray compat: numpy-first
orchestration: stateless pure function
```

The DPSS computation and eigenvalue weighting are shared across batch elements (same `N`, `NW`, `K`), so the tapered FFT can be computed as a single batched matrix multiply. Only the F-statistic reduction needs per-element computation.

---

## Schema adjustments

### Recommended additions (not blocking)

These follow the existing `validate_*` / `coerce_*` pattern in `datasets/schemas.py`.

#### `coerce_grid_spectrum`

```python
def coerce_grid_spectrum(da, *, fs=None):
    """Coerce to canonical grid spectrum dims ("AP", "ML", "freq")."""
    # Accept ("ML", "AP", "freq") from psdx() and transpose
    # Validate freq is 1D increasing
    # Warn if fs missing from attrs
```

This would eliminate the repeated `psd_da.transpose("AP", "ML", "freq")` in scripts.

#### `coerce_grid_feature_map`

```python
def coerce_grid_feature_map(da):
    """Coerce to canonical grid feature map dims ("AP", "ML")."""
    # Accept ("ML", "AP") and transpose
    # Validate AP, ML coords exist
    # Warn if feature_name missing from attrs
```

### Not recommended

- Do not add windowed spectrum validators yet. The demo does not serialize windowed PSD.
- Do not change existing dim conventions. The `("time", "ML", "AP")` vs `("AP", "ML")` split between signals and feature maps is stable and understood.
- Do not enforce `"time_win"` vs `"time"` for window centers. Scripts already use `"time"` and renaming would break the external pipeline.

---

## Composition helpers

These are small, stateless functions that reduce repeated script glue without turning CogPy into a workflow engine.

### 1. `grid_psd_to_spatial_order`

```python
# Location: src/cogpy/core/spectral/specx.py (or a new spectral/utils.py)
def grid_psd_to_spatial_order(psd_da: xr.DataArray) -> xr.DataArray:
    """Transpose grid PSD from psdx() output order to spatial-measure order.

    Input:  ("ML", "AP", "freq") — as returned by psdx() on grid signals
    Output: ("AP", "ML", "freq") — as expected by spatial measure functions
    """
```

This is a one-liner, but naming it makes the intent explicit and eliminates repeated `transpose("AP", "ML", "freq")` calls.

### 2. `score_to_bouts`

```python
# Location: src/cogpy/core/detect/utils.py
def score_to_bouts(
    score: np.ndarray,
    times: np.ndarray,
    *,
    low: float,
    high: float,
    min_duration: float = 0.0,
    merge_gap: float = 0.0,
) -> list[dict]:
    """Convert a 1D score time series to event bouts via dual threshold.

    Wraps dual_threshold_events_1d with duration filtering and gap merging.
    Returns list of dicts with keys: t0, t1, t, value, duration.
    """
```

This composes `dual_threshold_events_1d` with `merge_intervals` and a minimum-duration filter. The pattern is already used in ripple/burst detection but not yet exposed as a standalone helper for arbitrary scores.

### 3. `grid_feature_quantiles`

```python
# Location: src/cogpy/core/preprocess/badchannel/pipeline.py (or a new preprocess/feature_utils.py)
def grid_feature_quantiles(
    ds: xr.Dataset,
    *,
    quantiles: Sequence[float] = (0.8, 0.85, 0.9, 0.95),
    time_dim: str = "time",
) -> xr.DataArray:
    """Compute quantile summaries from a windowed feature dataset and stack to (ch, qfeat).

    Input:  xr.Dataset with variables of dims (AP, ML, time_dim)
    Output: xr.DataArray with dims (ch, qfeat) where ch = stacked(AP, ML)
            and qfeat = stacked(feature, quantile)
    """
```

This eliminates the duplicated quantile-stack-transpose code in `badlabel.py` and `plot_feature_maps.py`.

### Not recommended as helpers

- **Sidecar bundle class**: Useful but out of scope for the noise-characterization demo. Defer.
- **Window-aware repair helper**: Specific to the interpolation script. Not needed for noise characterization.
- **Feature dispatch registry**: Would be a framework, not a helper. External pipeline already dispatches via Snakemake config.

---

## Implementation order

### Phase 1: Core spectral primitives for the demo

#### 1. `narrowband_ratio` — already exists

No implementation needed. Verify it composes correctly with `psdx()` output on grid data:

```python
psd_da = psdx(sigx)
nb = narrowband_ratio(psd_da.transpose("AP", "ML", "freq").data, psd_da["freq"].values)
# nb shape: (AP, ML, freq)
```

**Test**: Apply to a synthetic grid signal with known 50 Hz line noise. Verify `nb` peaks at 50 Hz harmonics.

#### 2. `row/column energy profile` — already exists

`marginal_energy_outlier` is implemented and supports batch dims. Verify it works on per-frequency-band PSD grids:

```python
psd_band = band_power(psd_np, freqs, (45, 55))  # (AP, ML)
result = marginal_energy_outlier(psd_band)
# result["row_energy"]: (AP,), result["col_energy"]: (ML,)
```

**Test**: Apply to a grid with one high-energy row. Verify the row is flagged.

### Phase 2: Normalization and spatial helpers

#### 3. Spatial normalization helpers — partially exist

The building blocks exist in `badchannel/spatial.py`. For the demo, the composition pattern is:

```python
# Per-frequency normalization
psd_z = (psd_np - np.nanmedian(psd_np, axis=-1, keepdims=True)) / (
    np.nanmedian(np.abs(psd_np - np.nanmedian(psd_np, axis=-1, keepdims=True)), axis=-1, keepdims=True) * 1.4826 + EPS
)
```

This is already expressible with `zscorex(psd_da, dim="freq", robust=True)`. No new function needed.

For grid neighborhood normalization of feature maps, `local_robust_zscore_grid` already exists. Verify it works on `(AP, ML)` feature maps derived from PSD:

```python
from cogpy.preprocess.badchannel.spatial import local_robust_zscore_grid
from cogpy.preprocess.badchannel.grid import make_footprint
footprint = make_footprint(rank=2, connectivity=1, niter=1)
nb_z = local_robust_zscore_grid(nb_ratio_at_50hz, footprint=footprint)
```

**Test**: Apply to a grid with one spatially isolated high-noise electrode. Verify it scores high after neighborhood normalization.

### Phase 3: Event extraction and TTL reuse

#### 4. TTL artifact reuse

Not a new CogPy function. The external pipeline already handles TTL epoch selection via brainstate metadata. For the demo, the script loads epoch boundaries from a sidecar file and slices the signal. CogPy's role is to provide the compute primitives that operate on the sliced windows.

No CogPy changes needed. Document the composition pattern:

```python
# In external script
epoch_start, epoch_end = load_epoch_boundaries(sidecar_path)
sigx_epoch = sigx.sel(time=slice(epoch_start, epoch_end))
psd_da = psdx(sigx_epoch)
```

#### 5. `score_to_bouts` / event extraction helpers

Implement `score_to_bouts` as described in the Composition Helpers section. This composes `dual_threshold_events_1d` with gap merging and duration filtering.

**Location**: `src/cogpy/core/detect/utils.py`

```python
def score_to_bouts(score, times, *, low, high, min_duration=0.0, merge_gap=0.0):
    events = dual_threshold_events_1d(score, times, low=low, high=high)
    if merge_gap > 0:
        intervals = [(e["t0"], e["t1"]) for e in events]
        merged = merge_intervals(intervals, gap=int(merge_gap * fs))
        events = _intervals_to_events(score, times, merged)
    if min_duration > 0:
        events = [e for e in events if e["duration"] >= min_duration]
    return events
```

**Test**: Apply to a synthetic score with two close bouts. Verify merging works.

### Phase 4: Directional spatial measures

#### 6. Directional Moran's I — already exists

`moran_i(grid, adjacency="ap_only")` and `moran_i(grid, adjacency="ml_only")` are already implemented. The demo composes them as:

```python
I_ap = moran_i(feature_map, adjacency="ap_only")   # row structure
I_ml = moran_i(feature_map, adjacency="ml_only")   # column structure
# High I_ml + low I_ap → row-striped artifact
# High I_ap + low I_ml → column-striped artifact
```

Verify batch support works for per-frequency maps:

```python
psd_t = psd_np.transpose(2, 0, 1)  # (freq, AP, ML) — freq as batch dim
I_ap_per_freq = moran_i(psd_t, adjacency="ap_only")  # (freq,)
```

**Test**: Create a synthetic grid with row stripes at 50 Hz. Verify `I_ml >> I_ap` at 50 Hz.

### Phase 5: Sliding F-test

#### 7. Sliding F-test line scanner

This is the most significant new work. Two parts:

**Part A: Vectorize `ftest_line_scan` for batch input**

Modify `ftest_line_scan` to accept `(..., time)` input. The DPSS tapers and eigenvalues depend only on `N`, `NW`, `K` — shared across batch elements. Compute tapered FFTs as a single batched operation, then reduce to F-statistics per element.

```python
# Current: signal shape (time,) → fstat shape (freq,)
# New:     signal shape (..., time) → fstat shape (..., freq)
```

Estimated effort: medium. The core F-test math is already correct; the change is reshaping the input and vectorizing the taper application.

**Part B: Sliding window wrapper (external script)**

The sliding-window application is the orchestrator's job. In the demo script:

```python
from cogpy.utils.sliding_core import sliding_window_view
windows = sliding_window_view(signal_np, window_size, window_step)  # (..., n_win, win_size)
fstat, freqs, sig_mask = ftest_line_scan(windows, fs, NW=4.0, p_threshold=0.05)
# fstat: (..., n_win, freq)
```

This produces a spatially and temporally resolved line-noise map without any new CogPy abstraction.

**Test**: Apply to a synthetic signal with transient 50 Hz injection. Verify the F-test detects it only in the injected windows.

---

## Summary of implementation priorities

| Priority | Item | Status | Effort | Blocking? |
|----------|------|--------|--------|-----------|
| 1 | Verify `narrowband_ratio` grid composition | EXISTS | Test only | No |
| 2 | Verify `marginal_energy_outlier` per-frequency | EXISTS | Test only | No |
| 3 | Verify spatial normalization on PSD-derived maps | EXISTS | Test only | No |
| 4 | Document TTL epoch composition pattern | N/A | Docs only | No |
| 5 | Implement `score_to_bouts` | NEW | Low | No — useful but not blocking |
| 6 | Verify directional `moran_i` batch support | EXISTS | Test only | No |
| 7 | Vectorize `ftest_line_scan` for batch input | MODIFY | Medium | Yes — needed for grid line-noise maps |
| 8 | Implement `spectral_peak_freqs` | NEW | Low | No — convenience |
| 9 | Implement `spatial_kurtosis` | NEW | Low | No — one-liner |
| 10 | Implement `spatial_frequency_ratio` | NEW | Medium | No — optional for demo |
| 11 | Implement `spatial_noise_concentration` | NEW | Low | No — one-liner |
| 12 | Add `coerce_grid_spectrum` | NEW | Low | No — quality of life |
| 13 | Add `grid_feature_quantiles` helper | NEW | Low | No — DRY improvement |

The only blocking item is vectorizing `ftest_line_scan`. Everything else either exists or is a low-effort addition that improves but does not gate the demo.
