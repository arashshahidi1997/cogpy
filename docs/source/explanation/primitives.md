# Atomic Primitives — Design Philosophy

cogpy is a library of **atomic, composable operators** for electrophysiology
signal processing. It is *not* a pipeline framework.

## Core principle

Every function in cogpy should be:

- **Small**: does one thing.
- **Pure**: no hidden state, no side effects, no I/O.
- **Composable**: inputs and outputs are standard types (numpy arrays,
  xarray DataArrays, pandas DataFrames).
- **Domain-agnostic**: no function should "know" about specific hardware,
  experiments, or artifact types.

High-level orchestration — sequencing these primitives into a workflow for a
specific dataset — belongs **outside** cogpy, in Snakemake pipelines,
notebooks, or project-specific code.

## Quick-reference imports

Grouped by task. All paths below use the shortest public route.

```python
# --- Event detection & matching ---
from cogpy.detect import ThresholdDetector, BurstDetector
from cogpy.detect.utils import score_to_bouts, merge_intervals
from cogpy.events import EventCatalog
from cogpy.events.match import (
    match_nearest, match_nearest_symmetric,
    estimate_lag, estimate_drift, event_lag_histogram,
)

# --- Triggered analysis & templates ---
from cogpy.brainstates.intervals import perievent_epochs
from cogpy.triggered import (
    triggered_average, triggered_std, triggered_median, triggered_snr,
    estimate_template, fit_scaling, subtract_template,
)

# --- Regression ---
from cogpy.regression import (
    lagged_design_matrix, event_design_matrix,
    ols_fit, ols_predict, ols_residual,
)

# --- Spectral ---
from cogpy.spectral.psd import psd_welch, psd_multitaper
from cogpy.spectral.specx import spectrogramx, psdx
from cogpy.spectral.features import (
    band_power, relative_band_power, spectral_peak_freqs,
    ftest_line_scan, line_noise_ratio, narrowband_ratio,
    am_artifact_score, am_depth,
)
from cogpy.spectral.bivariate import coherence, plv, cross_spectrum

# --- Spatial measures & filtering ---
from cogpy.measures.spatial import (
    moran_i, gradient_anisotropy, spatial_kurtosis,
    marginal_energy_outlier, csd_power, spatial_coherence_profile,
)
from cogpy.preprocess.filtering.spatial import (
    gaussian_spatialx, median_spatialx, median_subtractx,
)
from cogpy.preprocess.filtering.reference import cmrx

# --- Decomposition ---
from cogpy.decomposition.pca import erpPCA

# --- Validation metrics ---
from cogpy.measures.comparison import (
    snr_improvement, residual_energy_ratio,
    bandpower_change, waveform_residual_rms,
)
```

## Capability areas

cogpy organizes its primitives into the following capability areas.
Each table lists the function, its import module, signature sketch, and purpose.

### Events

Detect, refine, and relate discrete events in signals.

| Primitive | Import from | Signature | Purpose |
|-----------|-------------|-----------|---------|
| `ThresholdDetector` | `cogpy.detect` | `(threshold, direction, bandpass, use_envelope, min_duration, merge_gap)` | Threshold-crossing detection |
| `BurstDetector` | `cogpy.detect` | `(h_quantile, h, nperseg, noverlap, bandwidth, footprint)` | Spectral peak (h-maxima) detection |
| `score_to_bouts` | `cogpy.detect.utils` | `(score, times, low, high, min_duration, merge_gap) → list[dict]` | Score → interval events via dual threshold |
| `merge_intervals` | `cogpy.detect.utils` | `(intervals, gap) → list[tuple]` | Merge adjacent intervals |
| `EventCatalog` | `cogpy.events` | container (DataFrame-backed) | Unified event container with filter/query methods |
| `detect_overlaps` | `cogpy.events.overlap` | `(catalog) → DataFrame` | Find overlapping intervals |
| `match_nearest` | `cogpy.events.match` | `(times_a, times_b, max_lag) → (idx_a, idx_b, lags)` | Nearest-neighbor event matching (greedy) |
| `match_nearest_symmetric` | `cogpy.events.match` | `(times_a, times_b, max_lag) → (idx_a, idx_b, lags)` | Bijective 1:1 event matching |
| `event_lag_histogram` | `cogpy.events.match` | `(times_a, times_b, max_lag, bin_width) → (counts, bin_edges)` | Cross-correlogram between event trains |
| `estimate_lag` | `cogpy.events.match` | `(times_a, times_b, max_lag, method) → float` | Constant lag estimate (median/mean/mode) |
| `estimate_drift` | `cogpy.events.match` | `(times_a, times_b, max_lag, degree) → ndarray` | Polynomial drift coefficients |

### Triggered analysis

Extract and summarize signal epochs aligned to events.

| Primitive | Import from | Signature | Purpose |
|-----------|-------------|-----------|---------|
| `perievent_epochs` | `cogpy.brainstates.intervals` | `(xsig, events, fs, *, pre, post) → DataArray(event, ..., lag)` | Extract time-locked windows |
| `triggered_average` | `cogpy.triggered` | `(epochs, *, event_dim) → DataArray` | Mean across epochs |
| `triggered_median` | `cogpy.triggered` | `(epochs, *, event_dim) → DataArray` | Robust median across epochs |
| `triggered_std` | `cogpy.triggered` | `(epochs, *, event_dim, ddof) → DataArray` | Variability across epochs |
| `triggered_snr` | `cogpy.triggered` | `(epochs, *, event_dim) → DataArray` | Consistency of event-locked response |
| `estimate_template` | `cogpy.triggered` | `(epochs, *, method, event_dim) → DataArray` | Template from epoch stack (mean/median/trimmean) |
| `fit_scaling` | `cogpy.triggered` | `(epochs, template) → ndarray(n_events,)` | Per-event amplitude via dot-product projection |
| `subtract_template` | `cogpy.triggered` | `(signal, event_samples, template, *, scaling) → array` | Remove scaled template at event locations |

### Regression

Linear model building blocks for component removal.

| Primitive | Import from | Signature | Purpose |
|-----------|-------------|-----------|---------|
| `lagged_design_matrix` | `cogpy.regression` | `(reference, lags, *, intercept) → ndarray(n_time, n_lags)` | Toeplitz matrix from reference signal |
| `event_design_matrix` | `cogpy.regression` | `(n_time, event_samples, template, *, intercept) → ndarray` | Place template at event times |
| `ols_fit` | `cogpy.regression` | `(X, Y, *, rcond) → ndarray(n_features, n_channels)` | Least-squares coefficient estimation |
| `ols_predict` | `cogpy.regression` | `(X, beta) → ndarray` | Predicted signal from design + coefficients |
| `ols_residual` | `cogpy.regression` | `(X, Y, beta) → ndarray` | `Y - X @ beta` |

### Spectral

Frequency-domain analysis.

| Primitive | Import from | Signature | Purpose |
|-----------|-------------|-----------|---------|
| `psd_welch` | `cogpy.spectral.psd` | `(arr, fs, *, nperseg, axis) → (psd, freqs)` | Welch PSD estimate |
| `psd_multitaper` | `cogpy.spectral.psd` | `(arr, fs, *, NW, K, axis) → (psd, freqs)` | Multitaper PSD estimate |
| `psdx` | `cogpy.spectral.specx` | `(sigx, *, method, bandwidth) → DataArray(freq)` | xarray PSD wrapper |
| `spectrogramx` | `cogpy.spectral.specx` | `(sigx, *, bandwidth, nperseg) → DataArray(..., freq, time)` | xarray multitaper spectrogram |
| `band_power` | `cogpy.spectral.features` | `(psd, freqs, band) → (...,)` | Integrate PSD over band (trapezoid) |
| `relative_band_power` | `cogpy.spectral.features` | `(psd, freqs, band, *, norm_range) → (...,)` | Normalized band power ratio |
| `spectral_peak_freqs` | `cogpy.spectral.features` | `(psd, freqs, *, prominence, min_distance_hz) → ndarray` | Find spectral peaks (Hz) |
| `ftest_line_scan` | `cogpy.spectral.features` | `(signal, fs, *, NW, p_threshold) → (fstat, freqs, sig_mask)` | F-test for sinusoidal lines |
| `line_noise_ratio` | `cogpy.spectral.features` | `(psd, freqs, *, f_line, bw) → (...,)` | Narrowband contamination at f_line |
| `narrowband_ratio` | `cogpy.spectral.features` | `(psd, freqs, *, flank_hz) → (..., freq)` | Per-bin peak-to-background ratio |
| `am_artifact_score` | `cogpy.spectral.features` | `(psd, freqs, *, fc, fm) → (...,)` | Amplitude-modulation sideband score |
| `coherence` | `cogpy.spectral.bivariate` | `(mtfft_x, mtfft_y) → (..., freq)` | Magnitude squared coherence |
| `plv` | `cogpy.spectral.bivariate` | `(mtfft_x, mtfft_y) → (..., freq)` | Phase locking value |
| `cross_spectrum` | `cogpy.spectral.bivariate` | `(mtfft_x, mtfft_y) → (..., freq) complex` | Cross-spectral density |

### Spatial

Grid-electrode spatial analysis and filtering.

| Primitive | Import from | Signature | Purpose |
|-----------|-------------|-----------|---------|
| `moran_i` | `cogpy.measures.spatial` | `(grid, adjacency) → (...,)` | Spatial autocorrelation (queen/rook/directional) |
| `gradient_anisotropy` | `cogpy.measures.spatial` | `(grid) → (...,)` | log₂ ratio of AP vs ML gradients |
| `spatial_kurtosis` | `cogpy.measures.spatial` | `(grid) → (...,)` | Excess kurtosis of spatial amplitude distribution |
| `spatial_noise_concentration` | `cogpy.measures.spatial` | `(grid, k) → (...,)` | Fraction of energy in top-k electrodes |
| `marginal_energy_outlier` | `cogpy.measures.spatial` | `(grid, robust, threshold) → dict` | Row/column energy outlier scores |
| `csd_power` | `cogpy.measures.spatial` | `(grid_signal, spacing_mm) → (AP, ML, time)` | Current source density via 2D Laplacian |
| `spatial_coherence_profile` | `cogpy.measures.spatial` | `(grid_signal, fs, spacing_mm, ...) → (profile, bins, freqs)` | Coherence vs inter-electrode distance |
| `cmrx` | `cogpy.preprocess.filtering.reference` | `(sigx, channel_dims, skipna) → DataArray` | Common median reference |
| `gaussian_spatialx` | `cogpy.preprocess.filtering.spatial` | `(sigx, sigma) → DataArray` | Spatial Gaussian lowpass |
| `median_spatialx` | `cogpy.preprocess.filtering.spatial` | `(sigx, size) → DataArray` | Spatial median lowpass |
| `median_subtractx` | `cogpy.preprocess.filtering.spatial` | `(sigx, dims, skipna) → DataArray` | Spatial median reference |
| `erpPCA` | `cogpy.decomposition.pca` | `(nfac, max_it, tol)` class | Varimax-rotated PCA estimator |

### Validation

Before/after comparison metrics.

| Primitive | Import from | Signature | Purpose |
|-----------|-------------|-----------|---------|
| `snr_improvement` | `cogpy.measures.comparison` | `(psd_before, psd_after, freqs, *, signal_band, noise_band) → float` | Change in SNR (dB) |
| `residual_energy_ratio` | `cogpy.measures.comparison` | `(original, cleaned, *, axis) → ndarray` | Energy fraction remaining |
| `bandpower_change` | `cogpy.measures.comparison` | `(psd_before, psd_after, freqs, *, band) → ndarray` | Fractional change in band power |
| `waveform_residual_rms` | `cogpy.measures.comparison` | `(template_before, template_after) → float` | Template waveform difference |

## How primitives compose

Primitives are designed to be chained in arbitrary order by the caller.
Here are conceptual composition patterns (pseudo-code, not runnable pipelines):

### Pattern: event-triggered template removal

```python
from cogpy.detect import ThresholdDetector
from cogpy.events.match import match_nearest
from cogpy.brainstates.intervals import perievent_epochs
from cogpy.triggered import estimate_template, fit_scaling, subtract_template
from cogpy.spectral.psd import psd_multitaper
from cogpy.measures.comparison import bandpower_change

# 1. Detect events in a reference signal
detector = ThresholdDetector(threshold=3.0, direction="positive")
catalog = detector.detect(reference_signal)
event_times = catalog.df["t"].values

# 2. Match events to another stream
idx_a, idx_b, lags = match_nearest(events_a, events_b, max_lag=0.01)

# 3. Extract epochs around events
epochs = perievent_epochs(signal, event_times, fs, pre=0.01, post=0.01)

# 4. Estimate and subtract template
template = estimate_template(epochs, method="median")
alpha = fit_scaling(epochs.values, template.values)
cleaned = subtract_template(signal, event_samples, template.values, scaling=alpha)

# 5. Validate
psd_before, freqs = psd_multitaper(signal.values, fs)
psd_after, _ = psd_multitaper(cleaned.values, fs)
delta = bandpower_change(psd_before, psd_after, freqs, band=(100, 140))
```

### Pattern: regression-based component removal

```python
from cogpy.regression import lagged_design_matrix, ols_fit, ols_residual

# 1. Build lagged design matrix from reference
X = lagged_design_matrix(reference, lags=range(0, 20))

# 2. Fit and subtract per channel
beta = ols_fit(X, multichannel_signal)
cleaned = ols_residual(X, multichannel_signal, beta)
```

### Pattern: spatial characterization

```python
from cogpy.brainstates.intervals import perievent_epochs
from cogpy.triggered import triggered_average
from cogpy.measures.spatial import gradient_anisotropy, moran_i

# Compute spatial structure of an event-locked map
epochs = perievent_epochs(grid_signal, event_times, fs, pre=0.005, post=0.005)
avg_map = triggered_average(epochs)
aniso = gradient_anisotropy(avg_map)
moran = moran_i(avg_map, adjacency="queen")
```

### Pattern: spectral artifact diagnostics

```python
from cogpy.spectral.psd import psd_multitaper
from cogpy.spectral.features import ftest_line_scan, narrowband_ratio, am_artifact_score

# 1. Scan for sinusoidal lines
fstat, freqs, sig_mask = ftest_line_scan(signal, fs, NW=4.0)

# 2. Per-bin peak-to-background ratio
ratio = narrowband_ratio(psd, freqs, flank_hz=5.0)

# 3. Amplitude modulation sidebands
score = am_artifact_score(psd, freqs, fc=120.0, fm=60.0)
```

Each step is independent and reusable. The *composition* lives outside cogpy.
