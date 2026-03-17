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

## Capability areas

cogpy organizes its primitives into the following capability areas.

### Events (`cogpy.core.detect`, `cogpy.core.events`)

Detect, refine, and relate discrete events in signals.

| Primitive | Module | Purpose |
|-----------|--------|---------|
| `ThresholdDetector` | `detect.threshold` | Threshold-crossing detection |
| `BurstDetector` | `detect.burst` | Spectral peak (h-maxima) detection |
| `score_to_bouts` | `detect.utils` | Score → interval events via dual threshold |
| `merge_intervals` | `detect.utils` | Merge adjacent intervals |
| `EventCatalog` | `events.catalog` | Unified event container |
| `detect_overlaps` | `events.overlap` | Find overlapping intervals |
| `match_nearest` | `events.match` | Nearest-neighbor event matching |
| `match_nearest_symmetric` | `events.match` | Bijective event matching |
| `event_lag_histogram` | `events.match` | Cross-correlogram between event trains |
| `estimate_lag` | `events.match` | Constant lag estimation |
| `estimate_drift` | `events.match` | Polynomial drift estimation |

### Triggered analysis (`cogpy.core.triggered`)

Extract and summarize signal epochs aligned to events.

| Primitive | Module | Purpose |
|-----------|--------|---------|
| `perievent_epochs` | `brainstates.intervals` | Extract time-locked windows |
| `triggered_average` | `triggered.stats` | Mean across epochs |
| `triggered_median` | `triggered.stats` | Robust median across epochs |
| `triggered_std` | `triggered.stats` | Variability across epochs |
| `triggered_snr` | `triggered.stats` | Consistency of event-locked response |
| `estimate_template` | `triggered.template` | Template from epoch stack |
| `fit_scaling` | `triggered.template` | Per-event amplitude coefficients |
| `subtract_template` | `triggered.template` | Remove template at event locations |

### Regression (`cogpy.core.regression`)

Linear model building blocks for component removal.

| Primitive | Module | Purpose |
|-----------|--------|---------|
| `lagged_design_matrix` | `regression.design` | Toeplitz matrix from reference signal |
| `event_design_matrix` | `regression.design` | Place template at event times |
| `ols_fit` | `regression.ols` | Least-squares coefficient estimation |
| `ols_predict` | `regression.ols` | Predict from design matrix + coefficients |
| `ols_residual` | `regression.ols` | Residual after prediction removal |

### Spectral (`cogpy.core.spectral`)

Frequency-domain analysis.

| Primitive | Module | Purpose |
|-----------|--------|---------|
| `psd_welch`, `psd_multitaper` | `spectral.psd` | Power spectral density |
| `spectrogramx` | `spectral.specx` | Time-frequency representation |
| `band_power` | `spectral.features` | Integrate PSD over frequency band |
| `spectral_peak_freqs` | `spectral.features` | Find spectral peaks |
| `ftest_line_scan` | `spectral.features` | F-test for sinusoidal lines |
| `line_noise_ratio` | `spectral.features` | Narrowband contamination metric |
| `narrowband_ratio` | `spectral.features` | Per-bin peak-to-background ratio |
| `coherence`, `plv` | `spectral.bivariate` | Cross-signal coupling |

### Spatial (`cogpy.core.measures.spatial`, `cogpy.core.preprocess.filtering.spatial`)

Grid-electrode spatial analysis.

| Primitive | Module | Purpose |
|-----------|--------|---------|
| `moran_i` | `measures.spatial` | Spatial autocorrelation |
| `gradient_anisotropy` | `measures.spatial` | Directional gradient ratio |
| `marginal_energy_outlier` | `measures.spatial` | Row/column energy outliers |
| `csd_power` | `measures.spatial` | Current source density |
| `gaussian_spatialx` | `preprocess.filtering.spatial` | Spatial smoothing |
| `median_subtractx` | `preprocess.filtering.spatial` | Spatial median reference |
| PCA / varimax | `decomposition.pca` | Low-rank spatial decomposition |

### Validation (`cogpy.core.measures.comparison`)

Before/after comparison metrics.

| Primitive | Module | Purpose |
|-----------|--------|---------|
| `snr_improvement` | `measures.comparison` | Change in SNR (dB) |
| `residual_energy_ratio` | `measures.comparison` | Energy fraction removed |
| `bandpower_change` | `measures.comparison` | Fractional change in band power |
| `waveform_residual_rms` | `measures.comparison` | Template waveform difference |

## How primitives compose

Primitives are designed to be chained in arbitrary order by the caller.
Here are conceptual composition patterns (pseudo-code, not runnable pipelines):

### Pattern: event-triggered template removal

```python
# 1. Detect events in a reference signal
events = threshold_detect(reference_signal, threshold=3.0)

# 2. Match events to another stream
idx_a, idx_b, lags = match_nearest(events_a, events_b, max_lag=0.01)

# 3. Extract epochs around events
epochs = perievent_epochs(signal, event_times, fs, pre=0.01, post=0.01)

# 4. Estimate and subtract template
template = estimate_template(epochs, method="median")
alpha = fit_scaling(epochs, template)
cleaned = subtract_template(signal, event_samples, template, scaling=alpha)

# 5. Validate
psd_before = psd_multitaper(signal, fs)
psd_after = psd_multitaper(cleaned, fs)
delta = bandpower_change(psd_before, psd_after, freqs, band=(100, 140))
```

### Pattern: regression-based component removal

```python
# 1. Build lagged design matrix from reference
X = lagged_design_matrix(reference, lags=range(0, 20))

# 2. Fit and subtract per channel
beta = ols_fit(X, multichannel_signal)
cleaned = ols_residual(X, multichannel_signal, beta)
```

### Pattern: spatial characterization

```python
# Compute spatial structure of an event-locked map
epochs = perievent_epochs(grid_signal, event_times, fs, pre=0.005, post=0.005)
avg_map = triggered_average(epochs)
aniso = gradient_anisotropy(avg_map)
moran = moran_i(avg_map, adjacency="queen")
```

Each step is independent and reusable. The *composition* lives outside cogpy.
