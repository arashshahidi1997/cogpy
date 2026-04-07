# Compose artifact analysis from cogpy primitives

This guide shows how to assemble cogpy's atomic primitives for artifact
analysis in an external project (notebook, Snakemake pipeline, or standalone
script). cogpy provides the operators; your project provides the orchestration.

For the full primitive catalog and design philosophy, see
{doc}`/explanation/primitives`.

## Setup

All examples assume:

```python
import numpy as np
import xarray as xr
```

## 1. Detect events in a reference signal

Use when you have a signal (e.g. TTL channel, sync pulse, stimulus trace) that
contains discrete events you want to locate.

```python
from cogpy.detect import ThresholdDetector

detector = ThresholdDetector(
    threshold=3.0,          # in signal units (or z-scored units)
    direction="positive",   # "positive", "negative", or "both"
    min_duration=0.001,     # seconds — reject transients shorter than this
    merge_gap=0.002,        # seconds — merge events closer than this
)
catalog = detector.detect(signal)       # signal: xr.DataArray with "time" dim
event_times = catalog.df["t"].values    # ndarray of event times in seconds
```

**Alternatively**, for a continuous score signal (e.g. envelope, power trace):

```python
from cogpy.detect.utils import score_to_bouts

bouts = score_to_bouts(
    score, times,
    low=2.0, high=3.0,       # dual-threshold: enter at high, exit at low
    min_duration=0.02,
    merge_gap=0.01,
)
# bouts: list of dicts with keys {t0, t, t1, duration, value}
```

## 2. Match events across streams

Use when you have events detected in two different signals (e.g. TTL pulses
recorded on two systems) and need to align them.

```python
from cogpy.events.match import (
    match_nearest_symmetric,
    estimate_lag,
    estimate_drift,
    event_lag_histogram,
)

# One-to-one matching
idx_a, idx_b, lags = match_nearest_symmetric(times_a, times_b, max_lag=0.05)

# Global lag estimate
lag = estimate_lag(times_a, times_b, max_lag=0.05, method="median")

# Drift over time (polynomial fit to lag vs time)
coeffs = estimate_drift(times_a, times_b, max_lag=0.05, degree=1)
# coeffs[0] = drift rate (s/s), coeffs[1] = offset (s)
# predicted lag at time t: np.polyval(coeffs, t)

# Visualize lag distribution
counts, bin_edges = event_lag_histogram(times_a, times_b, max_lag=0.05, bin_width=0.001)
```

## 3. Extract event-locked epochs

Use when you want to cut windows around events from a continuous signal.

```python
from cogpy.brainstates.intervals import perievent_epochs

epochs = perievent_epochs(
    signal,              # xr.DataArray with "time" dim
    event_times,         # 1-D array of event times (seconds)
    fs=1000.0,           # sampling rate
    pre=0.010,           # seconds before event
    post=0.010,          # seconds after event
)
# epochs: DataArray with dims (event, <original non-time dims>, lag)
# lag coordinate: relative time from -pre to +post
```

## 4. Compute triggered statistics

```python
from cogpy.triggered import (
    triggered_average,
    triggered_std,
    triggered_median,
    triggered_snr,
)

avg = triggered_average(epochs)    # mean across events
med = triggered_median(epochs)     # robust median
std = triggered_std(epochs)        # variability
snr = triggered_snr(epochs)        # mean / std ratio
```

## 5. Estimate and subtract a template

Use when each event produces a stereotyped waveform you want to remove.

```python
from cogpy.triggered import estimate_template, fit_scaling, subtract_template

# Estimate template (median is robust to outlier events)
template = estimate_template(epochs, method="median")

# Optional: fit per-event amplitude scaling
alpha = fit_scaling(epochs.values, template.values)

# Subtract from the continuous signal
event_samples = (event_times * fs).astype(int)
cleaned = subtract_template(
    signal, event_samples, template.values, scaling=alpha,
)
```

## 6. Regression-based component removal

Use when the artifact is correlated with a reference signal but the
waveform shape varies (e.g. different propagation delays per channel).

```python
from cogpy.regression import (
    lagged_design_matrix,
    event_design_matrix,
    ols_fit, ols_predict, ols_residual,
)

# Option A: lagged reference regression
X = lagged_design_matrix(reference_1d, lags=range(-10, 11), intercept=True)
beta = ols_fit(X, target_multichannel)      # (n_features, n_channels)
cleaned = ols_residual(X, target_multichannel, beta)

# Option B: event-based template regression
X = event_design_matrix(n_time, event_samples, template_1d, intercept=True)
beta = ols_fit(X, target_multichannel)
cleaned = ols_residual(X, target_multichannel, beta)
```

## 7. Spectral diagnostics

Use to characterize contamination or verify cleaning.

```python
from cogpy.spectral.psd import psd_multitaper
from cogpy.spectral.specx import spectrogramx, psdx
from cogpy.spectral.features import (
    band_power,
    spectral_peak_freqs,
    ftest_line_scan,
    narrowband_ratio,
    line_noise_ratio,
    am_artifact_score,
)

# PSD (numpy)
psd, freqs = psd_multitaper(signal_1d, fs, NW=4)

# PSD (xarray, preserves coords)
psd_da = psdx(signal_xr, method="multitaper", bandwidth=4.0)

# Spectrogram (xarray)
spec = spectrogramx(signal_xr, bandwidth=4.0, nperseg=256)

# Detect spectral peaks
peaks_hz = spectral_peak_freqs(psd, freqs, prominence=2.0)

# F-test for sinusoidal lines (more sensitive than peak detection)
fstat, freqs, sig_mask = ftest_line_scan(signal_1d, fs, NW=4.0)

# Per-bin narrowband ratio (values >> 1 = line artifact)
ratio = narrowband_ratio(psd, freqs, flank_hz=5.0)

# Amplitude modulation sidebands (e.g. carrier=fc, modulator=fm)
score = am_artifact_score(psd, freqs, fc=120.0, fm=60.0)

# Band power
power = band_power(psd, freqs, band=(100, 140))
```

## 8. Spatial analysis

Use for grid-electrode recordings to characterize spatial structure.

```python
from cogpy.measures.spatial import (
    moran_i,
    gradient_anisotropy,
    spatial_kurtosis,
    marginal_energy_outlier,
    csd_power,
)
from cogpy.preprocess.filtering.reference import cmrx
from cogpy.preprocess.filtering.spatial import gaussian_spatialx
from cogpy.decomposition.pca import erpPCA

# Spatial autocorrelation
I = moran_i(grid_map, adjacency="queen")   # +1 smooth, 0 random, -1 checker

# Directional anisotropy (detect row/column striping)
aniso = gradient_anisotropy(grid_map)

# Row/column energy outliers
outliers = marginal_energy_outlier(grid_map, robust=True)

# Common median reference
rereferenced = cmrx(signal_grid)

# Spatial smoothing
smoothed = gaussian_spatialx(signal_grid, sigma=1.0)

# PCA decomposition
pca = erpPCA(nfac=5)
pca.fit(design_matrix)
scores = pca.FSr       # factor scores
loadings = pca.LR      # rotated loadings
```

## 9. Validate before vs after

Use to quantify whether cleaning improved the signal.

```python
from cogpy.measures.comparison import (
    snr_improvement,
    residual_energy_ratio,
    bandpower_change,
    waveform_residual_rms,
)

# SNR change in dB
delta_snr = snr_improvement(
    psd_before, psd_after, freqs,
    signal_band=(1, 40), noise_band=(100, 140),
)

# Fraction of energy remaining (< 1 means energy was removed)
ratio = residual_energy_ratio(original, cleaned)

# Fractional bandpower change in a specific band
delta_bp = bandpower_change(psd_before, psd_after, freqs, band=(100, 140))

# Template waveform difference (for event-locked QC)
rms = waveform_residual_rms(template_before, template_after)
```

## Composition patterns

These primitives are designed to be mixed freely. Common patterns:

- **Detect → epoch → template → subtract → validate**
  (event-triggered template removal)
- **Detect → design matrix → regress → validate**
  (regression-based removal)
- **Detect → epoch → spatial measures**
  (characterize artifact spatial footprint)
- **Before PSD → clean → after PSD → compare**
  (spectral validation loop)

The choice of pattern depends on your data. cogpy provides the building
blocks; your project provides the assembly.
