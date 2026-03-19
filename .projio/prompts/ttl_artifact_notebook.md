# TTL Artifact Characterization and Removal — Notebook Prompt

## Context

You are building an exploratory Jupyter notebook for characterizing and removing a TTL-linked artifact in electrophysiology recordings.

### The problem

A TTL pulse at ~120 Hz contaminates broadband electrophysiology signals. The contamination:
- is event-like (brief transient at each TTL edge)
- is broadband (affects many frequency bands, not just 120 Hz and harmonics)
- varies across channels due to acquisition geometry (different amplitude, latency, waveform shape)
- may exhibit slow drift in timing between the TTL clock and the acquisition clock

This is NOT a simple line-noise problem. It requires:
1. detecting and characterizing the artifact events
2. understanding their spectral and spatial structure
3. removing them without distorting the underlying neural signal
4. validating the removal

### The data

You are working with grid ECoG data loaded as `xarray.DataArray` with dimensions `("AP", "ML", "time")` and a `fs` coordinate/attribute storing sampling rate in Hz. There is also a TTL reference channel available as a 1D signal.

---

## Available cogpy primitives

All operations below are atomic, composable functions. The notebook will compose them — cogpy does NOT contain pipelines.

### Data conventions

- Signals are `xarray.DataArray` with named dimensions
- Grid ECoG: dims `("AP", "ML", "time")` or `("time", "AP", "ML")`
- Flat channels: dims `("time", "ch")`
- Sampling rate: `sig.fs` (0D coordinate) or `sig.attrs["fs"]`
- Events: `np.ndarray` of times in seconds, or `EventCatalog` (pandas DataFrame wrapper)
- Epochs: `xr.DataArray` with dims `("event", ..., "lag")`

### 1. Event detection & matching

```python
from cogpy.detect.threshold import ThresholdDetector
from cogpy.detect.utils import (
    score_to_bouts,       # score timeseries → interval events via dual threshold
    find_true_runs,       # boolean mask → contiguous run indices
    merge_intervals,      # merge adjacent intervals within a gap
)
from cogpy.events.match import (
    match_nearest,            # match each event in A to nearest in B within max_lag
    match_nearest_symmetric,  # bijective (one-to-one) nearest matching
    event_lag_histogram,      # cross-correlogram between two event trains
    estimate_lag,             # constant lag estimation (median/mean/mode of matched lags)
    estimate_drift,           # polynomial drift estimation (polyfit on matched lags vs time)
)
from cogpy.events.catalog import EventCatalog  # unified event container
```

**ThresholdDetector** — generic threshold-crossing detector. Parameters: `threshold`, `direction` ("positive"/"negative"/"both"), optional `bandpass`, `use_envelope`, `min_duration`, `merge_gap`. Returns `EventCatalog`.

**match_nearest(times_a, times_b, max_lag=...)** → `(idx_a, idx_b, lags)`. Greedy nearest-neighbor. Positive lag means B follows A.

**match_nearest_symmetric(...)** → same but bijective (each B event matched at most once).

**estimate_lag(times_a, times_b, max_lag=..., method="median")** → float. Robust constant lag estimate.

**estimate_drift(times_a, times_b, max_lag=..., degree=1)** → polynomial coefficients. `np.polyval(coeffs, t)` gives lag at time t.

### 2. Epoch extraction & triggered statistics

```python
from cogpy.brainstates.intervals import perievent_epochs
# perievent_epochs(xsig, events, fs, pre, post) → xr.DataArray (event, ..., lag)
#   events: array of times in seconds
#   pre/post: seconds before/after event
#   fill_value: padding for out-of-bounds (default NaN)

from cogpy.triggered import (
    triggered_average,   # mean across events → template waveform
    triggered_std,       # std across events (ddof=1 default)
    triggered_median,    # robust median across events
    triggered_snr,       # mean / standard_error — consistency metric
)
```

All accept either `np.ndarray (n_events, ..., n_lag)` or `xr.DataArray` with `event_dim="event"`.

### 3. Template estimation & subtraction

```python
from cogpy.triggered import (
    estimate_template,   # mean/median/trimmean of epoch stack
    fit_scaling,         # per-event least-squares scaling: alpha_i = <epoch_i, template> / <template, template>
    subtract_template,   # subtract scaled template at event locations in continuous signal
)
```

**estimate_template(epochs, method="median")** — "median" is robust to outlier events; "trimmean" for 20% trimmed mean.

**fit_scaling(epochs, template)** → `(n_events,)` array of per-event amplitudes.

**subtract_template(signal, event_samples, template, scaling=None)** — subtracts `scaling[i] * template` at each `event_samples[i]`. Skips out-of-bounds. Preserves xarray metadata.

### 4. Regression-based removal

```python
from cogpy.regression import (
    lagged_design_matrix,  # Toeplitz-like matrix from reference signal + lag set
    event_design_matrix,   # place template at event onsets as separate regressors
    ols_fit,               # np.linalg.lstsq wrapper → beta coefficients
    ols_predict,           # X @ beta
    ols_residual,          # Y - X @ beta
)
```

**lagged_design_matrix(reference, lags, intercept=True)** — builds `(n_time, n_lags + 1)` matrix. Lag k means predictor at time t is reference[t−k]. Zero-padded at boundaries.

**event_design_matrix(n_time, event_samples, template, intercept=True)** — one column per event, template placed at onset. For jointly fitting amplitude of each occurrence.

### 5. Spectral analysis

```python
from cogpy.spectral.psd import psd_multitaper, psd_welch
from cogpy.spectral.specx import psdx, spectrogramx, normalize_spectrogram
from cogpy.spectral.features import (
    band_power,            # integrate PSD over (fmin, fmax) → scalar power
    relative_band_power,   # band_power / total_power
    spectral_peak_freqs,   # find peaks via scipy.signal.find_peaks
    ftest_line_scan,       # Thomson F-test: sinusoidal line vs broadband at each freq
    line_noise_ratio,      # power at f_line / flanking power
    narrowband_ratio,      # per-freq-bin ratio to flanking median
    am_artifact_score,     # sideband / background for AM artifacts
)
from cogpy.spectral.bivariate import coherence, cross_corr_lag
from cogpy.spectral.multitaper import multitaper_fft
```

**ftest_line_scan(signal, fs, NW=4.0, p_threshold=0.05)** → `(fstat, freqs, sig_mask)`. Identifies statistically significant narrowband lines. Essential for finding the 120 Hz fundamental and harmonics.

**narrowband_ratio(psd, freqs, flank_hz=5.0)** → per-bin ratio to flanking median. Values >> 1 indicate narrowband contamination.

### 6. Temporal filtering

```python
from cogpy.preprocess.filtering.temporal import (
    bandpassx,    # Butterworth bandpass (zero-phase SOS)
    highpassx,    # Butterworth highpass
    lowpassx,     # Butterworth lowpass
    notchx,       # single IIR notch (w0, Q)
    notchesx,     # multiple notches at specified frequencies
)
from cogpy.preprocess.filtering.reference import cmrx       # common median reference
from cogpy.preprocess.filtering.normalization import zscorex # z-score along dim
```

### 7. Spatial analysis

```python
from cogpy.measures.spatial import (
    moran_i,                   # spatial autocorrelation (queen/rook/ap_only/ml_only adjacency)
    gradient_anisotropy,       # log2(mean|dV/dAP| / mean|dV/dML|) — stripe direction
    marginal_energy_outlier,   # row/col energy z-scores + outlier flags
    spatial_kurtosis,          # concentration of energy
    spatial_noise_concentration,  # fraction of energy in top-k electrodes
    csd_power,                 # current source density via 2D Laplacian
    spatial_summary_xr,        # batch compute multiple spatial measures → xr.Dataset
)
from cogpy.preprocess.filtering.spatial import (
    gaussian_spatialx,   # spatial Gaussian smoothing
    median_subtractx,    # subtract spatial median
    median_highpassx,    # spatiotemporal median highpass
)
from cogpy.decomposition.pca import erpPCA  # varimax-rotated PCA (fit/transform)
```

### 8. Validation & comparison metrics

```python
from cogpy.measures.comparison import (
    snr_improvement,        # SNR_after - SNR_before (dB), needs signal_band + noise_band
    residual_energy_ratio,  # sum((orig-clean)^2) / sum(orig^2) per channel
    bandpower_change,       # (P_after - P_before) / P_before in a frequency band
    waveform_residual_rms,  # RMS of difference between two waveforms (e.g. triggered averages)
)
```

### 9. Plotting — HoloViews static primitives

All functions below live in `cogpy.plot.hv.signals` and return **static** HoloViews elements (Curve, Image, Overlay, Layout, HoloMap) — never DynamicMap. They render correctly in both live notebooks and static HTML export.

```python
import holoviews as hv
hv.extension("bokeh")

from cogpy.plot.hv.signals import (
    # --- Spectral ---
    psd_curve,            # single PSD as hv.Curve (logx/logy)
    psd_overlay,          # overlay multiple PSDs: {"before": (psd, f), "after": (psd, f)}
    psd_with_lines,       # PSD + vertical lines at detected peak frequencies

    # --- Spatial maps ---
    spatial_heatmap,      # (AP, ML) scalar field as hv.Image — artifact power, metrics, etc.
    spatial_heatmap_grid, # dict of {title: 2d_array} → hv.Layout grid of heatmaps

    # --- Triggered waveforms ---
    triggered_waveform,       # mean ± std band for one channel → hv.Overlay
    triggered_waveform_grid,  # (AP, ML, lag) templates → hv.Layout of waveforms in grid position

    # --- Histograms ---
    event_histogram,      # generic histogram (inter-event intervals, amplitudes, etc.)
    lag_histogram,         # cross-correlogram from event_lag_histogram output

    # --- Decomposition ---
    factor_loading_grid,  # (factor, AP, ML, freq) loadings → hv.Layout of spatial maps at peak freq

    # --- Drift / alignment ---
    drift_plot,           # scatter of matched lags vs time + polynomial fit curve

    # --- Time series ---
    signal_trace,            # single 1D signal as hv.Curve
    signal_traces_overlay,   # overlay multiple traces: {"raw": (t, y), "cleaned": (t, y)}
)
```

**Key functions explained:**

**`psd_overlay({"before": (psd, freqs), "after": (psd, freqs)})`** — before/after PSD comparison with legend. Use `logx=True, logy=True` (defaults).

**`psd_with_lines(psd, freqs, line_freqs)`** — PSD curve with red dashed verticals at detected lines. Feed `line_freqs` from `freqs[sig_mask]` after `ftest_line_scan`.

**`spatial_heatmap(data_2d, title="120 Hz power", cmap="viridis")`** — one (AP, ML) heatmap. Use `symmetric=True, cmap="RdBu_r"` for signed quantities (gradient anisotropy, bandpower change).

**`spatial_heatmap_grid({"120 Hz": map1, "240 Hz": map2, "Moran I": map3})`** — auto-layout multiple spatial maps in a grid.

**`triggered_waveform(template, std, lag_axis=lag_s)`** — mean waveform with shaded ±1 std band.

**`triggered_waveform_grid(templates_xr, stds_xr)`** — expects `xr.DataArray` with dims `(AP, ML, lag)`. Produces a grid of small waveform plots, one per electrode, laid out by grid position. Uses `hv.HoloMap` → static-safe.

**`factor_loading_grid(loadings_xr)`** — expects `xr.DataArray` with dims `(factor, AP, ML, freq)`. For each factor, finds the frequency of max absolute loading and shows the spatial map at that frequency. Produces `hv.Layout`.

**`drift_plot(matched_times, matched_lags, coeffs)`** — scatter of per-event lags over time with polynomial fit overlay. Shows clock drift visually.

**`signal_traces_overlay({"raw": (t, y), "cleaned": (t, y)})`** — overlay two (or more) time series with legend for before/after comparison.

#### Interactive plots (live notebook only)

For interactive exploration in a live Jupyter session, `cogpy.plot.hv.xarray_hv` provides DynamicMap-based viewers. These will NOT render in static HTML export but are useful during exploration:

```python
from cogpy.plot.hv.xarray_hv import (
    multichannel_view,     # stacked multichannel traces with minimap & RangeTool
    grid_movie,            # AP×ML movie scrubbed over time — DynamicMap
    grid_movie_with_time_curve,  # spatial frame + time curve with linked time hair
)
```

**Static export rule:** Use `signals.*` functions for all plots that must appear in exported HTML. Use `xarray_hv.*` functions only for live interactive exploration.

---

## Notebook narrative

Structure the notebook as a scientific investigation, not a pipeline. Each section asks a question, uses cogpy primitives to answer it, and interprets the result before deciding what to do next.

### Section 0 — Setup & data loading
- Load the grid ECoG signal and TTL reference channel
- Print shape, fs, duration, grid dimensions
- **Plot:** `signal_trace(ttl_ref, title="TTL reference")` — raw TTL channel
- **Plot:** `signal_traces_overlay({"ch (0,0)": (t, sig[0,0,:]), "TTL": (t, ttl_scaled)})` — a few ECoG channels + TTL overlay
- For interactive browsing: `multichannel_view(sig)` (live notebook only)

### Section 1 — Spectral fingerprint of the artifact
**Question:** What does the contamination look like in the frequency domain?

- Compute PSD (multitaper) of a representative channel
- Run `ftest_line_scan` to identify statistically significant narrowband lines
- Compute `narrowband_ratio` and `line_noise_ratio` at 120 Hz and harmonics
- Compute `am_artifact_score` if sidebands are suspected
- **Plot:** `psd_with_lines(psd, freqs, freqs[sig_mask], title="PSD + detected lines")` — PSD annotated with significant narrowband lines
- **Plot:** `psd_curve(narrowband_ratio_arr, freqs, logy=False, title="Narrowband ratio")` — per-frequency peak-to-background ratio (>>1 = contaminated)
- **Conclusion:** Identify the spectral signature — is it narrowband (lines only) or broadband (event-like transients)?

### Section 2 — Spatial structure of the artifact
**Question:** How does the artifact vary across the electrode grid?

- Compute per-channel PSD, extract `band_power` around 120 Hz
- Compute `moran_i`, `gradient_anisotropy`, `marginal_energy_outlier`
- **Plot:** `spatial_heatmap(power_120hz, title="120 Hz band power", cmap="inferno")` — artifact power map
- **Plot:** `spatial_heatmap_grid({"120 Hz": p120, "240 Hz": p240, "360 Hz": p360, "gradient_aniso": aniso_map})` — multi-metric spatial overview
- **Plot:** `event_histogram(row_energies, title="Row marginal energy", xlabel="Energy")` — distribution of per-row energy
- **Conclusion:** Is the artifact spatially uniform, graded, or striped?

### Section 3 — Event detection in the TTL reference
**Question:** Can we detect individual TTL edges?

- Use `ThresholdDetector` on the TTL reference channel
- Compute inter-event intervals (IEIs)
- **Plot:** `event_histogram(ieis * 1000, bins=100, xlabel="IEI (ms)", title="Inter-event interval distribution")` — should peak at ~8.33 ms
- **Plot:** `signal_trace(ttl_ref, title="TTL with detections")` overlaid with `hv.Spikes(event_times)` — verify detection visually on a short window
- Check for missing/extra events; store event times

### Section 4 — Event matching and clock alignment
**Question:** Are the TTL events aligned with the neural acquisition clock?

- Detect artifact events directly in a high-artifact ECoG channel (e.g. via envelope thresholding)
- Use `match_nearest` to pair TTL events with ECoG artifact events
- Compute `estimate_lag` and `estimate_drift`
- **Plot:** `lag_histogram(counts, edges, title="TTL ↔ ECoG cross-correlogram")` — from `event_lag_histogram` output
- **Plot:** `drift_plot(matched_times, matched_lags, coeffs, title="Clock drift: TTL vs ECoG")` — scatter + polynomial fit
- If drift exists, correct event times with `np.polyval(coeffs, event_times)`
- **Conclusion:** What is the timing relationship? Constant lag, linear drift, or jitter?

### Section 5 — Artifact waveform characterization
**Question:** What does a single artifact event look like, and how does it vary?

- Use `perievent_epochs` to extract windows around TTL events (e.g. ±5 ms)
- Compute `triggered_average`, `triggered_std`, `triggered_snr` for each channel
- **Plot:** `triggered_waveform(avg_ch, std_ch, lag_axis=lag_s, title="Template: ch (3,5)")` — single-channel template with ±std band
- **Plot:** `triggered_waveform_grid(templates_xr, stds_xr)` — full electrode grid of waveforms, one per (AP, ML) position. This is the key plot for seeing how the artifact varies spatially.
- **Plot:** `spatial_heatmap(snr_map, title="Triggered SNR", cmap="magma")` — which channels have the most consistent artifact?
- **Plot:** `spatial_heatmap_grid({"peak amplitude": amp_map, "peak latency": lat_map, "SNR": snr_map})` — summary spatial maps of template properties
- **Conclusion:** Is the artifact stereotyped enough for template subtraction?

### Section 6 — Removal approach 1: Template subtraction
**Question:** Can we remove the artifact by subtracting a per-channel template?

- Use `estimate_template(epochs, method="median")` per channel
- Use `fit_scaling(epochs, template)` to get per-event amplitudes
- Use `subtract_template(signal, event_samples, template, scaling=alpha)` per channel
- **Plot:** `psd_overlay({"before": (psd_pre, f), "after": (psd_post, f)}, title="PSD: template subtraction")` — before/after PSD comparison
- **Plot:** `triggered_waveform(avg_after, std_after, lag_axis=lag_s, title="Residual ETA after template sub")` — should be flat if artifact is removed
- **Plot:** `signal_traces_overlay({"raw": (t, raw_ch), "cleaned": (t, clean_ch)}, title="Time domain: template sub")` — before/after time series for a representative channel
- Compute `bandpower_change`, `waveform_residual_rms`
- **Conclusion:** How much artifact was removed? Is there signal distortion?

### Section 7 — Removal approach 2: Regression
**Question:** Can lagged regression against the TTL reference do better?

- Use `lagged_design_matrix(ttl_ref, lags=range(n_lags))` to model the channel-specific impulse response
- Use `ols_fit`, `ols_residual` per channel to subtract the predicted artifact
- **Plot:** `psd_overlay({"before": (psd_pre, f), "template": (psd_tmpl, f), "regression": (psd_reg, f)}, title="PSD: all methods")` — three-way comparison
- **Plot:** `signal_traces_overlay({"raw": (t, y_raw), "template": (t, y_tmpl), "regression": (t, y_reg)})` — time domain three-way
- **Plot:** `triggered_waveform(avg_reg, std_reg, lag_axis=lag_s, title="Residual ETA after regression")` — should be flat
- Compare: `snr_improvement`, `residual_energy_ratio`
- **Conclusion:** Which approach gives cleaner removal with less signal distortion?

### Section 8 — Validation summary
**Question:** Did we actually improve the signal?

- **Plot:** `psd_overlay({"raw": ..., "template sub": ..., "regression": ...}, title="Final PSD comparison")` — comprehensive PSD overlay
- **Plot:** `spatial_heatmap_grid({"Δpower 120Hz (template)": bp_tmpl, "Δpower 120Hz (regr)": bp_reg, "Δpower 1-30Hz (template)": bp_neural_tmpl, "Δpower 1-30Hz (regr)": bp_neural_reg}, symmetric=True)` — spatial maps of bandpower change: artifact band (should be negative) and neural band (should be ~0)
- **Plot:** `spatial_heatmap(energy_ratio_map, title="Residual energy ratio", cmap="magma")` — uniformity check
- Print summary table of metrics per method (use pandas DataFrame)
- **Conclusion:** Summary table of metrics per method

### Section 9 — Optional: spatial decomposition of the artifact
**Question:** Can PCA/SVD reveal the artifact's spatial structure?

- Stack epochs into (events × channels × lag), reshape to (events, channels*lag)
- Apply `erpPCA` or direct SVD to extract spatial modes
- **Plot:** `factor_loading_grid(loadings_xr)` — spatial loading maps at peak frequency for each factor. Shows whether the artifact loads on a small number of spatial modes (low-rank structure).
- **Plot:** `psd_overlay` of factor score time series — spectral content of each component
- Could inform a spatial filter approach (project out artifact subspace)

---

## Implementation notes

- Initialize holoviews once at the top: `import holoviews as hv; hv.extension("bokeh")`
- **Static-safe plots:** All `cogpy.plot.hv.signals.*` functions return static HoloViews elements. They render in both live notebooks and `nbconvert` / static HTML export. Use these for all plots that must appear in exported reports.
- **Interactive plots:** `cogpy.plot.hv.xarray_hv.*` functions (e.g. `multichannel_view`, `grid_movie`) return DynamicMap objects. These work only in live Jupyter sessions. Use them for interactive browsing but do NOT rely on them for exported output.
- **HoloMap vs DynamicMap:** `triggered_waveform_grid` uses `hv.HoloMap` internally, which is static-safe and exports to HTML. `grid_movie` uses `hv.DynamicMap`, which requires a live kernel.
- Use `%%time` or `tqdm` for long operations
- Keep each cell focused on one question + one answer
- Print intermediate shapes and sanity checks (e.g. expected number of TTL events at 120 Hz)
- All cogpy functions are pure — no hidden state. Cells can be re-run in any order after data is loaded.
- For large data, work on a representative time window first (e.g. 10 seconds), then scale up
