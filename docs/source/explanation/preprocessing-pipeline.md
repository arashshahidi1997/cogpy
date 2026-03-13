# Preprocessing Pipeline

This page explains the design of cogpy's preprocessing pipeline: how bad
channels are identified, why spatial normalization matters, and how the
Snakemake workflow ties it all together.

## Pipeline overview

The preprocessing pipeline converts raw ECoG recordings into clean,
analysis-ready signals. It runs as a Snakemake DAG with seven stages:

```
raw → lowpass → downsample → feature → badlabel → plot → interpolate
```

| Stage | Purpose | Module |
|-------|---------|--------|
| `raw_zarr` | Convert binary LFP + XML metadata to Zarr | `cogpy.io.ecog_io` |
| `lowpass` | Anti-alias filter before downsampling | `cogpy.preprocess.filtering` |
| `downsample` | Decimate to target sampling rate | `cogpy.preprocess.filtering` |
| `feature` | Extract channel quality features | `cogpy.preprocess.badchannel` |
| `badlabel` | Label bad channels via DBSCAN | `cogpy.preprocess.badchannel` |
| `plot_feature_maps` | QC visualizations | Matplotlib |
| `interpolate` | Spatially interpolate bad channels | `cogpy.preprocess.interpolate` |

Each stage reads from Zarr and writes to Zarr, making the pipeline
restartable and inspectable at any point.

## Filtering

cogpy provides xarray-aware filters in `cogpy.preprocess.filtering`:

| Function | Type | Use case |
|----------|------|----------|
| `bandpassx()` | Butterworth IIR | General frequency selection |
| `lowpassx()` | Butterworth IIR | Anti-aliasing before decimation |
| `highpassx()` | Butterworth IIR | DC removal |
| `notchx()` / `notchesx()` | Notch IIR | Line noise removal (50/60 Hz) |
| `decimatex()` | Polyphase | Downsampling with anti-alias |
| `cmrx()` | Spatial | Common-mode rejection (subtract channel median) |
| `gaussian_spatialx()` | Spatial | Gaussian smoothing across grid |
| `median_spatialx()` | Spatial | Median filtering across grid |

All filters use `xr.apply_ufunc` internally, preserving dimensions,
coordinates, and attributes. They are also dask-compatible for lazy
evaluation on large recordings.

**Why Butterworth IIR?** For online-style causal filtering with `filtfilt`
(zero-phase), Butterworth provides flat passband response with predictable
roll-off. The pipeline uses order-4 filters by default.

## Bad-channel detection

The bad-channel pipeline is the most complex preprocessing component. It
uses a four-stage approach designed to be robust to the spatial structure
of electrode grids.

### Stage 1: Feature extraction

`compute_features_sliding()` extracts seven channel-quality features in
sliding time windows:

| Feature | What it measures |
|---------|-----------------|
| `anticorrelation` | Spatial correlation with grid neighbors |
| `relative_variance` | Variance relative to neighbors |
| `deviation` | Amplitude deviation from neighbors |
| `amplitude` | Peak signal amplitude |
| `time_derivative` | Rate of signal change |
| `hurst_exponent` | Long-range temporal correlation |
| `kurtosis` | Distribution tail heaviness |

Output shape: `(n_features, AP, ML, n_windows)`.

### Stage 2: Spatial normalization

Raw features are not directly comparable across the grid — a channel near
the edge naturally has different variance than one in the center. Spatial
normalization corrects for this by comparing each channel to its grid
neighbors.

Four normalization modes:

| Mode | Formula | Used by |
|------|---------|---------|
| `identity` | Use raw value | `anticorrelation` |
| `ratio` | `x / (median_neighbor + ε)` | `relative_variance`, `amplitude`, `time_derivative` |
| `difference` | `x - median_neighbor` | `deviation` |
| `robust_z` | `(x - median) / (MAD × 1.4826 + ε)` | `kurtosis` |

Neighbor relationships are derived from the 2D grid layout using binary
dilation footprints (default: 2-iteration, connectivity-1).

**Why spatial normalization?** Without it, edge channels and channels near
sulci would be systematically flagged as bad. The normalization isolates
*locally anomalous* channels — those that differ from their immediate
spatial context.

### Stage 3: DBSCAN outlier labeling

After normalization, features are aggregated across time windows using
quantiles (75th–95th percentile, 5 levels), then stacked into a
`(n_channels, n_feature_quantile_combos)` matrix.

DBSCAN identifies outliers:

1. **StandardScaler** normalizes the feature matrix
2. **k-distance curve** estimates optimal `eps` via knee detection
   (`KneeLocator` on k=10 nearest-neighbor distances)
3. **DBSCAN** clusters channels; noise points (`label=-1`) are bad

**Why DBSCAN over threshold-based methods?** DBSCAN detects outliers in
the joint feature space without assuming any single feature is sufficient.
A channel might have acceptable variance but anomalous kurtosis — DBSCAN
catches multivariate outliers that per-feature thresholds miss.

**Why automatic eps?** Manual eps tuning is fragile across recordings with
different noise floors. The k-distance knee provides a data-driven estimate
that adapts to each recording's feature distribution.

### Stage 4: Interpolation

Bad channels are replaced by spatial interpolation from their grid
neighbors. This preserves the spatial sampling for downstream analyses
(CSD computation, spatial measures) that require a complete grid.

## Snakemake orchestration

The pipeline is packaged as a Snakemake workflow in
`cogpy.workflows.preprocess`. Key design choices:

- **Rules are thin orchestrators.** Each rule loads data via `cogpy.io`,
  calls `cogpy.core` functions, and saves via `cogpy.io`. No compute
  logic lives in the Snakefile.

- **Zarr as interchange format.** Every stage reads and writes Zarr,
  providing chunked storage, metadata preservation, and restartability.

- **Config-driven parameters.** All hyperparameters (filter cutoffs,
  window sizes, DBSCAN settings) live in YAML config files, not in code.

- **Dask chunking.** Scripts chunk along the time axis for memory-efficient
  processing of long recordings:
  `sigx.chunk({'time': 16*4096, 'AP': -1, 'ML': -1})`

## CLI entry point

The `cogpy-preproc` command wraps Snakemake with sensible defaults:

```bash
cogpy-preproc all data/sub-01/rec1.lfp -c 8
cogpy-preproc feature data/sub-01/rec1.lfp --configfile custom.yml
```

It loads the packaged default config, merges any user overrides, resolves
the packaged Snakefile path, and spawns a Snakemake subprocess targeting
the requested rule.

## Legacy modules

The canonical bad-channel pipeline lives in `cogpy.core.preprocess.badchannel`.
Three older modules are retained for backward compatibility but are deprecated:

| Legacy module | Replacement |
|---------------|-------------|
| `channel_feature.py` | `badchannel.channel_features` |
| `channel_feature_functions.py` | `badchannel.channel_features` |
| `detect_bads.py` | `badchannel.badlabel` |

New code should always use the `badchannel` subpackage.
