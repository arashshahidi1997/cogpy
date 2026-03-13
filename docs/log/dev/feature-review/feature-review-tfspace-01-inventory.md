# CogPy TF-Space Inventory

## Goal

This inventory assesses whether CogPy already contains the primitives needed for a time-frequency-space noise QC workflow for ECoG/iEEG, with emphasis on concrete code evidence in `core`, `datasets`, and existing review/docs areas rather than proposed API design.

## Existing spectral/time-frequency primitives

| Status | Module path | Symbol | Description | Notes |
|---|---|---|---|---|
| EXISTS | `src/cogpy/core/spectral/specx.py` | `spectrogramx` | Xarray multitaper spectrogram wrapper. | Returns `xr.DataArray` with non-time dims preserved plus `("freq","time")`; suitable base for dense TF maps. |
| EXISTS | `src/cogpy/core/spectral/specx.py` | `psdx` | Xarray PSD wrapper. | Returns `(..., freq)` from `xr.DataArray`; supports multitaper and Welch. |
| EXISTS | `src/cogpy/core/spectral/multitaper.py` | `mtm_spectrogram` | Multitaper spectrogram primitive using Ghostipy backend. | Lower-level NumPy/Dask TF primitive under `spectrogramx`. |
| EXISTS | `src/cogpy/core/spectral/multitaper.py` | `mtm_spectrogramx` | Xarray multitaper spectrogram wrapper. | Separate xarray interface exists in addition to `spectrogramx`. |
| PARTIAL | `src/cogpy/core/spectral/multitaper.py` | `multitaper_spectrogram` | Deprecated windowed spectrogram helper. | Evidence of older rolling-window TF path; docstring says to prefer newer sliding-core pattern. |
| EXISTS | `src/cogpy/core/detect/transforms/spectral.py` | `SpectrogramTransform` | Detection-pipeline transform wrapper around `spectrogramx`. | Confirms spectrogram is already part of reusable pipeline composition. |
| EXISTS | `src/cogpy/core/detect/burst.py` | `BurstDetector` | Detector that accepts spectrogram-like inputs or computes spectrogram implicitly. | `get_event_dims()` returns `["time", "freq", "AP", "ML"]`, which is direct TF-space evidence. |
| RELATED | `src/cogpy/core/spectral/spectrogram_burst.py` | `compute_multitaper_spectrogram` | Higher-level multitaper spectrogram workflow. | Returns dataset with `spec(channel, time, freq)` plus configurable normalization, but module is pipeline-specific rather than core-generic. |
| EXISTS | `src/cogpy/core/utils/sliding_core.py` | `sliding_window`, `running_reduce`, `running_blockwise`, `running_blockwise_xr` | NumPy/xarray sliding-window infrastructure. | General windowing substrate for time-windowed transforms or derived measures. |
| EXISTS | `src/cogpy/core/utils/sliding.py` | `rolling_win`, `running_measure`, `running_measure_sane`, `rolling_win_sane` | Xarray-aware rolling/windowed application helpers. | Evidence that windowed feature extraction is a first-class composition pattern. |
| PARTIAL | `src/cogpy/datasets/schemas.py` | `DIMS_GRID_WINDOWED_SPECTRUM`, `DIMS_CHANNEL_WINDOWED_SPECTRUM`, `DIMS_SPECTROGRAM4D` | Canonical dim constants for TF outputs. | Dim conventions exist, but validators/coercers for windowed spectra are not yet implemented beyond `validate_spectrogram4d()`. |
| EXISTS | `src/cogpy/datasets/schemas.py` | `validate_spectrogram4d`, `coerce_spectrogram4d` | Validator/coercer for grid spectrogram tensors. | Canonical order is `("ml","ap","time","freq")`; useful but not aligned with all uppercase AP/ML conventions elsewhere. |

## Existing spectral-normalization / whitening primitives

| Status | Module path | Symbol | Description | Notes |
|---|---|---|---|---|
| EXISTS | `src/cogpy/core/spectral/whitening.py` | `AR_whiten`, `AR_whitening` | Autoregressive whitening filter implementation. | Direct whitening primitive for time series; not spectrogram-specific, but relevant for upstream whitening. |
| EXISTS | `src/cogpy/core/spectral/features.py` | `narrowband_ratio` | Per-frequency-bin PSD prominence against flanking-bin median. | Direct evidence for frequency-local normalization/peak contrast on `(..., freq)`. |
| EXISTS | `src/cogpy/core/spectral/features.py` | `line_noise_ratio` | Ratio of line-band power to flank-band power. | Narrowband-vs-baseline primitive at known line frequency. |
| EXISTS | `src/cogpy/core/spectral/features.py` | `spectral_peak_freqs` | Peak picker over PSD. | Supportive for identifying narrow peaks after ratio/whitening steps. |
| EXISTS | `src/cogpy/core/spectral/features.py` | `ftest_line_scan` | Thomson F-test scan across frequency bins. | Strong evidence for narrowband line detection on 1D windows; currently 1D signal only. |
| EXISTS | `src/cogpy/core/spectral/features.py` | `aperiodic_exponent`, `fooof_periodic` | Aperiodic fit and periodic-above-background extraction. | Useful for spectral background normalization or pseudo-whitened summaries. |
| EXISTS | `src/cogpy/core/preprocess/filtering/normalization.py` | `zscorex` | Z-score / robust z-score along any xarray dimension. | Can normalize along `freq` if caller wraps spectra as xarray and chooses `dim="freq"`. |
| EXISTS | `src/cogpy/core/preprocess/badchannel/feature_normalization.py` | `normalize_windowed_features` | Robust or standard z-score normalization over a window dimension. | More general window-axis normalization for feature datasets. |
| EXISTS | `src/cogpy/core/preprocess/channel_feature_functions.py` | `local_robust_zscore`, `local_robust_zscore_dask` | Neighborhood-based robust z-scoring. | Spatial/local normalization primitives for feature maps. |
| EXISTS | `src/cogpy/core/preprocess/badchannel/spatial.py` | `normalize_ratio`, `normalize_difference`, `normalize_robust_z`, `local_robust_zscore_grid` | Spatial neighborhood normalization helpers. | Evidence for local baseline logic on 2D APxML maps. |
| RELATED | `src/cogpy/core/spectral/spectrogram_burst.py` | `normalization` config (`db`, `zscore`, `raw`) | Built-in spectrogram normalization modes in pipeline-specific workflow. | Confirms existing TF workflows already perform normalization, but not as a reusable standalone API. |
| EXISTS | `src/cogpy/core/preprocess/linenoise.py` | `LineNoiseEstimatorMultitaper`, `LineNoiseEstimatorICA` | Line-noise scoring/removal classes. | Includes explicit baseline logic around line harmonics; more preprocessing-oriented than generic whitening. |
| MISSING | `src/cogpy/core/spectral/` | Direct `wspec(...)`-style constructor | Explicit reusable builder for whitened / frequency-normalized spectrogram tensors. | Search found components for constructing this, but no single TF-normalization function returning a canonical whitened spectrogram object. |

## Existing spatial-slice measures

| Status | Module path | Symbol | Description | Notes |
|---|---|---|---|---|
| EXISTS | `src/cogpy/core/measures/spatial.py` | `moran_i` | Moran's I on `(..., AP, ML)` scalar grids. | Supports `queen`, `rook`, `ap_only`, `ml_only`; direct match for `moran_ap` / `moran_ml` via directional adjacency. |
| EXISTS | `src/cogpy/core/measures/spatial.py` | `gradient_anisotropy` | Directional gradient imbalance on `(..., AP, ML)`. | Direct match to desired per-slice anisotropy metric. |
| EXISTS | `src/cogpy/core/measures/spatial.py` | `marginal_energy_outlier` | Row/column energy profiles, z-scores, outlier flags. | Direct evidence for row/column outlier summaries per spatial slice. |
| EXISTS | `src/cogpy/core/measures/spatial.py` | `spatial_kurtosis` | Excess kurtosis of flattened APxML amplitudes. | Direct match. |
| EXISTS | `src/cogpy/core/measures/spatial.py` | `spatial_noise_concentration` | Fraction of total energy in top-k electrodes. | Direct match. |
| EXISTS | `src/cogpy/core/measures/spatial.py` | `spatial_coherence_profile` | Distance-binned coherence profile over a grid. | Adjacent spatial-structure metric; not one of the requested slice scalars, but relevant evidence of grid-aware spatial analysis. |
| RELATED | `src/cogpy/core/measures/spatial.py` | `csd_power` | 2D Laplacian/CSD transform over `(AP, ML, time)`. | Spatial derivative primitive, useful precursor for some spatial structure analyses. |
| EXISTS | `src/cogpy/core/preprocess/badchannel/spatial.py` | `neighbors_from_adjacency`, `neighborhood_median`, `neighborhood_mad` | Neighborhood statistics from grid adjacency. | Evidence of reusable local-spatial support for batched or normalized slice analysis. |
| EXISTS | `src/cogpy/core/preprocess/badchannel/grid.py` | `grid_adjacency`, `grid_edges`, `make_footprint` | Grid topology helpers for APxML layouts. | Necessary support for batched spatial-slice measures and neighborhood normalization. |
| RELATED | `src/cogpy/core/preprocess/badchannel/spatial.py` | `anticorrelation` | Median inverse neighbor correlation map. | Spatial artifact feature, but not the same as Moran's I. |
| RELATED | `src/cogpy/core/preprocess/channel_feature_functions.py` | `laplacian`, `spatial_gradient`, `gradient`, `gradient_fast` | Legacy spatial feature computations on `(AP, ML, time)`. | Additional evidence that APxML spatial derivatives already exist in preprocessing code. |
| EXISTS | `src/cogpy/core/measures/spatial.py` | batched `(..., AP, ML)` convention | Spatial measures accept arbitrary leading batch dims. | This directly supports applying metrics on each `(time_win, freq)` slice after a transpose. |

## Existing temporal/event helpers

| Status | Module path | Symbol | Description | Notes |
|---|---|---|---|---|
| EXISTS | `src/cogpy/core/detect/utils.py` | `score_to_bouts` | Convert a 1D score series to bouts via dual threshold, gap merge, and min duration. | Direct match to requested temporal/event primitive. |
| EXISTS | `src/cogpy/core/detect/utils.py` | `dual_threshold_events_1d` | Dual-threshold event extraction on a 1D score. | Direct lower-level helper. |
| EXISTS | `src/cogpy/core/detect/utils.py` | `merge_intervals` | Merge nearby intervals in sample space. | Direct lower-level helper. |
| EXISTS | `src/cogpy/core/detect/threshold.py` | `ThresholdDetector` | Generic threshold-crossing interval detector. | Already produces interval event tables with `duration` and `label`. |
| EXISTS | `src/cogpy/core/detect/ripple.py` | `RippleDetector`, `SpindleDetector` | Event detectors with duration filtering and event catalogs. | Evidence that event extraction conventions are already established. |
| EXISTS | `src/cogpy/datasets/schemas.py` | `EventCatalog`, `validate_event_catalog`, `coerce_event_catalog` | Canonical event-table contract plus validation/coercion. | Strong evidence for downstream TF-score-to-event summaries. |
| EXISTS | `src/cogpy/datasets/schemas.py` | `Intervals`, `Events` | Lightweight interval and point-event containers. | Include `total_duration()`, `to_dataframe()`, `to_intervals()`, and `restrict()`. |
| RELATED | `src/cogpy/core/preprocess/badchannel/feature_normalization.py` | `smooth_windowed_features`, `summarize_windowed_features` | Smooth or summarize windowed feature datasets. | Useful for temporal aggregation after TF/slice metrics are computed, but not event-specific. |
| RELATED | `src/cogpy/core/measures/temporal.py` | scalar temporal measures (`temporal_stability`, `mobility`, `complexity`, `jump_index`, etc.) | General temporal characterization functions. | These are signal/window measures, not occupancy/bout-summary utilities. |
| MISSING | `src/cogpy/core/detect/` / `src/cogpy/core/measures/` | Direct occupancy helper | No dedicated helper found to compute fraction-of-time occupancy from event catalogs or interval sets. | Can likely be composed from `Intervals.total_duration()` and recording duration, but no named primitive was found. |
| MISSING | `src/cogpy/core/detect/` / `src/cogpy/core/measures/` | Direct bout-duration distribution helper | No dedicated summary helper found for event duration histograms/distributions. | Durations exist in `EventCatalog` tables, but summarization is left to caller code. |
| MISSING | `src/cogpy/core/detect/` / `src/cogpy/core/measures/` | Direct frequency-distribution / joint spectral-spatial abnormality summary helper | No concrete helper found for summarizing event counts or occupancy over freq bands jointly with spatial abnormality scores. | Current code supports composing this, but no direct primitive is present. |

## Existing schema / validation helpers

| Status | Module path | Symbol | Description | Notes |
|---|---|---|---|---|
| EXISTS | `docs/source/explanation/architecture.md` | architecture guidance | Repository-level architecture explicitly treats `cogpy.core` as reusable in-memory compute primitives and stresses xarray-centered named dims. | This supports TF-space composition expectations without implying workflow ownership. |
| EXISTS | `src/cogpy/datasets/schemas.py` | `DIMS_IEEG_GRID`, `validate_ieeg_grid`, `coerce_ieeg_grid` | Canonical grid time-series schema. | Canonical order is `("time","ML","AP")`. |
| EXISTS | `src/cogpy/datasets/schemas.py` | `DIMS_IEEG_GRID_WINDOW`, `validate_ieeg_grid_windowed`, `coerce_ieeg_grid_windowed` | Canonical windowed grid schema. | Direct support for `(time_win, ML, AP)` feature maps. |
| EXISTS | `src/cogpy/datasets/schemas.py` | `DIMS_GRID_SPECTRUM`, `DIMS_GRID_WINDOWED_SPECTRUM`, `DIMS_CHANNEL_SPECTRUM`, `DIMS_CHANNEL_WINDOWED_SPECTRUM` | Canonical spectrum dim constants. | Important evidence that spectrum/windowed-spectrum schemas are already anticipated. |
| PARTIAL | `src/cogpy/datasets/schemas.py` | spectrum dim constants without validators | Windowed/grid spectral schemas are defined but lack `validate_*` / `coerce_*` implementations. | This is schema-level readiness, not full schema enforcement. |
| EXISTS | `src/cogpy/datasets/schemas.py` | `validate_spectrogram4d`, `coerce_spectrogram4d` | Spectrogram validator/coercer. | Canonical form is lower-case `("ml","ap","time","freq")`; may require boundary renaming from AP/ML code paths. |
| EXISTS | `src/cogpy/datasets/schemas.py` | `validate_multichannel_windowed`, `coerce_multichannel_windowed` | Windowed non-grid schema helpers. | Useful if some TF/event summaries are flattened to channel representations. |
| EXISTS | `src/cogpy/datasets/schemas.py` | `validate_ieeg_time_channel`, `coerce_ieeg_time_channel` | Canonical stacked-view schema with optional AP/ML coords on channel. | Useful boundary form for event catalogs or per-channel summaries. |
| EXISTS | `docs/source/explanation/datasets/schemas.md` and `docs/log/dev/feature-review/feature-review-02-schema.md` | prior schema review evidence | Existing review/docs already recognize canonical dim constants and partial validator coverage. | Confirms schema work is underway rather than absent. |

## Key gaps observed

- The repository already has the core computational pieces for TF-space QC: spectrogram generation, sliding-window infrastructure, PSD/spectral contrast measures, grid spatial measures on `(..., AP, ML)`, and score-to-bout event extraction.
- The strongest missing piece is not raw computation but a canonical reusable composition layer for `spec(time_win, freq, AP, ML)` and especially normalized/whitened `wspec(...)` tensors; today this must be assembled from existing primitives.
- Schema readiness is partial: `DIMS_GRID_WINDOWED_SPECTRUM` and related constants exist, but dedicated `validate_*` / `coerce_*` helpers for grid/windowed spectra do not.
- Spatial-slice measures are in better shape than expected: `gradient_anisotropy`, directional `moran_i`, `marginal_energy_outlier`, `spatial_kurtosis`, and `spatial_noise_concentration` all already exist and already accept leading batch dims.
- Temporal/event summarization is only partly productized: `score_to_bouts`, interval merging, and validated event catalogs exist, but occupancy, duration distributions, frequency distributions, and joint spectral-spatial summary helpers are not present as named reusable primitives.
