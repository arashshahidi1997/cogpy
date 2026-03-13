# CogPy Feature Inventory

## Context
This inventory was produced by scanning the CogPy codebase for concrete implementations and closely related helpers, guided by the conceptual registry at `docs/reference/feature-registry`.

Statuses used here:

| Status | Meaning |
|---|---|
| `EXISTS` | A direct implementation matching the conceptual feature is present. |
| `PARTIAL` | A close but narrower or differently scoped implementation is present. |
| `RELATED` | A supporting helper, adjacent implementation, or pipeline hook exists, but it is not a direct match for the conceptual feature. |

## Spectral features

| Concept | Status | Module path | Symbol name | Type | Short description | Notes |
|---|---|---|---|---|---|---|
| narrowband ratio | `EXISTS` | `src/cogpy/core/spectral/features.py` | `narrowband_ratio` | function | Per-frequency-bin PSD prominence ratio against flanking-bin median. | Direct match to registry; returns `(..., freq)`. |
| narrowband-to-broadband ratio | `RELATED` | `src/cogpy/core/spectral/features.py` | `line_noise_ratio` | function | Ratio of power around a line frequency to flank bands. | Directly useful for line-peak-vs-background contrast, but targeted to a specified line frequency. |
| narrowband-to-broadband ratio | `RELATED` | `src/cogpy/core/preprocess/badchannel/channel_features.py` | `noise_to_signal` | function | Welch high-band to low-band power ratio. | Broad spectral QC feature; not narrowband-specific. |
| narrowband-to-broadband ratio | `RELATED` | `src/cogpy/core/preprocess/badchannel/channel_features.py` | `snr` | function | Alias for `noise_to_signal`. | Same implementation under alternate name. |
| spectral peak detection | `RELATED` | `src/cogpy/core/spectral/features.py` | `fooof_periodic` | function | Extracts periodic component above fitted aperiodic background. | Useful for peak-bearing spectra, but does not itself return discrete peak locations. |
| spectral peak detection | `RELATED` | `src/cogpy/core/spectral/features.py` | `aperiodic_exponent` | function | Fits 1/f exponent over a frequency range. | Background-model helper adjacent to peak characterization. |
| spectral flatness | `EXISTS` | `src/cogpy/core/spectral/features.py` | `spectral_flatness` | function | Wiener entropy from PSD geometric/arithmetic means. | Direct registry match. |
| F-test line detection (`mtm_fstat`) | `EXISTS` | `src/cogpy/core/spectral/bivariate.py` | `mtm_fstat` | function | Multitaper F-statistic at a specified target frequency `f0`. | Direct match for known-frequency line testing. |
| F-test line detection (`mtm_fstat`) | `EXISTS` | `src/cogpy/core/spectral/features.py` | `ftest_line_scan` | function | Thomson multitaper F-test scan across all frequency bins. | Stronger unknown-frequency line scan than `mtm_fstat`; added as PSD-adjacent spectral feature utility. |
| AM artifact detection | `EXISTS` | `src/cogpy/core/spectral/features.py` | `am_artifact_score` | function | Sideband-vs-background log-ratio for AM artifact signatures. | Direct registry match. |
| AM artifact characterization | `RELATED` | `src/cogpy/core/spectral/features.py` | `am_depth` | function | Sideband-vs-carrier log-ratio. | Complementary AM measure, not the primary detector score. |
| line noise detection | `EXISTS` | `src/cogpy/core/preprocess/linenoise.py` | `LineNoiseEstimatorICA` | class | ICA-based line-noise component detection and reconstruction. | Detects line-noise-heavy ICs from spectrogram power around harmonics. |
| line noise detection | `EXISTS` | `src/cogpy/core/preprocess/linenoise.py` | `LineNoiseEstimatorMultitaper` | class | Multitaper line-noise F-test thresholding and reconstruction helper. | Contains `mu_f_stat_func`, `compute_f_test_threshold`, and `f_test`. |
| line noise detection | `RELATED` | `src/cogpy/core/preprocess/linenoise.py` | `get_linenoise_freqs` | helper | Enumerates harmonic frequency bins near `f0`. | Utility for harmonic selection rather than detection. |
| line noise detection | `RELATED` | `src/cogpy/core/preprocess/linenoise.py` | `drop_linenoise_freqs` | helper | Removes identified line-noise frequencies from a frequency grid. | Frequency-axis utility. |
| line noise detection | `RELATED` | `src/cogpy/core/preprocess/linenoise.py` | `drop_linenoise_harmonics` | helper | Drops harmonic bins from an xarray object. | Post-detection cleanup helper. |

## Spatial features

| Concept | Status | Module path | Symbol name | Type | Short description | Notes |
|---|---|---|---|---|---|---|
| Moran's I | `EXISTS` | `src/cogpy/core/measures/spatial.py` | `moran_i` | function | Moran's I spatial autocorrelation on `(..., AP, ML)` scalar grids. | Direct registry match; supports `queen`, `rook`, `ap_only`, `ml_only`. |
| spatial coherence | `EXISTS` | `src/cogpy/core/measures/spatial.py` | `spatial_coherence_profile` | function | Pairwise multitaper coherence binned by inter-electrode distance. | Direct registry match for distance-profile coherence. |
| spatial coherence | `RELATED` | `src/cogpy/core/spectral/bivariate.py` | `coherence` | function | Magnitude-squared coherence from multitaper FFTs. | Core spectral primitive used by `spatial_coherence_profile`. |
| spatial gradients | `EXISTS` | `src/cogpy/core/measures/spatial.py` | `gradient_anisotropy` | function | Log-ratio of AP vs ML gradient magnitudes. | Direct registry match for directional gradient imbalance. |
| spatial gradients / grid derivative | `RELATED` | `src/cogpy/core/measures/spatial.py` | `csd_power` | function | 2D Laplacian-based current source density transform on grid signals. | Spatial derivative/CSD operator, not the same as anisotropy metric. |
| row/column energy measures | `EXISTS` | `src/cogpy/core/measures/spatial.py` | `marginal_energy_outlier` | function | Computes row/column energy, z-scores, and outlier masks. | Direct registry match. |
| spatial autocorrelation | `PARTIAL` | `src/cogpy/core/preprocess/badchannel/spatial.py` | `anticorrelation` | function | Median neighbor correlation inversion over a grid. | Neighbor-based spatial dependence score, but not a general autocorrelation function. |
| neighborhood/grid-based measures | `RELATED` | `src/cogpy/core/preprocess/badchannel/channel_features.py` | `temporal_mean_laplacian` | function | Mean absolute spatial Laplacian over time for `(AP, ML, time)` data. | Legacy grid feature used in bad-channel preprocessing. |
| grid-based measures | `RELATED` | `src/cogpy/core/preprocess/badchannel/grid.py` | `grid_adjacency` | helper | Builds sparse adjacency for `(AP, ML)` layouts. | Infrastructure for neighborhood/grid features. |
| grid-based measures | `RELATED` | `src/cogpy/core/preprocess/badchannel/grid.py` | `make_footprint` | helper | Creates 2D neighborhood footprints. | Used by local spatial normalization / adjacency building. |
| grid-based measures | `RELATED` | `src/cogpy/core/preprocess/badchannel/grid.py` | `grid_edges` | helper | Enumerates source-destination neighbor pairs on a grid. | Low-level grid topology utility. |

## Normalization features

| Concept | Status | Module path | Symbol name | Type | Short description | Notes |
|---|---|---|---|---|---|---|
| z-score across channels/windows | `EXISTS` | `src/cogpy/core/preprocess/badchannel/feature_normalization.py` | `normalize_windowed_features` | function | Standard or robust z-score normalization across a chosen dataset dimension. | Default use is across `time_win`; general dim-based xarray normalization. |
| z-score across channels / signal axis | `EXISTS` | `src/cogpy/core/preprocess/filtering/normalization.py` | `zscorex` | function | xarray z-score or robust z-score along a selected dimension. | Signal-level normalization helper, often time-axis oriented. |
| robust z-score | `EXISTS` | `src/cogpy/core/preprocess/badchannel/spatial.py` | `normalize_robust_z` | helper | Robust z-score from value minus neighborhood median over neighborhood MAD. | Direct neighborhood robust-z primitive. |
| neighborhood normalization | `EXISTS` | `src/cogpy/core/preprocess/badchannel/pipeline.py` | `normalize_features_from_raw` | function | Applies ratio/difference/robust-z normalization to feature maps using spatial neighbors. | Direct pipeline-level neighborhood normalization pass. |
| neighborhood normalization | `EXISTS` | `src/cogpy/core/preprocess/badchannel/spatial.py` | `neighborhood_median` | helper | Computes local medians from adjacency neighborhoods. | Core support for neighborhood normalization. |
| neighborhood normalization | `EXISTS` | `src/cogpy/core/preprocess/badchannel/spatial.py` | `neighborhood_mad` | helper | Computes local MAD from adjacency neighborhoods. | Core support for robust neighborhood normalization. |
| neighborhood normalization | `EXISTS` | `src/cogpy/core/preprocess/badchannel/spatial.py` | `normalize_ratio` | helper | Normalizes values by neighborhood median ratio. | Ratio-style local normalization. |
| neighborhood normalization | `EXISTS` | `src/cogpy/core/preprocess/badchannel/spatial.py` | `normalize_difference` | helper | Normalizes values by subtracting neighborhood median. | Difference-style local normalization. |
| neighborhood normalization | `EXISTS` | `src/cogpy/core/preprocess/badchannel/spatial.py` | `local_robust_zscore_grid` | function | Local robust z-score over a 2D footprint neighborhood. | Grid-local robust normalization independent of adjacency lists. |
| channel normalization / feature-map normalization | `RELATED` | `src/cogpy/core/preprocess/badchannel/pipeline.py` | `compute_feature_maps_for_window` | function | Computes raw feature maps and applies per-feature neighborhood normalization specs. | Mixed extraction + normalization stage. |
| channel normalization / feature-map normalization | `RELATED` | `src/cogpy/core/preprocess/badchannel/pipeline.py` | `compute_features_sliding_legacy` | function | Sliding-window feature extraction with optional local robust z-scoring of each map. | Legacy compatibility path. |
| robust summary normalization context | `RELATED` | `src/cogpy/core/preprocess/badchannel/feature_normalization.py` | `summarize_windowed_features` | function | Collapses windowed features into summary statistics including MAD. | Summarization rather than normalization, but part of same preprocessing stack. |

## Detection / pipeline features

| Concept | Status | Module path | Symbol name | Type | Short description | Notes |
|---|---|---|---|---|---|---|
| burst detection | `EXISTS` | `src/cogpy/core/detect/burst.py` | `BurstDetector` | class | Event detector wrapping spectrogram + h-maxima burst peak detection. | Direct event-detector interface for burst peaks. |
| burst detection | `EXISTS` | `src/cogpy/core/burst/blob_detection.py` | `detect_hmaxima` | function | H-maxima burst candidate detection on n-D xarray data. | Primary low-level burst-peak extractor used by `BurstDetector`. |
| burst detection | `RELATED` | `src/cogpy/core/burst/blob_detection.py` | `detect_blobs` | function | Blob detection over scale-space with sigma controls. | Broader burst/blob candidate detector adjacent to h-maxima path. |
| burst aggregation pipeline | `RELATED` | `src/cogpy/core/spectral/spectrogram_burst.py` | `detect_blob_candidates` | function | Detects spectrogram blob candidates from time-frequency data. | Spectrogram-specific burst workflow outside `core/detect`. |
| burst aggregation pipeline | `RELATED` | `src/cogpy/core/spectral/spectrogram_burst.py` | `aggregate_bursts` | function | Merges burst candidates across channels/time/frequency neighborhoods. | Multi-channel burst grouping utility. |
| event detectors | `EXISTS` | `src/cogpy/core/detect/threshold.py` | `ThresholdDetector` | class | Generic threshold-crossing interval detector with optional bandpass/envelope preprocessing. | Reused in prebuilt ripple/gamma pipelines. |
| event detectors | `EXISTS` | `src/cogpy/core/detect/ripple.py` | `RippleDetector` | class | Bandpass, envelope, z-score, dual-threshold ripple detector. | Dedicated ripple event detector. |
| event detectors | `EXISTS` | `src/cogpy/core/detect/ripple.py` | `SpindleDetector` | class | Ripple-detector variant with spindle defaults. | Another concrete detector in same framework. |
| event detector helper | `EXISTS` | `src/cogpy/core/detect/utils.py` | `dual_threshold_events_1d` | function | Converts 1D threshold crossings into interval events. | Core helper used in ripple-style detection. |
| event detector helper | `RELATED` | `src/cogpy/core/detect/utils.py` | `zscore_1d` | helper | Safe z-scoring for 1D detector inputs. | Detector-local normalization stage. |
| pipeline-based detection | `EXISTS` | `src/cogpy/core/detect/pipeline.py` | `DetectionPipeline` | class | Chains transforms and detectors into reproducible workflows. | Direct pipeline abstraction. |
| pipeline-based detection | `EXISTS` | `src/cogpy/core/detect/pipelines.py` | `BURST_PIPELINE` | pipeline object | Prebuilt spectrogram + burst detector pipeline. | Concrete burst pipeline definition. |
| pipeline-based detection | `EXISTS` | `src/cogpy/core/detect/pipelines.py` | `RIPPLE_PIPELINE` | pipeline object | Prebuilt bandpass/envelope/z-score + threshold pipeline. | Concrete ripple pipeline definition. |
| pipeline-based detection | `EXISTS` | `src/cogpy/core/detect/pipelines.py` | `FAST_RIPPLE_PIPELINE` | pipeline object | Prebuilt fast-ripple threshold pipeline. | Concrete detector workflow. |
| pipeline-based detection | `EXISTS` | `src/cogpy/core/detect/pipelines.py` | `GAMMA_BURST_PIPELINE` | pipeline object | Prebuilt gamma-burst threshold pipeline. | Concrete detector workflow. |
| pipeline transform | `RELATED` | `src/cogpy/core/detect/transforms/spectral.py` | `SpectrogramTransform` | transform | Pipeline transform that computes a spectrogram. | Burst pipeline building block. |

## Observations

- The strongest direct matches to the conceptual registry are concentrated in `src/cogpy/core/spectral/features.py`, `src/cogpy/core/measures/spatial.py`, and `src/cogpy/core/preprocess/badchannel/*`.
- Line-noise functionality exists in two layers: PSD/spectral statistics (`narrowband_ratio`, `mtm_fstat`, `ftest_line_scan`, `line_noise_ratio`) and dedicated preprocessing/removal classes in `src/cogpy/core/preprocess/linenoise.py`.
- Spatial QC is implemented mainly as grid-level scalar measures and neighborhood-based bad-channel features; direct symbols for spatial FFT and spatial kurtosis were not found in the scanned modules.
- Moran-style spatial autocorrelation exists directly, while additional neighbor-dependent spatial structure is represented indirectly through `anticorrelation`, Laplacian/CSD-style transforms, and grid adjacency helpers.
- Normalization support is broad: global z-score utilities, robust windowed feature normalization, and explicit neighborhood normalization primitives all already exist.
- Detection support is already organized around a reusable detector/pipeline framework, with concrete burst, ripple, spindle, and threshold-based workflows present.
