# Package Map

Top-down view of cogpy's module tree. Every subpackage lives directly under
`cogpy/` — there is no indirection layer.

For the full primitive catalog with import paths and signatures, see
{doc}`primitives`.

## Subpackage overview

```
cogpy/
├── base.py                  Signal schemas (ECoGSchema), validation, ECoG wrapper
│
├── detect/                  Event detection framework
│   ├── threshold.py           ThresholdDetector — configurable threshold crossing
│   ├── burst.py               BurstDetector — spectral h-maxima detection
│   ├── ripple.py              RippleDetector, SpindleDetector — dual-threshold
│   ├── pipeline.py            DetectionPipeline — chain transforms + detector
│   ├── utils.py               score_to_bouts, merge_intervals, dual_threshold_events_1d
│   └── transforms/            Composable pre-detection transforms (bandpass, envelope, etc.)
│
├── events/                  Event data structures and matching
│   ├── catalog.py             EventCatalog — unified event container (DataFrame-backed)
│   ├── match.py               match_nearest, estimate_lag, estimate_drift
│   └── overlap.py             detect_overlaps between interval events
│
├── triggered/               Event-locked analysis
│   ├── stats.py               triggered_average, triggered_std, triggered_median, triggered_snr
│   └── template.py            estimate_template, fit_scaling, subtract_template
│
├── regression/              Linear model primitives
│   ├── design.py              lagged_design_matrix, event_design_matrix
│   └── ols.py                 ols_fit, ols_predict, ols_residual
│
├── spectral/                Frequency-domain analysis
│   ├── psd.py                 psd_welch, psd_multitaper
│   ├── specx.py               psdx, spectrogramx, coherencex (xarray wrappers)
│   ├── features.py            band_power, spectral_peak_freqs, ftest_line_scan,
│   │                          line_noise_ratio, narrowband_ratio, am_artifact_score
│   ├── bivariate.py           cross_spectrum, coherence, plv, cross_corr_peak/lag
│   ├── multitaper.py          dpss_tapers, multitaper_fft, mtm_spectrogram
│   ├── whitening.py           ARWhiten class, ar_whitening, ar_yule
│   ├── psd_utils.py           psd_to_db, stack_spatial_dims
│   └── process_spectrogram.py Spectrogram post-processing (outlier repair, smoothing)
│
├── measures/                Signal quality and spatial measures
│   ├── spatial.py             moran_i, gradient_anisotropy, spatial_kurtosis,
│   │                          csd_power, spatial_coherence_profile, marginal_energy_outlier
│   ├── temporal.py            relative_variance, deviation, hurst_exponent, kurtosis, etc.
│   └── comparison.py          snr_improvement, residual_energy_ratio, bandpower_change
│
├── preprocess/              Signal preprocessing
│   ├── filtering/             Canonical filter path
│   │   ├── temporal.py          bandpassx, lowpassx, highpassx, notchx, decimatex
│   │   ├── spatial.py           gaussian_spatialx, median_spatialx, median_highpassx
│   │   ├── reference.py         cmrx (common median reference)
│   │   └── normalization.py     zscorex
│   ├── badchannel/            Bad channel detection stack
│   │   ├── channel_features.py  Feature extraction
│   │   ├── spatial.py           Spatial normalization
│   │   ├── badlabel.py          Outlier labeling (DBSCAN)
│   │   └── pipeline.py          Full pipeline orchestrator
│   ├── interpolate.py         Spatial interpolation of bad channels
│   ├── linenoise.py           Line noise removal
│   └── resample.py            Resampling / decimation
│
├── decomposition/           Dimensionality reduction
│   ├── pca.py                 erpPCA (varimax-rotated PCA)
│   ├── spatspec.py            SpatSpecDecomposition, DesignMatrixReshaper
│   ├── scores.py              Factor score utilities
│   ├── match.py               Factor matching across decompositions
│   └── embed.py               Embedding utilities
│
├── brainstates/             Brain state analysis
│   ├── intervals.py           perievent_epochs, restrict, interval operations
│   ├── brainstates.py         State classification
│   └── EMG.py                 EMG-based state scoring
│
├── plot/                    Visualization
│   ├── hv/                    Interactive (HoloViews/Panel)
│   │   ├── xarray_hv.py        grid_movie, multichannel_view
│   │   ├── topomap.py          TopoMap (AP×ML heatmap)
│   │   ├── orthoslicer.py      Interactive orthoslicer
│   │   ├── ecog_viewer.py      Full ECoG viewer app
│   │   ├── processing_chain.py Filter pipeline UI
│   │   └── time_player.py      TimeHair, PlayerWithRealTime
│   ├── decomposition.py      Static PCA/factor plots
│   ├── specgram_plot.py       Static spectrogram plots
│   └── time_plot.py           Static time-series plots
│
├── io/                      File I/O (separate from compute)
│   ├── ecog_io.py             ECoG binary/XML format reader
│   ├── ieeg_io.py             iEEG BIDS reader
│   ├── converters/            Format converters (MNE, BIDS)
│   └── sidecars.py            JSON sidecar management
│
├── datasets/                Sample data and schemas
│   ├── load.py                load_sample, load_raw_sample
│   └── schemas.py             Data validation schemas (Events, Intervals, etc.)
│
├── utils/                   Shared utilities
│   ├── xarr.py                xarray helpers (reshape, wrap)
│   ├── grid_neighborhood.py   GridNeighborhood, adjacency matrices
│   ├── sliding.py             Rolling window operations
│   └── reshape.py             Array reshape utilities
│
├── model/                   Synthetic signal generation
├── wave/                    Travelling wave analysis
├── burst/                   Spectral burst detection (h-maxima morphology)
├── depth_probe/             Depth probe / CSD analysis
├── tensorscope/             Legacy TensorScope Panel app (historical reference)
├── cli/                     CLI entry points
└── workflows/               Snakemake preprocessing pipelines (packaged as data)
```

## Design boundaries

| Layer | Responsibility | Examples |
|-------|---------------|----------|
| **Compute** | Pure functions on arrays/DataArrays | `spectral`, `measures`, `detect`, `triggered`, `regression` |
| **I/O** | File reading/writing | `io`, `datasets` |
| **Visualization** | Plotting | `plot` |
| **Orchestration** | NOT in cogpy | Snakemake pipelines, notebooks, project repos |

Compute functions never do I/O. I/O functions never transform signals.
Visualization functions accept computed results.
