# CogPy Composition and Orchestration Review

## Context
This review treats the external project pipeline at `docs/reference/example-snakemake-pipeline` as the primary source of practical composition patterns. The point is not to infer composition from aspirational internal abstractions, but to inspect which `cogpy` components are actually called by real Snakemake rules and how those scripts pass data between stages.

The example flow is a preprocess pipeline centered on bad-channel detection, interpolation, line-noise handling, and QC/report generation. In practice, orchestration lives in the external Snakefile and script nodes. `cogpy` is used mainly as a library of IO helpers, bad-channel feature functions, interpolation helpers, and a small number of spectral utilities.

## CogPy components used in practice
The example pipeline uses a relatively small and concrete subset of `cogpy`.

Directly called from Snakefile or scripts:

- `cogpy.io.converters.bids_lfp_to_zarr`
  - Used in `raw_zarr` to convert BIDS `.lfp` plus sidecars into a Zarr-backed `sigx`.
- `cogpy.io.ecog_io.from_zarr`, `cogpy.io.ecog_io.to_zarr`, `cogpy.io.ecog_io.assert_ecog`
  - Used by filter, feature extraction, interpolation, and downsample rules on Zarr-backed grid signals.
- `cogpy.io.ieeg_io.from_file`
  - Used by visualization and line-noise QC scripts to load `.lfp` binaries plus sidecars as grid-shaped xarray signals.
- `cogpy.io.sidecars.read_json_metadata`, `update_sampling_frequency_json`, `propagate_sidecars`
  - Used in the line-noise branch for `.lfp` plus `.json` bundle management.
- `cogpy.core.preprocess.badchannel.grid.grid_adjacency`, `make_footprint`
  - Used by `scripts/feature.py` to build the neighborhood structure for feature normalization.
- `cogpy.core.preprocess.badchannel.pipeline.FeatureSpec`, `compute_features_sliding`
  - This is the main feature extraction primitive actually composed in the current workflow.
- `cogpy.core.preprocess.badchannel.badlabel.DbscanParams`, `grouped_dbscan_outliers`
  - Used by `scripts/badlabel.py` for grouped DBSCAN-based outlier detection.
- `cogpy.preprocess.interpolate.interpolate_bads`
  - Used by interpolation scripts to repair bad channels.
- `cogpy.core.utils.stats.robust_zscore`
  - Used by `plot_feature_maps.py` for quantile-based deviation scoring.
- `cogpy.core.spectral.multitaper.mtm_spectrogramx`
  - Used in `scripts/linenoise/sample_spectrogram_plot.py` for QC spectrogram generation.

Used in legacy or compatibility paths, but not the current preferred flow:

- `cogpy.core.preprocess.channel_feature.ChannelFeatures`, `save_features`
  - Used by `feature_deprecated.py` and by the internal `src/cogpy/workflows/preprocess` workflow, but explicitly marked legacy in the library.
- `cogpy.preprocess.detect_bads.OutlierDetector`
  - Used by the internal workflow and older scripts, but the module itself is deprecated in favor of `cogpy.core.preprocess.badchannel.badlabel`.

Present in `cogpy` but not used by this practical composition:

- `cogpy.core.detect.DetectionPipeline` and prebuilt detector pipelines.
- `cogpy.datasets.schemas.EventCatalog`, `Events`, `Intervals`.
- `cogpy.core.spectral.specx.psdx` and `spectrogramx`.
- `cogpy.core.preprocess.badchannel.channel_features.extract_channel_features_xr`.

Those APIs may be useful elsewhere, but they are not the source of truth for how this example preprocess pipeline is currently composed.

## Data flow patterns
### File-based outputs
The real DAG is file-first.

- Raw BIDS `.lfp` plus `.json`, `_channels.tsv`, `_electrodes.tsv` are converted to a Zarr product containing `sigx`.
- Zarr signal stages are used for early transforms:
  - `raw_zarr`
  - `lowpass`
  - `downsample`
- Feature extraction writes a feature Zarr dataset, not a tabular artifact.
- Bad-channel labels are serialized as a flat `.npy` boolean map.
- Row/column noise outputs are serialized as:
  - `.tsv` summary table
  - `.npz` arrays used by downstream interpolation
  - `.png` QC figure
- Interpolation switches back to `.lfp` binary output for compatibility with downstream viewers and line-noise tools.
- The line-noise branch passes `.lfp` plus `.json` bundles, plus small JSON comb-frequency profiles.
- Final products are mostly HTML/PNG report artifacts and symlinked sidecars.

This means composition is primarily across files, with xarray/NumPy objects existing only inside script boundaries.

### xarray objects
Within each script, xarray is the dominant in-memory interface.

- `ecog_io.from_zarr(...)[\"sigx\"]` returns a grid signal.
- `ieeg_io.from_file(..., grid=True)` returns a grid signal built from `.lfp` plus BIDS sidecars.
- Scripts then manually transpose these arrays to the shapes they need:
  - `sigx.transpose(\"AP\", \"ML\", \"time\")` for NumPy-heavy feature/interpolation code
  - `sigx.isel(AP=..., ML=...)` for 1D spectral QC
- After NumPy computation, scripts often reconstruct xarray objects by hand before saving or plotting.

There is little use of higher-level schema-aware composition inside the scripts. The flow relies on repeated local conventions instead.

### Tables
There are only a few real table-like intermediates.

- `rowcol_noise.py` writes a `.tsv` with row/column scores and outlier flags.
- Feature datasets are not serialized as long-form tables. They remain gridded xarray variables in Zarr.
- `plot_feature_maps.py` constructs temporary pandas DataFrames for pairplots, but these are visualization-only intermediates.

So the practical composition unit is not a stable feature table API. It is a grid-shaped feature dataset plus small ad hoc summary tables.

### Event-like outputs
Event-like objects are effectively absent from the current example pipeline.

- No rule passes `EventCatalog`, `Events`, or `Intervals` between steps.
- The only event-like external input is optional brainstate `.mat` content used to pick an SWS/NREM plotting window in `sample_spectrogram_plot.py`.
- Burst, ripple, spindle, or interval outputs are not part of the active preprocess flow.

This matters because it means current external composition is driven by grid-signal and feature-map products, not by a stable event object model.

### Intermediate features
The main intermediate semantic product is the feature Zarr dataset produced by `scripts/feature.py`.

- Each feature is stored as a separate variable.
- Variables use dims `(\"AP\", \"ML\", \"time\")`, where `time` is actually the window-center axis.
- Dataset attrs store `fs`, `window_size`, `window_step`, JSON-serialized geometry config, and JSON-serialized feature spec config.
- Downstream scripts repeatedly derive summary statistics from this dataset:
  - quantiles over `time`
  - stacked `AP/ML -> ch` feature matrices
  - per-window row/column aggregates
  - `composite_badness` movies

### PSD and spectrogram outputs
PSD and spectrogram products are not passed between Snakemake steps as stable serialized intermediates.

- `measure_combfreqs.py` uses SciPy Welch directly and writes only a JSON summary.
- `sample_spectrogram_plot.py` uses `mtm_spectrogramx` directly and writes only an HTML report.
- `comb_qc_plot.py` uses `ghostipy.mtm_spectrogram` directly and writes only an HTML report.

So spectral outputs are ephemeral analysis products computed inside leaf scripts, not a shared schema-stable pipeline product.

## Script-level composition patterns
Several composition patterns repeat across the external workflow.

### Thin script-node orchestration
Each Snakemake rule is mostly a path contract. The real logic sits in small Python entrypoints that:

- load one or more files
- convert to xarray or NumPy
- run one concrete `cogpy` function or a short composition of them
- write a single serialized artifact

This is pragmatic and reproducible, but it means composition knowledge is dispersed across scripts rather than centralized in reusable `cogpy` helpers.

### Manual schema bridging
Scripts frequently perform manual shape and coord adaptation rather than calling a schema helper.

- Grid signals are explicitly transposed to `(AP, ML, time)` before feature or interpolation logic.
- Feature datasets are manually rebuilt with dims `(AP, ML, time)` after `compute_features_sliding`.
- Quantile feature matrices are manually stacked from `AP/ML` into `ch`.

This repeated manual bridging is a strong signal that the underlying shapes are stable enough to rely on, but not yet encapsulated.

### Mixed xarray-at-boundary, NumPy-in-core style
The dominant pattern is:

1. load a labeled xarray object
2. strip to NumPy for core computation
3. reconstruct xarray when labels are needed again

That is especially visible in:

- `scripts/feature.py`
- `scripts/interpolate.py`
- `scripts/rowcol_noise.py`

This is consistent with the schema review: low-level compute is NumPy-first, while xarray is mainly used at the pipeline boundary.

### Sidecar-coupled binary products
Once the workflow leaves Zarr and returns to `.lfp`, the pipeline assumes a bundle abstraction even though no explicit bundle class exists.

- `.lfp` is only meaningful together with `.json` sidecar metadata.
- Some steps also need `_channels.tsv`, `_electrodes.tsv`, or `.xml`.
- Several scripts repeat logic for reading/updating/propagating those sidecars.

This is a real composition pattern and a good target for lightweight library support.

### Derived summaries over the same feature dataset
The feature Zarr is reused in multiple downstream branches:

- bad-channel labeling
- feature-map plotting
- row/column noise detection
- badness movie generation

Each script reimplements some combination of:

- quantile reduction over the window axis
- `AP/ML` stacking
- feature concatenation across variables
- robust z-scoring or Euclidean/deviation scoring

That repeated logic is currently stable enough to reuse, but it is still duplicated across scripts.

## Outdated internal pipeline abstractions
Several internal abstractions should not be treated as the source of truth for current practice.

### Explicitly legacy library modules
- `src/cogpy/core/preprocess/channel_feature.py`
  - Marked as a legacy compatibility surface.
  - The module itself warns that canonical implementation lives under `cogpy.core.preprocess.badchannel.*`.
- `src/cogpy/core/preprocess/detect_bads.py`
  - Also explicitly marked legacy and deprecated in favor of `cogpy.core.preprocess.badchannel.badlabel`.

These modules still work, but they describe an older composition style centered on `ChannelFeatures` and `DetectBadsPipe`, not the current external workflow.

### Internal workflow prototype
- `src/cogpy/workflows/preprocess/Snakefile`
- `src/cogpy/workflows/preprocess/scripts/*`

This internal workflow uses the deprecated bad-channel stack, older path conventions, and even a placeholder shell command for interpolation. It is best interpreted as an older prototype or internal example, not as the canonical representation of present-day composition.

### Deprecated scripts inside the external example pipeline
The example pipeline itself contains archival or compatibility layers:

- `scripts/feature_deprecated.py`
- `scripts/lfp_video_deprecated.py`
- `scripts/_deprecated_badchannel/*`
- `scripts/_deprecated_linenoise/*`

Those paths confirm active migration away from older bad-channel and line-noise composition patterns. They are useful historical context, but they should not drive new abstractions.

### Important distinction: active but unused abstractions
`cogpy.core.detect.DetectionPipeline` and the prebuilt detector pipelines are not obviously deprecated, but they are also not exercised by this preprocess example. They should therefore not be treated as evidence of how real external preprocess composition currently works.

## Interface assumptions imposed on CogPy
The external pipeline makes strong assumptions about `cogpy`, even when those assumptions are not formalized in validators.

### Input schemas
- Grid signals loaded from Zarr or `.lfp` are expected to be interpretable as `AP`, `ML`, and `time`.
- Scripts often expect `sigx.attrs[\"fs\"]` to exist.
- `ieeg_io.from_file(..., grid=True)` is expected to infer shape and coordinates from BIDS sidecars.
- `.lfp` binary readers assume sidecars provide at least `SamplingFrequency` and `ChannelCount`.

### Output schemas
- Zarr signal stages are expected to contain a variable named `sigx`.
- Feature Zarr is expected to be an `xr.Dataset` with one variable per feature.
- Feature variables are expected to align on the same `(AP, ML, time)` grid, where `time` is the window-center axis.
- `badlabel.npy` is expected to be a flat array that can be reshaped to `(AP, ML)` in the same order used by feature extraction.
- `noise_npz` is expected to expose arrays like `bad_rows_t` and `bad_cols_t` with shapes matching feature-window counts.

### Naming and coordinate conventions
- Uppercase `AP` and `ML` are the practical norm in this pipeline.
- The feature-window axis is serialized as `time`, not `time_win`.
- Some scripts replace physical `AP/ML` coordinates with integer grid indices before plotting or indexing.
- Group labels such as hemisphere may appear as a coord on `ML` or on `(AP, ML)`.

These conventions are stable enough for the current scripts, but they do not align perfectly with several design-target constants in `datasets.schemas.py`.

### Serialization expectations
- Zarr is the durable format for labeled xarray objects.
- `.lfp` plus sidecars is the durable format for interoperability with external tools and viewers.
- JSON is used for small metadata products and profile summaries.
- PNG/HTML are treated as terminal visualization artifacts, not reusable data products.

### xarray compatibility assumptions
The pipeline expects `cogpy` loaders to return xarray objects that are easy to transpose, index, stack, and serialize. It does not expect higher-level data classes or workflow objects. This is important: external scripts are already written against xarray-first interfaces, not against an internal pipeline framework.

### Are CogPy outputs schema-stable enough for scripted composition?
For this specific preprocess branch, the answer is: stable enough locally, not yet stable enough globally.

Stable enough locally:

- `sigx` Zarr layout
- `fs` in attrs
- feature dataset variable-per-feature pattern
- flat badlabel reshape convention
- `.lfp` plus sidecar handling

Not yet globally stable:

- validators are not used in the active scripts
- feature and spectrum design constants are mostly not enforced
- dim ordering differs between active outputs and some canonical constants
- the feature-window axis is called `time` in practice, not `time_win`
- spectral outputs are computed ad hoc rather than serialized through a stable schema

So the external pipeline is relying on conventions that are de facto stable, but still under-specified in reusable runtime contracts.

## Lightweight composition opportunities
The right opportunity is not a workflow engine inside `cogpy`. It is a small set of reusable helpers that eliminate repeated script glue.

### 1. A validated feature-dataset contract
Add a helper for the actual bad-channel feature product, for example:

- `coerce_badchannel_feature_dataset(ds)`
- `validate_badchannel_feature_dataset(ds)`

It should validate:

- variable-wise `(AP, ML, time_win)` or `(AP, ML, time)` convention
- shared coords
- required attrs such as `fs`, `window_size`, `window_step`
- optional provenance fields like `geometry`, `features`, `implementation`

This would directly reduce manual schema assumptions across `feature.py`, `badlabel.py`, `rowcol_noise.py`, and `badness_video.py`.

### 2. Grid-feature stacking/summarization helpers
The pipeline repeats the same transformations on feature datasets:

- quantile summaries over the window axis
- stacking `AP/ML` into channel rows
- concatenating features into a 2D design matrix

A small helper such as `stack_feature_dataset_for_ml(ds, quantiles=...)` would remove duplicated code and make downstream scripts easier to compare and test.

### 3. A binary LFP bundle helper
There is a practical bundle abstraction already:

- `.lfp`
- `.json`
- optional `_channels.tsv`
- optional `_electrodes.tsv`
- optional `.xml`

A lightweight helper or dataclass such as `LfpBundle` could centralize:

- metadata reads
- sidecar propagation
- sampling-rate updates after downsampling
- validation of file-size vs `ChannelCount`

This would remove repeated sidecar logic from multiple scripts without turning `cogpy` into an orchestrator.

### 4. Window-aware repair helpers
Interpolation currently reimplements a window loop that aligns:

- static bad labels
- dynamic row masks
- dynamic column masks
- feature window metadata

A helper that accepts `(sig, static_mask, bad_rows_t, bad_cols_t, window_size, window_step)` and returns repaired signal blocks would reduce one of the more fragile pieces of script duplication.

### 5. Spectral QC helpers that return xarray
The line-noise scripts compute Welch and spectrogram views ad hoc. A small utility that standardizes:

- single-channel extraction from grid signals
- spectrogram/PSD computation
- time/frequency coordinate naming
- optional brainstate-window restriction

would make those scripts more consistent without requiring persisted spectral products.

### 6. Public shims for current bad-channel primitives
The external pipeline already imports deep internal paths such as `cogpy.core.preprocess.badchannel.pipeline.compute_features_sliding`. That is a signal that a small public shim layer would help:

- `cogpy.preprocess.badchannel.compute_features_sliding`
- `cogpy.preprocess.badchannel.grouped_dbscan_outliers`
- `cogpy.preprocess.badchannel.grid_adjacency`

This would reduce direct coupling to internal module layout while preserving the current script-oriented composition model.

## Observations
The example pipeline shows that practical composition in CogPy today is not centered on an internal workflow framework. It is centered on a handful of stable-enough library functions combined by external Snakemake scripts.

The most mature real composition path is the bad-channel preprocessing branch:

- grid signal IO
- sliding-window feature extraction
- DBSCAN-based bad labeling
- transient-aware interpolation
- line-noise preparation and QC

The biggest architectural gap is not missing orchestration. It is missing small interface helpers around already-stable conventions:

- validated feature-dataset schemas
- reusable grid-feature summarization
- standardized `.lfp` plus sidecar bundle handling
- window-aware repair utilities

For future feature implementation, that implies:

- prioritize schema and IO contracts over internal workflow abstractions
- treat external script composition as the practical compatibility target
- do not use deprecated internal preprocess workflow code as a design reference
- add reusable helpers where the same script glue appears in multiple places

That path would improve external pipeline composition immediately while keeping `cogpy` a library, not a workflow engine.
