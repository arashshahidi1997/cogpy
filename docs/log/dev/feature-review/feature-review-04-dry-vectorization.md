# CogPy DRY and Vectorization Review

## Repeated logic patterns

- Robust z-scoring and MAD logic is implemented in several places with slightly different wrappers:
  - `src/cogpy/core/preprocess/filtering/normalization.py::zscorex`
  - `src/cogpy/core/preprocess/badchannel/feature_normalization.py::normalize_windowed_features`
  - `src/cogpy/core/preprocess/badchannel/spatial.py::{normalize_robust_z, local_robust_zscore_grid}`
  - `docs/reference/example-snakemake-pipeline/scripts/rowcol_noise.py` defines `_robust_z_1d` and `_robust_z_spatial_per_time`
  - `docs/reference/example-snakemake-pipeline/scripts/plot_feature_maps.py` uses `robust_zscore` again for channel deviation scoring
- Rolling / window logic is split between reusable core helpers and bespoke script loops:
  - reusable: `src/cogpy/core/utils/sliding_core.py`
  - bespoke: `src/cogpy/core/preprocess/badchannel/pipeline.py::compute_features_sliding` loops window-by-window
  - bespoke: `docs/reference/example-snakemake-pipeline/scripts/interpolate.py` loops window-by-window for transient repair and overlap-write semantics
  - bespoke: `docs/reference/example-snakemake-pipeline/scripts/linenoise/sample_spectrogram_plot.py` does ad hoc second-based slicing
- Axis handling is repeated throughout the preprocess path:
  - xarray grid schema prefers `("time","ML","AP")`
  - many spatial and pipeline routines expect `(AP, ML, time)` or `(..., AP, ML)`
  - scripts repeatedly call `transpose("AP", "ML", "time")`, then later flatten or rebuild arrays
- Xarray to NumPy conversion is a stable repeated pattern rather than an exception:
  - `feature.py`, `interpolate.py`, and `rowcol_noise.py` all drop to NumPy for core work and then manually rebuild xarray objects
  - `psdx()` and `spectrogramx()` already preserve labels, but many downstream spectral feature functions still consume raw `(psd, freqs)` arrays
- Grid reshaping / stacking is repeated across multiple scripts:
  - `badlabel.py`, `plot_feature_maps.py`, `feature_umap.py`, and `rowcol_noise.py` all use some variation of `quantile(...).stack(ch=("AP","ML"))`
  - flatten-to-binary conversion appears in both `src/cogpy/io/converters/bids.py::zarr_to_dat` and `docs/reference/example-snakemake-pipeline/scripts/interpolate.py`
- Neighborhood operations are partly modularized, but script-facing composition still repeats:
  - reusable primitives exist in `src/cogpy/core/preprocess/badchannel/spatial.py`
  - `src/cogpy/core/preprocess/badchannel/pipeline.py` repeatedly recomputes neighbors, medians, MADs, and normalizations feature-by-feature
- Spectral baseline estimation exists in more than one form:
  - PSD flank-based baseline ratios in `src/cogpy/core/spectral/features.py::{line_noise_ratio, narrowband_ratio}`
  - an elbow-interpolated 50 Hz baseline in `src/cogpy/core/preprocess/linenoise.py`
  - `docs/reference/example-snakemake-pipeline/scripts/linenoise/measure_combfreqs.py` computes a separate Welch-plus-prominence peak workflow rather than using the library spectral feature layer
- Score-to-table / score-to-event conversion is duplicated conceptually and structurally:
  - there are two `EventCatalog` implementations: `src/cogpy/datasets/schemas.py` and `src/cogpy/core/events/catalog.py`
  - both define overlapping `to_events`, `to_intervals`, and `to_event_stream` behavior
  - burst-to-event conversion is handled in multiple locations via `from_burst_dict` / `from_hmaxima`

## Candidate helper abstractions

- `grid_feature_matrix(ds, *, summary="quantile", q=0.9)`:
  - standardize `Dataset -> DataArray(feature, AP, ML) -> stacked (ch, feature)` conversion
  - would remove near-duplicate code in `badlabel.py`, `feature_umap.py`, `plot_feature_maps.py`, and `rowcol_noise.py`
- `grid_summary_features(ds, *, qmin, qmax, nq)`:
  - centralize the repeated quantile-feature construction used for DBSCAN, pairplots, and deviation scoring
  - likely belongs beside `feature_normalization.py`
- `robust_score(x, *, dim=None, mode="global"|"per_time")`:
  - unify robust z-score / MAD scaling semantics across signal normalization, feature normalization, and row/column scoring
- `transpose_grid(x, *, order="ap-ml-time")` or `coerce_grid_compute_view(...)`:
  - provide one sanctioned bridge from schema-native xarray to NumPy compute order
  - would reduce repeated manual `transpose(...).data` calls and make axis assumptions explicit
- `grid_stack_channels()` and `grid_unstack_channels()`:
  - lightweight helpers for AP×ML <-> flat channel transformations
  - should preserve or restore AP/ML coordinates and avoid repeated index bookkeeping
- `windowed_mask_to_sample_mask(...)`:
  - convert row/column or channel masks defined on window centers into sample-domain masks
  - would simplify `interpolate.py` and likely future transient-repair code
- `dat_bundle_from_sigx()` / `sigx_to_lfp_bundle()`:
  - internalize repeated `.transpose -> reshape -> int16 -> write` plus sidecar propagation logic
  - especially useful because the example pipeline treats `.lfp + .json (+ .tsv)` as a de facto bundle abstraction
- `spectral_peak_profile(...)`:
  - small wrapper around Welch / PSD features for line or comb-noise summaries
  - would let scripts use the same baseline and peak logic rather than embedding separate signal-processing recipes
- `feature_dataset_qc(ds)`:
  - compute reusable QC derivatives such as `composite_badness`, quantile maps, deviation score, row/column aggregates
  - good fit for small, composable preprocess helpers rather than a monolithic workflow class
- `event_catalog_from_scores(...)`:
  - unify score/peak/blob tables into one canonical catalog path and reduce the current split between dataset-schema and core-event catalog layers

## Vectorization observations

- `src/cogpy/core/preprocess/badchannel/pipeline.py::compute_features_sliding` is the main remaining obvious Python loop hotspot.
  - It iterates over windows and recomputes feature maps per block.
  - For reducers that are purely time-axis reductions, `sliding_core.sliding_window`, `running_reduce`, or `running_reduce_xr` could eliminate the outer Python loop.
  - This is especially relevant for amplitude, variance, deviation, derivative, and kurtosis.
- `compute_feature_maps_for_window` also loops feature-by-feature and repeatedly recomputes neighborhood summaries.
  - For features sharing the same normalization mode, neighborhood median/MAD could be computed batched across features instead of one feature at a time.
- `neighborhood_median` and `neighborhood_mad` still loop over nodes in Python.
  - That is acceptable for small grids, but it is the main non-vectorized core in neighborhood normalization.
  - If this becomes hot, sparse-matrix gather or padded-neighbor tensor approaches are the natural next step.
- `docs/reference/example-snakemake-pipeline/scripts/interpolate.py` uses a Python loop over windows.
  - This is partly justified because writeback semantics are overlap-aware.
  - Still, mask construction could be vectorized across all windows before the repair loop, and sample-range bookkeeping could be centralized.
- `docs/reference/example-snakemake-pipeline/scripts/lfp_video.py::_spatial_median_3x3` is already vectorized well with `sliding_window_view`.
- `docs/reference/example-snakemake-pipeline/scripts/rowcol_noise.py` is mostly vectorized already after the initial reshape.
  - Its main duplication problem is not speed but repeated local scoring logic.

## xarray interoperability observations

- Coordinates and attrs are often lost at the point where real computation begins.
  - `feature.py` converts `sigx` to NumPy early, then reconstructs a new dataset manually.
  - `interpolate.py` performs repair on a raw NumPy array and only later wraps the result back into `sigx.copy(data=...)`.
- There is a persistent dim-order mismatch:
  - schemas define grid signals as `("time","ML","AP")`
  - many spatial functions assume `(..., AP, ML)` or `(AP, ML, time)`
  - some design docs target `("AP","ML",...)`
  - this forces manual transposes at almost every composition boundary
- `psdx()` and `spectrogramx()` are better xarray citizens than many downstream consumers.
  - They preserve non-reduced coords and attrs.
  - But spectral feature functions remain numpy-first, so labels are typically discarded just after computation.
- `extract_channel_features_xr()` already represents a more xarray-native path than the current example pipeline uses.
  - The example pipeline instead builds datasets manually around `compute_features_sliding`.
  - That suggests either the xarray-native path is not yet ergonomic enough for practice, or it lacks one or two helpers needed by pipeline scripts.
- Grid coords are sometimes replaced by integer indices for computation safety, then stored separately as `AP_coord` / `ML_coord`.
  - Practical, but it indicates missing helper support for “index-safe compute, coord-preserving output.”

## Grid processing utilities

- High-value generic helpers for AP×ML grids:
  - `grid_stack_channels` and `grid_unstack_channels`
  - `grid_index_coords` / `grid_physical_coords`
  - `grid_to_dat_flat` and `dat_flat_to_grid`
  - `grid_summary_map(ds, reducer=...)`
  - `grid_quantile_features(ds, qmin, qmax, nq)`
  - `grid_rowcol_scores(score_map_t)`
  - `grid_dynamic_mask_from_rows_cols(bad_rows_t, bad_cols_t)`
  - `grid_window_centers_to_time(ds_or_attrs)`
- `src/cogpy/core/preprocess/badchannel/grid.py` already covers topology creation well.
  - The gap is not adjacency creation.
  - The gap is script-facing helpers for reshaping, stacking, and row/column summaries built on top of that topology.
- A thin “grid compute bridge” layer would likely deliver more value than more low-level math helpers.
  - Most duplication is about representation changes, not about missing core algorithms.

## Composition glue worth internalizing

- Quantile-feature extraction from feature datasets is repeated enough to belong in CogPy.
  - Same pattern appears in `badlabel.py`, `plot_feature_maps.py`, and `feature_umap.py`.
- Row/column noise scoring looks like reusable library code, not script-only glue.
  - `rowcol_noise.py` computes stable, generic outputs: score maps, row/column aggregates, robust z-scores, and boolean masks.
  - That should likely be a helper under preprocess/badchannel rather than remaining a script-local implementation.
- Binary signal bundle conversion also looks library-worthy.
  - Example scripts repeatedly assume that writing `.lfp` also implies preserving metadata sidecars and channel geometry context.
- Brainstate-window selection in line-noise QC is another small reusable composition primitive.
  - The logic in `sample_spectrogram_plot.py` is narrow, but the “pick centered window inside a state epoch” behavior is generic and likely to recur.
- Event conversion glue should be internalized by choosing one canonical catalog path.
  - Right now there is unnecessary ambiguity between `cogpy.datasets.schemas.EventCatalog` and `cogpy.core.events.catalog.EventCatalog`.

## Key refactor opportunities

- Highest value: introduce one small helper layer for grid feature dataset composition.
  - Focus on quantile summaries, AP×ML stacking, row/column aggregation, and flat-channel conversion.
  - This would remove the most duplicated script code immediately.
- Highest value: standardize one compute-order bridge for grid tensors.
  - A sanctioned xarray-to-NumPy adapter would remove many manual transposes and reduce axis bugs.
- High value: consolidate robust normalization helpers.
  - Keep one implementation for median/MAD scaling and reuse it across signal, feature, and row/column scoring.
- High value: unify event catalog abstractions.
  - Two overlapping `EventCatalog` implementations are unnecessary friction for any future score-to-event pipeline.
- Medium value: batch or vectorize the sliding bad-channel feature pipeline.
  - The current window loop is workable, but it is the clearest path for computational speedup if feature extraction becomes a bottleneck.
- Medium value: expose a small spectral summary helper for comb/line-noise workflows.
  - The codebase already has the ingredients, but current scripts recombine them ad hoc rather than through one stable API.
