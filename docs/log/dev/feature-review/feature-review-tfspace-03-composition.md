# CogPy TF-Space Composition Review

## Practical pipeline decomposition

Treating `docs/reference/example-snakemake-pipeline` as the practical source of truth, a TF-space QC workflow would stage most naturally as a small set of thin Snakemake rules with one durable artifact per rule:

1. `load_grid_signal`
   - Input: existing grid signal product, ideally the same Zarr-backed `sigx` pattern used by `raw_zarr`, `lowpass`, and `downsample`.
   - Practical form: `xr.DataArray` loaded inside a script, then manually transposed as needed.

2. `compute_spec`
   - Input: grid signal `(time, ML, AP)` or equivalent.
   - Compute: `spectrogramx()` or `mtm_spectrogramx()` over the full grid.
   - Practical boundary: save a durable spectrogram tensor rather than recomputing inside plotting scripts.

3. `compute_wspec`
   - Input: spectrogram tensor.
   - Compute: whitening / normalization along `freq`, or a background-normalized variant built from existing spectral and normalization helpers.
   - Practical boundary: save this separately if it will feed more than one downstream summary or plot.

4. `compute_spatial_summaries`
   - Input: `spec` or `wspec`, transposed once to `(..., AP, ML)`.
   - Compute: batched `moran_i`, `gradient_anisotropy`, `spatial_kurtosis`, `spatial_noise_concentration`, `marginal_energy_outlier`.
   - Practical boundary: emit reduced score tensors and selected structured row/column outputs.

5. `derive_scores_and_events`
   - Input: reduced `(time_win, freq)` score tensors.
   - Compute: band reductions, thresholded 1D score traces, bout extraction with `score_to_bouts()`, and summary tables.
   - Practical boundary: event catalogs / bout TSVs plus compact score arrays for aggregation.

6. `save_inspection_products`
   - Input: stage outputs above.
   - Compute: HTML/PNG plots, session-level summaries, and optional cross-session aggregation inputs.

This matches the existing preprocess example: rules stay file-first, scripts stay thin, and the real work happens as load -> transpose/bridge -> compute -> serialize.

## Steps that already compose well

- Step 1 -> 2 composes naturally with current practice.
  - The example pipeline already treats Zarr-backed grid signals as the durable intermediate for early stages.
  - `feature.py` and other scripts already use `ecog_io.from_zarr(...)["sigx"]` and work from an xarray boundary.

- Step 2 itself is mostly ready.
  - `spectrogramx()` and `mtm_spectrogramx()` already provide the core TF transform.
  - Repo tutorials explicitly show the intended TF-space pattern: compute spectrogram, transpose once, then batch spatial measures across `(time_win, freq)`.

- Step 4 is numerically in good shape.
  - The spatial measures already accept arbitrary leading batch dims with spatial axes last.
  - For a tensor transposed to `(time_win, freq, AP, ML)`, the main spatial summaries can run without Python loops.

- Step 5 partially composes.
  - `score_to_bouts()` already covers the final 1D score -> bout conversion.
  - The strict `datasets.schemas.EventCatalog` contract exists if the workflow wants validated event-like outputs at the final boundary.

- Step 6 matches current orchestration style exactly.
  - The example pipeline already writes HTML/PNG QC views as leaf outputs and keeps them separate from computational intermediates.

## Steps that currently require script glue

- Step 1 -> 2 still requires manual dim bridging.
  - The enforced grid signal schema is `(time, ML, AP)`.
  - Spatial code expects NumPy arrays with trailing `(AP, ML)`.
  - Existing scripts repeatedly transpose by hand before compute.

- Step 2 -> 3 is the biggest missing reusable composition boundary.
  - There is no single canonical `wspec(...)` constructor.
  - Whitening exists for time series, and normalization pieces exist for xarray features, but TF normalization is currently assembled ad hoc.

- Step 2 -> 4 still requires explicit transpose and xarray loss.
  - `spectrogramx()` output does not land directly in the spatial-measure batch convention.
  - Spatial reducers are NumPy-first, so coord preservation and re-wrapping are the caller's job.

- Step 4 -> 5 requires caller-defined reductions.
  - There is no current helper that turns a full `(time_win, freq)` abnormality tensor into:
    - a 1D score trace
    - per-band score traces
    - occupancy summaries
    - validated event catalogs
  - The workflow author still has to choose frequency collapsing, thresholds, merge-gap logic, and summary columns.

- Step 6 currently recomputes spectral views instead of reading a durable TF product.
  - In the example pipeline, `sample_spectrogram_plot.py` computes a spectrogram inside the plotting script and writes only HTML.
  - That is acceptable for one-off QC views, but awkward for TF-space workflows that want the same tensor reused by multiple downstream steps.

## Candidate helper boundaries

- `compute_grid_spectrogram(sigx, ...) -> xr.DataArray`
  - Purpose: normalize the emitted dim order and window-axis naming at the first TF boundary.
  - Target output: one canonical grid TF tensor, even if the internal compute still delegates to `spectrogramx()`.

- `normalize_grid_spectrogram(spec, method=...) -> xr.DataArray`
  - Purpose: provide a reusable `wspec` boundary.
  - Candidate methods: robust z-score over `freq`, ratio-to-local-baseline, dB transform, or background-subtracted variants.

- `tf_to_spatial_batch(spec) -> np.ndarray | xr.DataArray`
  - Purpose: encapsulate the recurring transpose from spectrogram layout into `(..., AP, ML)`.
  - This would eliminate the same manual bridge currently repeated across scripts and tutorials.

- `compute_tf_spatial_summaries(spec_or_wspec, measures=...) -> xr.Dataset`
  - Purpose: batch the scalar spatial reducers and return labeled `(time_win, freq)` tensors plus structured row/column outputs.
  - This is the cleanest reusable boundary between TF tensors and later event logic.

- `reduce_tf_scores(summary_ds, reduction=...) -> xr.Dataset`
  - Purpose: standardize frequency collapsing or band summarization before thresholding.
  - Without this helper, every script will choose its own reduction and metadata.

- `score_trace_to_bouts(score, time, ...) -> EventCatalog | DataFrame`
  - Purpose: wrap `score_to_bouts()` into the stricter output contract actually needed by downstream aggregation.
  - This is where detector metadata, score columns, and optional `f0/f1/f_peak` fields should be attached.

## Suggested output artifacts by stage

- Stage 1: loaded / cleaned grid signal
  - Best fit: Zarr.
  - Reason: this is already the dominant durable signal format in current practice for early pipeline stages.

- Stage 2: dense `spec(time_win, freq, AP, ML)`
  - Best fit: Zarr.
  - Reason: high-dimensional labeled xarray data is already persisted as Zarr in the preprocess example, and TF tensors will be larger than current feature maps.
  - NetCDF: possible technically, but not current practice in this repo and less aligned with chunked reuse.

- Stage 3: dense `wspec(time_win, freq, AP, ML)`
  - Best fit: Zarr.
  - Reason: same argument as `spec`; likely reused by multiple downstream summaries and plots.

- Stage 4: reduced spatial score tensors `(time_win, freq)` and selected structured slice outputs
  - Best fit:
    - Zarr for scalar score tensors and other labeled arrays that will be reused.
    - NPZ for compact unlabeled bundles such as row/column masks or a small set of auxiliary arrays, following `rowcol_noise.py`.
  - Practical recommendation: prefer Zarr if the product is an xarray `Dataset`; reserve NPZ for script-local numeric bundles.

- Stage 5: event / bout / summary outputs
  - Best fit:
    - TSV for inspectable event tables and session summaries.
    - NPZ for compact numeric sidecars such as score traces, band-wise occupancy arrays, or threshold masks.
  - Reason: this matches current practice where inspectable summaries are TSV and compact numeric auxiliaries are NPZ.

- Stage 6: inspectable outputs
  - Best fit: HTML and PNG, as already used by the example pipeline.

Overall recommendation by product type:

- Use Zarr for reusable xarray tensors and datasets.
- Use TSV for human-inspectable tables that may be aggregated later.
- Use NPZ for compact auxiliary numeric bundles when xarray labeling is not needed.
- Do not make NetCDF the default here; it is not the current repo practice for pipeline intermediates.

## Main blockers or high-friction steps

- No canonical persisted TF intermediate exists in the example orchestration.
  - Spectrograms are currently computed inside leaf QC scripts and discarded after plotting.

- No reusable `wspec` stage exists.
  - This is the largest gap between available primitives and a clean TF-space workflow.

- Dim/order conventions still force manual bridges.
  - Signal schema, spectrogram emission order, and spatial measure expectations do not line up cleanly.

- Spatial reducers drop xarray labels.
  - The numerical path is good, but pipeline scripts must currently manage coord preservation themselves.

- Event/bout summarization is only half productized.
  - `score_to_bouts()` exists, but occupancy, band summaries, and strict event-table construction are still script responsibilities.

- Structured slice summaries such as `marginal_energy_outlier()` are not yet wrapped into a stable serialized dataset contract.
  - That makes row/column-style TF summaries likely to devolve into bespoke script outputs unless a helper boundary is added.

Bottom line: the workflow can be staged cleanly with the current Snakemake style, but only if `spec`, `wspec`, and TF-spatial summary datasets become first-class persisted intermediates. Without those boundaries, the workflow will work, but it will inherit the same script-local transpose/recompute patterns seen in the current example pipeline.
