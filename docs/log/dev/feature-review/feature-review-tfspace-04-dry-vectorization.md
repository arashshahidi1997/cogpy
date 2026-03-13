# CogPy TF-Space DRY and Vectorization Review

## Likely performance-critical operations

- Dense spectrogram construction over grid signals is already the main compute boundary. `spectrogramx()` emits labeled TF tensors, but its native output order is `[non-time dims] + ["freq", "time"]`, so a canonical grid input produces `(ML, AP, freq, time)` rather than a direct TF-space batch layout (`src/cogpy/core/spectral/specx.py:153-220`).
- TF-to-spatial reductions are the next hot path. The spatial measures themselves are batch-friendly and already reduce `(..., AP, ML) -> ...` without explicit Python loops for the common scalar summaries (`moran_i`, `gradient_anisotropy`, `spatial_kurtosis`, `spatial_noise_concentration`; `src/cogpy/core/measures/spatial.py:73-152`, `386-452`, `455+`).
- Event extraction from TF summaries is cheap only after the score is reduced to 1D. `score_to_bouts()` is strictly a 1D time-series helper, so every TF-space pipeline still needs a custom reduction from `(time_win, freq)` or `(..., time_win, freq)` into a time trace before bout logic can run (`src/cogpy/core/detect/utils.py:155-220`).
- Pairwise spatial/frequency analyses can become expensive quickly. `spatial_coherence_profile()` still loops over every channel pair in Python after the multitaper FFT, so it is not ready for TF-space-style batching or repeated QC use on large grids (`src/cogpy/core/measures/spatial.py:210-326`).

## Existing vectorized strengths

- The core spatial summary functions are already designed for batched `(..., AP, ML)` input and use reshape-plus-NumPy reductions instead of loops. That is the strongest existing foundation for TF-space QC (`src/cogpy/core/measures/spatial.py:115-149`, `358-382`, `414-420`, `449-452`).
- The repo already contains reusable xarray windowing infrastructure that could support TF-space code without bespoke loops:
  - `running_blockwise_xr()` for blockwise window transforms with coord/attr propagation (`src/cogpy/core/utils/sliding_core.py:320-396`)
  - `running_reduce_xr()` for window reductions (`src/cogpy/core/utils/sliding_core.py:399+`)
  - `running_measure_sane()` / `xroll_apply()` as `xr.apply_ufunc`-based rolling wrappers with chunk-aware Dask handling (`src/cogpy/core/utils/sliding.py:220-274`, `277-418`)
- The newer bad-channel feature path already uses these wrappers instead of manual loops. `extract_channel_features_xr()` applies scalar and spectral features with `xr.apply_ufunc`, and windowed features with `running_blockwise_xr()` / `running_reduce_xr()` (`src/cogpy/core/preprocess/badchannel/channel_features.py:297-379`).
- Spatial neighborhood normalization is already batched over windows in the bad-channel pipeline by flattening spatial nodes and normalizing `(nodes, W)` in one pass instead of looping per window (`src/cogpy/core/preprocess/badchannel/pipeline.py:180-234`).

## Repeated glue / transpose / wrapping patterns

- The same manual transpose bridge is repeated in docs and implied in code:
  - `spec.values.transpose(2, 3, 0, 1)` in the how-to for batched spatial analysis (`docs/source/howto/batch-spatial-analysis.md:33-36`)
  - `spec.values.transpose(2, 3, 0, 1)` again in the spatial tutorial (`docs/source/tutorials/spatial-measures.md:90-99`)
  - `spectrogramx()` itself emits `(ML, AP, freq, time)` while the spatial reducers want `(..., AP, ML)` and the schema constants anticipate `(time_win, AP, ML, freq)` (`src/cogpy/core/spectral/specx.py:187-220`; `src/cogpy/datasets/schemas.py:55-69`)
- There is repeated xarray-to-NumPy boundary churn:
  - tutorials use `spec.values` before spatial reduction (`docs/source/howto/batch-spatial-analysis.md:35`, `docs/source/tutorials/spatial-measures.md:94`)
  - `detect_blob_candidates()` repeatedly slices a single channel, transposes, computes Dask eagerly if needed, converts to NumPy, and then wraps back into a temporary `xr.DataArray` for blob detection (`src/cogpy/core/spectral/spectrogram_burst.py:229-246`)
- Time-window naming and order are inconsistent enough to force caller glue:
  - enforced grid/window schema is `(time_win, ML, AP)` (`src/cogpy/datasets/schemas.py:55-56`)
  - enforced spectrogram schema is lower-case `(ml, ap, time, freq)` (`src/cogpy/datasets/schemas.py:59-60`)
  - `spectrogramx()` still uses `"time"` for window centers (`src/cogpy/core/spectral/specx.py:197-200`)
- There is duplicated robust-z logic in at least three places:
  - `zscorex(..., robust=True)` (`src/cogpy/core/preprocess/filtering/normalization.py:7-36`)
  - `normalize_windowed_features(..., robust=True)` (`src/cogpy/core/preprocess/badchannel/feature_normalization.py:11-89`)
  - `marginal_energy_outlier()` local `_zscore()` (`src/cogpy/core/measures/spatial.py:362-374`)
- Local spatial robust-z / neighborhood baseline logic is duplicated in old and new code:
  - deprecated `local_robust_zscore` / `local_robust_zscore_dask` (`src/cogpy/core/preprocess/channel_feature_functions.py:78-139`)
  - new `local_robust_zscore_grid` plus neighborhood median/MAD helpers (`src/cogpy/core/preprocess/badchannel/spatial.py:53-148`)

## Missing reusable helpers

- Missing canonical TF batch adapter. The repo needs one helper that takes a spectrogram-like `xr.DataArray` and returns a canonical labeled TF-space layout for spatial reducers, instead of repeating `values.transpose(...)`.
- Missing reusable TF normalization constructor. There is no core helper that turns `spec -> wspec` with explicit method choices like robust-z-over-freq, dB, local-flank ratio, or aperiodic-subtracted output. Current normalization is split across:
  - generic z-score helpers (`src/cogpy/core/preprocess/filtering/normalization.py:7-36`)
  - windowed feature normalization (`src/cogpy/core/preprocess/badchannel/feature_normalization.py:11-89`)
  - pipeline-specific spectrogram normalization (`src/cogpy/core/spectral/spectrogram_burst.py:147-156`)
- Missing xarray wrappers for spatial reductions. The numerical reducers are fine, but there is no `xr.apply_ufunc` layer that preserves `(time_win, freq)` coords and returns an `xr.Dataset` rather than plain NumPy arrays or dicts.
- Missing reusable TF-to-event reducers. `score_to_bouts()` is usable only after the caller decides how to collapse frequency, threshold, and annotate metadata. There is no shared helper for:
  - collapsing `(time_win, freq)` into time traces or per-band traces
  - computing occupancy / duration summaries from `Intervals`
  - assembling a strict `datasets.schemas.EventCatalog`
- Missing validators/coercers for the schema constants most relevant to TF-space work. `DIMS_GRID_WINDOWED_SPECTRUM` and friends exist, but there is no validator/coercer path comparable to `validate_ieeg_grid()` or `validate_spectrogram4d()` (`src/cogpy/datasets/schemas.py:63-76`).

## Highest-value vectorization opportunities

- Replace per-channel blob-detection loops with a batched wrapper. `detect_blob_candidates()` currently loops over `channel`, materializes each slice, rewraps it, and then loops again over DataFrame rows (`src/cogpy/core/spectral/spectrogram_burst.py:232-256`). Even if the blob detector itself stays 2D, an `xr.apply_ufunc` or reshape-based batch wrapper around `(channel, time, freq)` would remove repeated transpose/compute/wrap overhead.
- Generalize the bad-channel normalization pattern to TF-space. `normalize_features_from_raw()` already flattens `(AP, ML, W)` into `(nodes, W)` and applies neighborhood medians/MADs once (`src/cogpy/core/preprocess/badchannel/pipeline.py:215-232`). The same reshape-first strategy can be extended to `(time_win, freq, AP, ML)` by flattening batch dims and spatial dims separately, then unflattening back.
- Replace `narrowband_ratio()`'s Python loop over frequency bins with a windowed/rolling implementation. It currently iterates `for i in range(nf)` and recomputes flank masks per bin (`src/cogpy/core/spectral/features.py:314-320`). For dense TF-space use on many slices, this becomes a clear bottleneck. A rolling median or precomputed frequency-neighborhood index map would remove Python overhead.
- Treat `aperiodic_exponent()` and `fooof_periodic()` as batch wrappers around a core 1D fitter, then expose an xarray `apply_ufunc` path. Both currently rely on `np.apply_along_axis`, which is fine for PSD vectors but does not preserve labels or Dask friendliness for large batched TF reductions (`src/cogpy/core/spectral/features.py:456-490`, `501-540`).
- Rework `spatial_coherence_profile()` if it is expected to participate in QC pipelines. The current pairwise Python loop (`for pair_k, (i, j) in enumerate(zip(iu, ju))`) is the opposite of TF-space-ready (`src/cogpy/core/measures/spatial.py:315-319`). A vectorized pair-selection/bincount accumulation scheme would be much more scalable.
- Add xarray gufunc wrappers around the existing scalar spatial reducers. This is a high-value, low-risk change because the underlying NumPy math is already vectorized; only the label-preserving wrapper is missing.

## Recommendations for keeping TF-space code maintainable

- Standardize one canonical in-memory TF schema and make all wrappers target it. The least disruptive choice for new TF-space QC code is a labeled xarray form equivalent to `(time_win, freq, AP, ML)` or `(time_win, freq, ML, AP)`, plus one coercer that accepts `spectrogramx()` output and renames/reorders once.
- Add two thin public helpers instead of repeating glue in scripts and docs:
  - `coerce_grid_tf(...)` or `tf_to_spatial_batch(...)`
  - `reduce_tf_spatial(...)` returning an `xr.Dataset`
- Reuse existing wrapper infrastructure instead of inventing another TF-specific loop framework. The strongest candidate is `running_blockwise_xr()` / `running_measure_sane()` because those already solve coord propagation, output dim naming, and some Dask behavior (`src/cogpy/core/utils/sliding_core.py:320-396`; `src/cogpy/core/utils/sliding.py:277-418`).
- Consolidate normalization logic into one small set of core primitives:
  - robust z-score along a named xarray dim
  - local spatial robust z-score over `(AP, ML)`
  - local spectral baseline ratio over `freq`
  Then call those from bad-channel, spectrogram QC, and line-noise code rather than re-embedding median/MAD arithmetic.
- Keep xarray until the last unavoidable boundary. Right now docs normalize the NumPy path by teaching `spec.values.transpose(...)`; that is expedient but encourages repeated label loss. The maintainable path is `xr.apply_ufunc` wrappers around the existing NumPy kernels so coords survive across TF-space summaries.
- Treat TF-to-event conversion as an explicit reusable stage. A small helper that goes `score_da(time_win[, freq]) -> EventCatalog + occupancy summaries` would prevent every downstream QC script from re-implementing reduction, threshold, and metadata attachment.
