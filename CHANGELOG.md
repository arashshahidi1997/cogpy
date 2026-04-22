# Changelog

All notable changes to cogpy are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [0.3.0] — 2026-04-22

### Added
- `cogpy.preprocess.interpolate.interpolate_bads_coords` — coordinate-based bad
  channel interpolation. Takes explicit `(x, y)` electrode coordinates and uses
  Delaunay triangulation of good channels to interpolate bad ones. Supports
  arbitrary non-uniform layouts (ECoG grids with hemispheric gaps, checkerboard
  arrays, depth probes) without hard-coded device geometry. Vectorized over
  trailing dimensions.
- `cogpy.preprocess.interpolate.interpolate_bads_xarray` — xarray wrapper that
  reads `x`, `y` non-dimension coordinates from a DataArray (e.g. loaded from
  BIDS `_electrodes.tsv`) and returns a DataArray with interpolated values.
  Dimension order and metadata preserved.
- `cogpy.preprocess.interpolate.interpolate_bads_1d` — linear (1D) interpolation
  for depth probes where electrodes lie along a single axis. Uses `np.interp`
  along the probe axis with nearest-neighbor extrapolation at endpoints. Needed
  because `scipy.interpolate.griddata` fails on collinear points.
- `optimal_ref` parameter on `cogpy.decomposition.match.match_factors` — when
  `True` (default) auto-selects the reference recording whose transitive
  matches are best overall; when `False` uses index 0.
- `cogpy.decomposition.pca.compute_similarity_matrix` — pairwise factor
  similarity between two `SpatSpecDecomposition` instances, now a documented
  public entry point.
- Tutorial: *Cross-Recording Factor Matching* — walks through similarity
  matrix → Hungarian assignment → multi-recording matching → centroid, with
  a sketch of the two-stage within/across-animal workflow.
- API reference pages for 7 subpackages that were previously missing from
  the docs index: `decomposition`, `triggered`, `regression`, `wave`,
  `plot`, `depth_probe`, `model`.
- `Examples` sections on 10 most-used functions (filtering, spectral,
  detection, measures, triggered).
- `See Also` cross-references across filtering, spectral, detection,
  and triggered function groups.
- Expanded `__init__.py` module docstrings for all lazy-loaded subpackages
  (listings of submodules / public names).
- Lab-internal module notes on `cogpy.io` modules that assume Bhatt-Lab
  recording conventions (`ecog_io`, `ecephys_io`, `ieeg_io`,
  `ieeg_sidecars`, `xml_io`, `xml_anat_map`, `load_meta`).
- README and docs landing page now clarify the `ecogpy` (PyPI) vs `cogpy`
  (import) naming.

### Fixed
- `cogpy.decomposition.match.get_fac_match_df` now uses the optimal
  reference index instead of a hard-coded `match_fac_ref[1]` — the
  previous behavior silently ignored `optimal_refrec`'s result.
- `cogpy.decomposition.match.get_similx_flat` now iterates over all
  diagonal self-similarity matrices; previously hard-coded
  `simil_arr[2, 2]`, which crashed for `nrec != 3`.
- `cogpy.spectral.whitening.ARWhiten.fit` now accepts xarray input and
  uses the `method=` keyword for `np.percentile` (was positional).
- BIDS iEEG sidecar reader is now spec-compliant: channel count resolution,
  electrode coordinate loading, grid dimension inference from
  `_electrodes.tsv` when the JSON lacks `RowCount`/`ColumnCount`.
- `cogpy.io.ecog_io` module docstring replaced the blank-template
  header with a real description and a matrix-convention note.
- `cogpy.wave.features` — added module docstring + function docstrings;
  removed empty stub functions that were previously `pass` bodies.

### Changed
- `perievent_epochs` in `cogpy.brainstates.intervals` vectorized — roughly
  **220× faster** on typical workloads. No API change.
- `cogpy.decomposition.match.match_factors` accepts the new `optimal_ref`
  kwarg (defaults to `True`; was effectively ignored in 0.2.0).

### Removed
- Stale `bids_reader` tutorial reference.
- Empty stub functions in `cogpy.wave.features` (`peak`, `convexity`,
  `laplace`, `contour`, `split_waves`, etc. — none had implementations).
- `src/cogpy/workflows/preprocess/scripts/plot_features.ipynb` — unused.

## [0.1.0] — 2026-03-19

First release prepared for PyPI.

### Added
- `cogpy.triggered` — event-locked analysis: `triggered_average`, `triggered_std`,
  `triggered_median`, `triggered_snr`, `estimate_template`, `fit_scaling`,
  `subtract_template`
- `cogpy.regression` — linear model primitives: `lagged_design_matrix`,
  `event_design_matrix`, `ols_fit`, `ols_predict`, `ols_residual`
- `cogpy.measures.comparison` — before/after validation metrics: `snr_improvement`,
  `residual_energy_ratio`, `bandpower_change`, `waveform_residual_rms`
- `cogpy.events.match` — event matching and lag estimation: `match_nearest`,
  `match_nearest_symmetric`, `estimate_lag`, `estimate_drift`, `event_lag_histogram`
- MIT LICENSE file
- `__version__` attribute via `importlib.metadata`
- `llms.txt` for agent discoverability
- `docs/source/explanation/primitives.md` — full primitive catalog with imports
- `docs/source/explanation/package-map.md` — module tree overview
- `docs/source/howto/compose-artifact-analysis.md` — composition patterns
- `make build` / `make publish` targets

### Changed
- **Flat package structure:** all subpackages moved from `cogpy.core.*` to
  `cogpy.*` (e.g. `from cogpy.spectral.psd import psd_multitaper`).
  The `core/` indirection layer and `sys.modules` hack are removed.
- `AR_whiten` renamed to `ARWhiten` (PascalCase for classes)
- `AR_whitening` renamed to `ar_whitening`
- `AR_yule` renamed to `ar_yule`
- `preprocess.filtx` shim removed; use `preprocess.filtering` directly

### Removed
- `preprocess/channel_feature.py`, `channel_feature_functions.py`,
  `detect_bads.py` — use `preprocess.badchannel` instead
- `spectral/ssa.py` — incomplete archive (SSA noted as future addition)
- `spectral/superlet.py` — archive, zero callers
- `spectral/spectrogram_burst.py` — pipeline-specific, not atomic
- `plot/_legacy/` directory — zero callers
- Deprecated spectral functions: `assign_freqs`, `specx_coords`,
  `multitaper_spectrogram`, `multitaper_fftgram`, `running_spectral`
- Dependencies: `pyts`, `jinja2`, `quantities`, `numba` (unused or archive-only)
