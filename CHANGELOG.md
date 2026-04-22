# Changelog

All notable changes to cogpy are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

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
