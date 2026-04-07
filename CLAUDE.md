# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

cogpy is a Python toolkit for ECoG / iEEG signal processing and analysis. It provides:
- Reusable compute primitives (filtering, spectral analysis, event detection, bad-channel ID)
- Structured I/O for electrophysiology formats (binary LFP, Zarr, BIDS-iEEG)
- Reproducible Snakemake preprocessing pipelines
- A backend API for visualization frontends (e.g. TensorScope)

## Common Commands

```bash
# Install (editable dev install)
pip install -e .

# Quality checks
make check          # format + lint + typecheck + tests (all-in-one)
make format         # black .
make lint           # ruff check . --fix
make typecheck      # mypy src
make tests          # pytest

# Run a single test
python -m pytest tests/path/to/test_file.py::test_name

# Test markers
python -m pytest -m integration    # integration tests (slower, external deps)
python -m pytest -m benchmark      # performance benchmarks

# Documentation
make docs           # Build Sphinx docs to docs/build/html
make serve          # Serve docs at http://0.0.0.0:8000

# Data management (datalad)
make save SAVE_MSG="message"
make push
```

Note: The Makefile uses a shared conda Python at `/storage/share/python/environments/Anaconda3/envs/cogpy/bin/python`. Override with `PYTHON=python make tests`.

## Architecture

### Two-Layer Design

- **`cogpy.core`** — Pure compute on in-memory structures. File-agnostic and testable.
- **`cogpy.io`** — File reading/writing, sidecar management, format translation.

Never mix I/O and compute in the same function. Pipelines compose `cogpy.io` (load/save) with `cogpy.core` (transform).

### Core Data Model

Primary data structure is `xarray.DataArray` with standardized dimensions defined in `cogpy.core.base.ECoGSchema`:
- Time: `"time"`
- Grid ECoG spatial: `"AP"` (anterior-posterior), `"ML"` (medial-lateral)
- Flat ECoG channel: `"ch"`
- Sampling rate: accessible as `sig.fs` (0D coordinate or attribute)

### Key Modules

| Path | Purpose |
|------|---------|
| `core/base.py` | Signal schemas (`ECoGSchema`) and thin wrapper classes |
| `core/preprocess/filtering/` | xarray filters: `temporal.py` (IIR), `spatial.py` (grid), `reference.py` (CMR), `normalization.py` (z-score) |
| `core/preprocess/badchannel/` | Canonical bad-channel pipeline (features → spatial norm → DBSCAN) |
| `core/preprocess/filtx.py` | Backward-compat shim — re-exports from `filtering/` |
| `core/detect/` | Event detection (unified detector interface) |
| `core/events/` | EventCatalog data structure |
| `core/spectral/` | Spectral analysis (multitaper, whitening, etc.) |
| `core/measures/` | Spatial (`moran_i`, `gradient_anisotropy`) and temporal measures |
| `core/plot/hv/` | Interactive HoloViews/Panel visualization (`grid_movie`, `multichannel_view`, `add_time_hair`, `TopoMap`, `OrthoSlicerRanger`) |
| `core/plot/` | Static matplotlib/plotly helpers (`decomposition`, `specgram_plot`, `time_plot`) |
| `core/plot/_legacy/` | Deprecated viz modules (retained for reference) |
| `core/tensorscope/` | Legacy TensorScope Panel app (migrated to standalone React+TS) |
| `io/ecog_io.py` | ECoG-specific file I/O |
| `io/ieeg_io.py` | iEEG BIDS I/O |
| `cli/` | Thin CLI wrappers (keep minimal) |
| `workflows/` | Snakemake pipelines (packaged as data) |
| `datasets/` | Sample dataset loading utilities |

### Import Strategy

- `cogpy.__init__.py` uses `lazy_loader.attach()` for lazy submodule loading
- Re-exports `cogpy.core.*` as `cogpy.*` for backward compatibility
- Legacy modules (`channel_feature`, `channel_feature_functions`, `detect_bads`) emit `DeprecationWarning` on import; use `badchannel` instead
- **Filtering:** prefer `cogpy.preprocess.filtering` (new canonical path); `cogpy.preprocess.filtx` is a backward-compat shim

### Visualization (`core/plot/`)

- **`core/plot/hv/`** — All interactive HoloViews/Panel components. Key files:
  - `xarray_hv.py` — `grid_movie`, `grid_movie_with_time_curve`, `multichannel_view`, `add_time_hair`, `selected_channel_curve`
  - `time_player.py` — `TimeHair`, `AxisHair`, `PlayerWithRealTime`
  - `topomap.py` — AP×ML heatmap viewer
  - `multichannel_viewer.py` — stacked-trace viewer for `(time, ch)` signals
  - `orthoslicer.py` — interactive orthoslicer with linked time-window
  - `processing_chain.py` — filter pipeline UI (CMR, bandpass, notch, z-score)
  - `ecog_viewer.py` — `ChannelGridSelector` + full ECoG viewer app
- **`core/plot/`** root — Static matplotlib/plotly helpers (`decomposition.py`, `specgram_plot.py`, `time_plot.py`)
- **`core/plot/_legacy/`** — Deprecated modules retained for reference

### TensorScope Migration

TensorScope was originally a Panel/HoloViews app inside `cogpy.core.plot.tensorscope`.
It has been moved to `cogpy.core.tensorscope` and migrated to a standalone React +
TypeScript project. The cogpy subpackage and its archived spec docs
(`docs/source/explanation/plot/_archive/tensorscope-*.md`) are retained as
historical reference. New visualization work targets the standalone frontend using
cogpy's public API as backend.

### CLI Entry Points

- `cogpy-preproc` — Preprocessing pipeline (`cogpy.cli.preprocess:main`)
- `tensorscope` — Legacy CLI (`cogpy.core.tensorscope.cli:main`)

## Conventions

- **Python ≥ 3.10** required
- **Docstrings:** NumPy style (parsed by `sphinx.ext.napoleon`)
- **Tests:** Mirror source structure under `tests/`; use pytest fixtures
- **Snakemake rules:** Use `cogpy.io` to load/save, `cogpy.core` for transforms; keep rules as thin orchestrators
- **Bad channel detection:** Use canonical `cogpy.core.preprocess.badchannel` stack, not legacy `channel_feature_functions` / `detect_bads`
- **Filtering:** Use `cogpy.preprocess.filtering` (split into `temporal`, `spatial`, `reference`, `normalization`); `filtx` is a compat shim
- **Batch dimensions:** Spatial measures accept `(..., AP, ML)`, spectral features accept `(..., freq)`, temporal measures accept `(..., time)`
