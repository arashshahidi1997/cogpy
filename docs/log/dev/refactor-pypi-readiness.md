# Refactor Log: PyPI Readiness

Date: 2026-03-19
Status: In progress

## Goal

Prepare cogpy for initial PyPI publication (v0.1.0) with a clean, standard
API surface. Backward compatibility is NOT a constraint.

---

## Pass 1 — Metadata & cleanup (DONE, commit 8312204)

| Change | Detail |
|--------|--------|
| `pyproject.toml` | Added description, keywords, classifiers, project URLs |
| Dependencies | Removed `pyts`, `jinja2`, `quantities` (unused core deps) |
| LICENSE | Added MIT LICENSE file |
| `__version__` | Added via `importlib.metadata` |
| Hardcoded paths | Removed ad-hoc `test_ecog_npix()` from `ecog_io.py` |
| `quantities` import | Dropped; `seconds_to_samples` now accepts int/float (duck-types via `float()` for other numeric types) |
| `ssa.py` | Removed archive module; dropped `numba` from core deps. SSA noted as future addition in `spectral-conventions.md` |
| Docs cleanup | Removed duplicate diataxis stubs (`docs/explanation/`, `docs/how-to/`, `docs/tutorials/`). Updated `primitives.md` with import paths/signatures. Added `howto/compose-artifact-analysis.md` |

## Pass 1.5 — Flatten core/ (DONE, commit 387c3d4)

| Change | Detail |
|--------|--------|
| Structure | Moved all 16 subpackages from `cogpy/core/` to `cogpy/` top level |
| `__init__.py` | Removed `sys.modules` hack; subpackages are now real paths |
| Shim dirs | Removed all shim `__init__.py` and `.pyi` stub files |
| Imports | Replaced `cogpy.core.X` with `cogpy.X` across 367 files |
| Tests | Merged `tests/core/` into `tests/` |
| CLAUDE.md | Updated architecture section to reflect flat structure |
| Result | 547 tests pass |

## Pass 2 — API surface cleanup

### 2a. Remove deprecated preprocess modules

**Decision: REMOVE.** No backward compatibility needed.

| Module | Status | Replacement |
|--------|--------|-------------|
| `preprocess/channel_feature.py` | Deprecated, emits warning | `preprocess.badchannel.pipeline` |
| `preprocess/channel_feature_functions.py` | Deprecated, emits warning | `preprocess.badchannel.channel_features` |
| `preprocess/detect_bads.py` | Deprecated, emits warning | `preprocess.badchannel.badlabel` |

Callers to update: `tests/preprocess/test_channel_features.py` (remove),
`tests/preprocess/test_badchannel_phase_a_contracts.py` (update imports),
workflow scripts `02_feature.py`, `03_badlabel.py`.

### 2b. Remove deprecated spectral functions

**Decision: REMOVE** from `spectral/multitaper.py`.

| Function | Replacement |
|----------|-------------|
| `assign_freqs` | `sliding_core.running_blockwise_xr` |
| `specx_coords` | `sliding_core.running_blockwise_xr` |
| `multitaper_spectrogram` | `spectrogramx` |
| `multitaper_fftgram` | `spectrogramx` |
| `running_spectral` | `sliding_core.running_blockwise_xr` |

Callers: `tests/spectral/test_spectral.py` (remove deprecated test cases).

### 2c. Remove archive modules

**Decision: REMOVE.**

| Module | Reason |
|--------|--------|
| `spectral/superlet.py` | Archive, zero callers |
| `spectral/spectrogram_burst.py` | Pipeline-specific, not atomic |
| `plot/_legacy/` directory | Zero callers, deprecated |

### 2d. Fix ghost lazy-loader reference

`spectral/__init__.py` declares `bivariate_spectral` but the file doesn't
exist. Remove from lazy loader.

### 2e. Remove private helpers from public exports

Remove `_apply_full_array` and `_fs_scalar` from `preprocess/filtering/__init__.py`
re-exports. They are used internally via relative imports from `_utils.py` —
no need to be in `__all__` or public surface.

Also remove `preprocess/filtx.py` (backward-compat shim). Callers should use
`preprocess.filtering` directly.

### 2f. Rename camelCase to snake_case

| Old name | New name | Module |
|----------|----------|--------|
| `AR_whiten` | `ARWhiten` (class, keep PascalCase) | `spectral/whitening.py` |
| `AR_whitening` | `ar_whitening` | `spectral/whitening.py` |
| `AR_yule` | `ar_yule` | `spectral/whitening.py` |

Note: `FASLT` is inside `superlet.py` which is being removed, so no rename needed.

### 2g. Clean up preprocess/__init__.py

Remove legacy module declarations from lazy loader. Only export:
`filtering`, `interpolate`, `linenoise`, `resample`, `badchannel`.

---

## Pass 3 — Publishing infra (DONE)

| Change | Detail |
|--------|--------|
| Makefile | Added `make build` (sdist + wheel) and `make publish` (twine upload) targets |
| `llms.txt` | Agent-facing summary at repo root pointing to key docs |
| `package-map.md` | Full module tree with one-line descriptions in `docs/source/explanation/` |
| `pyproject.toml` | Fixed TOML ordering (`dependencies` before `[project.urls]`) |
| Build test | `cogpy-0.1.0-py3-none-any.whl` builds successfully (212 files, no `core/` refs) |
