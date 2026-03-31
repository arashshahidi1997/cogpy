# cogpy

Atomic, composable primitives for ECoG / iEEG signal processing.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**[Documentation](https://arashshahidi1997.github.io/cogpy/)** · **[Repository](https://github.com/arashshahidi1997/cogpy)**

## What it is

cogpy provides small, pure, domain-agnostic operators for electrophysiology
signal processing. It is **not** a pipeline framework — high-level
orchestration belongs in Snakemake pipelines, notebooks, or project repos.

### Capability areas

| Area | Key functions |
|------|--------------|
| **Event detection** | `ThresholdDetector`, `BurstDetector`, `score_to_bouts` |
| **Event matching** | `match_nearest`, `estimate_lag`, `estimate_drift` |
| **Triggered analysis** | `perievent_epochs`, `triggered_average`, `estimate_template`, `subtract_template` |
| **Regression** | `lagged_design_matrix`, `ols_fit`, `ols_residual` |
| **Spectral** | `psd_multitaper`, `spectrogramx`, `band_power`, `ftest_line_scan` |
| **Spatial measures** | `moran_i`, `gradient_anisotropy`, `csd_power` |
| **Filtering** | `bandpassx`, `cmrx`, `gaussian_spatialx`, `zscorex` |
| **Validation** | `snr_improvement`, `bandpower_change`, `residual_energy_ratio` |

## Install

```bash
pip install ecogpy               # core dependencies only
pip install ecogpy[viz]          # + matplotlib, holoviews, panel
pip install ecogpy[all]          # all optional dependencies
pip install -e .                # editable dev install
```

## Quick start

```python
from cogpy.detect import ThresholdDetector
from cogpy.brainstates.intervals import perievent_epochs
from cogpy.triggered import estimate_template, subtract_template
from cogpy.spectral.psd import psd_multitaper
from cogpy.measures.comparison import bandpower_change

# Detect events
detector = ThresholdDetector(threshold=3.0, direction="positive")
catalog = detector.detect(signal)

# Extract epochs and estimate template
epochs = perievent_epochs(signal, catalog.df["t"].values, fs, pre=0.01, post=0.01)
template = estimate_template(epochs, method="median")

# Subtract and validate
cleaned = subtract_template(signal, event_samples, template.values)
psd_before, freqs = psd_multitaper(signal.values, fs)
psd_after, _ = psd_multitaper(cleaned.values, fs)
delta = bandpower_change(psd_before, psd_after, freqs, band=(100, 140))
```

## Package structure

All subpackages live directly under `cogpy/` — no indirection layers.

```
cogpy/
├── detect/          Event detection (threshold, burst, ripple)
├── events/          EventCatalog, matching, lag estimation
├── triggered/       Epoch extraction, triggered stats, template subtraction
├── regression/      Design matrices, OLS fit/predict/residual
├── spectral/        PSD, spectrogram, coherence, multitaper, features
├── measures/        Spatial, temporal, and comparison metrics
├── preprocess/      Filtering, bad channel detection, interpolation
├── decomposition/   PCA, varimax rotation
├── brainstates/     Perievent epochs, interval operations
├── plot/            Static (matplotlib) and interactive (HoloViews) viz
├── io/              File I/O for ECoG/iEEG formats
├── datasets/        Sample data loaders
├── cli/             CLI entry points
└── workflows/       Snakemake preprocessing pipelines
```

## Development

```bash
make check          # format + lint + typecheck + tests
make format         # black .
make lint           # ruff check . --fix
make tests          # pytest
make docs           # build Sphinx docs
make build          # build sdist + wheel
```

## Documentation

- [Primitive catalog](https://arashshahidi1997.github.io/cogpy/explanation/primitives/) — all operators with imports and signatures
- [Package map](https://arashshahidi1997.github.io/cogpy/explanation/package-map/) — module tree overview
- [Composition patterns](https://arashshahidi1997.github.io/cogpy/howto/compose-artifact-analysis/) — how to assemble primitives

## License

MIT — see [LICENSE](LICENSE).
