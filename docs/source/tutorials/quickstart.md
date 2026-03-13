# Quickstart

This tutorial walks through loading data, inspecting the signal schema,
and running a basic spectral analysis. By the end you will understand
cogpy's data model and the two-layer design.

## Prerequisites

```bash
pip install -e ".[all]"
```

## Load sample data

cogpy ships with synthetic ECoG data for testing and tutorials.

```python
import cogpy
import numpy as np

sig = cogpy.datasets.load_sample()
print(sig)
# <xarray.DataArray (time: 10000, AP: 16, ML: 16)>
# Coordinates:
#   * time  (time) float64 0.0 0.001 0.002 ...
#   * AP    (AP) int64 0 1 2 ... 15
#   * ML    (ML) int64 0 1 2 ... 15
#     fs    float64 1000.0
```

The data is an `xarray.DataArray` with three dimensions:
- **time** — sample timestamps in seconds
- **AP** — anterior-posterior grid rows (0 = posterior)
- **ML** — medial-lateral grid columns (0 = medial)

The sampling rate `fs` is stored as a scalar coordinate.

## The two-layer design

cogpy separates **compute** from **I/O**:

```
cogpy.io       →  Load files into xarray
                      ↓
cogpy.core.*   →  Pure compute (filtering, spectral, detection, ...)
                      ↓
cogpy.io       →  Save results to files
```

Core functions never touch the filesystem. I/O functions never do heavy math.
Snakemake pipelines compose both.

## Compute a power spectral density

```python
from cogpy.spectral.psd import psd_multitaper

# Extract one channel's time series
channel = sig.sel(AP=8, ML=8).values  # shape: (10000,)
fs = float(sig.fs)

psd, freqs = psd_multitaper(channel, fs=fs)
```

The PSD function is **pure compute** — it accepts a numpy array and returns
numpy arrays. No files, no xarray required at this level.

## Spectral features

cogpy's spectral features follow the **PSD-first convention**: compute the PSD
once, then derive multiple features from it.

```python
from cogpy.spectral.features import (
    band_power,
    spectral_entropy,
    line_noise_ratio,
)

gamma_power = band_power(psd, freqs, band=(30, 100))
entropy = spectral_entropy(psd, freqs)
line_noise = line_noise_ratio(psd, freqs, line_freq=60.0)
```

All spectral features accept `(..., freq)` arrays — batch dimensions are
leading, frequency is always the last axis.

## Detect spectral bursts

cogpy's detection framework wraps transforms and detectors into reproducible
pipelines.

```python
from cogpy.detect import BURST_PIPELINE

events = BURST_PIPELINE.run(sig)
print(events)
# EventCatalog with columns: event_id, t, freq, AP, ML, ...
```

The pipeline computes a spectrogram internally, finds local maxima, and
returns an `EventCatalog` — a pandas DataFrame wrapper with standardized
columns.

## Next steps

- {doc}`spectral-analysis` — deeper dive into PSD, spectrograms, and spectral features
- {doc}`bad-channel-detection` — identify and interpolate bad electrodes
- {doc}`spatial-measures` — spatial grid characterization (Moran's I, gradient anisotropy)
