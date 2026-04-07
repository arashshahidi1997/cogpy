# How to run spatial measures on spectrograms

Spatial measures like `gradient_anisotropy`, `moran_i`, and
`marginal_energy_outlier` accept `(..., AP, ML)` batch dimensions. This
means you can apply them to an entire time-frequency spectrogram without
any Python loops.

## Setup

```python
import numpy as np
from cogpy.spectral.specx import spectrogramx
from cogpy.datasets.schemas import coerce_grid_windowed_spectrum
from cogpy.measures.spatial import (
    gradient_anisotropy,
    moran_i,
    marginal_energy_outlier,
)
```

## Compute the spectrogram

```python
# sig: xarray.DataArray with dims (time, ML, AP)
spec = spectrogramx(sig, nperseg=512, noverlap=384, bandwidth=4.0)
# spec dims: (ML, AP, freq, time)
```

## Coerce to compute-pipeline schema

Use `coerce_grid_windowed_spectrum` to transpose and rename in one step
(handles `time` → `time_win` and any dim order):

```python
spec = coerce_grid_windowed_spectrum(spec)
# spec dims: (time_win, AP, ML, freq)
```

Spatial measures expect `(..., AP, ML)` — AP and ML must be the last two
axes. Extract the numpy array with `(time_win, freq)` as batch dims:

```python
# Transpose: (time_win, AP, ML, freq) → (time_win, freq, AP, ML)
data = spec.values.transpose(0, 3, 1, 2)
print(data.shape)  # e.g., (300, 257, 16, 16)
```

## Apply measures (vectorized)

```python
# Each returns shape (time_win, freq) — no loops needed
aniso = gradient_anisotropy(data)                       # ~instant
energy = marginal_energy_outlier(data)                  # ~instant
moran_ap = moran_i(data, adjacency="ap_only")          # ~1-2 seconds
moran_ml = moran_i(data, adjacency="ml_only")          # ~1-2 seconds
```

## Performance notes

| Measure | Mechanism | Speed (300x257x16x16) |
|---------|-----------|----------------------|
| `gradient_anisotropy` | `np.diff` + `np.nanmean` | < 0.1s |
| `marginal_energy_outlier` | `np.nansum` + `np.nanmedian` | < 0.1s |
| `moran_i` | Batched matmul `xc @ W` | ~1-2s |

`moran_i` is slower because it does a matrix-vector product per batch
element (256x256 adjacency matrix). The matrix is built once and cached
via `lru_cache`.

## Interpreting results

```python
# Gradient anisotropy as a function of frequency and time
# aniso[t, f] > 0: row-striped pattern at that time-freq bin
# aniso[t, f] < 0: column-striped pattern
# aniso[t, f] ~ 0: isotropic (normal)

# Find time-freq bins with strong column-striped artifacts
artifact_mask = aniso < -2.0  # strong ML-dominant gradient

# Marginal energy: which columns are outliers at each time-freq bin?
bad_cols = energy["col_outlier"]  # (time_win, freq, ML) boolean
```

## Combining with spectral features

You can combine spatial and spectral diagnostics:

```python
from cogpy.spectral.features import narrowband_ratio

# Per-channel narrowband ratio
psd = np.mean(spec.values ** 2, axis=2)  # average over time → (AP, ML, freq)
nb = narrowband_ratio(psd, freqs, flank_hz=5.0)  # (AP, ML, freq)

# Spatial pattern at narrowband-peak frequencies
peak_mask = nb > 10  # strong narrowband peaks
# ... correlate with spatial outlier patterns
```
