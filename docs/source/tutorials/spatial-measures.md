# Spatial Grid Measures

This tutorial introduces cogpy's spatial characterization measures for 2D
electrode grids. These measures detect non-physiological spatial patterns
(striped artifacts, checkerboard noise) that per-channel temporal measures
miss.

## Grid convention

All spatial measures expect the grid as the **last two axes**:

```
grid : (..., AP, ML)
```

- **AP** — anterior-posterior (rows)
- **ML** — medial-lateral (columns)
- **...** — optional batch dimensions (time windows, frequency bins)

2D input returns a Python float. Higher-dimensional input returns an array
with the spatial axes reduced.

## Moran's I — spatial autocorrelation

Moran's I measures how similar neighboring electrodes are:

```python
from cogpy.measures.spatial import moran_i

# Single grid snapshot: (AP, ML)
grid = sig.sel(time=0.5).values  # shape: (16, 16)

I = moran_i(grid, adjacency="queen")
# I ~ +1: spatially smooth (biological)
# I ~  0: spatially random (independent noise)
# I ~ -1: anti-correlated (referencing artifact)
```

### Directional modes

Directional adjacency discriminates stripe axis from checkerboard:

```python
I_ap = moran_i(grid, adjacency="ap_only")   # vertical neighbors only
I_ml = moran_i(grid, adjacency="ml_only")   # horizontal neighbors only

# Row-striped artifact: I_ml >> I_ap (constant along rows)
# Column-striped artifact: I_ap >> I_ml (constant along columns)
# Checkerboard: both negative
```

## Gradient anisotropy

Measures directional imbalance of spatial gradients:

```python
from cogpy.measures.spatial import gradient_anisotropy

aniso = gradient_anisotropy(grid)
# 0.0 = isotropic (balanced gradients)
# positive = row-striped (large AP gradient, small ML gradient)
# negative = column-striped (large ML gradient, small AP gradient)
```

## Marginal energy outlier

Identifies which rows or columns carry anomalous energy:

```python
from cogpy.measures.spatial import marginal_energy_outlier

result = marginal_energy_outlier(grid, robust=True, threshold=3.0)

# result["col_outlier"]  — boolean mask, True for bad columns
# result["row_outlier"]  — boolean mask, True for bad rows
# result["col_zscore"]   — z-score per column
# result["row_zscore"]   — z-score per row
```

## Batch operation

All three measures accept arbitrary leading batch dimensions. This enables
efficient spatial characterization across time-frequency spectrograms
**without Python loops**:

```python
from cogpy.spectral.specx import spectrogramx
from cogpy.measures.spatial import gradient_anisotropy, moran_i

# Compute spectrogram: (AP, ML, time_win, freq)
spec = spectrogramx(sig, window_size=512, window_step=128)

# Transpose to (..., AP, ML) convention
spec_t = spec.values.transpose(2, 3, 0, 1)  # (time_win, freq, AP, ML)

# Vectorized — no loops, pure numpy
aniso_map = gradient_anisotropy(spec_t)       # (time_win, freq)
moran_map = moran_i(spec_t, adjacency="ap_only")  # (time_win, freq)

# aniso_map[t, f] = gradient anisotropy of the spatial grid at time t, freq f
```

This runs in seconds for typical grid sizes (16x16) even with hundreds of
time-frequency bins, because the computation is fully vectorized.

## CSD power

Current Source Density sharpens spatial specificity by computing the 2D
Laplacian:

```python
from cogpy.measures.spatial import csd_power

csd = csd_power(sig.values, spacing_mm=1.0)  # (AP, ML, time)
# Border electrodes are NaN (5-point stencil requires interior points)
```

## Next steps

- {doc}`/howto/batch-spatial-analysis` — applying spatial measures to full recordings
- {doc}`/explanation/data-model` — understanding the grid schema in depth
