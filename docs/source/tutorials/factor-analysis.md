---
title: Factor Analysis of Spectrograms with erpPCA
file_format: mystnb
kernelspec:
  name: cogpy
  display_name: cogpy
  language: python
---

# Factor Analysis of Spectrograms with erpPCA

This tutorial shows how to decompose a grid ECoG spectrogram into
spatio-spectral factors using **varimax-rotated PCA** (erpPCA).

The pipeline is:

1. Load a grid ECoG signal
2. Compute a multitaper spectrogram
3. Build a design matrix (log-power, z-scored)
4. Fit erpPCA → extract loadings and scores
5. Inspect loadings (which spatial pattern at which frequency)
6. Process factor scores (smooth, threshold, detect events)

Each factor captures a recurring spatio-spectral pattern —
e.g. "spindle-band activity over posterior cortex".

```{code-cell} python
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import holoviews as hv

hv.extension("bokeh")
```

## 1. Load sample data

`load_sample()` returns a preprocessed grid ECoG signal with dims
`(AP, ML, time)` at 125 Hz.

```{code-cell} python
from cogpy.datasets import load_sample

sig = load_sample()
print(sig)
```

Quick look — single-channel time trace and a spatial snapshot:

```{code-cell} python
fig, axes = plt.subplots(1, 2, figsize=(12, 3))

# Time trace at center electrode
sig.sel(AP=8, ML=8).plot(ax=axes[0])
axes[0].set(title="Time trace (AP=8, ML=8)", xlabel="Time (s)", ylabel="Amplitude")

# Spatial snapshot at one time point
sig.isel(time=sig.sizes["time"] // 2).plot(ax=axes[1], cmap="RdBu_r")
axes[1].set(title="Spatial snapshot (mid-recording)", aspect="equal")
plt.tight_layout()
```

## 2. Compute multitaper spectrogram

We use `spectrogramx` (ghostipy backend) to compute a time-frequency
representation for every electrode in the grid.

Two practical considerations for the PCA step:

- **Frequency range**: clip to frequencies of interest (1–50 Hz)
  to avoid inflating the variable count.
- **Grid subsampling**: for large grids, subsample spatially to keep
  the covariance matrix tractable.

The PCA variable count is `n_AP × n_ML × n_freq`. The covariance
matrix is `(n_vars × n_vars)`, so keeping this under ~3000 is
recommended for interactive use.

```{code-cell} python
from cogpy.spectral.specx import spectrogramx

spec = spectrogramx(sig, bandwidth=4.0, nperseg=64, noverlap=56)
spec = spec.compute() if hasattr(spec.data, "compute") else spec

# Clip frequency range and subsample grid
spec = spec.sel(freq=slice(1, 50))
spec = spec.isel(AP=slice(None, None, 2), ML=slice(None, None, 2))  # 8×8

n_vars = spec.sizes["AP"] * spec.sizes["ML"] * spec.sizes["freq"]
print(f"Spectrogram: {spec.shape}")
print(f"PCA variables: {spec.sizes['AP']}×{spec.sizes['ML']}×{spec.sizes['freq']} = {n_vars}")
```

Spectrogram at one electrode and mean spectrum across the grid:

```{code-cell} python
mid_ap = spec.AP.values[len(spec.AP) // 2]
mid_ml = spec.ML.values[len(spec.ML) // 2]

spec_img = hv.Image(
    spec.sel(AP=mid_ap, ML=mid_ml),
    kdims=["time", "freq"],
    label="Spectrogram",
).opts(cmap="viridis", colorbar=True, width=500, height=250,
       title=f"Spectrogram at AP={mid_ap}, ML={mid_ml}")

mean_spec = spec.mean(dim=("AP", "ML", "time"))
mean_curve = hv.Curve(
    (mean_spec.freq.values, mean_spec.values),
    kdims=["Frequency (Hz)"], vdims=["Power"],
    label="Mean spectrum",
).opts(width=350, height=250, title="Mean power spectrum", logy=True)

spec_img + mean_curve
```

## 3. Build the design matrix

erpPCA operates on a 2D matrix `(time, variables)`. We:

1. **Rename** dims to `SpatSpecDecomposition` convention (`AP→h`, `ML→w`)
2. **Flatten** spatial + frequency dims into one axis
3. **Log-transform** power (stabilises variance)
4. **Z-score** each variable across time

```{code-cell} python
from scipy.stats import zscore
from cogpy.core.decomposition.spatspec import SpatSpecDecomposition

mtx = spec.rename({"AP": "h", "ML": "w"})
ss = SpatSpecDecomposition(mtx)

X = ss.designmat(mtx, log=True)
Xz = X.copy()
Xz.data = zscore(X.data, axis=0, nan_policy="omit")
Xz.data[np.isnan(Xz.data)] = 0.0

print(f"Design matrix: {Xz.shape[0]} time points × {Xz.shape[1]} variables")
```

Visualise the design matrix as a heatmap — each row is a time window,
each column a (channel, frequency) variable:

```{code-cell} python
hv.Image(
    Xz.data,
    kdims=["Variable", "Time"],
    bounds=(0, 0, Xz.shape[1], Xz.shape[0]),
).opts(
    cmap="RdBu_r", colorbar=True, width=600, height=250,
    title="Design matrix (z-scored log-power)",
    xlabel="Variable index (h × w × freq)", ylabel="Time window",
    invert_yaxis=True,
)
```

## 4. Fit erpPCA

The `erpPCA` estimator follows the scikit-learn API:

1. Covariance matrix of the design matrix
2. Eigendecomposition
3. **Varimax rotation** to maximise simple structure
4. Sort factors by explained variance

```{code-cell} python
from cogpy.core.decomposition.pca import erpPCA

nfac = 8
erp = erpPCA(nfac=nfac, verbose=False)
erp.fit(Xz.data)

print(f"Loadings:   {erp.LR.shape}  (vars × factors)")
print(f"Scores:     {erp.FSr.shape}  (time × factors)")
print(f"Var explained (rotated): {erp.VT[3][:nfac].sum():.1f}%")
```

Variance explained per factor:

```{code-cell} python
var_pct = erp.VT[3][:nfac]
hv.Bars(
    [(f"F{i}", v) for i, v in enumerate(var_pct)],
    kdims=["Factor"], vdims=["Variance (%)"],
).opts(
    width=400, height=250, color="steelblue",
    title="Variance explained per factor (rotated)",
)
```

## 5. Reshape into loadings and scores

Reshape the flat loading matrix back into `(factor, h, w, freq)` and
wrap scores as `(time, factor)` DataArrays.

```{code-cell} python
ss.ldx_set(erp.LR)
ss.ldx_process()
scx = ss.scx_from_FSr(erp.FSr, Xz.time.values)

print(f"Loadings: {ss.ldx.shape}")
print(f"Scores:   {scx.shape}")
print(f"\nFactor summary:")
print(ss.ldx_df[["AP", "ML", "freqmax", "norm"]])
```

## 6. Visualise loadings

### Spatial maps (HoloViews)

Each factor's spatial map at its peak frequency, laid out in a grid.
The `×` marks the peak electrode. These are static `HoloMap`-compatible
objects that render on Sphinx sites.

```{code-cell} python
from cogpy.core.plot.hv.decomposition import loading_spatial_layout

loading_spatial_layout(ss.ldx_slc_maxfreq, ss.ldx_df)
```

### Browse factors with HoloMap

A `HoloMap` keyed by factor — use the slider or dropdown to browse:

```{code-cell} python
from cogpy.core.plot.hv.decomposition import factor_holomap

factor_holomap(ss.ldx, ss.ldx_df)
```

### Spectral profiles

Frequency loading at the peak electrode for each factor — the red
dashed line marks the peak frequency:

```{code-cell} python
from cogpy.core.plot.hv.decomposition import loading_spectral_profiles

loading_spectral_profiles(ss.ldx_slc_maxch, ss.ldx_df)
```

### Interactive explorer (notebook only)

The following `DynamicMap` provides a live slider to browse the full
4D loading tensor across factors and frequency bins. It will **not
render on a static site** — run this cell in a Jupyter notebook.

```{code-cell} python
:tags: [skip-execution]

# NOTE: DynamicMap — requires a live kernel (won't render on static docs site)
def _loading_explorer(factor, freq_idx):
    freq_val = float(ss.ldx.freq.values[freq_idx])
    slc = ss.ldx.sel(factor=factor, freq=freq_val, method="nearest")
    return hv.Image(slc, kdims=["w", "h"]).opts(
        cmap="RdBu_r", colorbar=True, width=350, height=300,
        symmetric=True, invert_yaxis=True,
        title=f"Factor {factor} @ {freq_val:.1f} Hz",
    )

hv.DynamicMap(
    _loading_explorer,
    kdims=[
        hv.Dimension("factor", values=list(range(nfac))),
        hv.Dimension("freq_idx", range=(0, ss.ldx.sizes["freq"] - 1), step=1),
    ],
)
```

## 7. Factor scores over time

Factor scores are how strongly each spatio-spectral pattern is
expressed at each time point.

```{code-cell} python
from cogpy.core.plot.hv.decomposition import score_traces

score_traces(scx, ss.ldx_df)
```

## 8. Score processing

Raw scores are noisy. The `scores` module provides a pipeline:

1. **Gaussian smoothing** — temporal low-pass
2. **Lower envelope removal** — removes slow baseline shifts
3. **Quantile thresholding** — keeps only prominent activations

```{code-cell} python
from cogpy.core.decomposition.scores import scx_process

scx_dict = scx_process(scx, sigma=0.25, quantile=0.25, return_all=True)
```

Compare processing stages for one factor:

```{code-cell} python
ifac = 0
peak_freq = ss.ldx_df.loc[ifac, "freqmax"]
stages = [
    ("Raw", scx_dict["scx"], "steelblue"),
    ("Smoothed (no envelope)", scx_dict["scx_noenv"], "darkorange"),
    ("Thresholded", scx_dict["scx_thresh"], "green"),
]

panels = []
for label, arr, color in stages:
    trace = arr.sel(factor=ifac)
    panels.append(
        hv.Curve(
            (trace.time.values, trace.values),
            kdims=["Time (s)"], vdims=["Score"],
        ).opts(
            width=700, height=150, color=color,
            title=f"F{ifac} ({peak_freq:.0f} Hz) — {label}",
        )
    )

hv.Layout(panels).cols(1)
```

## 9. Reconstruct and compare

Reconstruct the spectrogram from loadings × scores and compare to the
original at one electrode.

```{code-cell} python
mtx_hat = ss.reconstruct(scx)
mtx_z = ss.mtx_from_designmat(Xz, mtx)

h_sel = ss.ldx.h.values[len(ss.ldx.h) // 2]
w_sel = ss.ldx.w.values[len(ss.ldx.w) // 2]

orig_slc = mtx_z.sel(h=h_sel, w=w_sel)
recon_slc = mtx_hat.sel(h=h_sel, w=w_sel)
resid_slc = orig_slc - recon_slc

shared = dict(kdims=["time", "freq"], width=320, height=220, colorbar=True)

img_orig = hv.Image(orig_slc, **shared).opts(
    cmap="viridis", title="Original (z-scored)")
img_recon = hv.Image(recon_slc, **shared).opts(
    cmap="viridis", title=f"Reconstructed ({nfac} factors)")
img_resid = hv.Image(resid_slc, **shared).opts(
    cmap="RdBu_r", title="Residual", symmetric=True)

img_orig + img_recon + img_resid
```

## 10. Frequency band labelling

Tag factors by their peak frequency band using the generic
`mark_freq_band` method.

```{code-cell} python
ss.mark_freq_band("delta",      0.5,  4)
ss.mark_freq_band("theta",      4,    8)
ss.mark_freq_band("alpha",      8,   13)
ss.mark_freq_band("spindle",   10,   16)
ss.mark_freq_band("beta",      13,   30)
ss.mark_freq_band("low_gamma", 30,   80)

print(ss.ldx_df[["freqmax", "is_delta", "is_theta", "is_alpha",
                  "is_spindle", "is_beta", "is_low_gamma"]])
```

## Summary

| Step | Function / Class | Module |
|------|-----------------|--------|
| Load data | `load_sample()` | `cogpy.datasets` |
| Spectrogram | `spectrogramx()` | `cogpy.spectral.specx` |
| Design matrix | `SpatSpecDecomposition.designmat()` | `cogpy.core.decomposition.spatspec` |
| Fit PCA | `erpPCA.fit()` | `cogpy.core.decomposition.pca` |
| Reshape loadings | `SpatSpecDecomposition.ldx_set()` | `cogpy.core.decomposition.spatspec` |
| Score processing | `scx_process()` | `cogpy.core.decomposition.scores` |
| Factor matching | `match_factors()` | `cogpy.core.decomposition.match` |
| HV spatial layout | `loading_spatial_layout()` | `cogpy.core.plot.hv.decomposition` |
| HV factor browser | `factor_holomap()` | `cogpy.core.plot.hv.decomposition` |

For cross-recording factor matching (e.g. aligning factors across
sessions or subjects), see `cogpy.core.decomposition.match.match_factors`.

## Computational considerations

### What this tutorial subsampled

This tutorial uses a short sample recording (7.5 s). To keep the
eigendecomposition tractable for interactive use, we reduced the
variable count:

| Parameter | Full data | This tutorial | Effect |
|-----------|-----------|---------------|--------|
| Grid | 16×16 = 256 ch | 8×8 = 64 ch | 2× spatial subsample |
| Freq range | 0–62 Hz (33 bins) | 1–50 Hz (25 bins) | Drop DC + above-interest |
| Overlap | 48/64 → 55 time wins | 56/64 → 109 time wins | More samples for PCA |
| **PCA variables** | **8,448** | **1,600** | 5× reduction |

The computational bottleneck is **`np.linalg.eigh`** on the
`(n_vars × n_vars)` covariance matrix — an O(n³) operation.
The covariance computation itself (`X.T @ X`) is fast and scales
linearly with recording length.

### Scaling to real sessions (2 hours, 16×16 grid)

For a 2-hour recording at 125 Hz with `nperseg=256, noverlap=224`:

| | Tutorial | Real session |
|---|----------|-------------|
| Duration | 7.5 s | 7,200 s |
| Time windows | 109 | ~28,000 |
| Grid | 8×8 | 16×16 |
| Freq bins (1–50 Hz) | 25 | ~100 |
| **Variables** | **1,600** | **25,600** |
| Cov matrix entries | 2.5 M | 655 M |

With 28,000 time samples >> 25,600 variables the covariance is
well-conditioned. The cost is dominated by the eigendecomposition.

### Strategies for large-scale analysis

- **Frequency binning**: average power into ~10 canonical bands
  (delta, theta, alpha, ...) instead of keeping all freq bins.
  16×16×10 = 2,560 vars — fast and interpretable.
- **Truncated eigendecomposition**: replace `np.linalg.eigh` with
  `scipy.sparse.linalg.eigsh(k=nfac)` to compute only the top-k
  eigenvalues. Reduces O(n³) to O(n²·k).
- **Two-stage decomposition**: first reduce frequency dimensionality
  per channel (standard PCA), then run spatial varimax on the
  reduced representation.
- **Spatial subsampling**: for very large grids, subsample every
  2nd electrode (as done here) with minimal information loss for
  low-spatial-frequency patterns.

## Next steps

- {doc}`spectral-analysis` — spectral analysis stack (PSD, spectrograms, features)
- {doc}`spatial-measures` — spatial grid measures (Moran's I, gradient anisotropy)
