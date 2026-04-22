---
title: Cross-Recording Factor Matching
file_format: mystnb
kernelspec:
  name: cogpy
  display_name: cogpy
  language: python
mystnb:
  execution_mode: "auto"
---

# Cross-Recording Factor Matching

This tutorial demonstrates how to **match spatio-spectral factors across
multiple recordings** (sessions or animals) and compute a consensus
(centroid) decomposition.  It is a direct follow-up to
{doc}`factor-analysis`, which covers single-recording erpPCA.

:::{admonition} Setup
:class: dropdown

**Python 3.10+** required.  Install cogpy with the extras this notebook
needs (interactive plots + spectral backend + Jupyter kernel):

```bash
pip install "ecogpy[viz,signal,notebook]"
```

Or install everything at once: `pip install "ecogpy[all]"`.
For a development install see {doc}`install`.
:::

:::{admonition} Prerequisites
:class: important

Read {doc}`factor-analysis` first.  This tutorial assumes familiarity
with `SpatSpecDecomposition`, `erpPCA`, loadings, and scores.
:::

## The problem

Varimax-rotated PCA decomposes each recording's spectrogram into
spatio-spectral factors independently.  The factors are meaningful
within a single recording, but across recordings:

- **Factor ordering differs** — variance ranking changes with noise.
- **Factor polarity and shape vary** — slightly different spatial
  patterns emerge from different data segments.

To compare factors across recordings (e.g. "is this spindle pattern
consistent across sessions?"), we need to **match** corresponding
factors and quantify their similarity.

## The approach

The matching pipeline (Garcia-Cortadella et al. 2024,
{func}`cogpy.decomposition.match.match_factors`) works in five steps:

:::{admonition} Algorithm overview
:class: tip

1. **Pairwise similarity** — for every pair of recordings, compute a
   `(nfac x nfac)` similarity matrix using spatial correlation of
   loading maps at peak frequency, gated by a frequency threshold.
2. **Hungarian assignment** — find the optimal one-to-one factor
   mapping for each pair that maximises total similarity
   (`scipy.optimize.linear_sum_assignment`).
3. **Reference selection** — pick the recording whose transitive
   matches are best overall (the "medoid"), or use a fixed reference.
4. **Transitive alignment** — re-index all recordings' factors to
   match the reference's ordering.
5. **Centroid** — average the aligned, normalised loadings to get a
   consensus decomposition.
:::

```{code-cell} python
import warnings; warnings.filterwarnings("ignore")  # suppress xarray FutureWarnings
import numpy as np
import xarray as xr
import pandas as pd
import holoviews as hv

hv.extension("bokeh")
```

## 1. Generate synthetic multi-session data

We create **4 synthetic recordings** that share the same underlying
spatio-spectral factors but with different noise realisations and
factor orderings — mimicking what you'd get from independent PCA
fits on different sessions of the same animal.

:::{admonition} Why synthetic data?
:class: note

The bundled sample recording is a single 7.5 s segment. To
demonstrate matching we need multiple recordings with shared
structure.  We extract "ground truth" loadings from the real data,
then generate synthetic spectrograms by mixing these loadings with
random scores + noise.  Each independent PCA fit recovers the factors
in a **different order** — exactly the scenario matching solves.
:::

```{code-cell} python
from cogpy.datasets import load_sample
from cogpy.spectral.specx import spectrogramx
from cogpy.decomposition.spatspec import SpatSpecDecomposition
from cogpy.decomposition.pca import erpPCA

# Fit erpPCA on the real sample to get "ground truth" loadings
sig = load_sample()
spec = spectrogramx(sig, bandwidth=4.0, nperseg=64, noverlap=56)
spec = spec.compute() if hasattr(spec.data, "compute") else spec
spec = spec.sel(freq=slice(1, 50)).isel(
    AP=slice(None, None, 2), ML=slice(None, None, 2)
)
spec = spec.coarsen(freq=2, boundary="trim").mean()

mtx = spec.rename({"AP": "h", "ML": "w"})
ss_real = SpatSpecDecomposition(mtx)
X_real = ss_real.designmat(mtx, log=True)
Xc_real = X_real - X_real.mean("time")
Xc_real.data = np.nan_to_num(Xc_real.data)

nfac = 6
erp = erpPCA(nfac=nfac, verbose=False)
erp.fit(Xc_real.data)

# Ground truth loadings: (nfac, n_vars)
L_true = erp.LR.T  # (nfac, hwf)
print(f"Ground truth: {nfac} factors, {L_true.shape[1]} variables")
```

Now generate 4 synthetic spectrograms by mixing these loadings with
random scores, adding noise, and fitting erpPCA independently on
each:

```{code-cell} python
rng = np.random.default_rng(42)
nrec = 4
n_time = 200  # time windows per synthetic recording

synthetic_series = []

for irec in range(nrec):
    # Random scores (each factor has different variance)
    variances = rng.exponential(1.0, size=nfac)
    scores = rng.normal(size=(n_time, nfac)) * np.sqrt(variances)

    # Reconstruct spectrogram + add noise
    X_synth = scores @ L_true
    noise_level = 0.3 * np.std(X_synth)
    X_synth += rng.normal(scale=noise_level, size=X_synth.shape)

    # Fit erpPCA independently — factor order will differ
    erp_synth = erpPCA(nfac=nfac, verbose=False)
    erp_synth.fit(X_synth)

    # Build SpatSpecDecomposition
    ss = SpatSpecDecomposition(ss_real.spatspec_coords)
    ss.ldx_set(erp_synth.LR)
    ss.ldx_process()
    synthetic_series.append(ss)

ss_series = pd.Series(synthetic_series)
print(f"Created {nrec} synthetic recordings, each with {nfac} factors")
```

## 2. Inspect unmatched factors

Before matching, factors from different recordings are in arbitrary
order.  Compare the spatial maps at peak frequency for the first two
recordings — notice that **similar patterns appear at different factor
indices**:

```{code-cell} python
from cogpy.plot.hv.decomposition import loading_spatial_layout

layout_0 = loading_spatial_layout(
    ss_series.iloc[0].ldx_slc_maxfreq,
    ss_series.iloc[0].ldx_df,
).opts(title="Recording 0 (unmatched)")

layout_1 = loading_spatial_layout(
    ss_series.iloc[1].ldx_slc_maxfreq,
    ss_series.iloc[1].ldx_df,
).opts(title="Recording 1 (unmatched)")

layout_0 + layout_1
```

:::{admonition} What to look for
:class: hint

You should see similar spatial patterns (blobs in similar grid
locations) but at different factor indices across the two recordings.
For example, Recording 0's Factor 0 might look like Recording 1's
Factor 3.  This is the ordering ambiguity that matching resolves.
:::

## 3. Pairwise similarity matrix

The first step in matching is computing the similarity between every
factor pair across two recordings.  The metric is the **Pearson
correlation between spatial loading maps at peak frequency**.  Factor
pairs whose peak frequencies differ by more than `freq_threshold` Hz
are assigned zero similarity:

```{code-cell} python
from cogpy.decomposition.pca import compute_similarity_matrix

simil_01 = compute_similarity_matrix(
    ss_series.iloc[0], ss_series.iloc[1], freq_threshold=3
)

hv.HeatMap(
    [(f"Rec0 F{i}", f"Rec1 F{j}", simil_01[i, j])
     for i in range(nfac) for j in range(nfac)],
    kdims=["Recording 0", "Recording 1"],
    vdims=["Similarity"],
).opts(
    cmap="RdBu_r", colorbar=True, width=400, height=350,
    title="Pairwise factor similarity (Rec 0 vs Rec 1)",
    xrotation=45, symmetric=True,
)
```

:::{admonition} Reading the heatmap
:class: hint

Each cell `(i, j)` shows the spatial correlation between Factor `i`
of Recording 0 and Factor `j` of Recording 1.  Values close to 1
indicate a strong match.  Zero values mean the peak frequencies
differed by more than `freq_threshold` — those pairs are excluded
from consideration.
:::

## 4. Hungarian assignment (bipartite matching)

The **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`)
finds the one-to-one mapping that maximises total similarity.  It
solves the assignment problem optimally in O(n^3):

```{code-cell} python
from scipy.optimize import linear_sum_assignment

row_ind, col_ind = linear_sum_assignment(-simil_01)  # negate for maximisation

print("Optimal factor mapping (Rec 0 -> Rec 1):")
for r, c in zip(row_ind, col_ind):
    print(f"  Factor {r} -> Factor {c}  (similarity: {simil_01[r, c]:.3f})")
print(f"\nMean optimal similarity: {simil_01[row_ind, col_ind].mean():.3f}")
```

:::{admonition} Why Hungarian and not greedy?
:class: note

A greedy approach (pick the best pair, remove it, repeat) can get
stuck in suboptimal solutions.  The Hungarian algorithm guarantees
the **globally optimal** one-to-one assignment — the permutation that
maximises the sum of matched similarities.
:::

## 5. Multi-recording matching with `match_factors`

For more than 2 recordings, pairwise Hungarian can't be applied
directly (it's a bipartite algorithm — it only handles two sets).
`match_factors` uses a **hub-and-spoke** strategy:

```{code-cell} python
from cogpy.decomposition.match import match_factors

results = match_factors(
    ss_series,
    nrec=nrec,
    nfac=nfac,
    freq_threshold=3,
    simil_score_threshold=0.5,
    optimal_ref=True,
)

print(f"Optimal reference recording: {int(results['metadata_csv']['opt_ref'])}")
print(f"\nMatch table (factor index per recording):")
print(results["match_df_csv"])
```

:::{admonition} Understanding the match table
:class: hint

Each **row** is a matched factor group.  The numbered columns
(0, 1, 2, 3) show which factor index in each recording corresponds
to that matched factor.  The `simil` column is the **minimum**
pairwise similarity across all recording pairs for that group — a
conservative quality measure.

Factors with similarity below `simil_score_threshold` are dropped.
:::

:::{admonition} Optimal reference selection
:class: note

When `optimal_ref=True`, the algorithm tries **every recording** as
the reference, runs the full matching, and picks the one whose
transitive matches are best overall.  This is like finding the
"medoid" of the group — the most typical recording.  Set
`optimal_ref=False` to always use recording 0 as the reference
(faster, deterministic, but may give worse matches if recording 0
is atypical).
:::

## 6. Visualise matched loadings

After matching, factor indices are aligned across recordings.
Each column below shows the **same matched factor** across all 4
recordings — they should look spatially similar:

```{code-cell} python
ss_all = results["ldx_all_pkl"]
n_matched = min(4, len(results["match_df_csv"]))

panels = []
for ifac in range(n_matched):
    for irec in range(nrec):
        slc = ss_all.ldx_slc_maxfreq.sel(rec=irec, factor=ifac)
        panels.append(
            hv.Image(
                slc, kdims=["w", "h"],
            ).opts(
                cmap="RdBu_r", colorbar=False, width=160, height=140,
                symmetric=True, invert_yaxis=True,
                title=f"Rec {irec}, F{ifac}",
            )
        )

hv.Layout(panels).cols(nrec).opts(
    title="Matched factors: each column = same factor across recordings"
)
```

## 7. Centroid (consensus) loadings

The centroid is the mean of normalised, aligned loadings — a
**consensus factor** that captures the shared spatial-spectral
pattern across all recordings:

```{code-cell} python
ss_cent = results["ldx_pkl"]

loading_spatial_layout(ss_cent.ldx_slc_maxfreq, ss_cent.ldx_df)
```

### Centroid spectral profiles

The frequency profile at the peak electrode for each consensus
factor — the red dashed line marks the peak frequency:

```{code-cell} python
from cogpy.plot.hv.decomposition import loading_spectral_profiles

loading_spectral_profiles(ss_cent.ldx_slc_maxch, ss_cent.ldx_df)
```

## 8. Quality assessment

Not all factors match well across recordings.  The `simil` column
quantifies the worst-case pairwise similarity for each matched
factor.  Low scores indicate patterns that are session-specific or
noise-driven:

```{code-cell} python
match_df = results["match_df_csv"]
hv.Bars(
    [(f"F{i}", s) for i, s in zip(match_df.index, match_df["simil"])],
    kdims=["Matched factor"], vdims=["Min. similarity"],
).opts(
    width=400, height=250, color="steelblue",
    title="Match quality per factor (min pairwise similarity)",
    ylim=(0, 1),
)
```

:::{admonition} Interpreting match quality
:class: tip

- **> 0.9**: strong match — the factor is highly reproducible.
- **0.7 -- 0.9**: moderate — likely a real pattern with some
  session-to-session variability.
- **< 0.7**: weak — consider whether this factor is meaningful
  or noise-driven.  Increase `simil_score_threshold` to filter
  these out automatically.
:::

## Summary

| Step | Function | Module |
|------|----------|--------|
| Fit per-recording | `erpPCA.fit()` | `cogpy.decomposition.pca` |
| Pairwise similarity | `compute_similarity_matrix()` | `cogpy.decomposition.pca` |
| Full matching pipeline | `match_factors()` | `cogpy.decomposition.match` |
| HV spatial layout | `loading_spatial_layout()` | `cogpy.plot.hv.decomposition` |
| HV spectral profiles | `loading_spectral_profiles()` | `cogpy.plot.hv.decomposition` |

### Key parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `freq_threshold` | 3 Hz | Max peak-frequency difference to consider a match |
| `simil_score_threshold` | 0.7 | Min similarity to retain a matched factor |
| `optimal_ref` | `True` | Auto-select best reference recording |

(cross-animal-matching)=
## Cross-animal matching

When comparing factors **across animals**, the matching can be done
in two stages:

1. **Within-animal**: match factors across sessions of the same
   animal using `match_factors` to produce a per-animal centroid.
2. **Across-animal**: collect the per-animal centroids into a new
   Series and run `match_factors` again to produce a grand centroid.

```python
# Stage 1: within-animal centroids
animal_centroids = {}
for animal_id, session_series in sessions_by_animal.items():
    result = match_factors(session_series, nrec=len(session_series),
                           nfac=nfac, freq_threshold=3,
                           simil_score_threshold=0.7)
    animal_centroids[animal_id] = result["ldx_pkl"]

# Stage 2: across-animal matching
cross_animal_series = pd.Series(animal_centroids)
grand_result = match_factors(cross_animal_series,
                              nrec=len(cross_animal_series),
                              nfac=nfac, freq_threshold=5,
                              simil_score_threshold=0.5)
grand_centroid = grand_result["ldx_pkl"]
```

:::{admonition} Spatial correspondence across animals
:class: warning

The current similarity metric (spatial correlation) assumes
**pixel-to-pixel correspondence** — i.e. the same grid size and
placement across animals.  When grid placements differ, spatial
correlation may underestimate true similarity.

Garcia-Cortadella et al. (2024) use an alternative **feature-space
Euclidean distance** for cross-animal matching: a 7-dimensional
summary (AP/ML sign, grid position, peak frequency, spatial extent)
that does not require aligned grids.  A `method="feature_space"`
option for `match_factors` is planned for a future release.
:::

## Next steps

- {doc}`factor-analysis` — single-recording factor analysis
  (prerequisite)
- {doc}`spectral-analysis` — spectral analysis stack (PSD,
  spectrograms)
- `cogpy.decomposition.match` — API reference for matching functions

## References

Garcia-Cortadella, R. et al. (2024). DC-coupled, 2.5D
electrophysiological imaging of large-scale cortical dynamics.
*bioRxiv*. [DOI: 10.1101/2024.12.20.629545](https://doi.org/10.1101/2024.12.20.629545)
