# Bad Channel Detection

This tutorial walks through cogpy's canonical bad-channel detection pipeline:
extract per-channel features, normalize spatially, and label outliers.

## Overview

The pipeline has four stages:

```
Raw signal → Feature extraction → Spatial normalization → Outlier labeling
              (temporal measures)   (neighborhood stats)    (DBSCAN)
```

Each stage is a separate module under `cogpy.core.preprocess.badchannel`.

## Step 1: Extract channel features

```python
from cogpy.preprocess.badchannel.channel_features import extract_channel_features_xr

# sig: xarray.DataArray with dims (time, AP, ML) and fs coordinate
features = extract_channel_features_xr(
    sig,
    window_size=2048,
    window_step=1024,
    feature_names=["variance", "amplitude", "kurtosis", "hurst_exponent"],
)
# Returns xarray.Dataset with dims (time_win, AP, ML) and one variable per feature
```

Each feature is computed per channel per time window. The underlying
functions live in `cogpy.core.measures.temporal`:

```python
from cogpy.measures.temporal import (
    relative_variance,
    amplitude,
    kurtosis,
    hurst_exponent,
)
```

## Step 2: Spatial normalization

Raw feature values vary across the grid due to electrode impedance and
anatomy. Spatial normalization expresses each channel's feature value
relative to its neighborhood:

```python
from cogpy.preprocess.badchannel.pipeline import (
    compute_features_sliding,
    DEFAULT_FEATURE_SPECS,
)

# DEFAULT_FEATURE_SPECS defines 7 features with normalization strategies:
# - "ratio": channel / neighborhood_median (e.g., variance, amplitude)
# - "difference": channel - neighborhood_median (e.g., hurst_exponent)
# - "robust_z": (channel - median) / MAD (e.g., kurtosis)
# - "identity": no normalization (e.g., anticorrelation)

normalized = compute_features_sliding(
    sig,
    window_size=2048,
    window_step=1024,
    feature_specs=DEFAULT_FEATURE_SPECS,
)
```

## Step 3: Label outliers

DBSCAN clusters channels in feature space. Outlier channels (noise label
`-1`) are flagged as bad:

```python
from cogpy.preprocess.badchannel.badlabel import (
    DbscanParams,
    dbscan_outliers,
)

params = DbscanParams(eps=1.5, min_samples=5)
labels = dbscan_outliers(normalized, params=params)
# labels: array of 0 (good) / 1 (bad) per channel
```

## Full pipeline

For a complete example composing all stages:

```python
from cogpy.preprocess.badchannel.pipeline import compute_features_sliding
from cogpy.preprocess.badchannel.badlabel import dbscan_outliers, DbscanParams
from cogpy.preprocess.badchannel.feature_normalization import (
    normalize_windowed_features,
    summarize_windowed_features,
)

# Extract and normalize
features = compute_features_sliding(sig, window_size=2048, window_step=1024)

# Summarize across time windows (median per channel)
summary = summarize_windowed_features(features)

# Label outliers
bad_mask = dbscan_outliers(summary, params=DbscanParams())
```

## Interpolating bad channels

Once bad channels are identified, interpolate them:

```python
from cogpy.preprocess.interpolate import interpolate_bads

sig_clean = interpolate_bads(sig, bad_mask=bad_mask)
```

## Next steps

- {doc}`spatial-measures` — spatial grid characterization beyond bad-channel detection
- {doc}`/howto/custom-snakemake-pipeline` — automate this pipeline with Snakemake
