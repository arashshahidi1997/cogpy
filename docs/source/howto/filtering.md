# How to filter signals

## Bandpass filter

```python
from cogpy.preprocess.filtering import bandpassx

# Bandpass 1-200 Hz, 4th-order Butterworth, zero-phase
sig_bp = bandpassx(sig, wl=1.0, wh=200.0, order=4, axis="time")
```

All filtering functions accept `xarray.DataArray` and preserve coordinates
and metadata. Under the hood they use `scipy.signal.sosfiltfilt` (zero-phase
Butterworth).

## Lowpass / highpass

```python
from cogpy.preprocess.filtering import lowpassx, highpassx

sig_lp = lowpassx(sig, wl=100.0, order=4, axis="time")
sig_hp = highpassx(sig, wh=1.0, order=4, axis="time")
```

## Notch filter (line noise removal)

```python
from cogpy.preprocess.filtering import notchesx

# Remove 60 Hz and harmonics
sig_clean = notchesx(sig, freqs=[60.0, 120.0, 180.0])
```

For ICA-based line noise removal (more aggressive):

```python
from cogpy.preprocess.linenoise import LineNoiseEstimatorICA

estimator = LineNoiseEstimatorICA(line_freq=60.0)
sig_clean = estimator.fit_transform(sig)
```

## Common median reference

```python
from cogpy.preprocess.filtering import cmrx

# Subtract median across channels at each time sample
sig_cmr = cmrx(sig)  # auto-detects (AP, ML) or (channel,) dims
```

## Filter order and edge effects

All temporal filters default to `scipy.signal.sosfiltfilt` (zero-phase).
A 4th-order Butterworth applied forward and backward gives an effective
8th-order zero-phase response.

For short signals, edge effects can be significant. Consider:
- Padding the signal before filtering (cogpy pads with reflection by default)
- Using a lower filter order
- Trimming edges after filtering

## Spatial filtering (CSD)

To sharpen spatial specificity on grid data, apply the Current Source Density
(2D Laplacian):

```python
from cogpy.measures.spatial import csd_power

csd = csd_power(sig.values, spacing_mm=1.0)
# Border electrodes are NaN; interior uses 5-point finite-difference stencil
```
