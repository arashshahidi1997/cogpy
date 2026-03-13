# Spectral Conventions

This page explains the design choices behind cogpy's spectral analysis stack.

## PSD-first convention

Spectral scalar features (band power, entropy, line noise ratio, etc.)
accept **pre-computed PSD arrays**, not raw signals:

```python
# Step 1: Compute PSD once
psd, freqs = psd_multitaper(signal, fs=1000.0)

# Step 2: Derive multiple features from the same PSD
gamma = band_power(psd, freqs, band=(30, 100))
entropy = spectral_entropy(psd, freqs)
lnr = line_noise_ratio(psd, freqs, line_freq=60.0)
```

**Why?** PSD estimation is the expensive step. Computing it once and reusing
it across features avoids redundant work. This also makes features composable
— any PSD source (Welch, multitaper, averaged spectrogram) works.

**Exception:** `ftest_line_scan` operates on raw signal because it needs
access to individual tapered FFTs for the F-test statistic.

## Multitaper methods

cogpy prefers multitaper spectral estimation over Welch's method for short
windows. Multitaper provides:

- Better frequency resolution at a given window length
- Controlled spectral leakage via DPSS (Slepian) tapers
- Natural framework for F-test line detection

The key parameter is **NW** (time-bandwidth product). Higher NW = more
tapers = smoother spectrum but lower frequency resolution.

| NW | Tapers (K=2NW-1) | Use case |
|----|-------------------|----------|
| 2.0 | 3 | High frequency resolution |
| 4.0 | 7 | General purpose (default) |
| 8.0 | 15 | Heavy smoothing |

## Frequency axis convention

- Frequency arrays are **strictly increasing**, in Hz
- The last axis of PSD/spectrogram arrays is always `freq`
- DC (0 Hz) is included; Nyquist may or may not be included depending on
  signal length

## Two API levels

### numpy level (`cogpy.core.spectral`)

Pure functions on numpy arrays. No xarray dependency.

```python
from cogpy.spectral.psd import psd_multitaper
from cogpy.spectral.multitaper import multitaper_fft
```

### xarray level (`cogpy.core.spectral.specx`)

Dimension-aware wrappers that preserve coordinates:

```python
from cogpy.spectral.specx import psdx, spectrogramx, coherencex
```

These call the numpy-level functions internally and attach appropriate
xarray coordinates to the output.

## Feature categories

| Category | Input | Output | Examples |
|----------|-------|--------|----------|
| Scalar | `(..., freq)` | `(...)` | band_power, spectral_entropy, line_noise_ratio |
| Vector | `(..., freq)` | `(..., freq)` | narrowband_ratio, fooof_periodic |
| Raw-signal | `(time,)` | `(freq,)` + metadata | ftest_line_scan |

All features broadcast over arbitrary leading batch dimensions.

## Spectral whitening

For analyses where the `1/f` spectral shape obscures narrowband features,
cogpy provides spectral whitening:

```python
from cogpy.spectral.whitening import whiten_ar
```

This fits an autoregressive model and divides out the predicted spectrum,
flattening the broadband background while preserving narrowband peaks.
