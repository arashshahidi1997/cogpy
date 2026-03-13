# Spectral Analysis

This tutorial covers cogpy's spectral analysis stack: PSD estimation,
time-frequency spectrograms, and derived features.

## PSD estimation

cogpy provides two PSD backends:

```python
from cogpy.spectral.psd import psd_welch, psd_multitaper

# Welch's method (good for long signals)
psd_w, freqs = psd_welch(signal, fs=1000.0)

# Multitaper (better frequency resolution, recommended for short windows)
psd_m, freqs = psd_multitaper(signal, fs=1000.0)
```

Both return `(psd, freqs)` tuples. The PSD shape matches the input with the
time axis replaced by a frequency axis: `(..., freq)`.

## xarray interface

For xarray DataArrays, use the `specx` module:

```python
from cogpy.spectral.specx import psdx, spectrogramx

# PSD with xarray coordinates preserved
psd_da = psdx(sig, NW=4.0)
# <xarray.DataArray (AP: 16, ML: 16, freq: 513)>

# Time-frequency spectrogram
spec = spectrogramx(sig, window_size=512, window_step=128, NW=4.0)
# <xarray.DataArray (AP: 16, ML: 16, time_win: ..., freq: 257)>
```

## Spectral features

All spectral features follow the **PSD-first convention**: they accept
pre-computed `(psd, freqs)` and reduce over the frequency axis.

```python
from cogpy.spectral.features import (
    band_power,
    relative_band_power,
    spectral_entropy,
    spectral_flatness,
    spectral_edge,
    broadband_snr,
    line_noise_ratio,
    narrowband_ratio,
    am_artifact_score,
    aperiodic_exponent,
)

# Band power — integrate PSD over a frequency band
gamma = band_power(psd, freqs, band=(30, 100))      # (...,) scalar
rel_gamma = relative_band_power(psd, freqs, band=(30, 100))  # fraction

# Spectral shape
entropy = spectral_entropy(psd, freqs)     # high = broadband, low = peaked
flatness = spectral_flatness(psd, freqs)   # 1.0 = white noise, 0.0 = pure tone
edge_95 = spectral_edge(psd, freqs, p=0.95)  # frequency below which 95% of power

# Signal quality
snr = broadband_snr(psd, freqs)
lnr = line_noise_ratio(psd, freqs, line_freq=60.0)

# Narrowband artifact detection
nb_ratio = narrowband_ratio(psd, freqs, flank_hz=5.0)  # (..., freq)
# Values >> 1 indicate narrowband peaks (line noise, artifact)

# Amplitude modulation
am_score = am_artifact_score(psd, freqs)

# Aperiodic fit (requires specparam/FOOOF)
exponent = aperiodic_exponent(psd, freqs)
```

### Batch dimensions

All features broadcast over leading dimensions. For a grid PSD of shape
`(16, 16, 513)` (AP, ML, freq), each feature returns `(16, 16)`.

## Line noise detection

For detecting narrowband lines at **unknown** frequencies, use the F-test
scanner:

```python
from cogpy.spectral.features import ftest_line_scan

# Scans all frequency bins for sinusoidal components
fstat, freqs, sig_mask = ftest_line_scan(signal_1d, fs=1000.0, NW=4.0)

# sig_mask[i] == True means a significant line at freqs[i]
line_freqs = freqs[sig_mask]
```

This implements Thomson's (1982) eigenvalue-weighted F-test across all
frequency bins simultaneously.

## Next steps

- {doc}`bad-channel-detection` — use spectral features for channel quality assessment
- {doc}`/howto/filtering` — temporal and spatial filtering
