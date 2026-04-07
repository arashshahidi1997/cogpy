---
title: Travelling-Wave Detection and Analysis
file_format: mystnb
kernelspec:
  name: cogpy
  display_name: cogpy
  language: python
mystnb:
  execution_mode: "auto"
---

# Travelling-Wave Detection and Analysis

This tutorial introduces `cogpy.wave`'s travelling-wave analysis tools.
These methods estimate the **direction**, **speed**, and **spatial pattern**
of propagating activity across electrode grids.

The module provides three complementary estimation families:

- **Phase-gradient** plane-wave fitting (Zhang et al., 2018)
- **k--w spectral** analysis (wavenumber--frequency decomposition)
- **Optical-flow** velocity fields with pattern classification

All methods accept `xarray.DataArray` with the standard cogpy grid
dimensions `(time, AP, ML)`.

:::{note}
This tutorial uses **synthetic data** with known ground truth so that
every estimator can be verified.  For real ECoG data, replace the
synthetic generators with your loaded and bandpass-filtered signal.
:::

```{code-cell} python
import numpy as np
import xarray as xr
import holoviews as hv

hv.extension("bokeh")

from cogpy.wave._types import Geometry, PatternType
from cogpy.wave import synthetic, phase_gradient, kw_spectrum, surrogates
```

## 1. Synthetic travelling waves

### Plane wave

A plane wave propagating at 45 degrees, 2 mm/s, 10 Hz on an 8x8 grid
with 0.4 mm spacing:

```{code-cell} python
geo = Geometry.regular(dx=0.4, dy=0.4)

sig = synthetic.plane_wave(
    shape=(1000, 8, 8),    # 1 s at 1 kHz, 8x8 grid
    geometry=geo,
    direction=np.pi / 4,   # 45 degrees
    speed=2.0,             # 2 mm/s
    frequency=10.0,        # 10 Hz
    fs=1000.0,
    noise_std=0.1,
    rng=42,
)
print(sig)
```

Visualise the wave as a **HoloMap movie** -- use the slider to scrub
through time and watch the wave propagate across the grid:

```{code-cell} python
# Subsample to ~20 frames for a lightweight HoloMap
sig_sub = sig.isel(time=slice(0, 200, 10))

holomap = hv.HoloMap(
    {float(sig_sub.time[i]): hv.Image(
        sig_sub.isel(time=i).values,
        kdims=["ML", "AP"],
        bounds=(0, 0, sig_sub.sizes["ML"] - 1, sig_sub.sizes["AP"] - 1),
    ) for i in range(sig_sub.sizes["time"])},
    kdims=["time (s)"],
)
holomap.opts(
    hv.opts.Image(
        cmap="RdBu_r", colorbar=True, symmetric=True,
        width=400, height=350, title="Plane wave movie",
    )
)
```

And a time trace alongside a spatial snapshot:

```{code-cell} python
trace = hv.Curve(
    (sig.time.values, sig.isel(AP=4, ML=4).values),
    kdims=["Time (s)"], vdims=["Amplitude"],
).opts(width=500, height=200, title="Center electrode")

snapshot = hv.Image(
    sig.isel(time=50).values,
    kdims=["ML", "AP"],
    bounds=(0, 0, sig.sizes["ML"] - 1, sig.sizes["AP"] - 1),
).opts(cmap="RdBu_r", colorbar=True, symmetric=True,
       width=350, height=300, title="t = 50 ms")

trace + snapshot
```

### Spiral wave

```{code-cell} python
spiral = synthetic.spiral_wave(
    shape=(500, 16, 16),
    geometry=geo,
    center=(3.0, 3.0),
    angular_freq=2 * np.pi * 8,  # 8 Hz rotation
    fs=1000.0,
    noise_std=0.05,
    rng=42,
)

# HoloMap movie of the spiral -- scrub through time to see it rotate
sp_sub = spiral.isel(time=slice(0, 250, 12))

spiral_movie = hv.HoloMap(
    {float(sp_sub.time[i]): hv.Image(
        sp_sub.isel(time=i).values,
        kdims=["ML", "AP"],
        bounds=(0, 0, sp_sub.sizes["ML"] - 1, sp_sub.sizes["AP"] - 1),
    ) for i in range(sp_sub.sizes["time"])},
    kdims=["time (s)"],
)
spiral_movie.opts(
    hv.opts.Image(
        cmap="RdBu_r", colorbar=True, symmetric=True,
        width=400, height=350, title="Spiral wave movie",
    )
)
```

### Multi-component superposition

```{code-cell} python
wave1 = synthetic.plane_wave(
    shape=(1000, 8, 8), geometry=geo,
    direction=0.0, speed=2.0, frequency=10.0, fs=1000.0,
)
wave2 = synthetic.plane_wave(
    shape=(1000, 8, 8), geometry=geo,
    direction=np.pi / 2, speed=1.5, frequency=12.0, fs=1000.0,
)
mixed = synthetic.multi_wave([wave1, wave2], noise_std=0.1, rng=42)
print(f"Mixed signal shape: {mixed.shape}")
```

## 2. Phase-gradient analysis

The phase-gradient approach extracts the instantaneous phase of a
band-limited oscillation, computes its spatial gradient, and fits a
plane wave.  This follows Zhang et al. (2018).

:::{tip}
For real data, always **bandpass-filter** to the band of interest before
computing phase:

```python
from cogpy.preprocess.filtering.temporal import bandpass
sig_alpha = bandpass(sig, flo=8, fhi=12)
```

The synthetic plane wave is already narrowband, so we skip this step here.
:::

### Analytic phase

```{code-cell} python
phase = phase_gradient.hilbert_phase(sig, axis="time")

hv.Curve(
    (phase.time.values, phase.isel(AP=4, ML=4).values),
    kdims=["Time (s)"], vdims=["Phase (rad)"],
).opts(width=600, height=200, title="Unwrapped phase (center electrode)")
```

Browse the phase map over time -- the linear phase ramp sweeps across
the grid:

```{code-cell} python
phase_sub = phase.isel(time=slice(0, 200, 10))

phase_movie = hv.HoloMap(
    {float(phase_sub.time[i]): hv.Image(
        phase_sub.isel(time=i).values,
        kdims=["ML", "AP"],
        bounds=(0, 0, phase_sub.sizes["ML"] - 1, phase_sub.sizes["AP"] - 1),
    ) for i in range(phase_sub.sizes["time"])},
    kdims=["time (s)"],
)
phase_movie.opts(
    hv.opts.Image(
        cmap="twilight", colorbar=True,
        width=400, height=350, title="Phase map movie",
    )
)
```

### Phase-gradient directionality (PGD)

PGD measures how consistently the spatial phase gradient points in a
single direction.  Values near 1 indicate a clean plane wave; near 0
indicates no coherent propagation.

```{code-cell} python
pgd_score = phase_gradient.pgd(phase, geo)

pgd_curve = hv.Curve(
    (pgd_score.time.values, pgd_score.values),
    kdims=["Time (s)"], vdims=["PGD"],
).opts(width=600, height=200, ylim=(0, 1.05), title="Phase-gradient directionality")

chance_line = hv.HLine(0.5).opts(color="gray", line_dash="dashed", alpha=0.5)

pgd_curve * chance_line
```

```{code-cell} python
print(f"Mean PGD: {float(pgd_score.mean()):.3f}")
```

### Plane-wave fit

Fit a plane to the phase map at each time step to recover direction and
speed:

```{code-cell} python
estimates = phase_gradient.plane_wave_fit(phase, geo, freq=10.0)

directions = np.array([e.direction for e in estimates])
speeds = np.array([e.speed for e in estimates])
fit_quals = np.array([e.fit_quality for e in estimates])

dir_hist = hv.Histogram(
    np.histogram(np.degrees(directions), bins=30),
).opts(color="steelblue", width=280, height=220, title="Direction")
dir_line = hv.VLine(45).opts(color="red", line_dash="dashed")

spd_hist = hv.Histogram(
    np.histogram(speeds, bins=30),
).opts(color="darkorange", width=280, height=220, title="Speed")
spd_line = hv.VLine(2.0).opts(color="red", line_dash="dashed")

fq_hist = hv.Histogram(
    np.histogram(fit_quals, bins=30),
).opts(color="seagreen", width=280, height=220, title="Fit quality (R^2)")

(dir_hist * dir_line + spd_hist * spd_line + fq_hist).cols(3)
```

```{code-cell} python
print(f"Direction: {np.degrees(np.median(directions)):.1f} deg "
      f"(true: 45.0 deg)")
print(f"Speed: {np.median(speeds):.2f} mm/s (true: 2.00 mm/s)")
print(f"Fit quality: {np.median(fit_quals):.3f}")
```

## 3. k--w spectral analysis

The wavenumber--frequency spectrum `S(kx, ky, f)` provides an
independent view of wave structure. Peaks correspond to dominant
propagating components.

```{code-cell} python
spec = kw_spectrum.kw_spectrum_3d(sig, geo)
print(f"Spectrum shape: {spec.shape}  dims: {spec.dims}")
```

### Visualise the spectrum

Browse the k-space at each frequency using the slider -- look for
concentrated energy at the expected wavenumber at 10 Hz:

```{code-cell} python
# HoloMap: k-space slice at selected frequencies (subsample for page size)
freq_idx = np.linspace(0, spec.sizes["freq"] - 1, 20, dtype=int)

kw_movie = hv.HoloMap(
    {float(spec.freq.values[i]): hv.Image(
        np.log10(spec.isel(freq=i).values + 1e-10),
        kdims=["ky", "kx"],
        bounds=(
            float(spec.ky.values[0]), float(spec.kx.values[0]),
            float(spec.ky.values[-1]), float(spec.kx.values[-1]),
        ),
    ) for i in freq_idx},
    kdims=["freq (Hz)"],
)
kw_movie.opts(
    hv.opts.Image(
        cmap="hot", colorbar=True,
        width=400, height=350,
        title="k-w spectrum (log power)",
        xlabel="ky (cycles/mm)", ylabel="kx (cycles/mm)",
    )
)
```

Fixed-frequency slice at 10 Hz and a frequency-kx cross-section:

```{code-cell} python
f_idx = int(np.argmin(np.abs(spec.freq.values - 10.0)))
ky_idx = int(np.argmin(np.abs(spec.ky.values - 0.0)))

kslice = hv.Image(
    np.log10(spec.isel(freq=f_idx).values + 1e-10),
    kdims=["ky", "kx"],
    bounds=(
        float(spec.ky.values[0]), float(spec.kx.values[0]),
        float(spec.ky.values[-1]), float(spec.kx.values[-1]),
    ),
).opts(cmap="hot", colorbar=True, width=350, height=300,
       title=f"k-space at f = {spec.freq.values[f_idx]:.1f} Hz")

fk_slice = hv.Image(
    np.log10(spec.isel(ky=ky_idx).values + 1e-10),
    kdims=["kx", "freq"],
    bounds=(
        float(spec.kx.values[0]), float(spec.freq.values[0]),
        float(spec.kx.values[-1]), float(spec.freq.values[-1]),
    ),
).opts(cmap="hot", colorbar=True, width=350, height=300,
       title=f"f-kx at ky = {spec.ky.values[ky_idx]:.2f}")

kslice + fk_slice
```

### Peak extraction

```{code-cell} python
peaks = kw_spectrum.kw_peaks(spec, n_peaks=1)
pk = peaks[0]

print(f"Frequency:  {pk.frequency:.1f} Hz (true: 10.0 Hz)")
print(f"Direction:  {np.degrees(pk.direction):.1f} deg (true: 45.0 deg)")
print(f"Speed:      {pk.speed:.2f} mm/s (true: 2.00 mm/s)")
if pk.wavelength is not None:
    print(f"Wavelength: {pk.wavelength:.2f} mm")
```

### Multi-component detection

For the two-wave mixture, extract two peaks:

```{code-cell} python
spec_mixed = kw_spectrum.kw_spectrum_3d(mixed, geo)
peaks = kw_spectrum.kw_peaks(spec_mixed, n_peaks=2)

for i, pk in enumerate(peaks):
    print(f"Peak {i}: f = {pk.frequency:.1f} Hz, "
          f"dir = {np.degrees(pk.direction):.0f} deg, "
          f"speed = {pk.speed:.2f} mm/s")
```

## 4. Optical-flow velocity fields

Optical flow treats the spatial signal as a movie and estimates a
dense velocity field.  This naturally handles complex patterns
(rotations, spirals) beyond plane waves.

:::{note}
Optical flow requires `scikit-image`.  Install with
`pip install scikit-image`.
:::

```{code-cell} python
from cogpy.wave import optical_flow, vector_field

# Compute flow on a few frames of the plane wave
sig_short = sig.isel(time=slice(0, 50))
u, v = optical_flow.compute_flow(sig_short, method="ilk")

speed_map, dir_map = optical_flow.flow_to_speed_direction(u, v)
```

Browse the velocity field over time -- speed (left) and direction
(right) at each frame:

```{code-cell} python
# Subsample to ~15 frames for page size
flow_idx = np.linspace(0, speed_map.sizes["time"] - 1, 15, dtype=int)

speed_hm = hv.HoloMap(
    {float(speed_map.time.values[i]): hv.Image(
        speed_map.isel(time=i).values,
        kdims=["ML", "AP"],
        bounds=(0, 0, speed_map.sizes["ML"] - 1, speed_map.sizes["AP"] - 1),
    ) for i in flow_idx},
    kdims=["time (s)"],
).opts(hv.opts.Image(
    cmap="magma", colorbar=True, width=350, height=300, title="Speed field",
))

dir_hm = hv.HoloMap(
    {float(dir_map.time.values[i]): hv.Image(
        dir_map.isel(time=i).values,
        kdims=["ML", "AP"],
        bounds=(0, 0, dir_map.sizes["ML"] - 1, dir_map.sizes["AP"] - 1),
    ) for i in flow_idx},
    kdims=["time (s)"],
).opts(hv.opts.Image(
    cmap="hsv", colorbar=True, width=350, height=300, title="Direction field",
))

speed_hm + dir_hm
```

### Vector-field classification

Characterise the flow pattern using divergence, curl, and pattern
classification:

```{code-cell} python
u0 = u.isel(time=0)
v0 = v.isel(time=0)

div = vector_field.divergence(u0, v0, geo)
vort = vector_field.curl(u0, v0, geo)
pattern = vector_field.classify_pattern(u0, v0, geo)

div_img = hv.Image(
    div.values, kdims=["ML", "AP"],
    bounds=(0, 0, div.sizes["ML"] - 1, div.sizes["AP"] - 1),
).opts(cmap="RdBu_r", symmetric=True, colorbar=True,
       width=350, height=300, title="Divergence")

vort_img = hv.Image(
    vort.values, kdims=["ML", "AP"],
    bounds=(0, 0, vort.sizes["ML"] - 1, vort.sizes["AP"] - 1),
).opts(cmap="RdBu_r", symmetric=True, colorbar=True,
       width=350, height=300, title="Curl (vorticity)")

div_img + vort_img
```

```{code-cell} python
print(f"Classified pattern: {pattern}")
```

For a plane wave, divergence and curl should be near zero everywhere,
and the pattern should be classified as `planar`.

## 5. Statistical validation with surrogates

Wave-like patterns can appear by chance in noisy data.  Surrogate
testing establishes whether observed wave metrics are statistically
significant.

### Phase randomisation

Destroys phase coherence while preserving the power spectrum at each
channel:

```{code-cell} python
surr = surrogates.phase_randomize(sig, rng=42)

pgd_surr = phase_gradient.pgd(
    phase_gradient.hilbert_phase(surr), geo
)

real_curve = hv.Curve(
    (pgd_score.time.values, pgd_score.values),
    kdims=["Time (s)"], vdims=["PGD"], label="Real",
).opts(color="steelblue")

surr_curve = hv.Curve(
    (pgd_surr.time.values, pgd_surr.values),
    kdims=["Time (s)"], vdims=["PGD"], label="Surrogate",
).opts(color="gray", alpha=0.7)

(real_curve * surr_curve).opts(
    width=600, height=220, ylim=(0, 1.05),
    title="PGD: real vs surrogate", legend_position="top_right",
)
```

### Formal surrogate test

Test whether the mean PGD is significantly above chance:

```{code-cell} python
def mean_pgd(data):
    ph = phase_gradient.hilbert_phase(data)
    return float(phase_gradient.pgd(ph, geo).mean())

p_val, observed, null_dist = surrogates.surrogate_test(
    sig,
    estimator_fn=mean_pgd,
    n_surrogates=50,  # 200 recommended; 50 for fast docs build
    seed=42,
)

null_hist = hv.Histogram(
    np.histogram(null_dist, bins=20),
).opts(color="gray", alpha=0.7, xlabel="Mean PGD", ylabel="Count")

obs_line = hv.VLine(observed).opts(color="red", line_dash="dashed", line_width=2)

obs_label = hv.Text(
    observed + 0.01, 5, f"Observed = {observed:.3f}",
    halign="left", fontsize=10,
).opts(text_color="red")

(null_hist * obs_line * obs_label).opts(
    width=500, height=250,
    title=f"Surrogate test (p = {p_val:.4f})",
)
```

## 6. Interactive explorer (notebook only)

The following `DynamicMap` uses `cogpy.plot.hv.grid_movie` for a
full-resolution, smooth playback of the wave alongside its phase map.
It **requires a live kernel** and will not render on a static docs site.

```{code-cell} python
:tags: [skip-execution]

# NOTE: DynamicMap — requires a live kernel
from cogpy.plot.hv.xarray_hv import grid_movie

wave_dmap = grid_movie(sig, title="Plane wave")
phase_dmap = grid_movie(
    phase.isel(time=slice(0, 200)),
    cmap="twilight", symmetric=False, title="Phase map",
)

wave_dmap + phase_dmap
```

## 7. Typical workflow with real data

A complete analysis on real ECoG data follows the same steps with
a bandpass filter added:

```python
from cogpy.io.ecog_io import load_ecog
from cogpy.preprocess.filtering.temporal import bandpass
from cogpy.wave._types import Geometry
from cogpy.wave import phase_gradient, kw_spectrum, surrogates

# 1. Load and filter
sig = load_ecog("path/to/data.bin")
sig_alpha = bandpass(sig, flo=8, fhi=12)

# 2. Define geometry
geo = Geometry.regular(dx=0.4, dy=0.4)  # electrode pitch in mm

# 3. Phase-gradient analysis
phase = phase_gradient.hilbert_phase(sig_alpha)
pgd_score = phase_gradient.pgd(phase, geo)
estimates = phase_gradient.plane_wave_fit(phase, geo)

# 4. Cross-validate with k-w spectrum
spec = kw_spectrum.kw_spectrum_3d(sig_alpha, geo)
peaks = kw_spectrum.kw_peaks(spec, n_peaks=1)

# 5. Surrogate test
def mean_pgd(data):
    ph = phase_gradient.hilbert_phase(data)
    return float(phase_gradient.pgd(ph, geo).mean())

p_val, obs, null = surrogates.surrogate_test(
    sig_alpha, mean_pgd, n_surrogates=200, seed=0
)
```

## Additional methods

The module also provides:

| Module | Purpose | Reference |
|--------|---------|-----------|
| `beamforming` | f--k / Capon beamforming for irregular layouts | Capon (1969) |
| `multitaper_nd` | Separable N-D DPSS tapers for variance reduction | Hanssen (1997) |
| `generalized_phase` | Broadband phase stabilisation for wideband signals | Davis et al. (2020) |

## Summary

| Step | Function | Module |
|------|----------|--------|
| Synthetic data | `plane_wave`, `spiral_wave`, `multi_wave` | `cogpy.wave.synthetic` |
| Analytic phase | `hilbert_phase` | `cogpy.wave.phase_gradient` |
| PGD score | `pgd` | `cogpy.wave.phase_gradient` |
| Plane-wave fit | `plane_wave_fit` | `cogpy.wave.phase_gradient` |
| k--w spectrum | `kw_spectrum_3d`, `kw_peaks` | `cogpy.wave.kw_spectrum` |
| Optical flow | `compute_flow`, `flow_to_speed_direction` | `cogpy.wave.optical_flow` |
| Vector analysis | `divergence`, `curl`, `classify_pattern` | `cogpy.wave.vector_field` |
| Surrogates | `phase_randomize`, `surrogate_test` | `cogpy.wave.surrogates` |

## References

- Zhang et al. (2018). *Theta and Alpha Oscillations Are Traveling Waves in the Human Neocortex*. Neuron. [DOI: 10.1016/j.neuron.2018.05.019](https://doi.org/10.1016/j.neuron.2018.05.019)
- Davis et al. (2020). *Spontaneous travelling cortical waves gate perception in behaving primates*. Nature. [DOI: 10.1038/s41586-020-2802-y](https://doi.org/10.1038/s41586-020-2802-y)
- Townsend & Gong (2018). *Detection and analysis of spatiotemporal patterns in brain activity*. PLOS Comp Biol. [DOI: 10.1371/journal.pcbi.1006643](https://doi.org/10.1371/journal.pcbi.1006643)
- Afrashteh et al. (2017). *Optical-flow analysis toolbox*. NeuroImage. [DOI: 10.1016/j.neuroimage.2017.03.034](https://doi.org/10.1016/j.neuroimage.2017.03.034)
- Capon (1969). *High-resolution frequency-wavenumber spectrum analysis*. Proc IEEE. [DOI: 10.1109/PROC.1969.7278](https://doi.org/10.1109/PROC.1969.7278)
- Hanssen (1997). *Multidimensional multitaper spectral estimation*. Signal Processing. [DOI: 10.1016/S0165-1684(97)00076-5](https://doi.org/10.1016/S0165-1684(97)00076-5)

## Next steps

- {doc}`spectral-analysis` -- spectral analysis stack (PSD, spectrograms)
- {doc}`spatial-measures` -- spatial grid measures (Moran's I, gradient anisotropy)
