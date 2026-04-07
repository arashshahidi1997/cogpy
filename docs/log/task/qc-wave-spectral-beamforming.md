---
title: "QC: wave.kw_spectrum + beamforming + multitaper_nd vs paper equations"
date: 2026-04-01
status: done
actionable: true
project_primary: "cogpy"
tags: [task, qc, travelling-waves]
---

## Prompt

Quality-check `src/cogpy/wave/kw_spectrum.py`, `src/cogpy/wave/beamforming.py`, and `src/cogpy/wave/multitaper_nd.py` against their paper references. These modules have no codio MATLAB reference — they were implemented from first principles.

### Steps

1. Read `src/cogpy/wave/kw_spectrum.py`:
   - Verify 3D FFT produces correct wavenumber and frequency axes (check fftfreq scaling with geometry)
   - Verify peak extraction returns correct direction = atan2(ky, kx) and speed = omega/|k|
   - Test with a known synthetic plane wave and verify the peak location matches expected (kx, ky, freq)
   - Check windowing (Hann or similar) is applied correctly in all 3 dimensions

2. Read `src/cogpy/wave/beamforming.py`:
   - Verify conventional beamformer: P(s) = a(s)^H R a(s) where a is steering vector, R is CSD matrix
   - Verify Capon/MVDR: P(s) = 1 / (a(s)^H R^{-1} a(s)) — check matrix inversion regularization
   - Steering vector: a_i(s) = exp(-j 2pi f s . r_i) — verify sign convention and coordinate system
   - Test with synthetic plane wave on irregular array coords

3. Read `src/cogpy/wave/multitaper_nd.py`:
   - Verify separable construction: taper_nd = outer(dpss_x, dpss_y, dpss_t)
   - Check bandwidth parameters map correctly to scipy.signal.windows.dpss NW parameter
   - Verify integration with kw_spectrum produces reduced variance

4. Fix any issues found. Run all related tests.

### References
- Capon 1969, DOI: 10.1109/PROC.1969.7278
- Thomson 1982, DOI: 10.1109/PROC.1982.12433
- Hanssen 1997, DOI: 10.1016/S0165-1684(97)00076-5


## Batch Result

- status: done
- batch queue_id: `f9eff0043988`
- session: `d738b2f9-7755-4c25-a7fa-4e9b25695419`
- batch duration: 1349.7s
