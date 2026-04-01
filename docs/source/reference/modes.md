# Small vs Large Modes (Debugging vs Responsiveness)

Bundles and example generators should support two modes:

## `mode="small"` (quick debug)

**Goals**
- Starts fast (< 1–2s on a laptop/workstation).
- Low memory footprint.
- Enough structure to exercise selection, stacking, downsampling, and basic overlays.

**Typical sizes (guidelines)**
- Grid iEEG: `n_ap=8`, `n_ml=8`, `n_time=20_000` (float32 recommended)
- Spectrogram: `nml=10`, `nap=10`, `nt=80`, `nf=60`
- Bursts: `n_peaks=5–20`

## `mode="large"` (responsiveness check)

**Goals**
- Stresses GUI rendering and interaction (pan/zoom, selection changes) without being “impossible” to allocate.
- Large enough to reveal performance problems (downsampling budgets, Dask materialization, expensive transposes).

**Typical sizes (guidelines)**
- Grid iEEG: `n_ap=16`, `n_ml=16`, `n_time=200_000` (float32 recommended)
  - This is ~256 × 200k = 51.2M samples (~205MB as float32), big enough to matter but not instantly OOM.
- Multichannel (non-grid): `n_channel=32–64`, `n_time=1_000_000` (float32 recommended)
- Spectrogram: `nml=16`, `nap=16`, `nt=200–500`, `nf=100–200`

## Implementation notes (for when we implement)

- These defaults are implemented in `cogpy.datasets.entities` and used by `cogpy.datasets.gui_bundles`.
- `ieeg_grid_bundle(..., large_backend="dask")` is supported when `dask` is installed; it currently wraps the generated array in a dask array to support later “windowed materialization” work.

- Avoid `np.random.randn(n_channel, n_time)` in “large” if it creates >GB arrays unintentionally.
  Prefer:
  - float32,
  - chunked/Dask-backed arrays where possible,
  - or “large time, moderate channels” profiles.
- Always document the chosen defaults and estimated memory use per mode.
