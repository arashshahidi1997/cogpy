---
title: Sliding Core Utilities
---

# Sliding windows with `cogpy.utils.sliding_core`

This short tutorial shows how to transform NumPy arrays into sliding window
views and how to compute streaming features without copying any data. The
helpers in `cogpy.utils.sliding_core` operate directly on ndarray memory,
which makes them fast building blocks for preprocessing pipelines.

## Imports and synthetic data

We will build a toy ECoG-like array with three channels so the examples can run
quickly inside the documentation build.

```{code-cell} python
import numpy as np

from cogpy.utils.sliding_core import (
    running_blockwise,
    running_reduce,
    sliding_window,
    sliding_window_naive,
)

fs = 1_000  # Hz
duration = 2.0  # seconds
samples = int(fs * duration)
t = np.arange(samples) / fs

rng = np.random.default_rng(7)
signals = np.vstack(
    [
        np.sin(2 * np.pi * 6 * t),
        0.8 * np.sin(2 * np.pi * 10 * t + np.pi / 4),
        0.5 * np.sin(2 * np.pi * 22 * t),
    ]
)
noise = 0.15 * rng.standard_normal(signals.shape)
data = (signals + noise).astype(np.float32)
data.shape
```

## Non-copying sliding windows

The `sliding_window` function inserts a window axis that steps over the time
dimension while keeping the original data buffer intact.

```{code-cell} python
WINDOW = 200  # samples (200 ms at 1 kHz)
STEP = 50     # 75% overlap

windows = sliding_window(data, window_size=WINDOW, window_step=STEP, axis=1)
windows.shape, np.shares_memory(windows, data)
```

Notice that the view is read-only and references the source array. You can still
create a copy when needed, but for most downstream operations you can treat it
as a windowed block. If you need a quick correctness check, compare against the
reference implementation:

```{code-cell} python
ref = sliding_window_naive(data, window_size=WINDOW, window_step=STEP, axis=1)
np.allclose(windows, ref)
```

## Applying reducers per window

`running_reduce` wraps `sliding_window` and expects any reducer that accepts an
`axis` keyword (NumPy ufuncs, statistics helpers, or your own functions). The
example below computes the root-mean-square (RMS) value of each channel inside
every window.

```{code-cell} python
def rms(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.mean(arr**2, axis=axis))

window_rms = running_reduce(
    data,
    window_size=WINDOW,
    window_step=STEP,
    reducer=rms,
    axis=1,  # time axis
)
window_rms.shape
```

The output shape is `(n_channels, n_windows)` because the reducer removes the
per-window sample axis. Here are the first few RMS values per channel (rounded
for brevity):

```{code-cell} python
np.round(window_rms[:, :4], 3)
```

You can swap in any reducer such as `np.max`, `np.median`, or a custom function
that also accepts additional keyword arguments via `reducer_kwargs`.

## Block-wise feature extraction

For more advanced processing you might need to perform computations that return
multi-dimensional features per window (e.g., FFT magnitudes, PCA scores, or a
machine-learning model). `running_blockwise` handles this pattern by treating
every window as the "core block" provided to your function.

The snippet below extracts the average theta-band (6–12 Hz) amplitude for each
channel per window using the real FFT. The helper returns a `(n_windows,
n_channels)` array because our function produces one mean value per channel.

```{code-cell} python
freqs = np.fft.rfftfreq(WINDOW, 1 / fs)
theta_mask = (freqs >= 6) & (freqs <= 12)

def theta_band_amplitude(block: np.ndarray) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(block, axis=-1))
    return spectrum[:, theta_mask].mean(axis=-1)

theta_features = running_blockwise(
    data,
    window_size=WINDOW,
    window_step=STEP,
    func=theta_band_amplitude,
    axis=1,
)
theta_features.shape
```

```{code-cell} python
np.round(theta_features[:5], 2)
```

Because `running_blockwise` computes the feature shape from the first window, it
can cache an output array and reuse it for the remaining windows. This keeps the
loop in C/NumPy while still letting you write the per-window logic in pure
Python.

---

You now have read-only window views plus two helper APIs for running reducers
and block-wise feature extraction. Combine them to stitch together your own
feature pipelines without juggling manual loop bookkeeping.
