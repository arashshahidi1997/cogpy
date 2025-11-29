import numpy as np
from numpy.lib.stride_tricks import as_strided


def sliding_window_da(x: np.ndarray, window_size: int, window_step: int = 1) -> np.ndarray:
    """
    Minimal sliding-window view along the last axis of `x`.

    Assumptions
    -----------
    - `x` is a NumPy ndarray.
    - The last axis is the time axis.
    - No Dask, no xarray, no extra safety checks.
    - Returns a view (no copies) using `as_strided`.

    Parameters
    ----------
    x : np.ndarray
        Input array. Sliding windows are taken along the last axis.
    window_size : int
        Number of samples in each window.
    window_step : int, optional
        Step between consecutive windows (in samples). Default is 1.

    Returns
    -------
    np.ndarray
        View of shape (..., n_windows, window_size), where

            n_windows = 1 + (N - window_size) // window_step

        and N is the length of the last axis of `x`.
    """
    x = np.asarray(x)
    window_size = int(window_size)
    window_step = int(window_step)

    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    if window_step <= 0:
        raise ValueError(f"window_step must be > 0, got {window_step}")

    *batch_shape, N = x.shape
    if N < window_size:
        raise ValueError(f"Last axis length {N} < window_size={window_size}")

    # Number of windows
    n_windows = 1 + (N - window_size) // window_step

    # Strides: keep batch strides, and for the last axis create a
    # (n_windows, window_size) view with step between windows.
    stride_time = x.strides[-1]
    new_shape = tuple(batch_shape) + (n_windows, window_size)
    new_strides = x.strides[:-1] + (window_step * stride_time, stride_time)

    # read-only view for safety
    x_win = as_strided(x, shape=new_shape, strides=new_strides, writeable=False)
    return x_win

def sliding_window_naive(x: np.ndarray, window_size: int, window_step: int = 1) -> np.ndarray:
    """
    Naive implementation using explicit Python loops and copies,
    for comparison with sliding_window_da.
    """
    x = np.asarray(x)
    *batch_shape, N = x.shape
    n_windows = 1 + (N - window_size) // window_step

    # We'll build the output explicitly
    out_shape = tuple(batch_shape) + (n_windows, window_size)
    out = np.empty(out_shape, dtype=x.dtype)

    if not batch_shape:
        # 1D case
        for i in range(n_windows):
            start = i * window_step
            out[i, :] = x[start : start + window_size]
    else:
        # Flatten batch dims, loop over them in Python
        x_flat = x.reshape(-1, N)
        out_flat = out.reshape(-1, n_windows, window_size)
        for b in range(x_flat.shape[0]):
            row = x_flat[b]
            for i in range(n_windows):
                start = i * window_step
                out_flat[b, i, :] = row[start : start + window_size]

    return out


if __name__ == "__main__":
    import time

    rng = np.random.default_rng(0)

    # Example shapes
    n_channels = 64
    n_time = 1_000_000  # 1e6 samples
    window_size = 256
    window_step = 64

    x = rng.standard_normal((n_channels, n_time), dtype=np.float64)

    # Warm-up
    w1 = sliding_window_da(x, window_size, window_step)
    w2 = sliding_window_naive(x, window_size, window_step)
    # Sanity check: same values
    assert np.allclose(w1, w2)

    print("Input shape:", x.shape)
    print("Windowed shape:", w1.shape)
    print()

    # Timing: as_strided version
    t0 = time.perf_counter()
    w1 = sliding_window_da(x, window_size, window_step)
    t1 = time.perf_counter()
    print(f"sliding_window_da (as_strided) time: {t1 - t0:.4f} s")

    # Timing: naive copy version
    t0 = time.perf_counter()
    w2 = sliding_window_naive(x, window_size, window_step)
    t1 = time.perf_counter()
    print(f"sliding_window_naive (python loops) time: {t1 - t0:.4f} s")

    # To make sure the result is actually used (avoid overly aggressive optimization)
    print("Mean of w1:", float(w1.mean()))
    print("Mean of w2:", float(w2.mean()))

