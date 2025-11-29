"""Test and timing harness that compares every sliding window variant.

The script loads versions 0-3, runs them on shared inputs, asserts they
produce identical outputs, and prints timing comparisons for each case.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent


def _load_module(alias: str, filename: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


MODULES = {
    "v0": _load_module("sliding_version_0", "sliding-version-0-minimal.py"),
    "v1": _load_module("sliding_version_1", "sliding-version-1-axis.py"),
    "v2": _load_module("sliding_version_2", "sliding-version-2-dask.py"),
    "v3": _load_module("sliding_version_3", "sliding-version-3-xarray.py"),
}


def _time_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def _print_case_results(case_name: str, timings: List[Tuple[str, float]]):
    print(f"\nCase: {case_name}")
    for label, elapsed in timings:
        print(f"  {label:<30s} {elapsed*1e3:8.2f} ms")


def run_case(case_name: str, array: np.ndarray, *, window_size: int, window_step: int, axis: int):
    timings: List[Tuple[str, float]] = []

    ref, t_ref = _time_call(
        MODULES["v1"].sliding_window_naive, array, window_size, window_step, axis=axis
    )
    timings.append(("v1.naive (reference)", t_ref))

    if axis == array.ndim - 1:
        v0_naive, t_naive0 = _time_call(
            MODULES["v0"].sliding_window_naive, array, window_size, window_step
        )
        assert np.allclose(v0_naive, ref)
        timings.append(("v0.naive", t_naive0))

        view0, t_view0 = _time_call(
            MODULES["v0"].sliding_window_da, array, window_size, window_step
        )
        assert np.allclose(view0, ref)
        timings.append(("v0.as_strided", t_view0))

    v1_view, t_v1 = _time_call(
        MODULES["v1"].sliding_window_da, array, window_size, window_step, axis=axis
    )
    assert np.allclose(v1_view, ref)
    timings.append(("v1.as_strided", t_v1))

    v2_view, t_v2 = _time_call(
        MODULES["v2"].sliding_window_da, array, window_size, window_step, axis=axis
    )
    assert np.allclose(v2_view, ref)
    timings.append(("v2.NumPy", t_v2))

    mod_v2 = MODULES["v2"]
    has_dask = getattr(mod_v2, "da", None) is not None
    if has_dask:
        x_da = mod_v2.da.from_array(array, chunks=tuple(max(1, s // 4) for s in array.shape))
        windows_da, t_build = _time_call(
            mod_v2.sliding_window_da, x_da, window_size, window_step, axis=axis
        )
        start = time.perf_counter()
        computed = windows_da.compute()
        t_compute = time.perf_counter() - start
        assert np.allclose(computed, ref)
        timings.append(("v2.Dask graph", t_build))
        timings.append(("v2.Dask compute", t_compute))
    else:
        timings.append(("v2.Dask graph", float("nan")))

    dims = [f"dim_{i}" for i in range(array.ndim)]
    xr_input = xr.DataArray(array, dims=dims)
    window_dim = "window"
    sample_dim = "sample"
    xr_view, t_xr = _time_call(
        MODULES["v3"].sliding_window_xr,
        xr_input,
        window_size,
        window_step,
        dim=dims[axis],
        window_dim=window_dim,
        sample_dim=sample_dim,
    )
    assert np.allclose(xr_view.data, ref)
    timings.append(("v3.xarray", t_xr))

    _print_case_results(case_name, timings)


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    cases = {
        "batch_last": {
            "array": rng.normal(size=(32, 4096)),
            "window_size": 128,
            "window_step": 32,
            "axis": -1,
        },
        "batch_first": {
            "array": rng.normal(size=(2048, 4)),
            "window_size": 128,
            "window_step": 64,
            "axis": 0,
        },
    }

    for name, cfg in cases.items():
        run_case(name, **cfg)
