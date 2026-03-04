"""
Performance benchmarks for TensorScope.

These tests are intended to be run explicitly:

    python -m pytest code/lib/cogpy/tests/core/plot/tensorscope/benchmarks.py -m benchmark -v -s

They enforce interactive responsiveness budgets on a given machine. Because
performance depends on hardware and environment, the thresholds are
intentionally conservative and can be relaxed via:

    TENSORSCOPE_BENCH_RELAX=1
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from cogpy.datasets.entities import example_ieeg_grid
from cogpy.core.plot.tensorscope import TensorScopeApp, TensorScopeState
from cogpy.core.plot.tensorscope.layers import TimeseriesLayer


def _relax_factor() -> float:
    """Return a multiplicative relax factor for timings."""
    if os.environ.get("TENSORSCOPE_BENCH_RELAX", "").strip():
        return 3.0
    return 1.0


class Timer:
    """Simple timer context manager."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


@pytest.mark.benchmark
def test_state_creation_speed():
    """Benchmark state initialization."""
    data = example_ieeg_grid(mode="small")

    with Timer() as t:
        _ = TensorScopeState(data)

    ms = t.elapsed * 1000
    print(f"\nState creation: {ms:.1f}ms")
    assert t.elapsed < 0.5 * _relax_factor(), f"State creation too slow: {t.elapsed:.2f}s"


@pytest.mark.benchmark
def test_cursor_update_speed():
    """Benchmark cursor movement (target: <100ms average)."""
    data = example_ieeg_grid(mode="small")
    state = TensorScopeState(data)

    # Warmup
    state.current_time = float(state.time_window.bounds[0]) if state.time_window is not None else 0.0

    times = []
    tmin = float(state.time_window.bounds[0]) if state.time_window is not None else 0.0
    tmax = float(state.time_window.bounds[1]) if state.time_window is not None else 10.0
    for t in np.linspace(tmin, tmax, 50):
        with Timer() as timer:
            state.current_time = float(t)
        times.append(timer.elapsed)

    avg_ms = float(np.mean(times)) * 1000
    max_ms = float(np.max(times)) * 1000
    print(f"\nCursor update: avg={avg_ms:.1f}ms, max={max_ms:.1f}ms")
    assert avg_ms < 100 * _relax_factor(), f"Cursor updates too slow: {avg_ms:.1f}ms"


@pytest.mark.benchmark
def test_selection_update_speed():
    """Benchmark selection changes (selection wiring + viewer update)."""
    pytest.importorskip("panel")
    data = example_ieeg_grid(mode="small")
    state = TensorScopeState(data)
    _ = TimeseriesLayer(state)  # ensure watchers are registered

    with Timer() as t:
        for ap, ml in [(0, 0), (1, 1), (2, 2), (3, 3)]:
            state.channel_grid.select_cell(ap, ml)

    ms = t.elapsed * 1000
    print(f"\nSelection (4 channels): {ms:.1f}ms")
    assert t.elapsed < 0.3 * _relax_factor(), f"Selection too slow: {t.elapsed:.2f}s"


@pytest.mark.benchmark
def test_window_processing_speed():
    """Benchmark windowed data processing (target: <500ms)."""
    try:
        data = example_ieeg_grid(mode="large")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"large example data unavailable: {e}")

    state = TensorScopeState(data)
    tmin = float(state.time_window.bounds[0]) if state.time_window is not None else 0.0
    t0, t1 = tmin + 5.0, tmin + 7.0

    with Timer() as t:
        _ = state.processing.get_window(t0, t1)

    ms = t.elapsed * 1000
    print(f"\nWindow processing (2s window): {ms:.1f}ms")
    assert t.elapsed < 0.5 * _relax_factor(), f"Window processing too slow: {t.elapsed:.2f}s"


@pytest.mark.benchmark
def test_app_initialization_speed():
    """Benchmark full app creation (state + layers)."""
    pytest.importorskip("panel")
    data = example_ieeg_grid(mode="small")

    with Timer() as t:
        _ = (
            TensorScopeApp(data)
            .add_layer("timeseries")
            .add_layer("spatial_map")
            .add_layer("selector")
            .add_layer("processing")
        )

    ms = t.elapsed * 1000
    print(f"\nApp initialization: {ms:.1f}ms")
    assert t.elapsed < 2.0 * _relax_factor(), f"App init too slow: {t.elapsed:.2f}s"


@pytest.mark.benchmark
def test_memory_footprint():
    """Benchmark memory usage (requires psutil)."""
    psutil = pytest.importorskip("psutil")
    import gc

    process = psutil.Process()
    gc.collect()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    try:
        data = example_ieeg_grid(mode="large")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"large example data unavailable: {e}")

    _ = TensorScopeApp(data).add_layer("timeseries").add_layer("spatial_map")

    gc.collect()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_used = float(mem_after - mem_before)

    print(f"\nMemory footprint: {mem_used:.1f}MB")
    assert mem_used < 500 * _relax_factor(), f"Memory usage too high: {mem_used:.1f}MB"


@pytest.mark.benchmark
def test_layer_dispose_cleanup():
    """Memory leak smoke test: repeated create/dispose cycles (requires psutil)."""
    psutil = pytest.importorskip("psutil")
    import gc

    process = psutil.Process()
    data = example_ieeg_grid(mode="small")

    gc.collect()
    mem_before = process.memory_info().rss

    for _ in range(100):
        state = TensorScopeState(data)
        layer = TimeseriesLayer(state)
        layer.dispose()
        del layer
        del state

    gc.collect()
    mem_after = process.memory_info().rss
    mem_growth_mb = (mem_after - mem_before) / 1024 / 1024

    print(f"\nMemory growth (100 cycles): {mem_growth_mb:.1f}MB")
    assert mem_growth_mb < 50 * _relax_factor(), f"Memory leak detected: {mem_growth_mb:.1f}MB growth"


def run_benchmarks() -> None:
    """Run all benchmarks and print report."""
    print("\n" + "=" * 60)
    print("TensorScope Performance Benchmarks")
    print("=" * 60)

    import pytest as _pytest

    _pytest.main([__file__, "-v", "-m", "benchmark", "-s"])

    print("\n" + "=" * 60)
    print("Benchmark Report Complete")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmarks()

