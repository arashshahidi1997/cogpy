"""Shared test fixtures for TensorScope tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def small_ieeg():
    """Small iEEG dataset for fast tests."""

    example_ieeg_grid = pytest.importorskip("cogpy.datasets.entities").example_ieeg_grid
    return example_ieeg_grid(mode="small")


@pytest.fixture
def medium_ieeg():
    """Medium iEEG dataset for integration tests."""

    example_ieeg_grid = pytest.importorskip("cogpy.datasets.entities").example_ieeg_grid
    return example_ieeg_grid(mode="large")


@pytest.fixture
def synthetic_large_ieeg():
    """
    Large synthetic iEEG for performance tests.

    10 minutes, 8x8 grid, 1kHz (~3MB).
    Generated on-demand, not committed to repo.
    """

    np = pytest.importorskip("numpy")
    xr = pytest.importorskip("xarray")

    duration = 600  # 10 minutes
    fs = 1000
    n_ap, n_ml = 8, 8

    n_samples = int(duration * fs)

    rng = np.random.RandomState(42)
    data = rng.randn(n_samples, n_ap, n_ml).cumsum(axis=0)

    return xr.DataArray(
        data,
        dims=("time", "AP", "ML"),
        coords={
            "time": np.arange(n_samples) / fs,
            "AP": np.arange(n_ap),
            "ML": np.arange(n_ml),
        },
    )

