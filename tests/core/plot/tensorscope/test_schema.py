"""Tests for schema validation."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cogpy.core.plot.tensorscope.schema import (
    SchemaError,
    flatten_grid_to_channels,
    validate_and_normalize_grid,
)


def test_validate_correct_schema():
    """Test validation passes for correct schema."""
    data = xr.DataArray(
        np.random.randn(100, 8, 8),
        dims=("time", "AP", "ML"),
        coords={
            "time": np.arange(100),
            "AP": np.arange(8),
            "ML": np.arange(8),
        },
    )

    result = validate_and_normalize_grid(data)
    assert result.dims == ("time", "AP", "ML")


def test_validate_transposes_wrong_order():
    """Test validation corrects dimension order."""
    data = xr.DataArray(
        np.random.randn(100, 8, 8),
        dims=("time", "ML", "AP"),
        coords={
            "time": np.arange(100),
            "ML": np.arange(8),
            "AP": np.arange(8),
        },
    )

    result = validate_and_normalize_grid(data)
    assert result.dims == ("time", "AP", "ML")


def test_validate_rejects_missing_dims():
    """Test validation rejects missing dimensions."""
    data = xr.DataArray(
        np.random.randn(100, 64),
        dims=("time", "channel"),
    )

    with pytest.raises(SchemaError, match="must have dimensions"):
        validate_and_normalize_grid(data)


def test_validate_rejects_non_monotonic_time():
    """Test validation rejects non-monotonic time."""
    data = xr.DataArray(
        np.random.randn(10, 8, 8),
        dims=("time", "AP", "ML"),
        coords={
            "time": np.array([0, 1, 2, 5, 4, 6, 7, 8, 9, 10]),
            "AP": np.arange(8),
            "ML": np.arange(8),
        },
    )

    with pytest.raises(SchemaError, match="monotonically increasing"):
        validate_and_normalize_grid(data)


def test_flatten_grid_row_major():
    """Test flattening uses row-major convention."""
    data = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("time", "AP", "ML"),
        coords={
            "time": [0, 1],
            "AP": [0, 1, 2],
            "ML": [0, 1, 2, 3],
        },
    )

    flat = flatten_grid_to_channels(data)

    assert flat.dims == ("time", "channel")
    assert len(flat.channel) == 12

    assert flat.AP.values[0] == 0
    assert flat.ML.values[0] == 0

    assert flat.AP.values[5] == 1
    assert flat.ML.values[5] == 1

    assert flat.AP.values[11] == 2
    assert flat.ML.values[11] == 3

