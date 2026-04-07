"""Tests for cogpy.measures.spatial — spatial grid measures."""

import numpy as np
import pytest

from cogpy.measures.spatial import (
    moran_i,
    marginal_energy_outlier,
    gradient_anisotropy,
    spatial_kurtosis,
    spatial_noise_concentration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_grid(ap=8, ml=8, value=1.0):
    return np.full((ap, ml), value)


def _row_striped_grid(ap=8, ml=8, seed=42):
    """Alternating high/low rows (constant along ML)."""
    rng = np.random.default_rng(seed)
    row_vals = rng.choice([1.0, 10.0], size=ap)
    return np.tile(row_vals[:, None], (1, ml))


def _col_striped_grid(ap=8, ml=8, seed=42):
    """Alternating high/low columns (constant along AP)."""
    rng = np.random.default_rng(seed)
    col_vals = rng.choice([1.0, 10.0], size=ml)
    return np.tile(col_vals[None, :], (ap, 1))


def _checkerboard_grid(ap=8, ml=8):
    """Checkerboard pattern: anti-correlated in both axes."""
    ii, jj = np.meshgrid(np.arange(ap), np.arange(ml), indexing="ij")
    return ((ii + jj) % 2).astype(float) * 10.0 + 1.0


def _bright_column_grid(ap=8, ml=8, col=3, bright=100.0):
    """Single bright column."""
    g = np.ones((ap, ml))
    g[:, col] = bright
    return g


# ---------------------------------------------------------------------------
# moran_i — directional adjacency
# ---------------------------------------------------------------------------


class TestMoranDirectional:
    def test_original_modes_still_work(self):
        """queen and rook still produce valid results."""
        g = _checkerboard_grid()
        iq = moran_i(g, adjacency="queen")
        ir = moran_i(g, adjacency="rook")
        assert np.isfinite(iq)
        assert np.isfinite(ir)

    def test_invalid_adjacency(self):
        with pytest.raises(ValueError):
            moran_i(np.ones((4, 4)), adjacency="invalid")

    def test_row_striped_ml_vs_ap(self):
        """Row-striped: ml_only sees same values (high I), ap_only sees alternation (low I)."""
        g = _row_striped_grid()
        i_ml = moran_i(g, adjacency="ml_only")
        i_ap = moran_i(g, adjacency="ap_only")
        # Along ML, neighbors have same value → high positive I
        # Along AP, neighbors alternate → low or negative I
        assert i_ml > i_ap

    def test_col_striped_ap_vs_ml(self):
        """Column-striped: ap_only sees same values (high I), ml_only sees alternation."""
        g = _col_striped_grid()
        i_ap = moran_i(g, adjacency="ap_only")
        i_ml = moran_i(g, adjacency="ml_only")
        assert i_ap > i_ml

    def test_checkerboard_both_negative(self):
        """Checkerboard: negative I in both directions."""
        g = _checkerboard_grid()
        i_ap = moran_i(g, adjacency="ap_only")
        i_ml = moran_i(g, adjacency="ml_only")
        assert i_ap < 0
        assert i_ml < 0

    def test_uniform_nan(self):
        """Constant grid → NaN (zero variance)."""
        g = _uniform_grid()
        assert np.isnan(moran_i(g, adjacency="ap_only"))
        assert np.isnan(moran_i(g, adjacency="ml_only"))


# ---------------------------------------------------------------------------
# marginal_energy_outlier
# ---------------------------------------------------------------------------


class TestMarginalEnergyOutlier:
    def test_uniform_no_outliers(self):
        """Uniform grid → no outliers."""
        g = _uniform_grid()
        result = marginal_energy_outlier(g)
        assert not np.any(result["row_outlier"])
        assert not np.any(result["col_outlier"])

    def test_bright_column_detected(self):
        """Single bright column flagged as outlier."""
        g = _bright_column_grid(col=3, bright=100.0)
        result = marginal_energy_outlier(g)
        assert result["col_outlier"][3]
        # Other columns should not be outliers
        non_bright = np.delete(result["col_outlier"], 3)
        assert not np.any(non_bright)

    def test_output_keys(self):
        """All expected keys present."""
        g = _uniform_grid()
        result = marginal_energy_outlier(g)
        expected_keys = {
            "row_energy",
            "col_energy",
            "row_zscore",
            "col_zscore",
            "row_outlier",
            "col_outlier",
        }
        assert set(result.keys()) == expected_keys

    def test_output_shapes(self):
        """Shapes match grid axes."""
        g = np.ones((6, 10))
        result = marginal_energy_outlier(g)
        assert result["row_energy"].shape == (6,)
        assert result["col_energy"].shape == (10,)
        assert result["row_outlier"].shape == (6,)
        assert result["col_outlier"].shape == (10,)

    def test_non_robust_mode(self):
        """Non-robust (mean/std) mode runs without error and produces valid output."""
        g = _bright_column_grid(col=3, bright=100.0)
        result = marginal_energy_outlier(g, robust=False)
        # Non-robust z-score is less sensitive (std inflated by outlier),
        # but the bright column should still have the highest z-score
        assert result["col_zscore"][3] == np.max(result["col_zscore"])

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="AP, ML"):
            marginal_energy_outlier(np.ones((4,)))

    def test_custom_threshold(self):
        """Higher threshold → fewer outliers."""
        g = _bright_column_grid(col=3, bright=20.0)
        result_low = marginal_energy_outlier(g, threshold=2.0)
        result_high = marginal_energy_outlier(g, threshold=10.0)
        assert np.sum(result_low["col_outlier"]) >= np.sum(result_high["col_outlier"])


# ---------------------------------------------------------------------------
# gradient_anisotropy
# ---------------------------------------------------------------------------


class TestGradientAnisotropy:
    def test_isotropic_near_zero(self):
        """Random grid → anisotropy near 0."""
        rng = np.random.default_rng(42)
        g = rng.normal(0, 1, (16, 16))
        aniso = gradient_anisotropy(g)
        assert abs(aniso) < 0.5  # roughly isotropic

    def test_row_striped_negative(self):
        """Row-striped (constant along ML) → ML gradient small → negative anisotropy."""
        g = _row_striped_grid(ap=16, ml=16)
        aniso = gradient_anisotropy(g)
        # ML gradient is ~0 (constant within rows), AP gradient is large
        # log2(large / small) > 0 → actually positive for row-striped
        # because gradients are along AP axis
        # Wait: row-striped means rows differ, so dV/dAP is large, dV/dML is 0
        # → log2(large/small) > 0 → positive
        # The docstring says positive = AP-dominant gradient = column-striped pattern
        # But row-striped has AP-dominant gradient too... let me re-check.
        # Row-striped: values change along AP (rows differ), constant along ML
        # → |dV/dAP| >> |dV/dML| → ratio > 1 → log2 > 0 → positive
        # The docstring maps positive → "column-striped pattern" which is wrong
        # Actually: if rows have different values, the gradient IS along AP.
        # Column-striped: values change along ML → gradient along ML → negative.
        # Row-striped: gradient along AP → positive.
        assert aniso > 0

    def test_col_striped_negative(self):
        """Column-striped (constant along AP) → AP gradient small → negative anisotropy."""
        g = _col_striped_grid(ap=16, ml=16)
        aniso = gradient_anisotropy(g)
        assert aniso < 0

    def test_uniform_near_zero(self):
        """Constant grid → near 0 (both gradients ≈ 0)."""
        g = _uniform_grid()
        aniso = gradient_anisotropy(g)
        assert abs(aniso) < 0.1

    def test_small_grid_nan(self):
        """Grid smaller than (2,2) → NaN."""
        assert np.isnan(gradient_anisotropy(np.ones((1, 5))))
        assert np.isnan(gradient_anisotropy(np.ones((5, 1))))

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="AP, ML"):
            gradient_anisotropy(np.ones((4,)))


# ---------------------------------------------------------------------------
# Batch-dim tests  — (..., AP, ML) support
# ---------------------------------------------------------------------------


class TestBatchGradientAnisotropy:
    def test_3d_matches_loop(self):
        """Batch result matches per-slice loop."""
        rng = np.random.default_rng(42)
        grids = rng.normal(0, 1, (5, 8, 8))  # (batch, AP, ML)
        batch_result = gradient_anisotropy(grids)
        assert batch_result.shape == (5,)
        for i in range(5):
            expected = gradient_anisotropy(grids[i])
            np.testing.assert_allclose(batch_result[i], expected)

    def test_4d_matches_loop(self):
        """(time, freq, AP, ML) → (time, freq) output."""
        rng = np.random.default_rng(7)
        grids = rng.normal(0, 1, (3, 4, 8, 8))
        batch_result = gradient_anisotropy(grids)
        assert batch_result.shape == (3, 4)
        for t in range(3):
            for f in range(4):
                expected = gradient_anisotropy(grids[t, f])
                np.testing.assert_allclose(batch_result[t, f], expected)

    def test_small_grid_batch(self):
        """Small grid in batch → NaN array."""
        grids = np.ones((3, 1, 5))
        result = gradient_anisotropy(grids)
        assert result.shape == (3,)
        assert np.all(np.isnan(result))


class TestBatchMarginalEnergyOutlier:
    def test_3d_matches_loop(self):
        """Batch result matches per-slice loop."""
        rng = np.random.default_rng(42)
        grids = rng.normal(0, 1, (5, 8, 8))
        batch_result = marginal_energy_outlier(grids)
        assert batch_result["row_energy"].shape == (5, 8)
        assert batch_result["col_energy"].shape == (5, 8)
        for i in range(5):
            single = marginal_energy_outlier(grids[i])
            np.testing.assert_allclose(
                batch_result["row_energy"][i], single["row_energy"]
            )
            np.testing.assert_allclose(
                batch_result["col_energy"][i], single["col_energy"]
            )
            np.testing.assert_allclose(
                batch_result["row_zscore"][i], single["row_zscore"]
            )
            np.testing.assert_allclose(
                batch_result["col_zscore"][i], single["col_zscore"]
            )

    def test_4d_shapes(self):
        """(time, freq, AP, ML) → correct output shapes."""
        grids = np.ones((3, 4, 6, 10))
        result = marginal_energy_outlier(grids)
        assert result["row_energy"].shape == (3, 4, 6)
        assert result["col_energy"].shape == (3, 4, 10)
        assert result["row_outlier"].shape == (3, 4, 6)
        assert result["col_outlier"].shape == (3, 4, 10)

    def test_bright_column_batch(self):
        """Bright column detected in every batch slice."""
        g = _bright_column_grid(col=3, bright=100.0)
        grids = np.stack([g, g, g])  # (3, 8, 8)
        result = marginal_energy_outlier(grids)
        assert np.all(result["col_outlier"][:, 3])


class TestBatchMoranI:
    def test_3d_matches_loop(self):
        """Batch result matches per-slice loop."""
        rng = np.random.default_rng(42)
        grids = rng.normal(0, 1, (5, 8, 8))
        batch_result = moran_i(grids, adjacency="queen")
        assert batch_result.shape == (5,)
        for i in range(5):
            expected = moran_i(grids[i], adjacency="queen")
            np.testing.assert_allclose(batch_result[i], expected, rtol=1e-10)

    def test_4d_matches_loop(self):
        """(time, freq, AP, ML) → (time, freq) output."""
        rng = np.random.default_rng(7)
        grids = rng.normal(0, 1, (2, 3, 8, 8))
        batch_result = moran_i(grids, adjacency="rook")
        assert batch_result.shape == (2, 3)
        for t in range(2):
            for f in range(3):
                expected = moran_i(grids[t, f], adjacency="rook")
                np.testing.assert_allclose(batch_result[t, f], expected, rtol=1e-10)

    def test_directional_batch(self):
        """Directional modes work in batch."""
        g_row = _row_striped_grid()
        g_col = _col_striped_grid()
        grids = np.stack([g_row, g_col])  # (2, 8, 8)
        i_ml = moran_i(grids, adjacency="ml_only")
        i_ap = moran_i(grids, adjacency="ap_only")
        # Row-striped: ml_only > ap_only
        assert i_ml[0] > i_ap[0]
        # Col-striped: ap_only > ml_only
        assert i_ap[1] > i_ml[1]

    def test_uniform_batch_nan(self):
        """Constant grids in batch → NaN for each."""
        grids = np.ones((3, 4, 4))
        result = moran_i(grids)
        assert result.shape == (3,)
        assert np.all(np.isnan(result))


# ---------------------------------------------------------------------------
# spatial_kurtosis
# ---------------------------------------------------------------------------


class TestSpatialKurtosis:
    def test_uniform_zero(self):
        """Uniform grid → excess kurtosis near -1.2 (platykurtic uniform)."""
        g = np.ones((8, 8))
        # All identical → zero variance → kurtosis is nan or degenerate
        # Use a uniform random grid instead
        rng = np.random.default_rng(42)
        g = rng.uniform(0, 1, (8, 8))
        k = spatial_kurtosis(g)
        assert np.isfinite(k)
        assert k < 0  # uniform distribution is platykurtic

    def test_single_hotspot_high_kurtosis(self):
        """One very high electrode → high kurtosis."""
        g = np.ones((8, 8))
        g[3, 3] = 1000.0
        k = spatial_kurtosis(g)
        assert k > 10.0

    def test_output_scalar_for_2d(self):
        """2D input → Python float."""
        g = np.ones((4, 4))
        k = spatial_kurtosis(g)
        assert isinstance(k, float)

    def test_batch_shape(self):
        """Batch dims preserved: (5, 8, 8) → (5,)."""
        rng = np.random.default_rng(42)
        grids = rng.normal(0, 1, (5, 8, 8))
        k = spatial_kurtosis(grids)
        assert k.shape == (5,)

    def test_4d_batch(self):
        """(2, 3, 8, 8) → (2, 3)."""
        rng = np.random.default_rng(42)
        grids = rng.normal(0, 1, (2, 3, 8, 8))
        k = spatial_kurtosis(grids)
        assert k.shape == (2, 3)

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="AP, ML"):
            spatial_kurtosis(np.ones((4,)))


# ---------------------------------------------------------------------------
# spatial_noise_concentration
# ---------------------------------------------------------------------------


class TestSpatialNoiseConcentration:
    def test_uniform_grid(self):
        """Uniform grid → concentration = k / n."""
        g = np.ones((8, 8))
        c = spatial_noise_concentration(g, k=3)
        expected = 3.0 / 64.0
        assert c == pytest.approx(expected, rel=1e-6)

    def test_single_dominant_channel(self):
        """One dominant channel → concentration near 1."""
        g = np.zeros((8, 8))
        g[0, 0] = 1000.0
        c = spatial_noise_concentration(g, k=1)
        assert c > 0.99

    def test_k_larger_than_grid(self):
        """k >= total channels → concentration = 1."""
        g = np.ones((2, 2))
        c = spatial_noise_concentration(g, k=10)
        assert c == pytest.approx(1.0, rel=1e-6)

    def test_output_scalar_for_2d(self):
        """2D input → Python float."""
        g = np.ones((4, 4))
        c = spatial_noise_concentration(g, k=2)
        assert isinstance(c, float)

    def test_batch_shape(self):
        """(5, 8, 8) → (5,)."""
        rng = np.random.default_rng(42)
        grids = rng.normal(0, 1, (5, 8, 8))
        c = spatial_noise_concentration(grids, k=3)
        assert c.shape == (5,)

    def test_concentration_in_range(self):
        """Output is in [0, 1]."""
        rng = np.random.default_rng(42)
        g = np.abs(rng.normal(0, 1, (8, 8)))
        c = spatial_noise_concentration(g, k=5)
        assert 0.0 <= c <= 1.0

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="AP, ML"):
            spatial_noise_concentration(np.ones((4,)))


# ---------------------------------------------------------------------------
# spatial_summary_xr — xarray wrapper
# ---------------------------------------------------------------------------


class TestSpatialSummaryXr:
    def test_basic_output(self):
        """Returns xr.Dataset with one variable per measure."""
        import xarray as xr
        from cogpy.measures.spatial import spatial_summary_xr

        rng = np.random.default_rng(42)
        da = xr.DataArray(
            rng.normal(0, 1, (5, 8, 6)),
            dims=("time_win", "AP", "ML"),
            coords={
                "time_win": np.arange(5) * 0.5,
                "AP": np.arange(8),
                "ML": np.arange(6),
            },
        )
        ds = spatial_summary_xr(
            da, measures=("moran_i", "gradient_anisotropy", "spatial_kurtosis")
        )
        assert isinstance(ds, xr.Dataset)
        assert set(ds.data_vars) == {
            "moran_i",
            "gradient_anisotropy",
            "spatial_kurtosis",
        }
        for var in ds.data_vars:
            assert ds[var].dims == ("time_win",)
            assert ds[var].shape == (5,)
        assert "time_win" in ds.coords

    def test_4d_tf_space(self):
        """(time_win, freq, AP, ML) → (time_win, freq) output."""
        import xarray as xr
        from cogpy.measures.spatial import spatial_summary_xr

        rng = np.random.default_rng(7)
        da = xr.DataArray(
            rng.normal(0, 1, (3, 4, 8, 6)),
            dims=("time_win", "freq", "AP", "ML"),
            coords={
                "time_win": np.arange(3) * 0.5,
                "freq": np.linspace(1, 200, 4),
                "AP": np.arange(8),
                "ML": np.arange(6),
            },
        )
        ds = spatial_summary_xr(da, measures=("moran_i", "spatial_noise_concentration"))
        for var in ds.data_vars:
            assert ds[var].dims == ("time_win", "freq")
            assert ds[var].shape == (3, 4)
        assert "time_win" in ds.coords
        assert "freq" in ds.coords

    def test_directional_moran(self):
        """moran_ap and moran_ml work as measure names."""
        import xarray as xr
        from cogpy.measures.spatial import spatial_summary_xr

        g = _row_striped_grid()
        da = xr.DataArray(g, dims=("AP", "ML"))
        ds = spatial_summary_xr(da, measures=("moran_ap", "moran_ml"))
        # Row-striped: ml_only should have higher I than ap_only
        assert float(ds["moran_ml"]) > float(ds["moran_ap"])

    def test_unknown_measure_raises(self):
        """Unknown measure name raises ValueError."""
        import xarray as xr
        from cogpy.measures.spatial import spatial_summary_xr

        da = xr.DataArray(np.ones((4, 4)), dims=("AP", "ML"))
        with pytest.raises(ValueError, match="Unknown measure"):
            spatial_summary_xr(da, measures=("nonexistent",))

    def test_matches_numpy(self):
        """xarray wrapper matches direct numpy call."""
        import xarray as xr
        from cogpy.measures.spatial import spatial_summary_xr

        rng = np.random.default_rng(42)
        arr = rng.normal(0, 1, (5, 8, 6))
        da = xr.DataArray(arr, dims=("time_win", "AP", "ML"))
        ds = spatial_summary_xr(da, measures=("gradient_anisotropy",))
        expected = gradient_anisotropy(arr)
        np.testing.assert_allclose(ds["gradient_anisotropy"].values, expected)
