"""Tests for cogpy.core.decomposition.pca — varimax-rotated PCA."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def random_data():
    """Random data matrix (50 samples, 10 variables)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((50, 10))


@pytest.fixture
def low_rank_data():
    """Rank-3 data embedded in 10 dimensions."""
    rng = np.random.default_rng(0)
    latent = rng.standard_normal((100, 3))
    mixing = rng.standard_normal((3, 10))
    return latent @ mixing + 0.01 * rng.standard_normal((100, 10))


# ---------------------------------------------------------------------------
# varimax_rotation
# ---------------------------------------------------------------------------


class TestVarimax:
    def test_output_shape(self, random_data):
        from cogpy.core.decomposition.pca import varimax_rotation

        Y, G = varimax_rotation(random_data, maxit=10, IfVerbose=False)
        assert Y.shape == random_data.shape
        assert G.ndim == 2

    def test_norm_preserved(self, random_data):
        from cogpy.core.decomposition.pca import varimax_rotation

        Y, _ = varimax_rotation(random_data, maxit=10, norm=False, IfVerbose=False)
        # Column norms should be preserved (orthogonal rotation)
        orig_norms = np.linalg.norm(random_data, axis=0)
        rot_norms = np.linalg.norm(Y, axis=0)
        np.testing.assert_allclose(orig_norms, rot_norms, rtol=1e-6)


# ---------------------------------------------------------------------------
# erppca function
# ---------------------------------------------------------------------------


class TestErppca:
    def test_basic_output_keys(self, low_rank_data):
        from cogpy.core.decomposition.pca import erppca

        out = erppca(low_rank_data, IfVerbose=False)
        assert "LU" in out
        assert "LR" in out
        assert "FSr" in out
        assert "VT" in out

    def test_return_attrs(self, low_rank_data):
        from cogpy.core.decomposition.pca import erppca

        out = erppca(low_rank_data, IfVerbose=False, return_attrs=True)
        assert "unmixing_" in out
        assert "cov_" in out

    def test_nfac_truncation(self, low_rank_data):
        from cogpy.core.decomposition.pca import erppca

        out = erppca(low_rank_data, nFac=2, IfVerbose=False)
        assert out["LR"].shape[1] == 2
        assert out["FSr"].shape[1] == 2


# ---------------------------------------------------------------------------
# erpPCA estimator (sklearn API)
# ---------------------------------------------------------------------------


class TestErpPCAEstimator:
    def test_fit_transform(self, low_rank_data):
        from cogpy.core.decomposition.pca import erpPCA

        model = erpPCA(nfac=3, verbose=False)
        model.fit(low_rank_data)
        result = model.transform(low_rank_data)
        assert result.shape == (100, 3)

    def test_cov_diag(self, low_rank_data):
        from cogpy.core.decomposition.pca import erpPCA

        model = erpPCA(nfac=3, verbose=False)
        model.fit(low_rank_data)
        cd = model.cov_diag()
        assert cd.shape[0] == 10  # nvars


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_pseudo_inverse_scaled(self):
        from cogpy.core.decomposition.pca import pseudo_inverse_scaled

        rng = np.random.default_rng(1)
        LR = rng.standard_normal((10, 3))
        cov = np.eye(10)
        result = pseudo_inverse_scaled(LR, cov)
        assert result.shape == (10, 3)

    def test_redirect_loadings(self):
        from cogpy.core.decomposition.pca import redirect_loadings

        L = np.array([[-5, 3], [-2, -4], [-1, 2]])
        L_red = redirect_loadings(L)
        # Column 0 was mostly negative → should flip
        assert L_red[0, 0] > 0
        # Column 1: max(abs) at row 1 with value -4, so should flip
        assert L_red[1, 1] > 0

    def test_sort_by_eigv(self):
        from cogpy.core.decomposition.pca import sort_by_eigv

        L = np.array([[1, 2, 3], [4, 5, 6]])
        ev = np.array([0.1, 0.5, 0.3])
        L_sorted, ev_sorted = sort_by_eigv(L, ev)
        np.testing.assert_array_equal(ev_sorted, [0.5, 0.3, 0.1])

    def test_simplicity_criterion(self):
        from cogpy.core.decomposition.pca import simplicity_criterion

        # Identity-like matrix should have low simplicity
        Y = np.eye(5, 3)
        g = simplicity_criterion(Y)
        assert isinstance(g, (float, np.floating))
