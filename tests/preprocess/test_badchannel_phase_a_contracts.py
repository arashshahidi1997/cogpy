import numpy as np

from cogpy.core.preprocess.channel_feature_functions import (
    anticorrelation,
    amplitude,
    deviation,
    hurst_exponent,
    relative_variance,
    time_derivative,
)
from cogpy.core.utils.grid_neighborhood import adjacency_matrix, make_footprint


def test_grid_adjacency_matrix_excludes_self():
    footprint = make_footprint(rank=2, connectivity=1, niter=1)
    adj = adjacency_matrix((3, 4), footprint=footprint, exclude=True)
    assert adj.shape == (12, 12)
    assert np.all(np.diag(adj) == 0)


def test_anticorrelation_accepts_adjacency_argument():
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(3, 3, 64)).astype(np.float32)

    footprint = make_footprint(rank=2, connectivity=1, niter=1)
    adj = adjacency_matrix((3, 3), footprint=footprint, exclude=True)

    out = anticorrelation(arr, adj=adj)
    assert out.shape == (3, 3)
    assert np.isfinite(out).all()


def test_temporal_feature_shapes():
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(2, 3, 128)).astype(np.float32)

    assert relative_variance(arr).shape == (2, 3)
    assert deviation(arr).shape == (2, 3)
    assert amplitude(arr).shape == (2, 3)
    assert time_derivative(arr).shape == (2, 3)
    assert hurst_exponent(arr).shape == (2, 3)

