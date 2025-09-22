import numpy as np
import pandas as pd
import scipy.ndimage as nd
import pytest
from cogpy.utils.grid_neighborhood import (
    GridNeighborhood, make_footprint, build_neighbor_masks, gather_neighbors
)


def _expected_adj_2x2_4conn():
    """
    Helper: adjacency for a 2x2 grid with 4-connectivity (exclude self).
    Index layout:
        0 1
        2 3
    Neighbors:
        0: {1,2}
        1: {0,3}
        2: {0,3}
        3: {1,2}
    """
    A = np.zeros((4, 4), dtype=bool)
    A[0, [1, 2]] = True
    A[1, [0, 3]] = True
    A[2, [0, 3]] = True
    A[3, [1, 2]] = True
    return A


@pytest.fixture
def footprint_4conn():
    # 3x3 cross (center + N/E/S/W)
    return make_footprint(rank=2, connectivity=1, niter=1)


@pytest.fixture
def gn_2x2(footprint_4conn):
    return GridNeighborhood(AP=2, ML=2, footprint=footprint_4conn)


def test_adjacency_matrix_matches_expected_for_2x2_grid(gn_2x2):
    """Adjacency (exclude self) should match hand-computed 4-connectivity graph."""
    A = gn_2x2.adjacency_matrix()
    assert A.shape == (4, 4)
    assert np.array_equal(A, _expected_adj_2x2_4conn())


def test_neighbor_pairs_df_has_correct_pairs_and_columns(gn_2x2):
    """neighbor_pairs_df should list all directed edges and have correct schema."""
    df = gn_2x2.neighbor_pairs_df
    # Schema
    assert list(df.columns) == ['ch_ref', 'ch_neighbor']
    # There are 8 directed edges in the 2x2 4-conn grid
    assert len(df) == 8

    # Cross-check against adjacency
    A = gn_2x2.adjacency_matrix()
    expected_pairs = {(i, j) for i in range(4) for j in range(4) if A[i, j]}
    got_pairs = set(map(tuple, df[['ch_ref', 'ch_neighbor']].to_numpy()))
    assert got_pairs == expected_pairs


def test_gather_neighbors_splits_values_correctly(gn_2x2):
    """
    Using the include-mask (self + neighbors), node 0 should collect values
    at indices {0,1,2} and non-neighbors {3}.
    """
    grid_values = np.arange(4).reshape(2, 2)  # values equal to linear indices
    include_masks = build_neighbor_masks(gn_2x2.footprint, gn_2x2.grid_shape, exclude=False)

    neigh = gather_neighbors(
        grid_values=grid_values,
        neighbor_mask=include_masks,  # pass include (self+neighbors) mask
        node_idx=0,
    )

    assert set(neigh.tolist()) == {0, 1, 2}
