"""
Module: grid_neighborhood
Status: STABLE
Last Updated: 2025-09-07
Author: Arash Shahidi, A.Shahidi@campus.lmu.de

Summary:
    Utilities for working with per-node neighborhoods on a 2D grid, derived from
    a position-agnostic footprint (stencil). Includes the `GridNeighborhood` class
    which precomputes per-node neighborhood masks and neighbor relations, as well
    as standalone functions for constructing and manipulating such neighborhoods.

Functions:
    make_footprint: Create a binary structuring element (stencil) via morphological iteration.
    remove_center: Return a copy of the stencil with the center element cleared (set to False).
    grid_index_array: Create a 2D array of linear indices shaped like the grid.
    build_neighbor_masks: Construct per-node neighborhood masks from a position-agnostic footprint.
    build_neighbor_pairs_df: Generate a long-form DataFrame of (reference, neighbor) pairs.
    adjacency_matrix: Build an adjacency matrix from neighborhood masks.
    adjacency_edges: Get the adjacency edges (source, destination) for the grid.
    gather_neighbors: Get grid values of neighbors of a reference node.
    map_neighbors: Return a function that applies an ensemble function to each node's neighborhood.
    apply_neighborfunc: Apply a function to each node's neighborhood and return the results.
    apply_neighborfunc: Apply a function to each node's neighborhood and return the results.

Classes:
    GridNeighborhood: Class encapsulating grid neighborhood utilities and precomputations.
        attributes:
            footprint: The position-agnostic neighborhood stencil.
            grid_shape: Shape of the grid as (rows, cols).
            num_nodes: Total number of nodes in the grid.
            include_mask_per_node: Per-node neighborhood masks including the reference node.
            exclude_mask_per_node: Per-node neighborhood masks excluding the reference node.
            neighbor_pairs_df: DataFrame of directed neighbor relations (excluding self).
            adj: Adjacency matrix of the grid (self excluded).
            adj_src: Source indices of adjacency edges.
            adj_dst: Destination indices of adjacency edges.

        methods:
            get_neighbor_mask: Return the neighborhood mask for a given node.
            adjacency_matrix: Build an adjacency matrix from the precomputed masks.
            adjacency_edges: Get the adjacency edges (source, destination) for the grid.
            neighbor_indices: Get linear indices of neighbors for a given node (self excluded).
            apply_neighborhoodfunc: Apply a function to each node's neighborhood and return the results.
            neighborhood_mapper: Return a function that applies an ensemble function to each node's neighborhood.
        
Constants:
    None

Example:
    >>> from cogpy.utils import grid_neighborhood as gn
    >>> footprint = gn.make_footprint(2, 1, 2)
    >>> gneigh = gn.GridNeighborhood(AP=16, ML=16, footprint=footprint)
    >>> gneigh.grid_shape
    (16, 16)
    >>> gneigh.num_nodes
    256
    >>> gneigh.get_neighbor_mask(node_idx=0, exclude=True).shape
    (16, 16)
    >>> gneigh.adj.shape
    (256, 256)
    >>> gneigh.neighbor_indices(node_idx=0)
    array([ 1,  2, 16, 17, 32])
"""

from pickletools import read_uint1
import numpy as np
import pandas as pd
import scipy.ndimage as nd
from scipy import signal
import matplotlib.pyplot as plt
from tabulate import tabulate
from functools import partial

class GridNeighborhood:
    """
    Utilities for working with per-node neighborhoods on a 2D grid, derived from
    a position-agnostic footprint (stencil).

    Parameters
    ----------
    footprint : np.ndarray, dtype=bool, shape (H, W)
        Boolean stencil defining the neighborhood motif relative to a center cell.
    grid_shape : tuple[int, int]
        Grid shape as (n_rows, n_cols).

    Attributes
    ----------
    neighbor_mask : np.ndarray, dtype=bool, shape (H, W)
        Alias of the provided `footprint`; the position-agnostic stencil.
    grid_shape : tuple[int, int]
        Grid shape (rows, cols).
    num_nodes : int
        Total number of grid nodes (rows * cols).
    include_mask_per_node : np.ndarray, dtype=bool, shape (num_nodes, H, W)
        Per-node neighborhood masks including the reference node.
    exclude_mask_per_node : np.ndarray, dtype=bool, shape (num_nodes, H, W)
        Per-node neighborhood masks excluding the reference node.
    neighbor_pairs_df : pandas.DataFrame with columns ['ch_ref', 'ch_neighbor']
        Long-form table of directed neighbor relations (excluding self).
    adj : np.ndarray, dtype=bool, shape (num_nodes, num_nodes)
        Adjacency matrix of the grid (self excluded).
    adj_src : np.ndarray, dtype=int, 1D
        Source indices of adjacency edges.
    adj_dst : np.ndarray, dtype=int, 1D
        Destination indices of adjacency edges.
    
    Methods
    -------
    get_neighbor_mask(node_idx, exclude=False):
        Return the neighborhood mask for a given node.
    adjacency_matrix():
        Build an adjacency matrix from the precomputed masks.
    adjacency_edges():
        Get the adjacency edges (source, destination) for the grid.
    neighbor_indices(node_idx):
        Get linear indices of neighbors for a given node (self excluded).
    apply_neighborhoodfunc(ensemble_func, grid_values):
        Apply a function to each node's neighborhood and return the results.
    neighborhood_mapper(ensemble_func):
        Return a function that applies an ensemble function to each node's neighborhood.
    __repr__():
        Return a human-readable, bordered table summary of members and methods.
    """
    def __init__(self, AP:int, ML:int, footprint=None):
        """Initialize and precompute per-node neighborhood masks and neighbor pairs."""
        if footprint is None:
            footprint = make_footprint(rank=2, connectivity=1, niter=2)
        self.footprint = footprint
        self.grid_shape = (AP, ML)
        self.num_nodes = AP * ML
        self.include_mask_per_node = build_neighbor_masks(footprint, self.grid_shape, exclude=False)
        self.exclude_mask_per_node = build_neighbor_masks(footprint, self.grid_shape, exclude=True)
        self.neighbor_pairs_df = build_neighbor_pairs_df(footprint, self.grid_shape, self.num_nodes)
        self.adj = self.adjacency_matrix()
        self.adj_src, self.adj_dst = self.adjacency_edges()

    def get_neighbor_mask(self, node_idx, exclude=False):
        """
        Return the neighborhood mask for a given node.

        Parameters
        ----------
        node_idx : int
            Linear index of the reference node in row-major order (0..num_nodes-1).
        exclude : bool, default False
            If True, the returned mask excludes the reference node; otherwise it includes it.

        Returns
        -------
        mask : np.ndarray, dtype=bool, shape grid_shape
            Boolean mask marking neighbors of the given node.
        """
        neighbor_mask = self.exclude_mask_per_node if exclude else self.include_mask_per_node
        return neighbor_mask[node_idx]

    def adjacency_matrix(self):
        """
        Build an adjacency matrix from the precomputed (exclude-self) masks.

        Returns
        -------
        A : np.ndarray, dtype=bool, shape (num_nodes, num_nodes)
            Row i, column j is True if node j is a neighbor of node i (self excluded).
        """
        return adjacency_matrix(self.grid_shape, exclude=True, exclude_mask_per_node=self.exclude_mask_per_node, num_nodes=self.num_nodes)

    def adjacency_edges(self):
        """
        Get the adjacency edges (source, destination) for the grid.

        Returns
        -------
        edges : tuple[np.ndarray, np.ndarray]
            Source and destination indices of the adjacency edges.
        """
        src, dst = np.where(self.adj)
        src_order = np.argsort(src)
        return src[src_order], dst[src_order]

    def neighbor_indices(self, node_idx):
        """
        Get linear indices of neighbors for a given node (self excluded).

        Parameters
        ----------
        node_idx : int
            Linear index of the reference node (0..num_nodes-1).

        Returns
        -------
        idx : np.ndarray, dtype=int, 1D
            Linear indices into the flattened grid for neighbors of `node_idx`.
        """
        return np.ravel_multi_index(np.where(self.exclude_mask_per_node[node_idx]), dims=self.grid_shape)

    def apply_neighborhoodfunc(self, ensemble_func, grid_values):
        """
        Apply a function to each node's neighborhood and return the results.
        
        Parameters
        ----------
        ensemble_func : callable
            Function to apply to each neighborhood. It should accept a 1D array of
            neighbor values and return a single value (e.g., np.mean, np.median).
        grid_values : np.ndarray, shape grid_shape
            Values laid out on the same grid as the masks.
        
        Returns
        -------
        results : list
            List of results from applying `ensemble_func` to each node's neighborhood.
        """
        return apply_neighborfunc(ensemble_func, grid_values, self.exclude_mask_per_node)
    
    def neighborhood_mapper(self, ensemble_func):
        """
        Return a function that applies `ensemble_func` to each node's neighborhood.
        
        Parameters
        ----------
        ensemble_func : callable
            Function to apply to each neighborhood. It should accept a 1D array of
            neighbor values and return a single value (e.g., np.mean, np.median).
        
        Returns
        -------
        mapper : callable
            Function that takes `grid_values` and applies `ensemble_func` to each
            node's neighborhood, returning the results.
        """
        return partial(self.apply_neighborhoodfunc, ensemble_func=ensemble_func)

    def __repr__(self):
        """
        Return a human-readable, bordered table summary of members and methods.

        Notes
        -----
        Uses `tabulate(..., tablefmt='grid')` to render an ASCII table for
        interactive inspection (e.g., in a REPL).
        """
        header = "<GridNeighborhood>"
        data = {
            "Method/Attribute": [
                "footprint", "grid_shape", "nch", "loc_include", "exclude_mask_per_node",
                "exclude_mask_per_node_df", "mask_cache", "get_neighborhood_arr",
                "adjacency_matrix", "neighbor_indices"
            ],
            "Description": [
                "Footprint array", "Shape of the grid", "Number of channels",
                "Included locations", "Excluded locations",
                "DataFrame of excluded locations", "Collection of locations",
                "Get neighborhood array", "Get adjacency matrix",
                "Get neighbor indices",
            ],
        }
        df = pd.DataFrame(data)
        table = tabulate(df, headers="keys", tablefmt="grid", showindex=False)
        return f"{header}\n{table}"

def build_neighbor_masks(footprint, grid_shape, exclude=False):
    """
    Construct per-node neighborhood masks from a position-agnostic footprint.

    Parameters
    ----------
    footprint : np.ndarray, dtype=bool, shape (H, W)
        Boolean stencil defining the neighborhood motif relative to center.
    grid_shape : tuple[int, int]
        Grid shape as (n_rows, n_cols).
    exclude : bool, default False
        If True, the returned mask excludes the reference node; otherwise it includes it.

    Returns
    -------
    mask_per_node_ : np.ndarray, dtype=bool, shape (num_nodes, n_rows, n_cols)
        For each node, the neighborhood mask (self included if `exclude` is False).

    Notes
    -----
    Internally, for each node the footprint is "centered" via 2D convolution
    against a Kronecker delta image and then converted to a boolean mask.
    """
    mask_per_node = []
    nrow = grid_shape[0]
    ncol = grid_shape[1]
    num_nodes = nrow * ncol
    for node_idx in range(num_nodes):
        mask_ = np.zeros(grid_shape)
        mask_[np.unravel_index(node_idx, grid_shape)] = 1
        mask_ = signal.convolve2d(mask_, footprint, mode='same')
        if exclude:
            mask_[np.unravel_index(node_idx, grid_shape)] = 0
        mask_per_node.append(mask_)

    mask_per_node = np.array(mask_per_node, dtype=bool)
    return mask_per_node

def build_neighbor_pairs_df(footprint, grid_shape, num_nodes):
    """
    Generate a long-form DataFrame of (reference, neighbor) pairs.

    Parameters
    ----------
    footprint : np.ndarray, dtype=bool, shape (H, W)
        Boolean stencil used to define neighbor relations.
    grid_shape : tuple[int, int]
        Grid shape as (n_rows, n_cols).
    num_nodes : int
        Total number of nodes (n_rows * n_cols).

    Returns
    -------
    neighbor_pairs_df : pandas.DataFrame
        DataFrame with columns ['ch_ref', 'ch_neighbor'] listing all directed
        neighbor relations (self excluded). Indices are linear (row-major).
    """
    exclude_mask_per_node = build_neighbor_masks(footprint, grid_shape, exclude=True)
    neighbor_pairs_df = pd.DataFrame(np.argwhere(exclude_mask_per_node.reshape(-1, num_nodes)), columns=['ch_ref', 'ch_neighbor'])
    return neighbor_pairs_df

def adjacency_matrix(grid_shape, exclude=True, footprint=None, exclude_mask_per_node=None, num_nodes=None):
    """
    Build an adjacency matrix from neighborhood masks.
        
    Parameters
    ----------
    grid_shape : tuple[int, int]
        Grid shape as (n_rows, n_cols).
    exclude : bool, default True
        If True, the adjacency excludes self-loops; otherwise includes them.
    footprint : np.ndarray, dtype=bool, optional
        Boolean stencil defining the neighborhood motif. Required if
        `exclude_mask_per_node` is not provided.
    exclude_mask_per_node : np.ndarray, dtype=bool, optional
        Precomputed per-node masks excluding self. If provided, `footprint`
        is ignored.
    num_nodes : int, optional
        Total number of nodes (n_rows * n_cols). If not provided, it is
        computed from `grid_shape`.

    Returns
    -------
    A : np.ndarray, dtype=bool, shape (num_nodes, num_nodes)
        Row i, column j is True if node j is a neighbor of node i.
    """
    if num_nodes is None:
        num_nodes = grid_shape[0] * grid_shape[1]
    if exclude_mask_per_node is None:
        if footprint is None:
            footprint = make_footprint(2,1,2)
        exclude_mask_per_node = build_neighbor_masks(footprint, grid_shape, exclude=exclude)
    adj = np.array([exclude_mask_per_node[node_idx].reshape(-1) for node_idx in range(num_nodes)])
    return adj

def adjacency_edges(*args, **kwargs):
    """
    Get the adjacency edges (source, destination) for the grid.

    Parameters
    ----------
    *args, **kwargs : passed through to `adjacency_matrix`
        See `adjacency_matrix` for details.
    Returns
    -------
    edges : tuple[np.ndarray, np.ndarray]
        Source and destination indices of the adjacency edges.
    """
    src, dst = np.where(adjacency_matrix(*args, **kwargs))
    src_order = np.argsort(src)
    return src[src_order], dst[src_order]

def gather_neighbors(grid_values, neighbor_mask, node_idx, **kwargs):
    """
    Get grid values of neighbors of a reference node.

    Parameters
    ----------
    grid_values : np.ndarray, shape grid_shape
        Values laid out on the same grid as the masks.
    neighbor_mask: 
    node_idx : int
        Passed through to `select_neighbor_mask`.

    Returns
    -------
    neigh : np.ndarray, 1D
        Values at positions marked True in the selected neighbor mask.

    Notes
    -----
    This function relies on `select_neighbor_mask` to obtain a boolean mask
    for the reference node. Ensure the kwargs you pass select the intended mask.
    """

    include_mask_node = neighbor_mask[node_idx]
    neigh = grid_values[np.where(include_mask_node)]
    return neigh

def map_neighbors(ensemble_func, neighbor_mask):
    """
    Return a function that applies `ensemble_func` to each node's neighborhood.
        
    Parameters
    ----------
    ensemble_func : callable
        Function to apply to each neighborhood. It should accept a 1D array of
        neighbor values and return a single value (e.g., np.mean, np.median).
    neighbor_mask : np.ndarray, dtype=bool, shape (num_nodes, H, W)
        Per-node neighborhood masks.

    Returns
    -------
    apply_func_to_neighbors : callable
        Function that takes `node_idx` and `grid_values`, applies `ensemble_func`
        to the neighborhood of `node_idx`, and returns the result.
    """
    def apply_func_to_neighbors(node_idx, grid_values):
        neigh = gather_neighbors(grid_values, neighbor_mask, node_idx)
        return ensemble_func(neigh)

    return apply_func_to_neighbors

def apply_neighborfunc(ensemble_func, grid_values, neighbor_mask):
    """
    Apply a function to each node's neighborhood and return the results.

    Parameters
    ----------
    ensemble_func : callable
        Function to apply to each neighborhood. It should accept a 1D array of
        neighbor values and return a single value (e.g., np.mean, np.median).
    grid_values : np.ndarray, shape grid_shape
        Values laid out on the same grid as the masks.
    neighbor_mask : np.ndarray, dtype=bool, shape (num_nodes, H, W
        Per-node neighborhood masks.
    
    Returns
    -------
    results : list
        List of results from applying `ensemble_func` to each node's neighborhood.
    """
    num_nodes = grid_values.shape[0] * grid_values.shape[1]
    neighborfunc = map_neighbors(ensemble_func, neighbor_mask)
    return [neighborfunc(node_idx, grid_values) for node_idx in range(num_nodes)]

def make_footprint(rank=2, connectivity=1, niter=2):
    """
    Create a binary structuring element (stencil) via morphological iteration.

    Parameters
    ----------
    rank : int, default 2
        Dimensionality of the structuring element.
    connectivity : int, default 1
        Neighborhood connectivity (e.g., 1 => 4-connectivity in 2D).
    niter : int, default 2
        Number of times to iteratively expand the base structuring element.

    Returns
    -------
    footprint : np.ndarray, dtype=bool, shape determined by `rank` and `niter`
        The generated Boolean stencil (position-agnostic neighborhood motif).
    """
    footprint = nd.iterate_structure(nd.generate_binary_structure(rank, connectivity), niter)
    return footprint

def remove_center(footprint):
    """
    Return a copy of the stencil with the center element cleared (set to False).

    Parameters
    ----------
    footprint : np.ndarray, dtype=bool
        Boolean stencil with an identifiable central element.

    Returns
    -------
    footprint_exclude : np.ndarray, dtype=bool
        Copy of `footprint` where the center (floor-div midpoint) is False.
    """
    footprint_exclude = np.copy(footprint)
    footprint_exclude[footprint.shape[0]//2, footprint.shape[1]//2] = False
    return footprint_exclude

def grid_index_array(grid_shape):
    """
    Create a 2D array of linear indices shaped like the grid.

    Parameters
    ----------
    grid_shape : tuple[int, int]
        Grid shape as (n_rows, n_cols).

    Returns
    -------
    indices : np.ndarray, dtype=int, shape grid_shape
        Each cell contains its linear (row-major) index in [0, n_rows*n_cols).
    """
    nch = grid_shape[0] * grid_shape[1]
    channels_arr = np.arange(nch).reshape(grid_shape)
    return channels_arr


def center_index_from_footprint(footprint: np.ndarray) -> int:
    """
    Return the flat index of the footprint's logical center
    (row = H//2, col = W//2, etc.) among the True elements.

    Parameters
    ----------
    footprint : ndarray of bool, shape (H, W[, ...])
        Boolean mask defining neighborhood.

    Returns
    -------
    center_idx : int
        Index of the center element in the flattened neighborhood vector.

    Example
    -------
    >>> import numpy as np
    >>> import cogpy.utils.grid_neighborhood as gn
    >>> footprint = gn.make_footprint(rank=2, connectivity=1, niter=2)
    >>> assert gn.center_index_from_footprint(footprint) == 6
    """
    if footprint.ndim < 1:
        raise ValueError("Footprint must have at least 1 dimension.")

    # logical center coordinates
    center_coords = tuple(s // 2 for s in footprint.shape)

    if not footprint[center_coords]:
        raise ValueError(
            f"Footprint does not include its logical center {center_coords}."
        )

    # flatten order = ravel (row-major)
    flat_indices = np.flatnonzero(footprint.ravel())

    # find where the center coordinate maps into this list
    center_flat = np.ravel_multi_index(center_coords, footprint.shape)
    try:
        return np.where(flat_indices == center_flat)[0][0]
    except IndexError:
        raise RuntimeError("Could not locate center in footprint.")
