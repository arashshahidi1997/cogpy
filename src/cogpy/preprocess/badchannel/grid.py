"""Grid neighborhood utilities for 2D (AP, ML) layouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as nd
import scipy.sparse as sp


def make_footprint(
    *, rank: int = 2, connectivity: int = 1, niter: int = 2
) -> np.ndarray:
    if rank != 2:
        raise ValueError(
            "Only rank=2 footprints are supported in this pipeline module."
        )
    if niter < 1:
        raise ValueError("niter must be >= 1")
    fp = nd.generate_binary_structure(rank, connectivity).astype(bool)
    for _ in range(niter - 1):
        fp = nd.binary_dilation(fp, structure=fp)
    return fp.astype(bool)


def remove_center(footprint: np.ndarray) -> np.ndarray:
    fp = np.asarray(footprint, dtype=bool).copy()
    r0 = fp.shape[0] // 2
    c0 = fp.shape[1] // 2
    fp[r0, c0] = False
    return fp


@dataclass(frozen=True)
class GridAdjacency:
    nrows: int
    ncols: int
    footprint: np.ndarray
    adj: sp.csr_matrix

    @property
    def n_nodes(self) -> int:
        return self.nrows * self.ncols


def grid_edges(
    nrows: int, ncols: int, *, footprint: np.ndarray, exclude_center: bool = True
):
    fp = np.asarray(footprint, dtype=bool)
    if fp.ndim != 2:
        raise ValueError("footprint must be 2D")
    if fp.shape[0] % 2 != 1 or fp.shape[1] % 2 != 1:
        raise ValueError("footprint dimensions must be odd")

    if exclude_center:
        fp = remove_center(fp)

    offsets = np.argwhere(fp) - np.array([fp.shape[0] // 2, fp.shape[1] // 2])
    src_list: list[int] = []
    dst_list: list[int] = []

    for r in range(nrows):
        for c in range(ncols):
            src = r * ncols + c
            for dr, dc in offsets:
                rr = r + int(dr)
                cc = c + int(dc)
                if 0 <= rr < nrows and 0 <= cc < ncols:
                    dst = rr * ncols + cc
                    src_list.append(src)
                    dst_list.append(dst)

    return np.asarray(src_list, dtype=np.int64), np.asarray(dst_list, dtype=np.int64)


def grid_adjacency(
    nrows: int,
    ncols: int,
    *,
    footprint: np.ndarray | None = None,
    exclude_center: bool = True,
    group_labels: np.ndarray | None = None,
) -> GridAdjacency:
    if footprint is None:
        footprint = make_footprint(rank=2, connectivity=1, niter=2)
    src, dst = grid_edges(
        nrows, ncols, footprint=footprint, exclude_center=exclude_center
    )
    if group_labels is not None:
        labels = np.asarray(group_labels)
        if labels.shape != (nrows, ncols):
            raise ValueError("group_labels must have shape (nrows, ncols)")
        flat = labels.reshape(-1)
        keep = flat[src] == flat[dst]
        src = src[keep]
        dst = dst[keep]
    n = nrows * ncols
    adj = sp.csr_matrix((np.ones_like(src, dtype=bool), (src, dst)), shape=(n, n))
    adj.eliminate_zeros()
    return GridAdjacency(
        nrows=nrows, ncols=ncols, footprint=np.asarray(footprint, bool), adj=adj
    )
