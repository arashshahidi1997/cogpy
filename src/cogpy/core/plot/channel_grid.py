"""
channel_grid.py
===============
Grid-layout-aware channel selection primitive.

Works entirely in integer grid indices (0..N-1 for both AP and ML).
Physical coordinate mapping is the caller's responsibility — use
``normalize_coords_to_index`` before constructing a ChannelGrid.

Modes
-----
row          : all channels in one AP row
column       : all channels in one ML column
sparse       : strided subgrid covering the full extent (stride + offset)
neighborhood : filled square (Chebyshev) around a center electrode
manual       : explicit toggle of individual cells

Usage
-----
    from channel_grid import ChannelGrid

    grid = ChannelGrid(n_ap=16, n_ml=16)

    grid.select_row(3)
    grid.select_column(7)
    grid.select_sparse(stride=2)           # 8×8 covering full 16×16
    grid.select_sparse(stride=4, offset=1) # offset phase
    grid.select_neighborhood(ap=8, ml=8, radius=2)
    grid.select_manual({(0,0), (1,1)})

    # Watch output
    grid.param.watch(lambda e: print(e.new), "selected")

    # Downstream use
    grid.selected       # frozenset of (ap, ml) int tuples
    grid.flat_indices   # list[int] into row-major (n_ap*n_ml,) flat array
    grid.as_array       # bool ndarray (n_ap, n_ml)
    grid.n_selected     # int
"""

from __future__ import annotations

from typing import FrozenSet, Tuple

import numpy as np
import param

__all__ = ["ChannelGrid"]

APMLPair = Tuple[int, int]


class ChannelGrid(param.Parameterized):
    """
    Grid-layout-aware channel selection primitive.

    All coordinates are integer grid indices (0-based).
    ``selected`` is always a ``frozenset`` of ``(ap, ml)`` int tuples and is
    the single source of truth — watch it for downstream updates.

    Parameters
    ----------
    n_ap : int
        Number of AP rows (constant after construction).
    n_ml : int
        Number of ML columns (constant after construction).
    """

    # ------------------------------------------------------------------
    # Grid shape — constant after construction
    # ------------------------------------------------------------------
    n_ap = param.Integer(default=16, bounds=(1, None), constant=True)
    n_ml = param.Integer(default=16, bounds=(1, None), constant=True)

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------
    mode = param.Selector(
        default="row",
        objects=["row", "column", "sparse", "neighborhood", "manual"],
    )

    # ------------------------------------------------------------------
    # Mode-specific inputs
    # ------------------------------------------------------------------
    row    = param.Integer(default=0, doc="AP row index for 'row' mode.")
    column = param.Integer(default=0, doc="ML column index for 'column' mode.")

    # sparse: strided subgrid covering full extent
    sparse_stride = param.Integer(
        default=2, bounds=(1, None),
        doc="Step between selected electrodes in both AP and ML directions.",
    )
    sparse_offset = param.Integer(
        default=0, bounds=(0, None),
        doc="Phase offset (applied equally to AP and ML). 0 = start from (0,0).",
    )

    # neighborhood: filled Chebyshev square
    neighborhood_center = param.Tuple(
        default=(0, 0),
        doc="(ap, ml) center for 'neighborhood' mode.",
    )
    neighborhood_radius = param.Integer(
        default=2, bounds=(0, None),
        doc="Chebyshev (L∞) radius in grid steps. r=1 → 3×3, r=2 → 5×5.",
    )

    # manual: explicit set, toggled cell by cell
    manual_selection = param.Parameter(
        default=frozenset(),
        doc="Explicit frozenset of (ap, ml) pairs for 'manual' mode.",
    )

    # ------------------------------------------------------------------
    # Output — single source of truth, watch this downstream
    # ------------------------------------------------------------------
    selected = param.Parameter(
        default=frozenset(),
        doc="Currently selected (ap, ml) pairs as a frozenset.",
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, n_ap: int = 16, n_ml: int = 16, **params):
        params["n_ap"] = n_ap
        params["n_ml"] = n_ml
        super().__init__(**params)
        self._clamp_inputs()
        self._recompute()
        self.param.watch(
            self._on_change,
            [
                "mode", "row", "column",
                "sparse_stride", "sparse_offset",
                "neighborhood_center", "neighborhood_radius",
                "manual_selection",
            ],
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _clamp_inputs(self):
        self.row    = int(np.clip(self.row,    0, self.n_ap - 1))
        self.column = int(np.clip(self.column, 0, self.n_ml - 1))
        ap_c, ml_c = self.neighborhood_center
        self.neighborhood_center = (
            int(np.clip(ap_c, 0, self.n_ap - 1)),
            int(np.clip(ml_c, 0, self.n_ml - 1)),
        )

    def _on_change(self, *_):
        self._recompute()

    def _recompute(self):
        self.selected = self._compute_selected()

    def _compute_selected(self) -> FrozenSet[APMLPair]:
        mode = self.mode

        if mode == "row":
            ap = int(np.clip(self.row, 0, self.n_ap - 1))
            return frozenset((ap, ml) for ml in range(self.n_ml))

        if mode == "column":
            ml = int(np.clip(self.column, 0, self.n_ml - 1))
            return frozenset((ap, ml) for ap in range(self.n_ap))

        if mode == "sparse":
            s   = max(1, self.sparse_stride)
            off = self.sparse_offset % s
            return frozenset(
                (ap, ml)
                for ap in range(off, self.n_ap, s)
                for ml in range(off, self.n_ml, s)
            )

        if mode == "neighborhood":
            ap_c, ml_c = self.neighborhood_center
            r = self.neighborhood_radius
            return frozenset(
                (ap, ml)
                for ap in range(max(0, ap_c - r), min(self.n_ap, ap_c + r + 1))
                for ml in range(max(0, ml_c - r), min(self.n_ml, ml_c + r + 1))
            )

        if mode == "manual":
            return frozenset(
                (int(ap), int(ml))
                for ap, ml in self.manual_selection
                if 0 <= ap < self.n_ap and 0 <= ml < self.n_ml
            )

        return frozenset()

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def flat_indices(self) -> list[int]:
        """Row-major flat indices: index = ap * n_ml + ml."""
        return sorted(ap * self.n_ml + ml for ap, ml in self.selected)

    @property
    def as_array(self) -> np.ndarray:
        """Boolean mask of shape (n_ap, n_ml). True where selected."""
        mask = np.zeros((self.n_ap, self.n_ml), dtype=bool)
        for ap, ml in self.selected:
            mask[ap, ml] = True
        return mask

    @property
    def n_selected(self) -> int:
        return len(self.selected)

    # ------------------------------------------------------------------
    # Fluent setters
    # ------------------------------------------------------------------
    def select_row(self, ap: int) -> "ChannelGrid":
        self.mode = "row"
        self.row  = int(np.clip(ap, 0, self.n_ap - 1))
        return self

    def select_column(self, ml: int) -> "ChannelGrid":
        self.mode   = "column"
        self.column = int(np.clip(ml, 0, self.n_ml - 1))
        return self

    def select_sparse(self, stride: int = 2, offset: int = 0) -> "ChannelGrid":
        """Strided subgrid covering the full extent.

        stride=2 on a 16×16 → 8×8 = 64 channels evenly spaced.
        stride=4 on a 16×16 → 4×4 = 16 channels evenly spaced.
        offset shifts the phase: offset=1, stride=2 picks (1,3,5,…) instead of (0,2,4,…).
        """
        self.sparse_stride = max(1, int(stride))
        self.sparse_offset = int(offset)
        self.mode          = "sparse"
        return self

    def select_neighborhood(self, ap: int, ml: int, radius: int = 2) -> "ChannelGrid":
        self.mode                = "neighborhood"
        self.neighborhood_radius = int(radius)
        self.neighborhood_center = (
            int(np.clip(ap, 0, self.n_ap - 1)),
            int(np.clip(ml, 0, self.n_ml - 1)),
        )
        return self

    def select_manual(self, pairs: set[APMLPair]) -> "ChannelGrid":
        self.mode             = "manual"
        self.manual_selection = frozenset(pairs)
        return self

    def toggle_manual(self, ap: int, ml: int) -> "ChannelGrid":
        """Toggle one cell in manual mode (switches to manual if needed)."""
        self.mode = "manual"
        pair    = (int(ap), int(ml))
        current = set(self.manual_selection)
        if pair in current:
            current.discard(pair)
        else:
            current.add(pair)
        self.manual_selection = frozenset(current)
        return self

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"ChannelGrid({self.n_ap}×{self.n_ml}  "
            f"mode={self.mode!r}  "
            f"n_selected={self.n_selected})"
        )