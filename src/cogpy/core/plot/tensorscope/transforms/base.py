"""
Transform base utilities.

This module contains small, reusable primitives for TensorScope "v2.x" view
composition, including shared coordinate selection state.
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["CoordinateSpace"]


class CoordinateSpace:
    """
    Shared coordinate selection state.

    Manages current selection in linked dimensions.
    When selection changes, notifies all watchers.

    Notes
    -----
    This is intentionally lightweight (not param.Parameterized). It acts as a
    simple pub-sub store for selections that multiple views can share.
    """

    def __init__(self, name: str, *, dims: set[str]):
        self.name = str(name)
        self.dims = set(dims)
        self.selections: dict[str, object] = {}
        self._watchers: list[Callable[[str, object], object]] = []

    def has_dim(self, dim: str) -> bool:
        return str(dim) in self.dims

    def set_selection(self, dim: str, value) -> None:
        dim = str(dim)
        if dim not in self.dims:
            raise ValueError(f"Dimension {dim!r} not in space {self.name!r}")
        self.selections[dim] = value
        self._notify(dim, value)

    def get_selection(self, dim: str):
        return self.selections.get(str(dim))

    def watch(self, callback: Callable[[str, object], object]):
        self._watchers.append(callback)
        return callback

    def unwatch(self, callback) -> None:
        try:
            self._watchers.remove(callback)
        except ValueError:
            return

    def _notify(self, dim: str, value) -> None:
        for cb in list(self._watchers):
            try:
                cb(dim, value)
            except Exception:  # noqa: BLE001
                # Watchers should never crash the app.
                pass

