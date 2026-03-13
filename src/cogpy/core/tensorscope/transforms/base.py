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
        self.stream = None
        self._sync_from_space_disabled = False

    def has_dim(self, dim: str) -> bool:
        return str(dim) in self.dims

    def set_selection(self, dim: str, value) -> None:
        self._set_selection(str(dim), value, source="space")

    def get_selection(self, dim: str):
        return self.selections.get(str(dim))

    def create_stream(self):
        """
        Create a HoloViews stream for this coordinate space.

        Notes
        -----
        - This is optional: views can still link by calling `watch(...)`.
        - Stream-to-space sync is guarded to avoid recursion when `set_selection()`
          triggers stream updates.
        """
        import holoviews as hv

        hv.extension("bokeh")

        dims = sorted(self.dims)
        # Use float defaults so Stream parameters accept both ints and floats.
        # (Param infers parameter type from the default value.)
        defaults = {d: 0.0 for d in dims}
        StreamClass = hv.streams.Stream.define(self.name.title(), **defaults)

        init = {d: self.get_selection(d) if self.get_selection(d) is not None else 0 for d in dims}
        self.stream = StreamClass(**init)

        def _on_stream_param(event) -> None:
            if getattr(self, "_sync_from_space_disabled", False):
                return
            dim = str(getattr(event, "name", ""))
            if dim not in self.dims:
                return
            self._set_selection(dim, getattr(event, "new", None), source="stream")

        for d in dims:
            self.stream.param.watch(_on_stream_param, d)

        return self.stream

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

    def _set_selection(self, dim: str, value, *, source: str) -> None:
        if dim not in self.dims:
            raise ValueError(f"Dimension {dim!r} not in space {self.name!r}")

        old = self.selections.get(dim, None)
        self.selections[dim] = value

        if (source != "stream") and (self.stream is not None) and (old != value):
            try:
                self._sync_from_space_disabled = True
                self.stream.event(**{dim: value})
            finally:
                self._sync_from_space_disabled = False

        self._notify(dim, value)
