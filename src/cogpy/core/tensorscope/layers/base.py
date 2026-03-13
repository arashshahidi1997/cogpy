"""
Base layer interface for TensorScope.

All layers follow this pattern:
1. Receive TensorScopeState in __init__
2. Wrap existing component (thin wrapper, not reimplementation)
3. Wire up reactive bindings to state
4. Track watchers/streams for cleanup
5. Implement dispose() to clean up

See the TensorScope design principles docs for component and lifecycle guidance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import panel as pn


WatcherHandle = object


@dataclass(frozen=True, slots=True)
class _WatcherRef:
    owner: Any
    handle: WatcherHandle


class TensorLayer(ABC):
    """
    Base class for TensorScope visualization layers.

    Layers are thin wrappers around existing cogpy.core.plot components.
    They adapt components to work with TensorScopeState and handle lifecycle.
    """

    layer_id: str = "base_layer"
    title: str = "Base Layer"

    def __init__(self, state: Any):
        self.state = state
        self._watchers: list[_WatcherRef] = []
        self._streams: list[Any] = []
        self._data_refs: list[Any] = []
        self._panel: pn.viewable.Viewable | None = None

    @abstractmethod
    def panel(self) -> pn.viewable.Viewable:
        """
        Return the Panel viewable for this layer.

        Implementations should return the same instance on repeated calls.
        """

    def _watch(self, owner: Any, fn: Callable[..., Any], param_name: str | list[str]):
        """
        Register a param watcher and track it for disposal.

        Parameters
        ----------
        owner
            The Parameterized instance whose ``.param.watch`` is called.
        fn
            Callback.
        param_name
            Parameter name or list of names.
        """
        h = owner.param.watch(fn, param_name)
        self._watchers.append(_WatcherRef(owner=owner, handle=h))
        return h

    def _add_stream(self, stream: Any) -> Any:
        self._streams.append(stream)
        return stream

    def _add_data_ref(self, obj: Any) -> Any:
        self._data_refs.append(obj)
        return obj

    def dispose(self) -> None:
        """
        Clean up resources.

        MUST be called when a layer is removed from an application.
        Unregisters tracked param watchers and clears references to allow GC.
        """
        for ref in self._watchers:
            try:
                ref.owner.param.unwatch(ref.handle)
            except Exception:  # noqa: BLE001
                pass

        for s in self._streams:
            try:
                if hasattr(s, "clear"):
                    s.clear()
            except Exception:  # noqa: BLE001
                pass

        for obj in self._data_refs:
            try:
                if hasattr(obj, "dispose"):
                    obj.dispose()
            except Exception:  # noqa: BLE001
                pass

        self._panel = None
        self._watchers.clear()
        self._streams.clear()
        self._data_refs.clear()


@dataclass(frozen=True, slots=True)
class LayerMetadata:
    """Metadata for a layer type (registration/display)."""

    layer_id: str
    title: str
    description: str = ""
    layer_type: str = "generic"

