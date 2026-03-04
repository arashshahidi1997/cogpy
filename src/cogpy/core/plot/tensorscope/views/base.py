"""View base class (v2.1)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable

__all__ = ["View"]


@dataclass(frozen=True, slots=True)
class _WatcherRef:
    owner: Any
    handle: Any


class View:
    """Base class for all TensorScope views."""

    def __init__(self, state, signal_id: str, view_id: str | None = None):
        self.state = state
        self.signal_id = str(signal_id)
        self.view_id = str(view_id or str(uuid.uuid4())[:8])
        self._watchers: list[_WatcherRef] = []

    def _watch(self, owner: Any, fn: Callable[..., Any], param_name: str | list[str]):
        h = owner.param.watch(fn, param_name)
        self._watchers.append(_WatcherRef(owner=owner, handle=h))
        return h

    def get_signal(self):
        reg = getattr(self.state, "signal_registry", None)
        if reg is None:
            return None
        return reg.get(self.signal_id)

    def get_config(self) -> dict[str, Any]:
        """Get current configuration for duplication."""
        return {}

    def duplicate(self, **overrides) -> "View":
        """Create a duplicate view with the same configuration."""
        config = dict(self.get_config())
        config.update(overrides)
        return self.__class__(state=self.state, signal_id=self.signal_id, **config)

    def panel(self):
        raise NotImplementedError

    def dispose(self) -> None:
        for ref in self._watchers:
            try:
                ref.owner.param.unwatch(ref.handle)
            except Exception:  # noqa: BLE001
                pass
        self._watchers.clear()

