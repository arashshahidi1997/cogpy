"""
TensorScope state model (v3.0).

Unified state with a tensor registry and a single global selection object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import param
import xarray as xr

__all__ = ["TensorNode", "TensorRegistry", "SelectionState", "TensorScopeState"]


@dataclass(frozen=True, slots=True)
class TensorNode:
    """
    Immutable tensor with lineage metadata.

    Parameters
    ----------
    name
        Unique tensor identifier.
    data
        Tensor data with labeled dimensions.
        Convention: (time, AP, ML) for grids, (time, channel) for linear arrays.
    source
        Parent tensor name (None for root/original).
    transform
        Transform type: 'signal', 'psd', 'spectrogram', etc.
    params
        Transform parameters (empty for original signals).
    """

    name: str
    data: xr.DataArray
    source: str | None = None
    transform: str = "signal"
    params: dict[str, Any] = field(default_factory=dict)

    def lineage_str(self) -> str:
        if self.source is None:
            return f"{self.name} (original)"
        return f"{self.name} \u2190 {self.transform}({self.source})"

    @property
    def dims(self) -> tuple[str, ...]:
        return tuple(str(d) for d in self.data.dims)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.data.shape)


class TensorRegistry:
    """Registry of named tensors."""

    def __init__(self) -> None:
        self._tensors: dict[str, TensorNode] = {}

    def add(self, node: TensorNode) -> None:
        if node.name in self._tensors:
            raise ValueError(f"Tensor {node.name!r} already registered")
        self._tensors[str(node.name)] = node

    def get(self, name: str) -> TensorNode:
        key = str(name)
        if key not in self._tensors:
            raise KeyError(f"Tensor {key!r} not found in registry")
        return self._tensors[key]

    def list(self) -> list[str]:
        return list(self._tensors.keys())

    def lineage_str(self, name: str) -> str:
        return self.get(name).lineage_str()

    def __contains__(self, name: object) -> bool:
        try:
            key = str(name)
        except Exception:  # noqa: BLE001
            return False
        return key in self._tensors

    def __len__(self) -> int:
        return len(self._tensors)


class SelectionState(param.Parameterized):
    """
    Global selection coordinates (reactive).

    All views should subscribe to this state for linked selection.
    """

    time = param.Number(default=0.0, bounds=(0, None), doc="Time coordinate (seconds)")
    freq = param.Number(default=0.0, bounds=(0, None), doc="Frequency coordinate (Hz)")
    ap = param.Integer(default=0, bounds=(0, None), doc="AP spatial index")
    ml = param.Integer(default=0, bounds=(0, None), doc="ML spatial index")
    channel = param.Parameter(default=None, doc="Channel index (for linear arrays)")

    def update(self, **kwargs: Any) -> None:
        valid: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in self.param:
                valid[str(key)] = value
        if valid:
            self.param.update(**valid)


class TensorScopeState:
    """
    Unified TensorScope state.

    Single source of truth for:
    - tensor registry (all tensors)
    - selection state (global coordinates)
    - active tensor (current tab)
    """

    def __init__(self) -> None:
        self.tensors = TensorRegistry()
        self.selection = SelectionState()
        self.active_tensor: str = "signal"

    def set_active_tensor(self, name: str) -> None:
        if name not in self.tensors:
            raise ValueError(f"Tensor {name!r} not in registry. Available: {self.tensors.list()}")
        self.active_tensor = str(name)

    def get_active_node(self) -> TensorNode:
        return self.tensors.get(self.active_tensor)

    def update_selection(self, **kwargs: Any) -> None:
        self.selection.update(**kwargs)
