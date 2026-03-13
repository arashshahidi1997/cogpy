"""
Layer manager for TensorScope.

Handles layer registration, instantiation, and lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .base import TensorLayer


@dataclass(frozen=True, slots=True)
class LayerSpec:
    """
    Specification for a layer type.

    Attributes
    ----------
    layer_id
        Unique identifier for this layer type.
    title
        Display title.
    factory
        Callable creating a layer instance: (state) -> TensorLayer.
    description
        Human-readable description.
    layer_type
        Category: 'spatial', 'timeseries', 'controls', 'navigation', etc.
    """

    layer_id: str
    title: str
    factory: Callable[[Any], TensorLayer]
    description: str = ""
    layer_type: str = "generic"


class LayerManager:
    """Manages layer instances and specifications."""

    def __init__(self, state: Any):
        self.state = state
        self._specs: dict[str, LayerSpec] = {}
        self._layers: dict[str, TensorLayer] = {}
        self._instance_types: dict[str, str] = {}
        self._instance_counter = 0

    def register(self, spec: LayerSpec) -> None:
        self._specs[str(spec.layer_id)] = spec

    def add(self, layer_id: str, instance_id: str | None = None) -> TensorLayer:
        if layer_id not in self._specs:
            raise ValueError(
                f"Layer type {layer_id!r} not registered. Available: {sorted(self._specs.keys())}"
            )

        if instance_id is None:
            instance_id = f"{layer_id}_{self._instance_counter}"
            self._instance_counter += 1

        spec = self._specs[layer_id]
        layer = spec.factory(self.state)

        # Attach instance metadata (useful for apps / debugging).
        setattr(layer, "instance_id", instance_id)
        setattr(layer, "layer_type_id", layer_id)

        self._layers[instance_id] = layer
        self._instance_types[instance_id] = layer_id
        return layer

    def remove(self, instance_id: str) -> None:
        if instance_id not in self._layers:
            raise KeyError(f"Layer instance {instance_id!r} not found")
        layer = self._layers.pop(instance_id)
        self._instance_types.pop(instance_id, None)
        layer.dispose()

    def get(self, instance_id: str) -> TensorLayer | None:
        return self._layers.get(instance_id)

    def list_instances(self) -> list[str]:
        return list(self._layers.keys())

    def list_types(self) -> list[str]:
        return list(self._specs.keys())

    def list_instance_types(self) -> list[str]:
        """Return active layer type IDs, in the order of instances() output."""
        return [self._instance_types[i] for i in self.list_instances()]

    def dispose_all(self) -> None:
        for instance_id in list(self._layers.keys()):
            self.remove(instance_id)

