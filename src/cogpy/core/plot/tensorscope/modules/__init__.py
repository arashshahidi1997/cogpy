"""TensorScope view preset modules (v2.2)."""

from __future__ import annotations

from .base import ModuleRegistry, ViewPresetModule
from .event_explorer import EventExplorerModule
from .psd_explorer import PSDExplorerModule

__all__ = ["EventExplorerModule", "ModuleRegistry", "PSDExplorerModule", "ViewPresetModule"]
