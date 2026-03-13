"""TensorScope layers."""

from __future__ import annotations

from .base import LayerMetadata, TensorLayer
from .controls import ChannelSelectorLayer, ProcessingControlsLayer
from .events import EventOverlayLayer, EventTableLayer
from .navigation import TimeNavigatorLayer
from .psd_explorer import PSDExplorerLayer
from .psd_settings import PSDSettings, PSDSettingsLayer
from .signal_manager import SignalManagerLayer
from .spatial import SpatialMapLayer
from .spectrogram import SpectrogramLayer
from .timeseries import TimeseriesLayer

__all__ = [
    "TensorLayer",
    "LayerMetadata",
    "TimeseriesLayer",
    "SpatialMapLayer",
    "SpectrogramLayer",
    "ChannelSelectorLayer",
    "ProcessingControlsLayer",
    "SignalManagerLayer",
    "TimeNavigatorLayer",
    "EventTableLayer",
    "EventOverlayLayer",
    "PSDSettingsLayer",
    "PSDExplorerLayer",
    "PSDSettings",
]
