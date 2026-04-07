"""
Signal object: data + processing pipeline + metadata.

A SignalObject represents one "version" of the data - raw, filtered,
high-gamma extracted, etc. Users can duplicate signals, modify their
processing, and compare them side-by-side.
"""

from __future__ import annotations

import uuid
from typing import Any

import param
import xarray as xr

from ..plot.hv.processing_chain import ProcessingChain
from .schema import flatten_grid_to_channels, validate_and_normalize_grid

__all__ = ["SignalObject", "SignalRegistry"]


def _processing_input(data: xr.DataArray) -> xr.DataArray:
    """
    Choose the data view used for ProcessingChain.

    For grid-shaped inputs (time, AP, ML), we prefer a stacked-flat
    (time, channel) view with AP/ML channel coords so:
    - channel subsetting works for timeseries viewers
    - spatial viewers can reconstruct frames from AP/ML coords
    """
    if ("AP" in data.dims) and ("ML" in data.dims) and ("time" in data.dims):
        # Normalize to canonical grid schema (also normalizes AP/ML coords),
        # then flatten to (time, channel) for ProcessingChain.
        grid = validate_and_normalize_grid(data)
        return flatten_grid_to_channels(grid)
    return data


class SignalObject:
    """
    A signal with its processing pipeline and metadata.

    Signal = Data + Processing Pipeline.

    SignalObject provides:
    - Data reference (shared across duplicates)
    - Processing pipeline (independent per signal)
    - Window extraction: get_window(t0, t1)

    For analysis (PSD, spectrogram, coherence, etc.):
    use ``cogpy.core.spectral`` / ``cogpy.core.measure`` functions on windows.

    Attributes
    ----------
    data : xr.DataArray
        Raw data reference (shared across duplicates)
    name : str
        Human-readable name
    processing : ProcessingChain
        Processing pipeline for this signal
    metadata : dict
        Additional metadata (type, is_base, etc.)
    id : str
        Unique identifier (8-char UUID)

    Examples
    --------
    >>> from cogpy.core.spectral.specx import psdx, spectrogramx
    >>> signal = SignalObject(data, "Raw LFP")
    >>> signal.processing.bandpass_on = True
    >>> signal.processing.bandpass_lo = 1.0
    >>> signal.processing.bandpass_hi = 100.0
    >>> win = signal.get_window(5.0, 10.0)
    >>> psd = psdx(win, method="multitaper", bandwidth=4.0)
    >>> spec = spectrogramx(win, bandwidth=4.0, nperseg=256)
    """

    def __init__(
        self,
        data: xr.DataArray,
        name: str,
        processing: ProcessingChain | None = None,
        metadata: dict | None = None,
        signal_id: str | None = None,
    ) -> None:
        self.data = data
        self.name = str(name)
        self._proc_input = _processing_input(data)
        self.processing = processing or ProcessingChain(self._proc_input)
        self.metadata = metadata or {}
        self.id = str(signal_id or str(uuid.uuid4())[:8])

    def get_window(
        self, t0: float, t1: float, channels: list[int] | None = None
    ) -> xr.DataArray:
        """Get processed data window (delegates to ProcessingChain.get_window)."""
        return self.processing.get_window(float(t0), float(t1), channels=channels)

    def duplicate(self, name: str | None = None) -> "SignalObject":
        """
        Create a copy with independent processing chain.

        The duplicated signal shares the same raw data reference but has its own
        ProcessingChain instance with copied settings.
        """
        new_name = str(name) if name is not None else f"{self.name} (copy)"

        new_processing = ProcessingChain(_processing_input(self.data))
        # Only copy processing params (not private fields like _data/_fs).
        new_processing.param.update(self.processing.to_dict())

        return SignalObject(
            data=self.data,
            name=new_name,
            processing=new_processing,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize signal (for session save)."""
        return {
            "id": self.id,
            "name": self.name,
            "processing": self.processing.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any], data: xr.DataArray) -> "SignalObject":
        """Restore signal from serialized config."""
        processing = ProcessingChain(_processing_input(data))
        proc_config = config.get("processing", {}) or {}
        for key, value in dict(proc_config).items():
            if key in processing.param:
                setattr(processing, key, value)

        return cls(
            data=data,
            name=str(config["name"]),
            processing=processing,
            metadata=dict(config.get("metadata", {}) or {}),
            signal_id=str(config.get("id") or "") or None,
        )


class SignalRegistry(param.Parameterized):
    """
    Registry for managing multiple signal objects.

    Parameters
    ----------
    signals : dict
        Mapping of signal_id → SignalObject
    active_signal_id : str | None
        ID of currently active signal
    """

    signals = param.Dict(default={}, doc="Mapping of signal_id → SignalObject", instantiate=True)
    active_signal_id = param.String(default=None, allow_None=True, doc="ID of currently active signal")

    def register(self, signal: SignalObject) -> str:
        """Register a signal and return its ID."""
        new_signals = dict(self.signals)
        new_signals[signal.id] = signal
        self.signals = new_signals

        if self.active_signal_id is None:
            self.active_signal_id = signal.id
        return signal.id

    def get(self, signal_id: str) -> SignalObject | None:
        return self.signals.get(str(signal_id))

    def get_active(self) -> SignalObject | None:
        sid = self.active_signal_id
        return self.signals.get(sid) if sid else None

    def set_active(self, signal_id: str) -> None:
        sid = str(signal_id)
        if sid not in self.signals:
            raise ValueError(f"Signal {sid!r} not found")
        self.active_signal_id = sid

    def list(self) -> list[str]:
        return list(self.signals.keys())

    def list_names(self) -> list[tuple[str, str]]:
        return [(sid, sig.name) for sid, sig in self.signals.items()]

    def duplicate(self, signal_id: str, new_name: str | None = None) -> str:
        source = self.get(signal_id)
        if source is None:
            raise ValueError(f"Signal {str(signal_id)!r} not found")
        new_signal = source.duplicate(new_name)
        return self.register(new_signal)

    def remove(self, signal_id: str) -> None:
        sid = str(signal_id)
        if sid not in self.signals:
            return

        new_signals = dict(self.signals)
        del new_signals[sid]
        self.signals = new_signals

        if self.active_signal_id == sid:
            self.active_signal_id = next(iter(self.signals.keys()), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signals": {sid: sig.to_dict() for sid, sig in self.signals.items()},
            "active_signal_id": self.active_signal_id,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any], data: xr.DataArray) -> "SignalRegistry":
        registry = cls()
        for _sid, sig_config in (config.get("signals", {}) or {}).items():
            try:
                signal = SignalObject.from_dict(dict(sig_config), data)
            except Exception:  # noqa: BLE001
                continue
            registry.register(signal)

        active_id = config.get("active_signal_id")
        if active_id and str(active_id) in registry.signals:
            registry.active_signal_id = str(active_id)
        return registry
