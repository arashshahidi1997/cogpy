"""Core schemas and thin wrappers.

This module introduces a minimal, project-local schema for common neural signals.
For now the focus is **ECoG/iEEG grids** represented as ``xarray.DataArray``.

Why a schema?
------------
Many CogPy utilities assume a few conventions:
- a time dimension named ``"time"``
- for grid ECoG: spatial dimensions named ``"AP"`` and ``"ML"``
- a sampling rate available as ``sig.fs`` (either as a 0D coordinate named
  ``"fs"`` or an attribute ``attrs["fs"]``)

This file provides small helpers to standardize/validate those conventions and
an *optional* thin wrapper class that makes common preprocessing operations
discoverable as methods while still delegating to the existing functional APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import xarray as xr


ECoGSchemaKind = Literal["grid", "flat"]


@dataclass(frozen=True)
class ECoGSchema:
    """Dimension/coordinate names used across the codebase."""

    time: str = "time"
    ap: str = "AP"
    ml: str = "ML"
    ch: str = "ch"
    fs: str = "fs"

    @property
    def grid_dims(self) -> tuple[str, str, str]:
        return (self.time, self.ap, self.ml)

    @property
    def flat_dims(self) -> tuple[str, str]:
        return (self.time, self.ch)


SCHEMA = ECoGSchema()


def _infer_fs_from_time_coord(time_values: np.ndarray) -> float | None:
    t = np.asarray(time_values, dtype=float).reshape(-1)
    if t.size < 2:
        return None
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return None
    # robust to occasional glitches
    dt0 = float(np.median(dt))
    if dt0 <= 0 or not np.isfinite(dt0):
        return None
    return 1.0 / dt0


def get_fs(sig: xr.DataArray, *, schema: ECoGSchema = SCHEMA) -> float | None:
    """Return sampling rate in Hz from a signal (coordinate or attrs)."""
    if not isinstance(sig, xr.DataArray):
        return None

    # Prefer a 0D coordinate named 'fs' (works well with xarray-style attribute access).
    if schema.fs in sig.coords:
        try:
            return float(sig.coords[schema.fs].values)
        except Exception:
            pass

    # Fall back to attrs
    fs = sig.attrs.get(schema.fs, None)
    if fs is not None:
        try:
            return float(fs)
        except Exception:
            return None

    # Infer from time coordinate if present and numeric
    if schema.time in sig.coords:
        return _infer_fs_from_time_coord(sig.coords[schema.time].values)

    return None


def ensure_fs(
    sig: xr.DataArray,
    *,
    fs: float | None = None,
    schema: ECoGSchema = SCHEMA,
) -> xr.DataArray:
    """Ensure ``sig`` has an accessible sampling rate (``sig.fs``)."""
    if not isinstance(sig, xr.DataArray):
        raise TypeError("ensure_fs expects an xarray.DataArray")

    fs_value = float(fs) if fs is not None else get_fs(sig, schema=schema)
    if fs_value is None or not np.isfinite(fs_value) or fs_value <= 0:
        raise ValueError("Sampling rate not found; provide fs=... or set sig.coords['fs'] / sig.attrs['fs']")

    # Use a 0D coordinate so existing code can do `sig.fs` or `sig.fs.item()`.
    out = sig
    if schema.fs not in out.coords:
        out = out.assign_coords({schema.fs: fs_value})
    else:
        try:
            out = out.assign_coords({schema.fs: fs_value})
        except Exception:
            # If assignment fails (e.g., non-assignable coord), keep existing.
            pass
    out.attrs[schema.fs] = fs_value
    return out


def ensure_time_coord(
    sig: xr.DataArray,
    *,
    fs: float | None = None,
    schema: ECoGSchema = SCHEMA,
) -> xr.DataArray:
    """Ensure ``sig`` has a numeric time coordinate in seconds."""
    sig = ensure_fs(sig, fs=fs, schema=schema)
    fs_value = float(get_fs(sig, schema=schema))  # now guaranteed

    if schema.time in sig.coords:
        return sig

    n = int(sig.sizes.get(schema.time, sig.shape[0] if sig.ndim else 0))
    if schema.time in sig.dims:
        n = int(sig.sizes[schema.time])
    else:
        # If time isn't a dim, don't guess.
        raise ValueError(f"Signal has no '{schema.time}' dim; cannot create time coordinate.")

    t = np.arange(n, dtype=float) / fs_value
    return sig.assign_coords({schema.time: t})


def validate_ecog(
    sig: xr.DataArray,
    *,
    kind: ECoGSchemaKind = "grid",
    require_fs: bool = True,
    schema: ECoGSchema = SCHEMA,
) -> None:
    """Validate that a DataArray follows the minimal ECoG schema."""
    if not isinstance(sig, xr.DataArray):
        raise TypeError("Expected an xarray.DataArray")

    if kind == "grid":
        missing = [d for d in schema.grid_dims if d not in sig.dims]
        if missing:
            raise ValueError(f"Expected dims {schema.grid_dims}; missing {missing}. Got dims={tuple(sig.dims)}")
    elif kind == "flat":
        missing = [d for d in schema.flat_dims if d not in sig.dims]
        if missing:
            raise ValueError(f"Expected dims {schema.flat_dims}; missing {missing}. Got dims={tuple(sig.dims)}")
    else:
        raise ValueError(f"Unknown kind={kind!r}")

    if require_fs and get_fs(sig, schema=schema) is None:
        raise ValueError("Missing sampling rate; set sig.coords['fs'] or sig.attrs['fs'] (or pass fs=...)")


def standardize_ecog(
    sig: xr.DataArray,
    *,
    kind: ECoGSchemaKind | None = None,
    fs: float | None = None,
    schema: ECoGSchema = SCHEMA,
) -> xr.DataArray:
    """Return a signal with ensured ``fs`` (and time coord if possible)."""
    if not isinstance(sig, xr.DataArray):
        raise TypeError("standardize_ecog expects an xarray.DataArray")

    out = ensure_fs(sig, fs=fs, schema=schema)
    if schema.time in out.dims:
        out = ensure_time_coord(out, fs=get_fs(out, schema=schema), schema=schema)

    if kind is not None:
        validate_ecog(out, kind=kind, require_fs=True, schema=schema)
    return out


class ECoG:
    """Thin wrapper around ``xr.DataArray`` with schema-aware methods.

    This is intentionally minimal: it does not introduce new storage formats and
    is safe to ignore. The main benefit is discoverability and chaining.
    """

    def __init__(
        self,
        sig: xr.DataArray,
        *,
        kind: ECoGSchemaKind = "grid",
        fs: float | None = None,
        schema: ECoGSchema = SCHEMA,
    ):
        self.schema = schema
        self.kind = kind
        self.x = standardize_ecog(sig, kind=kind, fs=fs, schema=schema)

    @property
    def fs(self) -> float:
        fs = get_fs(self.x, schema=self.schema)
        assert fs is not None
        return float(fs)

    def as_xarray(self) -> xr.DataArray:
        return self.x

    def pipe(self, func, *args, **kwargs) -> "ECoG":
        out = func(self.x, *args, **kwargs)
        if not isinstance(out, xr.DataArray):
            raise TypeError("ECoG.pipe expects func to return an xarray.DataArray")
        return ECoG(out, kind=self.kind, schema=self.schema)

    # ---- preprocessing (delegate to existing functions) ----
    def bandpass(
        self,
        *,
        low_hz: float,
        high_hz: float,
        order: int = 4,
        time_dim: str | None = None,
    ) -> "ECoG":
        from cogpy.preprocess import filtering

        td = self.schema.time if time_dim is None else str(time_dim)
        out = filtering.bandpassx(self.x, wl=float(low_hz), wh=float(high_hz), order=int(order), axis=td)
        out = ensure_fs(out, fs=self.fs, schema=self.schema)
        return ECoG(out, kind=self.kind, schema=self.schema)

    def lowpass(
        self,
        *,
        cutoff_hz: float,
        order: int = 4,
        time_dim: str | None = None,
    ) -> "ECoG":
        from cogpy.preprocess import filtering

        td = self.schema.time if time_dim is None else str(time_dim)
        out = filtering.lowpassx(self.x, wl=float(cutoff_hz), order=int(order), axis=td)
        out = ensure_fs(out, fs=self.fs, schema=self.schema)
        return ECoG(out, kind=self.kind, schema=self.schema)

    def highpass(
        self,
        *,
        cutoff_hz: float,
        order: int = 4,
        time_dim: str | None = None,
    ) -> "ECoG":
        from cogpy.preprocess import filtering

        td = self.schema.time if time_dim is None else str(time_dim)
        out = filtering.highpassx(self.x, wh=float(cutoff_hz), order=int(order), axis=td)
        out = ensure_fs(out, fs=self.fs, schema=self.schema)
        return ECoG(out, kind=self.kind, schema=self.schema)

    def notch(
        self,
        *,
        w0_hz: float = 60.0,
        q: float = 30.0,
        time_dim: str | None = None,
    ) -> "ECoG":
        from scipy import signal

        td = self.schema.time if time_dim is None else str(time_dim)
        axis = self.x.get_axis_num(td)
        b, a = signal.iirnotch(float(w0_hz), float(q), fs=self.fs)
        y = signal.filtfilt(b, a, self.x.values, axis=axis)
        out = xr.DataArray(y, coords=self.x.coords, dims=self.x.dims, attrs=dict(self.x.attrs), name=self.x.name)
        out = ensure_fs(out, fs=self.fs, schema=self.schema)
        return ECoG(out, kind=self.kind, schema=self.schema)

