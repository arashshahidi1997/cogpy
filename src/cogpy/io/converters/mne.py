from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import xarray as xr

from . import _deps
from ._units import Unit, scale_to_volts
from ._xarray_views import as_ieeg_time_channel, extract_time_channel_view

GridOrder = Literal["ML_AP"]


@dataclass(frozen=True)
class XarrayToMNEReport:
    mode: Literal["tc", "grid"]
    sfreq: float
    n_times: int
    n_channels: int
    grid_shape: tuple[int, int] | None  # (n_ml, n_ap) if grid else None
    order: GridOrder | None
    unit_in: Unit
    scaled_to_volts: bool


def to_mne(
    da: xr.DataArray,
    *,
    ch_type: str = "ecog",
    unit: Unit = "V",
    time_slice_s: tuple[float, float] | None = None,
    pick_channel: Sequence[int] | None = None,
    ch_name_style: Literal["index", "coord"] = "coord",
    verbose: str | None = "error",
    return_report: bool = False,
):
    """
    Convert PixECoG-style xarray DataArray to an MNE RawArray.

    Notes
    -----
    - MNE expects Volts for eeg/ecog/seeg channels. If your data are in uV/µV,
      set unit='uV' to scale by 1e-6.
    """
    if not isinstance(da, xr.DataArray):
        raise TypeError("da must be an xarray.DataArray")

    mne = _deps.require("mne", extra="interop-mne")

    mode: Literal["tc", "grid"] = "grid" if {"ML", "AP"}.issubset(set(da.dims)) else "tc"
    grid_shape: tuple[int, int] | None = None
    if mode == "grid":
        try:
            grid_shape = (int(da.sizes["ML"]), int(da.sizes["AP"]))
        except Exception:  # noqa: BLE001
            grid_shape = None

    da_tc = as_ieeg_time_channel(da)

    if time_slice_s is not None:
        t0, t1 = time_slice_s
        da_tc = da_tc.sel(time=slice(float(t0), float(t1)))

    if pick_channel is not None:
        da_tc = da_tc.isel(channel=list(pick_channel))

    view = extract_time_channel_view(da_tc, ch_name_style=ch_name_style)
    x_tc_v, scaled_to_v = scale_to_volts(view.x_tc, unit)

    x_ct = x_tc_v.T  # (n_ch, n_times)
    info = mne.create_info(ch_names=view.ch_names, sfreq=float(view.fs), ch_types=ch_type)
    raw = mne.io.RawArray(x_ct, info, verbose=verbose)

    raw.info["description"] = (
        f"source=xarray; mode={mode}; unit_in={unit}; scaled_to_V={scaled_to_v}; "
        + (f"grid_shape={grid_shape}; order=ML_AP" if mode == "grid" else "")
    )

    if return_report:
        rep = XarrayToMNEReport(
            mode=mode,
            sfreq=float(view.fs),
            n_times=int(x_ct.shape[1]),
            n_channels=int(x_ct.shape[0]),
            grid_shape=grid_shape,
            order="ML_AP" if mode == "grid" else None,
            unit_in=unit,
            scaled_to_volts=scaled_to_v,
        )
        return raw, rep

    return raw

