"""
Title: converters.py
Status: STABLE
Summary: Reusable converters for BIDS LFP <-> Zarr used by PixECoG pipelines.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
import xarray as xr
import mne

from pathlib import Path


def bids_lfp_to_zarr(lfp_path: str, zarr_path: str) -> None:
    """Convert a BIDS IEEG `.lfp` (with JSON/TSV sidecars) to a Zarr dataset.

    Parameters
    ----------
    lfp_path:
        Path to the BIDS IEEG LFP file (e.g., `*_ieeg.lfp`).
        Sidecars (`*_ieeg.json`, `*_channels.tsv`, `*_electrodes.tsv`) are expected
        next to the file as per `cogpy.io.ieeg_io`.
    zarr_path:
        Output Zarr directory path.
    """
    from cogpy.io import ecog_io, ieeg_io

    sigx = ieeg_io.from_file(lfp_path, grid=True, as_float=True)
    sigx.name = "sigx"

    dst = Path(zarr_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ecog_io.to_zarr(str(dst), sigx)


def zarr_to_dat(zarr_path: str, dat_path: str) -> None:
    """Convert a Zarr dataset containing `sigx` back to a flat binary signal.

    This mirrors the Snakemake `zarr2dat` rule behavior:
    - Load `sigx` from Zarr with shape (AP, ML, time)
    - Transpose to (time, ML, AP)
    - Flatten to (time, channels)
    - Write to a `.lfp` (or `.dat`) binary using `ecog_io.save_dat`

    Parameters
    ----------
    zarr_path:
        Path to the input Zarr directory containing a `sigx` variable.
    dat_path:
        Output file path (typically `.lfp`).
    """
    import numpy as np

    from cogpy.io import ecog_io

    sigx = ecog_io.from_zarr(zarr_path)["sigx"]  # AP, ML, time
    arr = sigx.data.transpose(2, 1, 0)  # time, ML, AP
    arr_flat = arr.reshape(arr.shape[0], -1)  # time, channels

    dst = Path(dat_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ecog_io.save_dat(arr_flat.compute(), str(dst), extension=dst.suffix, dtype=np.int16)

GridOrder = Literal["ML_AP", "AP_ML"]
Unit = Literal["V", "uV", "µV"]


@dataclass(frozen=True)
class XarrayToMNEReport:
    mode: Literal["tc", "grid"]
    sfreq: float
    n_times: int
    n_channels: int
    grid_shape: Optional[tuple[int, int]]  # (n_ml, n_ap) if grid else None
    order: Optional[GridOrder]
    unit_in: Unit
    scaled_to_volts: bool


def xr_to_mne(
    da: xr.DataArray,
    *,
    ch_type: str = "ecog",
    unit: Unit = "V",
    grid_order: GridOrder = "ML_AP",
    # optional xarray-side cropping before compute:
    time_slice_s: tuple[float, float] | None = None,    # (t0, t1) in seconds
    pick_ch: Sequence[int] | None = None,               # for tc
    pick_ml: Sequence[int] | None = None,               # for grid
    pick_ap: Sequence[int] | None = None,               # for grid
    # naming:
    ch_name_style: Literal["index", "coord"] = "coord",
    # misc:
    verbose: str | None = "error",
    return_report: bool = False,
) -> mne.io.Raw | tuple[mne.io.Raw, XarrayToMNEReport]:
    """
    Convert PixECoG-style xarray DataArray to an MNE RawArray.

    Supported inputs
    ----------------
    - dims ('time', 'ch')              -> tc mode
    - dims ('time', 'ML', 'AP')        -> grid mode

    Notes
    -----
    - MNE expects Volts for eeg/ecog/seeg channels. If your data are in uV/µV,
      set unit='uV' to scale by 1e-6.
    - RawArray is in-memory. Use time_slice_s / picks to reduce memory use.
    """
    if not isinstance(da, xr.DataArray):
        raise TypeError("da must be an xarray.DataArray")

    # ---- sampling rate ----
    sfreq = float(da.attrs.get("fs", np.nan))
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Missing/invalid sampling rate: da.attrs['fs'] must be a positive number")

    # ---- optional time crop (xarray-side) ----
    da2 = da
    if time_slice_s is not None:
        t0, t1 = time_slice_s
        if "time" not in da2.coords:
            raise ValueError("time_slice_s requested but da has no 'time' coordinate")
        da2 = da2.sel(time=slice(float(t0), float(t1)))

    # ---- detect mode by dims ----
    dims = tuple(da2.dims)
    if dims == ("time", "ch"):
        mode: Literal["tc", "grid"] = "tc"
        if pick_ch is not None:
            da2 = da2.isel(ch=list(pick_ch))

        x_tc = np.asarray(da2.data)  # (n_times, n_ch)
        n_times, n_ch = x_tc.shape

        # channel names
        if ch_name_style == "coord" and "ch" in da2.coords:
            ch_names = [str(v) for v in da2["ch"].values]
        else:
            ch_names = [f"ch{ii:03d}" for ii in range(n_ch)]

        order = None
        grid_shape = None

    elif dims == ("time", "ML", "AP"):
        mode = "grid"
        if pick_ml is not None:
            da2 = da2.isel(ML=list(pick_ml))
        if pick_ap is not None:
            da2 = da2.isel(AP=list(pick_ap))

        x_tmlap = np.asarray(da2.data)  # (n_times, n_ml, n_ap)
        n_times, n_ml, n_ap = x_tmlap.shape
        grid_shape = (n_ml, n_ap)
        order = grid_order

        ml_vals = da2["ML"].values if "ML" in da2.coords else np.arange(n_ml)
        ap_vals = da2["AP"].values if "AP" in da2.coords else np.arange(n_ap)

        if grid_order == "ML_AP":
            # channel index = ml * n_ap + ap  (matches your from_file comment)
            x_tc = x_tmlap.reshape(n_times, n_ml * n_ap)
            if ch_name_style == "coord":
                ch_names = [f"ML{ml_vals[ml]}_AP{ap_vals[ap]}" for ml in range(n_ml) for ap in range(n_ap)]
            else:
                ch_names = [f"ch{ii:03d}" for ii in range(n_ml * n_ap)]
        elif grid_order == "AP_ML":
            # channel index = ap * n_ml + ml
            x_tc = np.transpose(x_tmlap, (0, 2, 1)).reshape(n_times, n_ap * n_ml)
            if ch_name_style == "coord":
                ch_names = [f"AP{ap_vals[ap]}_ML{ml_vals[ml]}" for ap in range(n_ap) for ml in range(n_ml)]
            else:
                ch_names = [f"ch{ii:03d}" for ii in range(n_ap * n_ml)]
        else:
            raise ValueError("grid_order must be 'ML_AP' or 'AP_ML'")

        n_ch = x_tc.shape[1]

    else:
        raise ValueError(
            f"Unsupported dims {dims}. Expected ('time','ch') or ('time','ML','AP')."
        )

    # ---- unit scaling to volts ----
    scaled_to_volts = False
    if unit in ("uV", "µV"):
        x_tc = x_tc * 1e-6
        scaled_to_volts = True
    elif unit == "V":
        pass
    else:
        raise ValueError("unit must be 'V', 'uV', or 'µV'")

    # ---- build MNE RawArray ----
    x_ct = x_tc.T  # (n_ch, n_times)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_type)
    raw = mne.io.RawArray(x_ct, info, verbose=verbose)

    # store useful provenance
    raw.info["description"] = (
        f"source=xarray; mode={mode}; unit_in={unit}; scaled_to_V={scaled_to_volts}; "
        + (f"grid_shape={grid_shape}; order={order}" if mode == "grid" else "")
    )

    if return_report:
        rep = XarrayToMNEReport(
            mode=mode,
            sfreq=sfreq,
            n_times=int(n_times),
            n_channels=int(n_ch),
            grid_shape=grid_shape,
            order=order,
            unit_in=unit,
            scaled_to_volts=scaled_to_volts,
        )
        return raw, rep

    return raw

