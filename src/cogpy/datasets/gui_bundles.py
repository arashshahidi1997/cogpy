from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from cogpy.core.plot.hv.grid_indexing import to_apml_view

from .entities import example_ieeg_grid
from .schemas import validate_ieeg_grid

__all__ = [
    "IEEGGridGuiBundle",
    "SpectrogramGuiBundle",
    "ieeg_grid_bundle",
    "spectrogram_bursts_bundle",
]


Mode = Literal["small", "large"]
LargeBackend = Literal["numpy", "dask"]


@dataclass(frozen=True)
class IEEGGridGuiBundle:
    sig_grid: xr.DataArray  # ("time","ML","AP")
    sig_apml: xr.DataArray  # ("time","AP","ML") view
    sig_tc: xr.DataArray  # ("time","channel") stacked from sig_apml
    rms_apml: np.ndarray  # (n_ap, n_ml) from sig_apml.std("time")
    n_ap: int
    n_ml: int
    fs: float | None
    meta: dict
    ap_coords: np.ndarray
    ml_coords: np.ndarray
    atlas_image: np.ndarray | None = None


def ieeg_grid_bundle(
    *,
    mode: Mode = "small",
    seed: int = 0,
    with_atlas: bool = False,
    large_backend: LargeBackend = "numpy",
) -> IEEGGridGuiBundle:
    """
    Bundle for developing the iEEG grid + stacked-trace GUIs.

    Notes
    -----
    `large_backend="dask"` is supported only if dask is installed, and currently
    wraps a generated numpy array into a dask array. True end-to-end lazy
    viewing requires viewer support (implemented in later phases).
    """
    sig_grid = example_ieeg_grid(mode=mode, seed=seed)
    validate_ieeg_grid(sig_grid)

    if mode == "large" and large_backend == "dask":
        try:
            import dask.array as da  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise ImportError("large_backend='dask' requested but dask is not installed") from e

        arr = da.from_array(sig_grid.data, chunks=(50_000, -1, -1))
        sig_grid = sig_grid.copy(data=arr)

    sig_apml = to_apml_view(sig_grid)
    n_ap = int(sig_apml.sizes["AP"])
    n_ml = int(sig_apml.sizes["ML"])

    sig_tc = sig_apml.stack(channel=("AP", "ML")).reset_index("channel")  # ("time","channel"), with AP/ML coords
    sig_tc = sig_tc.transpose("time", "channel")
    sig_tc.name = "ieeg_tc"
    sig_tc.attrs.update(sig_grid.attrs)

    rms = sig_apml.std(dim="time").transpose("AP", "ML").values
    rms_apml = np.asarray(rms)

    ap_coords = np.asarray(sig_grid["AP"].values)
    ml_coords = np.asarray(sig_grid["ML"].values)

    atlas_image = None
    if with_atlas:
        # Placeholder hook for future atlas bundle support.
        atlas_image = None

    return IEEGGridGuiBundle(
        sig_grid=sig_grid,
        sig_apml=sig_apml,
        sig_tc=sig_tc,
        rms_apml=rms_apml,
        n_ap=n_ap,
        n_ml=n_ml,
        fs=sig_grid.attrs.get("fs", None),
        meta={"mode": mode, "seed": int(seed), "large_backend": large_backend},
        ap_coords=ap_coords,
        ml_coords=ml_coords,
        atlas_image=atlas_image,
    )


@dataclass(frozen=True)
class SpectrogramGuiBundle:
    spec: xr.DataArray  # ("ml","ap","time","freq")
    bursts: pd.DataFrame
    meta: dict


def spectrogram_bursts_bundle(
    *,
    mode: Mode = "small",
    seed: int = 0,
    kind: Literal["toy", "ar_grid"] = "toy",
) -> SpectrogramGuiBundle:
    if kind == "toy":
        from .entities import example_bursts_table, example_spectrogram4d

        spec = example_spectrogram4d(mode=mode, seed=seed)
        bursts = example_bursts_table(spec)
        return SpectrogramGuiBundle(spec=spec, bursts=bursts, meta={"mode": mode, "seed": int(seed), "kind": kind})

    if kind == "ar_grid":
        from .tensor import AROscillatorGrid

        if mode == "small":
            grid = AROscillatorGrid.make(duration=4.0, fs=1250.0, nap=10, nml=10, n_bursts=8, seed=seed)
        elif mode == "large":
            grid = AROscillatorGrid.make(duration=8.0, fs=1250.0, nap=16, nml=16, n_bursts=20, seed=seed)
        else:
            raise ValueError(f"mode must be 'small' or 'large', got {mode!r}")

        return SpectrogramGuiBundle(
            spec=grid.spectrogram,
            bursts=grid.bursts,
            meta={"mode": mode, "seed": int(seed), "kind": kind, "fs": grid.fs},
        )

    raise ValueError(f"kind must be 'toy' or 'ar_grid', got {kind!r}")
