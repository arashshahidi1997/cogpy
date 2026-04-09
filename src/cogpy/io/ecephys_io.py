"""BIDS ECEPHYS reader for electrophysiology ``.dat`` files.

.. note:: **Lab-internal module.** Assumes the Bhatt Lab BIDS-ECEPHYS
   directory layout and sidecar conventions.  Not part of the stable
   public API — external users should prefer :mod:`cogpy.io.converters`.

Functions
---------
load_ecephys_metadata
    Load metadata from BIDS sidecar files.
from_file
    Create an ``xarray.DataArray`` from a ``.dat`` file.

Example
-------
::

    from cogpy.io import ecephys_io
    da = ecephys_io.from_file('sub-01_ses-01_task-free_ecephys.lfp')
"""

# ✅ **1. Load BIDS Metadata**
import json

import pandas as pd
from pathlib import Path
import numpy as np
import xarray as xr
from cogpy.utils.imports import import_optional

da = import_optional("dask.array")
from typing import Union


def resolve_ecephys_sidecars(lfp_path: Union[str, Path]) -> dict[str, Path]:
    """Resolve BIDS sidecar paths for an ecephys LFP file."""
    lfp_path = Path(lfp_path)
    stem = lfp_path.name.replace("_ecephys.lfp", "")
    base = lfp_path.with_name(stem)
    return {
        "json": base.with_name(stem + "_ecephys.json"),
        "channels": base.with_name(stem + "_channels.tsv"),
        # electrodes sidecar in this repo drops the recording entity.
        "electrodes": base.with_name(
            (stem + "_electrodes.tsv").replace("_recording-lf", "")
        ),
    }


def load_ecephys_metadata(lfp_path, sidecars=None) -> dict:
    """
    Read BIDS JSON, channels.tsv, and electrodes.tsv.
    Returns a dictionary of everything needed downstream.
    """
    if sidecars is None:
        sidecars = resolve_ecephys_sidecars(lfp_path)
    json_path = sidecars["json"]
    ch_path = sidecars["channels"]
    elec_path = sidecars["electrodes"]

    with open(json_path) as f:
        meta = json.load(f)

    ch = pd.read_csv(ch_path, sep="\t")
    elec = pd.read_csv(elec_path, sep="\t")

    return {
        "fs": meta["SamplingFrequency"],
        "nch": meta["ChannelCount"],
        "dtype": np.dtype(ch["dtype"].iloc[0]),
        "AP": elec["AP"].to_numpy(),
        "ML": elec["ML"].to_numpy(),
        "DV": elec["DV"].to_numpy(),
    }


def from_file(
    dat_file: Union[str, Path], sidecars=None, as_float=False
) -> xr.DataArray:
    """
    Creates a memory-mapped (dask) array from a .dat file, reshaped to (num_channels, num_samples).

    Parameters:
            - dat_file (str): Path to the .dat file.
            - sidecars (dict, optional): Dictionary with paths to sidecar files. If None, will attempt to resolve based on dat_file name.

    Returns:
            - xr.DataArray: Dask-backed DataArray with dimensions ('AP', 'ML', 'time') and attribute 'fs'.
    """
    metadata = load_ecephys_metadata(dat_file, sidecars=sidecars)
    memmap_array = np.memmap(dat_file, dtype=metadata["dtype"], mode="r")
    arr = memmap_array.reshape(-1, metadata["nch"])
    dask_array = da.from_array(arr, asarray=False, fancy=False)
    time_coords = np.arange(arr.shape[0]) / metadata["fs"]
    DV_coords = metadata["DV"]
    data_array = xr.DataArray(
        dask_array, dims=("time", "DV"), coords={"time": time_coords, "DV": DV_coords}
    )
    data_array.attrs["fs"] = metadata["fs"]
    return data_array.astype(float) if as_float else data_array
