"""
**three tiny helper functions**:

1. **`load_bids_metadata()`** → reads JSON, electrodes, channels
2. **`load_binary()`** → memmaps and reshapes `(time, channel)`
3. **`reshape_to_grid()`** → reorders channels to `(AP, ML, time)`

And finally:

4. **`make_xarray()`** → builds the actual `xr.DataArray`
5. **`load_ieeg()`** → a tiny wrapper that ties everything together
"""

# ✅ **1. Load BIDS Metadata**
import json

import pandas as pd
from pathlib import Path
import numpy as np
import xarray as xr
import dask.array as da
from typing import Union, Dict, Any


def load_ieeg_metadata(lfp_path):
    """
    Read BIDS JSON, channels.tsv, and electrodes.tsv.
    Returns a dictionary of everything needed downstream.
    """
    lfp_path = Path(lfp_path)

    stem = lfp_path.name.replace("_ieeg.lfp", "")
    base = lfp_path.with_name(stem)

    json_path = base.with_name(stem + "_ieeg.json")
    ch_path = base.with_name(stem + "_channels.tsv")
    elec_path = base.with_name(stem + "_electrodes.tsv")

    with open(json_path) as f:
        meta = json.load(f)

    ch = pd.read_csv(ch_path, sep="\t")
    elec = pd.read_csv(elec_path, sep="\t")

    return {
        "fs": meta["SamplingFrequency"],
        "nch": meta["ChannelCount"],
        "nrow": meta["RowCount"],
        "ncol": meta["ColumnCount"],
        "dtype": np.dtype(ch["dtype"].iloc[0]),
        "rows": elec["row"].to_numpy(int),
        "cols": elec["col"].to_numpy(int),
        "AP": elec.groupby("row")["AP"]
        .mean()
        .reindex(range(meta["RowCount"]))
        .to_numpy(),
        "ML": elec.groupby("col")["ML"]
        .mean()
        .reindex(range(meta["ColumnCount"]))
        .to_numpy(),
    }


def from_file(dat_file: Union[str, Path], grid=False, as_float=False) -> xr.DataArray:
    """
    Creates a memory-mapped (dask) array from a .dat file, reshaped to (num_channels, num_samples).

    Parameters:
            - dat_file (str): Path to the .dat file.

    Returns:
            - xr.DataArray: Dask-backed DataArray with dimensions ('AP', 'ML', 'time') and attribute 'fs'.
    """
    metadata = load_ieeg_metadata(dat_file)
    memmap_array = np.memmap(dat_file, dtype=metadata["dtype"], mode="r")
    if grid:
        arr = memmap_array.reshape(-1, metadata["ncol"], metadata["nrow"])
        dask_array = da.from_array(arr, asarray=False, fancy=False)
        time_coords = np.arange(arr.shape[0]) / metadata["fs"]
        data_array = xr.DataArray(
            dask_array,
            dims=("time", "ML", "AP"),
            coords={"time": time_coords, "ML": metadata["ML"], "AP": metadata["AP"]},
        )
        data_array.attrs["fs"] = metadata["fs"]
        data_array = data_array.transpose("AP", "ML", "time")
    else:
        arr = memmap_array.reshape(-1, metadata["nch"])
        dask_array = da.from_array(arr, asarray=False, fancy=False)
        time_coords = np.arange(arr.shape[0]) / metadata["fs"]
        channel_coords = np.arange(metadata["nch"])
        data_array = xr.DataArray(
            dask_array,
            dims=("time", "ch"),
            coords={"time": time_coords, "ch": channel_coords},
        )
        data_array.attrs["fs"] = metadata["fs"]
        data_array.attrs["ncol"] = metadata["ncol"]
        data_array.attrs["nrow"] = metadata["nrow"]
    return data_array.astype(float) if as_float else data_array
