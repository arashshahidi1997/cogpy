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


def load_ecephys_metadata(lfp_path):
    """
    Read BIDS JSON, channels.tsv, and electrodes.tsv.
    Returns a dictionary of everything needed downstream.
    """
    lfp_path = Path(lfp_path)

    stem = lfp_path.name.replace("_ecephys.lfp", "")
    base = lfp_path.with_name(stem)

    json_path = base.with_name(stem + "_ecephys.json")
    ch_path = base.with_name(stem + "_channels.tsv")
    elec_path = base.with_name((stem + "_electrodes.tsv").replace("_recording-lf", ""))

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


def from_file(dat_file: Union[str, Path], as_float=False) -> xr.DataArray:
    """
    Creates a memory-mapped (dask) array from a .dat file, reshaped to (num_channels, num_samples).

    Parameters:
            - dat_file (str): Path to the .dat file.

    Returns:
            - xr.DataArray: Dask-backed DataArray with dimensions ('AP', 'ML', 'time') and attribute 'fs'.
    """
    metadata = load_ecephys_metadata(dat_file)
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
