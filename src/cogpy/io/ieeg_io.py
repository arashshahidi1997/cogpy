"""
Module: ieeg_io
Status: WIP
tag: pixecog
Last Updated: 2025-11-21
Author: Arash Shahidi, A.Shahidi@campus.lmu.de

Summary:
    Utilities for reading intracranial EEG data in BIDS IEEG format.

    Functions:
    load_ieeg_metadata: Load metadata from BIDS sidecar files.
    from_file: Create xarray.DataArray from .dat file.

Example:
    from cogpy.io import ieeg_io
    da = ieeg_io.from_file('sub-01_ses-01_task-free_ieeg.lfp')
"""

from typing import Union
import xarray as xr
import dask.array as da
import numpy as np
from pathlib import Path
from .ieeg_sidecars import load_ieeg_metadata

def from_file(
    dat_file: Union[str, Path], 
    meta_file: Union[str, Path] = None, 
    grid: bool = False, 
    as_float: bool = False
) -> xr.DataArray:
    
    dat_file = Path(dat_file)
    # If meta_file is None, we look for sidecars next to the dat_file.
    # If meta_file is provided (json or lfp), we look for sidecars relative to it.
    metadata = load_ieeg_metadata(meta_file or dat_file)

    if metadata.dtype is None:
        # If channels.tsv wasn't found relative to the meta_source,
        # we might want a fail-safe or a default.
        raise ValueError(f"Dtype missing. No channels.tsv found for {meta_file or dat_file}")

    memmap_array = np.memmap(dat_file, dtype=metadata.dtype, mode="r")

    if grid:
        # Ensure we have the dimensions needed for reshaping
        if not (metadata.nrow and metadata.ncol):
            raise ValueError("Grid reshape requested but RowCount/ColumnCount missing from metadata.")
            
        # Reshape: (Samples, ML, AP) based on your original logic
        arr = memmap_array.reshape(-1, metadata.ncol, metadata.nrow)
        dask_array = da.from_array(arr, chunks="auto") # Let dask optimize chunks
        
        time_coords = np.arange(arr.shape[0]) / metadata.fs
        
        # Use metadata coordinates if they exist, otherwise fallback to indices
        ml_coords = metadata.ml_coords if metadata.ml_coords is not None else np.arange(metadata.ncol)
        ap_coords = metadata.ap_coords if metadata.ap_coords is not None else np.arange(metadata.nrow)

        data_array = xr.DataArray(
            dask_array,
            dims=("time", "ML", "AP"),
            coords={"time": time_coords, "ML": ml_coords, "AP": ap_coords},
        )
        data_array = data_array.transpose("AP", "ML", "time")
    
    else:
        # Linear channel representation
        arr = memmap_array.reshape(-1, metadata.nch)
        dask_array = da.from_array(arr, chunks="auto")
        
        time_coords = np.arange(arr.shape[0]) / metadata.fs
        data_array = xr.DataArray(
            dask_array,
            dims=("time", "ch"),
            coords={"time": time_coords, "ch": np.arange(metadata.nch)},
        )
        # Store grid info in attributes for later reconstruction
        data_array.attrs["ncol"] = metadata.ncol
        data_array.attrs["nrow"] = metadata.nrow

    data_array.attrs["fs"] = metadata.fs
    return data_array.astype(float) if as_float else data_array