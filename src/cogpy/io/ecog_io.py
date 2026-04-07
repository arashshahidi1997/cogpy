"""
Module: ecog_io
Status: WIP
Last Updated: 2025-08-26
Author: Arash Shahidi, A.Shahidi@campus.lmu.de

Summary:


Functions:

Classes:

Constants:

Note:
        Matrix conventions in Python & Matlab:
        # Row Major channel index (C-order, Python convention)
        # sigx_flat = sigx.stack(ch=('AP','ML')) # time, ch=(AP, ML)

        # Column Major (F-order, MATLAB convention) channel index
        # sigx_flat = sigx.stack(ch=('ML','AP')) # time, ch=(ML, AP)


Example:
"""

from pathlib import Path
import numpy as np
import xarray as xr
from typing import Dict, Any, Union
from cogpy.utils.imports import import_optional

da = import_optional("dask.array")
from dask.diagnostics import ProgressBar
from . import xml_io
from .save_utils import save_options
from xarray import open_zarr as from_zarr


def from_arr(arr: np.ndarray, time_ax: int, ap_ax: int, ml_ax: int) -> xr.DataArray:
    """
    Converts a numpy array to an xarray.DataArray and selects a time slice.

    Parameters
    ----------
    arr : np.ndarray
            Input array with shape (time, h, w).
    time_ax : int
            Index of the time axis.
    ap_ax : int
            Index of the anterior-posterior axis.
    ml_ax : int
            Index of the medial-lateral axis.

    Returns
    -------
    sigx : xarray.DataArray
            DataArray with (time, AP, ML) dimensions.

    Note
    ----
    Matrix conventions in Python & Matlab:
    # if array was loaded from matlab (Column Major)
    sigx = ecog_io.from_numpy(arr, time_ax=0, ap_ax=2, ml_ax=1)
    # otherwise (Row Major)
    sigx = ecog_io.from_numpy(arr, time_ax=0, ap_ax=1, ml_ax=2)

    """
    sigx = xr.DataArray(arr, dims=("time", "AP", "ML"))
    return sigx


def from_file(
    dat_file: Union[str, Path],
    xml_file: str = None,
    metadata: Dict[str, Any] = None,
    ML_coords: np.ndarray = None,
    AP_coords: np.ndarray = None,
) -> xr.DataArray:
    """
    Creates a memory-mapped (dask) array from a .dat file, reshaped to (num_channels, num_samples).

    Parameters:
            - dat_file (str): Path to the .dat file.
            - xml_file (str, optional): Path to the XML file containing metadata. If not provided, metadata must be supplied.
            - metadata (dict, optional): Dictionary containing metadata such as 'fs', 'ncol', 'nrow', and 'dtype'.
            - ML_coords (np.ndarray, optional): Array of medial-lateral coordinates. If not provided, defaults to range of ncol.
            - AP_coords (np.ndarray, optional): Array of anterior-posterior coordinates.
                    If not provided, defaults to range of nrow.

    Returns:
            - xr.DataArray: Dask-backed DataArray with dimensions ('AP', 'ML', 'time') and attribute 'fs'.
    """
    # if metadata is not directly provided, attempt to read from xml_file
    if metadata is None:
        # if xml_file is not provided assume same path as dat_file except with .xml extension
        xml_file = Path(dat_file).with_suffix(".xml")
        metadata = xml_io.parse_meta_from_xml(xml_file)
        # assert 'fs', 'ncol', 'nrow' in metadata
        assert "fs" in metadata
        assert "ncol" in metadata
        assert "nrow" in metadata

    if ML_coords is None:
        ML_coords = np.arange(metadata["ncol"])
    if AP_coords is None:
        AP_coords = np.arange(metadata["nrow"])
    memmap_array = np.memmap(dat_file, dtype=metadata["dtype"], mode="r")
    arr = memmap_array.reshape(-1, metadata["ncol"], metadata["nrow"])
    dask_array = da.from_array(arr, asarray=False, fancy=False)
    time_coords = np.arange(arr.shape[0]) / metadata["fs"]
    data_array = xr.DataArray(
        dask_array,
        dims=("time", "ML", "AP"),
        coords={"time": time_coords, "ML": ML_coords, "AP": AP_coords},
    )
    data_array.attrs["fs"] = metadata["fs"]
    data_array = data_array.transpose("AP", "ML", "time")
    return data_array.astype(float)


def assert_ecog(sigx: xr.DataArray) -> None:
    assert "fs" in sigx.attrs, "Sampling frequency 'fs' not found in sigx attributes."
    assert (
        "AP" in sigx.dims and "ML" in sigx.dims and "time" in sigx.dims
    ), "sigx must have dimensions 'AP', 'ML', and 'time'."


def save_dat(
    arr: np.ndarray, dat_file: str, extension=".dat", dtype=np.int16, **save_kwargs
):
    """
    arr: array (samples, channels)
    dat_file: target file
    operation: default '.copy'
    overwrite: default False
    """
    dat_out = save_options(dat_file, extension, **save_kwargs)
    arr.reshape(-1).tofile(dat_out)
    print(f"signal saved to \n {dat_out}")


# OpenEphys
def oephys_load(directory, nsamples=30000):
    from open_ephys.analysis import Session

    session = Session(directory)
    continuous_obj = session.recordnodes[0].recordings[0].continuous[0]
    data = continuous_obj.get_samples(start_sample_index=0, end_sample_index=nsamples)
    time_vec = continuous_obj.timestamps[:nsamples]
    fs = continuous_obj.metadata["sample_rate"]
    return data, time_vec, fs


# Output
def da_to_zarr(output_file, sigx, **kwargs):
    with ProgressBar():
        name = getattr(sigx, "name")
        sigx.to_dataset(name=name).to_zarr(
            output_file, mode="w", zarr_format=2, **kwargs
        )


def ds_to_zarr(output_file, ds, **kwargs):
    with ProgressBar():
        ds.to_zarr(output_file, mode="w", zarr_format=2, **kwargs)


def to_zarr(output_file, obj, **kwargs):
    if isinstance(obj, xr.DataArray):
        da_to_zarr(output_file, obj, **kwargs)
    elif isinstance(obj, xr.Dataset):
        ds_to_zarr(output_file, obj, **kwargs)
    else:
        raise TypeError("obj must be either an xarray.DataArray or xarray.Dataset.")


# conversions
def zarr_to_dat(zarr_file: str, dat_file: str):
    sigx = from_zarr(zarr_file)["sigx"]
    assert_ecog(sigx)
    arr = sigx.transpose("time", "AP", "ML").data
    arr = arr.reshape(arr.shape[0], -1)
    save_dat(arr, dat_file, extension=".dat", dtype=np.int16, overwrite=True)


# neuropixels
import numpy as np
from pathlib import Path
from ..utils import xarr as xut


def load_ecog_npix(DATA_FILE, XML_FILE, xml_ecog_file, reshape=True):
    data = np.fromfile(DATA_FILE, dtype=np.int16)

    # load xml
    xml_ecog = xml_io.parse_meta_from_xml(xml_ecog_file)

    dat = from_file(DATA_FILE, XML_FILE).rename({"AP": "ch"})
    fs = xml_ecog["fs"]
    dat["fs"] = fs
    ecog_dat, npix_dat = dat[:, 0, :256], dat[:, 0, 256:]
    if reshape:
        ecog_dat = xut.reshape_dimension(
            ecog_dat, "ch", (16, 16), new_dims=("ML", "AP")
        )
    return ecog_dat, npix_dat, fs


def separate_ecog_npix(DATA_FILE, XML_FILE, xml_ecog_file, DATA_DIR):
    print("Loading data...")
    ecog_flat_dat, npix_dat, fs = load_ecog_npix(
        DATA_FILE, XML_FILE, xml_ecog_file, reshape=False
    )

    ecog_file = DATA_DIR / "ecog.zarr"
    npix_file = DATA_DIR / "npix.zarr"
    # save the data to disk
    ecog_flat_dat.to_zarr(ecog_file)
    npix_dat.to_zarr(npix_file)
    print(f"Data saved to {ecog_file} and {npix_file}")
