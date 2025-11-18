import numpy as np
import xarray as xr
import pandas as pd
from typing import Tuple, Dict, Any
from ..utils.reshape import reshape_axes
from ..utils import xarr as xut


def create_empty_anat_map(width, height):
    nchan = width * height
    return pd.DataFrame(
        {
            "id": np.arange(nchan),
            "grp": np.repeat(np.arange(width), height),
            "skip": np.zeros(nchan),
        }
    )


# %% Anatomical Remapping
def read_anat_map(xml_dict):
    """
    columnwise
    """
    # nchan = int(xml_dict['parameters']['acquisitionSystem']['nChannels'])
    groups = xml_dict["parameters"]["anatomicalDescription"]["channelGroups"]["group"]

    # if isinstance(groups, dict):
    #     chan = [[int(ch['#text']), igrp, int(ch['@skip'])] for ch in groups['channel']]
    #     anat_map = np.array(chan).transpose(1,0).reshape(2,-1).T

    # else:
    chan = []
    if isinstance(groups, dict):
        groups = [groups]

    for igrp, grp in enumerate(groups):
        chan.append(
            [[int(ch["#text"]), igrp, int(ch["@skip"])] for ch in grp["channel"]]
        )

    anat_map = np.array(chan).transpose(2, 0, 1).reshape(3, -1).T
    # channels, groups, (id, skip), original: info:2, channel(row):1, grp(column):0

    return pd.DataFrame(anat_map, columns=["id", "grp", "skip"])


def write_anat_map(anat_map, xml_dict, grid_shape=(16, 16)):
    if 1 in grid_shape:
        for ich in range(len(anat_map)):
            anat_map_ihiw = xml_dict["parameters"]["anatomicalDescription"][
                "channelGroups"
            ]["group"]["channel"][ich]

            anat_map_ihiw["@skip"] = str(int(anat_map.skip[ich]))
            anat_map_ihiw["#text"] = str(int(anat_map.id[ich]))

    else:
        for ich, (iw, ih) in enumerate(np.ndindex(grid_shape)):
            anat_map_ihiw = xml_dict["parameters"]["anatomicalDescription"][
                "channelGroups"
            ]["group"][iw]["channel"][ih]

            anat_map_ihiw["@skip"] = str(int(anat_map.skip[ich]))
            anat_map_ihiw["#text"] = str(int(anat_map.id[ich]))


def update_anat_map(bad_channels, anat_map):
    """
    saves the bad channels in the skip field of the `anat_map` dataframe.

    bad_channels:
        1d boolean array (Row/Column order)
    """
    anat_map.skip = bad_channels


def remap_array(arr: np.ndarray, anat_map: np.ndarray, gridshape: Tuple) -> np.ndarray:
    """
    Remaps an array to a specified grid shape based on an anatomical map.

    Parameters:
    - arr: np.array, the array to be remapped (channel, samples).
    - anat_map: np.array, the anatomical map with channel IDs.
    - gridshape: tuple (nrows, ncols), the target grid shape.
    - nchan: int, the number of channels.

    Returns:
    - np.array: The remapped array.
    """
    assert arr.shape[0] == len(anat_map)
    order = detect_order(anat_map.reshape(gridshape))
    if order is None:
        arr = arr[anat_map]
    arr = reshape_axes(arr, 0, gridshape)
    if order == "F":
        arr = np.swapaxes(arr, 0, 1)
    return arr


def detect_order(arr):
    """
    Detects the ordering of a 2D square array.

    Parameters:
    - arr: A 2D numpy array.

    Returns:
    - 'C' for C order, 'F' for Fortran order, or None if unordered.
    """
    c_order = arr.ravel(order="C")
    f_order = arr.ravel(order="F")

    is_c_order = np.all(np.diff(c_order) > 0)
    is_f_order = np.all(np.diff(f_order) > 0)

    if is_c_order and not is_f_order:
        return "C"
    elif is_f_order and not is_c_order:
        return "F"
    else:
        return None


def test_detect_order():
    """
    Tests the detect_order function.
    """
    arr_c = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    arr_f = np.array([[0, 3, 6], [1, 4, 7], [2, 5, 8]])

    arr_unordered = np.array([[0, 2, 4], [5, 3, 1], [7, 6, 8]])

    assert detect_order(arr_c) == "C"
    assert detect_order(arr_f) == "F"
    assert detect_order(arr_unordered) is None


def reshape_dataarray_to_grid(
    data_array: xr.DataArray, metadata: Dict[str, Any]
) -> xr.DataArray:
    """
    Reshapes and transposes a DataArray to grid format based on provided metadata.

    Parameters:
    - data_array (xr.DataArray): Input DataArray.
    - metadata (Dict[str, Any]): Metadata containing 'ncol', 'nrow'.

    Returns:
    - xr.DataArray: Reshaped and transposed DataArray with dimensions ('AP', 'ML', 'time').
    """
    reshaped_array = xut.reshape_dimension(
        data_array, "ch", (metadata["ncol"], metadata["nrow"]), ("ML", "AP")
    )
    reshaped_array = reshaped_array.transpose("AP", "ML", "time")
    reshaped_array.assign_coords(
        AP=np.arange(metadata["nrow"]), ML=np.arange(metadata["ncol"])
    )
    return reshaped_array
