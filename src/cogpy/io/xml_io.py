from pathlib import Path
import xmltodict
import numpy as np
import pandas as pd
from typing import Dict, Any
from .save_utils import save_options


# %% Anatomical Remapping
def read_anat_map(xml_dict: Dict[str, Any]) -> pd.DataFrame:
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


# %% .xml file
def parse_xml(xml_file):
    xml_file = Path(xml_file).with_suffix(".xml")
    with open(xml_file, "rb") as file:
        xml_dict = xmltodict.parse(file.read())

    return xml_dict


def unparse_xml(xml_dict, xml_file, **save_kwargs):
    xml_output = save_options(xml_file, ".xml", **save_kwargs)
    # write to output file
    with open(xml_output, "w") as out:
        out.write(xmltodict.unparse(xml_dict))

    print(f"metadata saved to \n {xml_output}")


def parse_meta_from_xml(xml_file):
    # Use the existing 'ld' module functions directly
    xml_dict = parse_xml(xml_file)

    # Acquisition system information
    acq = read_acquistionSystem(xml_dict)
    fs = float(acq["samplingRate"])
    nbits = acq["nBits"]

    # Process 'nbits' to ensure it's an integer
    if isinstance(nbits, str):
        if "float" in nbits:
            nbits = nbits.replace("float", "")
        elif "int" in nbits:
            nbits = nbits.replace("int", "")
    nbits = int(nbits)
    if nbits > 16:
        dtype = f"float{nbits}"
    else:
        dtype = f"int{nbits}"

    # Anatomical map and dimensions
    anat_map = read_anat_map(xml_dict)
    anat = anat_map.set_index("grp")

    ncol = anat.index.nunique()
    shank_heights = anat.index.value_counts()
    nrow = shank_heights.iloc[0]
    nchan = ncol * nrow

    # Ensure all shanks have the same height
    if not (shank_heights == nrow).all():
        raise ValueError(
            "Shanks have different heights! The algorithms are written for channel grids."
        )

    # Compile metadata into a dictionary
    metadata = {
        "nchan": nchan,
        "nrow": nrow,
        "ncol": ncol,
        "fs": fs,
        "dtype": dtype,
        "anat_map": anat_map,
    }

    return metadata


def read_dtype(xml_dict):
    acq = read_acquistionSystem(xml_dict)
    return "int{nbits}".format(nbits=acq["nBits"])


def read_acquistionSystem(xml_dict):
    return xml_dict["parameters"]["acquisitionSystem"]
