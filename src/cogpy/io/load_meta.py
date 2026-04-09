"""Parse and write XML metadata for OpenEphys-style recordings.

.. note:: **Lab-internal module.** Assumes OpenEphys XML metadata files
   as used in the Bhatt Lab recording setup.  Not part of the stable
   public API.
"""

from pathlib import Path
from cogpy.utils.imports import import_optional

xmltodict = import_optional("xmltodict")
from ..io.save_utils import save_options
from .xml_anat_map import read_anat_map


def parse_xml(xml_file):
    """Read an XML metadata file and return its contents as a nested dict.

    Parameters
    ----------
    xml_file : str or Path
        Path to the ``.xml`` file (extension added if missing).

    Returns
    -------
    dict
        Parsed XML contents.
    """
    xml_file = Path(xml_file).with_suffix(".xml")
    with open(xml_file, "rb") as file:
        xml_dict = xmltodict.parse(file.read())

    return xml_dict


def unparse_xml(xml_dict, xml_file, **save_kwargs):
    """Write a dict back to an XML file.

    Parameters
    ----------
    xml_dict : dict
        Nested dict to serialize (as returned by :func:`parse_xml`).
    xml_file : str or Path
        Output path.
    **save_kwargs
        Forwarded to :func:`cogpy.io.save_utils.save_options`.
    """
    xml_output = save_options(xml_file, ".xml", **save_kwargs)
    # write to output file
    with open(xml_output, "w") as out:
        out.write(xmltodict.unparse(xml_dict))

    print(f"metadata saved to \n {xml_output}")


def parse_meta_from_xml(xml_file):
    """Extract recording metadata (grid shape, fs, dtype) from an XML file.

    Parameters
    ----------
    xml_file : str or Path
        Path to the ``.xml`` metadata file.

    Returns
    -------
    dict
        Keys: ``nchan``, ``nrow``, ``ncol``, ``fs``, ``dtype``, ``anat_map``.

    Raises
    ------
    ValueError
        If electrode shanks have unequal heights (non-rectangular grid).
    """
    xml_dict = parse_xml(xml_file)

    # Acquisition system information
    acq = read_acquistionSystem(xml_dict)
    fs = float(acq["samplingRate"])
    nbits = acq["nBits"]

    # Process 'nbits' to ensure it's an integer
    if "float" in nbits and not isinstance(nbits, int):
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
    """Return the data-type string (e.g. ``'int16'``) from a parsed XML dict."""
    acq = read_acquistionSystem(xml_dict)
    return "int{nbits}".format(nbits=acq["nBits"])


def read_acquistionSystem(xml_dict):
    """Return the ``acquisitionSystem`` subtree from a parsed XML dict."""
    return xml_dict["parameters"]["acquisitionSystem"]
