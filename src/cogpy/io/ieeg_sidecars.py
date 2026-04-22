"""BIDS iEEG sidecar readers — JSON metadata, channel TSVs, electrode TSVs.

.. note:: **Lab-internal module.** Assumes the Bhatt Lab BIDS-iEEG
   sidecar conventions.  Not part of the stable public API.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path
from .sidecars import (
    sidecar_json,
    sidecar_channels,
    sidecar_electrodes,
    read_json_metadata,
    resolve_channel_count,
)
from dataclasses import dataclass


def read_ieeg_json(lfp_path: Path) -> Dict[str, Any]:
    """Reads the mandatory JSON sidecar."""
    return read_json_metadata(
        sidecar_json(lfp_path), required_keys=("SamplingFrequency",)
    )


def read_ieeg_channels(meta_source_path: Path) -> Optional[np.dtype]:
    path = sidecar_channels(meta_source_path)
    # TOLERANCE: Check existence before reading
    if not path.exists():
        return None

    df = pd.read_csv(path, sep="\t")
    # TOLERANCE: Check if column exists within the file
    if "dtype" in df.columns:
        valid = df["dtype"].dropna()
        if not valid.empty:
            return np.dtype(valid.iloc[0])
    return None


def read_ieeg_electrodes(
    meta_source_path: Path, nrow: int = 0, ncol: int = 0
) -> Dict[str, Any]:
    """Read BIDS _electrodes.tsv and return per-channel + grid metadata.

    If ``nrow`` / ``ncol`` are not supplied (or are 0) but the file has
    ``row`` / ``col`` columns, grid dimensions are inferred from the data
    as ``max(row) + 1`` / ``max(col) + 1``. Inferred values are returned
    under ``"nrow"`` / ``"ncol"`` so callers can use them when the JSON
    sidecar lacks ``RowCount`` / ``ColumnCount`` (those keys are lab
    extensions, not BIDS-standard).
    """
    path = sidecar_electrodes(meta_source_path)
    # TOLERANCE: Default values if file is missing
    data = {
        "rows": None,
        "cols": None,
        "ap": None,
        "ml": None,
        "x": None,
        "y": None,
        "nrow": nrow if nrow > 0 else None,
        "ncol": ncol if ncol > 0 else None,
    }

    if not path.exists():
        return data

    df = pd.read_csv(path, sep="\t")

    # Per-channel (x, y) coordinates — BIDS-iEEG standard columns.
    if "x" in df.columns:
        data["x"] = df["x"].to_numpy(float)
    if "y" in df.columns:
        data["y"] = df["y"].to_numpy(float)

    # TOLERANCE: Only attempt grid-row/col mapping if specific columns exist
    if {"row", "col"}.issubset(df.columns):
        rows = df["row"].to_numpy(int)
        cols = df["col"].to_numpy(int)
        data["rows"] = rows
        data["cols"] = cols

        # Infer grid dims from data if caller didn't supply them.
        nrow_eff = nrow if nrow > 0 else int(rows.max() + 1)
        ncol_eff = ncol if ncol > 0 else int(cols.max() + 1)
        data["nrow"] = nrow_eff
        data["ncol"] = ncol_eff

        if "AP" in df.columns:
            data["ap"] = (
                df.groupby("row")["AP"].mean().reindex(range(nrow_eff)).to_numpy()
            )
        if "ML" in df.columns:
            data["ml"] = (
                df.groupby("col")["ML"].mean().reindex(range(ncol_eff)).to_numpy()
            )

    return data


@dataclass(frozen=True)
class IEEGMetadata:
    fs: float
    nch: int
    dtype: Optional[np.dtype]
    nrow: Optional[int] = None
    ncol: Optional[int] = None
    rows: Optional[np.ndarray] = None
    cols: Optional[np.ndarray] = None
    ap_coords: Optional[np.ndarray] = None
    ml_coords: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None

    def __repr__(self):
        return f"IEEGMetadata(fs={self.fs}Hz, nch={self.nch}, grid={self.nrow}x{self.ncol})"


def load_ieeg_metadata(lfp_path: str | Path) -> IEEGMetadata:
    """
    Orchestrator: Coordinates the reading of all available BIDS sidecars.
    """
    lfp_path = Path(lfp_path)

    # 1. JSON is the primary source
    meta_json = read_ieeg_json(lfp_path)

    # 2. Extract types and counts. RowCount/ColumnCount are lab-extension
    # keys (not BIDS-standard); fall back to inference from _electrodes.tsv
    # if missing.
    nrow = meta_json.get("RowCount", 0)
    ncol = meta_json.get("ColumnCount", 0)

    # 3. Call sub-readers
    dtype = read_ieeg_channels(lfp_path)
    elec_data = read_ieeg_electrodes(lfp_path, nrow=nrow, ncol=ncol)

    return IEEGMetadata(
        fs=meta_json["SamplingFrequency"],
        nch=resolve_channel_count(meta_json),
        dtype=dtype,
        nrow=elec_data["nrow"],
        ncol=elec_data["ncol"],
        rows=elec_data["rows"],
        cols=elec_data["cols"],
        ap_coords=elec_data["ap"],
        ml_coords=elec_data["ml"],
        x=elec_data["x"],
        y=elec_data["y"],
    )


def load_ieeg_metadata(
    meta_source: str | Path, default_dtype: str = "int16"
) -> IEEGMetadata:
    """
    Orchestrator: Resolves sidecar paths.
    Tolerates missing channels/electrodes by using defaults.
    """
    p = Path(meta_source)
    json_path = p if p.suffix == ".json" else sidecar_json(p)

    meta_json = read_json_metadata(
        json_path, required_keys=("SamplingFrequency",)
    )

    nrow = meta_json.get("RowCount", 0)
    ncol = meta_json.get("ColumnCount", 0)

    # Try to get dtype from channels.tsv, fallback to default_dtype if None
    dtype = read_ieeg_channels(json_path) or np.dtype(default_dtype)

    elec_data = read_ieeg_electrodes(json_path, nrow=nrow, ncol=ncol)

    return IEEGMetadata(
        fs=meta_json["SamplingFrequency"],
        nch=resolve_channel_count(meta_json),
        dtype=dtype,
        nrow=elec_data["nrow"],
        ncol=elec_data["ncol"],
        rows=elec_data["rows"],
        cols=elec_data["cols"],
        ap_coords=elec_data["ap"],
        ml_coords=elec_data["ml"],
        x=elec_data["x"],
        y=elec_data["y"],
    )
