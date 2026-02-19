import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path
from .sidecars import sidecar_json, sidecar_channels, sidecar_electrodes, read_json_metadata
from dataclasses import dataclass

def read_ieeg_json(lfp_path: Path) -> Dict[str, Any]:
    """Reads the mandatory JSON sidecar."""
    return read_json_metadata(
        sidecar_json(lfp_path), 
        required_keys=("SamplingFrequency", "ChannelCount")
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

def read_ieeg_electrodes(meta_source_path: Path, nrow: int = 0, ncol: int = 0) -> Dict[str, Any]:
    path = sidecar_electrodes(meta_source_path)
    # TOLERANCE: Default values if file is missing
    data = {"rows": None, "cols": None, "ap": None, "ml": None}
    
    if not path.exists():
        return data

    df = pd.read_csv(path, sep="\t")
    
    # TOLERANCE: Only attempt spatial mapping if specific columns exist
    if {"row", "col"}.issubset(df.columns):
        data["rows"] = df["row"].to_numpy(int)
        data["cols"] = df["col"].to_numpy(int)
        
        if "AP" in df.columns and nrow > 0:
            data["ap"] = df.groupby("row")["AP"].mean().reindex(range(nrow)).to_numpy()
        if "ML" in df.columns and ncol > 0:
            data["ml"] = df.groupby("col")["ML"].mean().reindex(range(ncol)).to_numpy()
            
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

    def __repr__(self):
        return f"IEEGMetadata(fs={self.fs}Hz, nch={self.nch}, grid={self.nrow}x{self.ncol})"

def load_ieeg_metadata(lfp_path: str | Path) -> IEEGMetadata:
    """
    Orchestrator: Coordinates the reading of all available BIDS sidecars.
    """
    lfp_path = Path(lfp_path)
    
    # 1. JSON is the primary source
    meta_json = read_ieeg_json(lfp_path)
    
    # 2. Extract types and counts
    nrow = meta_json.get("RowCount", 0)
    ncol = meta_json.get("ColumnCount", 0)
    
    # 3. Call sub-readers
    dtype = read_ieeg_channels(lfp_path)
    elec_data = read_ieeg_electrodes(lfp_path, nrow=nrow, ncol=ncol)
    
    return IEEGMetadata(
        fs=meta_json["SamplingFrequency"],
        nch=meta_json["ChannelCount"],
        dtype=dtype,
        nrow=nrow if nrow > 0 else None,
        ncol=ncol if ncol > 0 else None,
        rows=elec_data["rows"],
        cols=elec_data["cols"],
        ap_coords=elec_data["ap"],
        ml_coords=elec_data["ml"]
    )


def load_ieeg_metadata(meta_source: str | Path, default_dtype: str = "int16") -> IEEGMetadata:
    """
    Orchestrator: Resolves sidecar paths.
    Tolerates missing channels/electrodes by using defaults.
    """
    p = Path(meta_source)
    json_path = p if p.suffix == ".json" else sidecar_json(p)
    
    meta_json = read_json_metadata(json_path, required_keys=("SamplingFrequency", "ChannelCount"))
    
    nrow = meta_json.get("RowCount", 0)
    ncol = meta_json.get("ColumnCount", 0)
    
    # Try to get dtype from channels.tsv, fallback to default_dtype if None
    dtype = read_ieeg_channels(json_path) or np.dtype(default_dtype)
    
    elec_data = read_ieeg_electrodes(json_path, nrow=nrow, ncol=ncol)
    
    return IEEGMetadata(
        fs=meta_json["SamplingFrequency"],
        nch=meta_json["ChannelCount"],
        dtype=dtype,
        nrow=nrow if nrow > 0 else None,
        ncol=ncol if ncol > 0 else None,
        rows=elec_data["rows"],
        cols=elec_data["cols"],
        ap_coords=elec_data["ap"],
        ml_coords=elec_data["ml"]
    )