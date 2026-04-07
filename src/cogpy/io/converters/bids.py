from __future__ import annotations

from pathlib import Path


def bids_lfp_to_zarr(lfp_path: str, zarr_path: str) -> None:
    """Convert a BIDS IEEG `.lfp` (with JSON/TSV sidecars) to a Zarr dataset.

    Parameters
    ----------
    lfp_path:
        Path to the BIDS IEEG LFP file (e.g., `*_ieeg.lfp`).
        Sidecars (`*_ieeg.json`, `*_channels.tsv`, `*_electrodes.tsv`) are expected
        next to the file as per `cogpy.io.ieeg_io`.
    zarr_path:
        Output Zarr directory path.
    """
    from cogpy.io import ecog_io, ieeg_io

    sigx = ieeg_io.from_file(lfp_path, grid=True, as_float=True)
    sigx.name = "sigx"

    dst = Path(zarr_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ecog_io.to_zarr(str(dst), sigx)


def zarr_to_dat(zarr_path: str, dat_path: str) -> None:
    """Convert a Zarr dataset containing `sigx` back to a flat binary signal.

    This mirrors the Snakemake `zarr2dat` rule behavior:
    - Load `sigx` from Zarr with shape (AP, ML, time)
    - Transpose to (time, ML, AP)
    - Flatten to (time, channels)
    - Write to a `.lfp` (or `.dat`) binary using `ecog_io.save_dat`
    """
    import numpy as np

    from cogpy.io import ecog_io

    sigx = ecog_io.from_zarr(zarr_path)["sigx"]  # AP, ML, time
    arr = sigx.data.transpose(2, 1, 0)  # time, ML, AP
    arr_flat = arr.reshape(arr.shape[0], -1)  # time, channels

    dst = Path(dat_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ecog_io.save_dat(arr_flat.compute(), str(dst), extension=dst.suffix, dtype=np.int16)
