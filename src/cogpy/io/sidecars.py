"""BIDS-style sidecar management: JSON metadata, channel/electrode TSVs, and symlink propagation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def read_json_metadata(
    path: str | Path, *, required_keys: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Read a JSON sidecar file and optionally validate required keys.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.
    required_keys : tuple of str, optional
        Keys that must be present; raises ``KeyError`` if missing.

    Returns
    -------
    dict
        Parsed JSON contents.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing JSON sidecar: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON sidecar {p}: {e}") from e
    for key in required_keys:
        if key not in meta:
            raise KeyError(f"{p} missing required key '{key}'")
    return meta


def write_json_metadata(path: str | Path, meta: dict[str, Any]) -> None:
    """Write a dict as a formatted JSON sidecar file."""
    p = Path(path)
    p.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def update_sampling_frequency_json(
    meta: dict[str, Any],
    out_json: str | Path,
    *,
    sampling_frequency_hz: float,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write updated JSON sidecar with new SamplingFrequency and optional extra fields."""
    meta_out = dict(meta)
    meta_out["SamplingFrequency"] = float(sampling_frequency_hz)
    if extra_fields:
        meta_out.update(extra_fields)
    write_json_metadata(out_json, meta_out)
    return meta_out


def _strip_datatype_suffix(stem: str) -> str:
    for dtype in ("ieeg", "ecephys"):
        suffix = f"_{dtype}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def sidecar_json(lfp_path: str | Path) -> Path:
    """Return the ``.json`` sidecar path for a given LFP file."""
    return Path(lfp_path).with_suffix(".json")


def sidecar_channels(lfp_path: str | Path) -> Path:
    """Return the ``_channels.tsv`` sidecar path for a given LFP file."""
    p = Path(lfp_path)
    stem = _strip_datatype_suffix(p.stem)
    return p.with_name(f"{stem}_channels.tsv")


def sidecar_electrodes(lfp_path: str | Path) -> Path:
    """Return the ``_electrodes.tsv`` sidecar path for a given LFP file."""
    p = Path(lfp_path)
    stem = _strip_datatype_suffix(p.stem)
    return p.with_name(f"{stem}_electrodes.tsv")


def sidecar_xml(lfp_path: str | Path) -> Path:
    """Return the ``.xml`` sidecar path for a given LFP file."""
    return Path(lfp_path).with_suffix(".xml")


def propagate_sidecars(
    input_lfp: str | Path,
    output_lfp: str | Path,
    *,
    overwrite: bool = False,
    extra: Iterable[Path] | None = None,
) -> None:
    """Symlink sidecars next to output_lfp when they exist for input_lfp."""
    in_lfp = Path(input_lfp)
    out_lfp = Path(output_lfp)
    out_lfp.parent.mkdir(parents=True, exist_ok=True)

    candidates = [
        sidecar_json(in_lfp),
        sidecar_channels(in_lfp),
        sidecar_electrodes(in_lfp),
        sidecar_xml(in_lfp),
    ]
    if extra:
        candidates.extend(Path(p) for p in extra)

    for src in candidates:
        if not src.exists():
            continue
        dst = out_lfp.with_name(src.name)
        if dst.exists() or dst.is_symlink():
            if overwrite:
                dst.unlink()
            else:
                continue
        os.symlink(os.path.realpath(src), str(dst))
