from __future__ import annotations

from typing import Iterable, Literal

import xarray as xr

__all__ = [
    "grid_shape",
    "to_apml_view",
    "flat_index_from_apml",
    "apml_from_flat_index",
    "flat_indices_from_selected",
    "flat_index_from_apml_order",
    "apml_from_flat_index_order",
]

FlatOrder = Literal["row-major", "col-major"]


def grid_shape(sig: xr.DataArray) -> tuple[int, int]:
    """
    Return (n_ap, n_ml) for a grid-shaped signal with dims including AP and ML.
    """
    if "AP" not in sig.dims or "ML" not in sig.dims:
        raise ValueError(f"sig must have dims including 'AP' and 'ML'; got dims={tuple(sig.dims)}")
    return int(sig.sizes["AP"]), int(sig.sizes["ML"])


def to_apml_view(sig: xr.DataArray) -> xr.DataArray:
    """
    Return a view with canonical grid axes ordered as ("time","AP","ML").

    This is used to enforce consistent row-major flattening with:
        flat = ap * n_ml + ml
        stack(channel=("AP","ML"))
    """
    if "time" not in sig.dims or "AP" not in sig.dims or "ML" not in sig.dims:
        raise ValueError(f"sig must include dims ('time','AP','ML'); got dims={tuple(sig.dims)}")
    return sig.transpose("time", "AP", "ML")


def flat_index_from_apml(ap: int, ml: int, n_ml: int) -> int:
    return int(ap) * int(n_ml) + int(ml)


def apml_from_flat_index(ix: int, n_ml: int) -> tuple[int, int]:
    ix_i = int(ix)
    n_ml_i = int(n_ml)
    if n_ml_i <= 0:
        raise ValueError("n_ml must be > 0")
    ap, ml = divmod(ix_i, n_ml_i)
    return int(ap), int(ml)

def flat_index_from_apml_order(
    ap: int,
    ml: int,
    *,
    n_ap: int,
    n_ml: int,
    order: FlatOrder = "row-major",
) -> int:
    """
    Convert (ap, ml) to a flat channel index under a specified convention.

    - row-major: flat = ap * n_ml + ml  (AP changes slowest in the grid plane)
    - col-major: flat = ml * n_ap + ap  (MATLAB/NeuroScope-style column-major)
    """
    ap_i = int(ap)
    ml_i = int(ml)
    n_ap_i = int(n_ap)
    n_ml_i = int(n_ml)
    if order == "row-major":
        return ap_i * n_ml_i + ml_i
    if order == "col-major":
        return ml_i * n_ap_i + ap_i
    raise ValueError(f"order must be 'row-major' or 'col-major', got {order!r}")


def apml_from_flat_index_order(
    ix: int,
    *,
    n_ap: int,
    n_ml: int,
    order: FlatOrder = "row-major",
) -> tuple[int, int]:
    """
    Inverse mapping for `flat_index_from_apml_order`.
    """
    ix_i = int(ix)
    n_ap_i = int(n_ap)
    n_ml_i = int(n_ml)
    if n_ap_i <= 0 or n_ml_i <= 0:
        raise ValueError("n_ap and n_ml must be > 0")

    if order == "row-major":
        ap, ml = divmod(ix_i, n_ml_i)
        return int(ap), int(ml)
    if order == "col-major":
        ml, ap = divmod(ix_i, n_ap_i)
        return int(ap), int(ml)
    raise ValueError(f"order must be 'row-major' or 'col-major', got {order!r}")


def flat_indices_from_selected(
    selected: Iterable[tuple[int, int]],
    *,
    n_ap: int,
    n_ml: int,
    order: FlatOrder = "row-major",
    n_ch: int | None = None,
) -> list[int]:
    """
    Convert a set/iterable of (ap, ml) pairs to sorted row-major flat indices.
    Optionally clamps indices to < n_ch.
    """
    out: list[int] = []
    for ap, ml in selected:
        ix = flat_index_from_apml_order(int(ap), int(ml), n_ap=int(n_ap), n_ml=int(n_ml), order=order)
        if n_ch is None or ix < int(n_ch):
            out.append(ix)
    return sorted(set(out))
