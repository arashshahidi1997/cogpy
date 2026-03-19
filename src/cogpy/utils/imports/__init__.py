"""Optional dependency helpers.

Provides :func:`import_optional` which gives a clear error message when an
optional dependency is missing, directing the user to the right ``pip install
cogpy[extra]`` command.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["import_optional"]

# Map package import names to the cogpy extra that provides them.
_EXTRAS: dict[str, str] = {
    "bokeh": "viz",
    "datashader": "viz",
    "holoviews": "viz",
    "hvplot": "viz",
    "ipympl": "viz",
    "matplotlib": "viz",
    "param": "viz",
    "panel": "viz",
    "plotly": "viz",
    "seaborn": "viz",
    "ipykernel": "notebook",
    "ipywidgets": "notebook",
    "h5py": "io",
    "tables": "io",
    "zarr": "io",
    "openpyxl": "io",
    "xmltodict": "io",
    "dask": "perf",
    "pyfftw": "perf",
    "ghostipy": "signal",
    "skimage": "signal",
    "emd": "signal",
    "mne": "interop-mne",
}


def import_optional(name: str, *, extra: str | None = None) -> ModuleType:
    """Import an optional dependency, raising a helpful error if missing.

    Parameters
    ----------
    name : str
        Dotted module name, e.g. ``"holoviews"`` or ``"dask.array"``.
    extra : str, optional
        Override the cogpy extra name.  Inferred from *name* if omitted.

    Returns
    -------
    module
        The imported module.

    Raises
    ------
    ImportError
        With an install hint like ``pip install cogpy[viz]``.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        top = name.split(".")[0]
        ext = extra or _EXTRAS.get(top, "")
        if ext:
            hint = f'pip install "cogpy[{ext}]"'
        else:
            hint = f"pip install {top}"
        raise ImportError(
            f"{name!r} is required but not installed. Install it with:\n\n"
            f"    {hint}\n"
        ) from None
