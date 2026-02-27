from __future__ import annotations

import importlib


def require(module: str, *, extra: str, pip_name: str | None = None):
    """
    Import an optional dependency with a helpful error message.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        name = pip_name or module
        raise ImportError(
            f"Missing optional dependency {module!r}. Install with `pip install cogpy[{extra}]` "
            f"(which will install {name!r})."
        ) from e

