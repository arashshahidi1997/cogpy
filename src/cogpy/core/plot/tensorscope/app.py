"""
TensorScope application shell.

TensorScopeApp is the composition root that wires:
- TensorScopeState (authoritative model)
- LayerManager / layers (views)
- Layout manager (Panel template + persistence)

Phase 0: Stub only.
Phase 3+: Full implementation of the application shell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from .state import TensorScopeState

if TYPE_CHECKING:
    import panel as pn
    import xarray as xr


@dataclass(frozen=True, slots=True)
class TensorScopeConfig:
    """
    Configuration container for TensorScopeApp.

    Phase 0: placeholder for future configuration options (themes, layer presets,
    performance budgets, etc.).
    """

    title: str = "TensorScope"


class TensorScopeApp:
    """
    TensorScope application shell (composition root).

    Parameters
    ----------
    data
        Primary dataset for the session.
    config
        Optional configuration for the app.
    """

    def __init__(self, data: "xr.DataArray", *, config: TensorScopeConfig | None = None) -> None:
        self.config = config or TensorScopeConfig()
        self.state = TensorScopeState(data)

    def build(self) -> "pn.viewable.Viewable":
        """
        Build and return the Panel view for this app.

        Phase 0: not implemented.
        """

        raise NotImplementedError("Phase 3 implementation pending")

    def dispose(self) -> None:
        """
        Dispose of any resources held by layers (Bokeh documents, streams, etc.).

        Phase 0: no-op.
        """

        return None

