"""
View specification for declarative view creation.

`ViewSpec` describes *what* to display without specifying *how* to render it.
`ViewFactory` converts a `ViewSpec` into reactive HoloViews objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["ViewSpec"]


@dataclass
class ViewSpec:
    """
    Declarative view specification.

    Parameters
    ----------
    kdims
        Dimensions to display (e.g., ["ML", "AP"] or ["time"]).
    controls
        Dimensions controlled by widgets/streams (e.g., ["time"] or ["AP", "ML"]).
    iterate
        Optional iteration dims (e.g., ["signal"]) used by higher-level layout code.
    signal_id
        Which signal to visualize (None = active signal).
    view_type
        HoloViews element type: "auto", "Image", "Curve", ...
    colormap
        Colormap for image-like views.
    title
        Optional title override.
    operation
        Optional post-processing function applied to numeric values.
    clim
        Optional color limits (vmin, vmax) for image-like views.
    symmetric_clim
        If True and `clim` is not provided, uses +/- max(|x|).
    """

    kdims: list[str]
    controls: list[str]
    iterate: list[str] = field(default_factory=list)
    signal_id: str | None = None

    view_type: str = "auto"
    colormap: str = "viridis"
    title: str | None = None

    operation: Callable[[Any], Any] | None = None
    clim: tuple[float, float] | None = None
    symmetric_clim: bool = False
    fixed_values: dict[str, object] = field(default_factory=dict)
    coord_spaces: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.kdims, list):
            self.kdims = list(self.kdims)
        if not isinstance(self.controls, list):
            self.controls = list(self.controls)
        if not isinstance(self.iterate, list):
            self.iterate = list(self.iterate) if self.iterate else []
        if not isinstance(self.coord_spaces, list):
            self.coord_spaces = list(self.coord_spaces) if self.coord_spaces else []
        if self.fixed_values is None:
            self.fixed_values = {}
        if not isinstance(self.fixed_values, dict):
            self.fixed_values = dict(self.fixed_values)

        overlap = (set(self.kdims) & set(self.controls)) | (set(self.kdims) & set(self.iterate)) | (
            set(self.controls) & set(self.iterate)
        )
        if overlap:
            raise ValueError(
                "Dimensions cannot appear in multiple categories "
                f"(overlap={sorted(overlap)!r}; kdims={self.kdims!r}, controls={self.controls!r}, "
                f"iterate={self.iterate!r})"
            )

        fixed_dims = {str(k) for k in self.fixed_values.keys()}
        conflict = fixed_dims & (set(self.kdims) | set(self.controls) | set(self.iterate))
        if conflict:
            raise ValueError(
                "Fixed dimensions cannot also be in kdims/controls/iterate "
                f"(conflict={sorted(conflict)!r})"
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a JSON-friendly dict.

        Notes
        -----
        `operation` is intentionally not serialized.
        """
        return {
            "kdims": list(self.kdims),
            "controls": list(self.controls),
            "iterate": list(self.iterate),
            "signal_id": self.signal_id,
            "view_type": self.view_type,
            "colormap": self.colormap,
            "title": self.title,
            "clim": self.clim,
            "symmetric_clim": self.symmetric_clim,
            "fixed_values": dict(self.fixed_values),
            "coord_spaces": list(self.coord_spaces),
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ViewSpec":
        """Deserialize from a dict (inverse of `to_dict`)."""
        return cls(
            kdims=list(config["kdims"]),
            controls=list(config["controls"]),
            iterate=list(config.get("iterate") or []),
            signal_id=config.get("signal_id"),
            view_type=str(config.get("view_type") or "auto"),
            colormap=str(config.get("colormap") or "viridis"),
            title=config.get("title"),
            clim=tuple(config["clim"]) if config.get("clim") is not None else None,
            symmetric_clim=bool(config.get("symmetric_clim") or False),
            fixed_values=dict(config.get("fixed_values") or {}),
            coord_spaces=list(config.get("coord_spaces") or []),
        )

