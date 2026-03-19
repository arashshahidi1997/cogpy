from __future__ import annotations

import numpy as np
from bokeh.models import Toolbar
from bokeh.palettes import Plasma256, RdBu11, Viridis256
from bokeh.plotting import figure

__all__ = [
    # Backgrounds
    "BG",
    "BG_PANEL",
    "BORDER",
    # Text
    "TEXT",
    "TEXT_SMALL",
    "TEXT_MED",
    # Accent
    "BLUE",
    "TEAL",
    # Palettes / colormaps
    "PALETTE",
    "COLORMAPS",
    # Helpers
    "style_figure",
]

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# Backgrounds
BG = "#181825"  # figure background
BG_PANEL = "#1e1e2e"  # panel/widget background
BORDER = "#3a3a5c"  # axis lines, cell borders

# Text
TEXT = "#cdd6f4"
TEXT_SMALL = "9px"
TEXT_MED = "11px"

# Accent
BLUE = "#4a90d9"
TEAL = "#4fc3f7"

# Electrode / trace palette (20 colours, used by MultichannelViewer + ChannelGridWidget)
PALETTE = [
    "#4fc3f7",
    "#81c784",
    "#ffb74d",
    "#e57373",
    "#ba68c8",
    "#4dd0e1",
    "#aed581",
    "#ffd54f",
    "#ff8a65",
    "#f06292",
    "#4db6ac",
    "#dce775",
    "#a1887f",
    "#90a4ae",
    "#fff176",
    "#ce93d8",
    "#80cbc4",
    "#ef9a9a",
    "#80deea",
    "#c5e1a5",
]


def _hex_to_rgb01(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def _rgb01_to_hex(rgb: np.ndarray) -> str:
    r, g, b = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(int).tolist()
    return f"#{r:02x}{g:02x}{b:02x}"


def _interp_palette(base: list[str], n: int) -> list[str]:
    """
    Linearly interpolate a hex palette to length n.

    Bokeh provides Viridis256/Plasma256 but not RdBu256 in some versions.
    """
    if n <= 1:
        return [base[0]]
    m = len(base)
    if m < 2:
        return [base[0]] * n

    base_rgb = np.stack([_hex_to_rgb01(c) for c in base], axis=0)  # (m,3)
    x = np.linspace(0.0, 1.0, m)
    xi = np.linspace(0.0, 1.0, n)
    out = np.empty((n, 3), dtype=float)
    for k in range(3):
        out[:, k] = np.interp(xi, x, base_rgb[:, k])
    return [_rgb01_to_hex(out[i]) for i in range(n)]


def _jet_palette(n: int) -> list[str]:
    """
    Approximate MATLAB-style "jet" colormap (blue → cyan → yellow → red).

    Implemented analytically to avoid matplotlib as a dependency.
    """
    n = int(n)
    if n <= 1:
        return ["#00007f"]
    x = np.linspace(0.0, 1.0, n)

    # Standard jet approximation.
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)

    rgb = np.stack([r, g, b], axis=1)
    return [_rgb01_to_hex(rgb[i]) for i in range(n)]


_RDBU256 = _interp_palette(list(RdBu11), 256)
_JET256 = _jet_palette(256)

# Colormaps (Bokeh palette lists)
COLORMAPS: dict[str, list[str]] = {
    "viridis": list(Viridis256),
    "plasma": list(Plasma256),
    "rdbu": list(reversed(_RDBU256)),  # red=high, blue=low
    "rdbu_r": _RDBU256,  # blue=high, red=low
    "jet": _JET256,
}


# ---------------------------------------------------------------------
# Styling helper
# ---------------------------------------------------------------------


def style_figure(fig: figure, *, xlabel: str = "", ylabel: str = "", toolbar: bool = False) -> figure:
    """
    Apply the cogpy dark theme to a Bokeh figure in-place. Returns fig.

    Sets: background, border, grid visibility, axis line/tick/label colours,
    axis label text, toolbar location.
    """
    fig.background_fill_color = BG
    fig.border_fill_color = BG
    fig.grid.visible = False

    fig.axis.axis_line_color = BORDER
    fig.axis.major_tick_line_color = BORDER
    fig.axis.minor_tick_line_color = None
    fig.axis.major_label_text_color = TEXT
    fig.axis.major_label_text_font_size = TEXT_SMALL
    fig.axis.axis_label_text_color = TEXT

    if xlabel is not None:
        fig.xaxis.axis_label = str(xlabel)
    if ylabel is not None:
        fig.yaxis.axis_label = str(ylabel)

    if toolbar:
        # If tools are set, keep toolbar above by default.
        fig.toolbar_location = "above"
        if isinstance(fig.toolbar, Toolbar):
            fig.toolbar.logo = None
    else:
        fig.toolbar_location = None

    return fig
