"""HoloViews helpers for xarray objects.

This module contains small utilities that make it easier to create quick
"viewer-like" HoloViews objects from xarray tensors.

The functions are intentionally lightweight and avoid any repo-specific
assumptions (no pipeline context, no derivative paths).
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def normalize_coords_to_index(sigx: xr.DataArray, coos: list[str] | tuple[str, ...] = ("AP", "ML")) -> xr.DataArray:
    """
    Replace selected coordinate values with 0..N-1 indices.

    This is useful when coordinates are non-uniform/unsorted (or when you just
    want a consistent grid indexing for quick visualization).
    """
    out = sigx
    for coo in coos:
        if coo in out.dims:
            out = out.assign_coords(**{coo: np.arange(out.sizes[coo])})
    return out


def grid_movie(
    sigx: xr.DataArray,
    *,
    x_dim: str = "ML",
    y_dim: str = "AP",
    normalize_index: bool = True,
    title: str | None = None,
    width: int = 400,
    height: int = 350,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    colorbar: bool = True,
):
    """
    Create a dynamic HoloViews Image "movie" over remaining dimensions.

    Parameters
    ----------
    sigx
        Input DataArray with at least ``(x_dim, y_dim)`` dimensions and one or
        more additional dimensions (e.g., ``time``) to animate over.
    x_dim, y_dim
        Spatial dimensions to plot as an image.
    normalize_index
        If True, replaces ``x_dim`` and ``y_dim`` coordinate values with
        integer indices 0..N-1 before plotting.
    title
        Plot title. Defaults to ``sigx.name``.

    Returns
    -------
    hv.DynamicMap
        A dynamic image that can be scrubbed/animated over remaining dims.
    """
    import holoviews as hv
    from holoviews import opts

    if x_dim not in sigx.dims or y_dim not in sigx.dims:
        raise ValueError(f"sigx must have dims {x_dim!r} and {y_dim!r}, got {list(sigx.dims)}")

    data = normalize_coords_to_index(sigx, (y_dim, x_dim)) if normalize_index else sigx

    hm = hv.Dataset(data).to(hv.Image, [x_dim, y_dim], dynamic=True)
    hm = hm.opts(
        opts.Image(
            cmap=cmap,
            symmetric=bool(symmetric),
            colorbar=bool(colorbar),
            width=int(width),
            height=int(height),
            title=(data.name if title is None else str(title)),
        )
    )
    return hm

def add_time_hair(
    obj,
    *,
    time_kdim: str = "time",
    t=None,
    snap: bool = True,
    return_controller: bool = False,
    **attach_kwargs,
):
    """
    Add a clickable "time hair" (vertical line) to a HoloViews Curve or Layout.

    This is a small umbrella wrapper around :func:`cogpy.core.plot.time_player.add_time_hair`
    so plotting helpers can be imported from a single place.

    Examples
    --------
    >>> import holoviews as hv
    >>> hv.extension("bokeh")  # doctest: +SKIP
    >>> import numpy as np
    >>> from cogpy.core.plot.xarray_hv import add_time_hair
    >>> curve = hv.Curve((np.linspace(0, 1, 10), np.random.randn(10)), kdims=["time"], vdims=["y"])
    >>> view, hair = add_time_hair(curve, time_kdim="time", return_controller=True)
    >>> hair.param.watch(lambda e: print("t:", e.new), "t")  # doctest: +SKIP
    >>> view  # doctest: +SKIP
    """
    from .time_player import add_time_hair as _add_time_hair

    return _add_time_hair(
        obj,
        time_kdim=time_kdim,
        t=t,
        snap=snap,
        return_controller=return_controller,
        **attach_kwargs,
    )


def grid_movie_with_time_curve(
    sigx: xr.DataArray,
    *,
    time_dim: str = "time",
    x_dim: str = "ML",
    y_dim: str = "AP",
    indexers: dict[str, object] | None = None,
    normalize_index: bool = True,
    title: str | None = None,
    width: int = 400,
    height: int = 350,
    curve_height: int = 120,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    colorbar: bool = True,
    snap: bool = True,
    return_controller: bool = False,
):
    """
    Common viewer layout: spatial "grid movie" + time curve with a linked time hair.

    The top panel is a 2D image at the current time ``t``; the bottom panel is a
    1D summary curve over time with a clickable vertical hair. Clicking the
    curve moves the hair and updates the image.

    Parameters
    ----------
    sigx
        Input DataArray with at least ``(x_dim, y_dim, time_dim)``. Any other
        dimensions are fixed using ``indexers`` (or the first index if not
        provided).
    indexers
        Optional mapping for additional dims (e.g. ``{"freq": 10.0}``).
    return_controller
        If True, returns ``(layout, controller)`` where ``controller.t`` is the
        selected time.

    Examples
    --------
    >>> import holoviews as hv
    >>> hv.extension("bokeh")  # doctest: +SKIP
    >>> import numpy as np
    >>> import xarray as xr
    >>> from cogpy.core.plot.xarray_hv import grid_movie_with_time_curve
    >>> da = xr.DataArray(
    ...     np.random.randn(20, 10, 12),
    ...     dims=["time", "AP", "ML"],
    ...     coords={"time": np.linspace(0, 1, 20), "AP": np.arange(10), "ML": np.arange(12)},
    ...     name="val",
    ... )
    >>> view, ctrl = grid_movie_with_time_curve(da, time_dim="time", y_dim="AP", x_dim="ML", return_controller=True)
    >>> ctrl.param.watch(lambda e: print("t:", e.new), "t")  # doctest: +SKIP
    >>> view  # doctest: +SKIP
    """
    import holoviews as hv
    import numpy as np
    from holoviews import streams
    from holoviews import opts

    if x_dim not in sigx.dims or y_dim not in sigx.dims or time_dim not in sigx.dims:
        raise ValueError(
            f"sigx must have dims {x_dim!r}, {y_dim!r}, {time_dim!r}, got {list(sigx.dims)}"
        )

    data = normalize_coords_to_index(sigx, (y_dim, x_dim)) if normalize_index else sigx
    indexers = {} if indexers is None else dict(indexers)

    # Fix any extra dims beyond (x, y, time).
    fixed = data
    extra_dims = [d for d in fixed.dims if d not in (x_dim, y_dim, time_dim)]
    if extra_dims:
        isel = {d: 0 for d in extra_dims if d not in indexers}
        if isel:
            fixed = fixed.isel(**isel)
        for d in extra_dims:
            if d in indexers:
                try:
                    fixed = fixed.sel({d: indexers[d]}, method="nearest")
                except Exception:
                    fixed = fixed.sel({d: indexers[d]})

    # 1D summary curve used for the time hair interaction (mean over x/y).
    curve_da = fixed.mean(dim=(y_dim, x_dim))
    curve = hv.Curve(curve_da, kdims=[time_dim]).opts(
        width=int(width),
        height=int(curve_height),
        framewise=False,
        tools=[],
        default_tools=[],
    )

    curve_with_hair, controller = add_time_hair(
        curve, time_kdim=time_dim, t=None, snap=bool(snap), return_controller=True
    )

    # Default t if unset (helps keep the image from rendering an empty placeholder).
    if controller.t is None:
        t_vals = np.asarray(fixed[time_dim].values)
        if t_vals.size:
            controller.t = t_vals[0].item() if hasattr(t_vals[0], "item") else t_vals[0]

    t_stream = streams.Params(controller, ["t"])

    def _image_frame(t=None, **_):
        if t is None:
            return hv.Image([]).opts(alpha=0.0)
        frame = fixed.sel({time_dim: t}, method="nearest")
        img = hv.Image(frame, kdims=[x_dim, y_dim], vdims=[frame.name or "val"])
        return img.opts(
            opts.Image(
                cmap=cmap,
                symmetric=bool(symmetric),
                colorbar=bool(colorbar),
                width=int(width),
                height=int(height),
                title=(fixed.name if title is None else str(title)),
                tools=["tap", "pan", "wheel_zoom", "box_zoom", "reset"],
                active_tools=["tap"],
            )
        )

    img_dm = hv.DynamicMap(_image_frame, streams=[t_stream])

    return (img_dm + curve_with_hair, controller) if bool(return_controller) else (img_dm + curve_with_hair)
