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


def snap_ch_from_apml(
    ap: float,
    ml: float,
    *,
    ap_ch: np.ndarray,
    ml_ch: np.ndarray,
) -> int:
    """Return nearest channel index given an (AP, ML) point.

    Parameters
    ----------
    ap, ml
        Query point.
    ap_ch, ml_ch
        Per-channel coordinate arrays of shape ``(n_ch,)``.
    """
    ap = float(ap)
    ml = float(ml)
    ap_ch = np.asarray(ap_ch, dtype=float).reshape(-1)
    ml_ch = np.asarray(ml_ch, dtype=float).reshape(-1)
    if ap_ch.shape != ml_ch.shape:
        raise ValueError(f"ap_ch and ml_ch must have the same shape, got {ap_ch.shape} vs {ml_ch.shape}")
    if ap_ch.size == 0:
        raise ValueError("ap_ch/ml_ch are empty")
    d2 = (ap_ch - ap) ** 2 + (ml_ch - ml) ** 2
    return int(np.argmin(d2))


def channel_geometry_from_time_channel(
    sig_tc: xr.DataArray,
    *,
    ch_dim: str = "ch",
    ap_name: str = "AP",
    ml_name: str = "ML",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-channel (AP, ML) arrays from a standardized ``(time, ch)`` signal.

    Expects ``sig_tc`` to already have 1D coordinates ``AP(ch)`` and ``ML(ch)``,
    e.g. after ``to_time_channel(...)`` + ``reset_index("ch")`` in ECoG grid data.
    """
    if ch_dim not in sig_tc.dims:
        raise ValueError(f"Expected ch_dim={ch_dim!r} in sig_tc.dims={tuple(sig_tc.dims)}")
    if ap_name not in sig_tc.coords or ml_name not in sig_tc.coords:
        raise ValueError(f"Expected coords {ap_name!r} and {ml_name!r} on sig_tc")
    ap_ch = np.asarray(sig_tc.coords[ap_name].values, dtype=float).reshape(-1)
    ml_ch = np.asarray(sig_tc.coords[ml_name].values, dtype=float).reshape(-1)
    return ap_ch, ml_ch


def standardize_time_channel_with_geometry(
    sig: xr.DataArray,
    *,
    time_dim: str = "time",
    ch_dim: str = "ch",
    grid_dims: tuple[str, str] = ("AP", "ML"),
    normalize_index: bool = True,
    nsamples: int | None = None,
    materialize: bool = False,
) -> tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """Standardize a grid signal to ``(time, ch)`` and return per-channel geometry.

    This is a convenience wrapper for the common notebook pattern:
    - optional time slice
    - optional ``.compute()`` (materialize dask-backed arrays)
    - ``to_time_channel`` (stacking ``(AP, ML)`` into ``ch``)
    - ``reset_index('ch')`` to expose ``AP(ch)`` and ``ML(ch)`` coords
    - set a simple integer channel coordinate ``0..nch-1``

    Returns
    -------
    sig_tc : xr.DataArray
        Signal with dims ``(time, ch)`` and coords ``AP(ch)``, ``ML(ch)``.
    ap_ch, ml_ch : np.ndarray
        Per-channel AP/ML coordinate arrays, each shape ``(nch,)``.
    """
    if not isinstance(sig, xr.DataArray):
        raise TypeError("sig must be an xarray.DataArray")
    s = sig
    if nsamples is not None:
        if time_dim not in s.dims:
            raise ValueError(f"time_dim={time_dim!r} not in sig.dims={tuple(s.dims)}")
        s = s.isel({time_dim: slice(0, int(nsamples))})
    if materialize:
        try:
            s = s.compute()
        except Exception:
            s = s.load()

    s_tc = to_time_channel(
        s,
        time_dim=time_dim,
        ch_dim=ch_dim,
        grid_dims=grid_dims,
        normalize_index=normalize_index,
    )

    # If ch is a MultiIndex, expose its components as coords (AP(ch), ML(ch)).
    try:
        s_tc = s_tc.reset_index(ch_dim)
    except Exception:
        pass

    # Ensure ch is a simple 0..nch-1 coordinate.
    s_tc = s_tc.assign_coords({ch_dim: np.arange(s_tc.sizes[ch_dim])})

    ap_ch, ml_ch = channel_geometry_from_time_channel(
        s_tc, ch_dim=ch_dim, ap_name=grid_dims[0], ml_name=grid_dims[1]
    )
    return s_tc, ap_ch, ml_ch


def to_time_channel(
    sig: xr.DataArray,
    *,
    time_dim: str = "time",
    ch_dim: str = "ch",
    grid_dims: tuple[str, str] = ("AP", "ML"),
    normalize_index: bool = True,
) -> xr.DataArray:
    """
    Umbrella helper to standardize signals to ``(time, ch)``.

    Accepts either:
    - already-stacked: ``(time_dim, ch_dim)``
    - grid signal: ``(time_dim, *grid_dims)`` (default ``(time, AP, ML)``),
      which is stacked into a single ``ch`` dimension via
      ``sig.stack(ch=(AP, ML))``.

    Parameters
    ----------
    sig
        Input signal.
    time_dim, ch_dim
        Names of the time and channel dimensions in the output.
    grid_dims
        Dimension names to stack into channels when ``sig`` is a grid tensor.
    normalize_index
        If True and stacking from a grid, replaces the grid coord values with
        0..N-1 indices before stacking. This keeps channel indexing stable even
        when coords are non-uniform/unsorted.
    """
    if time_dim not in sig.dims:
        raise ValueError(f"sig must have time dim {time_dim!r}, got {list(sig.dims)}")

    # Case 1: already (time, ch)
    if ch_dim in sig.dims:
        extra = [d for d in sig.dims if d not in (time_dim, ch_dim)]
        if extra:
            raise ValueError(
                f"sig has dims {list(sig.dims)}; expected only ({time_dim!r}, {ch_dim!r})"
            )
        if sig.dims != (time_dim, ch_dim):
            sig = sig.transpose(time_dim, ch_dim)
        return sig

    # Case 2: (time, AP, ML) -> (time, ch)
    d0, d1 = grid_dims
    if d0 in sig.dims and d1 in sig.dims:
        extra = [d for d in sig.dims if d not in (time_dim, d0, d1)]
        if extra:
            raise ValueError(
                f"sig has dims {list(sig.dims)}; expected only ({time_dim!r}, {d0!r}, {d1!r})"
            )
        s = normalize_coords_to_index(sig, (d0, d1)) if bool(normalize_index) else sig
        s = s.stack({ch_dim: (d0, d1)})
        return s.transpose(time_dim, ch_dim)

    raise ValueError(
        f"sig dims {list(sig.dims)} not understood; expected ({time_dim!r}, {ch_dim!r}) "
        f"or ({time_dim!r}, {grid_dims[0]!r}, {grid_dims[1]!r})."
    )


def multichannel_view(
    sig: xr.DataArray,
    *,
    time_dim: str = "time",
    ch_dim: str = "ch",
    grid_dims: tuple[str, str] = ("AP", "ML"),
    normalize_index: bool = True,
    title: str | None = None,
    ylabel: str = "Channel (stacked)",
    boundsx: tuple[float, float] | None = None,
    width: int | None = None,
    overlay_height: int | None = None,
    minimap_height: int = 140,
    responsive: bool = True,
    return_parts: bool = False,
):
    """
    Umbrella wrapper for :func:`cogpy.core.plot.multichannel_timeseries.multichannel_timeseries_view`
    that accepts common ECoG xarray schemas.

    Accepted input schemas
    ----------------------
    - Flat: ``(time, ch)``
    - Grid: ``(time, AP, ML)`` (stacked into ``ch`` using ``sig.stack(ch=("AP","ML"))``)

    Parameters
    ----------
    sig
        Input signal DataArray.
    time_dim, ch_dim, grid_dims
        Dim names used to interpret/stack the input.
    normalize_index
        If True and stacking from a grid, normalizes ``grid_dims`` coords to
        integer indices before stacking.
    title
        Plot title. Defaults to ``sig.name``.
    boundsx
        Initial RangeTool x-bounds ``(t0, t1)``. If omitted, the helper uses a
        small window from the start of the signal.
    return_parts
        If True, returns the dict from ``multichannel_timeseries_view(..., return_parts=True)``.
    """
    from .multichannel_timeseries import multichannel_timeseries_view

    sig_tc = to_time_channel(
        sig,
        time_dim=time_dim,
        ch_dim=ch_dim,
        grid_dims=grid_dims,
        normalize_index=normalize_index,
    )

    t = np.asarray(sig_tc[time_dim].values, dtype=float).reshape(-1)
    x = np.asarray(sig_tc.values, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected stacked signal values to be 2D (time, ch), got shape {x.shape}.")

    # Default labels: use channel coordinate values (handles MultiIndex nicely via str()).
    try:
        ch_vals = sig_tc[ch_dim].values
        channel_labels = [str(v) for v in ch_vals]
    except Exception:
        channel_labels = None

    return multichannel_timeseries_view(
        t,
        x,
        title=(sig_tc.name if title is None else str(title)),
        channel_labels=channel_labels,
        ylabel=ylabel,
        boundsx=boundsx,
        width=width,
        overlay_height=overlay_height,
        minimap_height=int(minimap_height),
        responsive=bool(responsive),
        return_parts=bool(return_parts),
    )


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


def add_click_hair(
    obj,
    *,
    kdim: str,
    value=None,
    snap: bool = True,
    orientation: str = "v",
    return_controller: bool = False,
    **attach_kwargs,
):
    """Add a clickable 1D hair (vertical/horizontal line) to a HoloViews object.

    This is an umbrella wrapper around :func:`cogpy.core.plot.time_player.add_axis_hair`.

    Parameters
    ----------
    kdim
        Key dimension to bind to. The plot must contain an element with this
        kdim in its ``.kdims``.
    orientation
        ``"v"`` (default) uses tap.x and draws a ``hv.VLine``.
        ``"h"`` uses tap.y and draws a ``hv.HLine``.
    """
    from .time_player import add_axis_hair as _add_axis_hair

    return _add_axis_hair(
        obj,
        kdim=kdim,
        value=value,
        snap=snap,
        orientation=orientation,
        return_controller=return_controller,
        **attach_kwargs,
    )


def grid_movie_linked_to_controller(
    sigx: xr.DataArray,
    *,
    controller,
    controller_param: str = "value",
    time_dim: str = "time",
    x_dim: str = "ML",
    y_dim: str = "AP",
    normalize_index: bool = True,
    title: str | None = None,
    width: int = 400,
    height: int = 350,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    colorbar: bool = True,
    snap: bool = True,
):
    """Grid movie whose displayed frame follows a Param controller value.

    This is useful to *reuse* a single "time hair" controller across multiple
    views (e.g., a time×channel heatmap, a selected-channel trace, and an AP×ML
    movie).

    Parameters
    ----------
    sigx
        Input DataArray with dims including ``(time_dim, y_dim, x_dim)``.
    controller
        A Param object with attribute ``controller_param`` (default: ``"value"``),
        e.g. a controller returned by :func:`add_click_hair` (AxisHair).
    controller_param
        Name of the Param attribute to read (typically ``"value"``).
    snap
        If True, snaps the controller value to the nearest available time coordinate.
    """
    import holoviews as hv
    import numpy as np
    from holoviews import streams, opts

    if x_dim not in sigx.dims or y_dim not in sigx.dims or time_dim not in sigx.dims:
        raise ValueError(
            f"sigx must have dims {x_dim!r}, {y_dim!r}, {time_dim!r}, got {list(sigx.dims)}"
        )

    data = normalize_coords_to_index(sigx, (y_dim, x_dim)) if normalize_index else sigx
    t_vals = np.asarray(data[time_dim].values)

    def _snap_t(t):
        if t is None or t_vals.size == 0:
            return None
        if not bool(snap):
            return t
        # robust nearest for numeric time vectors
        try:
            tf = float(t)
            key = t_vals.astype(float)
            return float(t_vals[int(np.argmin(np.abs(key - tf)))])
        except Exception:
            # fallback for datetime-like
            try:
                t64 = np.datetime64(t, "ns")
                key = t_vals.astype("datetime64[ns]").astype("int64")
                q = t64.astype("int64")
                return t_vals[int(np.argmin(np.abs(key - q)))]
            except Exception:
                return t

    # Stream on controller value
    params = streams.Params(controller, [controller_param])

    def _frame(**kw):
        t = kw.get(controller_param)
        t = _snap_t(t)
        if t is None:
            return hv.Image([]).opts(alpha=0.0)
        fr = data.sel({time_dim: t}, method="nearest")
        img = hv.Image(fr, kdims=[x_dim, y_dim], vdims=[fr.name or "val"])
        return img.opts(
            opts.Image(
                cmap=cmap,
                symmetric=bool(symmetric),
                colorbar=bool(colorbar),
                width=int(width),
                height=int(height),
                title=(data.name if title is None else str(title)),
                tools=["tap", "pan", "wheel_zoom", "box_zoom", "reset"],
            )
        )

    return hv.DynamicMap(_frame, streams=[params])


def selected_channel_curve(
    sig_tc: xr.DataArray,
    *,
    ch_controller,
    time_dim: str = "time",
    ch_dim: str = "ch",
    ap_name: str = "AP",
    ml_name: str = "ML",
    width: int | None = None,
    height: int | None = None,
    as_numpy: bool = False,
    adaptive_ylim: bool = False,
    ylim_pad_frac: float = 0.08,
    framewise: bool = False,
):
    """Return a DynamicMap curve for the channel selected by ``ch_controller.value``."""
    import holoviews as hv
    import numpy as np
    from holoviews import streams
    from holoviews import opts

    hv.extension("bokeh", logo=False)

    if time_dim not in sig_tc.dims or ch_dim not in sig_tc.dims:
        raise ValueError(f"sig_tc must have dims ({time_dim!r}, {ch_dim!r}); got {tuple(sig_tc.dims)}")

    ch_vals = np.asarray(sig_tc[ch_dim].values)

    def _snap_ch(v):
        if ch_vals.size == 0:
            return 0
        try:
            vf = float(v)
            key = ch_vals.astype(float)
            return int(ch_vals[int(np.argmin(np.abs(key - vf)))])
        except Exception:
            # fallback: nearest by string match isn't meaningful, just cast
            try:
                return int(v)
            except Exception:
                return int(ch_vals[0])

    params = streams.Params(ch_controller, ["value"])

    def _curve(value=None, **_):
        ch = _snap_ch(value) if value is not None else _snap_ch(ch_vals[0] if ch_vals.size else 0)
        da = sig_tc.sel({ch_dim: ch})
        title = f"{ch_dim}={ch}"
        try:
            ap = float(da.coords[ap_name].values)
            ml = float(da.coords[ml_name].values)
            if np.isfinite(ap) and np.isfinite(ml):
                title = f"{ch_dim}={ch} ({ap_name}={ap:.2f}, {ml_name}={ml:.2f})"
        except Exception:
            pass
        if bool(as_numpy):
            tvals = np.asarray(da[time_dim].values)
            yvals = np.asarray(da.values)
            c = hv.Curve((tvals, yvals), kdims=[time_dim], vdims=[sig_tc.name or "val"])
        else:
            c = hv.Curve(da, kdims=[time_dim])
        c = c.opts(title=title, tools=["tap", "wheel_zoom", "box_zoom", "reset"])
        if bool(adaptive_ylim):
            y = np.asarray(da.values, dtype=float)
            if y.size:
                y_min = float(np.nanmin(y))
                y_max = float(np.nanmax(y))
                if np.isfinite(y_min) and np.isfinite(y_max):
                    if y_max <= y_min:
                        y_max = y_min + 1.0
                    pad = float(ylim_pad_frac) * (y_max - y_min)
                    c = c.opts(ylim=(y_min - pad, y_max + pad))
        if width is not None:
            c = c.opts(width=int(width))
        if height is not None:
            c = c.opts(height=int(height))
        return c

    return hv.DynamicMap(_curve, streams=[params]).opts(opts.Curve(framewise=bool(framewise)))


def apml_crosshair_from_channel(
    *,
    ch_controller,
    ap_ch: np.ndarray,
    ml_ch: np.ndarray,
    color: str = "orange",
    line_width: int = 2,
):
    """Return a DynamicMap crosshair (HLine*VLine) from a selected channel controller."""
    import holoviews as hv
    from holoviews import streams, opts

    hv.extension("bokeh", logo=False)

    ap_ch = np.asarray(ap_ch, dtype=float).reshape(-1)
    ml_ch = np.asarray(ml_ch, dtype=float).reshape(-1)
    if ap_ch.shape != ml_ch.shape:
        raise ValueError("ap_ch and ml_ch must have the same shape")

    params = streams.Params(ch_controller, ["value"])

    def _cross(value=None, **_):
        if value is None or ap_ch.size == 0:
            return hv.HLine(0).opts(alpha=0.0, line_width=0) * hv.VLine(0).opts(alpha=0.0, line_width=0)
        try:
            ch = int(round(float(value)))
        except Exception:
            ch = 0
        ch = int(np.clip(ch, 0, ap_ch.size - 1))
        ap = float(ap_ch[ch])
        ml = float(ml_ch[ch])
        return (hv.HLine(ap) * hv.VLine(ml)).opts(
            opts.HLine(color=str(color), line_width=int(line_width)),
            opts.VLine(color=str(color), line_width=int(line_width)),
        )

    return hv.DynamicMap(_cross, streams=[params])


def bind_apml_tap_to_channel_controller(
    obj,
    *,
    ch_controller,
    ap_ch: np.ndarray,
    ml_ch: np.ndarray,
):
    """Bind Tap on an AP×ML view to update ``ch_controller.value``.

    Assumes the tapped element reports ``x=ML`` and ``y=AP`` (as HoloViews does
    for ``hv.Image`` with kdims ``(ML, AP)``).

    Returns the Tap stream (keep a reference to avoid GC).
    """
    import holoviews as hv
    from holoviews import streams

    hv.extension("bokeh", logo=False)

    ap_ch = np.asarray(ap_ch, dtype=float).reshape(-1)
    ml_ch = np.asarray(ml_ch, dtype=float).reshape(-1)

    tap = streams.Tap(source=obj)

    def _on_tap(x=None, y=None, **_):
        if x is None or y is None:
            return
        ch = snap_ch_from_apml(float(y), float(x), ap_ch=ap_ch, ml_ch=ml_ch)
        setattr(ch_controller, "value", ch)

    tap.add_subscriber(_on_tap)
    return tap


def hair_from_controller(
    *,
    controller,
    controller_param: str = "value",
    orientation: str = "v",
    color: str = "red",
    line_width: int = 2,
    line_alpha: float = 0.9,
):
    """Create a DynamicMap hair (VLine/HLine) driven by a Param controller.

    Unlike directly constructing ``hv.VLine(controller.value)``, this helper is
    safe when the controller value is ``None`` (returns an invisible placeholder).
    """
    import holoviews as hv
    from holoviews import streams

    hv.extension("bokeh", logo=False)

    o = str(orientation).lower()
    if o not in {"v", "h"}:
        raise ValueError("orientation must be 'v' or 'h'")

    params = streams.Params(controller, [controller_param])

    def _hair(**kw):
        v = kw.get(controller_param)
        if v is None:
            placeholder = hv.VLine(0) if o == "v" else hv.HLine(0)
            return placeholder.opts(alpha=0.0, line_width=0)
        el = hv.VLine(v) if o == "v" else hv.HLine(v)
        return el.opts(color=str(color), line_width=int(line_width), alpha=float(line_alpha))

    return hv.DynamicMap(_hair, streams=[params])


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
