"""
Command-line interface for TensorScope.

This CLI follows the style of other cogpy CLIs (argparse, minimal deps).

Examples
--------
Serve a dataset:

    tensorscope serve recording.nc --layout default --port 5008 --show

Serve a dataset into a module view:

    tensorscope serve recording.nc --module psd_explorer --port 5008 --show

List presets:

    tensorscope presets

List modules:

    tensorscope modules
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tensorscope", description="TensorScope visualization app")
    ap.add_argument(
        "--version",
        action="store_true",
        help="Print TensorScope version and exit.",
    )

    sub = ap.add_subparsers(dest="cmd", required=False)

    serve = sub.add_parser("serve", help="Launch TensorScope on an xarray DataArray file")
    serve.add_argument(
        "data_path",
        type=Path,
        help="Path to data: xarray (.nc/.zarr) or BIDS iEEG binary (.lfp with sidecars)",
    )
    serve.add_argument("--layout", default="default", help="Layout preset name")
    serve.add_argument(
        "--module",
        default=None,
        help="Module to display (e.g., psd_explorer). If set, runs TensorScope in module view mode.",
    )
    serve.add_argument("--port", type=int, default=5006, help="Server port")
    serve.add_argument("--show", action=argparse.BooleanOptionalAction, default=True, help="Open browser")
    serve.add_argument("--title", default="TensorScope", help="Application title")

    sub.add_parser("presets", help="List available layout presets")
    sub.add_parser("modules", help="List available modules")
    cfg = sub.add_parser("config", help="Show config (placeholder)")
    cfg.add_argument("--show", action="store_true", help="Show current config")

    return ap


def _cmd_presets() -> int:
    from cogpy.core.tensorscope.layout import LayoutManager

    mgr = LayoutManager()
    print("Available Layout Presets:\n")
    for name in mgr.preset_names():
        preset = mgr.get_preset(name)
        print(f"  {preset.name}:\n    {preset.description}\n")
    return 0


def _cmd_config(show: bool) -> int:
    if not show:
        return 0
    from cogpy.core.tensorscope import __version__

    print("TensorScope Configuration:")
    print(f"  Version: {__version__}")
    print("  Config file: ~/.tensorscope/config.yaml (not yet implemented)")
    print("\nAvailable layouts:")
    print("  - default")
    print("  - spatial_focus")
    print("  - timeseries_focus")
    return 0


def _cmd_modules() -> int:
    from cogpy.core.tensorscope.modules import ModuleRegistry

    reg = ModuleRegistry()
    print("Available Modules:\n")
    for name in sorted(reg.list()):
        mod = reg.get(name)
        desc = getattr(mod, "description", "") if mod is not None else ""
        if desc:
            print(f"  {name}:\n    {desc}\n")
        else:
            print(f"  {name}\n")
    return 0


def _cmd_serve(
    data_path: Path, layout: str, module: str | None, port: int, show: bool, title: str
) -> int:
    if not data_path.exists():
        print(f"Error: file not found: {data_path}", file=sys.stderr)
        return 2

    try:
        import xarray as xr
    except Exception as e:  # noqa: BLE001
        print(f"Error: xarray not available: {e}", file=sys.stderr)
        return 1

    try:
        import panel as pn
    except Exception as e:  # noqa: BLE001
        print(
            "Error: panel is required for serving TensorScope.\n"
            "Install with: pip install 'cogpy[viz]'",
            file=sys.stderr,
        )
        print(f"Details: {e}", file=sys.stderr)
        return 1

    def _load_data(p: Path):
        p = Path(p)
        suf = p.suffix.lower()

        # Case 1: BIDS iEEG binary with sidecars (e.g., *.lfp).
        if suf in {".lfp", ".dat"}:
            from cogpy.io import ieeg_io

            # Prefer grid=True (TensorScope expects grid-capable data), but fall back
            # to linear channel mode if metadata is incomplete.
            try:
                return ieeg_io.from_file(p, grid=True)
            except Exception:  # noqa: BLE001
                da = ieeg_io.from_file(p, grid=False)
                if ("ch" in getattr(da, "dims", ())) and ("channel" not in da.dims):
                    da = da.rename({"ch": "channel"})
                return da

        # Case 2: Zarr store (directory or *.zarr path).
        if suf == ".zarr" or (p.is_dir() and p.name.lower().endswith(".zarr")):
            ds = xr.open_zarr(p)
            # open_zarr returns a Dataset; pick a DataArray deterministically.
            if hasattr(ds, "data_vars"):
                if len(ds.data_vars) == 1:
                    return next(iter(ds.data_vars.values()))
                if "ieeg" in ds.data_vars:
                    return ds["ieeg"]
                raise ValueError(
                    f"Zarr store has multiple variables; specify a single DataArray. "
                    f"Found: {list(ds.data_vars)}"
                )
            return ds

        # Case 3: NetCDF or other xarray-serializable.
        try:
            return xr.load_dataarray(p)
        except Exception:  # noqa: BLE001
            # Fall back to Dataset load and pick a variable.
            ds = xr.load_dataset(p)
            if len(ds.data_vars) == 1:
                return next(iter(ds.data_vars.values()))
            if "ieeg" in ds.data_vars:
                return ds["ieeg"]
            raise ValueError(
                f"File contains multiple variables; expected a single DataArray. "
                f"Found: {list(ds.data_vars)}"
            )

    print(f"Loading data from: {data_path}")
    try:
        data = _load_data(data_path)
    except Exception as e:  # noqa: BLE001
        print(f"Error: failed to load data: {e}", file=sys.stderr)
        return 1

    print(f"  Loaded: dims={data.dims}, shape={data.shape}")

    if module:
        if str(module) == "psd_explorer":
            from cogpy.core.tensorscope import TensorScopeApp

            # If the user didn't explicitly choose a layout, default to the PSD explorer preset.
            layout2 = "psd_explorer" if str(layout) == "default" else str(layout)

            print(f"Creating TensorScope app (layout: {layout2}, module: {module})...")
            try:
                app = (
                    TensorScopeApp(data, title=title)
                    .with_layout(layout2)
                    .add_layer("timeseries")
                    .add_layer("spatial_map")
                    .add_layer("selector")
                    .add_layer("processing")
                    .add_layer("psd_settings")
                    .add_layer("navigator")
                    .add_layer("psd_explorer")
                )
            except Exception as e:  # noqa: BLE001
                print(f"Error: failed to create app: {e}", file=sys.stderr)
                return 1

            template = app.build()

            print(f"Starting server on port {port}...")
            pn.serve({"/": template}, port=int(port), show=bool(show), title=str(title))
            return 0

        from cogpy.core.tensorscope import TensorScopeState
        from cogpy.core.tensorscope.layers.controls import ProcessingControlsLayer
        from cogpy.core.tensorscope.modules import ModuleRegistry

        reg = ModuleRegistry()
        mod = reg.get(str(module))
        if mod is None:
            print(f"Error: unknown module: {module!r}", file=sys.stderr)
            print("Run `tensorscope modules` to list available modules.", file=sys.stderr)
            return 2

        if layout and str(layout) != "default":
            print("Note: --layout is ignored for module-view mode (non-layer modules).")

        print(f"Creating TensorScope module view (module: {module})...")
        try:
            state = TensorScopeState(data)
            module_view = mod.activate(state)
        except Exception as e:  # noqa: BLE001
            print(f"Error: failed to activate module: {e}", file=sys.stderr)
            return 1

        template = pn.template.FastGridTemplate(
            title=str(title),
            theme="dark",
            sidebar_width=320,
            row_height=80,
        )
        try:
            template.sidebar.append(ProcessingControlsLayer(state).panel())
        except Exception:  # noqa: BLE001
            pass

        template.main[0:10, 0:12] = pn.panel(module_view, sizing_mode="stretch_both")

        print(f"Starting server on port {port}...")
        pn.serve({"/": template}, port=int(port), show=bool(show), title=str(title))
        return 0

    from cogpy.core.tensorscope import TensorScopeApp

    print(f"Creating TensorScope app (layout: {layout})...")
    try:
        app = (
            TensorScopeApp(data, title=title)
            .with_layout(layout)
            .add_layer("timeseries")
            .add_layer("spatial_map")
            .add_layer("selector")
            .add_layer("signal_manager")
            .add_layer("processing")
            .add_layer("navigator")
        )
    except Exception as e:  # noqa: BLE001
        print(f"Error: failed to create app: {e}", file=sys.stderr)
        return 1

    template = app.build()

    print(f"Starting server on port {port}...")
    pn.serve({"/": template}, port=int(port), show=bool(show), title=str(title))
    return 0


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    ap = _build_parser()
    ns = ap.parse_args(argv)

    if ns.version:
        from cogpy.core.tensorscope import __version__

        print(__version__)
        raise SystemExit(0)

    if ns.cmd == "presets":
        raise SystemExit(_cmd_presets())
    if ns.cmd == "modules":
        raise SystemExit(_cmd_modules())
    if ns.cmd == "config":
        raise SystemExit(_cmd_config(bool(ns.show)))
    if ns.cmd == "serve":
        raise SystemExit(
            _cmd_serve(
                data_path=ns.data_path,
                layout=str(ns.layout),
                module=str(ns.module) if ns.module else None,
                port=int(ns.port),
                show=bool(ns.show),
                title=str(ns.title),
            )
        )

    ap.print_help()
    raise SystemExit(0)


if __name__ == "__main__":
    main()
