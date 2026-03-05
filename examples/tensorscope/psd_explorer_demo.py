"""
PSD Explorer Demo (v2.8.0).

Runs the `psd_explorer` module on example grid data.

Run with:
    /storage/share/python/environments/Anaconda3/envs/cogpy/bin/python \\
        code/lib/cogpy/examples/tensorscope/psd_explorer_demo.py
"""

from __future__ import annotations

import holoviews as hv
import panel as pn

from cogpy.core.plot.tensorscope.modules import ModuleRegistry
from cogpy.core.plot.tensorscope.state import TensorScopeState
from cogpy.datasets.entities import example_ieeg_grid


def main():
    pn.extension()
    hv.extension("bokeh")

    data = example_ieeg_grid(mode="small")
    state = TensorScopeState(data)

    reg = ModuleRegistry()
    mod = reg.get("psd_explorer")
    layout = mod.activate(state) if mod is not None else hv.Div("<b>psd_explorer not found</b>")

    template = pn.template.FastListTemplate(
        title="TensorScope v2.8.0: PSD Explorer Demo",
        main=[pn.pane.HoloViews(layout, sizing_mode="stretch_both")],
    )
    return template


if __name__ == "__main__":
    main().show()

