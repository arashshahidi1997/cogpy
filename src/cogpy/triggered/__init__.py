"""Triggered analysis primitives for event-locked epochs.

Statistics: ``triggered_average``, ``triggered_std``, ``triggered_median``,
``triggered_snr``.
Template ops: ``estimate_template``, ``fit_scaling``, ``subtract_template``.

Consumes epoch arrays (e.g. from ``cogpy.brainstates.intervals.perievent_epochs``)
and returns summary statistics, templates, or cleaned signals.
"""

from cogpy.triggered.stats import (
    triggered_average,
    triggered_std,
    triggered_median,
    triggered_snr,
)
from cogpy.triggered.template import (
    estimate_template,
    fit_scaling,
    subtract_template,
)

__all__ = [
    "triggered_average",
    "triggered_std",
    "triggered_median",
    "triggered_snr",
    "estimate_template",
    "fit_scaling",
    "subtract_template",
]
