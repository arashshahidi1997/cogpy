"""
Triggered analysis primitives.

Functions for computing statistics on event-locked epochs
and for template-based signal manipulation.

These primitives consume epoch arrays (typically produced by
``cogpy.brainstates.intervals.perievent_epochs``) and return
summary statistics, templates, or cleaned signals.
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
