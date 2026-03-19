"""
Linear regression primitives for signal decomposition.

Provides atomic building blocks for constructing design matrices,
fitting linear models, and subtracting predicted components from signals.
"""

from cogpy.regression.design import (
    lagged_design_matrix,
    event_design_matrix,
)
from cogpy.regression.ols import (
    ols_fit,
    ols_predict,
    ols_residual,
)

__all__ = [
    "lagged_design_matrix",
    "event_design_matrix",
    "ols_fit",
    "ols_predict",
    "ols_residual",
]
