from __future__ import annotations

from typing import Literal

import numpy as np

Unit = Literal["V", "uV", "µV"]


def scale_to_volts(x: np.ndarray, unit: Unit) -> tuple[np.ndarray, bool]:
    """
    Return (x_in_volts, scaled_to_volts).

    MNE expects Volts for eeg/ecog/seeg channels.
    """
    if unit in ("uV", "µV"):
        return x * 1e-6, True
    if unit == "V":
        return x, False
    raise ValueError("unit must be 'V', 'uV', or 'µV'")

