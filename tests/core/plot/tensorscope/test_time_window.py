"""Tests for TimeWindowCtrl."""

from __future__ import annotations

import pytest

from cogpy.core.plot.tensorscope.time_window import TimeWindowCtrl


def test_time_window_set_window_validation():
    ctrl = TimeWindowCtrl(bounds=(0.0, 10.0))
    with pytest.raises(ValueError, match="t0 < t1"):
        ctrl.set_window(2.0, 2.0)


def test_time_window_snaps_to_bounds():
    ctrl = TimeWindowCtrl(bounds=(0.0, 10.0), snap=True)
    ctrl.set_window(-5.0, 2.0)
    assert ctrl.window[0] == 0.0
    assert ctrl.window[1] == 2.0


def test_time_window_recenter():
    ctrl = TimeWindowCtrl(bounds=(0.0, 10.0), snap=True)
    ctrl.recenter(7.0, width_s=2.0)
    assert ctrl.window == (6.0, 8.0)

