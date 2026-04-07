import numpy as np
import pytest


def test_intervals_construction_valid():
    from cogpy.datasets.schemas import Intervals

    iv = Intervals(starts=[1.0, 5.0], ends=[3.0, 7.0], name="PerSWS")
    assert isinstance(iv.starts, np.ndarray)
    assert isinstance(iv.ends, np.ndarray)
    assert iv.starts.dtype == float
    assert iv.ends.dtype == float
    assert iv.name == "PerSWS"
    assert len(iv) == 2
    assert np.allclose(iv.starts, [1.0, 5.0])
    assert np.allclose(iv.ends, [3.0, 7.0])


def test_intervals_construction_invalid_ends_before_starts():
    from cogpy.datasets.schemas import Intervals

    with pytest.raises(ValueError, match="ends must be strictly greater"):
        Intervals(starts=[1.0, 5.0], ends=[1.0, 4.9])


def test_intervals_construction_mismatched_lengths():
    from cogpy.datasets.schemas import Intervals

    with pytest.raises(ValueError, match="same length"):
        Intervals(starts=[1.0, 5.0], ends=[3.0])


def test_intervals_construction_nonfinite():
    from cogpy.datasets.schemas import Intervals

    with pytest.raises(ValueError, match="must be finite"):
        Intervals(starts=[1.0, np.nan], ends=[3.0, 4.0])

    with pytest.raises(ValueError, match="must be finite"):
        Intervals(starts=[1.0, 2.0], ends=[3.0, np.inf])


def test_intervals_from_array_roundtrip():
    from cogpy.datasets.schemas import Intervals

    iv = Intervals(starts=[1.0, 5.0], ends=[3.0, 7.0], name="PerSWS")
    arr = iv.to_array()
    iv2 = Intervals.from_array(arr, name="PerSWS")
    assert np.allclose(iv2.starts, iv.starts)
    assert np.allclose(iv2.ends, iv.ends)
    assert iv2.name == "PerSWS"


def test_intervals_from_state_dict():
    from cogpy.datasets.schemas import Intervals

    states = {"PerSWS": [[1.0, 3.0], [5.0, 7.0]], "PerREM": [[3.0, 5.0]]}
    iv = Intervals.from_state_dict(states, "PerSWS")
    assert iv.name == "PerSWS"
    assert len(iv) == 2
    assert np.allclose(iv.starts, [1.0, 5.0])
    assert np.allclose(iv.ends, [3.0, 7.0])


def test_intervals_from_state_dict_missing_key():
    from cogpy.datasets.schemas import Intervals

    states = {"PerSWS": [[1.0, 3.0]]}
    with pytest.raises(KeyError, match="not found in states dict"):
        Intervals.from_state_dict(states, "PerREM")


def test_intervals_total_duration():
    from cogpy.datasets.schemas import Intervals

    iv = Intervals(starts=[1.0, 5.0, 8.0], ends=[3.0, 7.0, 9.5])
    assert iv.total_duration() == pytest.approx(5.5)


def test_intervals_len_repr():
    from cogpy.datasets.schemas import Intervals

    iv = Intervals(starts=[1.0, 5.0], ends=[3.0, 7.0], name="PerSWS")
    assert len(iv) == 2
    s = repr(iv)
    assert "Intervals(" in s
    assert "name='PerSWS'" in s
    assert "n=2" in s

    empty = Intervals(starts=[], ends=[], name="empty")
    assert len(empty) == 0
    s0 = repr(empty)
    assert "Intervals(" in s0
    assert "name='empty'" in s0
    assert "n=0" in s0


def test_events_construction_valid():
    from cogpy.datasets.schemas import Events

    ev = Events(times=[2.0, 5.5, 8.5], name="spindles")
    assert len(ev) == 3
    assert ev.name == "spindles"
    assert ev.times.dtype == float
    assert ev.labels.dtype.kind in ("U", "S", "O")
    assert np.all(ev.labels == np.array(["", "", ""], dtype=str))


def test_events_construction_sorted():
    from cogpy.datasets.schemas import Events

    ev = Events(times=[5.0, 1.0, 3.0], labels=["b", "a", "c"], name="ev")
    assert np.allclose(ev.times, [1.0, 3.0, 5.0])
    assert np.array_equal(ev.labels, np.array(["a", "c", "b"], dtype=str))


def test_events_construction_mismatched_labels():
    from cogpy.datasets.schemas import Events

    with pytest.raises(ValueError, match="same length as times"):
        Events(times=[1.0, 2.0], labels=["a"])


def test_events_to_intervals():
    from cogpy.datasets.schemas import Events

    ev = Events(times=[2.0, 5.5, 8.5], name="spindles")
    iv = ev.to_intervals(pre=0.5, post=1.0)
    assert len(iv) == 3
    assert iv.name == "spindles"
    assert np.allclose(iv.starts, ev.times - 0.5)
    assert np.allclose(iv.ends, ev.times + 1.0)


def test_events_restrict():
    from cogpy.datasets.schemas import Events, Intervals

    ev = Events(
        times=[0.5, 2.0, 4.0, 8.5], labels=["a", "b", "c", "d"], name="spindles"
    )
    iv = Intervals(starts=[1.0, 8.0], ends=[3.0, 9.0], name="keep")
    ev2 = ev.restrict(iv)
    assert np.allclose(ev2.times, [2.0, 8.5])
    assert np.array_equal(ev2.labels, np.array(["b", "d"], dtype=str))
    assert ev2.name == "spindles"


def test_events_len_repr():
    from cogpy.datasets.schemas import Events

    ev = Events(times=[2.0, 5.5, 8.5], name="spindles")
    assert len(ev) == 3
    s = repr(ev)
    assert "Events(" in s
    assert "name='spindles'" in s
    assert "n=3" in s

    empty = Events(times=[], name="empty")
    assert len(empty) == 0
    s0 = repr(empty)
    assert "Events(" in s0
    assert "name='empty'" in s0
    assert "n=0" in s0
