import numpy as np
import pytest
import xarray as xr


def _make_multichannel_signal(*, fs: float = 1000.0, T: float = 10.0, n_chan: int = 4):
    rng = np.random.default_rng(42)
    N = int(T * fs)
    t = np.arange(N) / fs
    sig = xr.DataArray(
        rng.standard_normal((n_chan, N)),
        dims=("channel", "time"),
        coords={"channel": np.arange(n_chan), "time": t},
        attrs={"fs": fs, "meta": "keep"},
    )
    return sig, t


def test_restrict_intervals_object():
    from cogpy.brainstates.intervals import restrict
    from cogpy.datasets.schemas import Intervals

    sig, t = _make_multichannel_signal()
    iv = Intervals(starts=[1.0, 5.0, 8.0], ends=[3.0, 7.0, 9.5], name="PerSWS")
    out = restrict(sig, iv)

    expected_mask = (
        ((t >= 1.0) & (t < 3.0)) | ((t >= 5.0) & (t < 7.0)) | ((t >= 8.0) & (t < 9.5))
    )
    assert out.sizes["time"] == int(expected_mask.sum())


def test_restrict_state_dict():
    from cogpy.brainstates.intervals import restrict

    sig, t = _make_multichannel_signal()
    states = {"PerSWS": [[1.0, 3.0], [5.0, 7.0]], "PerREM": [[3.0, 5.0]]}
    out = restrict(sig, states)

    expected_mask = (t >= 1.0) & (t < 7.0)
    assert out.sizes["time"] == int(expected_mask.sum())


def test_restrict_plain_array():
    from cogpy.brainstates.intervals import restrict

    sig, t = _make_multichannel_signal()
    out = restrict(sig, [[1.0, 3.0], [5.0, 7.0]])

    expected_mask = ((t >= 1.0) & (t < 3.0)) | ((t >= 5.0) & (t < 7.0))
    assert out.sizes["time"] == int(expected_mask.sum())


def test_restrict_preserves_attrs():
    from cogpy.brainstates.intervals import restrict
    from cogpy.datasets.schemas import Intervals

    sig, _ = _make_multichannel_signal()
    iv = Intervals(starts=[1.0], ends=[2.0], name="iv")
    out = restrict(sig, iv)
    assert out.attrs == sig.attrs


def test_restrict_preserves_dims():
    from cogpy.brainstates.intervals import restrict
    from cogpy.datasets.schemas import Intervals

    sig, _ = _make_multichannel_signal()
    iv = Intervals(starts=[1.0], ends=[2.0], name="iv")
    out = restrict(sig, iv)
    assert out.dims == sig.dims


def test_restrict_single_interval():
    from cogpy.brainstates.intervals import restrict

    sig, t = _make_multichannel_signal()
    out = restrict(sig, [1.0, 3.0])
    expected_mask = (t >= 1.0) & (t < 3.0)
    assert out.sizes["time"] == int(expected_mask.sum())


def test_restrict_empty_result():
    from cogpy.brainstates.intervals import restrict

    sig, _ = _make_multichannel_signal()
    out = restrict(sig, [[100.0, 101.0]])
    assert out.sizes["time"] == 0


def test_perievent_epochs_shape():
    from cogpy.brainstates.intervals import perievent_epochs

    sig, _ = _make_multichannel_signal(fs=1000.0, T=10.0, n_chan=4)
    events = np.array([2.0, 5.5, 8.5])
    pre, post = 0.5, 1.0
    epochs = perievent_epochs(sig, events, fs=1000.0, pre=pre, post=post)
    n_samples = int(round((pre + post) * 1000.0)) + 1
    assert epochs.dims == ("event", "channel", "lag")
    assert epochs.sizes["event"] == 3
    assert epochs.sizes["channel"] == 4
    assert epochs.sizes["lag"] == n_samples


def test_perievent_epochs_lag_coords():
    from cogpy.brainstates.intervals import perievent_epochs

    sig, _ = _make_multichannel_signal(fs=1000.0, T=10.0, n_chan=2)
    pre, post = 0.5, 1.0
    epochs = perievent_epochs(sig, [2.0], fs=1000.0, pre=pre, post=post)
    lag = epochs.coords["lag"].values
    assert lag[0] == pytest.approx(-pre)
    assert lag[-1] == pytest.approx(post)
    assert len(lag) == int(round((pre + post) * 1000.0)) + 1


def test_perievent_epochs_event_coords():
    from cogpy.brainstates.intervals import perievent_epochs

    sig, _ = _make_multichannel_signal(fs=1000.0, T=10.0, n_chan=2)
    events = np.array([2.0, 5.5, 8.5])
    epochs = perievent_epochs(sig, events, fs=1000.0, pre=0.5, post=1.0)
    assert np.allclose(epochs.coords["event"].values, events)


def test_perievent_epochs_preserves_attrs():
    from cogpy.brainstates.intervals import perievent_epochs

    sig, _ = _make_multichannel_signal(fs=1000.0, T=10.0, n_chan=1)
    epochs = perievent_epochs(sig, [2.0], fs=1000.0, pre=0.5, post=1.0)
    assert epochs.attrs["meta"] == "keep"
    assert epochs.attrs["fs"] == pytest.approx(1000.0)
    assert epochs.attrs["pre"] == pytest.approx(0.5)
    assert epochs.attrs["post"] == pytest.approx(1.0)


def test_perievent_epochs_near_boundary_padded():
    from cogpy.brainstates.intervals import perievent_epochs

    fs = 10.0
    T = 2.0
    N = int(T * fs)
    t = np.arange(N) / fs
    sig = xr.DataArray(
        np.arange(N, dtype=float)[np.newaxis, :],
        dims=("channel", "time"),
        coords={"channel": [0], "time": t},
        attrs={"fs": fs},
    )

    # event near start, needs left padding with NaNs for pre=0.5s at fs=10Hz -> 5 samples
    epochs = perievent_epochs(sig, [0.1], fs=fs, pre=0.5, post=0.5)
    ep0 = epochs.isel(event=0, channel=0).values
    assert np.isnan(ep0[:4]).all()
    assert np.isfinite(ep0[4:]).any()


def test_perievent_epochs_triggered_average_shape():
    from cogpy.brainstates.intervals import perievent_epochs

    sig, _ = _make_multichannel_signal(fs=1000.0, T=10.0, n_chan=4)
    epochs = perievent_epochs(sig, [2.0, 5.5, 8.5], fs=1000.0, pre=0.5, post=1.0)
    mean_epoch = epochs.mean("event")
    assert mean_epoch.dims == ("channel", "lag")


def test_perievent_epochs_events_object_input():
    from cogpy.brainstates.intervals import perievent_epochs
    from cogpy.datasets.schemas import Events

    sig, _ = _make_multichannel_signal(fs=1000.0, T=10.0, n_chan=2)
    ev = Events(times=[2.0, 5.5, 8.5], name="spindles")
    epochs = perievent_epochs(sig, ev, fs=1000.0, pre=0.5, post=1.0)
    assert np.allclose(epochs.coords["event"].values, ev.times)
