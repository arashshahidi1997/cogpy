import pytest


def test_ieeg_toolkit_app_constructs():
    pn = pytest.importorskip("panel")
    pytest.importorskip("holoviews")
    pytest.importorskip("param")

    from cogpy.plot.hv.ieeg_toolkit import ieeg_toolkit_app

    app = ieeg_toolkit_app(mode="small", seed=0)
    assert isinstance(app, pn.viewable.Viewable)


def test_spectrogram_bursts_app_constructs():
    pn = pytest.importorskip("panel")
    pytest.importorskip("holoviews")
    pytest.importorskip("param")

    from cogpy.plot._legacy.spectrogram_bursts_app import spectrogram_bursts_app

    app = spectrogram_bursts_app(mode="small", seed=0, kind="toy")
    assert isinstance(app, pn.viewable.Viewable)

