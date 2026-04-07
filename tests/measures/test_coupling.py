"""Tests for cogpy.measures.coupling — CCG, PETH, spectral power xcorr."""

import numpy as np
import pandas as pd
import pytest

from cogpy.measures.coupling import (
    cross_correlogram,
    peri_event_histogram,
    spectral_power_xcorr,
)
from cogpy.events.catalog import EventCatalog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def poisson_spikes(rng):
    """~20 Hz Poisson process over 60 s."""
    n = rng.poisson(20 * 60)
    return np.sort(rng.uniform(0, 60, n))


@pytest.fixture
def events_a(rng):
    """50 trigger events spread across 60 s."""
    return np.sort(rng.uniform(1, 59, 50))


@pytest.fixture
def events_b(rng):
    """40 independent events spread across 60 s."""
    return np.sort(rng.uniform(1, 59, 40))


# ---------------------------------------------------------------------------
# cross_correlogram
# ---------------------------------------------------------------------------


class TestCrossCorrelogram:
    def test_output_shape(self, events_a, events_b):
        res = cross_correlogram(events_a, events_b, bin_size=0.01, window=0.5)
        n_bins = int(2 * 0.5 / 0.01)
        assert res["lags"].shape[0] == pytest.approx(n_bins, abs=2)
        assert res["ccg"].shape == res["lags"].shape

    def test_lags_symmetric(self, events_a, events_b):
        res = cross_correlogram(events_a, events_b, bin_size=0.01, window=0.5)
        lags = res["lags"]
        np.testing.assert_allclose(lags + lags[::-1], 0.0, atol=1e-10)

    def test_normalize_units(self, events_a, events_b):
        """Normalized output should be in Hz (> 0 for Poisson events)."""
        res = cross_correlogram(
            events_a, events_b, bin_size=0.01, window=0.5, normalize=True
        )
        assert res["ccg"].sum() > 0

    def test_raw_counts_are_integers(self, events_a, events_b):
        res = cross_correlogram(
            events_a, events_b, bin_size=0.01, window=0.5, normalize=False
        )
        np.testing.assert_array_equal(res["ccg"] % 1, 0)

    def test_surrogates_shape(self, events_a, events_b):
        res = cross_correlogram(
            events_a, events_b, bin_size=0.01, window=0.5, n_surrogates=5, seed=0
        )
        assert res["surrogates"].shape == (5, len(res["lags"]))

    def test_no_surrogates_key_absent(self, events_a, events_b):
        res = cross_correlogram(
            events_a, events_b, bin_size=0.01, window=0.5, n_surrogates=0
        )
        assert "surrogates" not in res

    def test_accepts_event_catalog(self, events_a, events_b):
        df_a = pd.DataFrame({"event_id": range(len(events_a)), "t": events_a})
        cat_a = EventCatalog(df=df_a)
        res = cross_correlogram(cat_a, events_b, bin_size=0.01, window=0.5)
        assert res["ccg"].shape == res["lags"].shape

    def test_coarse_timescale(self, events_a, events_b):
        """Coarse bins: 100 ms, ±5 s window."""
        res = cross_correlogram(events_a, events_b, bin_size=0.1, window=5.0)
        assert len(res["lags"]) == pytest.approx(100, abs=2)

    def test_self_correlogram_zero_lag_peak(self, events_a):
        """Self-CCG should have its peak at lag 0."""
        res = cross_correlogram(
            events_a, events_a, bin_size=0.005, window=0.1, normalize=False
        )
        zero_idx = np.argmin(np.abs(res["lags"]))
        assert res["ccg"][zero_idx] == res["ccg"].max()


# ---------------------------------------------------------------------------
# peri_event_histogram
# ---------------------------------------------------------------------------


class TestPeriEventHistogram:
    def test_output_shape(self, poisson_spikes, events_a):
        res = peri_event_histogram(poisson_spikes, events_a, bin_size=0.01, window=0.5)
        n_bins = int(2 * 0.5 / 0.01)
        assert res["lags"].shape[0] == pytest.approx(n_bins, abs=2)
        assert res["peth"].shape == res["lags"].shape
        assert res["peth_sem"].shape == res["lags"].shape

    def test_n_events_recorded(self, poisson_spikes, events_a):
        res = peri_event_histogram(poisson_spikes, events_a, bin_size=0.01, window=0.5)
        assert res["n_events"] == len(events_a)

    def test_rate_approximately_correct(self, poisson_spikes, events_a):
        """Mean PETH rate should be near the overall firing rate (~20 Hz)."""
        res = peri_event_histogram(poisson_spikes, events_a, bin_size=0.01, window=0.5)
        assert 5 < res["peth"].mean() < 40

    def test_baseline_normalization(self, poisson_spikes, events_a):
        """After baseline norm, the mean rate in baseline window should be ~1."""
        res = peri_event_histogram(
            poisson_spikes,
            events_a,
            bin_size=0.01,
            window=0.5,
            baseline=(-0.5, -0.1),
        )
        lags = res["lags"]
        bl_mask = (lags >= -0.5) & (lags <= -0.1)
        np.testing.assert_allclose(res["peth"][bl_mask].mean(), 1.0, atol=0.2)

    def test_surrogates_shape(self, poisson_spikes, events_a):
        res = peri_event_histogram(
            poisson_spikes, events_a, bin_size=0.01, window=0.5, n_surrogates=3, seed=7
        )
        assert res["surrogates"].shape == (3, len(res["lags"]))

    def test_accepts_event_catalog_for_events(self, poisson_spikes, events_a):
        df = pd.DataFrame({"event_id": range(len(events_a)), "t": events_a})
        cat = EventCatalog(df=df)
        res = peri_event_histogram(poisson_spikes, cat, bin_size=0.01, window=0.5)
        assert res["n_events"] == len(events_a)


# ---------------------------------------------------------------------------
# spectral_power_xcorr
# ---------------------------------------------------------------------------


class TestSpectralPowerXcorr:
    def test_output_shape_1d(self, rng):
        pa = rng.uniform(0, 1, 500)
        pb = rng.uniform(0, 1, 500)
        res = spectral_power_xcorr(pa, pb, max_lag=20)
        assert res["lags"].shape == (41,)
        assert res["xcorr"].shape == (41,)

    def test_output_shape_2d(self, rng):
        pa = rng.uniform(0, 1, (10, 500))
        pb = rng.uniform(0, 1, (10, 500))
        res = spectral_power_xcorr(pa, pb, max_lag=10)
        assert res["xcorr"].shape == (10, 21)

    def test_normalized_range(self, rng):
        """Normalized xcorr must be in [-1, 1]."""
        pa = rng.uniform(0, 5, 1000)
        pb = rng.uniform(0, 5, 1000)
        res = spectral_power_xcorr(pa, pb, max_lag=50, normalize=True)
        assert res["xcorr"].max() <= 1.0 + 1e-9
        assert res["xcorr"].min() >= -1.0 - 1e-9

    def test_self_xcorr_peaks_at_zero(self, rng):
        """Self-correlation should peak at lag 0."""
        pa = rng.uniform(0, 1, 300)
        res = spectral_power_xcorr(pa, pa, max_lag=30)
        zero_idx = np.argmin(np.abs(res["lags"]))
        assert res["xcorr"][zero_idx] == pytest.approx(1.0, abs=1e-9)

    def test_known_lag_recovery(self):
        """Shifted signal should produce peak at expected lag.

        pb = base[shift:], so pb[i] = base[i + shift] = pa[i + shift].
        Convention: xcorr[l] = sum(pa[t] * pb[t+l]).
        For pa[t] ≈ pb[t + l], need l = -shift (pb leads → negative lag).
        """
        rng = np.random.default_rng(0)
        n = 2000
        shift = 10
        base = rng.uniform(0, 1, n + shift)
        pa = base[:n]
        pb = base[shift:]  # pb leads pa by `shift` samples
        res = spectral_power_xcorr(pa, pb, max_lag=30, log_transform=False)
        # pb leads pa → peak at negative lag = -shift
        peak_lag = res["lags"][np.argmax(res["xcorr"])]
        assert peak_lag == -shift

    def test_mismatched_lengths_raises(self, rng):
        pa = rng.uniform(0, 1, 100)
        pb = rng.uniform(0, 1, 200)
        with pytest.raises(ValueError, match="time length"):
            spectral_power_xcorr(pa, pb, max_lag=10)
