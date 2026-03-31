"""Tests for cogpy.measures.pac — PAC modulation index, preferred phase, comodulogram, surrogates."""

import numpy as np
import pytest

from cogpy.measures.pac import (
    modulation_index,
    preferred_phase,
    comodulogram,
    surrogate_pac,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def uniform_phase(rng):
    """Random phase uniformly distributed in [-pi, pi]."""
    return rng.uniform(-np.pi, np.pi, 5000)


@pytest.fixture
def constant_amplitude():
    """Constant amplitude — no coupling."""
    return np.ones(5000)


@pytest.fixture
def coupled_signals():
    """Synthetic signal with known PAC: amplitude peaks at phase=0.

    Low-freq phase at 2 Hz, high-freq amplitude modulated by phase.
    """
    fs = 1000.0
    t = np.arange(0, 5, 1 / fs)  # 5 seconds
    phase = 2 * np.pi * 2.0 * t  # 2 Hz phase signal
    phase_wrapped = (phase + np.pi) % (2 * np.pi) - np.pi

    # Amplitude envelope: 1 + cos(phase) → peaks at phase=0
    amplitude = 1.0 + np.cos(phase_wrapped)
    return phase_wrapped, amplitude


# ---------------------------------------------------------------------------
# modulation_index
# ---------------------------------------------------------------------------


class TestModulationIndex:
    def test_no_coupling_tort(self, uniform_phase, constant_amplitude):
        """Constant amplitude over uniform phase → MI ≈ 0."""
        mi = modulation_index(uniform_phase, constant_amplitude, method="tort")
        assert mi < 0.01

    def test_strong_coupling_tort(self, coupled_signals):
        phase, amp = coupled_signals
        mi = modulation_index(phase, amp, method="tort")
        assert mi > 0.1

    def test_no_coupling_ozkurt(self, uniform_phase, constant_amplitude):
        mi = modulation_index(uniform_phase, constant_amplitude, method="ozkurt")
        assert mi < 0.05

    def test_strong_coupling_ozkurt(self, coupled_signals):
        phase, amp = coupled_signals
        mi = modulation_index(phase, amp, method="ozkurt")
        assert mi > 0.01

    def test_no_coupling_canolty(self, uniform_phase, constant_amplitude):
        mi = modulation_index(uniform_phase, constant_amplitude, method="canolty")
        assert mi < 0.05

    def test_strong_coupling_canolty(self, coupled_signals):
        phase, amp = coupled_signals
        mi = modulation_index(phase, amp, method="canolty")
        assert mi > 0.1

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            modulation_index(np.zeros(10), np.zeros(20))

    def test_unknown_method(self, uniform_phase, constant_amplitude):
        with pytest.raises(ValueError, match="Unknown method"):
            modulation_index(uniform_phase, constant_amplitude, method="bogus")

    def test_n_bins_parameter(self, coupled_signals):
        """Different n_bins should produce different (but similar) MI values."""
        phase, amp = coupled_signals
        mi_18 = modulation_index(phase, amp, n_bins=18)
        mi_36 = modulation_index(phase, amp, n_bins=36)
        assert mi_18 > 0.05
        assert mi_36 > 0.05


# ---------------------------------------------------------------------------
# preferred_phase
# ---------------------------------------------------------------------------


class TestPreferredPhase:
    def test_coupled_preferred_angle(self, coupled_signals):
        """Amplitude peaks at phase=0, so preferred angle should be near 0."""
        phase, amp = coupled_signals
        res = preferred_phase(phase, amp)
        assert abs(res["angle"]) < 0.3  # within ~17 degrees of 0

    def test_mvl_strong_coupling(self, coupled_signals):
        phase, amp = coupled_signals
        res = preferred_phase(phase, amp)
        assert res["mvl"] > 0.1

    def test_mvl_no_coupling(self, uniform_phase, constant_amplitude):
        res = preferred_phase(uniform_phase, constant_amplitude)
        assert res["mvl"] < 0.05

    def test_pvalue_present(self, coupled_signals):
        phase, amp = coupled_signals
        res = preferred_phase(phase, amp)
        assert "pvalue" in res
        assert 0 <= res["pvalue"] <= 1.0

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            preferred_phase(np.zeros(10), np.zeros(20))


# ---------------------------------------------------------------------------
# comodulogram
# ---------------------------------------------------------------------------


class TestComodulogram:
    def test_output_shape(self):
        """Check that output dimensions match frequency grid."""
        rng = np.random.default_rng(99)
        fs = 500.0
        signal = rng.standard_normal(int(fs * 4))

        res = comodulogram(
            signal,
            freq_phase_range=(1.0, 5.0),
            freq_amp_range=(10.0, 30.0),
            fs=fs,
        )
        assert res["mi"].ndim == 2
        assert res["mi"].shape[0] == len(res["freq_phase"])
        assert res["mi"].shape[1] == len(res["freq_amp"])

    def test_all_nonnegative(self):
        """Tort MI should be non-negative everywhere."""
        rng = np.random.default_rng(99)
        fs = 500.0
        signal = rng.standard_normal(int(fs * 4))

        res = comodulogram(
            signal,
            freq_phase_range=(1.0, 5.0),
            freq_amp_range=(10.0, 30.0),
            fs=fs,
            method="tort",
        )
        assert np.all(res["mi"] >= 0)

    def test_pac_signal_hotspot(self):
        """Synthetic PAC signal should produce elevated MI at the coupling pair."""
        rng = np.random.default_rng(0)
        fs = 500.0
        t = np.arange(0, 10, 1 / fs)

        # 2 Hz phase modulates 20 Hz amplitude
        phase_2hz = np.sin(2 * np.pi * 2.0 * t)
        amp_mod = (1 + 0.8 * np.sin(2 * np.pi * 2.0 * t))
        hf_carrier = amp_mod * np.sin(2 * np.pi * 20.0 * t)
        signal = phase_2hz + hf_carrier + 0.1 * rng.standard_normal(len(t))

        res = comodulogram(
            signal,
            freq_phase_range=(1.0, 5.0),
            freq_amp_range=(12.0, 30.0),
            fs=fs,
        )
        # The peak MI should be meaningfully above zero
        assert res["mi"].max() > 0.01

    def test_custom_step(self):
        rng = np.random.default_rng(42)
        fs = 500.0
        signal = rng.standard_normal(int(fs * 3))

        res = comodulogram(
            signal,
            freq_phase_range=(1.0, 5.0),
            freq_amp_range=(10.0, 30.0),
            fs=fs,
            freq_phase_step=1.0,
            freq_amp_step=5.0,
        )
        assert len(res["freq_phase"]) >= 1
        assert len(res["freq_amp"]) >= 1


# ---------------------------------------------------------------------------
# surrogate_pac
# ---------------------------------------------------------------------------


class TestSurrogatePAC:
    def test_output_keys(self, coupled_signals):
        phase, amp = coupled_signals
        res = surrogate_pac(phase, amp, n_surrogates=10, seed=0)
        assert "observed" in res
        assert "zscore" in res
        assert "pvalue" in res
        assert "surrogates" in res

    def test_surrogates_shape(self, coupled_signals):
        phase, amp = coupled_signals
        n_surr = 20
        res = surrogate_pac(phase, amp, n_surrogates=n_surr, seed=0)
        assert res["surrogates"].shape == (n_surr,)

    def test_strong_coupling_high_zscore(self):
        """Strong non-stationary PAC should yield positive z-score."""
        rng = np.random.default_rng(123)
        n = 10000
        # Non-stationary: coupling only in first half
        phase = rng.uniform(-np.pi, np.pi, n)
        amp = np.ones(n)
        # First half: amplitude correlated with phase
        half = n // 2
        amp[:half] = 2.0 + 1.5 * np.cos(phase[:half])
        # Second half: no coupling
        amp[half:] = rng.uniform(0.5, 3.5, n - half)
        res = surrogate_pac(phase, amp, n_surrogates=50, seed=0)
        assert res["zscore"] > 1.0

    def test_no_coupling_low_zscore(self, uniform_phase, constant_amplitude):
        """No coupling → z-score near 0."""
        res = surrogate_pac(uniform_phase, constant_amplitude, n_surrogates=50, seed=0)
        assert abs(res["zscore"]) < 3.0

    def test_circular_shift_method(self, coupled_signals):
        phase, amp = coupled_signals
        res = surrogate_pac(phase, amp, n_surrogates=10, surrogate_method="circular_shift", seed=0)
        assert res["surrogates"].shape == (10,)

    def test_swap_phase_method(self, coupled_signals):
        phase, amp = coupled_signals
        res = surrogate_pac(phase, amp, n_surrogates=10, surrogate_method="swap_phase", seed=0)
        assert res["surrogates"].shape == (10,)

    def test_time_block_method(self, coupled_signals):
        phase, amp = coupled_signals
        res = surrogate_pac(phase, amp, n_surrogates=10, surrogate_method="time_block", seed=0)
        assert res["surrogates"].shape == (10,)

    def test_unknown_surrogate_method(self, coupled_signals):
        phase, amp = coupled_signals
        with pytest.raises(ValueError, match="Unknown surrogate_method"):
            surrogate_pac(phase, amp, n_surrogates=5, surrogate_method="bogus")

    def test_pvalue_range(self, coupled_signals):
        phase, amp = coupled_signals
        res = surrogate_pac(phase, amp, n_surrogates=20, seed=0)
        assert 0.0 <= res["pvalue"] <= 1.0

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            surrogate_pac(np.zeros(10), np.zeros(20), n_surrogates=5)

    def test_ozkurt_method(self, coupled_signals):
        phase, amp = coupled_signals
        res = surrogate_pac(phase, amp, n_surrogates=10, method="ozkurt", seed=0)
        assert res["observed"] > 0
