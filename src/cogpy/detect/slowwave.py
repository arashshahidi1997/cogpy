"""Slow-wave / UP-DOWN state detection (v2.6.5).

Bandpass → zero-crossing trough detection → duration & amplitude gating.
Follows the same EventDetector pattern as RippleDetector.

References
----------
- Isomura et al. 2006: DOWN = 200-500 ms silence, UP = 0.3-1 s activity
- Swanson et al. 2025: 25th percentile binarization, DOWN >80 ms
- Battaglia et al. 2004: delta 0.8-4 Hz, SPW coincides with D-U transitions
- yasa.detection.sw_detect: negative half-wave approach
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .base import EventDetector
from .utils import bandpass_filter

__all__ = ["SlowWaveDetector", "gamma_envelope_validator"]


def _find_zero_crossings(x: np.ndarray) -> np.ndarray:
    """Find indices where signal crosses zero (negative-to-positive or vice versa).

    Returns indices i where sign(x[i]) != sign(x[i+1]).
    """
    signs = np.sign(x)
    # Treat exact zeros as positive to avoid ambiguity.
    signs[signs == 0] = 1
    crossings = np.where(np.diff(signs) != 0)[0]
    return crossings


def _detect_slow_waves_1d(
    x: np.ndarray,
    t: np.ndarray,
    *,
    dur_neg_min: float,
    dur_neg_max: float,
    dur_cycle_min: float,
    dur_cycle_max: float,
    amp_ptp_percentile: float,
) -> list[dict[str, Any]]:
    """Detect slow waves on a single 1D filtered signal.

    Uses yasa-style negative half-wave approach:
    1. Find zero crossings
    2. Identify negative half-waves (neg-to-pos crossing → pos-to-neg crossing)
    3. Find negative trough within each half-wave
    4. Find positive peak in the subsequent positive half-wave
    5. Apply duration and amplitude criteria

    Parameters
    ----------
    x : 1D filtered signal
    t : corresponding time array
    dur_neg_min, dur_neg_max : duration bounds for negative half-wave (seconds)
    dur_cycle_min, dur_cycle_max : duration bounds for full cycle (seconds)
    amp_ptp_percentile : percentile threshold for peak-to-trough amplitude
    """
    if x.size < 4:
        return []

    zc = _find_zero_crossings(x)
    if len(zc) < 3:
        return []

    # Classify crossings: negative-going (pos→neg) or positive-going (neg→pos).
    # A neg-going crossing at index i means x[i] >= 0 and x[i+1] < 0.
    neg_going = []  # indices into zc where signal goes positive→negative
    pos_going = []  # indices into zc where signal goes negative→positive
    for idx in zc:
        if x[idx] >= 0 and x[idx + 1] < 0:
            neg_going.append(idx)
        elif x[idx] < 0 and x[idx + 1] >= 0:
            pos_going.append(idx)

    neg_going = np.array(neg_going, dtype=int)
    pos_going = np.array(pos_going, dtype=int)

    if len(neg_going) < 1 or len(pos_going) < 1:
        return []

    # For each negative half-wave: find a neg-going crossing followed by a pos-going crossing.
    # Then find the subsequent pos-going → neg-going as the full cycle end.
    candidates = []
    for ng in neg_going:
        # Find next positive-going crossing after this negative-going one.
        pg_after = pos_going[pos_going > ng]
        if len(pg_after) == 0:
            continue
        mid_crossing = pg_after[0]  # midcrossing = neg→pos zero crossing

        # Negative half-wave: from ng to mid_crossing
        neg_start = ng
        neg_end = mid_crossing

        # Find next negative-going crossing after midcrossing = end of positive half-wave
        ng_after = neg_going[neg_going > mid_crossing]
        if len(ng_after) == 0:
            # Use end of signal as approximate cycle end
            cycle_end = len(x) - 1
        else:
            cycle_end = ng_after[0]

        # Negative trough
        seg_neg = x[neg_start : neg_end + 1]
        if seg_neg.size == 0:
            continue
        trough_local = int(np.argmin(seg_neg))
        trough_idx = neg_start + trough_local

        # Positive peak in positive half-wave
        seg_pos = x[neg_end : cycle_end + 1]
        if seg_pos.size == 0:
            continue
        peak_local = int(np.argmax(seg_pos))
        peak_idx = neg_end + peak_local

        candidates.append(
            {
                "onset_idx": neg_start,
                "trough_idx": trough_idx,
                "mid_idx": mid_crossing,
                "peak_idx": peak_idx,
                "offset_idx": cycle_end,
                "val_trough": float(x[trough_idx]),
                "val_peak": float(x[peak_idx]),
                "ptp": float(x[peak_idx] - x[trough_idx]),
            }
        )

    if not candidates:
        return []

    # Compute PTP amplitude threshold from all candidates.
    ptps = np.array([c["ptp"] for c in candidates])
    ptp_thresh = float(np.percentile(ptps, amp_ptp_percentile)) if len(ptps) > 1 else 0.0

    events = []
    for c in candidates:
        t_onset = float(t[c["onset_idx"]])
        t_trough = float(t[c["trough_idx"]])
        t_mid = float(t[c["mid_idx"]])
        t_peak = float(t[c["peak_idx"]])
        t_offset = float(t[c["offset_idx"]])

        dur_neg = t_mid - t_onset
        dur_cycle = t_offset - t_onset

        # Duration gating
        if dur_neg < dur_neg_min or dur_neg > dur_neg_max:
            continue
        if dur_cycle < dur_cycle_min or dur_cycle > dur_cycle_max:
            continue

        # Amplitude gating
        if c["ptp"] < ptp_thresh:
            continue

        events.append(
            {
                "t0": t_onset,
                "t": t_trough,  # primary time = trough (DOWN state center)
                "t1": t_offset,
                "trough_time": t_trough,
                "midcrossing_time": t_mid,
                "peak_time": t_peak,
                "duration": dur_cycle,
                "duration_neg": dur_neg,
                "amplitude": c["ptp"],
                "val_trough": c["val_trough"],
                "val_peak": c["val_peak"],
                "frequency": 1.0 / dur_cycle if dur_cycle > 0 else np.nan,
                "state": "DOWN",
            }
        )

    return events


class SlowWaveDetector(EventDetector):
    """Slow-wave detector: bandpass → zero-crossing trough detection → gating.

    Produces interval events with columns:
    (t0, t, t1, trough_time, midcrossing_time, peak_time,
     duration, duration_neg, amplitude, val_trough, val_peak, frequency, state)

    Parameters
    ----------
    freq_range : tuple[float, float]
        Bandpass frequency range in Hz. Default (0.5, 4.0) for delta band.
        Use (0.5, 2.0) for strict slow oscillation.
    dur_neg : tuple[float, float]
        Min/max duration of negative half-wave in seconds.
        Default (0.08, 1.0) — DOWN >80 ms per Swanson et al.
    dur_cycle : tuple[float, float]
        Min/max duration of full cycle in seconds. Default (0.3, 1.5).
    amp_ptp_percentile : float
        Only keep events with peak-to-trough amplitude above this percentile
        of all candidate amplitudes. Default 25.0 (per Swanson et al.).
    filter_order : int
        Butterworth filter order. Default 4.
    """

    def __init__(
        self,
        *,
        freq_range: tuple[float, float] = (0.5, 4.0),
        dur_neg: tuple[float, float] = (0.08, 1.0),
        dur_cycle: tuple[float, float] = (0.3, 1.5),
        amp_ptp_percentile: float = 25.0,
        filter_order: int = 4,
    ) -> None:
        super().__init__("SlowWaveDetector")

        self.freq_range = (float(freq_range[0]), float(freq_range[1]))
        self.dur_neg = (float(dur_neg[0]), float(dur_neg[1]))
        self.dur_cycle = (float(dur_cycle[0]), float(dur_cycle[1]))
        self.amp_ptp_percentile = float(amp_ptp_percentile)
        self.filter_order = int(filter_order)

        self.params = {
            "freq_range": self.freq_range,
            "dur_neg": self.dur_neg,
            "dur_cycle": self.dur_cycle,
            "amp_ptp_percentile": self.amp_ptp_percentile,
            "filter_order": self.filter_order,
        }

    def can_accept(self, data: xr.DataArray) -> bool:
        return "time" in data.dims

    def detect(self, data: xr.DataArray, **_kwargs: Any):
        from cogpy.events import EventCatalog

        x_f = bandpass_filter(
            data, self.freq_range[0], self.freq_range[1], order=self.filter_order
        )

        events: list[dict[str, Any]] = []
        if ("AP" in x_f.dims) and ("ML" in x_f.dims):
            for ap_i in range(int(x_f.sizes["AP"])):
                for ml_i in range(int(x_f.sizes["ML"])):
                    ts = x_f.isel(AP=ap_i, ML=ml_i)
                    events.extend(
                        self._detect_1d(ts, AP=int(ap_i), ML=int(ml_i))
                    )
        elif "channel" in x_f.dims:
            for ch_i in range(int(x_f.sizes["channel"])):
                ts = x_f.isel(channel=ch_i)
                events.extend(self._detect_1d(ts, channel=int(ch_i)))
        else:
            events.extend(self._detect_1d(x_f))

        df = (
            pd.DataFrame.from_records(events)
            if events
            else pd.DataFrame(columns=["event_id", "t", "t0", "t1"])
        )
        if not df.empty:
            df = df.sort_values("t").reset_index(drop=True)
            df["event_id"] = [f"sw_{i:06d}" for i in range(len(df))]
            df["label"] = "slow_wave"

        return EventCatalog(
            df=df,
            name="slow_wave_events",
            metadata={"detector": self.name, **self.params},
        )

    def _detect_1d(
        self, ts: xr.DataArray, **loc: Any
    ) -> list[dict[str, Any]]:
        if "time" not in ts.dims:
            return []

        t = np.asarray(ts["time"].values, dtype=float)
        y = ts.data
        try:
            y = y.compute()
        except Exception:  # noqa: BLE001
            pass
        y = np.asarray(y, dtype=float).reshape(-1)

        if t.size != y.size or t.size < 4:
            return []

        evs = _detect_slow_waves_1d(
            y,
            t,
            dur_neg_min=self.dur_neg[0],
            dur_neg_max=self.dur_neg[1],
            dur_cycle_min=self.dur_cycle[0],
            dur_cycle_max=self.dur_cycle[1],
            amp_ptp_percentile=self.amp_ptp_percentile,
        )
        for ev in evs:
            ev.update(loc)
        return evs

    def get_event_dims(self) -> list[str]:
        return ["time"]

    def get_transform_info(self) -> dict:
        return {
            "required": True,
            "transform_type": "BandpassZeroCrossing",
            "params": {
                "freq_range": self.freq_range,
                "filter_order": self.filter_order,
                "dur_neg": self.dur_neg,
                "dur_cycle": self.dur_cycle,
                "amp_ptp_percentile": self.amp_ptp_percentile,
            },
            "implicit": True,
            "explicit": False,
        }


def gamma_envelope_validator(
    lfp: xr.DataArray,
    events: Any,
    *,
    gamma_band: tuple[float, float] = (30.0, 140.0),
    smooth_ms: float = 50.0,
    filter_order: int = 4,
) -> pd.DataFrame:
    """Validate slow-wave events against broadband gamma envelope.

    DOWN-state troughs should coincide with gamma envelope minima.
    Computes normalized gamma power at each event trough and returns
    a validation score.

    Parameters
    ----------
    lfp : xr.DataArray
        Raw LFP signal with ``time`` dimension.
    events : EventCatalog
        Slow-wave events (must have ``trough_time`` column).
    gamma_band : tuple[float, float]
        Gamma bandpass range. Default (30, 140) Hz for broadband gamma.
    smooth_ms : float
        Gaussian smoothing kernel width in milliseconds for the envelope.
    filter_order : int
        Butterworth filter order for gamma bandpass.

    Returns
    -------
    pd.DataFrame
        Copy of events.df with added columns:
        - ``gamma_at_trough``: z-scored gamma envelope value at trough
        - ``gamma_valid``: True if gamma_at_trough < 0 (trough during gamma minimum)
    """
    from .utils import hilbert_envelope, zscore_1d

    # Compute gamma envelope.
    gamma_filt = bandpass_filter(
        lfp, gamma_band[0], gamma_band[1], order=filter_order
    )
    gamma_env = hilbert_envelope(gamma_filt)

    # Smooth the envelope.
    t_arr = np.asarray(lfp["time"].values, dtype=float)
    dt = float(np.median(np.diff(t_arr))) if t_arr.size > 1 else 1e-3
    smooth_samples = max(1, int(round(smooth_ms / 1000.0 / dt)))
    env_vals = np.asarray(gamma_env.values, dtype=float).reshape(-1)
    if smooth_samples > 1:
        from scipy.ndimage import gaussian_filter1d

        env_vals = gaussian_filter1d(env_vals, sigma=smooth_samples / 2.35)

    # Z-score the envelope.
    env_z = zscore_1d(env_vals)

    df = events.df.copy()
    trough_col = "trough_time" if "trough_time" in df.columns else "t"

    gamma_at_trough = []
    for _, row in df.iterrows():
        tt = float(row[trough_col])
        idx = int(np.argmin(np.abs(t_arr - tt)))
        gamma_at_trough.append(float(env_z[idx]))

    df["gamma_at_trough"] = gamma_at_trough
    df["gamma_valid"] = df["gamma_at_trough"] < 0.0

    return df
