"""
Spectrogram + burst detection workflow (iEEG-specific).

This module provides a higher-level workflow for computing multitaper
spectrograms from memmapped iEEG/LFP data and integrating with burst/blob
detection logic.

Status
------
STATUS: PIPELINE-SPECIFIC
Reason: Higher-level workflow coupled to ieeg_io/memmap. Useful as reference but not a general-purpose utility.
Superseded by: n/a
Safe to remove: no
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr

from cogpy.burst import blob_detection
from cogpy.spectral import multitaper
from cogpy.io import ieeg_io


@dataclass(frozen=True)
class IEEGMeta:
    fs_hz: float
    nch: int
    dtype: np.dtype
    rows: np.ndarray  # (nch,)
    cols: np.ndarray  # (nch,)


def load_ieeg_meta(lfp_path: str | Path) -> IEEGMeta:
    meta = ieeg_io.load_ieeg_metadata(str(lfp_path))
    return IEEGMeta(
        fs_hz=float(meta["fs"]),
        nch=int(meta["nch"]),
        dtype=np.dtype(meta["dtype"]),
        rows=np.asarray(meta["rows"], dtype=np.int64),
        cols=np.asarray(meta["cols"], dtype=np.int64),
    )


def lfp_memmap(lfp_path: str | Path, meta: IEEGMeta) -> np.memmap:
    p = Path(lfp_path)
    n_items = p.stat().st_size // int(meta.dtype.itemsize)
    if meta.nch <= 0:
        raise ValueError(f"Invalid nch={meta.nch}")
    if n_items % meta.nch != 0:
        raise ValueError(
            f"LF P size does not divide evenly by nch: items={n_items}, nch={meta.nch}, dtype={meta.dtype}"
        )
    nsamples = n_items // meta.nch
    return np.memmap(p, dtype=meta.dtype, mode="r", shape=(nsamples, meta.nch), order="C")


def _cfg_float(cfg: dict[str, Any], key: str, default: float) -> float:
    v = cfg.get(key, default)
    return float(v)


def _cfg_int(cfg: dict[str, Any], key: str, default: int) -> int:
    v = cfg.get(key, default)
    return int(v)


def compute_multitaper_spectrogram(
    *,
    lfp_path: str | Path,
    spec_cfg: dict[str, Any],
    max_time_bins: int | None = None,
    max_freq_bins: int | None = None,
) -> xr.Dataset:
    meta = load_ieeg_meta(lfp_path)
    mm = lfp_memmap(lfp_path, meta)

    fs = meta.fs_hz
    nch = meta.nch

    window_duration_s = _cfg_float(spec_cfg, "window_duration_s", 1.0)
    step_duration_s = _cfg_float(spec_cfg, "step_duration_s", 0.05)
    time_bandwidth = _cfg_float(spec_cfg, "time_bandwidth", 3.0)  # NW
    num_tapers = _cfg_int(spec_cfg, "num_tapers", 5)
    remove_mean = bool(spec_cfg.get("remove_mean", True))
    min_lambda = _cfg_float(spec_cfg, "min_lambda", 0.9)

    fmin_hz = _cfg_float(spec_cfg, "fmin_hz", 0.5)
    fmax_hz = _cfg_float(spec_cfg, "fmax_hz", 300.0)
    normalization = str(spec_cfg.get("normalization", "db"))

    window_size = max(1, int(round(window_duration_s * fs)))
    window_step = max(1, int(round(step_duration_s * fs)))
    noverlap = max(0, window_size - window_step)
    bandwidth_hz = (time_bandwidth * fs) / float(window_size)

    # Use the cogpy multitaper wrapper (ghostipy backend) with Dask outer parallelism.
    # Important: keep the full time axis in one chunk for correctness.
    nsamples = int(mm.shape[0])
    ch_chunk = max(1, min(8, nch))
    x_tc = da.from_array(mm, chunks=(nsamples, ch_chunk)).astype(np.float32, copy=False)  # (time, ch)
    x = x_tc.T  # (ch, time)
    try:
        spec, f, t = multitaper.mtm_spectrogram(
            x,
            bandwidth=float(bandwidth_hz),
            fs=float(fs),
            nperseg=int(window_size),
            noverlap=int(noverlap),
            n_tapers=int(num_tapers),
            min_lambda=float(min_lambda),
            remove_mean=bool(remove_mean),
            n_fft_threads=1,
        )
    except ValueError as e:
        # Ghostipy can reject `n_tapers` when too many fail the energy concentration cutoff.
        # Fall back to a permissive cutoff rather than failing the whole pipeline.
        if "n_tapers" not in str(e):
            raise
        spec, f, t = multitaper.mtm_spectrogram(
            x,
            bandwidth=float(bandwidth_hz),
            fs=float(fs),
            nperseg=int(window_size),
            noverlap=int(noverlap),
            n_tapers=int(num_tapers),
            min_lambda=0.0,
            remove_mean=bool(remove_mean),
            n_fft_threads=1,
        )
        min_lambda = 0.0

    f = np.asarray(f, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    fmask = (f >= fmin_hz) & (f <= fmax_hz)
    f = f[fmask]
    spec = spec[:, fmask, :]

    if normalization == "db":
        spec = 10.0 * da.log10(spec.astype(np.float64) + 1e-12)
    elif normalization == "zscore":
        mu = da.nanmean(spec, axis=-1, keepdims=True)
        sd = da.nanstd(spec, axis=-1, keepdims=True)
        spec = (spec - mu) / (sd + 1e-12)
    elif normalization == "raw":
        spec = spec.astype(np.float64)
    else:
        raise ValueError(f"Unknown spectrogram.normalization: {normalization!r}")

    # Optional preview decimation.
    if max_time_bins is not None and t.size > max_time_bins and max_time_bins > 0:
        step = int(math.ceil(t.size / max_time_bins))
        t = t[::step]
        spec = spec[:, :, ::step]
    if max_freq_bins is not None and f.size > max_freq_bins and max_freq_bins > 0:
        step = int(math.ceil(f.size / max_freq_bins))
        f = f[::step]
        spec = spec[:, ::step, :]

    # Reorder to contract dims: (channel, time, freq)
    # (ch, freq, time) -> (ch, time, freq)
    spec_ctf = da.moveaxis(spec, 1, -1).astype(np.float32)  # (ch, time, freq)

    spec_da = xr.DataArray(
        spec_ctf,
        dims=("channel", "time", "freq"),
        coords={
            "channel": np.arange(nch, dtype=np.int32),
            "time": t.astype(np.float64, copy=False),
            "freq": f.astype(np.float64, copy=False),
        },
        name="spec",
    )
    spec_da.attrs["time_semantics"] = "window_center_seconds"

    ds = spec_da.to_dataset()
    ds.attrs.update(
        fs=float(fs),
        engine=str(spec_cfg.get("engine", "ghostipy")),
        time_bandwidth=float(time_bandwidth),
        num_tapers=int(num_tapers),
        min_lambda=float(min_lambda),
        window_duration_s=float(window_duration_s),
        step_duration_s=float(step_duration_s),
        fmin_hz=float(fmin_hz),
        fmax_hz=float(fmax_hz),
        normalization=str(normalization),
        time_semantics="window_center_seconds",
        remove_mean=bool(remove_mean),
    )
    return ds


def detect_blob_candidates(
    *,
    spec_ds: xr.Dataset,
    blob_cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    if "spec" not in spec_ds:
        raise KeyError("spec_ds must contain data var 'spec'")

    min_duration_s = _cfg_float(blob_cfg, "min_duration_s", 0.02)
    max_duration_s = _cfg_float(blob_cfg, "max_duration_s", 2.0)
    min_bandwidth_hz = _cfg_float(blob_cfg, "min_bandwidth_hz", 1.0)
    max_bandwidth_hz = _cfg_float(blob_cfg, "max_bandwidth_hz", 200.0)
    num_sigma = _cfg_int(blob_cfg, "num_sigma", 10)

    # Convert duration/bandwidth bounds to approximate LoG sigma ranges.
    sigma_dict_raw = {
        "time": (min_duration_s / 4.0, max_duration_s / 4.0),
        "freq": (min_bandwidth_hz / 4.0, max_bandwidth_hz / 4.0),
    }

    t_all: list[float] = []
    f_all: list[float] = []
    t_sigma_all: list[float] = []
    f_sigma_all: list[float] = []
    amp_all: list[float] = []
    ch_all: list[int] = []

    spec = spec_ds["spec"]
    nch = int(spec.sizes["channel"])

    for ch in range(nch):
        s = spec.isel(channel=ch).transpose("time", "freq")
        arr = s.data
        try:
            import dask.array as _da  # type: ignore
        except Exception:  # pragma: no cover
            _da = None  # type: ignore
        if _da is not None and isinstance(arr, _da.Array):
            arr = arr.compute()
        arr = np.asarray(arr)
        if arr.size == 0 or not np.isfinite(arr).any():
            continue
        da_tf = xr.DataArray(arr, dims=("time", "freq"), coords={"time": s["time"], "freq": s["freq"]})

        df = blob_detection.detect_blobs(da_tf, num_sigma=int(num_sigma), sigma_dict_raw=sigma_dict_raw)
        if df.empty:
            continue

        for _, row in df.iterrows():
            t_all.append(float(row["time"]))
            f_all.append(float(row["freq"]))
            t_sigma_all.append(float(row["time_sigma"]))
            f_sigma_all.append(float(row["freq_sigma"]))
            amp_all.append(float(row["amp"]))
            ch_all.append(int(ch))

    if not ch_all:
        empty = np.asarray([], dtype=np.float32)
        return dict(
            t0_s=empty,
            t1_s=empty,
            t_peak_s=empty,
            f0_hz=empty,
            f1_hz=empty,
            f_peak_hz=empty,
            channel=np.asarray([], dtype=np.int32),
            score=empty,
        )

    t_peak = np.asarray(t_all, dtype=np.float64)
    f_peak = np.asarray(f_all, dtype=np.float64)
    t_sig = np.asarray(t_sigma_all, dtype=np.float64)
    f_sig = np.asarray(f_sigma_all, dtype=np.float64)
    amp = np.asarray(amp_all, dtype=np.float64)
    ch = np.asarray(ch_all, dtype=np.int32)

    t0 = t_peak - 2.0 * t_sig
    t1 = t_peak + 2.0 * t_sig
    f0 = f_peak - 2.0 * f_sig
    f1 = f_peak + 2.0 * f_sig

    # Filter to configured bounds.
    dur = t1 - t0
    bw = f1 - f0
    ok = (
        np.isfinite(t_peak)
        & np.isfinite(f_peak)
        & (dur >= min_duration_s)
        & (dur <= max_duration_s)
        & (bw >= min_bandwidth_hz)
        & (bw <= max_bandwidth_hz)
    )

    t0 = t0[ok]
    t1 = t1[ok]
    t_peak = t_peak[ok]
    f0 = f0[ok]
    f1 = f1[ok]
    f_peak = f_peak[ok]
    ch = ch[ok]
    amp = amp[ok]

    return dict(
        t0_s=t0.astype(np.float32, copy=False),
        t1_s=t1.astype(np.float32, copy=False),
        t_peak_s=t_peak.astype(np.float32, copy=False),
        f0_hz=f0.astype(np.float32, copy=False),
        f1_hz=f1.astype(np.float32, copy=False),
        f_peak_hz=f_peak.astype(np.float32, copy=False),
        channel=ch.astype(np.int32, copy=False),
        score=amp.astype(np.float32, copy=False),
    )


def _neighbors_within_gap(meta: IEEGMeta, spatial_gap_tolerance: int) -> list[np.ndarray]:
    rows = meta.rows.astype(np.int64, copy=False)
    cols = meta.cols.astype(np.int64, copy=False)
    nch = int(meta.nch)

    pos_to_idx: dict[tuple[int, int], int] = {(int(rows[i]), int(cols[i])): i for i in range(nch)}
    direct: list[list[int]] = [[] for _ in range(nch)]
    for i in range(nch):
        ri = int(rows[i])
        ci = int(cols[i])
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = pos_to_idx.get((ri + dr, ci + dc))
            if j is not None and j != i:
                direct[i].append(int(j))

    max_steps = max(0, int(spatial_gap_tolerance)) + 1
    out: list[np.ndarray] = []
    for i in range(nch):
        seen = {i}
        frontier = {i}
        for _ in range(max_steps):
            nxt: set[int] = set()
            for u in frontier:
                for v in direct[u]:
                    if v not in seen:
                        seen.add(v)
                        nxt.add(v)
            frontier = nxt
            if not frontier:
                break
        seen.remove(i)
        out.append(np.asarray(sorted(seen), dtype=np.int64))
    return out


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def aggregate_bursts(
    *,
    blob_candidates: dict[str, np.ndarray],
    lfp_path: str | Path,
    bursts_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[tuple[str, int]]]:
    meta = load_ieeg_meta(lfp_path)

    max_dt_s = _cfg_float(bursts_cfg, "max_dt_s", 0.05)
    max_df_hz = _cfg_float(bursts_cfg, "max_df_hz", 5.0)
    min_channels = _cfg_int(bursts_cfg, "min_channels", 2)
    spatial_gap_tolerance = _cfg_int(bursts_cfg, "spatial_gap_tolerance", 1)

    t_peak = np.asarray(blob_candidates["t_peak_s"], dtype=np.float64)
    f_peak = np.asarray(blob_candidates["f_peak_hz"], dtype=np.float64)
    ch = np.asarray(blob_candidates["channel"], dtype=np.int64)
    score = np.asarray(blob_candidates.get("score", np.ones_like(t_peak)), dtype=np.float64)
    t0 = np.asarray(blob_candidates["t0_s"], dtype=np.float64)
    t1 = np.asarray(blob_candidates["t1_s"], dtype=np.float64)
    f0 = np.asarray(blob_candidates["f0_hz"], dtype=np.float64)
    f1 = np.asarray(blob_candidates["f1_hz"], dtype=np.float64)

    n = int(t_peak.size)
    if n == 0:
        return [], []

    neighbors = _neighbors_within_gap(meta, spatial_gap_tolerance)
    neighbor_sets = [set(n.tolist()) for n in neighbors]
    uf = _UnionFind(n)

    order = np.argsort(t_peak, kind="mergesort")
    t_sorted = t_peak[order]

    j0 = 0
    for ii in range(n):
        i = int(order[ii])
        while j0 < n and t_sorted[j0] < (t_peak[i] - max_dt_s):
            j0 += 1
        jj = j0
        while jj < n:
            j = int(order[jj])
            if j == i:
                jj += 1
                continue
            dt = abs(t_peak[j] - t_peak[i])
            if dt > max_dt_s:
                if t_peak[j] > t_peak[i]:
                    break
                jj += 1
                continue
            if abs(f_peak[j] - f_peak[i]) > max_df_hz:
                jj += 1
                continue
            ci = int(ch[i])
            cj = int(ch[j])
            if ci != cj and cj not in neighbor_sets[ci]:
                jj += 1
                continue
            uf.union(i, j)
            jj += 1

    groups: dict[int, list[int]] = {}
    for i in range(n):
        r = uf.find(i)
        groups.setdefault(r, []).append(i)

    bursts: list[dict[str, Any]] = []
    memberships: list[tuple[str, int]] = []

    # Stable ordering: sort groups by earliest time, then peak time.
    group_items = []
    for members in groups.values():
        members = sorted(members)
        group_items.append((float(np.min(t0[members])), float(np.min(t_peak[members])), members))
    group_items.sort(key=lambda x: (x[0], x[1]))

    burst_idx = 0
    for _, _, members in group_items:
        chans = sorted(set(int(ch[i]) for i in members))
        if len(chans) < min_channels:
            continue

        # Representative peak: max score, tie-break by earliest peak time.
        best = sorted(members, key=lambda i: (-float(score[i]), float(t_peak[i]), int(ch[i])))[0]

        burst_idx += 1
        burst_id = f"b{burst_idx:06d}"

        bursts.append(
            dict(
                burst_id=burst_id,
                t0_s=float(np.min(t0[members])),
                t1_s=float(np.max(t1[members])),
                t_peak_s=float(t_peak[best]),
                f0_hz=float(np.min(f0[members])),
                f1_hz=float(np.max(f1[members])),
                f_peak_hz=float(f_peak[best]),
                n_channels=int(len(chans)),
                ch_min=int(min(chans)),
                ch_max=int(max(chans)),
            )
        )

        for c in chans:
            memberships.append((burst_id, int(c)))

    return bursts, memberships
