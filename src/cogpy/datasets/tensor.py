import numpy as np
import pandas as pd
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from ..burst.blob_detection import detect_hmaxima


def make_dataset(
    *,
    duration: float = 10.0,
    nt: int = 50,
    f_min: float = 1.0,
    f_max: float = 40.0,
    nf: int = 25,
    ml_range: tuple[float, float] = (0.0, 1.0),
    nml: int = 10,
    ap_range: tuple[float, float] = (0.0, 1.0),
    nap: int = 10,
    scale: float = 100.0,
    n_blobs: int = 3,
    blob_sigma_idx: tuple[float, float, float, float] = (0.8, 0.8, 2.0, 0.8),
    blob_amp: float | None = None,
    seed: int | None = None,
) -> xr.DataArray:
    """
    Create a small deterministic 4D toy tensor for demos/tests.

    The returned DataArray uses dims ``(ml, ap, time, freq)`` and smooth
    separable structure so slicing/visualization looks reasonable.

    Parameters
    ----------
    duration
        Total time span (seconds) for the ``time`` coordinate.
    nt
        Number of time samples.
    f_min, f_max
        Frequency range (Hz) for the ``freq`` coordinate.
    nf
        Number of frequency samples.
    ml_range, nml
        Range and number of samples for the ``ml`` coordinate.
    ap_range, nap
        Range and number of samples for the ``ap`` coordinate.
    scale
        Overall amplitude scale.
    n_blobs
        Number of compact 4D Gaussian blobs to add (small relative to tensor size).
    blob_sigma_idx
        Blob width (σ) in **index units** for (ml, ap, time, freq). Keep these
        small (≈ 1–2) to ensure compact blobs.
    blob_amp
        Peak amplitude of each blob. Defaults to ``3 * scale``.
    seed
        Optional RNG seed for a small noise term (kept very small so the main
        structure dominates).
    """
    if duration <= 0:
        raise ValueError("duration must be > 0")
    if nt < 2:
        raise ValueError("nt must be >= 2")
    if nf < 2:
        raise ValueError("nf must be >= 2")
    if nml < 2:
        raise ValueError("nml must be >= 2")
    if nap < 2:
        raise ValueError("nap must be >= 2")
    if f_min <= 0:
        raise ValueError("f_min must be > 0")
    if f_max <= f_min:
        raise ValueError("f_max must be > f_min")
    if n_blobs < 0:
        raise ValueError("n_blobs must be >= 0")

    t_vals = np.linspace(0.0, float(duration), int(nt))
    f_vals = np.linspace(float(f_min), float(f_max), int(nf))
    ml_vals = np.linspace(float(ml_range[0]), float(ml_range[1]), int(nml))
    ap_vals = np.linspace(float(ap_range[0]), float(ap_range[1]), int(nap))

    # A smooth positive background so log transforms are well-behaved.
    arr_spectrogram = (0.25 * scale) * (
        1.0 + np.outer(np.cos(2 * np.pi * t_vals / max(duration, 1e-9)), np.sin(f_vals))
    )  # (time, freq)
    arr_specmap = (0.25 * scale) * (
        1.0 + np.outer(np.sin(np.pi * ml_vals), np.cos(np.pi * ap_vals))
    )  # (ml, ap)

    arr = (
        arr_specmap[:, :, np.newaxis, np.newaxis]
        * arr_spectrogram[np.newaxis, np.newaxis, :, :]
    )

    # Add compact 4D Gaussian blobs (small in index space).
    if blob_amp is None:
        blob_amp = 3.0 * scale
    sigma_ml, sigma_ap, sigma_t, sigma_f = map(float, blob_sigma_idx)
    grid_ml, grid_ap, grid_t, grid_f = np.meshgrid(
        np.arange(nml),
        np.arange(nap),
        np.arange(nt),
        np.arange(nf),
        indexing="ij",
        sparse=False,
    )
    rng = np.random.default_rng(0 if seed is None else int(seed))
    for _ in range(int(n_blobs)):
        c_ml = rng.integers(0, nml)
        c_ap = rng.integers(0, nap)
        c_t = rng.integers(0, nt)
        c_f = rng.integers(0, nf)
        blob = np.exp(
            -(
                ((grid_ml - c_ml) ** 2) / (2 * sigma_ml**2)
                + ((grid_ap - c_ap) ** 2) / (2 * sigma_ap**2)
                + ((grid_t - c_t) ** 2) / (2 * sigma_t**2)
                + ((grid_f - c_f) ** 2) / (2 * sigma_f**2)
            )
        )
        arr = arr + float(blob_amp) * blob

    # Optional small noise (kept small so blobs remain compact & salient).
    if seed is not None:
        arr = arr + (scale * 1e-3) * rng.standard_normal(size=arr.shape)

    da = xr.DataArray(
        arr,
        dims=("ml", "ap", "time", "freq"),
        coords={"ml": ml_vals, "ap": ap_vals, "time": t_vals, "freq": f_vals},
        name="val",
    )
    return da


def make_flat_blob_dataset(
    *,
    duration: float = 2.0,
    nt: int = 80,
    f_min: float = 1.0,
    f_max: float = 40.0,
    nf: int = 60,
    ml_range: tuple[float, float] = (0.0, 1.0),
    nml: int = 10,
    ap_range: tuple[float, float] = (0.0, 1.0),
    nap: int = 10,
    background: float = 0.0,
    noise_scale: float = 0.0,
    n_peaks: int = 5,
    peak_dist: str = "uniform",
    peak_loc: tuple[float, float, float, float] | None = None,
    peak_scale: tuple[float, float, float, float] | None = None,
    blob_sigma_idx: tuple[float, float, float, float] = (0.8, 0.8, 2.0, 0.8),
    blob_amp: float = 1.0,
    seed: int | None = None,
    return_peaks: bool = False,
) -> xr.DataArray | tuple[xr.DataArray, pd.DataFrame]:
    """
    Create a flat-background tensor with compact 4D Gaussian blob(s).

    This is useful when you want the signal to be "mostly empty" except for
    localized peaks, with peak coordinates drawn from a simple distribution.

    Parameters
    ----------
    peak_dist
        How to draw peak coordinates (in index space):
        - ``"uniform"``: each dim index is sampled uniformly
        - ``"normal"``: each dim index is sampled from a normal distribution
          with ``peak_loc`` and ``peak_scale`` (in *index* units), then clipped.
    peak_loc, peak_scale
        Only used when ``peak_dist="normal"``. Each is a 4-tuple giving
        (ml, ap, time, freq) mean/std in index units. If omitted, defaults to
        centered means and moderate std.
    blob_sigma_idx
        Blob width (σ) in **index units** for (ml, ap, time, freq).
    return_peaks
        If True, also returns a DataFrame listing the sampled peak coordinates.
    """
    if duration <= 0:
        raise ValueError("duration must be > 0")
    if nt < 2 or nf < 2 or nml < 2 or nap < 2:
        raise ValueError("nt/nf/nml/nap must be >= 2")
    if f_min <= 0:
        raise ValueError("f_min must be > 0")
    if f_max <= f_min:
        raise ValueError("f_max must be > f_min")
    if n_peaks < 0:
        raise ValueError("n_peaks must be >= 0")
    if peak_dist not in {"uniform", "normal"}:
        raise ValueError("peak_dist must be 'uniform' or 'normal'")

    rng = np.random.default_rng(0 if seed is None else int(seed))

    t_vals = np.linspace(0.0, float(duration), int(nt))
    f_vals = np.linspace(float(f_min), float(f_max), int(nf))
    ml_vals = np.linspace(float(ml_range[0]), float(ml_range[1]), int(nml))
    ap_vals = np.linspace(float(ap_range[0]), float(ap_range[1]), int(nap))

    arr = np.full((nml, nap, nt, nf), float(background), dtype=float)

    sigma_ml, sigma_ap, sigma_t, sigma_f = map(float, blob_sigma_idx)
    grid_ml, grid_ap, grid_t, grid_f = np.meshgrid(
        np.arange(nml),
        np.arange(nap),
        np.arange(nt),
        np.arange(nf),
        indexing="ij",
        sparse=False,
    )

    if peak_dist == "normal":
        if peak_loc is None:
            peak_loc = ((nml - 1) / 2, (nap - 1) / 2, (nt - 1) / 2, (nf - 1) / 2)
        if peak_scale is None:
            peak_scale = (nml / 6, nap / 6, nt / 10, nf / 6)

    peaks = []
    for peak_id in range(int(n_peaks)):
        if peak_dist == "uniform":
            c_ml = int(rng.integers(0, nml))
            c_ap = int(rng.integers(0, nap))
            c_t = int(rng.integers(0, nt))
            c_f = int(rng.integers(0, nf))
        else:
            c_ml = int(np.clip(rng.normal(peak_loc[0], peak_scale[0]), 0, nml - 1))
            c_ap = int(np.clip(rng.normal(peak_loc[1], peak_scale[1]), 0, nap - 1))
            c_t = int(np.clip(rng.normal(peak_loc[2], peak_scale[2]), 0, nt - 1))
            c_f = int(np.clip(rng.normal(peak_loc[3], peak_scale[3]), 0, nf - 1))

        blob = np.exp(
            -(
                ((grid_ml - c_ml) ** 2) / (2 * sigma_ml**2)
                + ((grid_ap - c_ap) ** 2) / (2 * sigma_ap**2)
                + ((grid_t - c_t) ** 2) / (2 * sigma_t**2)
                + ((grid_f - c_f) ** 2) / (2 * sigma_f**2)
            )
        )
        amp = float(blob_amp)
        arr = arr + amp * blob

        peaks.append(
            {
                "peak_id": peak_id,
                "i_ml": c_ml,
                "i_ap": c_ap,
                "i_time": c_t,
                "i_freq": c_f,
                "ml": float(ml_vals[c_ml]),
                "ap": float(ap_vals[c_ap]),
                "time": float(t_vals[c_t]),
                "freq": float(f_vals[c_f]),
                "amp": amp,
            }
        )

    if noise_scale and float(noise_scale) != 0.0:
        arr = arr + float(noise_scale) * rng.standard_normal(size=arr.shape)

    da = xr.DataArray(
        arr,
        dims=("ml", "ap", "time", "freq"),
        coords={"ml": ml_vals, "ap": ap_vals, "time": t_vals, "freq": f_vals},
        name="val",
    )
    if return_peaks:
        return da, pd.DataFrame(peaks)
    return da


def detect_bursts_hmaxima(
    datax: xr.DataArray,
    *,
    h_quantile: float = 0.8,
    h: float | None = None,
    footprint=None,
) -> pd.DataFrame:
    """
    Detect 4D burst peaks on the *raw* tensor using an h-maxima transform.

    This is a thin wrapper around :func:`cogpy.burst.blob_detection.detect_hmaxima`
    that returns a minimal, orthoslicer-friendly table.

    Returns
    -------
    pd.DataFrame
        Columns: ``burst_id, x, y, t, z, value`` where:
        ``x=ml``, ``y=ap``, ``t=time``, ``z=freq``, and ``value=amp``.
    """
    bursts_raw = detect_hmaxima(datax, h_quantile=h_quantile, h=h, footprint=footprint)
    if bursts_raw is None or len(bursts_raw) == 0:
        return pd.DataFrame(columns=["burst_id", "x", "y", "t", "z", "value"])

    required = {"ml", "ap", "time", "freq", "amp"}
    missing = sorted(required.difference(bursts_raw.columns))
    if missing:
        raise ValueError(f"detect_hmaxima output missing columns: {missing}")

    bursts = pd.DataFrame(
        {
            "burst_id": np.arange(len(bursts_raw), dtype=int),
            "x": bursts_raw["ml"].astype(float).to_numpy(),
            "y": bursts_raw["ap"].astype(float).to_numpy(),
            "t": bursts_raw["time"].astype(float).to_numpy(),
            "z": bursts_raw["freq"].astype(float).to_numpy(),
            "value": bursts_raw["amp"].astype(float).to_numpy(),
        }
    )
    return bursts


class TensorExample:
    """
    Convenience namespace for a toy tensor + detected bursts table.

    Attributes
    ----------
    data
        4D DataArray with dims ``(ml, ap, time, freq)``.
    bursts
        DataFrame with columns ``burst_id, x, y, t, z, value`` (from h-maxima).
    """

    def __init__(self, data: xr.DataArray, bursts: pd.DataFrame):
        self.data = data
        self.bursts = bursts

    @classmethod
    def make(
        cls,
        *,
        duration: float = 10.0,
        nt: int = 50,
        h_quantile: float = 0.8,
        h: float | None = None,
        footprint=None,
        **dataset_kwargs,
    ) -> "TensorExample":
        """
        Build a toy dataset via :func:`make_dataset` and detect bursts via h-maxima.
        """
        data = make_dataset(duration=duration, nt=nt, **dataset_kwargs)
        bursts = detect_bursts_hmaxima(data, h_quantile=h_quantile, h=h, footprint=footprint)
        return cls(data=data, bursts=bursts)


class AROscillatorGrid:
    """
    Suite for simulating a 2D grid of channels with oscillatory bursts.

    The raw signal is a 3D tensor with dims ``(ap, ml, time)``. Bursts are
    defined by *4D peak parameters* ``(x, y, t, z)`` where ``z`` is the peak
    frequency (Hz). Each burst is generated by:

    1) drawing a (x, y, t, freq) peak,
    2) generating an AR oscillator at ``freq`` (see :func:`emd.simulate.ar_oscillator`),
    3) applying a short temporal Gaussian envelope around ``t``,
    4) multiplying by a compact spatial Gaussian bump over (ap, ml),
    5) adding into the background 3D signal.

    A 4D time–frequency tensor can then be computed using
    :func:`cogpy.spectral.multitaper.mtm_spectrogramx`.

    Attributes
    ----------
    raw
        3D DataArray with dims ``(ap, ml, time)``.
    spectrogram
        4D DataArray with dims ``(ml, ap, time, freq)`` (computed from ``raw``),
        suitable for orthoslicers (map ``ml->x``, ``ap->y``, ``time->t``, ``freq->z``).
    bursts
        DataFrame with columns ``burst_id, x, y, t, z, value`` describing the
        ground-truth burst peaks used to generate the signal.
    """

    def __init__(self, *, raw: xr.DataArray, spectrogram: xr.DataArray, bursts: pd.DataFrame, fs: float):
        self.raw = raw
        self.spectrogram = spectrogram
        self.bursts = bursts
        self.fs = float(fs)

    @staticmethod
    def _gaussian_bump_2d(ny: int, nx: int, cy: float, cx: float, sy: float, sx: float) -> np.ndarray:
        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        return np.exp(-(((yy - cy) ** 2) / (2 * sy**2) + ((xx - cx) ** 2) / (2 * sx**2)))

    @classmethod
    def make(
        cls,
        *,
        duration: float = 10.0,
        fs: float = 1250.0,
        nap: int = 16,
        nml: int = 16,
        ap_range: tuple[float, float] = (0.0, 1.0),
        ml_range: tuple[float, float] = (0.0, 1.0),
        n_bursts: int = 10,
        f_min: float = 4.0,
        f_max: float = 30.0,
        burst_amp: float = 1.0,
        burst_seconds: float = 0.6,
        burst_time_sigma_s: float | None = None,
        spatial_sigma_idx: tuple[float, float] = (0.9, 0.9),  # (ap, ml) in index units
        ar_r: float = 0.98,
        background_noise: float = 0.05,
        spectrogram_bandwidth: float = 4.0,
        spectrogram_nperseg: int | None = None,
        spectrogram_noverlap: int | None = None,
        seed: int | None = 0,
    ) -> "AROscillatorGrid":
        if duration <= 0:
            raise ValueError("duration must be > 0")
        if fs <= 0:
            raise ValueError("fs must be > 0")
        if nap < 1 or nml < 1:
            raise ValueError("nap/nml must be >= 1")
        if n_bursts < 0:
            raise ValueError("n_bursts must be >= 0")
        if f_min <= 0 or f_max <= f_min:
            raise ValueError("invalid frequency range")
        if burst_seconds <= 0:
            raise ValueError("burst_seconds must be > 0")

        rng = np.random.default_rng(0 if seed is None else int(seed))

        nt = int(round(duration * fs))
        t_vals = np.arange(nt, dtype=float) / float(fs)
        ap_vals = np.linspace(float(ap_range[0]), float(ap_range[1]), int(nap))
        ml_vals = np.linspace(float(ml_range[0]), float(ml_range[1]), int(nml))

        # background
        raw = background_noise * rng.standard_normal(size=(nap, nml, nt))

        # burst peak table (ground truth)
        peaks = []
        for burst_id in range(int(n_bursts)):
            i_ap = int(rng.integers(0, nap))
            i_ml = int(rng.integers(0, nml))
            i_t = int(rng.integers(0, nt))
            f0 = float(rng.uniform(float(f_min), float(f_max)))

            peaks.append(
                {
                    "burst_id": int(burst_id),
                    "x": float(ml_vals[i_ml]),
                    "y": float(ap_vals[i_ap]),
                    "t": float(t_vals[i_t]),
                    "z": float(f0),
                    "value": float(burst_amp),
                    "i_ml": i_ml,
                    "i_ap": i_ap,
                    "i_time": i_t,
                }
            )

        if burst_time_sigma_s is None:
            burst_time_sigma_s = float(burst_seconds) / 6.0

        # add bursts
        from emd.simulate import ar_oscillator

        n_burst = int(round(float(burst_seconds) * float(fs)))
        n_burst = max(8, n_burst)
        tb = (np.arange(n_burst, dtype=float) / float(fs))
        env = np.exp(-0.5 * ((tb - tb.mean()) / float(burst_time_sigma_s)) ** 2)
        env = env / (env.max() if env.max() != 0 else 1.0)

        sy, sx = map(float, spatial_sigma_idx)

        for p in peaks:
            i_ap = int(p["i_ap"])
            i_ml = int(p["i_ml"])
            i_t = int(p["i_time"])
            f0 = float(p["z"])

            # temporal waveform (localized AR oscillation)
            w = np.asarray(ar_oscillator(f0, float(fs), tb[-1] + (1.0 / fs), r=float(ar_r), random_seed=int(rng.integers(0, 2**31 - 1)))).reshape(-1)
            w = w[:n_burst] * env
            # normalize then scale
            sd = float(np.std(w)) if np.std(w) != 0 else 1.0
            w = (w / sd) * float(burst_amp)

            # spatial bump (compact)
            bump = cls._gaussian_bump_2d(nap, nml, float(i_ap), float(i_ml), float(sy), float(sx))

            # insert into raw around i_t
            j0 = int(i_t - n_burst // 2)
            j1 = j0 + n_burst
            src0 = 0
            src1 = n_burst
            if j0 < 0:
                src0 = -j0
                j0 = 0
            if j1 > nt:
                src1 = n_burst - (j1 - nt)
                j1 = nt
            if j1 <= j0 or src1 <= src0:
                continue
            raw[:, :, j0:j1] += bump[:, :, None] * w[src0:src1][None, None, :]

        raw_da = xr.DataArray(
            raw,
            dims=("ap", "ml", "time"),
            coords={"ap": ap_vals, "ml": ml_vals, "time": t_vals},
            name="val",
        )

        # 4D spectrogram (freq,time added)
        from ..spectral.multitaper import mtm_spectrogramx

        spec_kwargs = dict(fs=float(fs), bandwidth=float(spectrogram_bandwidth))
        if spectrogram_nperseg is not None:
            spec_kwargs["nperseg"] = int(spectrogram_nperseg)
        if spectrogram_noverlap is not None:
            spec_kwargs["noverlap"] = int(spectrogram_noverlap)

        spec = mtm_spectrogramx(raw_da, dim="time", **spec_kwargs)  # dims: (ap, ml, freq, time)
        # Reorder for consistency with other toy tensors: (ml, ap, time, freq)
        spec = spec.transpose("ml", "ap", "time", "freq")
        spec.name = "val"
        # mtm_spectrogramx uses dask internally; for interactive plotting demos
        # we want an in-memory array to keep downstream code simple.
        if hasattr(spec.data, "compute"):
            spec = spec.compute()

        bursts_df = pd.DataFrame(peaks)[["burst_id", "x", "y", "t", "z", "value"]].copy()

        return cls(raw=raw_da, spectrogram=spec, bursts=bursts_df, fs=float(fs))


# ---------------------------------------------------------------------------
# TensorScope demo bundle helper
# ---------------------------------------------------------------------------

def make_tensorscope_demo_bundle(
    *,
    duration: float = 6.0,
    fs: float = 200.0,
    nap: int = 8,
    nml: int = 8,
    n_bursts: int = 6,
    f_min: float = 6.0,
    f_max: float = 24.0,
    burst_amp: float = 1.5,
    background_noise: float = 0.025,
    seed: int = 0,
) -> dict:
    """Build a complete, deterministic TensorScope demo bundle from one simulation.

    All outputs share the same underlying ``AROscillatorGrid`` simulation so
    that signal, spectrogram, events, and brainstates have a consistent timebase
    and ground truth.

    Parameters
    ----------
    duration, fs, nap, nml, n_bursts, f_min, f_max, burst_amp, background_noise, seed
        Forwarded to :meth:`AROscillatorGrid.make`.

    Returns
    -------
    dict with keys:

    ``"signal"``
        :class:`xarray.DataArray` with dims ``(time, AP, ML)``.
    ``"spectrogram"``
        :class:`xarray.DataArray` with dims ``(time, freq, AP, ML)``.
    ``"events"``
        :class:`pandas.DataFrame` with columns ``event_id, t, AP, ML,
        freq, amplitude, label``.
    ``"brainstates"``
        :class:`xarray.DataArray` with dim ``(time,)``, integer state codes
        (0=theta, 1=alpha, 2=beta) derived deterministically from the
        spectrogram.  The ``state_names`` attr maps code → label.
    ``"meta"``
        Plain dict with provenance metadata (seed, fs, shapes, …).
    """
    grid = AROscillatorGrid.make(
        duration=duration,
        fs=fs,
        nap=nap,
        nml=nml,
        n_bursts=n_bursts,
        f_min=f_min,
        f_max=f_max,
        burst_amp=burst_amp,
        background_noise=background_noise,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # signal: (ap, ml, time) → rename → (time, AP, ML)
    # ------------------------------------------------------------------
    signal = grid.raw.rename({"ap": "AP", "ml": "ML"}).transpose("time", "AP", "ML")
    signal.name = "signal"
    signal.attrs.update(
        {
            "units": "a.u.",
            "source": "cogpy.datasets.tensor.AROscillatorGrid.make",
            "description": "Deterministic demo LFP grid for TensorScope UI development",
            "fs": fs,
            "seed": seed,
        }
    )

    # ------------------------------------------------------------------
    # spectrogram: (ml, ap, time, freq) → rename → (time, freq, AP, ML)
    # ------------------------------------------------------------------
    spectrogram = (
        grid.spectrogram
        .rename({"ap": "AP", "ml": "ML"})
        .transpose("time", "freq", "AP", "ML")
    )
    spectrogram.name = "spectrogram"
    spectrogram.attrs.update(
        {
            "units": "power",
            "source": "cogpy.datasets.tensor.AROscillatorGrid.make",
            "description": "Multitaper spectrogram of demo LFP grid",
            "fs": fs,
            "seed": seed,
        }
    )

    # ------------------------------------------------------------------
    # events: map cogpy burst columns → TensorScope field names
    #   x=ml → ML, y=ap → AP, z=freq → freq, t=t, value=amplitude
    # ------------------------------------------------------------------
    bursts = grid.bursts.copy()
    events = pd.DataFrame(
        {
            "event_id": bursts["burst_id"].astype(int).values,
            "t": bursts["t"].astype(float).values,
            "AP": bursts["y"].astype(float).values,
            "ML": bursts["x"].astype(float).values,
            "freq": bursts["z"].astype(float).values,
            "amplitude": bursts["value"].astype(float).values,
            "label": "burst",
        }
    ).sort_values("t").reset_index(drop=True)

    # ------------------------------------------------------------------
    # brainstates: dominant spectral band per spectrogram time step
    #   theta=0 (4–8 Hz), alpha=1 (8–13 Hz), beta=2 (13–30 Hz)
    # ------------------------------------------------------------------
    _bands = [("theta", 4.0, 8.0), ("alpha", 8.0, 13.0), ("beta", 13.0, 30.0)]
    freq_vals = spectrogram.coords["freq"].values

    # Mean power over space → (time, freq)
    spec_mean = spectrogram.mean(dim=["AP", "ML"])

    band_powers = []
    for _bname, lo, hi in _bands:
        mask = (freq_vals >= lo) & (freq_vals <= hi)
        if mask.any():
            band_powers.append(spec_mean.isel(freq=mask).mean(dim="freq").values)
        else:
            # Band not covered by this spectrogram — assign zero power
            band_powers.append(np.zeros(int(spectrogram.sizes["time"]), dtype=float))

    # Shape (time, n_bands) → dominant band index per time step
    band_matrix = np.stack(band_powers, axis=1)
    state_codes = np.argmax(band_matrix, axis=1).astype(np.int8)
    state_names = [b[0] for b in _bands]

    brainstates = xr.DataArray(
        state_codes,
        dims=("time",),
        coords={"time": spectrogram.coords["time"].values},
        name="brainstate",
        attrs={
            "description": "Dominant spectral band per time step (0=theta,1=alpha,2=beta)",
            # Stored as comma-joined string for NetCDF compatibility; split on ',' to decode.
            "state_names": ",".join(state_names),
            "source": "cogpy.datasets.tensor.make_tensorscope_demo_bundle",
            "seed": seed,
        },
    )

    # ------------------------------------------------------------------
    # meta
    # ------------------------------------------------------------------
    meta: dict = {
        "seed": seed,
        "fs": fs,
        "duration": duration,
        "nap": nap,
        "nml": nml,
        "n_bursts": n_bursts,
        "f_min": f_min,
        "f_max": f_max,
        "burst_amp": burst_amp,
        "background_noise": background_noise,
        "source": "cogpy.datasets.tensor.AROscillatorGrid.make + make_tensorscope_demo_bundle",
        "files": {
            "signal.nc": {
                "dims": list(signal.dims),
                "shape": list(signal.shape),
                "description": "Raw LFP grid",
            },
            "spectrogram.nc": {
                "dims": list(spectrogram.dims),
                "shape": list(spectrogram.shape),
                "description": "Multitaper spectrogram",
            },
            "events.parquet": {
                "columns": list(events.columns),
                "n_events": len(events),
                "description": "Ground-truth burst events",
            },
            "brainstates.nc": {
                "dims": list(brainstates.dims),
                "shape": list(brainstates.shape),
                "state_names": state_names,  # list form preserved in manifest
                "description": "Dominant spectral band per time step",
            },
        },
    }

    return {
        "signal": signal,
        "spectrogram": spectrogram,
        "events": events,
        "brainstates": brainstates,
        "meta": meta,
    }


def example_smooth_multichannel_sigx(nchannel=16, ntime=1000_000, fs=1000):
    sig = xr.DataArray(
        np.random.randn(nchannel, ntime),  # (channel, time)
        dims=["channel", "time"],
        coords={
            "channel": [f"ch{i}" for i in range(nchannel)],
            "time": np.arange(ntime) / fs,  # 1000 Hz sampling rate → seconds
        },
        name="Example iEEG signal",
    )
    # apply smoothing
    sig_smooth = xr.apply_ufunc(
        lambda x: gaussian_filter1d(x, sigma=5),  # Adjust sigma for desired smoothing
        sig,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
    )
    return sig_smooth

def example_ieeg(nrows=16, ncols=16, ntime=1000_000):
    nchannels = nrows * ncols
    sig = example_smooth_multichannel_sigx(nchannel=nchannels, ntime=ntime).transpose("time", "channel")
    # reshape into 16, 16 and add AP, ML 
    ap_vals = np.linspace(-4, 1, nrows)
    ml_vals = np.linspace(-4, 4, ncols)
    sig_reshaped = xr.DataArray(
        sig.data.reshape(ntime, ncols, nrows),  # (time, ml, ap) column major
        dims=["time", "ML", "AP"],
        coords={"time": sig.coords["time"], "AP": ap_vals, "ML": ml_vals},
        name="Example iEEG signal",
    )
    return sig_reshaped
