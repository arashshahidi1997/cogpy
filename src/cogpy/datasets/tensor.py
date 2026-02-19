import numpy as np
import pandas as pd
import xarray as xr

from ..core.burst.blob_detection import detect_hmaxima


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

    This is a thin wrapper around :func:`cogpy.core.burst.blob_detection.detect_hmaxima`
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
