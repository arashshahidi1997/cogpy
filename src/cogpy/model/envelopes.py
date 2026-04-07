import numpy as np


def biexp(t, A=1.0, alpha=1.0, beta=0.5, t0=0.0):
    """
    f(t) = A * [exp(-alpha*(t-t0)) - exp(-beta*(t-t0))] for t >= t0; else 0.
    """
    t = np.asarray(t, dtype=float)
    y = np.zeros_like(t)
    m = t >= t0
    x = t[m] - t0
    y[m] = A * (np.exp(-alpha * x) - np.exp(-beta * x))
    return y


def smoothstep_cutoff(t, t0, T):
    """
    C¹ window w(t) with compact support on [t0, t0+T]:
      w=1 for t<=t0; w=1-3x^2+2x^3 for 0<x<1; w=0 for t>=t0+T,
      where x=(t - t0)/T.
    """
    t = np.asarray(t, dtype=float)
    w = np.ones_like(t)
    x = (t - t0) / T
    m1 = (x >= 0) & (x < 1)
    w[x >= 1] = 0.0
    w[m1] = 1.0 - 3.0 * x[m1] ** 2 + 2.0 * x[m1] ** 3
    return w


def tukey_cutoff(t, t0, T, alpha=0.5):
    """
    Tukey (tapered cosine) window on [t0, t0+T]:
      flat top for the first (1-alpha) fraction, cosine taper to 0 afterward.
    alpha in [0,1]: 0 = rectangular (hard cut), 1 = Hann.
    """
    t = np.asarray(t, dtype=float)
    w = np.zeros_like(t)
    x = (t - t0) / T
    # regions
    m_flat = (x >= 0) & (x <= (1 - alpha) / 2)
    m_taper = (x > (1 - alpha) / 2) & (x < 1)
    w[m_flat] = 1.0
    if alpha > 0:
        # cosine from (1-alpha)/2 to 1
        phi = (x[m_taper] - (1 - alpha) / 2) / (alpha / 2)
        w[m_taper] = 0.5 * (1 + np.cos(np.pi * phi))
    w[x >= 1] = 0.0
    return w


def finite_duration_biexp(
    t,
    A=1.0,
    alpha=1.0,
    beta=0.5,
    t0=0.0,
    T=1.0,
    window="smoothstep",
    renorm=None,
    **window_kwargs,
):
    """
    Multiply bi-exponential by a compact-support window so it's exactly 0 after t0+T.

    window: "smoothstep" (C¹), "tukey" (C⁰/C¹ depending on alpha).
    renorm: None | "area" | "peak"
        - None: no rescaling
        - "area": rescales so integral over [t0, t0+T] matches the infinite-area A*(1/alpha - 1/beta)
        - "peak": rescales so pre-taper peak (if inside flat region) matches original peak
    """
    t = np.asarray(t, dtype=float)
    base = biexp(t, A=A, alpha=alpha, beta=beta, t0=t0)

    if window == "smoothstep":
        w = smoothstep_cutoff(t, t0, T)
    elif window == "tukey":
        w = tukey_cutoff(t, t0, T, **window_kwargs)
    else:
        raise ValueError("window must be 'smoothstep' or 'tukey'")

    y = base * w

    # Optional renormalization
    if renorm is not None:
        if renorm == "area":
            # target area of the infinite bi-exponential (for common-shift case)
            target_area = A * (1.0 / alpha - 1.0 / beta)
            num = np.trapz(y, t)
            if num != 0:
                y *= target_area / num
        elif renorm == "peak":
            # find peak of untapered curve
            # peak time (for alpha!=beta) relative to t0: t* = ln(alpha/beta)/(alpha - beta)
            if alpha != beta:
                t_rel = np.log(alpha / beta) / (alpha - beta)
                t_peak = t0 + max(0.0, t_rel)  # clamp to >= t0
            else:
                t_peak = t0  # degenerate case
            # scale so the value at t_peak matches the untapered one
            base_peak = biexp(np.array([t_peak]), A=A, alpha=alpha, beta=beta, t0=t0)[0]
            y_peak = finite_duration_biexp.__dict__.get("_tmp_peak_cache")
            # compute y at peak with current window:
            y_at_peak = biexp(np.array([t_peak]), A=A, alpha=alpha, beta=beta, t0=t0)[
                0
            ] * (
                smoothstep_cutoff(np.array([t_peak]), t0, T)[0]
                if window == "smoothstep"
                else tukey_cutoff(np.array([t_peak]), t0, T, **window_kwargs)[0]
            )
            if y_at_peak != 0:
                y *= base_peak / y_at_peak
        else:
            raise ValueError("renorm must be None, 'area', or 'peak'")

    return y


def example_usage():
    t = np.linspace(0, 5, 1000)
    y_inf = biexp(t, A=1.0, alpha=1.5, beta=0.3, t0=0.5)

    # Finite duration version with smoothstep cutoff at T=2.0 s
    y_finite = finite_duration_biexp(
        t, A=1.0, alpha=1.5, beta=0.3, t0=0.5, T=2.0, window="smoothstep", renorm="area"
    )

    # Or with a Tukey window (50% taper)
    y_tukey = finite_duration_biexp(
        t, A=1.0, alpha=1.5, beta=0.3, t0=0.5, T=2.0, window="tukey"
    )
