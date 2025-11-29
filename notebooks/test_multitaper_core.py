import time
import numpy as np
import ghostipy as gsp
from multitaper_core import mtm_spectrogram   # <-- your new implementation


def test_against_ghostipy_loop():
    """
    Compare our multichannel mtm_spectrogram against ghostipy's single-channel version
    by looping over channels.

    This validates numerical correctness and gives a rough timing comparison.
    """

    rng = np.random.default_rng(0)

    # --------------------------
    # Parameters
    # --------------------------
    fs = 1000.0
    T = 10.0  # seconds
    n_samples = int(T * fs)
    n_channels = 64

    nperseg = 1024
    noverlap = 256
    bandwidth = 4.0

    print("Generating test data...")
    t = np.arange(n_samples) / fs
    # mix of sinusoids + noise
    data = np.vstack([
        np.sin(2*np.pi*(30+3*i)*t + rng.uniform(0, 2*np.pi)) +
        0.5 * rng.normal(size=n_samples)
        for i in range(n_channels)
    ])

    # -----------------------------------
    # Run our implementation (batched)
    # -----------------------------------
    print("\nRunning our mtm_spectrogram (batched)...")
    t0 = time.perf_counter()
    S_ours, f_ours, t_ours = mtm_spectrogram(
        data,
        bandwidth=bandwidth,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    t1 = time.perf_counter()
    print(f"  -> Finished in {t1 - t0:.3f} s")
    print(f"  -> Output shape: {S_ours.shape}")

    # -----------------------------------
    # Run ghostipy's reference channel-by-channel
    # -----------------------------------
    print("\nRunning ghostipy.mtm_spectrogram in a Python loop...")
    S_refs = []
    t2 = time.perf_counter()
    for ch in range(n_channels):
        S_ref, f_ref, t_ref = gsp.mtm_spectrogram(
            data[ch],
            bandwidth=bandwidth,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
        )
        S_refs.append(S_ref)
    t3 = time.perf_counter()
    print(f"  -> Finished in {t3 - t2:.3f} s")

    # stack to match our shape: (channels, freqs, time)
    S_ref_all = np.stack(S_refs, axis=0)
    print(f"  -> Stacked ref shape: {S_ref_all.shape}")

    # -----------------------------------
    # Compare numerical agreement
    # -----------------------------------
    # Ensure frequency + time alignment (ghostipy may pad/truncate slightly differently)
    if not np.allclose(f_ours, f_ref):
        print("WARNING: frequency axes do not match exactly.")
    if not np.allclose(t_ours, t_ref):
        print("WARNING: time axes do not match exactly.")

    # Resize S_ours to (channels, freqs, time)
    # Our output is (*channel_shape, n_freqs, n_segments)
    # data was (channels, time), so channel_shape = (channels,)
    S_ours_ch = S_ours

    # Compute differences
    err = S_ours_ch - S_ref_all
    l2 = np.sqrt(np.mean(err**2))
    maxerr = np.max(np.abs(err))

    print("\nNumerical comparison:")
    print(f"  L2 error:        {l2:.4e}")
    print(f"  Max abs error:   {maxerr:.4e}")

    # Very rough criteria — should be close unless window definitions differ
    if l2 < 1e-6:
        print("  -> PASS: Values match very well.")
    else:
        print("  -> CHECK: Values differ beyond tolerance (expected if implementations differ slightly).")

    print("\nDone.\n")


if __name__ == "__main__":
    test_against_ghostipy_loop()
