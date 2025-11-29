import time
import numpy as np
import ghostipy as gsp

# TODO: adjust this import to your actual module name / path
# from my_mtm_fftgram_module import mtm_spectrogram
from mtm_core import mtm_spectrogram  # <-- change as needed


def run_single_test(
    fs=1000.0,
    duration=5.0,
    freq0=30.0,
    bandwidth=4.0,
    nperseg=256,
    noverlap=128,
    nfft=None,
    remove_mean=False,
):
    """
    Compare our single-channel mtm_spectrogram against ghostipy.gsp.mtm_spectrogram
    on one synthetic time series.

    Prints:
      - shapes
      - timings
      - L2 and max abs error between spectrograms
    """
    rng = np.random.default_rng(0)

    # --------------------------
    # Generate test signal
    # --------------------------
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs
    x = (
        np.sin(2 * np.pi * freq0 * t + rng.uniform(0, 2 * np.pi))
        + 0.5 * rng.normal(size=n_samples)
    )

    print("Generating test data...")
    print(f"  fs = {fs} Hz, duration = {duration} s, n_samples = {n_samples}")
    print(f"  nperseg = {nperseg}, noverlap = {noverlap}, nfft = {nfft}\n")

    # --------------------------
    # Our implementation
    # --------------------------
    print("Running our mtm_spectrogram...")
    t0 = time.perf_counter()
    S_ours, f_ours, t_ours = mtm_spectrogram(
        x,
        bandwidth=bandwidth,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        n_tapers=None,
        min_lambda=0.95,
        remove_mean=remove_mean,
        nfft=nfft,
    )
    t1 = time.perf_counter()
    print(f"  -> Finished in {t1 - t0:.3f} s")
    print(f"  -> S_ours shape: {S_ours.shape}")
    print(f"  -> f_ours shape: {f_ours.shape}")
    print(f"  -> t_ours shape: {t_ours.shape}")

    # --------------------------
    # ghostipy reference
    # --------------------------
    print("\nRunning ghostipy.gsp.mtm_spectrogram...")
    t2 = time.perf_counter()
    S_ref, f_ref, t_ref = gsp.mtm_spectrogram(
        x,
        bandwidth=bandwidth,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        min_lambda=0.95,
        remove_mean=remove_mean,
        nfft=nfft,
    )
    t3 = time.perf_counter()
    print(f"  -> Finished in {t3 - t2:.3f} s")
    print(f"  -> S_ref shape:  {S_ref.shape}")
    print(f"  -> f_ref shape:  {f_ref.shape}")
    print(f"  -> t_ref shape:  {t_ref.shape}")

    # --------------------------
    # Compare axes
    # --------------------------
    print("\nAxis comparison:")
    print(f"  freq axes equal? {np.allclose(f_ours, f_ref)}")
    print(f"  time axes equal? {np.allclose(t_ours, t_ref)}")

    # If shapes differ, bail out early
    if S_ours.shape != S_ref.shape:
        print("\nShape mismatch between ours and ghostipy — cannot compare values.")
        return

    # --------------------------
    # Numerical comparison
    # --------------------------
    diff = S_ours - S_ref
    l2 = np.sqrt(np.mean(diff**2))
    max_err = np.max(np.abs(diff))
    print("\nNumerical comparison:")
    print(f"  L2 error:      {l2:.4e}")
    print(f"  Max abs error: {max_err:.4e}")
    if l2 < 1e-8 and max_err < 1e-6:
        print("  -> PASS: spectrograms match very well.")
    else:
        print("  -> CHECK: noticeable differences, inspect axis conventions / scaling.")

    # --------------------------
    # Timings summary
    # --------------------------
    print("\nTiming summary:")
    print(f"  Ours:      {t1 - t0:.3f} s")
    print(f"  ghostipy:  {t3 - t2:.3f} s")
    print("\nDone.\n")


if __name__ == "__main__":
    # A few example runs with different durations / nfft
    print("=== Test 1: 5 s, default nfft ===")
    run_single_test(duration=5.0, nperseg=256, noverlap=128)

    print("\n=== Test 2: 20 s, nfft = 512 ===")
    run_single_test(duration=20.0, nperseg=1024, noverlap=256)

    print("\n=== Test 3: 60 s, nfft = 1024 ===")
    run_single_test(duration=60.0, nperseg=1024, noverlap=256)

    print("\n=== Test 4: 500 s, nfft = 1024 ===")
    run_single_test(duration=500.0, nperseg=1024, noverlap=256)