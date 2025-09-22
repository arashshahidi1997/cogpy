from numpy.testing import assert_allclose
from scipy.signal import fftconvolve
from cogpy.model.poisson_process import *

def _ref_same_from_full(full_conv, L, K):
    # Crop the 'full' conv along the last axis to length L, centered with floor(K/2)
    start = K // 2
    stop = start + L
    return full_conv[..., start:stop]

def test_convolve_impulses_sparse(num_trials=20, seed=0):
    rng = np.random.default_rng(seed)

    for _ in range(num_trials):
        # random sizes, include even/odd K to catch off-by-one
        L = rng.integers(1, 25)
        spatial = tuple(rng.integers(1, 6, size=rng.integers(1, 4)))  # 1..3 spatial dims
        K = int(rng.integers(1, 15))
        kshape = spatial + (K,)

        sig = rng.integers(0, 2, size=L)            # 0/1 impulses
        # allow weights too to test generality
        if rng.random() < 0.5:
            sig = sig * rng.normal(size=L)

        kernel = rng.normal(size=kshape)

        # Build N-D reference via FFT on 'full', then crop for 'same'
        sig_nd = sig.reshape((1,) * (len(kshape) - 1) + (L,))
        ref_full = fftconvolve(sig_nd, kernel, mode="full")  # shape: spatial + (L+K-1,)

        # FULL
        out_full = convolve_impulses_sparse(sig, kernel, mode="full")
        assert out_full.shape == ref_full.shape
        assert_allclose(out_full, ref_full, rtol=1e-12, atol=1e-12)

        # SAME
        ref_same = _ref_same_from_full(ref_full, L, K)      # shape: spatial + (L,)
        out_same = convolve_impulses_sparse(sig, kernel, mode="same")
        assert out_same.shape == ref_same.shape
        assert_allclose(out_same, ref_same, rtol=1e-12, atol=1e-12)

    print("All tests passed!")
