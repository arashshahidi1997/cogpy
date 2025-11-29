# **Overview of Sliding-Window Function Types and Use Cases**

This module provides a set of tools for building efficient sliding-window computations on NumPy arrays (and xarray wrappers), using the following conceptual layers:

---

## **1. Sliding-Window Construction (Core Infrastructure)**

### **Function type:**

**Window builders**

### **API:**

`sliding_window(x, window_size, window_step, axis=-1)`

### **Responsibility:**

Create a **non-copying view** over the time axis (or any axis), returning an array where:

* one new axis is the **window index** (mapped/loop dim),
* the last axis is **window samples** (core dim).

### **Use cases:**

* Building windows for any time-based or sequence-based analysis.
* Used as the foundation for all subsequent operations.
* Works for 1D, 2D, or ND data.

### **Characteristics:**

* Pure NumPy (`as_strided`) or safe variant if needed.
* No computations or reductions.
* Maximally reusable primitive.

---

## **2. Simple Running Reductions**

### **Function type:**

**Running reducers**

### **API:**

`running_reduce(x, window_size, window_step, reducer)`

### **Responsibility:**

Apply a reducer over the **window_samples** axis for each window index.

### **Core dims:**

`(window_samples,)`

### **Mapped dims:**

All other dims, including `window_index`, channels, spatial dims, etc.

### **Use cases:**

* Running mean, std, variance, RMS
* Envelope measures, kurtosis, skewness
* Small 1D FFTs for quick spectral features
* Compact feature vectors per window

### **Characteristics:**

* Simple: reducer takes `(…, window_samples)` → reduces last axis.
* Vectorized computation, minimal overhead.
* Good for lightweight per-window operations.

---

## **3. Generalized Running Functions (gufunc-like)**

### **Function type:**

**Running gufunc / block-wise transforms**

### **API:**

`running_gufunc(x, window_size, window_step, func, core_axes=(..., window_samples))`

### **Responsibility:**

Apply a function that consumes **multiple core dimensions** simultaneously.

### **Core dims examples:**

* `(x, y, window_samples)`
* `(channels, freq, window_samples)`
* `(sensors, window_samples)`

### **Mapped dims:**

* `window_index`, subject, trial, etc.

### **Use cases:**

* Running PCA / SVD over `(channels, window_samples)`
* Running correlation matrices inside each window
* Running ICA on `(channels, window_samples)`
* Any block-structured decomposition that uses multiple axes jointly

### **Characteristics:**

* `func` receives a **full core block**, not a vector.
* Usually requires a Python loop over `window_index` (because each block is independent).
* Ideal for **moderate-cost** per-window algorithms.

---

## **4. Specialized Core Functions (High-Performance Pipelines)**

### **Function type:**

**Domain-specific optimized kernels**

### **API examples:**

* `mtm_spectrogram_core(...)`
* `stft_core(...)`
* `running_cp_decomposition_core(...)`

### **Responsibility:**

Own the **entire** multi-step pipeline for heavy sliding-window algorithms, including:

* constructing sliding windows,
* adding extra axes (tapers, channels),
* batched FFTs,
* FFTW planning and threading,
* tensor decomposition algorithms,
* shape/layout control,
* multi-output semantics.

### **Use cases:**

* MTM (multi-taper) spectrograms
* STFT (short-time Fourier transform)
* Running tensor decompositions (CP/Tucker)
* Wavelet transforms
* Any multi-step numerical pipeline requiring:

  * axis control
  * multi-threading
  * plan reuse
  * batching

### **Characteristics:**

* Treats **all windows at once**, not one by one.
* Controls axis layout for cache + FFT efficiency.
* Integrates FFTW / JIT / BLAS optimizations internally.
* Returns multiple structured outputs (S, f, t, etc.).
* Can't be expressed cleanly as a single “measure of one window”.

---

# **5. Why Generic `running_measure` Has Natural Limits**

The function:

```python
running_measure(windows, measure)
```

is useful but inherently limited.

### **Designed for:**

* “One window in → one small output”
* reductions or small vector extraction
* simple per-window analysis
* lightweight feature computation

### **Not suitable for:**

### **1. Multi-step algorithms**

MTM/STFT pipelines need multiple stages:

* tapers
* batching
* zero padding
* FFTW planning
* taper weighting

A “single-window measure” can’t express that pipeline.

---

### **2. Algorithms that require all windows at once**

Optimized FFT logic wants:

```python
rfft(windows, axis=-1)   # batched FFT over all windows
```

But `running_measure` would feed **one window at a time**, killing performance and preventing batched transforms.

---

### **3. Multi-axis cores**

Running tensor decomposition over `(channels, window_samples)` must see the whole block:

```python
block = windows[i, :, :, :]   # multiple core axes
```

`running_measure` only handles 1 core dim by default.

---

### **4. Multi-output results**

Spectrograms produce:

* S(f, t)
* frequency vector
* time vector

Reductions do not.

---

### **5. Axis & memory layout control**

Optimized cores must choose very specific layouts for:

* FFT performance
* contiguous memory
* threading
* cache reuse

`running_measure` must stay axis-agnostic and cannot enforce such constraints.

---

# **6. Recommended Architectural Layering**

Use the layers like this:

### **Low-level:**

* `sliding_window`
  (core infrastructure)

### **Mid-level:**

* `running_reduce`
* `running_gufunc`
  (generic operations)

### **High-level:**

* `mtm_spectrogram_core`
* `stft_core`
* `running_cp_decomposition_core`
  (optimized domain-specific algorithms)

Each layer serves a different purpose, and trying to force them into one abstraction would hurt clarity and performance.
