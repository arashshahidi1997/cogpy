# How to do triggered analysis

Triggered analysis locks a continuous signal to discrete events and computes
summary statistics or templates across those events.

The typical workflow is: **detect events → extract epochs → compute stats / template → (optional) subtract template**.

---

## Extract peri-event epochs

Use `perievent_epochs` from `cogpy.brainstates.intervals` to cut windows
around each event time. The result is an xarray with an `"event"` dimension
and a `"lag"` dimension (time relative to event onset).

```python
from cogpy.brainstates.intervals import perievent_epochs

# event_times: 1D array of event onset times in seconds
# sig: xr.DataArray with a "time" dimension, any other dims preserved
epochs = perievent_epochs(
    sig,
    event_times,
    fs=1000.0,        # sampling rate
    pre=0.5,          # 500 ms before event
    post=1.0,         # 1000 ms after event
)
# epochs.dims → ("event", ..., "lag")
# epochs.coords["lag"] → array from -0.5 to 1.0
```

Events near recording boundaries are padded with NaN rather than dropped.

---

## Triggered average (ETA)

The event-triggered average is the mean across events at each lag. It reveals
the consistent, time-locked component of the signal.

```python
from cogpy.triggered import triggered_average

eta = triggered_average(epochs)
# eta has the same dims as a single epoch (no "event" dim)
```

---

## Variability and SNR

```python
from cogpy.triggered import triggered_std, triggered_snr

std = triggered_std(epochs)     # cross-event std at each lag
snr = triggered_snr(epochs)     # mean / standard-error
```

`triggered_snr` is `mean / (std / sqrt(n_events))` — high values indicate a
reliable event-locked component. `triggered_median` is also available as a
robust alternative to the mean.

---

## Template estimation

`estimate_template` is a more flexible aggregator that supports `"mean"`,
`"median"`, and `"trimmean"` (20% trimmed mean via scipy).

```python
from cogpy.triggered import estimate_template

template = estimate_template(epochs, method="trimmean")
```

---

## Per-event scaling

If the event-locked waveform varies in amplitude across events (e.g. stimulus
artifacts with variable intensity), fit a per-event scaling coefficient via
least-squares projection:

```python
from cogpy.triggered import fit_scaling

# Requires numpy arrays
alpha = fit_scaling(epochs.values, template.values)
# alpha[i] = <epochs_i, template> / <template, template>
```

---

## Template subtraction (artifact removal)

Given event onset sample indices and a template, subtract the template from
the continuous signal:

```python
from cogpy.triggered import subtract_template
import numpy as np

# Convert event times to sample indices
event_samples = np.round(event_times * fs).astype(int)

# Subtract with uniform scaling (alpha=1 for all events)
cleaned = subtract_template(sig, event_samples, template.values)

# Or with per-event scaling
cleaned = subtract_template(sig, event_samples, template.values, scaling=alpha)
```

Out-of-bounds events (where the template would extend beyond the signal) are
silently skipped. The return type matches the input (xr.DataArray or ndarray).

---

## Restrict signal to intervals

Before epoch extraction, you may want to restrict the signal to valid
intervals (e.g. excluding artifact periods):

```python
from cogpy.brainstates.intervals import restrict

# intervals: list of [t_start, t_end] or (n, 2) array
sig_clean = restrict(sig, intervals=[[10.0, 60.0], [120.0, 300.0]])
```

`restrict` supports `Intervals` objects, `(n, 2)` arrays, and dictionaries of
state intervals (union of all states). Boundary convention is `[t0, t1)`.

---

## Full example: stimulus artifact removal

```python
from cogpy.brainstates.intervals import perievent_epochs
from cogpy.triggered import (
    estimate_template,
    fit_scaling,
    subtract_template,
    triggered_snr,
)
import numpy as np

fs = 1000.0

# 1. Cut epochs around stimulus events
epochs = perievent_epochs(sig, stim_times, fs=fs, pre=0.001, post=0.010)

# 2. Estimate a robust template
template = estimate_template(epochs, method="median")

# 3. Check SNR — high values confirm a consistent artifact
snr = triggered_snr(epochs)
print("Peak SNR:", float(snr.max()))

# 4. Fit per-event amplitude scaling
alpha = fit_scaling(epochs.values, template.values)

# 5. Subtract from continuous signal
event_samples = np.round(stim_times * fs).astype(int)
sig_clean = subtract_template(sig, event_samples, template.values, scaling=alpha)
```
