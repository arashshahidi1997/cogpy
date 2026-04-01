# How to work with brain states and regression

`cogpy.brainstates` provides interval-based state classification (sleep stages,
behavioral states, transitions).  `cogpy.regression` builds design matrices and
solves OLS problems for artifact removal or signal decomposition.  They are
independent modules that compose naturally in state-stratified analyses.

---

## State dictionary format

Brain states are represented as a dict mapping labels to lists of `[start, end)`
intervals (seconds):

```python
states = {
    "PerSWS": [[10.0, 45.0], [120.0, 200.0]],
    "PerREM": [[50.0, 80.0], [210.0, 260.0]],
}
```

---

## Convert states to a DataFrame

```python
from cogpy.brainstates.brainstates import get_states_df

states_df = get_states_df(states)
# Columns: state, iseg, t0, t1
```

## Get per-state durations

```python
from cogpy.brainstates.brainstates import get_state_durations

durations = get_state_durations(states_df)
# Returns pd.Series indexed by state label
```

---

## Label time points by state

Given a time array and a state dict, assign each sample to a state period:

```python
from cogpy.brainstates.brainstates import label_numbers_by_state_intervals

t = sig.coords["time"].values
labeled = label_numbers_by_state_intervals(t, states)
# DataFrame: one column per state, values = period index or -1
```

---

## Filter events by state

```python
from cogpy.brainstates.brainstates import sort_col_into_states, filter_by_states

# Add state labels to an event DataFrame
events_df = sort_col_into_states(events_df, "time", states)

# Keep only events during SWS
sws_events = filter_by_states(events_df, include_states=["PerSWS"], exclude_states=[])
```

---

## Restrict signal to intervals

```python
from cogpy.brainstates.intervals import restrict

# Keep only SWS portions of the signal
sig_sws = restrict(sig, states["PerSWS"])

# Or pass the full states dict (union of all intervals)
sig_any_state = restrict(sig, states)
```

---

## Detect state transitions

```python
from cogpy.brainstates.brainstates import state_transitions, state_transition_interval

# All transitions
trans_df = state_transitions(states)
# Columns: transition_time, prev_state, next_state

# Extract SWS→REM transition windows (30 s before, 30 s after)
windows = state_transition_interval(states, "PerSWS", "PerREM", 30.0, 30.0)
# Returns (n_transitions, 2) array of [t_start, t_end]
```

Add transition intervals back as a new state:

```python
from cogpy.brainstates.brainstates import append_transition_intervals

states = append_transition_intervals(states, "PerSWS", "PerREM", window_before=30, window_after=0)
# Adds key "PerSWS_To_PerREM" to states dict
```

---

## Purify macro states (remove micro-state overlap)

When macrostates (e.g. SWS) contain embedded microstates (e.g. high-velocity
spindles), subtract the microstates:

```python
from cogpy.brainstates.brainstates import purify_macro_states

states = purify_macro_states(
    states,
    macro_states=["PerSWS", "PerREM"],
    micro_states=["PerHVS", "PerMicroA"],
)
```

---

## EMG proxy from intracranial recordings

Estimate muscle activity from high-frequency inter-channel correlations
(Watson & Buzsaki protocol):

```python
from cogpy.brainstates.EMG import compute_emg_proxy

emg_df = compute_emg_proxy(
    sig,
    fs=1000.0,
    coords_df=channel_coords_df,   # columns: ml, ap, dv (micrometers)
    min_distance=400.0,             # only use distant channel pairs
    window_size=0.5,                # 500 ms windows
    window_step=0.25,               # 250 ms step
)
# Returns DataFrame with columns: time, emg_proxy, emg_proxy_std
```

---

## Build a lagged design matrix

`lagged_design_matrix` creates a Toeplitz matrix where each column is the
reference signal shifted by a given lag.  Positive lags look backward in time.

```python
from cogpy.regression import lagged_design_matrix

# Reference signal (e.g. stimulation TTL), lags 0..5 samples
X = lagged_design_matrix(reference, lags=range(6), intercept=True)
# X.shape → (n_time, 7)  — 1 intercept + 6 lag columns
```

---

## Build an event design matrix

When you have discrete events and a known template waveform, build a design
matrix with one column per event:

```python
from cogpy.regression import event_design_matrix

# template: (n_lag,) waveform, event_samples: (n_events,) sample indices
X = event_design_matrix(len(sig), event_samples, template, intercept=True)
# X.shape → (n_time, n_events + 1)
```

---

## Fit, predict, and clean with OLS

```python
from cogpy.regression import ols_fit, ols_predict, ols_residual

# Y: (n_time,) or (n_time, n_channels) — supports multi-channel
beta = ols_fit(X, Y)            # least-squares coefficients
Y_hat = ols_predict(X, beta)    # predicted artifact
cleaned = ols_residual(X, Y, beta)  # Y - Y_hat
```

---

## Full example: state-stratified artifact removal

Remove stimulation artifacts from SWS epochs using lagged regression:

```python
from cogpy.brainstates.intervals import restrict
from cogpy.regression import lagged_design_matrix, ols_fit, ols_residual
import numpy as np

# 1. Restrict signal and reference to SWS intervals
sig_sws = restrict(sig, states["PerSWS"])
ref_sws = restrict(stim_ref, states["PerSWS"])

# 2. Build lagged design matrix (0–10 ms at 1 kHz = lags 0..10)
X = lagged_design_matrix(ref_sws.values, lags=range(11), intercept=True)

# 3. Fit and subtract per-channel
Y = sig_sws.values  # (n_time, n_channels)
beta = ols_fit(X, Y)
sig_clean = ols_residual(X, Y, beta)
```

## Full example: event-locked template regression

Remove a known artifact template at each event:

```python
from cogpy.triggered import estimate_template
from cogpy.brainstates.intervals import perievent_epochs
from cogpy.regression import event_design_matrix, ols_fit, ols_residual
import numpy as np

# 1. Estimate template from epochs
epochs = perievent_epochs(sig, event_times, fs=1000.0, pre=0.001, post=0.010)
template = estimate_template(epochs, method="median").values.ravel()

# 2. Build design matrix
event_samples = np.round(event_times * 1000).astype(int)
X = event_design_matrix(sig.sizes["time"], event_samples, template)

# 3. Fit per-event amplitudes and subtract
Y = sig.values
beta = ols_fit(X, Y)
cleaned = ols_residual(X, Y, beta)
```

## See also

- {doc}`triggered-analysis` — triggered analysis workflows
- {doc}`/api/brainstates` — full API reference
