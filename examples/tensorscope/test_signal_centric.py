"""
Test Signal-Centric Architecture

Demonstrates:
- Creating signals
- Duplicating and modifying processing
- Comparing signals side-by-side
- SignalManagerLayer UI (indirectly, via state/signal objects)

Run with:
    conda run -n cogpy python code/lib/cogpy/examples/tensorscope/test_signal_centric.py
"""

from __future__ import annotations

from cogpy.datasets.entities import example_ieeg_grid
from cogpy.plot.tensorscope import TensorScopeState

print("=" * 60)
print("TensorScope Signal-Centric Architecture Test")
print("=" * 60)

# Load data
print("\n1. Loading data...")
data = example_ieeg_grid(mode="small")
print(f"   ✅ Loaded: {data.dims}, shape={data.shape}")

# Create state
print("\n2. Creating TensorScope state...")
state = TensorScopeState(data)
print(f"   ✅ Signals: {state.signal_registry.list_names()}")

# Get base signal
print("\n3. Accessing base signal...")
base_id = state.signal_registry.list()[0]
base_signal = state.signal_registry.get(base_id)
print(f"   - ID: {base_signal.id}")
print(f"   - Name: {base_signal.name}")
print(f"   - Is base: {base_signal.metadata.get('is_base')}")
print("   ✅ Base signal accessible")

# Duplicate signal
print("\n4. Duplicating signal...")
dup_id = state.signal_registry.duplicate(base_id, "Filtered LFP")
dup_signal = state.signal_registry.get(dup_id)
print(f"   - Original ID: {base_id}")
print(f"   - Duplicate ID: {dup_id}")
print(f"   - Duplicate name: {dup_signal.name}")
assert dup_signal.data is base_signal.data
assert dup_signal.processing is not base_signal.processing
print("   ✅ Duplication successful (shared data, independent processing)")

# Modify duplicate's processing
print("\n5. Modifying duplicate processing...")
dup_signal.processing.bandpass_on = True
dup_signal.processing.bandpass_lo = 1.0
dup_signal.processing.bandpass_hi = 100.0
print(f"   - Set bandpass: {dup_signal.processing.bandpass_lo}-{dup_signal.processing.bandpass_hi} Hz")
assert base_signal.processing.bandpass_on is False
print("   ✅ Processing modified independently")

# Create derived signal via state helper
print("\n6. Creating high-gamma signal...")
hg_id = state.create_derived_signal(
    base_id,
    "High Gamma (70-150Hz)",
    {
        "bandpass_on": True,
        "bandpass_lo": 70.0,
        "bandpass_hi": 150.0,
        "zscore_on": True,
    },
)
hg_signal = state.signal_registry.get(hg_id)
print(f"   - ID: {hg_signal.id}")
print(f"   - Name: {hg_signal.name}")
print(f"   - Bandpass: {hg_signal.processing.bandpass_lo}-{hg_signal.processing.bandpass_hi} Hz")
print(f"   - Z-score: {hg_signal.processing.zscore_on}")
print("   ✅ Derived signal created with config")

# Compare windows
print("\n7. Comparing signal windows...")
t0, t1 = 5.0, 7.0
base_win = base_signal.get_window(t0, t1)
filt_win = dup_signal.get_window(t0, t1)
hg_win = hg_signal.get_window(t0, t1)
print(f"   - Window: [{t0}, {t1}]s")
print(f"   - Base shape: {base_win.shape}")
print(f"   - Filtered shape: {filt_win.shape}")
print(f"   - High-gamma shape: {hg_win.shape}")
print("   ✅ All signals produce windows")

# PSD via external analysis on windows
print("\n8. Computing PSD from window...")
from cogpy.spectral.specx import psdx

win = base_signal.get_window(5.0, 7.0)
psd = psdx(win, method="multitaper", bandwidth=4.0, nperseg=512)
print(f"   - Window shape: {win.shape}")
print(f"   - PSD shape: {psd.shape}")
print(f"   - PSD dims: {psd.dims}")
print("   ✅ Analysis via cogpy.spectral.specx")

# Test selected_time
print("\n9. Testing selected_time feature...")
state.time_hair.t = 6.0
state.set_selected_time_from_cursor()
print(f"   - Cursor: {state.time_hair.t}s")
print(f"   - Selected: {state.selected_time}s")
state.time_hair.t = 8.0
print(f"   - Cursor moved to: {state.time_hair.t}s")
print(f"   - Selected unchanged: {state.selected_time}s")
assert state.selected_time == 6.0
print("   ✅ Selected time independent of cursor")

# Test serialization
print("\n10. Testing serialization...")
state_dict = state.to_dict()
print(f"   - Keys: {list(state_dict.keys())}")
print(f"   - Signal count: {len((state_dict.get('signal_registry') or {}).get('signals') or {})}")
print("   ✅ Serialization working")

# List all signals
print("\n11. Summary of signals...")
for sid, name in state.signal_registry.list_names():
    sig = state.signal_registry.get(sid)
    proc_desc = sig.processing.describe()
    print(f"   - {name} ({sid[:6]}...): {proc_desc}")
print(f"   ✅ {len(state.signal_registry.list())} signals total")

print("\n" + "=" * 60)
print("✅ Signal-Centric Architecture: ALL TESTS PASSED!")
print("=" * 60)
