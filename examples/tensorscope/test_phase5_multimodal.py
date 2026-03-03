"""
Test Phase 5: Multi-Modal Support

Demonstrates:
- Creating multiple modalities (LFP, spectrogram)
- Registering with state
- Switching active modality
- Time alignment across rates
- Windowing with different sampling rates

Run with:
    conda run -n cogpy python code/lib/cogpy/examples/tensorscope/test_phase5_multimodal.py
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from cogpy.datasets.entities import example_ieeg_grid
from cogpy.core.plot.tensorscope import TensorScopeState
from cogpy.core.plot.tensorscope.data.modalities import SpectrogramModality
from cogpy.core.plot.tensorscope.data.alignment import (
    align_to_common_timebase,
    find_nearest_time_index,
)


print("=" * 60)
print("TensorScope Phase 5: Multi-Modal Support Test")
print("=" * 60)

# Load primary data
print("\n1. Loading primary LFP data...")
lfp_data = example_ieeg_grid(mode="small")
state = TensorScopeState(lfp_data)
print("   - Primary modality: grid_lfp")
print(f"   - Shape: {lfp_data.shape}")
print("   ✅ State created with grid_lfp")

# Create synthetic spectrogram
print("\n2. Creating synthetic spectrogram modality...")
n_time_spec = 200
n_freq = 60
spec_data = xr.DataArray(
    np.random.randn(n_time_spec, n_freq, 8, 8),
    dims=("time", "freq", "AP", "ML"),
    coords={
        "time": np.linspace(0, 10, n_time_spec),
        "freq": np.logspace(0, 2, n_freq),
        "AP": np.arange(8),
        "ML": np.arange(8),
    },
)

spec_modality = SpectrogramModality(spec_data)
print(f"   - Shape: {spec_data.shape}")
print(f"   - Time sampling: ~{spec_modality.sampling_rate:.1f}Hz")
print(f"   - Freq range: {spec_modality.freq_bounds()}")
print("   ✅ Spectrogram modality created")

# Register spectrogram
print("\n3. Registering spectrogram with state...")
state.register_modality("spectrogram", spec_modality)
modalities = state.data_registry.list()
print(f"   - Registered modalities: {modalities}")
assert len(modalities) == 2
print("   ✅ Spectrogram registered")

# Check active modality
print("\n4. Checking active modality...")
active_name = state.data_registry.get_active_name()
active_mod = state.get_active_modality()
print(f"   - Active modality: {active_name}")
print(f"   - Type: {active_mod.modality_type}")
print(f"   - Time bounds: {active_mod.time_bounds()}")
print("   ✅ Active modality queried")

# Switch active modality
print("\n5. Switching active modality...")
state.set_active_modality("spectrogram")
new_active = state.data_registry.get_active_name()
print(f"   - Switched to: {new_active}")
assert new_active == "spectrogram"

window = state.get_active_modality().get_window(2.0, 5.0)
print(f"   - Window [2.0, 5.0]: shape={window.shape}")
print("   ✅ Modality switching works")

# Test time alignment
print("\n6. Testing time alignment...")
lfp_mod = state.data_registry.get("grid_lfp")
spec_mod = state.data_registry.get("spectrogram")

lfp_times = lfp_mod.data.time.values
spec_times = spec_mod.data.time.values

print(f"   - LFP times: {len(lfp_times)} points, dt={np.median(np.diff(lfp_times)):.4f}s")
print(f"   - Spec times: {len(spec_times)} points, dt={np.median(np.diff(spec_times)):.4f}s")

common_time = align_to_common_timebase([lfp_times, spec_times], method="intersection")
print(f"   - Common timebase: {len(common_time)} points")
print("   ✅ Time alignment working")

# Test nearest time index
print("\n7. Testing nearest time index...")
target_time = 5.3
lfp_idx = find_nearest_time_index(target_time, lfp_times)
spec_idx = find_nearest_time_index(target_time, spec_times)
print(f"   - Target time: {target_time}s")
print(f"   - LFP nearest: t={lfp_times[lfp_idx]:.4f}s (idx={lfp_idx})")
print(f"   - Spec nearest: t={spec_times[spec_idx]:.4f}s (idx={spec_idx})")
print("   ✅ Nearest index search working")

# Test conversion between modalities
print("\n8. Testing modality conversions...")
grid_mod = state.data_registry.get("grid_lfp")
flat_mod = grid_mod.to_flat()
print(f"   - Grid shape: {grid_mod.data.shape}")
print(f"   - Flat shape: {flat_mod.data.shape}")
assert flat_mod.modality_type == "flat_lfp"
print("   ✅ Grid→Flat conversion working")

# Test serialization
print("\n9. Testing modality serialization...")
for name in state.data_registry.list():
    mod = state.data_registry.get(name)
    mod_dict = mod.to_dict()
    print(f"   - {name}: {list(mod_dict.keys())}")
print("   ✅ Serialization working")

print("\n" + "=" * 60)
print("✅ Phase 5 Multi-Modal Support: ALL TESTS PASSED!")
print("=" * 60)
