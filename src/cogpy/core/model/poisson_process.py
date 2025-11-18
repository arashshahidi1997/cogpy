"""
PointProcess:
    Abstract base class for point processes.
PoissonProcess:
    Implements a Poisson point process using exponential inter-spike intervals (ISIs).
PointBurst:
    Implements a burst process based on a given point process and event duration distribution.
PoissonBurst:
    Implements a Poisson burst process with exponentially distributed event durations.
ModeProcess:
    Combines a mode function with a point process to generate a signal.
ModeMixer:
    Mixes multiple ModeProcess signals and provides access to individual mode signals.
    Stores individual generated signals for each mode.

workflow:
    1. Define mode functions (e.g., Gaussian modes from a GaussianCover).
    2. Create PointProcess instances (e.g., PoissonProcess).
    3. Create ModeProcess instances by combining mode functions with point processes.
    4. Use ModeMixer to mix the signals from multiple ModeProcess instances.

example usage:
    from src.model.poisson_process import PoissonProcess, ModeProcess, ModeMixer
    from src.model.gaussian_cover import GaussianCover
    import xarray as xr
    import numpy as np

    # Define mode functions (e.g., Gaussian modes)
    gc = GaussianCover(shape=(16, 16, 70), sigma=[3, 3, 7])
    modes = gc.modes  # List of xarray.DataArray mode functions
    mode_processes = []
    for mode in modes:
        pp = PoissonProcess(rate=5)  # 5 events per unit time
        mp = ModeProcess(mode_function=mode, process=pp)
        mode_processes.append(mp)
    mixer = ModeMixer(mode_processes)
    mixed_signal = mixer.mix(duration=1000)  # Mix signals over 1000
    unmixed_signals = mixer.get_unmixed_modes()  # Access individual mode signals
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List
from scipy.stats import rv_continuous
import xarray as xr
import scipy.ndimage as nd
from functools import partial
from .envelopes import finite_duration_biexp


class PointProcess(ABC):
    @abstractmethod
    def generate(self, duration):
        """
        Generate a realization of the point process for a given duration.

        :param duration: The duration for which the process should be generated.
        :return: A realization of the point process.
        """
        pass


class PoissonProcess(PointProcess):
    def __init__(self, rate):
        """
        Initializes the Poisson Process.

        :param rate: The rate (λ) of the Poisson Process, i.e., average number of events per unit time.
        """
        self.rate = rate

    def generate(self, duration):
        """
        Generates a realization of the Poisson process over a duration using exponential ISIs.

        :param duration: Duration over which the Poisson process should be generated.
        :return: A binary array with "1"s indicating events and "0"s indicating no events.
        """
        # Estimate total number of events based on Poisson mean
        n = int(np.ceil(self.rate * duration + 3 * np.sqrt(self.rate * duration)))

        # Generate ISI using exponential distribution
        isis = np.random.exponential(1.0 / self.rate, n)

        # Compute event times
        events = np.cumsum(isis)

        # Filter events that are beyond the duration
        events = events[events < duration]

        # Create a binary signal of events over the duration
        signal = np.zeros(duration)
        signal[events.astype(int)] = 1

        return signal


class PointBurst:
    def __init__(
        self,
        point_process: PointProcess,
        duration_distribution: rv_continuous,
        refractory_period: int = 0,
    ):
        """
        Initializes the Point Burst Process.

        :param point_process: An instance of a point process with a generate(duration) method.
        :param duration_distribution: An instance of a scipy.stats distribution for event durations.
        :param refractory_period: The refractory period after each event during which no new events can start.
        """
        self.point_process = point_process
        self.duration_distribution = duration_distribution
        self.refractory_period = refractory_period

    def generate(self, duration: int):
        """
        Generates a realization of the point burst process over a duration.

        :param duration: Duration over which the point burst process should be generated.
        :return: A binary array with "1"s indicating burst events and "0"s indicating no events.
        """
        # Generate base point process
        base_signal = self.point_process.generate(duration)

        # Initialize burst signal
        burst_signal = np.zeros_like(base_signal)

        # Iterate over each event in the base signal
        last_event_end = -1
        for event_time in np.where(base_signal == 1)[0]:
            if event_time <= last_event_end:
                continue  # Skip if within refractory period

            # Sample burst duration
            burst_duration = int(np.ceil(self.duration_distribution.rvs()))

            # Ensure burst doesn't exceed the total duration
            end_time = min(event_time + burst_duration, duration)
            last_event_end = end_time + self.refractory_period

            # Mark the burst duration in the signal
            burst_signal[event_time:end_time] = 1

        return burst_signal


class PoissonBurst(PoissonProcess):
    def __init__(
        self, rate: float, duration_mean: float, envelope_func: callable = None
    ):
        """
        Initializes the Poisson Burst Process.

        :param rate: The rate (λ) of the Poisson Process, i.e., average number of events per unit time.
        :param duration_mean: The mean of the exponential distribution for event duration.
        """
        super().__init__(rate)
        self.duration_mean = duration_mean
        self.envelope_func = envelope_func

        # generate a signal of given envelope
        if self.envelope_func is None:
            self.envelope_func = partial(
                finite_duration_biexp,
                A=1.0,
                alpha=0.3,
                beta=1.5,
                t0=0,
                window="smoothstep",
                renorm="area",
            )

    def generate(self, duration):
        """
        Generates a realization of the Poisson burst process over a duration.

        :param duration: Duration over which the Poisson burst process should be generated.
        :return: A binary array with "1"s indicating burst events and "0"s indicating no events.
        """
        # Generate base Poisson process
        base_signal = super().generate(duration)

        # Initialize burst signal
        burst_signal = np.zeros_like(base_signal)

        # Iterate over each event in the base signal
        for event_time in np.where(base_signal == 1)[0]:
            # Sample burst duration from exponential distribution
            burst_duration = int(np.ceil(sample_exponential(self.duration_mean)))

            # Ensure burst doesn't exceed the total duration
            end_time = min(event_time + burst_duration, duration)

            envelope = self.envelope_func(
                np.arange(end_time - event_time), T=burst_duration
            )
            burst_signal[event_time:end_time] = envelope

        return burst_signal


class ModeProcess:
    def __init__(
        self, mode_function: xr.DataArray, process: PointProcess, convolve=False
    ):
        """
        Initialize a ModeProcess with a mode function and a corresponding point process.

        :param mode_function: An instance of a mode function (numpy array).
        :param point_process: An instance of a PointProcess.
        """
        # check type of mode_function
        if not isinstance(mode_function, xr.DataArray):
            raise TypeError("mode_function must be an instance of xarray.DataArray.")

        self.mode_function = mode_function
        self.point_process = process
        self.convolve = convolve

    def generate(self, duration):
        """
        Generate the signal for this mode.

        :param duration: The duration for which to generate the signal.
        :return: Generated signal for this mode.
        """
        envelope_signal = self.point_process.generate(duration)
        envex = xr.DataArray(
            envelope_signal, dims=["time"], coords={"time": np.arange(duration)}
        )
        if self.convolve:
            # Apply impulse response
            return convolve_impulses_sparse(
                envelope_signal, self.mode_function.values, mode="same"
            )
        return self.mode_function * envex


class ModeMixer:
    def __init__(self, mode_processes: List[ModeProcess]):
        """
        Initialize a ModeMixer with a list of ModeProcess objects.

        :param mode_processes: A list of ModeProcess instances.
        """
        if not all(
            mode_processes[0].mode_function.shape == mp.mode_function.shape
            for mp in mode_processes
        ):
            raise ValueError("All mode functions must have the same dimensionality.")

        self.mode_processes = mode_processes
        self.unmixed_modes = []  # To store individual generated signals

    def mix(self, duration):
        """
        Mix the signals from the provided ModeProcess objects and store each generated signal.

        :param duration: The duration for which to mix the signals.
        :return: Mixed signal.
        """
        mixed_modes = []
        self.unmixed_modes = []

        for mode_process in self.mode_processes:
            generated_signal = mode_process.generate(duration)
            self.unmixed_modes.append(generated_signal)

        self.unmixed_modes = xr.concat(self.unmixed_modes, dim="mode")
        mixed_modes = self.unmixed_modes.sum(dim="mode")  # Sum over all modes
        return mixed_modes

    def get_unmixed_modes(self):
        """
        Retrieve the individual generated signals for each mode.

        :return: List of generated signals.
        """
        return self.unmixed_modes


class ImpulseResponseMixer:
    def __init__(self, point_process: PointProcess, impulse_response: xr.DataArray):
        """
        Initialize an ImpulseResponseMixer with a point process and an impulse response.

        :param point_process: An instance of a PointProcess.
        :param impulse_response: An instance of an impulse response (numpy array).
        """
        if not isinstance(impulse_response, xr.DataArray):
            raise TypeError("impulse_response must be an instance of xarray.DataArray.")

        self.point_process = point_process
        self.impulse_response = impulse_response


def sample_exponential(mean):
    """
    Generate a sample from an exponential distribution given a mean.

    :param mean: The mean of the exponential distribution.
    :return: A sample from the exponential distribution.
    """
    rate = 1 / mean
    return np.random.exponential(scale=1 / rate)


# your implementation from earlier
def convolve_impulses_sparse(sig, kernel, mode="full"):
    sig = np.asarray(sig)
    k = np.asarray(kernel)
    L, K = sig.size, k.shape[-1]

    if mode == "full":
        out_len = L + K - 1
        start_shift = 0
    elif mode == "same":
        out_len = L
        start_shift = -(K // 2)  # center like np.convolve(..., 'same')
    else:
        raise ValueError("mode must be 'full' or 'same'")

    out = np.zeros(k.shape[:-1] + (out_len,), dtype=np.result_type(sig, k))
    idx = np.flatnonzero(sig)
    if idx.size == 0:
        return out

    for t in idx:
        w = sig[t]
        o0 = t + start_shift
        dst_lo = max(o0, 0)
        dst_hi = min(o0 + K, out_len)
        if dst_lo >= dst_hi:
            continue
        src_lo = dst_lo - o0
        src_hi = src_lo + (dst_hi - dst_lo)
        out[..., dst_lo:dst_hi] += w * k[..., src_lo:src_hi]
    return out
