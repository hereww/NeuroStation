"""Bounded, read-only statistics over the worker's raw EEG preview stream."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ChannelMetrics:
    mean_uv: float | None
    rms_uv: float | None
    peak_to_peak_uv: float | None
    valid_samples: int
    invalid_samples: int


class AcquisitionMonitorModel:
    """Last two seconds of raw samples; never substitutes zeros for missing data."""

    WINDOW_SECONDS = 2

    def __init__(self):
        self.channel_names: tuple[str, ...] = ()
        self.sample_rate_hz = 0
        self.sample_count = 0
        self.invalid_value_count = 0
        self.buffers: list[deque[float]] = []

    def configure(self, names, sample_rate_hz: int) -> None:
        names = tuple(str(name) for name in names)
        if not names or sample_rate_hz <= 0:
            raise ValueError("invalid live EEG configuration")
        if names == self.channel_names and sample_rate_hz == self.sample_rate_hz:
            return
        self.channel_names = names
        self.sample_rate_hz = sample_rate_hz
        self.sample_count = 0
        self.invalid_value_count = 0
        self.buffers = [deque(maxlen=sample_rate_hz * self.WINDOW_SECONDS) for _ in names]

    def append(self, samples: list[list[float]]) -> None:
        if len(samples) != len(self.buffers) or not self.buffers:
            raise ValueError("live EEG channel count")
        count = len(samples[0])
        if any(len(channel) != count for channel in samples):
            raise ValueError("live EEG sample lengths")
        converted = [[float(value) for value in channel] for channel in samples]
        for buffer, channel in zip(self.buffers, converted):
            self.invalid_value_count += sum(not math.isfinite(value) for value in channel)
            buffer.extend(channel)
        self.sample_count += count

    def metrics(self) -> tuple[ChannelMetrics, ...]:
        result = []
        for buffer in self.buffers:
            values = [value for value in buffer if math.isfinite(value)]
            invalid = len(buffer) - len(values)
            if not values:
                result.append(ChannelMetrics(None, None, None, 0, invalid))
                continue
            mean = math.fsum(values) / len(values)
            # AC RMS is insensitive to the board's DC electrode offset.
            rms = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))
            result.append(ChannelMetrics(mean, rms, max(values) - min(values), len(values), invalid))
        return tuple(result)
