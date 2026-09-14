"""Shared, non-clinical helpers for displaying and validating EEG samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_CHANNEL_NAMES = tuple(f"CH{index}" for index in range(1, 9))


def channel_labels_from_config(
    config: dict[str, Any] | None, channel_count: int
) -> tuple[str, ...]:
    """Return labels that never claim an electrode position unless configured."""

    count = max(0, int(channel_count))
    entries = config.get("channels") if isinstance(config, dict) else None
    labels: list[str] = []
    if isinstance(entries, list):
        for index, entry in enumerate(entries[:count], start=1):
            if not isinstance(entry, dict):
                labels.append(f"CH{index}")
                continue
            name = str(entry.get("name") or f"CH{index}").strip() or f"CH{index}"
            position = str(entry.get("electrode_position") or "").strip()
            if position and position.casefold() not in {"unspecified", "unknown", "null"}:
                labels.append(f"{name} · {position}")
            else:
                labels.append(name)
    while len(labels) < count:
        labels.append(f"CH{len(labels) + 1}")
    return tuple(labels)


def channel_labels_from_path(path: Path | None, channel_count: int) -> tuple[str, ...]:
    if path is None:
        return channel_labels_from_config(None, channel_count)
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        value = None
    return channel_labels_from_config(value if isinstance(value, dict) else None, channel_count)


def _interpolate_nonfinite(values: np.ndarray) -> tuple[np.ndarray, int]:
    finite = np.isfinite(values)
    invalid_count = int(values.size - np.count_nonzero(finite))
    if invalid_count == 0:
        return values.astype(float, copy=True), 0
    if not np.any(finite):
        return np.zeros_like(values, dtype=float), invalid_count
    indexes = np.arange(values.size, dtype=float)
    repaired = np.interp(indexes, indexes[finite], values[finite])
    return repaired.astype(float, copy=False), invalid_count


def display_filter(
    values: Sequence[float],
    sample_rate_hz: float,
    *,
    low_hz: float = 1.0,
    high_hz: float = 45.0,
) -> tuple[list[float], int]:
    """Return a transparent FFT display filter and the number of repaired values.

    This is intentionally a display-only filter. It uses a smooth frequency
    mask, removes the per-window median, and does not modify the saved raw EEG.
    The second return value lets callers surface non-finite input instead of
    silently treating it as valid data.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("waveform values must be one-dimensional")
    if array.size == 0:
        return [], 0
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    repaired, invalid_count = _interpolate_nonfinite(array)
    repaired = repaired - float(np.median(repaired))
    if repaired.size < 16:
        return repaired.tolist(), invalid_count

    nyquist = sample_rate_hz / 2.0
    low = max(0.0, min(float(low_hz), nyquist))
    high = max(low, min(float(high_hz), nyquist))
    frequencies = np.fft.rfftfreq(repaired.size, d=1.0 / sample_rate_hz)
    mask = np.zeros_like(frequencies)
    passband = (frequencies >= low) & (frequencies <= high)
    mask[passband] = 1.0
    # Smooth the two edges to reduce ringing in a short rolling display window.
    transition = max(1.0, sample_rate_hz / repaired.size)
    low_edge = (frequencies >= max(0.0, low - transition)) & (frequencies < low)
    high_edge = (frequencies > high) & (frequencies <= min(nyquist, high + transition))
    if low > 0:
        mask[low_edge] = (frequencies[low_edge] - (low - transition)) / transition
    if high < nyquist:
        mask[high_edge] = (high + transition - frequencies[high_edge]) / transition
    filtered = np.fft.irfft(np.fft.rfft(repaired) * mask, n=repaired.size)
    return filtered.astype(float, copy=False).tolist(), invalid_count

