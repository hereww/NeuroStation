"""Build paired training windows for the learned EEG denoiser.

The builder uses recorded sessions as clean-ish bases and injects artifact
segments from deterministic generators. It is deliberately explicit about the
fact that these are engineering targets, not clinical ground truth.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import signal

from .learned_denoise import LearnedDenoiseError


def read_session_eeg(session_dir: Path, expected_channels: int = 8) -> tuple[np.ndarray, float, np.ndarray]:
    """Read raw_brainflow.tsv as [channels, samples], sample rate and validity."""
    raw_path = session_dir / "raw_brainflow.tsv"
    if not raw_path.exists():
        raise LearnedDenoiseError(f"session is missing raw_brainflow.tsv: {session_dir}")
    try:
        with raw_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = list(reader.fieldnames or [])
            eeg_names = [name for name in fieldnames if name.lower().startswith("eeg")]
            if len(eeg_names) < expected_channels:
                eeg_names = [name for name in fieldnames if name.lower().startswith("ch")]
            if len(eeg_names) < expected_channels:
                raise LearnedDenoiseError(f"could not find {expected_channels} EEG columns in {raw_path}")
            eeg_names = eeg_names[:expected_channels]
            rows = []
            for row in reader:
                rows.append([_float_or_nan(row.get(name, "")) for name in eeg_names])
    except OSError as error:
        raise LearnedDenoiseError(f"cannot read {raw_path}: {error}") from error
    if not rows:
        raise LearnedDenoiseError(f"session has no EEG rows: {raw_path}")
    eeg = np.asarray(rows, dtype=np.float32).T
    metadata = {}
    session_path = session_dir / "session.json"
    if session_path.exists():
        try:
            loaded = json.loads(session_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metadata = loaded
        except (OSError, ValueError, json.JSONDecodeError):
            metadata = {}
    sample_rate = float(metadata.get("sampling_rate_hz", 250.0))
    valid = np.isfinite(eeg)
    return eeg, sample_rate, valid


def _float_or_nan(value: object) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def discover_sessions(root: Path) -> list[Path]:
    if (root / "raw_brainflow.tsv").exists():
        return [root]
    return sorted({path.parent for path in root.rglob("raw_brainflow.tsv") if path.parent.is_dir()})


def iter_windows(values: np.ndarray, window_samples: int, hop_samples: int) -> Iterable[tuple[int, np.ndarray]]:
    if values.ndim != 2:
        raise LearnedDenoiseError("values must have shape [channels, samples]")
    _, samples = values.shape
    if samples < window_samples:
        return
    for start in range(0, samples - window_samples + 1, hop_samples):
        yield start, values[:, start : start + window_samples]


def cleanish_window(window: np.ndarray, min_valid_fraction: float = 0.98) -> bool:
    valid_fraction = float(np.mean(np.isfinite(window))) if window.size else 0.0
    if valid_fraction < min_valid_fraction:
        return False
    finite = np.where(np.isfinite(window), window, 0.0)
    for channel in finite:
        dynamic_range = float(np.percentile(channel, 99.5) - np.percentile(channel, 0.5))
        if not np.isfinite(dynamic_range) or dynamic_range <= 1e-9:
            return False
        if np.mean(np.abs(np.diff(channel)) <= 1e-12) > 0.85:
            return False
    return True


def _band_limited_noise(rng: np.random.Generator, length: int, sample_rate: float, low: float, high: float) -> np.ndarray:
    raw = rng.normal(0.0, 1.0, length).astype(np.float32)
    nyquist = sample_rate / 2.0
    high = min(high, nyquist - 1.0)
    if not 0 < low < high:
        return raw
    sos = signal.butter(3, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, raw).astype(np.float32)


def synthesize_artifacts(
    clean: np.ndarray,
    sample_rate: float,
    rng: np.random.Generator,
    *,
    max_artifacts: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return noisy signal, [channels, samples] mask and integer type labels."""
    clean = np.asarray(clean, dtype=np.float32)
    channels, samples = clean.shape
    noise = np.zeros_like(clean)
    mask = np.zeros((channels, samples), dtype=bool)
    labels = np.zeros((channels, samples), dtype=np.int8)
    artifact_count = int(rng.integers(1, max(2, max_artifacts + 1)))
    for _ in range(artifact_count):
        kind = int(rng.integers(1, 6))
        start = int(rng.integers(0, max(1, samples - max(4, int(sample_rate * 0.25)))))
        duration = int(rng.integers(max(4, int(sample_rate * 0.1)), max(5, int(sample_rate * 1.8))))
        end = min(samples, start + duration)
        if end <= start:
            continue
        selected_count = int(rng.integers(1, channels + 1))
        selected = rng.choice(channels, size=selected_count, replace=False)
        envelope = np.ones(end - start, dtype=np.float32)
        if envelope.size > 8:
            ramp = min(envelope.size // 4, int(sample_rate * 0.1))
            if ramp > 0:
                envelope[:ramp] = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
                envelope[-ramp:] = np.linspace(1.0, 0.0, ramp, dtype=np.float32)
        if kind == 1:  # blink / EOG-like low-frequency pulse
            t = np.linspace(-1.0, 1.0, end - start, dtype=np.float32)
            artifact = np.exp(-4.0 * t * t) * rng.uniform(1.5, 5.0)
        elif kind == 2:  # EMG-like broadband burst
            artifact = _band_limited_noise(rng, end - start, sample_rate, 30.0, min(100.0, sample_rate / 2.0 - 2.0))
            artifact *= rng.uniform(0.5, 2.5)
        elif kind == 3:  # line interference
            frequency = float(rng.choice([50.0, 60.0]))
            phase = float(rng.uniform(0.0, 2 * np.pi))
            artifact = np.sin(2 * np.pi * frequency * np.arange(end - start) / sample_rate + phase).astype(np.float32)
            artifact *= rng.uniform(0.2, 1.5)
        elif kind == 4:  # movement / slow drift
            artifact = np.cumsum(rng.normal(0.0, 0.02, end - start)).astype(np.float32)
            artifact -= np.mean(artifact)
            artifact *= rng.uniform(1.0, 8.0) / max(float(np.std(artifact)), 1e-4)
        else:  # electrode pop / transient step
            artifact = np.zeros(end - start, dtype=np.float32)
            midpoint = max(1, (end - start) // 3)
            artifact[midpoint:] = rng.uniform(-4.0, 4.0)
            artifact += rng.normal(0.0, 0.08, end - start).astype(np.float32)
        artifact *= envelope
        for channel in selected:
            scale = rng.uniform(0.7, 1.3)
            noise[channel, start:end] += scale * artifact
            mask[channel, start:end] = True
            labels[channel, start:end] = kind
    noisy = clean + noise
    return noisy.astype(np.float32), mask, labels


def build_training_arrays(
    session_root: Path,
    *,
    window_samples: int = 1000,
    hop_samples: int = 500,
    expected_channels: int = 8,
    max_windows: int = 20000,
    seed: int = 20261005,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    clean_windows: list[np.ndarray] = []
    noisy_windows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    session_ids: list[str] = []
    sample_rates: list[float] = []
    for session_dir in discover_sessions(session_root):
        try:
            eeg, sample_rate, _ = read_session_eeg(session_dir, expected_channels)
        except LearnedDenoiseError:
            continue
        for _, window in iter_windows(eeg, window_samples, hop_samples):
            if not cleanish_window(window):
                continue
            finite = np.isfinite(window)
            fill = np.nanmedian(np.where(finite, window, np.nan), axis=1)
            fill = np.where(np.isfinite(fill), fill, 0.0)
            clean = np.where(finite, window, fill[:, None]).astype(np.float32)
            noisy, mask, label = synthesize_artifacts(clean, sample_rate, rng)
            clean_windows.append(clean)
            noisy_windows.append(noisy)
            masks.append(mask)
            labels.append(label)
            session_ids.append(session_dir.name)
            sample_rates.append(sample_rate)
            if len(clean_windows) >= max_windows:
                break
        if len(clean_windows) >= max_windows:
            break
    if not clean_windows:
        raise LearnedDenoiseError(f"no clean-ish windows found under {session_root}")
    return {
        "clean": np.stack(clean_windows).astype(np.float32),
        "noisy": np.stack(noisy_windows).astype(np.float32),
        "artifact_mask": np.stack(masks).astype(np.uint8),
        "artifact_type": np.stack(labels).astype(np.int8),
        "session_id": np.asarray(session_ids),
        "sample_rate_hz": np.asarray(sample_rates, dtype=np.float32),
    }



def _load_npy_first(dataset_dir: Path, stem: str) -> np.ndarray:
    path = dataset_dir / f"{stem}.npy"
    if not path.exists():
        raise LearnedDenoiseError(f"EEGdenoiseNet file is missing: {path}")
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise LearnedDenoiseError(f"cannot load {path}: {error}") from error
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise LearnedDenoiseError(f"invalid finite 2D array: {path}")
    return np.asarray(values, dtype=np.float32)


def build_eegdenoisenet_arrays(
    dataset_dir: Path,
    *,
    window_samples: int = 1000,
    target_rate_hz: float = 250.0,
    max_windows: int = 20000,
    seed: int = 20261005,
) -> dict[str, np.ndarray]:
    """Build 8-channel-shaped supervised windows from EEGdenoiseNet epochs.

    EEGdenoiseNet is a single-channel clean/EOG/EMG epoch pool. The same
    artifact source is mixed into one or more synthetic channels, while the
    clean waveform remains the target. This is a training source, not a claim
    that the files are simultaneous multi-channel recordings.
    """
    clean_pool = _load_npy_first(dataset_dir, "EEG_all_epochs")
    eog_pool = _load_npy_first(dataset_dir, "EOG_all_epochs")
    emg_pool = _load_npy_first(dataset_dir, "EMG_all_epochs")
    source_rate_hz = 256.0
    epoch_samples = max(1, int(round(clean_pool.shape[1] * target_rate_hz / source_rate_hz)))
    if window_samples < epoch_samples:
        raise LearnedDenoiseError("window_samples is shorter than a resampled EEGdenoiseNet epoch")

    def resample_pool(pool: np.ndarray) -> np.ndarray:
        return signal.resample(pool, epoch_samples, axis=1).astype(np.float32)

    clean_pool = resample_pool(clean_pool)
    eog_pool = resample_pool(eog_pool)
    emg_pool = resample_pool(emg_pool)
    rng = np.random.default_rng(seed)
    clean_windows: list[np.ndarray] = []
    noisy_windows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    session_ids: list[str] = []
    channel_count = 8
    chunks_per_window = int(np.ceil(window_samples / epoch_samples))
    for index in range(max_windows):
        clean_parts: list[np.ndarray] = []
        noisy_parts: list[np.ndarray] = []
        mask_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        for _ in range(chunks_per_window):
            clean_epoch = clean_pool[int(rng.integers(0, len(clean_pool)))]
            clean_part = clean_epoch.copy()
            noisy_part = clean_part.copy()
            mask_part = np.zeros((channel_count, epoch_samples), dtype=bool)
            label_part = np.zeros((channel_count, epoch_samples), dtype=np.int8)
            if rng.random() >= 0.2:
                artifact_kind = int(rng.integers(1, 3))
                artifact_pool = eog_pool if artifact_kind == 1 else emg_pool
                artifact = artifact_pool[int(rng.integers(0, len(artifact_pool)))].copy()
                artifact -= np.mean(artifact)
                clean_scale = float(np.std(clean_part))
                artifact_scale = max(float(np.std(artifact)), 1e-6)
                mix_scale = rng.uniform(0.25, 2.0) * max(clean_scale, 1e-3) / artifact_scale
                artifact *= mix_scale
                selected_count = int(rng.integers(1, channel_count + 1))
                selected = rng.choice(channel_count, size=selected_count, replace=False)
                for channel in selected:
                    noisy_part = noisy_part  # keep the scalar epoch as the base signal below
                    if channel == 0:
                        pass
                # Channel-specific gains create a spatially imperfect but useful
                # approximation of EOG/EMG spread across the montage.
                noisy_channels = np.repeat(noisy_part[None, :], channel_count, axis=0)
                clean_channels = np.repeat(clean_part[None, :], channel_count, axis=0)
                for channel in selected:
                    noisy_channels[channel] += artifact * rng.uniform(0.7, 1.3)
                    mask_part[channel] = True
                    label_part[channel] = artifact_kind
            else:
                noisy_channels = np.repeat(noisy_part[None, :], channel_count, axis=0)
                clean_channels = np.repeat(clean_part[None, :], channel_count, axis=0)
            clean_parts.append(clean_channels)
            noisy_parts.append(noisy_channels)
            mask_parts.append(mask_part)
            label_parts.append(label_part)
        clean_window = np.concatenate(clean_parts, axis=1)[:, :window_samples]
        noisy_window = np.concatenate(noisy_parts, axis=1)[:, :window_samples]
        mask_window = np.concatenate(mask_parts, axis=1)[:, :window_samples]
        label_window = np.concatenate(label_parts, axis=1)[:, :window_samples]
        clean_windows.append(clean_window.astype(np.float32))
        noisy_windows.append(noisy_window.astype(np.float32))
        masks.append(mask_window)
        labels.append(label_window)
        session_ids.append("EEGdenoiseNet")
    return {
        "clean": np.stack(clean_windows).astype(np.float32),
        "noisy": np.stack(noisy_windows).astype(np.float32),
        "artifact_mask": np.stack(masks).astype(np.uint8),
        "artifact_type": np.stack(labels).astype(np.int8),
        "session_id": np.asarray(session_ids),
        "sample_rate_hz": np.full(len(clean_windows), target_rate_hz, dtype=np.float32),
    }


def concatenate_training_arrays(*items: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    items = tuple(item for item in items if item and "clean" in item and len(item["clean"]))
    if not items:
        raise LearnedDenoiseError("no training arrays to concatenate")
    keys = ("clean", "noisy", "artifact_mask", "artifact_type", "session_id", "sample_rate_hz")
    return {key: np.concatenate([item[key] for item in items], axis=0) for key in keys}

def save_training_arrays(output_path: Path, arrays: dict[str, np.ndarray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


__all__ = [
    "build_eegdenoisenet_arrays",
    "build_training_arrays",
    "concatenate_training_arrays",
    "cleanish_window",
    "discover_sessions",
    "iter_windows",
    "read_session_eeg",
    "save_training_arrays",
    "synthesize_artifacts",
]
