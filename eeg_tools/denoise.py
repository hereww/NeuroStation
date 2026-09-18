"""Reproducible, non-destructive EEG denoising and quality metrics.

The module intentionally keeps the first denoising pipeline transparent.  It
does not overwrite a session's raw BrainFlow TSV and it does not claim that an
artifact detector has recovered the participant's neural signal.  It produces
an annotated raw track plus conventional filtered tracks for comparison.
"""

from __future__ import annotations

import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
from typing import Any, Iterable

import numpy as np
import scipy
from scipy import signal

from .offline_analysis import (
    _float,
    _int,
    _marker_sample_indexes,
    _read_tsv,
    _ssvep_score,
)


PIPELINE_VERSION = "denoise_v1"
DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "sampling_rate_hz": 250,
    "bandpass_hz": [1.0, 45.0],
    "bandpass_order": 4,
    "line_noise_hz": [50.0, 60.0],
    "notch_quality_factor": 30.0,
    "max_interpolation_s": 0.2,
    "artifact_detection": {
        "flatline_fraction": 0.2,
        "robust_z_threshold": 8.0,
        "high_frequency_band_hz": [30.0, 45.0],
        "high_frequency_ratio": 0.35,
        "motion_z_threshold": 8.0,
        "window_s": 1.0,
        "step_s": 0.5,
    },
}

_RAW_REQUIRED = {"sample_index"}
_SESSION_INPUTS = ("raw_brainflow.tsv", "raw_columns.tsv", "session.json", "events.tsv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON object: {path.name}")
    return value


def _merged_config(value: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    config.update({key: item for key, item in value.items() if key != "artifact_detection"})
    artifact = dict(DEFAULT_CONFIG["artifact_detection"])
    supplied = value.get("artifact_detection")
    if isinstance(supplied, dict):
        artifact.update(supplied)
    config["artifact_detection"] = artifact
    return config


def _validate_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("unsupported denoise pipeline schema_version")
    sample_rate = float(config.get("sampling_rate_hz", 0))
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("denoise sampling_rate_hz must be positive")
    band = config.get("bandpass_hz")
    if not isinstance(band, (list, tuple)) or len(band) != 2:
        raise ValueError("denoise bandpass_hz must contain two values")
    low, high = (float(item) for item in band)
    if not 0 < low < high < sample_rate / 2:
        raise ValueError("denoise bandpass_hz must be inside the Nyquist range")
    if int(config.get("bandpass_order", 0)) < 1:
        raise ValueError("denoise bandpass_order must be positive")
    line_noise = config.get("line_noise_hz")
    if not isinstance(line_noise, (list, tuple)) or not line_noise:
        raise ValueError("denoise line_noise_hz must contain at least one frequency")
    if {round(float(item), 6) for item in line_noise} != {50.0, 60.0}:
        raise ValueError("denoise line_noise_hz must contain exactly 50 and 60 Hz")
    for frequency in line_noise:
        if not 0 < float(frequency) < sample_rate / 2:
            raise ValueError("denoise line noise must be inside the Nyquist range")
    if float(config.get("notch_quality_factor", 0)) <= 0:
        raise ValueError("denoise notch_quality_factor must be positive")
    if float(config.get("max_interpolation_s", -1)) < 0:
        raise ValueError("denoise max_interpolation_s must not be negative")


def _load_raw(session_dir: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    path = session_dir / "raw_brainflow.tsv"
    try:
        headers, rows = _read_tsv(path)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"cannot read raw_brainflow.tsv: {error}") from error
    if not headers or not rows:
        raise ValueError("raw_brainflow.tsv is empty or has no header")
    missing = _RAW_REQUIRED - set(headers)
    if missing:
        raise ValueError(f"raw_brainflow.tsv missing columns: {', '.join(sorted(missing))}")
    values: dict[str, list[float]] = {header: [] for header in headers}
    for row in rows:
        for header in headers:
            value = _float(row.get(header, ""))
            values[header].append(np.nan if value is None else value)
    arrays = {key: np.asarray(item, dtype=float) for key, item in values.items()}
    if not arrays["sample_index"].size:
        raise ValueError("raw_brainflow.tsv contains no samples")
    return headers, arrays


def _validate_session_manifest(session_dir: Path) -> Path | None:
    """Validate raw file provenance when an acquisition manifest is present."""

    manifest_path = session_dir / "manifest.csv"
    if not manifest_path.is_file():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"cannot read manifest.csv: {error}") from error
    raw_digest = next((str(row.get("sha256") or "") for row in rows if str(row.get("file_name") or "") == "raw_brainflow.tsv"), "")
    if raw_digest and raw_digest.casefold() != _sha256(session_dir / "raw_brainflow.tsv").casefold():
        raise ValueError("manifest.csv raw_brainflow.tsv hash does not match the session file")
    return manifest_path


def _target_list(protocol: dict[str, Any]) -> list[tuple[str, float]]:
    targets: list[tuple[str, float]] = []
    for item in protocol.get("targets", []):
        if isinstance(item, dict) and item.get("id") is not None and item.get("frequency_hz") is not None:
            targets.append((str(item["id"]), float(item["frequency_hz"])))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            targets.append((str(item[0]), float(item[1])))
    if not targets:
        raise ValueError("protocol contains no analyzable targets")
    return targets


def _contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or not np.any(mask):
        return []
    padded = np.concatenate(([False], mask, [False]))
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    ends = np.flatnonzero(~padded[1:] & padded[:-1])
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _robust_z(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.zeros_like(values, dtype=float)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    if scale <= np.finfo(float).eps:
        scale = float(np.std(finite))
    if scale <= np.finfo(float).eps:
        return np.zeros_like(values, dtype=float)
    return np.abs((values - median) / scale)


def _timestamp_for_sample(timestamps: np.ndarray | None, index: int, sample_rate: float) -> float:
    if timestamps is not None and 0 <= index < timestamps.size and np.isfinite(timestamps[index]):
        return float(timestamps[index])
    return float(index / sample_rate)


def _record_segments(
    records: list[dict[str, Any]],
    mask: np.ndarray,
    *,
    channel: str,
    artifact_type: str,
    source: str,
    severity: str,
    timestamps: np.ndarray | None,
    sample_rate: float,
) -> None:
    for start, end in _contiguous_segments(mask):
        records.append(
            {
                "channel": channel,
                "artifact_type": artifact_type,
                "source": source,
                "severity": severity,
                "start_sample": start,
                "end_sample_exclusive": end,
                "sample_count": end - start,
                "start_time_s": round(_timestamp_for_sample(timestamps, start, sample_rate), 9),
                "end_time_s": round(_timestamp_for_sample(timestamps, max(start, end - 1), sample_rate), 9),
            }
        )


def _detect_artifacts(
    eeg: np.ndarray,
    raw: dict[str, np.ndarray],
    sample_rate: float,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    detection = config["artifact_detection"]
    channel_count, sample_count = eeg.shape
    artifact_mask = np.zeros((channel_count, sample_count), dtype=bool)
    records: list[dict[str, Any]] = []
    timestamps = raw.get("timestamp_s")
    flat_fraction_limit = float(detection["flatline_fraction"])
    z_threshold = float(detection["robust_z_threshold"])
    window = max(1, round(float(detection["window_s"]) * sample_rate))
    step = max(1, round(float(detection["step_s"]) * sample_rate))

    for channel_index in range(channel_count):
        values = eeg[channel_index]
        name = f"CH{channel_index + 1}"
        nonfinite = ~np.isfinite(values)
        _record_segments(records, nonfinite, channel=name, artifact_type="nonfinite", source="rule", severity="error", timestamps=timestamps, sample_rate=sample_rate)
        artifact_mask[channel_index, nonfinite] = True

        finite = np.isfinite(values)
        differences = np.diff(values, prepend=values[0] if values.size else 0.0)
        flat = finite & (np.abs(differences) <= 1e-12)
        for start, end in _contiguous_segments(flat):
            if end - start >= max(2, round(flat_fraction_limit * sample_rate)):
                _record_segments(records, np.arange(sample_count) >= start, channel=name, artifact_type="flatline", source="rule", severity="warning", timestamps=timestamps, sample_rate=sample_rate)
                # Replace the just-created overlong mask with the exact segment.
                records.pop()
                records.append({
                    "channel": name, "artifact_type": "flatline", "source": "rule", "severity": "warning",
                    "start_sample": start, "end_sample_exclusive": end, "sample_count": end - start,
                    "start_time_s": round(_timestamp_for_sample(timestamps, start, sample_rate), 9),
                    "end_time_s": round(_timestamp_for_sample(timestamps, max(start, end - 1), sample_rate), 9),
                })
                artifact_mask[channel_index, start:end] = True

        outlier = finite & (_robust_z(values) > z_threshold)
        _record_segments(records, outlier, channel=name, artifact_type="amplitude_outlier", source="rule", severity="warning", timestamps=timestamps, sample_rate=sample_rate)
        artifact_mask[channel_index, outlier] = True

        # A high-frequency power ratio is evaluated on overlapping windows so
        # transient muscle activity is retained as a reviewable segment.
        high_low, high_high = (float(item) for item in detection["high_frequency_band_hz"])
        for start in range(0, sample_count, step):
            end = min(sample_count, start + window)
            if end - start < max(16, round(sample_rate / 2)):
                continue
            segment = values[start:end]
            if not np.all(np.isfinite(segment)):
                continue
            frequencies, power = signal.welch(segment, fs=sample_rate, nperseg=min(segment.size, max(32, round(sample_rate))))
            total_band = (frequencies >= 1.0) & (frequencies <= min(45.0, sample_rate / 2.0))
            high_band = (frequencies >= high_low) & (frequencies <= min(high_high, sample_rate / 2.0))
            denominator = float(np.sum(power[total_band]))
            ratio = float(np.sum(power[high_band]) / denominator) if denominator > 0 else 0.0
            if ratio >= float(detection["high_frequency_ratio"]):
                mask = np.zeros(sample_count, dtype=bool)
                mask[start:end] = True
                _record_segments(records, mask, channel=name, artifact_type="high_frequency", source="rule", severity="warning", timestamps=timestamps, sample_rate=sample_rate)
                artifact_mask[channel_index, start:end] = True

    acceleration = [raw.get(f"accel_{axis}") for axis in ("x", "y", "z")]
    if all(item is not None and item.size == sample_count for item in acceleration):
        accel_matrix = np.vstack([item for item in acceleration])
        motion_signal = np.sqrt(np.sum(np.square(np.diff(accel_matrix, axis=1, prepend=accel_matrix[:, :1])), axis=0))
        motion = _robust_z(motion_signal) > float(detection["motion_z_threshold"])
        motion_mask = np.zeros(sample_count, dtype=bool)
        for start, end in _contiguous_segments(motion):
            expanded_start = max(0, start - round(0.25 * sample_rate))
            expanded_end = min(sample_count, end + round(0.25 * sample_rate))
            motion_mask[expanded_start:expanded_end] = True
        if np.any(motion_mask):
            _record_segments(records, motion_mask, channel="ALL", artifact_type="motion", source="accelerometer", severity="warning", timestamps=timestamps, sample_rate=sample_rate)
            artifact_mask[:, motion_mask] = True

    records.sort(key=lambda item: (int(item["start_sample"]), str(item["channel"]), str(item["artifact_type"])))
    return artifact_mask, records


def _interpolate_for_filter(values: np.ndarray, max_gap_samples: int) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    if np.all(finite):
        return values.astype(float, copy=True), np.zeros(values.size, dtype=bool)
    if not np.any(finite):
        return np.zeros_like(values, dtype=float), np.ones(values.size, dtype=bool)
    indexes = np.arange(values.size, dtype=float)
    filled = np.interp(indexes, indexes[finite], values[finite]).astype(float, copy=False)
    long_gap = np.zeros(values.size, dtype=bool)
    for start, end in _contiguous_segments(~finite):
        if end - start > max_gap_samples:
            long_gap[start:end] = True
    return filled, long_gap


def _filter_track(eeg: np.ndarray, sample_rate: float, config: dict[str, Any], line_noise_hz: float) -> tuple[np.ndarray, np.ndarray]:
    low, high = (float(item) for item in config["bandpass_hz"])
    bandpass = signal.butter(int(config["bandpass_order"]), [low, high], btype="bandpass", fs=sample_rate, output="sos")
    notch_b, notch_a = signal.iirnotch(float(line_noise_hz), float(config["notch_quality_factor"]), fs=sample_rate)
    notch = signal.tf2sos(notch_b, notch_a)
    sos = np.vstack((notch, bandpass))
    max_gap_samples = max(0, round(float(config["max_interpolation_s"]) * sample_rate))
    output = np.empty_like(eeg, dtype=float)
    long_gap_mask = np.zeros_like(eeg, dtype=bool)
    for channel_index, values in enumerate(eeg):
        filled, long_gap = _interpolate_for_filter(values, max_gap_samples)
        if filled.size <= 3 * max(1, sos.shape[0]):
            raise ValueError("EEG session is too short for zero-phase denoising")
        padlen = min(filled.size - 1, max(0, 3 * (2 * sos.shape[0] + 1)))
        filtered = signal.sosfiltfilt(sos, filled, padlen=padlen)
        filtered[long_gap] = np.nan
        output[channel_index] = filtered
        long_gap_mask[channel_index] = long_gap
    return output, long_gap_mask


def _psd_metrics(eeg: np.ndarray, sample_rate: float, targets: list[tuple[str, float]], line_noise: Iterable[float]) -> dict[str, Any]:
    channel_metrics: dict[str, Any] = {}
    line_values: dict[str, list[float]] = {str(float(frequency)): [] for frequency in line_noise}
    target_values: dict[str, list[float]] = {target_id: [] for target_id, _ in targets}
    for channel_index, values in enumerate(eeg):
        finite = values[np.isfinite(values)]
        if finite.size < 16:
            channel_metrics[f"CH{channel_index + 1}"] = {"valid_fraction": float(finite.size / max(1, values.size)), "rms": None, "flat_fraction": 1.0, "saturation_fraction": None}
            continue
        frequencies, power = signal.welch(finite, fs=sample_rate, nperseg=min(finite.size, max(32, round(sample_rate * 4))))
        band = (frequencies >= 1.0) & (frequencies <= min(45.0, sample_rate / 2.0))
        total_power = float(np.median(power[band])) if np.any(band) else 0.0
        differences = np.diff(finite)
        flat_fraction = float(np.mean(np.abs(differences) <= 1e-12)) if differences.size else 1.0
        channel_metrics[f"CH{channel_index + 1}"] = {
            "valid_fraction": float(finite.size / max(1, values.size)),
            "rms": float(np.sqrt(np.mean(np.square(finite)))),
            "flat_fraction": flat_fraction,
            "saturation_fraction": float(np.mean(np.abs(finite) >= 24_000.0)),
            "psd_frequency_hz": frequencies.tolist(),
            "psd_power": power.tolist(),
        }
        for frequency in line_noise:
            index = int(np.argmin(np.abs(frequencies - float(frequency))))
            target_power = float(power[index])
            neighbor = (frequencies >= float(frequency) - 2.0) & (frequencies <= float(frequency) + 2.0)
            neighbor[index] = False
            baseline = float(np.median(power[neighbor])) if np.any(neighbor) else max(total_power, 1e-18)
            line_values[str(float(frequency))].append(float(target_power / max(baseline, 1e-18)))
        for target_id, frequency in targets:
            index = int(np.argmin(np.abs(frequencies - frequency)))
            target_power = float(power[index])
            neighbor = (frequencies >= frequency - 2.0) & (frequencies <= frequency + 2.0)
            neighbor &= np.abs(frequencies - frequency) > max(0.75, sample_rate / max(1, finite.size))
            baseline = float(np.median(power[neighbor])) if np.any(neighbor) else max(total_power, 1e-18)
            target_values[target_id].append(float(10.0 * np.log10(max(target_power, 1e-18) / max(baseline, 1e-18))))
    return {
        "channels": channel_metrics,
        "line_noise_ratio": {key: float(np.median(values)) if values else None for key, values in line_values.items()},
        "target_snr_db": {key: float(np.median(values)) if values else None for key, values in target_values.items()},
    }


def _trial_snr_metrics(
    eeg: np.ndarray,
    sample_rate: float,
    protocol: dict[str, Any],
    events_source: list[dict[str, str]],
    artifact_mask: np.ndarray | None,
    marker: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Return per-trial target SNR and validity summaries for audit reports."""

    targets = _target_list(protocol)
    events = [dict(item) for item in events_source]
    _marker_sample_indexes(events, marker)
    results: list[dict[str, Any]] = []
    for onset in (event for event in events if event.get("event_name") == "stimulus_onset"):
        trial_index = onset.get("trial_index", "")
        start = _int(onset.get("sample_index"))
        if start is None:
            continue
        offset = next((event for event in events if event.get("event_name") == "stimulus_offset" and event.get("trial_index") == trial_index), None)
        end = _int(offset.get("sample_index")) if offset else None
        if end is None or end <= start:
            duration = float(protocol.get("stimulus_s") or protocol.get("trial", {}).get("stimulus_s", 5.0))
            end = start + round(duration * sample_rate)
        end = min(end, eeg.shape[1])
        trim = min(round(0.5 * sample_rate), max(0, (end - start) // 5))
        window_start = min(start + trim, max(start, end - 16))
        window = eeg[:, window_start:end]
        snr_by_target: dict[str, float | None] = {}
        for target_id, frequency in targets:
            channel_values: list[float] = []
            for values in window:
                finite = values[np.isfinite(values)]
                if finite.size < 16:
                    continue
                frequencies, power = signal.welch(finite, fs=sample_rate, nperseg=min(finite.size, max(32, round(sample_rate * 4))))
                target_index = int(np.argmin(np.abs(frequencies - frequency)))
                neighbor = (frequencies >= frequency - 2.0) & (frequencies <= frequency + 2.0)
                neighbor &= np.abs(frequencies - frequency) > max(0.75, sample_rate / max(1, finite.size))
                baseline = float(np.median(power[neighbor])) if np.any(neighbor) else 1e-18
                channel_values.append(float(10.0 * np.log10(max(float(power[target_index]), 1e-18) / max(baseline, 1e-18))))
            snr_by_target[target_id] = float(np.median(channel_values)) if channel_values else None
        results.append({
            "trial_index": trial_index,
            "window_start_sample_index": window_start,
            "window_samples": max(0, end - window_start),
            "artifact_rejected": int(artifact_mask is not None and np.any(artifact_mask[:, window_start:end])),
            "target_snr_db": snr_by_target,
        })
    return results


def _analyze_array(
    eeg: np.ndarray,
    sample_rate: float,
    protocol: dict[str, Any],
    events_source: list[dict[str, str]],
    artifact_mask: np.ndarray | None,
    marker: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    targets = _target_list(protocol)
    events = [dict(item) for item in events_source]
    _marker_sample_indexes(events, marker)
    rows: list[dict[str, Any]] = []
    onsets = [event for event in events if event.get("event_name") == "stimulus_onset"]
    target_by_frequency = {round(frequency, 6): target_id for target_id, frequency in targets}
    for onset in onsets:
        trial_index = onset.get("trial_index", "")
        start = _int(onset.get("sample_index"))
        if start is None:
            continue
        offset = next((event for event in events if event.get("event_name") == "stimulus_offset" and event.get("trial_index") == trial_index), None)
        end = _int(offset.get("sample_index")) if offset else None
        if end is None or end <= start:
            duration = float(protocol.get("stimulus_s") or protocol.get("trial", {}).get("stimulus_s", 5.0))
            end = start + round(duration * sample_rate)
        end = min(end, eeg.shape[1])
        trim = min(round(0.5 * sample_rate), max(0, (end - start) // 5))
        window_start = min(start + trim, max(start, end - 16))
        window = eeg[:, window_start:end]
        rejected = bool(artifact_mask is not None and np.any(artifact_mask[:, window_start:end]))
        try:
            score = _ssvep_score(window, sample_rate, targets)
        except ValueError:
            continue
        presented = onset.get("presented_target_id") or onset.get("target_id") or ""
        if not presented:
            presented = target_by_frequency.get(round(_float(onset.get("frequency_hz")) or 0.0, 6), "")
        gaze_marks = [event for event in events if event.get("event_name") == "gaze_target_mark" and event.get("trial_index") == trial_index and event.get("gaze_target_id")]
        gaze = gaze_marks[-1].get("gaze_target_id", "") if gaze_marks else ""
        rows.append({
            "trial_index": trial_index,
            "onset_sample_index": start,
            "offset_sample_index": end,
            "window_start_sample_index": window_start,
            "window_samples": score["samples"],
            "presented_target_id": presented,
            "gaze_target_id": gaze,
            "eeg_predicted_target_id": score["predicted_target_id"],
            "eeg_confidence": round(score["confidence"], 6),
            "eeg_relative_score": round(score["relative_score"], 6),
            "presented_target_match": int(bool(presented and presented == score["predicted_target_id"])),
            "gaze_target_match": int(bool(gaze and gaze == score["predicted_target_id"])),
            "artifact_rejected": int(rejected),
            "scores_json": json.dumps(score["scores"], ensure_ascii=False, sort_keys=True),
        })
    presented_matches = [row["presented_target_match"] for row in rows]
    gaze_rows = [row for row in rows if row["gaze_target_id"]]
    return {
        "schema_version": 1,
        "analysis": "fft_ssvep_baseline",
        "sample_rate_hz": sample_rate,
        "trial_count": len(rows),
        "presented_target_accuracy": float(np.mean(presented_matches)) if presented_matches else None,
        "gaze_mark_count": len(gaze_rows),
        "gaze_target_accuracy": float(np.mean([row["gaze_target_match"] for row in gaze_rows])) if gaze_rows else None,
        "rejected_trial_count": int(sum(row["artifact_rejected"] for row in rows)),
        "label_definitions": {
            "presented_target_id": "Target rendered by the protocol; not proof of gaze.",
            "gaze_target_id": "Manual/self-report annotation; not an eye tracker measurement.",
            "eeg_predicted_target_id": "FFT score winner from EEG; an inference requiring validation.",
        },
        "limitations": ["A predicted target is not proof that the participant looked at that target."],
    }, rows


def _write_features(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "trial_index", "onset_sample_index", "offset_sample_index", "window_start_sample_index", "window_samples",
        "presented_target_id", "gaze_target_id", "eeg_predicted_target_id", "eeg_confidence", "eeg_relative_score",
        "presented_target_match", "gaze_target_match", "artifact_rejected", "scores_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_artifacts(path: Path, records: list[dict[str, Any]]) -> None:
    fields = ["channel", "artifact_type", "source", "severity", "start_sample", "end_sample_exclusive", "sample_count", "start_time_s", "end_time_s"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)


def _write_npz(path: Path, eeg: np.ndarray, sample_rate: float, raw: dict[str, np.ndarray], artifact_mask: np.ndarray, channel_names: list[str]) -> None:
    np.savez_compressed(path, eeg=eeg, sample_rate_hz=np.asarray(sample_rate), sample_index=raw["sample_index"], timestamp_s=raw.get("timestamp_s", np.arange(eeg.shape[1]) / sample_rate), artifact_mask=artifact_mask, channel_names=np.asarray(channel_names))


def _manifest(path: Path, session_id: str, input_hashes: dict[str, str], files: list[Path]) -> None:
    fields = ["session_id", "role", "file_name", "size_bytes", "sha256", "source_input_sha256"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, digest in input_hashes.items():
            writer.writerow({"session_id": session_id, "role": "input", "file_name": name, "size_bytes": "", "sha256": digest, "source_input_sha256": digest})
        for item in files:
            writer.writerow({"session_id": session_id, "role": "derived", "file_name": item.name, "size_bytes": item.stat().st_size, "sha256": _sha256(item), "source_input_sha256": _json_sha256(input_hashes)})


def _cached_result(output_dir: Path, input_hashes: dict[str, str], config_sha: str) -> dict[str, Any] | None:
    preprocessing = output_dir / "preprocessing.json"
    metrics = output_dir / "denoising_metrics.json"
    if not preprocessing.is_file() or not metrics.is_file():
        return None
    try:
        metadata = _load_json(preprocessing)
        summary = _load_json(metrics)
    except ValueError:
        return None
    if metadata.get("input_hashes") != input_hashes or metadata.get("pipeline_sha256") != config_sha:
        return None
    required = ["artifact_segments.tsv", "annotated_raw.npz", "denoised_50hz.npz", "denoised_60hz.npz", "trial_features_annotated.tsv", "trial_features_50hz.tsv", "trial_features_60hz.tsv", "analysis_annotated.json", "analysis_50hz.json", "analysis_60hz.json", "derived_manifest.csv"]
    if not all((output_dir / item).is_file() for item in required):
        return None
    return {**summary, "status": "up_to_date", "output_dir": str(output_dir)}


def run_denoise_pipeline(session_dir: Path, pipeline_path: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """Run the configured denoising pipeline without changing raw session files."""

    session_dir = Path(session_dir).expanduser().resolve()
    pipeline_path = Path(pipeline_path).expanduser().resolve()
    output_dir = Path(output_dir or session_dir / "derived" / PIPELINE_VERSION).expanduser().resolve()
    if not session_dir.is_dir():
        raise ValueError(f"session directory was not found: {session_dir}")
    if not pipeline_path.is_file():
        raise ValueError(f"pipeline configuration was not found: {pipeline_path}")
    input_paths = [session_dir / name for name in _SESSION_INPUTS]
    for path in input_paths:
        if not path.is_file():
            raise ValueError(f"required session file was not found: {path.name}")
    raw_headers, raw = _load_raw(session_dir)
    raw_column_fields, raw_column_rows = _read_tsv(session_dir / "raw_columns.tsv")
    if not raw_column_fields:
        raise ValueError("raw_columns.tsv has no header")
    declared_columns = {str(row.get("column_name") or "") for row in raw_column_rows}
    if declared_columns and not set(raw_headers).issubset(declared_columns):
        missing = sorted(set(raw_headers) - declared_columns)
        raise ValueError(f"raw_columns.tsv is missing raw columns: {', '.join(missing)}")
    session = _load_json(session_dir / "session.json")
    events_fields, events = _read_tsv(session_dir / "events.tsv")
    if not events_fields:
        raise ValueError("events.tsv has no header")
    protocol_path = session_dir / "protocol.json"
    if not protocol_path.is_file():
        protocol_path = session_dir / "ssvep_config.json"
    if not protocol_path.is_file():
        raise ValueError("protocol.json or ssvep_config.json was not found")
    protocol = _load_json(protocol_path)
    config = _merged_config(_load_json(pipeline_path))
    _validate_config(config)
    input_paths.append(protocol_path)
    manifest_path = _validate_session_manifest(session_dir)
    if manifest_path is not None:
        input_paths.append(manifest_path)
    input_hashes = {path.name: _sha256(path) for path in input_paths}
    config_sha = _sha256(pipeline_path)
    cached = _cached_result(output_dir, input_hashes, config_sha)
    if cached is not None:
        return cached

    sample_rate = float(session.get("sampling_rate_hz") or protocol.get("sampling_rate_hz") or protocol.get("hardware", {}).get("sampling_rate_hz", config["sampling_rate_hz"]))
    if not np.isclose(sample_rate, float(config["sampling_rate_hz"])):
        raise ValueError(f"sampling rate mismatch: session={sample_rate:g}, pipeline={float(config['sampling_rate_hz']):g}")
    eeg_names = sorted((name for name in raw_headers if re.fullmatch(r"eeg_ch\d+", name)), key=lambda name: int(re.search(r"\d+", name).group()))
    if not eeg_names:
        raise ValueError("raw_brainflow.tsv has no eeg_chN columns")
    eeg = np.vstack([raw[name] for name in eeg_names])
    if eeg.shape[1] < max(32, round(sample_rate)):
        raise ValueError("session contains too few samples for denoising")
    artifact_mask, artifact_records = _detect_artifacts(eeg, raw, sample_rate, config)
    line_noise = [float(item) for item in config["line_noise_hz"]]
    marker = raw.get("marker")
    annotated_analysis, annotated_rows = _analyze_array(eeg, sample_rate, protocol, events, artifact_mask, marker)
    filtered_tracks: dict[str, np.ndarray] = {}
    analyses: dict[str, dict[str, Any]] = {"annotated": annotated_analysis}
    rows_by_track: dict[str, list[dict[str, Any]]] = {"annotated": annotated_rows}
    long_gap_masks: dict[str, np.ndarray] = {}
    for frequency in line_noise:
        key = f"{int(frequency)}hz"
        filtered, long_gap = _filter_track(eeg, sample_rate, config, frequency)
        filtered_tracks[key] = filtered
        long_gap_masks[key] = long_gap
        analyses[key], rows_by_track[key] = _analyze_array(filtered, sample_rate, protocol, events, artifact_mask | long_gap, marker)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_artifacts(output_dir / "artifact_segments.tsv", artifact_records)
    _write_npz(output_dir / "annotated_raw.npz", eeg, sample_rate, raw, artifact_mask, eeg_names)
    for key, values in filtered_tracks.items():
        _write_npz(output_dir / f"denoised_{key}.npz", values, sample_rate, raw, artifact_mask | long_gap_masks[key], eeg_names)
    _write_features(output_dir / "trial_features_annotated.tsv", annotated_rows)
    _write_features(output_dir / "trial_features_50hz.tsv", rows_by_track["50hz"])
    _write_features(output_dir / "trial_features_60hz.tsv", rows_by_track["60hz"])
    for key, analysis in analyses.items():
        analysis["session_id"] = session.get("session_id", session_dir.name)
        analysis["eeg_columns"] = eeg_names
        analysis["analysis_track"] = key
        analysis["trial_features"] = str(output_dir / f"trial_features_{'annotated' if key == 'annotated' else key}.tsv")
        (output_dir / f"analysis_{key}.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics: dict[str, Any] = {
        "schema_version": 1,
        "pipeline": PIPELINE_VERSION,
        "session_id": session.get("session_id", session_dir.name),
        "sample_rate_hz": sample_rate,
        "eeg_columns": eeg_names,
        "artifact_segment_count": len(artifact_records),
        "artifact_sample_count": int(np.count_nonzero(artifact_mask)),
        "artifact_fraction": float(np.mean(artifact_mask)) if artifact_mask.size else 0.0,
        "tracks": {
            "annotated": _psd_metrics(eeg, sample_rate, _target_list(protocol), line_noise),
            "50hz": _psd_metrics(filtered_tracks["50hz"], sample_rate, _target_list(protocol), line_noise),
            "60hz": _psd_metrics(filtered_tracks["60hz"], sample_rate, _target_list(protocol), line_noise),
        },
        "analysis": {key: {item: value for item, value in report.items() if item in {"trial_count", "presented_target_accuracy", "gaze_mark_count", "gaze_target_accuracy", "rejected_trial_count"}} for key, report in analyses.items()},
        "label_definitions": analyses["annotated"]["label_definitions"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    }
    metrics["tracks"]["annotated"]["trial_metrics"] = _trial_snr_metrics(eeg, sample_rate, protocol, events, artifact_mask, marker)
    metrics["tracks"]["50hz"]["trial_metrics"] = _trial_snr_metrics(filtered_tracks["50hz"], sample_rate, protocol, events, artifact_mask | long_gap_masks["50hz"], marker)
    metrics["tracks"]["60hz"]["trial_metrics"] = _trial_snr_metrics(filtered_tracks["60hz"], sample_rate, protocol, events, artifact_mask | long_gap_masks["60hz"], marker)
    metrics_path = output_dir / "denoising_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    preprocessing = {
        "schema_version": 1,
        "pipeline": PIPELINE_VERSION,
        "pipeline_config": config,
        "pipeline_sha256": config_sha,
        "input_hashes": input_hashes,
        "session_id": session.get("session_id", session_dir.name),
        "source_session_dir": str(session_dir),
        "output_dir": str(output_dir),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "tracks": {"annotated": "raw values with artifact mask", "50hz": "zero-phase bandpass plus 50 Hz notch", "60hz": "zero-phase bandpass plus 60 Hz notch"},
    }
    (output_dir / "preprocessing.json").write_text(json.dumps(preprocessing, ensure_ascii=False, indent=2), encoding="utf-8")
    output_files = [path for path in sorted(output_dir.iterdir()) if path.is_file() and path.name != "derived_manifest.csv"]
    _manifest(output_dir / "derived_manifest.csv", str(session.get("session_id", session_dir.name)), input_hashes, output_files)
    metrics["status"] = "completed"
    metrics["output_dir"] = str(output_dir)
    metrics["generated_files"] = [path.name for path in output_files] + ["derived_manifest.csv"]
    return metrics


__all__ = ["DEFAULT_CONFIG", "PIPELINE_VERSION", "run_denoise_pipeline"]
