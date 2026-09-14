"""Offline, reproducible SSVEP feature extraction and target scoring.

This module deliberately implements a transparent FFT baseline with NumPy
only. It is suitable for audit and technical validation; it is not a clinical
decoder and an EEG frequency peak is not proof of eye position.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _read_raw(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    headers, rows = _read_tsv(path)
    if not headers:
        return [], {}
    values: dict[str, list[float]] = {header: [] for header in headers}
    for row in rows:
        for header in headers:
            value = _float(row.get(header, ""))
            values[header].append(np.nan if value is None else value)
    return headers, {key: np.asarray(value, dtype=float) for key, value in values.items()}


def _marker_sample_indexes(
    events: list[dict[str, str]], marker: np.ndarray | None
) -> None:
    if marker is None or marker.size == 0:
        return
    used: dict[int, int] = {}
    for event in events:
        if event.get("sample_index", "").strip():
            continue
        marker_code = _int(event.get("marker_code"))
        if marker_code is None:
            continue
        matches = np.flatnonzero(np.isclose(marker, marker_code, atol=1e-6))
        offset = used.get(marker_code, 0)
        if offset >= len(matches):
            continue
        event["sample_index"] = str(int(matches[offset]))
        used[marker_code] = offset + 1


def _nearest_power(power: np.ndarray, frequencies: np.ndarray, target: float) -> float:
    if target <= 0 or target >= frequencies[-1]:
        return 0.0
    return float(power[..., int(np.argmin(np.abs(frequencies - target)))].mean())


def _ssvep_score(eeg: np.ndarray, sample_rate_hz: float, targets: list[tuple[str, float]]) -> dict[str, Any]:
    if eeg.ndim != 2 or eeg.shape[1] < max(16, int(sample_rate_hz)):
        raise ValueError("stimulus window is too short for a stable FFT baseline")
    centered = eeg - np.nanmean(eeg, axis=1, keepdims=True)
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
    window = np.hanning(centered.shape[1])
    spectrum = np.fft.rfft(centered * window, axis=1)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(centered.shape[1], d=1.0 / sample_rate_hz)
    valid = (frequencies >= 1.0) & (frequencies <= min(45.0, sample_rate_hz / 2.0))
    scale = np.nanmedian(power[:, valid], axis=1) + 1e-12
    normalized = power / scale[:, None]
    scores: dict[str, float] = {}
    for target_id, frequency in targets:
        fundamental = _nearest_power(normalized, frequencies, frequency)
        harmonic = _nearest_power(normalized, frequencies, 2.0 * frequency)
        scores[target_id] = fundamental + 0.5 * harmonic
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    predicted, best = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else 0.0
    total = sum(scores.values()) + 1e-12
    return {
        "predicted_target_id": predicted,
        "confidence": float(max(0.0, (best - second) / (best + 1e-12))),
        "relative_score": float(best / total),
        "scores": scores,
        "fft_frequency_resolution_hz": float(sample_rate_hz / centered.shape[1]),
        "samples": int(centered.shape[1]),
    }


def analyze_session(session_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """Analyze one saved session and write derived, non-destructive outputs."""

    session_dir = Path(session_dir).expanduser().resolve()
    output_dir = Path(output_dir or session_dir).expanduser().resolve()
    session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    protocol_path = session_dir / "protocol.json"
    if not protocol_path.is_file():
        protocol_path = session_dir / "ssvep_config.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    targets: list[tuple[str, float]] = []
    for item in protocol.get("targets", []):
        if isinstance(item, dict) and item.get("id") and item.get("frequency_hz"):
            targets.append((str(item["id"]), float(item["frequency_hz"])))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            targets.append((str(item[0]), float(item[1])))
    if not targets:
        raise ValueError("protocol contains no analyzable targets")
    sample_rate = float(
        session.get("sampling_rate_hz")
        or protocol.get("sampling_rate_hz")
        or protocol.get("hardware", {}).get("sampling_rate_hz", 250)
    )

    raw_headers, raw = _read_raw(session_dir / "raw_brainflow.tsv")
    eeg_names = sorted(
        (name for name in raw_headers if re.fullmatch(r"eeg_ch\d+", name)),
        key=lambda name: int(re.search(r"\d+", name).group()),
    )
    if not eeg_names:
        raise ValueError("raw_brainflow.tsv has no eeg_chN columns")
    marker = raw.get("marker")
    events_fields, events = _read_tsv(session_dir / "events.tsv")
    del events_fields
    _marker_sample_indexes(events, marker)
    target_by_frequency = {round(frequency, 6): target_id for target_id, frequency in targets}
    onsets = [event for event in events if event.get("event_name") == "stimulus_onset"]
    rows: list[dict[str, Any]] = []
    for onset in onsets:
        trial_index = onset.get("trial_index", "")
        start = _int(onset.get("sample_index"))
        if start is None:
            continue
        offset = next(
            (
                event
                for event in events
                if event.get("event_name") == "stimulus_offset"
                and event.get("trial_index") == trial_index
            ),
            None,
        )
        end = _int(offset.get("sample_index")) if offset else None
        if end is None or end <= start:
            end = start + int(float(protocol.get("stimulus_s", 5.0)) * sample_rate)
        end = min(end, len(raw[eeg_names[0]]))
        trim = min(int(0.5 * sample_rate), max(0, (end - start) // 5))
        window_start = min(start + trim, max(start, end - 16))
        eeg = np.vstack([raw[name][window_start:end] for name in eeg_names])
        score = _ssvep_score(eeg, sample_rate, targets)
        presented = onset.get("presented_target_id") or onset.get("target_id") or ""
        if not presented:
            frequency = _float(onset.get("frequency_hz"))
            presented = target_by_frequency.get(round(frequency or 0.0, 6), "")
        gaze_marks = [
            event
            for event in events
            if event.get("event_name") == "gaze_target_mark"
            and event.get("trial_index") == trial_index
            and event.get("gaze_target_id")
        ]
        gaze = gaze_marks[-1].get("gaze_target_id", "") if gaze_marks else ""
        row = {
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
            "scores_json": json.dumps(score["scores"], ensure_ascii=False, sort_keys=True),
        }
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "trial_features.tsv"
    feature_fields = [
        "trial_index", "onset_sample_index", "offset_sample_index",
        "window_start_sample_index", "window_samples", "presented_target_id",
        "gaze_target_id", "eeg_predicted_target_id", "eeg_confidence",
        "eeg_relative_score", "presented_target_match", "gaze_target_match",
        "scores_json",
    ]
    with feature_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=feature_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    presented_matches = [row["presented_target_match"] for row in rows]
    gaze_rows = [row for row in rows if row["gaze_target_id"]]
    gaze_matches = [row["gaze_target_match"] for row in gaze_rows]
    summary = {
        "schema_version": 1,
        "analysis": "fft_ssvep_baseline",
        "session_id": session.get("session_id", session_dir.name),
        "session_dir": str(session_dir),
        "sample_rate_hz": sample_rate,
        "eeg_columns": eeg_names,
        "trial_count": len(rows),
        "presented_target_accuracy": float(np.mean(presented_matches)) if presented_matches else None,
        "gaze_mark_count": len(gaze_rows),
        "gaze_target_accuracy": float(np.mean(gaze_matches)) if gaze_matches else None,
        "label_definitions": {
            "presented_target_id": "Target rendered by the protocol; not proof of gaze.",
            "gaze_target_id": "Manual/self-report annotation from 1-4 key input; not an eye tracker measurement.",
            "eeg_predicted_target_id": "FFT score winner from EEG; an inference requiring validation.",
        },
        "limitations": [
            "The baseline uses frequency-domain power at the fundamental and second harmonic.",
            "A predicted target is not proof that the participant looked at that target.",
            "For formal evidence, add eye tracking or photodiode/TTL validation and compare against surrogates.",
        ],
        "trial_features": str(feature_path),
    }
    summary_path = output_dir / "analysis.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        result = analyze_session(arguments.session_dir, arguments.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
