from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = ROOT / "analysis_outputs" / "history" / "stimulus_integrity_20261014.json"
DEFAULT_SESSIONS = [
    Path(r"C:\Users\Administrator\Documents\NeuroStation\Datasets\session_20261004_004101_105"),
    Path(r"C:\Users\Administrator\Documents\NeuroStation\Datasets\session_20261004_004555_413"),
]


def sha256_files(source: Path) -> dict[str, str]:
    result = {}
    for path in sorted(source.rglob("*")):
        if path.is_file():
            result[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def read_events(source: Path) -> list[dict[str, str]]:
    with (source / "events.tsv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t", strict=True))


def read_frame_rows(source: Path) -> list[dict[str, str]]:
    with (source / "frame_timing.tsv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t", strict=True))


def trial_report(rows: list[dict[str, str]], event: dict[str, str], refresh_rate_hz: float) -> dict[str, Any]:
    frame_index = np.asarray([int(row["frame_index"]) for row in rows], dtype=int)
    lit = np.asarray([float(row["lit"]) for row in rows], dtype=float)
    actual = np.asarray([float(row["actual_s"]) for row in rows], dtype=float)
    intervals = np.diff(actual)
    spectrum = np.abs(np.fft.rfft(lit - float(np.mean(lit))))
    frequencies = np.fft.rfftfreq(len(lit), d=1.0 / refresh_rate_hz)
    peak_index = int(np.argmax(spectrum[1:]) + 1) if len(spectrum) > 1 else 0
    requested = float(event.get("frequency_hz") or 0.0)
    transitions = int(np.sum(lit[1:] != lit[:-1]))
    duration = float(actual[-1] - actual[0]) if len(actual) >= 2 else 0.0
    transition_estimate = transitions / max(2.0 * duration, 1e-12)
    dominant = float(frequencies[peak_index]) if peak_index else None
    dropped = int(sum(max(0, int(row.get("dropped_since_previous") or 0)) for row in rows))
    median_interval = float(np.median(intervals)) if len(intervals) else None
    return {
        "trial_index": int(event.get("trial_index") or -1),
        "target_id": event.get("presented_target_id") or event.get("target_id") or "",
        "requested_frequency_hz": requested,
        "frame_count": int(len(rows)),
        "first_frame_index": int(frame_index[0]) if len(frame_index) else None,
        "last_frame_index": int(frame_index[-1]) if len(frame_index) else None,
        "observed_duration_s": duration,
        "transition_count": transitions,
        "transition_frequency_estimate_hz": transition_estimate,
        "fft_dominant_frequency_hz": dominant,
        "frequency_error_hz": None if dominant is None else dominant - requested,
        "median_frame_interval_ms": None if median_interval is None else median_interval * 1000.0,
        "dropped_frame_count": dropped,
        "duty_cycle": float(np.mean(lit)) if len(lit) else None,
        "integrity_passed": bool(
            requested > 0
            and dominant is not None
            and abs(dominant - requested) <= max(0.5, refresh_rate_hz / max(len(rows), 1))
            and 0.35 <= float(np.mean(lit)) <= 0.65
            and duration >= 4.5
        ),
    }


def audit_session(source: Path) -> dict[str, Any]:
    before = sha256_files(source)
    session = json.loads((source / "session.json").read_text(encoding="utf-8-sig"))
    display = session.get("display") or {}
    refresh = float(display.get("refresh_rate_hz") or display.get("protocol_refresh_rate_hz") or 60.0)
    events = read_events(source)
    frames = read_frame_rows(source)
    onset_events = {int(row["trial_index"]): row for row in events if row.get("event_name") == "stimulus_onset"}
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in frames:
        grouped.setdefault(int(row["trial_index"]), []).append(row)
    trials = [trial_report(grouped[index], onset_events[index], refresh) for index in sorted(grouped) if index in onset_events]
    after = sha256_files(source)
    if before != after:
        raise RuntimeError(f"source changed during audit: {source}")
    return {
        "source_session": str(source),
        "source_files_unchanged": True,
        "historical_status": session.get("status"),
        "historical_validation_mode": session.get("validation_mode"),
        "refresh_rate_hz": refresh,
        "trial_count": len(trials),
        "integrity_passed_trial_count": sum(item["integrity_passed"] for item in trials),
        "integrity_failed_trial_count": sum(not item["integrity_passed"] for item in trials),
        "trials": trials,
        "all_stimulus_trials_passed": bool(trials) and all(item["integrity_passed"] for item in trials),
    }


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing report: {path}")
    pending = path.with_name(path.name + ".pending")
    with pending.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", action="append", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sessions = [item.resolve() for item in (args.session or DEFAULT_SESSIONS)]
    reports = [audit_session(source) for source in sessions]
    result = {
        "schema_version": 1,
        "status": "formal",
        "completion_status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "read_only_ssvep_frame_timing_integrity_audit",
        "source_mutated": False,
        "historical_conclusions_rewritten": False,
        "runtime_enabled": False,
        "default_deployment_authorized": False,
        "limitations": [
            "Frame timing integrity confirms the visual stimulus schedule, not the participant's gaze or neural response.",
            "A passed stimulus trial does not prove the EEG contains a recoverable SSVEP response.",
            "This audit does not alter the acquisition protocol or runtime.",
        ],
        "sessions": reports,
        "all_sessions_all_trials_passed": bool(reports) and all(item["all_stimulus_trials_passed"] for item in reports),
    }
    write_atomic(args.output.resolve(), result)
    print(json.dumps({"session_count": len(reports), "all_sessions_all_trials_passed": result["all_sessions_all_trials_passed"], "trials": sum(item["trial_count"] for item in reports), "passed": sum(item["integrity_passed_trial_count"] for item in reports)}, ensure_ascii=False, indent=2))
    print(f"report={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
