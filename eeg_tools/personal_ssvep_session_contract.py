"""Read-only personal SSVEP label, timing and source-file contract audit.

A contract check never verifies physiological quality, gaze authenticity or
model performance. All runtime and independent-validation flags stay false.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


class PersonalSSVEPSessionContractError(ValueError):
    """The native session cannot be read safely and unambiguously."""


CORE_EVENT_FIELDS = {
    "event_name", "trial_index", "target_id", "presented_target_id",
    "gaze_target_id", "source", "label_source", "sample_index", "frequency_hz",
}
REQUIRED_FILES = (
    "session.json", "channel_config.json", "ssvep_config.json",
    "events.tsv", "raw_brainflow.tsv", "quality.json",
)
OPTIONAL_FILES = ("preflight.json", "gaze_truth.json")
GAZE_EVIDENCE_ROLES = {"raw_gaze", "calibration", "synchronization", "target_mapping"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PersonalSSVEPSessionContractError(message)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise PersonalSSVEPSessionContractError(f"nonfinite JSON constant: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"),
                       object_pairs_hook=_unique_keys, parse_constant=_invalid_constant)
    _require(isinstance(value, dict), f"{path.name} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_file(root: Path, name: str) -> Path:
    _require(isinstance(name, str) and bool(name), "evidence path must be nonempty")
    relative = Path(name)
    _require(not relative.is_absolute() and ".." not in relative.parts,
             "evidence path must be relative and inside the session")
    path = (root / relative).resolve()
    _require(path.is_relative_to(root) and path.is_file(),
             f"missing regular file or evidence outside session: {name}")
    return path


def _index(value: Any) -> int | None:
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value):
        return int(value)
    return None


def _table(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", strict=True)
        header = next(reader, [])
        _require(bool(header) and all(header) and len(set(header)) == len(header),
                 f"{path.name}: unique nonempty column names required")
        yield header
        for number, row in enumerate(reader, 2):
            _require(len(row) == len(header), f"{path.name}: malformed row {number}; no repair performed")
            yield row


def _raw_summary(path: Path) -> dict:
    rows = _table(path)
    header = next(rows)
    eeg = [f"eeg_ch{i}" for i in range(1, 9)]
    _require("sample_index" in header and {x for x in header if x.startswith("eeg_ch")} == set(eeg),
             "raw_brainflow.tsv: sample_index and exactly eeg_ch1..8 are required")
    index_column = header.index("sample_index")
    eeg_columns = [header.index(name) for name in eeg]
    count = 0
    nonfinite_rows = 0
    for row in rows:
        _require(_index(row[index_column]) == count, "raw sample_index must be contiguous 0..N-1")
        try:
            values = [float(row[column]) for column in eeg_columns]
        except ValueError as exc:
            raise PersonalSSVEPSessionContractError("raw EEG contains nonnumeric samples") from exc
        nonfinite_rows += int(not all(math.isfinite(value) for value in values))
        count += 1
    return {"sample_count": count, "nonfinite_eeg_rows": nonfinite_rows,
            "signal_quality_accepted": False, "packet_continuity_assessed": False}


def _gaze_evidence(root: Path, session: dict, trials: dict, targets: dict, receipts: dict) -> dict:
    result = {"available": "gaze_truth.json" in receipts, "independent": False,
              "independent_declared": False, "trial_label_count": 0,
              "evidence_contract_passed": False, "issues": []}
    if not result["available"]:
        result["issues"].append("no gaze_truth.json; independent gaze evidence unavailable")
        return result
    truth = _read_json(root / "gaze_truth.json")
    issues = result["issues"]
    result["source_type"] = truth.get("source_type")
    result["independent_declared"] = truth.get("independent") is True
    if type(truth.get("schema_version")) is not int or truth["schema_version"] != 1:
        issues.append("gaze schema_version must be integer 1")
    if not session.get("session_id") or truth.get("session_id") != session["session_id"]:
        issues.append("gaze session_id must match the EEG session")
    if truth.get("source_type") != "eye_tracker" or truth.get("independent") is not True:
        issues.append("eye_tracker source and boolean independent declaration required")
    labels = truth.get("trial_labels")
    seen = set()
    if not isinstance(labels, list):
        issues.append("trial_labels must be a list")
        labels = []
    for label in labels:
        if not isinstance(label, dict):
            issues.append("malformed gaze trial label")
            continue
        index = _index(label.get("trial_index"))
        start, end = _index(label.get("start_sample_index")), _index(label.get("end_sample_index"))
        if index not in trials or index in seen:
            issues.append("gaze trial_index is unknown or duplicated")
            continue
        seen.add(index)
        onset, offset = trials[index]
        if label.get("target_id") not in targets or start is None or end is None or not (onset <= start < end <= offset):
            issues.append(f"invalid gaze target or sample interval for trial {index}")
            continue
        result["trial_label_count"] += 1
    if not trials or result["trial_label_count"] != len(trials) or seen != set(trials):
        issues.append("gaze labels must cover every paired trial exactly once")
    evidence = truth.get("evidence_files")
    roles = set()
    paths = set()
    if not isinstance(evidence, list):
        issues.append("evidence_files must include raw gaze, calibration, synchronization and target mapping")
        evidence = []
    for item in evidence:
        if not isinstance(item, dict):
            issues.append("malformed gaze evidence file entry")
            continue
        role, name, expected = item.get("role"), item.get("path"), item.get("sha256")
        if not isinstance(role, str) or role not in GAZE_EVIDENCE_ROLES or role in roles:
            issues.append("unknown or duplicated gaze evidence role")
            continue
        path = _local_file(root, name)
        _require(path not in paths and path not in {(root / n).resolve() for n in REQUIRED_FILES + OPTIONAL_FILES},
                 "gaze evidence roles must reference separate evidence files")
        paths.add(path)
        _require(path.stat().st_size > 0, f"empty gaze evidence file: {name}")
        actual = _sha256(path)
        key = path.relative_to(root).as_posix()
        receipts[key] = actual
        if not isinstance(expected, str) or expected != actual:
            issues.append(f"gaze evidence hash mismatch: {name}")
        else:
            roles.add(role)
    if roles != GAZE_EVIDENCE_ROLES:
        issues.append("all four gaze evidence roles must have matching file hashes")
    result["evidence_contract_passed"] = not issues
    result["verification_status"] = "independent_review_required"
    # Presence and hash checks cannot establish calibration/synchronization
    # validity, label authenticity, or the subject's intention.
    return result


def audit_session(session_dir: str | Path, *, minimum_repetitions_per_target: int = 8) -> dict[str, Any]:
    root = Path(session_dir).expanduser().resolve()
    _require(root.is_dir(), f"session directory does not exist: {root}")
    _require(type(minimum_repetitions_per_target) is int and minimum_repetitions_per_target >= 1,
             "minimum_repetitions_per_target must be a positive integer")
    paths = {name: _local_file(root, name) for name in REQUIRED_FILES}
    optional_presence = {name: (root / name).exists() for name in OPTIONAL_FILES}
    paths.update({name: _local_file(root, name) for name, present in optional_presence.items() if present})
    before = {name: _sha256(path) for name, path in paths.items()}
    session = _read_json(paths["session.json"])
    channels = _read_json(paths["channel_config.json"])
    quality = _read_json(paths["quality.json"])
    protocol = _read_json(paths["ssvep_config.json"])
    preflight = _read_json(paths["preflight.json"]) if "preflight.json" in paths else session.get("preflight")
    preflight = preflight if isinstance(preflight, dict) else {}
    raw = _raw_summary(paths["raw_brainflow.tsv"])
    event_rows = _table(paths["events.tsv"])
    fields = next(event_rows)
    _require(CORE_EVENT_FIELDS.issubset(fields), "events.tsv is missing required label/event fields")
    events = [dict(zip(fields, row)) for row in event_rows]
    target_list = protocol.get("targets")
    _require(isinstance(target_list, list) and bool(target_list), "protocol targets must be a nonempty list")
    targets = {}
    for target in target_list:
        _require(isinstance(target, dict) and isinstance(target.get("id"), str) and bool(target["id"]),
                 "protocol target must have a nonempty string id")
        frequency = target.get("frequency_hz")
        _require(type(frequency) in (int, float) and math.isfinite(frequency) and frequency > 0,
                 "protocol target frequency must be finite and positive")
        _require(target["id"] not in targets and frequency not in targets.values(), "protocol targets must be unique")
        targets[target["id"]] = frequency

    onsets = [row for row in events if row["event_name"] == "stimulus_onset"]
    offsets = [row for row in events if row["event_name"] == "stimulus_offset"]
    counts = {target: 0 for target in targets}
    event_issues = []
    grouped = {"onset": {}, "offset": {}}
    for kind, rows in (("onset", onsets), ("offset", offsets)):
        for row in rows:
            index, sample = _index(row["trial_index"]), _index(row["sample_index"])
            target = row["presented_target_id"]
            try:
                frequency = float(row["frequency_hz"])
            except ValueError:
                frequency = None
            if index is None or sample is None or sample >= raw["sample_count"]:
                event_issues.append(f"invalid {kind} trial/sample index")
                continue
            if index in grouped[kind]:
                event_issues.append(f"duplicate {kind} for trial {index}")
                continue
            grouped[kind][index] = (sample, target)
            if target not in targets or target != row["target_id"] or frequency != targets.get(target):
                event_issues.append(f"{kind} target/frequency disagrees with protocol for trial {index}")
            elif kind == "onset":
                counts[target] += 1
    trials = {}
    last_end = -1
    for index in sorted(set(grouped["onset"]) | set(grouped["offset"])):
        onset, offset = grouped["onset"].get(index), grouped["offset"].get(index)
        if onset is None or offset is None or onset[0] >= offset[0] or onset[1] != offset[1]:
            event_issues.append(f"missing, reversed or inconsistent event pair for trial {index}")
            continue
        if onset[0] < last_end:
            event_issues.append(f"overlapping or reordered stimulus intervals for trial {index}")
        last_end = offset[0]
        if offset[0] - onset[0] < 1125:
            event_issues.append(f"trial {index} cannot provide 125-sample offset plus 1000-sample EEG window")
        trials[index] = (onset[0], offset[0])
    manual = [row for row in events if row["event_name"] == "gaze_target_mark" and row["gaze_target_id"]]
    gaze = _gaze_evidence(root, session, trials, targets, before)
    expected_channels = [f"CH{i}" for i in range(1, 9)]
    entries = channels.get("channels")
    entry_names = [entry.get("name") for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []
    checks = {
        "native_cyton": str(session.get("board", "")).casefold() in {"cyton", "openbci cyton"},
        "sampling_rate_250_hz": type(session.get("sampling_rate_hz")) in (int, float) and session["sampling_rate_hz"] == 250,
        "eight_channels": type(session.get("channel_count")) is int and session["channel_count"] == 8,
        "not_simulated": session.get("simulated") is False,
        "session_completed": session.get("status") == "completed",
        "protocol_formal": protocol.get("status") == "formal",
        "channel_order_ch1_to_ch8": channels.get("channel_order") == expected_channels,
        "complete_channel_entries": len(entry_names) == 8 and set(entry_names) == set(expected_channels),
        "raw_nonempty_and_finite": raw["sample_count"] > 0 and raw["nonfinite_eeg_rows"] == 0,
        "recorded_sample_count_matches": type(session.get("recorded_samples_per_channel")) is int and session["recorded_samples_per_channel"] == raw["sample_count"],
        "stimulus_pairs_valid": bool(onsets) and len(onsets) == len(offsets) == len(trials) and not event_issues,
        "minimum_repetitions_each_protocol_target": all(count >= minimum_repetitions_per_target for count in counts.values()),
        "handshake_recorded_passed": any(isinstance(check, dict) and check.get("name") == "handshake" and check.get("status") == "passed" for check in (preflight.get("checks") or [])),
        "quality_status_ok": quality.get("status") == "ok" and quality.get("recording_status") == "completed",
        "display_cadence_pass": isinstance(quality.get("frame_timing"), dict) and quality["frame_timing"].get("cadence_status") == "pass",
        "timestamp_gaps_zero": type(quality.get("timestamp_gap_count")) is int and quality["timestamp_gap_count"] == 0,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    # The proxy candidate means a complete input for offline calibration,
    # never hardware, physiological, model-performance or clinical acceptance.
    for name, present in optional_presence.items():
        _require((root / name).exists() == present, f"optional source changed during audit: {name}")
    after = {name: _sha256(_local_file(root, name)) for name in before}
    _require(before == after, "source session changed during audit")
    return {
        "schema_version": 1, "status": "formal",
        "audit_stage": "personal_ssvep_session_label_and_validation_contract",
        "source_session": str(root), "session_id": session.get("session_id", ""),
        "validation_mode_recorded": session.get("validation_mode"),
        "protocol_status_recorded": protocol.get("status"),
        "minimum_repetitions_per_target": minimum_repetitions_per_target,
        "checks": checks, "blocking_checks": blockers, "raw": raw,
        "events": {"event_count": len(events), "onset_count": len(onsets), "offset_count": len(offsets),
                   "paired_trial_count": len(trials), "target_counts": counts,
                   "structural_issues": event_issues,
                   "manual_gaze_mark_count": len(manual),
                   "manual_gaze_label_sources": sorted({row["label_source"] or row["source"] or "<missing>" for row in manual})},
        "label_evidence": {"presented_target_count": sum(bool(row["presented_target_id"]) for row in onsets),
                           "presented_target_is_proxy": True, "manual_self_report_count": len(manual),
                           "independent_gaze": gaze, "independent_intent_labels_available": False},
        "hardware_evidence": {"physical_electrode_positions_verified_declared":
                              channels.get("physical_electrode_positions_verified") is True or session.get("physical_electrode_positions_verified") is True,
                              "physical_electrode_positions_independently_verified": False},
        "eligibility": {"proxy_calibration_candidate": not blockers,
                        "independent_personal_validation_candidate": False,
                        "runtime_enabled": False, "production_validated": False,
                        "default_deployment_authorized": False},
        "source_files_unchanged": True, "source_sha256": before,
        "limitations": [
            "Presentation and keyboard labels are proxies; this audit never establishes gaze or intention truth.",
            "Gaze file hashes and label intervals establish traceability only; independent evidence review is required.",
            "Recorded quality and handshake statuses are software records, not fresh hardware or clinical acceptance.",
            "Timestamp gaps require review; this audit does not diagnose packet loss or repair EEG.",
            "No model training, inference, deployment, physiological quality or performance acceptance is performed.",
        ],
    }
