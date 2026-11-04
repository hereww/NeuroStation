from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(r"C:\Users\Administrator\Documents\NeuroStation\Datasets\Trash\Datasets")
sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "analysis_outputs" / "history" / "trash_ssvep_inventory_20261014.json"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_events(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "parseable": False,
        "row_count": 0,
        "event_name_counts": {},
        "stimulus_onset_count": 0,
        "stimulus_offset_count": 0,
        "paired_trial_count": 0,
        "unpaired_onset_count": 0,
        "target_counts": {},
        "frequency_counts_hz": {},
        "gaze_annotation_count": 0,
        "independent_gaze_truth_available": False,
        "parse_error": None,
    }
    if not path.is_file() or path.stat().st_size == 0:
        result["parse_error"] = "missing_or_empty_events_tsv"
        return result
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t", strict=True)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError("missing_or_duplicate_events_header")
            events = list(reader)
        result["row_count"] = len(events)
        counts = Counter((row.get("event_name") or "") for row in events)
        result["event_name_counts"] = dict(sorted(counts.items()))
        onsets = [row for row in events if row.get("event_name") == "stimulus_onset"]
        offsets = [row for row in events if row.get("event_name") == "stimulus_offset"]
        result["stimulus_onset_count"] = len(onsets)
        result["stimulus_offset_count"] = len(offsets)
        offset_by_trial = Counter(row.get("trial_index", "") for row in offsets)
        paired = 0
        unpaired = 0
        target_counts = Counter()
        frequencies = Counter()
        for row in onsets:
            trial_index = row.get("trial_index", "")
            if offset_by_trial[trial_index] == 1:
                paired += 1
            else:
                unpaired += 1
            target = row.get("presented_target_id") or row.get("target_id") or ""
            if target:
                target_counts[target] += 1
            frequency = safe_float(row.get("frequency_hz"))
            if frequency is not None:
                frequencies[f"{frequency:g}"] += 1
        result["paired_trial_count"] = paired
        result["unpaired_onset_count"] = unpaired
        result["target_counts"] = dict(sorted(target_counts.items()))
        result["frequency_counts_hz"] = dict(sorted(frequencies.items(), key=lambda item: float(item[0])))
        result["gaze_annotation_count"] = sum(1 for row in events if (row.get("gaze_target_id") or "").strip())
        result["independent_gaze_truth_available"] = False
        result["parseable"] = True
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        result["parse_error"] = str(exc)
    return result


def parse_raw(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "parseable": False,
        "row_count": 0,
        "header": [],
        "eeg_columns": [],
        "finite_fraction": {},
        "flat_fraction": {},
        "saturation_like_fraction": {},
        "rms": {},
        "min": {},
        "max": {},
        "timestamp_median_diff_s": None,
        "timestamp_gap_count_gt_1_5x_median": None,
        "nonzero_marker_count": 0,
        "first_sample_index": None,
        "last_sample_index": None,
        "parse_error": None,
    }
    if not path.is_file() or path.stat().st_size == 0:
        result["parse_error"] = "missing_or_empty_raw_brainflow_tsv"
        return result
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t", strict=True)
            header = reader.fieldnames or []
            if not header or len(set(header)) != len(header):
                raise ValueError("missing_or_duplicate_raw_header")
            eeg_columns = [name for name in header if re.fullmatch(r"eeg_ch\d+", name or "")]
            if not eeg_columns:
                raise ValueError("no_eeg_columns_named_eeg_chN")
            result["header"] = header
            result["eeg_columns"] = eeg_columns
            finite = Counter()
            flat = Counter()
            saturated = Counter()
            counts = Counter()
            sumsq = Counter()
            mins: dict[str, float] = {name: math.inf for name in eeg_columns}
            maxs: dict[str, float] = {name: -math.inf for name in eeg_columns}
            previous: dict[str, float | None] = {name: None for name in eeg_columns}
            timestamp_diffs: list[float] = []
            last_timestamp: float | None = None
            row_count = 0
            for row in reader:
                row_count += 1
                sample_index = safe_float(row.get("sample_index"))
                if sample_index is not None:
                    if result["first_sample_index"] is None:
                        result["first_sample_index"] = int(sample_index)
                    result["last_sample_index"] = int(sample_index)
                marker = safe_float(row.get("marker"))
                if marker is not None and abs(marker) > 1e-12:
                    result["nonzero_marker_count"] += 1
                timestamp = safe_float(row.get("timestamp_s"))
                if timestamp is not None and last_timestamp is not None:
                    delta = timestamp - last_timestamp
                    if math.isfinite(delta) and delta >= 0:
                        timestamp_diffs.append(delta)
                if timestamp is not None:
                    last_timestamp = timestamp
                for name in eeg_columns:
                    value = safe_float(row.get(name))
                    counts[name] += 1
                    if value is None:
                        continue
                    finite[name] += 1
                    sumsq[name] += value * value
                    mins[name] = min(mins[name], value)
                    maxs[name] = max(maxs[name], value)
                    previous_value = previous[name]
                    if previous_value is not None and value == previous_value:
                        flat[name] += 1
                    previous[name] = value
                    if abs(value) >= 180000.0:
                        saturated[name] += 1
            if row_count == 0:
                raise ValueError("raw_has_header_but_no_data_rows")
            result["row_count"] = row_count
            result["finite_fraction"] = {name: finite[name] / counts[name] for name in eeg_columns}
            result["flat_fraction"] = {name: flat[name] / max(1, counts[name] - 1) for name in eeg_columns}
            result["saturation_like_fraction"] = {name: saturated[name] / counts[name] for name in eeg_columns}
            result["rms"] = {name: math.sqrt(sumsq[name] / finite[name]) if finite[name] else None for name in eeg_columns}
            result["min"] = {name: mins[name] if mins[name] != math.inf else None for name in eeg_columns}
            result["max"] = {name: maxs[name] if maxs[name] != -math.inf else None for name in eeg_columns}
            if timestamp_diffs:
                median = statistics.median(timestamp_diffs)
                result["timestamp_median_diff_s"] = median
                result["timestamp_gap_count_gt_1_5x_median"] = sum(delta > median * 1.5 for delta in timestamp_diffs)
            result["parseable"] = True
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        result["parse_error"] = str(exc)
    return result


def classify_session(session: dict[str, Any] | None, protocol: dict[str, Any] | None,
                     events: dict[str, Any], raw: dict[str, Any], channel_config: dict[str, Any] | None) -> dict[str, Any]:
    session = session or {}
    protocol = protocol or {}
    board = str(session.get("board") or "").casefold()
    status = str(session.get("status") or "")
    validation_mode = str(session.get("validation_mode") or "")
    protocol_status = str((session.get("protocol_provenance") or {}).get("status") or "")
    has_native_raw = raw["parseable"]
    has_labels = events["paired_trial_count"] > 0 and bool(events["target_counts"])
    quality = []
    if not has_native_raw:
        quality.append("raw_unavailable_or_unparseable")
    if not events["parseable"]:
        quality.append("events_unavailable_or_unparseable")
    if events["unpaired_onset_count"]:
        quality.append("one_or_more_unpaired_stimulus_onsets")
    if raw["parseable"]:
        if any(value < 1.0 for value in raw["finite_fraction"].values()):
            quality.append("nonfinite_eeg_values")
        if any(value > 0.01 for value in raw["flat_fraction"].values()):
            quality.append("flat_or_repeated_eeg_values")
        if any(value > 0.001 for value in raw["saturation_like_fraction"].values()):
            quality.append("saturation_like_eeg_values")
    if status != "completed":
        quality.append(f"session_status_{status or 'missing'}")
    if validation_mode != "formal_candidate":
        quality.append(f"validation_mode_{validation_mode or 'missing'}")
    if protocol_status != "formal":
        quality.append(f"protocol_status_{protocol_status or 'missing'}")
    if session.get("simulated") is True:
        quality.append("simulated_capture")
    if channel_config is None:
        quality.append("channel_config_missing")
    else:
        positions = [entry.get("electrode_position") for entry in channel_config.get("channels", []) if isinstance(entry, dict)]
        if len(positions) != 8 or any(not isinstance(position, str) or not position.strip() for position in positions):
            quality.append("electrode_positions_missing_or_incomplete")
    target_count = len(events["target_counts"])
    repetitions = protocol.get("repetitions")
    expected = target_count * int(repetitions) if isinstance(repetitions, (int, float)) and repetitions and target_count else None
    labeled = events["paired_trial_count"]
    exploration_usable = bool(has_native_raw and has_labels and labeled >= 4)
    formal_candidate_shape = bool(
        board in {"cyton", "openbci cyton"}
        and session.get("simulated") is False
        and validation_mode == "formal_candidate"
        and protocol_status == "formal"
        and status == "completed"
        and has_native_raw and has_labels
    )
    return {
        "session_kind": "native_neurostation" if has_native_raw else "imported_or_non_native",
        "exploration_usable": exploration_usable,
        "formal_candidate_shape": formal_candidate_shape,
        "personal_calibration_eligible": False,
        "default_runtime_authorized": False,
        "deployment_decision": "do_not_deploy_from_inventory",
        "expected_labeled_trials_from_protocol": expected,
        "quality_flags": sorted(set(quality)),
        "label_caveat": "presentation labels are not independent gaze truth",
        "historical_fields_preserved": {
            "status": session.get("status"),
            "validation_mode": session.get("validation_mode"),
            "protocol_status": protocol_status,
            "simulated": session.get("simulated"),
        },
    }


def classification_inputs_probe(source: Path, raw_rows: int) -> dict[str, Any]:
    result = {"available": False, "trial_count": 0, "target_count": 0, "reason": None}
    if not (source / "raw_brainflow.tsv").is_file() or raw_rows <= 0:
        result["reason"] = "raw_unavailable_or_empty"
        return result
    try:
        import numpy as np
        from eeg_tools.workstation import v9_offline
        targets, trials, reason = v9_offline._classification_inputs(source, np.empty((raw_rows, 8), dtype=np.float32))
        result.update({"available": reason is None, "trial_count": len(trials), "target_count": len(targets), "reason": reason})
    except Exception as exc:  # diagnostic probe must not stop inventory
        result["reason"] = f"probe_error: {type(exc).__name__}: {exc}"
    return result


def inventory(root: Path) -> dict[str, Any]:
    entries = []
    if not root.is_dir():
        raise SystemExit(f"root directory not found: {root}")
    for source in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name.casefold()):
        session = read_json(source / "session.json")
        protocol = read_json(source / "protocol.json") or read_json(source / "ssvep_config.json")
        channel_config = read_json(source / "channel_config.json")
        quality = read_json(source / "quality.json")
        events = parse_events(source / "events.tsv")
        raw = parse_raw(source / "raw_brainflow.tsv")
        classification = classification_inputs_probe(source, int(raw.get("row_count") or 0))
        decision = classify_session(session, protocol, events, raw, channel_config)
        entries.append({
            "source_directory": str(source),
            "source_name": source.name,
            "session": session or {"parse_error": "missing_or_invalid_session_json"},
            "protocol": protocol or {"parse_error": "missing_or_invalid_protocol_json"},
            "channel_config_present": channel_config is not None,
            "channel_config_summary": {
                "board": channel_config.get("board") if channel_config else None,
                "channel_order": channel_config.get("channel_order") if channel_config else None,
                "electrode_positions": [entry.get("electrode_position") for entry in channel_config.get("channels", []) if isinstance(entry, dict)] if channel_config else [],
            },
            "quality_json": quality,
            "events": events,
            "raw": raw,
            "classification_inputs_probe": classification,
            "decision": decision,
        })
    native = [item for item in entries if item["decision"]["session_kind"] == "native_neurostation"]
    usable = [item for item in entries if item["decision"]["exploration_usable"]]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "read_only_historical_ssvep_inventory",
        "source_root": str(root),
        "source_mutated": False,
        "historical_conclusions_rewritten": False,
        "deployment_authorization_granted": False,
        "limitations": [
            "Trash and historical sessions are inventoried only; no source file is modified.",
            "Presentation labels are not independent gaze truth and cannot establish gaze accuracy.",
            "Technical-validation, simulated, aborted, imported, saturated or incomplete sessions are not formal personal calibration data.",
            "This inventory does not validate physical electrode placement, clinical efficacy, or production runtime behavior.",
        ],
        "summary": {
            "directory_count": len(entries),
            "native_neurostation_count": len(native),
            "imported_or_non_native_count": len(entries) - len(native),
            "exploration_usable_count": len(usable),
            "formal_candidate_shape_count": sum(item["decision"]["formal_candidate_shape"] for item in entries),
            "classification_probe_available_count": sum(item["classification_inputs_probe"]["available"] for item in entries),
            "paired_trial_total": sum(item["events"]["paired_trial_count"] for item in entries),
        },
        "entries": entries,
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
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
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = inventory(args.root.resolve())
    write_json_atomic(args.output.resolve(), report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


