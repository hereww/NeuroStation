from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_ROOT = Path(r"C:\Users\Administrator\Documents\NeuroStation\Datasets\Trash\Datasets")
DEFAULT_OUTPUT = ROOT / "analysis_outputs" / "history" / "historical_cyton_v3_exploration_20261014.json"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def source_hashes(source: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(source.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            result[str(path.relative_to(source))] = digest
    return result


def parse_events(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", strict=True)
        if not reader.fieldnames:
            raise ValueError("events.tsv header missing")
        rows = list(reader)
    return rows


def native_session_candidates(root: Path, minimum_trials: int) -> list[Path]:
    candidates = []
    for source in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name.casefold()):
        try:
            session = read_json(source / "session.json")
            if str(session.get("board", "")).casefold() not in {"cyton", "openbci cyton"}:
                continue
            if session.get("simulated") is True:
                continue
            if not (source / "raw_brainflow.tsv").is_file() or (source / "raw_brainflow.tsv").stat().st_size <= 0:
                continue
            events = parse_events(source / "events.tsv")
            paired = sum(
                1 for onset in events if onset.get("event_name") == "stimulus_onset"
                and sum(1 for offset in events if offset.get("event_name") == "stimulus_offset" and offset.get("trial_index") == onset.get("trial_index")) == 1
            )
            if paired >= minimum_trials:
                candidates.append(source)
        except (OSError, UnicodeError, json.JSONDecodeError, csv.Error, ValueError):
            continue
    return candidates


def make_targets(protocol: dict[str, Any]) -> list[tuple[str, float]]:
    targets = []
    for item in protocol.get("targets", []):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            targets.append((str(item[0]), float(item[1])))
        elif isinstance(item, dict):
            targets.append((str(item["id"]), float(item["frequency_hz"])))
    if not targets:
        raise ValueError("protocol has no targets")
    return targets


def evaluate_session(source: Path) -> dict[str, Any]:
    import numpy as np
    from eeg_tools import personal_ssvep_decoder_v3 as decoder
    from eeg_tools.workstation import v9_offline
    from scripts.diagnose_neurostation_v9_output import parse_raw, validate_metadata

    session = read_json(source / "session.json")
    channels = read_json(source / "channel_config.json")
    protocol = read_json(source / "protocol.json") if (source / "protocol.json").is_file() else read_json(source / "ssvep_config.json")
    before = source_hashes(source)
    validate_metadata(session, channels)
    raw, header = parse_raw((source / "raw_brainflow.tsv").read_bytes())
    targets, trials, reason = v9_offline._classification_inputs(source, raw)
    if reason:
        raise ValueError(reason)
    if len(trials) < 1:
        raise ValueError("no classifiable trials")
    rows = []
    for trial in trials:
        start = trial.get("window_start_sample_index")
        end = trial.get("offset_sample_index")
        onset = trial.get("onset_sample_index")
        row: dict[str, Any] = {
            "trial_index": trial.get("trial_index"),
            "presented_target_id": trial.get("presented_target_id", ""),
            "onset_sample_index": onset,
            "offset_sample_index": end,
            "window_start_sample_index": start,
            "prediction": None,
        }
        try:
            if start is None or end is None or onset is None:
                raise ValueError("missing trial indexes")
            rest_start = max(0, onset - 750)
            rest_end = onset
            prediction = decoder.decode_window(raw[start:end].T, raw[rest_start:rest_end].T, 250.0, targets)
            row["prediction"] = prediction
        except (ValueError, ArithmeticError, IndexError) as exc:
            row["prediction"] = {
                "available": False,
                "accepted": False,
                "target_id": None,
                "rejection_reason": "computation_failed",
                "reason": str(exc),
            }
        rows.append(row)
    accepted = [row for row in rows if row["prediction"] and row["prediction"].get("accepted")]
    matches = [row for row in accepted if row["prediction"].get("target_id") == row.get("presented_target_id")]
    confusion = Counter()
    for row in accepted:
        confusion[f"{row.get('presented_target_id')}->{row['prediction'].get('target_id')}"] += 1
    after = source_hashes(source)
    if before != after:
        raise RuntimeError("source data changed during read-only diagnostic")
    return {
        "source_session": str(source),
        "source_files_unchanged": True,
        "source_header": header,
        "historical_status": session.get("status"),
        "historical_validation_mode": session.get("validation_mode"),
        "historical_protocol_status": (session.get("protocol_provenance") or {}).get("status"),
        "targets": targets,
        "recorded_trial_count": len(rows),
        "accepted_trial_count": len(accepted),
        "unknown_trial_count": sum(1 for row in rows if not row["prediction"].get("accepted")),
        "accepted_presented_match_count": len(matches),
        "accepted_presented_match_rate": len(matches) / len(accepted) if accepted else None,
        "all_trial_match_rate": len(matches) / len(rows) if rows else None,
        "confusion_accepted": dict(sorted(confusion.items())),
        "trials": rows,
        "decoder": decoder.config_summary(),
        "limitations": [
            "Exploratory historical diagnostic only; it does not authorize runtime deployment.",
            "These sessions retain technical_validation or technical_validation_override history.",
            "Presentation matches are not independent gaze accuracy.",
            "Physical electrode positions and clean physiological truth remain unverified.",
            "No model checkpoint was trained or selected by this diagnostic.",
        ],
        "runtime_enabled": False,
        "production_validated": False,
        "default_deployment_authorized": False,
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
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-trials", type=int, default=4)
    args = parser.parse_args()
    sources = native_session_candidates(args.root.resolve(), args.minimum_trials)
    reports = []
    failures = []
    for source in sources:
        try:
            reports.append(evaluate_session(source))
        except Exception as exc:
            failures.append({"source_session": str(source), "error": f"{type(exc).__name__}: {exc}"})
    aggregate_matches = sum(int(report["accepted_presented_match_count"]) for report in reports)
    aggregate_accepted = sum(int(report["accepted_trial_count"]) for report in reports)
    result = {
        "schema_version": 1,
        "status": "formal",
        "completion_status": "complete" if not failures else "complete_with_failures",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.root.resolve()),
        "source_mutated": False,
        "historical_conclusions_rewritten": False,
        "candidate_count": len(sources),
        "evaluated_count": len(reports),
        "failure_count": len(failures),
        "aggregate": {
            "accepted_trial_count": aggregate_accepted,
            "accepted_presented_match_count": aggregate_matches,
            "accepted_presented_match_rate": aggregate_matches / aggregate_accepted if aggregate_accepted else None,
        },
        "reports": reports,
        "failures": failures,
        "runtime_enabled": False,
        "production_validated": False,
        "default_deployment_authorized": False,
    }
    write_atomic(args.output.resolve(), result)
    print(json.dumps({"candidate_count": len(sources), "evaluated_count": len(reports), "failures": len(failures), "aggregate": result["aggregate"]}, ensure_ascii=False, indent=2))
    print(f"report={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
