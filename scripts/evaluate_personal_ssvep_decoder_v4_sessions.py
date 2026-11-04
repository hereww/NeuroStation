"""Evaluate the v4 adaptive candidate on NeuroStation session directories.

This is an explicit offline diagnostic adapter. Calibration labels are used only
inside ``fit_calibration``; evaluation labels are compared after the forward pass
as a presentation-proxy diagnostic. The result is never enabled in the desktop
runtime and does not claim gaze accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FALSE_FLAGS = {
    "runtime_enabled": False,
    "production_validated": False,
    "source_domain_validated": False,
    "independent_personal_validation": False,
    "clean_ground_truth_available": False,
    "presented_matches_are_gaze_accuracy": False,
    "training_performed": False,
    "default_deployment_authorized": False,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("v4 session evaluator: " + message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    require(not path.exists(), f"report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)


def _load_session(source: Path) -> tuple[Any, dict, list[tuple[str, float]], list[dict], dict]:
    import numpy as np
    from eeg_tools.workstation import v9_offline as baseline
    from scripts.diagnose_neurostation_v9_output import parse_raw, validate_metadata

    source = source.resolve()
    require(source.is_dir(), f"session directory does not exist: {source}")
    before = baseline._source_hashes(source)
    session_metadata = read_json(source / "session.json")
    validate_metadata(session_metadata, read_json(source / "channel_config.json"))
    raw, header = parse_raw((source / "raw_brainflow.tsv").read_bytes())
    targets, trials, reason = baseline._classification_inputs(source, raw)
    require(not reason, reason or "classification metadata unavailable")
    require(raw.ndim == 2 and raw.dtype.kind == "f" and bool(np.isfinite(raw).all()), "session raw array is invalid")
    after = baseline._source_hashes(source)
    require(before == after, "session source changed during load")
    return raw, header, targets, trials, {
        "source_session": str(source),
        "session_id": session_metadata.get("session_id", ""),
        "source_sha256": before,
    }


def _register_session_identity(source: Path, info: dict, role: str, seen: dict[str, tuple[str, str]]) -> None:
    """Reject repeated sessions and copied raw recordings across all split inputs."""
    resolved = os.path.normcase(str(source.resolve()))
    identities = [f"path:{resolved}"]
    session_id = info.get("session_id")
    if isinstance(session_id, str) and session_id.strip():
        identities.append(f"session_id:{session_id.strip()}")
    hashes = info.get("source_sha256")
    if isinstance(hashes, dict):
        raw_hash = hashes.get("raw_brainflow.tsv")
        if isinstance(raw_hash, str) and raw_hash:
            identities.append(f"raw_sha256:{raw_hash}")

    for identity in identities:
        previous = seen.get(identity)
        if previous is not None:
            require(
                False,
                f"{role} session overlaps {previous[0]} session by {identity.split(':', 1)[0]}: "
                f"{previous[1]} and {source.resolve()}",
            )
    for identity in identities:
        seen[identity] = (role, str(source.resolve()))


def _pairs_for_session(source: Path, *, include_signals: bool) -> tuple[list[dict], list[tuple[str, float]], dict]:
    import numpy as np
    from eeg_tools import personal_ssvep_decoder_v4 as decoder

    raw, header, targets, trials, source_info = _load_session(source)
    pairs: list[dict] = []
    for trial in trials:
        start = int(trial["window_start_sample_index"])
        end = int(trial["offset_sample_index"])
        onset = int(trial["onset_sample_index"])
        rest_start = max(0, onset - 750)
        rest_end = onset
        require(0 <= rest_start < rest_end <= raw.shape[0], "rest window is outside raw recording")
        require(0 <= start < end <= raw.shape[0], "stimulus window is outside raw recording")
        require(end - start >= decoder.get_config()["minimum_samples"], "stimulus window is too short")
        require(rest_end - rest_start >= decoder.get_config()["minimum_rest_samples"], "rest window is too short")
        entry = {
            "trial_index": int(trial["trial_index"]),
            "target_id": trial.get("presented_target_id", ""),
            "onset_sample_index": onset,
            "window_start_sample_index": start,
            "offset_sample_index": end,
            "rest_start_sample_index": rest_start,
            "rest_end_sample_index": rest_end,
        }
        if include_signals:
            entry["signal"] = np.asarray(raw[start:end].T, dtype=np.float64)
            entry["rest_signal"] = np.asarray(raw[rest_start:rest_end].T, dtype=np.float64)
        pairs.append(entry)
    return pairs, targets, {"header": header, **source_info}


def evaluate(calibration_sessions: list[Path], evaluation_sessions: list[Path], output: Path) -> dict:
    import numpy as np
    from eeg_tools import personal_ssvep_decoder_v4 as decoder

    require(calibration_sessions, "at least one --calibration-session is required")
    require(evaluation_sessions, "at least one --evaluation-session is required")
    require(not output.exists(), f"output already exists: {output}")
    calibration_paths = [os.path.normcase(str(source.resolve())) for source in calibration_sessions]
    evaluation_paths = [os.path.normcase(str(source.resolve())) for source in evaluation_sessions]
    require(
        len(set(calibration_paths + evaluation_paths)) == len(calibration_paths + evaluation_paths),
        "calibration and evaluation session paths must be disjoint and each session may appear only once",
    )

    calibration_examples: list[dict[str, Any]] = []
    calibration_sources = []
    source_identities: dict[str, tuple[str, str]] = {}
    targets: list[tuple[str, float]] | None = None
    for source in calibration_sessions:
        rows, session_targets, info = _pairs_for_session(source, include_signals=True)
        _register_session_identity(source, info, "calibration", source_identities)
        if targets is None:
            targets = session_targets
        require(session_targets == targets, "calibration target definitions differ")
        for row in rows:
            calibration_examples.append({"target_id": row["target_id"], "signal": row["signal"], "rest_signal": row["rest_signal"]})
        calibration_sources.append({"source_session": info["source_session"], "source_sha256": info["source_sha256"], "trial_count": len(rows), "header": info["header"]})
    require(targets is not None, "no calibration targets loaded")
    evaluation_inputs = []
    evaluation_sources = []
    for source in evaluation_sessions:
        rows, session_targets, info = _pairs_for_session(source, include_signals=True)
        _register_session_identity(source, info, "evaluation", source_identities)
        require(session_targets == targets, "evaluation target definitions differ")
        evaluation_inputs.append((rows, session_targets, info))
        evaluation_sources.append({"source_session": info["source_session"], "source_sha256": info["source_sha256"], "trial_count": len(rows), "header": info["header"]})

    sample_rate_hz = 250.0
    calibration = decoder.fit_calibration(calibration_examples, sample_rate_hz, targets)

    trials = []
    for rows, session_targets, info in evaluation_inputs:
        source_rows = []
        for row in rows:
            prediction = decoder.decode_window(row["signal"], row["rest_signal"], sample_rate_hz, targets, calibration)
            recorded = row["target_id"]
            source_rows.append({
                key: value for key, value in row.items()
                if key not in {"signal", "rest_signal"}
            })
            source_rows[-1].update({
                "recorded_presentation_target_id": recorded,
                "prediction": prediction,
                "recorded_label_used_in_forward": False,
                "presented_match": bool(recorded and prediction.get("accepted") and prediction.get("target_id") == recorded),
            })
        trials.extend(source_rows)

    accepted = [row for row in trials if row["prediction"].get("accepted")]
    target_ids = [target_id for target_id, _ in targets]
    counts = {target_id: sum(row["recorded_presentation_target_id"] == target_id for row in trials) for target_id in target_ids}
    report = {
        "schema_version": 1,
        "status": "formal",
        "completion_status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_sources": calibration_sources,
        "evaluation_sources": evaluation_sources,
        "source_session_identity_disjoint": True,
        "targets": [[target_id, float(frequency)] for target_id, frequency in targets],
        "calibration": calibration,
        "trials": trials,
        "summary": {
            "calibration_trial_count": int(sum(item["trial_count"] for item in calibration_sources)),
            "evaluation_trial_count": len(trials),
            "accepted_trial_count": len(accepted),
            "unknown_trial_count": sum(row["prediction"].get("target_id") is None for row in trials),
            "presented_match_count": sum(row["presented_match"] for row in trials),
            "accepted_presented_match_count": sum(row["presented_match"] for row in accepted),
            "accepted_presented_match_rate": (float(sum(row["presented_match"] for row in accepted) / len(accepted)) if accepted else None),
            "counts_by_target": counts,
            "matching_definition": "match to recorded presentation proxy after accepted forward decoding; not independent gaze accuracy",
        },
        "decoder": decoder.config_summary(),
        "limitations": [
            "Candidate/offline diagnostic only; no default desktop runtime change.",
            "Calibration labels are recorded presentation labels and are not independent gaze truth.",
            "A cross-session evaluation is a proxy holdout, not an independent gaze validation.",
            "Physical electrode positions and source-domain validity still require independent checks.",
            "Unknown is a valid fail-closed result when calibration or physiological evidence gates fail.",
        ],
        **FALSE_FLAGS,
    }
    atomic_json(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-session", action="append", required=True, type=Path)
    parser.add_argument("--evaluation-session", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.calibration_session, args.evaluation_session, args.output)
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
        print(json.dumps({"status": "formal", "completion_status": "incomplete", "error": str(exc), **FALSE_FLAGS}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"status": "formal", "completion_status": "complete", "report": str(args.output.resolve()), "summary": report["summary"], **FALSE_FLAGS}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
