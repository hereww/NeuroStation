"""Evaluate the v4 candidate decoder from an explicit offline manifest.

The manifest contains calibration sessions and evaluation sessions.  Each
session points to a NumPy ``.npy`` array with shape ``[8, samples]`` and a
matching rest array.  Calibration labels assign centroids; evaluation labels,
when present, are read only after decoding to report agreement with the
recorded presentation proxy.  They are not independent gaze truth.

Example manifest::

    {
      "sample_rate_hz": 250.0,
      "targets": [["a", 10.0], ["b", 12.0]],
      "calibration_sessions": [
        {"target_id": "a", "signal_path": "cal/a.npy", "rest_signal_path": "cal/a_rest.npy"}
      ],
      "evaluation_sessions": [
        {"signal_path": "eval/one.npy", "rest_signal_path": "eval/one_rest.npy",
         "recorded_target_id": "a"}
      ]
    }
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

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
    "physiological_clean_truth": False,
    "clinical_validation": False,
    "training_performed": False,
    "deep_model_training_performed": False,
    "default_deployment_authorized": False,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("v4 personal decoder evaluator: " + message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def _array_path(manifest_path: Path, session: dict, key: str):
    import numpy as np

    path_value = session.get(key)
    require(isinstance(path_value, str) and path_value, f"session is missing {key}")
    path = (manifest_path.parent / path_value).resolve()
    require(path.exists() and path.is_file(), f"array file does not exist: {path}")
    before = sha256(path)
    array = np.load(path, allow_pickle=False)
    require(array.dtype.kind in "iuf" and array.ndim == 2, f"{key} must be a numeric [channels, samples] .npy array")
    after = sha256(path)
    require(before == after, f"array changed while it was read: {path}")
    return array, str(path), before


def evaluate_manifest(manifest_path: Path, output_path: Path) -> dict:
    import numpy as np
    from eeg_tools import personal_ssvep_decoder_v4 as decoder

    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    require(isinstance(manifest, dict), "manifest must be an object")
    sample_rate_hz = float(manifest.get("sample_rate_hz", decoder.get_config()["sample_rate_hz"]))
    targets = manifest.get("targets")
    require(isinstance(targets, list) and targets, "manifest targets must be a nonempty list")
    calibration_entries = manifest.get("calibration_sessions")
    evaluation_entries = manifest.get("evaluation_sessions", [])
    require(isinstance(calibration_entries, list) and calibration_entries, "manifest calibration_sessions must be nonempty")
    require(isinstance(evaluation_entries, list), "manifest evaluation_sessions must be a list")

    calibration_sessions = []
    source_hashes_before = {}
    for session in calibration_entries:
        require(isinstance(session, dict), "calibration session must be an object")
        signal, signal_path, signal_hash = _array_path(manifest_path, session, "signal_path")
        rest, rest_path, rest_hash = _array_path(manifest_path, session, "rest_signal_path")
        calibration_sessions.append({"target_id": session.get("target_id"), "signal": signal, "rest_signal": rest})
        source_hashes_before[signal_path] = signal_hash
        source_hashes_before[rest_path] = rest_hash
    calibration = decoder.fit_calibration(calibration_sessions, sample_rate_hz, targets)

    rows = []
    for index, session in enumerate(evaluation_entries):
        require(isinstance(session, dict), "evaluation session must be an object")
        signal, signal_path, signal_hash = _array_path(manifest_path, session, "signal_path")
        rest, rest_path, rest_hash = _array_path(manifest_path, session, "rest_signal_path")
        source_hashes_before[signal_path] = signal_hash
        source_hashes_before[rest_path] = rest_hash
        prediction = decoder.decode_window(signal, rest, sample_rate_hz, targets, calibration)
        row = {
            "trial_index": index,
            "recorded_target_id": session.get("recorded_target_id", ""),
            "prediction": prediction,
            "recorded_label_used_in_forward": False,
            "presented_match": bool(session.get("recorded_target_id") and prediction.get("target_id") == session.get("recorded_target_id")),
        }
        rows.append(row)

    source_hashes_after = {path: sha256(Path(path)) for path in source_hashes_before}
    require(source_hashes_before == source_hashes_after, "source arrays changed during evaluation")
    accepted = [row for row in rows if row["prediction"].get("accepted")]
    report = {
        "schema_version": 1,
        "status": "formal",
        "completion_status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "targets": targets,
        "calibration": calibration,
        "trials": rows,
        "summary": {
            "evaluation_trial_count": len(rows),
            "accepted_trial_count": len(accepted),
            "unknown_trial_count": sum(row["prediction"].get("target_id") is None for row in rows),
            "presented_match_count": sum(row["presented_match"] for row in rows),
            "presented_matching_definition": "match to recorded presentation proxy, not independent gaze accuracy",
        },
        "source_hashes_unchanged": True,
        "decoder": decoder.config_summary(),
        "limitations": [
            "This is a candidate/offline diagnostic and is not enabled by the default desktop.",
            "Source domain validation is false; hardware and session quality still require independent checks.",
            "Recorded presentation labels are proxy metadata only and do not establish gaze accuracy.",
            "Unknown is a valid fail-closed result when calibration or evidence gates do not pass.",
        ],
        **FALSE_FLAGS,
    }
    atomic_json(output_path.resolve(), report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="offline JSON manifest with calibration/evaluation .npy paths")
    parser.add_argument("--output", required=True, type=Path, help="new report JSON path")
    args = parser.parse_args(argv)
    try:
        report = evaluate_manifest(args.manifest, args.output)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "formal", "completion_status": "incomplete", "error": str(exc), **FALSE_FLAGS}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"status": "formal", "completion_status": "complete", "report": str(args.output.resolve()), **FALSE_FLAGS}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
