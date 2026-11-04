"""Evaluate the evidence-gated v3 decoder on personal SSVEP recordings.

This is an explicit read-only diagnostic. Recorded presentation labels are used
only after decoding to report agreement; they never enter the forward pass.
An unknown result is a valid fail-closed outcome and is not counted as a match.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRACKS = ("raw", "notch", "traditional", "v14")
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
    "checkpoint_selection_performed": False,
    "default_deployment_authorized": False,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("v3 personal decoder evaluator: " + message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    require(not path.exists(), f"report already exists: {path}")
    pending = path.with_name(path.name + ".pending")
    with pending.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)


def create_output(path: Path) -> Path:
    base = (ROOT / "analysis_outputs").resolve()
    candidate = path.resolve()
    require(candidate.is_relative_to(base), "output must be under analysis_outputs")
    require(not candidate.exists(), f"output directory already exists: {candidate}")
    candidate.mkdir(parents=True)
    return candidate


def evaluate_session(source: Path, v14_personal: Path, output: Path) -> dict:
    import numpy as np
    from eeg_tools import personal_ssvep_decoder_v3 as v3
    from eeg_tools.workstation import v9_offline as baseline
    from scripts.diagnose_neurostation_v9_output import parse_raw, validate_metadata

    before = baseline._source_hashes(source)
    validate_metadata(read_json(source / "session.json"), read_json(source / "channel_config.json"))
    raw, header = parse_raw((source / "raw_brainflow.tsv").read_bytes())
    targets, trials, reason = baseline._classification_inputs(source, raw)
    require(not reason, reason or "classification metadata unavailable")
    target_ids = {item[0] for item in targets}
    arrays = {
        "raw": raw,
        "notch": np.load(v14_personal / "notch.npy", allow_pickle=False),
        "traditional": np.load(v14_personal / "traditional.npy", allow_pickle=False),
        "v14": np.load(v14_personal / "v14.npy", allow_pickle=False),
    }
    require(all(value.shape == raw.shape and value.dtype.kind == "f" and np.isfinite(value).all() for value in arrays.values()), "track arrays do not match source raw")
    rows = []
    summaries = {track: {} for track in TRACKS}
    for trial in trials:
        start, end = trial["window_start_sample_index"], trial["offset_sample_index"]
        rest_start = max(0, trial["onset_sample_index"] - 750)
        rest_end = trial["onset_sample_index"]
        row = {
            "trial_index": trial["trial_index"],
            "presented_target_id": trial.get("presented_target_id", ""),
            "onset_sample_index": trial["onset_sample_index"],
            "offset_sample_index": trial["offset_sample_index"],
            "window_start_sample_index": start,
            "rest_start_sample_index": rest_start,
            "rest_end_sample_index": rest_end,
            "predictions": {},
        }
        for track, value in arrays.items():
            try:
                require(rest_end - rest_start >= 250, "rest window has fewer than 250 samples")
                prediction = v3.decode_window(value[start:end].T, value[rest_start:rest_end].T, 250.0, targets)
            except (ValueError, ArithmeticError) as exc:
                prediction = {"available": False, "accepted": False, "target_id": None, "rejection_reason": "computation_failed", "reason": str(exc)}
            row["predictions"][track] = prediction
        rows.append(row)
    for track in TRACKS:
        available = [row["predictions"][track] for row in rows if row["predictions"][track].get("available")]
        accepted = [item for item in available if item.get("accepted")]
        summaries[track] = {
            "recorded_trial_count": len(rows),
            "available_trial_count": len(available),
            "accepted_trial_count": len(accepted),
            "unknown_trial_count": sum(item.get("target_id") is None for item in available),
            "accepted_presented_match_count": sum(item.get("accepted") and item.get("target_id") == row["presented_target_id"] for row, item in ((row, row["predictions"][track]) for row in rows)),
            "accepted_presented_match_rate": (sum(item.get("accepted") and item.get("target_id") == row["presented_target_id"] for row, item in ((row, row["predictions"][track]) for row in rows)) / len(accepted)) if accepted else None,
            "mean_top_two_margin_accepted": (math.fsum(item["confidence"] for item in accepted) / len(accepted)) if accepted else None,
            "confidence_is_probability": False,
        }
    after = baseline._source_hashes(source)
    require(before == after, "source data changed")
    report = {
        "status": "formal",
        "completion_status": "complete",
        "source_session": str(source),
        "source_header": header,
        "source_sha256_before": before,
        "source_sha256_after": after,
        "source_hashes_match": True,
        "targets": targets,
        "trials": rows,
        "tracks": summaries,
        "decoder_v3": v3.config_summary(),
        "rest_window_definition": "pre-stimulus samples from max(0,onset-750) through onset; at least 250 samples required",
        "presented_matching_definition": "accepted predictions compared to recorded presentation labels; this is not independent gaze accuracy",
        "channel_selection": {"decoder_v3_indices": [4, 5, 6, 7], "decoder_v3_names": ["CH5", "CH6", "CH7", "CH8"], "reason": "a priori posterior visual-channel subset from preset map"},
        "limitations": [
            "The v3 gates are transparent reference thresholds and are not independently detection-validated.",
            "Unknown is preferred to forcing a target when stimulus/rest evidence is insufficient or ambiguous.",
            "The configured electrode positions require physical verification.",
            "Presented matches are not independent gaze accuracy.",
            "No clean physiological reference or independent gaze truth is available.",
            "No default desktop runtime was changed.",
        ],
        **FALSE_FLAGS,
    }
    atomic_json(output / "report.json", report)
    return report


def evaluate(v14_evaluation: Path, output_path: Path) -> dict:
    summary = read_json(v14_evaluation / "summary.json")
    require(summary.get("model", {}).get("selection_gate_passed") is False, "input is not failed-selection diagnostic")
    require(summary.get("presented_matches_are_gaze_accuracy") is False, "unsafe gaze flag in input")
    output = create_output(output_path)
    reports = []
    for index, item in enumerate(summary.get("personal", [])):
        personal_output = output / f"personal_{index:02d}"
        personal_output.mkdir()
        reports.append(evaluate_session(Path(item["source_session"]), v14_evaluation / f"personal_{index:02d}", personal_output))
    result = {
        "schema_version": 1,
        "status": "formal",
        "completion_status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "v14_evaluation_directory": str(v14_evaluation.resolve()),
        "personal": [{"source_session": report["source_session"], "report": f"personal_{i:02d}/report.json", "tracks": report["tracks"]} for i, report in enumerate(reports)],
        "source_files_unchanged": all(report["source_hashes_match"] for report in reports),
        "v3_is_evidence_gated_diagnostic": True,
        "accepted_predictions_allow_unknown": True,
        "limitations": [
            "This result is an evidence-gated diagnostic only; it does not prove denoising quality or gaze accuracy.",
            "No default desktop runtime was changed.",
            "Thresholds require independent validation on new formal calibration sessions.",
        ],
        **FALSE_FLAGS,
    }
    atomic_json(output / "summary.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v14-evaluation-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args(argv)
    try:
        result = evaluate(Path(args.v14_evaluation_directory).resolve(), Path(args.output_directory))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "formal", "completion_status": "incomplete", "error": str(exc), **FALSE_FLAGS}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"status": "formal", "completion_status": "complete", "report": str(Path(args.output_directory).resolve() / "summary.json"), **FALSE_FLAGS}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
