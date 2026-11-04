"""Frozen event-window sensitivity diagnostic for the v14 personal outputs.

This script measures how much the transparent CCA proxy changes under a small
pre-registered grid of event-window trims. It does not choose a deployment
window, train, use gaze truth, or authorize any checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRACKS = ("raw", "notch", "traditional", "v14")
TRIMS = (0, 50, 100, 125, 150, 200)
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
        raise ValueError("v14 event-window diagnostic: " + message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def evaluate_session(source: Path, evaluation_personal: Path, output: Path) -> dict:
    import numpy as np
    from eeg_tools.personal_ssvep_decoder import decode_window, config_summary
    from eeg_tools.workstation import v9_offline as baseline
    from scripts.diagnose_neurostation_v9_output import parse_raw, validate_metadata

    before = baseline._source_hashes(source)
    validate_metadata(read_json(source / "session.json"), read_json(source / "channel_config.json"))
    raw, _ = parse_raw((source / "raw_brainflow.tsv").read_bytes())
    targets, trials, metadata_reason = baseline._classification_inputs(source, raw)
    require(not metadata_reason, metadata_reason or "classification metadata unavailable")
    target_ids = {target[0] for target in targets}
    arrays = {
        track: np.load(evaluation_personal / f"{track}.npy", allow_pickle=False)
        for track in TRACKS
    }
    require(all(value.shape == raw.shape and np.isfinite(value).all() for value in arrays.values()),
            "evaluation arrays do not match source raw shape")
    rows = []
    for start_trim in TRIMS:
        for end_trim in TRIMS:
            predictions = {track: [] for track in TRACKS}
            for trial in trials:
                start, end = trial["onset_sample_index"], trial["offset_sample_index"]
                if start is None or end is None or end - start - start_trim - end_trim < 250:
                    continue
                window_start = start + start_trim
                window_end = end - end_trim
                for track, value in arrays.items():
                    try:
                        result = decode_window(value[window_start:window_end].T, 250.0, targets)
                    except (ValueError, ArithmeticError) as exc:
                        predictions[track].append({"available": False, "reason": str(exc)})
                    else:
                        predictions[track].append({
                            "available": True,
                            "target_id": result["target_id"],
                            "confidence": float(result["confidence"]),
                            "presented_target_id": trial.get("presented_target_id", ""),
                        })
            summary = {}
            for track in TRACKS:
                available = [item for item in predictions[track] if item.get("available") and item.get("presented_target_id") in target_ids]
                margins = [item["confidence"] for item in available]
                summary[track] = {
                    "recorded_trial_count": len(trials),
                    "available_presented_trial_count": len(available),
                    "presented_match_count": sum(item["target_id"] == item["presented_target_id"] for item in available),
                    "mean_top_two_margin": math.fsum(margins) / len(margins) if margins else None,
                    "margin_is_probability": False,
                }
            rows.append({"start_trim_samples": start_trim, "end_trim_samples": end_trim, "tracks": summary})
    after = baseline._source_hashes(source)
    require(before == after, "source data changed")
    return {
        "status": "formal",
        "completion_status": "complete",
        "source_session": str(source),
        "source_sha256_before": before,
        "source_sha256_after": after,
        "source_hashes_match": True,
        "targets": targets,
        "trial_count": len(trials),
        "trim_grid_samples": {"start": list(TRIMS), "end": list(TRIMS)},
        "current_baseline_rule": {"start_trim_samples": 125, "end_trim_samples": 0},
        "rows": rows,
        "decoder": config_summary(),
        "interpretation": [
            "This is a fixed timing sensitivity grid, not hyperparameter selection.",
            "Presented matching is a recorded-label proxy, not gaze accuracy.",
            "No row authorizes a runtime window or model deployment.",
        ],
        **FALSE_FLAGS,
    }


def evaluate(evaluation: Path, output_path: Path) -> dict:
    import numpy as np  # noqa: F401  # keep the dependency check explicit
    summary = read_json(evaluation / "summary.json")
    require(summary.get("model", {}).get("selection_gate_passed") is False,
            "evaluation is not bound to failed-selection checkpoint")
    require(summary.get("presented_matches_are_gaze_accuracy") is False,
            "unsafe gaze flag in evaluation summary")
    sessions = summary.get("personal", [])
    require(len(sessions) == 2, "exactly two personal sessions expected")
    output = create_output(output_path)
    reports = []
    for index, session in enumerate(sessions):
        source = Path(session["source_session"])
        personal_output = output / f"personal_{index:02d}"
        personal_output.mkdir(parents=True, exist_ok=False)
        report = evaluate_session(source, evaluation / f"personal_{index:02d}", personal_output)
        atomic_json(personal_output / "report.json", report)
        reports.append(report)
    result = {
        "schema_version": 1,
        "status": "formal",
        "completion_status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_directory": str(evaluation.resolve()),
        "personal": [
            {"source_session": report["source_session"], "report": f"personal_{i:02d}/report.json", "trial_count": report["trial_count"]}
            for i, report in enumerate(reports)
        ],
        "source_files_unchanged": all(report["source_hashes_match"] for report in reports),
        "fixed_grid_not_selection": True,
        "limitations": [
            "The grid describes sensitivity to recorded sample-index window trims only.",
            "It does not correct hardware-clock mismatch or establish a better true neural onset.",
            "Presented labels are not independent gaze truth.",
            "The v14 checkpoint remains failed-selection diagnostic only.",
        ],
        **FALSE_FLAGS,
    }
    atomic_json(output / "summary.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args(argv)
    try:
        result = evaluate(Path(args.evaluation_directory).resolve(), Path(args.output_directory))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "formal", "completion_status": "incomplete", "error": str(exc), **FALSE_FLAGS}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"status": "formal", "completion_status": "complete", "report": str(Path(args.output_directory).resolve() / "summary.json"), **FALSE_FLAGS}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

