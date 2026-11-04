"""Evaluate the fixed posterior-channel CCA decoder on personal recordings.

This is an explicit offline diagnostic. It uses configured channel metadata and
recorded presentation labels only for the report comparison; labels never enter
the decoder forward pass. It does not claim gaze accuracy or deployment safety.
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
        raise ValueError("v2 personal decoder evaluator: " + message)


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


def evaluate_session(source: Path, v14_personal: Path, output: Path) -> dict:
    import numpy as np
    from eeg_tools import personal_ssvep_decoder as v1
    from eeg_tools import personal_ssvep_decoder_v2 as v2
    from eeg_tools.workstation import v9_offline as baseline
    from scripts.diagnose_neurostation_v9_output import parse_raw, validate_metadata

    before = baseline._source_hashes(source)
    validate_metadata(read_json(source / "session.json"), read_json(source / "channel_config.json"))
    raw, header = parse_raw((source / "raw_brainflow.tsv").read_bytes())
    targets, trials, reason = baseline._classification_inputs(source, raw)
    require(not reason, reason or "classification metadata unavailable")
    target_ids = {item[0] for item in targets}
    arrays = {"raw": raw, "notch": None, "traditional": None, "v14": None}
    # Keep v1 baseline paths and v14 path bound to the already completed v14 evaluation.
    arrays["notch"] = np.load(v14_personal / "notch.npy", allow_pickle=False)
    arrays["traditional"] = np.load(v14_personal / "traditional.npy", allow_pickle=False)
    arrays["v14"] = np.load(v14_personal / "v14.npy", allow_pickle=False)
    require(all(value.shape == raw.shape and value.dtype.kind == "f" and np.isfinite(value).all() for value in arrays.values()), "track arrays do not match source raw")
    rows = []
    summaries = {"v1": {}, "v2": {}}
    for trial in trials:
        row = {"trial_index": trial["trial_index"], "presented_target_id": trial.get("presented_target_id", ""), "onset_sample_index": trial["onset_sample_index"], "offset_sample_index": trial["offset_sample_index"], "window_start_sample_index": trial["window_start_sample_index"], "predictions": {"v1": {}, "v2": {}}}
        start, end = trial["window_start_sample_index"], trial["offset_sample_index"]
        for decoder_name, decoder in (("v1", v1.decode_window), ("v2", v2.decode_window)):
            for track, value in arrays.items():
                try:
                    prediction = decoder(value[start:end].T, 250.0, targets)
                except (ValueError, ArithmeticError) as exc:
                    prediction = {"available": False, "target_id": None, "reason": str(exc)}
                row["predictions"][decoder_name][track] = prediction
        rows.append(row)
    for decoder_name in ("v1", "v2"):
        for track in TRACKS:
            available = [row["predictions"][decoder_name][track] for row in rows if row["predictions"][decoder_name][track].get("available") and row["presented_target_id"] in target_ids]
            margins = [item["confidence"] for item in available]
            summaries[decoder_name][track] = {
                "recorded_trial_count": len(rows),
                "available_presented_trial_count": len(available),
                "presented_match_count": sum(
                    row["predictions"][decoder_name][track].get("available") and row["predictions"][decoder_name][track].get("target_id") == row["presented_target_id"]
                    for row in rows
                ),
                "mean_top_two_margin": math.fsum(margins) / len(margins) if margins else None,
                "margin_is_probability": False,
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
        "decoder_v1": v1.config_summary(),
        "decoder_v2": v2.config_summary(),
        "presented_matching_definition": "matches to recorded presentation, not independent gaze accuracy",
        "channel_selection": {"decoder_v2_indices": [4, 5, 6, 7], "decoder_v2_names": ["CH5", "CH6", "CH7", "CH8"], "reason": "a priori posterior visual-channel subset from preset map"},
        "limitations": [
            "Posterior-channel v2 is a fixed offline diagnostic, not production validation.",
            "The configured electrode positions require physical verification.",
            "Presented matches are not independent gaze accuracy.",
            "No clean physiological reference or no-target rejection threshold is available.",
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
        "personal": [{"source_session": report["source_session"], "report": f"personal_{i:02d}/report.json", "v1": report["tracks"]["v1"], "v2": report["tracks"]["v2"]} for i, report in enumerate(reports)],
        "source_files_unchanged": all(report["source_hashes_match"] for report in reports),
        "v2_is_fixed_posterior_diagnostic": True,
        "limitations": [
            "This result improves the decoder diagnostic only; it does not prove denoising quality.",
            "No default desktop runtime was changed.",
            "Presented labels are not independent gaze truth.",
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


