"""Build a fail-closed personal SSVEP calibration diagnostic.

The script uses recorded presentation targets only for calibration and post-hoc
holdout comparison. It never treats those labels as independent gaze truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
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
    "checkpoint_selection_performed": False,
    "default_deployment_authorized": False,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("personal calibration evaluator: " + message)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_session(source: Path) -> dict:
    import numpy as np
    from eeg_tools import personal_ssvep_decoder_v2 as v2
    from eeg_tools.workstation import v9_offline as baseline
    from scripts.diagnose_neurostation_v9_output import parse_raw, validate_metadata

    config = __import__("eeg_tools.personal_ssvep_calibration_v1", fromlist=["get_config"]).get_config()
    before = baseline._source_hashes(source)
    validate_metadata(read_json(source / "session.json"), read_json(source / "channel_config.json"))
    raw, header = parse_raw((source / "raw_brainflow.tsv").read_bytes())
    targets, trials, reason = baseline._classification_inputs(source, raw)
    require(not reason, reason or "classification metadata unavailable")
    start_offset = int(config["window_start_after_onset_samples"])
    length = int(config["window_samples"])
    rows = []
    for trial in trials:
        start = int(trial["onset_sample_index"]) + start_offset
        end = start + length
        require(0 <= start < end <= raw.shape[0], "calibration window falls outside raw recording")
        score = v2.decode_window(raw[start:end].T, 250.0, targets)
        rows.append({
            "trial_index": trial["trial_index"],
            "presented_target_id": trial.get("presented_target_id", ""),
            "onset_sample_index": trial["onset_sample_index"],
            "window_start_sample_index": start,
            "window_end_sample_index_exclusive": end,
            "scores": score["scores"],
            "forced_reference_target_id": score["target_id"],
            "forced_reference_confidence": score["confidence"],
        })
    after = baseline._source_hashes(source)
    require(before == after, "source data changed")
    return {
        "source_session": str(source),
        "source_header": header,
        "source_sha256_before": before,
        "source_sha256_after": after,
        "source_hashes_match": True,
        "targets": targets,
        "rows": rows,
        "counts_by_target": dict(Counter(row["presented_target_id"] for row in rows)),
    }


def _loto_diagnostic(scores, labels, target_ids, calibration):
    import numpy as np
    from eeg_tools import personal_ssvep_calibration_v1 as cal

    matches = []
    predictions = []
    for index in range(len(labels)):
        train_scores = np.delete(scores, index, axis=0)
        train_labels = [label for j, label in enumerate(labels) if j != index]
        model = cal.fit_calibrator(train_scores, train_labels, target_ids, minimum_repetitions=2, allow_ineligible=True)
        if not model.get("parameters"):
            matches.append(False)
            predictions.append(None)
            continue
        diagnostic_model = dict(model)
        diagnostic_model["eligible"] = True
        diagnostic_model["holdout_validation_passed"] = True
        prediction = cal.predict(diagnostic_model, scores[index])
        predictions.append(prediction.get("target_id"))
        matches.append(prediction.get("target_id") == labels[index])
    count = len(labels)
    rate = float(sum(matches) / count) if count else None
    return {"trial_count": count, "accepted_proxy_match_count": int(sum(matches)), "accepted_proxy_match_rate": rate, "predictions": predictions, "definition": "leave-one-trial-out match to recorded presentation proxy; not gaze accuracy"}


def evaluate(sessions: list[Path], output_path: Path) -> dict:
    import numpy as np
    from eeg_tools import personal_ssvep_calibration_v1 as cal

    config = cal.get_config()
    output = create_output(output_path)
    reports = [_extract_session(source.resolve()) for source in sessions]
    target_ids = [target[0] for target in reports[0]["targets"]]
    require(all([report["targets"] == reports[0]["targets"] for report in reports]), "session target definitions differ")
    score_rows = []
    for report in reports:
        for row in report["rows"]:
            score_rows.append((row["scores"], row["presented_target_id"]))
    scores = np.asarray([[float(item[target]) for target in target_ids] for item, _ in score_rows], dtype=np.float64)
    labels = [label for _, label in score_rows]
    model = cal.fit_calibrator(scores, labels, target_ids)
    diagnostic = _loto_diagnostic(scores, labels, target_ids, model)
    holdout_passed = bool(model.get("eligible") and diagnostic["trial_count"] >= config["minimum_holdout_trials"] and diagnostic["accepted_proxy_match_rate"] is not None and diagnostic["accepted_proxy_match_rate"] >= config["minimum_holdout_match_rate"])
    model["holdout_validation_passed"] = holdout_passed
    model["runtime_enabled"] = False
    model["default_deployment_authorized"] = False
    for index, report in enumerate(reports):
        atomic_json(output / f"personal_{index:02d}.json", {
            "status": "formal",
            "completion_status": "complete",
            "source_session": report["source_session"],
            "source_sha256_before": report["source_sha256_before"],
            "source_sha256_after": report["source_sha256_after"],
            "source_hashes_match": report["source_hashes_match"],
            "targets": report["targets"],
            "counts_by_target": report["counts_by_target"],
            "trial_count": len(report["rows"]),
            "window": {"start_after_onset_samples": config["window_start_after_onset_samples"], "samples": config["window_samples"]},
            "label_source": config["label_source"],
            **FALSE_FLAGS,
        })
    combined_counts = dict(Counter(labels))
    minimum_repetitions = int(config["minimum_repetitions_per_target"])
    missing_targets = {
        target: count for target, count in combined_counts.items()
        if count < minimum_repetitions
    }
    missing_targets.update({
        target: 0 for target in target_ids if target not in combined_counts
    })
    limitations = []
    if missing_targets:
        details = ", ".join(
            f"{target}={missing_targets[target]}" for target in target_ids
            if target in missing_targets
        )
        limitations.append(
            f"Per-target repetition minimum is {minimum_repetitions}; insufficient targets: {details}."
        )
    else:
        limitations.append(
            f"Per-target repetition minimum of {minimum_repetitions} is met; this does not establish independent gaze truth."
        )
    if not holdout_passed:
        limitations.append(
            "Holdout validation has not passed; the calibrated model remains diagnostic-only."
        )
    limitations.extend([
        "Recorded presentation labels are not independent gaze truth.",
        "The calibrated model cannot be deployed without holdout validation and hardware/electrode verification.",
    ])
    result = {
        "schema_version": 1,
        "status": "formal",
        "completion_status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "personal": [{"source_session": report["source_session"], "report": f"personal_{i:02d}.json", "counts_by_target": report["counts_by_target"]} for i, report in enumerate(reports)],
        "combined_trial_count": len(labels),
        "combined_counts_by_target": combined_counts,
        "calibration_config": cal.config_summary(),
        "calibrator": model,
        "diagnostic_leave_one_trial_out": diagnostic,
        "source_files_unchanged": all(report["source_hashes_match"] for report in reports),
        "calibration_eligible": bool(model.get("eligible")),
        "holdout_validation_passed": holdout_passed,
        "label_definition": config["label_source"],
        "limitations": limitations,
        **FALSE_FLAGS,
    }
    atomic_json(output / "summary.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", action="append", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.session, args.output_directory)
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
        print(json.dumps({"status": "formal", "completion_status": "incomplete", "error": str(exc), **FALSE_FLAGS}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"status": "formal", "completion_status": "complete", "report": str((args.output_directory / "summary.json").resolve()), **FALSE_FLAGS}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

