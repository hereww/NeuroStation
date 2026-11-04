"""Fail-closed final verification for the v14 personal diagnostic artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
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
        raise ValueError("v14 final verification: " + message)


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


def verify(evaluation: Path, figures: Path) -> dict:
    import numpy as np
    summary = read_json(evaluation / "summary.json")
    require(summary.get("status") == "formal", "evaluation status is not formal")
    require(summary.get("completion_status") == "complete", "evaluation is incomplete")
    require(summary.get("allow_failed_selection_diagnostic") is True,
            "diagnostic opt-in is missing")
    model = summary.get("model", {})
    require(model.get("selected_epoch") is None, "selected epoch must remain null")
    require(model.get("selection_gate_passed") is False, "selection gate must remain false")
    require(model.get("checkpoint_role") == "diagnostic_last_epoch_no_eligible_checkpoint",
            "checkpoint role is not failed-selection diagnostic")
    for key, value in FALSE_FLAGS.items():
        require(summary.get(key) is value, f"unsafe summary flag: {key}")
        require(model.get(key) is value, f"unsafe model flag: {key}")

    source_bindings = {}
    track_results = []
    for index, personal in enumerate(summary.get("personal", [])):
        report_path = evaluation / personal["report"]
        report = read_json(report_path)
        require(report.get("source_hashes_match") is True, f"source changed for personal_{index:02d}")
        require(report.get("presented_matches_are_gaze_accuracy") is False,
                f"unsafe gaze flag in personal_{index:02d}")
        source = Path(report["source_session"])
        source_bindings[str(source)] = {
            "before": report["source_sha256_before"],
            "after": report["source_sha256_after"],
            "match": report["source_sha256_before"] == report["source_sha256_after"],
        }
        personal_dir = report_path.parent
        arrays = {}
        for track in TRACKS:
            path = personal_dir / f"{track}.npy"
            require(path.is_file(), f"missing {path}")
            value = np.load(path, allow_pickle=False)
            require(value.ndim == 2 and value.shape[1] == 8, f"bad shape for {path}")
            require(value.dtype.kind == "f" and np.isfinite(value).all(), f"bad values for {path}")
            arrays[track] = value
        shape = arrays["raw"].shape
        require(all(value.shape == shape for value in arrays.values()),
                f"track shape mismatch in personal_{index:02d}")
        track_results.append({
            "personal_index": index,
            "source_session": report["source_session"],
            "shape": list(shape),
            "tracks": report["tracks"],
            "source_hashes_match": report["source_hashes_match"],
        })

    receipt_path = figures / "figures_receipt.json"
    receipt = read_json(receipt_path)
    require(receipt.get("evaluation_summary_sha256") == sha256(evaluation / "summary.json"),
            "figure receipt does not bind to evaluation summary")
    require(len(receipt.get("figures", {})) == 6, "expected six diagnostic figures")
    for name, metadata in receipt["figures"].items():
        path = figures / name
        require(path.is_file(), f"missing figure {path}")
        require(metadata.get("sha256") == sha256(path), f"figure SHA mismatch: {name}")

    return {
        "schema_version": 1,
        "status": "formal",
        "completion_status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_directory": str(evaluation.resolve()),
        "figures_directory": str(figures.resolve()),
        "evaluation_summary_sha256": sha256(evaluation / "summary.json"),
        "figures_receipt_sha256": sha256(receipt_path),
        "source_bindings": source_bindings,
        "track_results": track_results,
        "checks": {
            "two_personal_sessions": len(track_results) == 2,
            "source_files_unchanged": all(item["match"] for item in source_bindings.values()),
            "all_track_shapes_match": True,
            "all_track_arrays_finite": True,
            "six_figures_present_and_bound": True,
            "failed_selection_remains_fail_closed": True,
        },
        "conclusion": "diagnostic_artifacts_complete_but_not_deployable",
        "limitations": [
            "v14 failed checkpoint is included only under explicit diagnostic opt-in.",
            "Recorded-presentation matches are not gaze accuracy.",
            "No clean physiological reference, independent gaze truth, hardware calibration, or clinical validation.",
            "The verification does not establish that the original personal data are invalid or that v14 improves physiology.",
        ],
        **FALSE_FLAGS,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-directory", required=True)
    parser.add_argument("--figures-directory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = verify(Path(args.evaluation_directory).resolve(), Path(args.figures_directory).resolve())
        output = create_output(Path(args.output))
        atomic_json(output / "report.json", result)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "formal", "completion_status": "incomplete", "error": str(exc), **FALSE_FLAGS}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"status": "formal", "completion_status": "complete", "report": str(output / "report.json"), **FALSE_FLAGS}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
