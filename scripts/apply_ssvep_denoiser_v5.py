"""Materialize a formal, non-overwriting SSVEP v5 candidate.

This exporter is deliberately fail-closed.  It accepts only the frozen reference
inside the trusted workspace, performs the evaluator's metadata-only catalog
check before loading a checkpoint or waveform, and publishes each artifact
exclusively.  It never removes a candidate directory recursively.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eeg_tools import ssvep_denoise_v5 as core
from eeg_tools import ssvep_task_eval_v5 as task_eval


class ApplySSVEPDenoiserV5Error(ValueError):
    """A safety, binding, or publication contract failed."""


def _fail(message: str) -> None:
    raise ApplySSVEPDenoiserV5Error(message)


def _sha(path: Path) -> str:
    path = Path(path)
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _stamp(path: Path) -> tuple[int, int, int, int, int]:
    try:
        stat = Path(path).stat()
    except OSError as exc:
        raise ApplySSVEPDenoiserV5Error(f"missing/unreadable input: {path}") from exc
    if not Path(path).is_file():
        _fail(f"input must be a regular file: {path}")
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino


def _safe_inside(value: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a workspace path while rejecting links, traversal, and foreign drives."""
    text = str(value)
    parts = text.replace("\\", "/").split("/")
    if ".." in parts:
        _fail("path traversal refused")
    if PureWindowsPath(text).drive.casefold() == "d:":
        _fail("D: paths are forbidden")
    try:
        path = task_eval._inside(ROOT, value)
    except Exception as exc:
        raise ApplySSVEPDenoiserV5Error(str(exc)) from exc
    if must_exist and not path.exists():
        _fail(f"path does not exist: {path}")
    return path


def _trusted_file(value: str | Path) -> Path:
    path = _safe_inside(value, must_exist=True)
    _stamp(path)
    return path


def _trusted_reference(value: str | Path) -> Path:
    requested = _safe_inside(value, must_exist=True)
    fixed = _safe_inside(core.REFERENCE_RELATIVE, must_exist=True)
    if requested != fixed:
        _fail("reference-dir must be the fixed trusted v5 reference directory")
    if not requested.is_dir():
        _fail("trusted reference must be a directory")
    return requested


def _candidate_output(value: str | Path) -> Path:
    path = _safe_inside(value)
    processed = _safe_inside(Path("datasets") / "processed", must_exist=True)
    if path.parent != processed:
        _fail("output-dir must be an independent direct child of datasets/processed")
    if path == processed or path.name in {"", ".", ".."}:
        _fail("invalid candidate output directory")
    if path.exists() or path.is_symlink():
        _fail("refuse overwrite candidate artifacts")
    return path


def _report_output(value: str | Path) -> Path:
    path = _safe_inside(value)
    reports = _safe_inside(Path("analysis_outputs"), must_exist=True)
    if path.parent != reports:
        _fail("output-report must be a direct child of analysis_outputs")
    if path.suffix.casefold() != ".json":
        _fail("output-report must be a JSON file")
    if path.exists() or path.is_symlink():
        _fail("refuse overwrite report artifact")
    return path


def _json_bytes(value: dict[str, Any]) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ApplySSVEPDenoiserV5Error("metadata must be finite JSON") from exc


def _publish_file(path: Path, writer) -> None:
    """Write a temporary sibling and hard-link it into place, never replacing."""
    path.parent.mkdir(parents=False, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pending", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise ApplySSVEPDenoiserV5Error(f"refuse overwrite artifact: {path}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _publish_json(path: Path, value: dict[str, Any]) -> None:
    payload = _json_bytes(value)
    _publish_file(path, lambda handle: handle.write(payload))


def _publish_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    # Passing the open handle is intentional: np.savez_compressed(path, ...) can
    # append .npz and leave the exclusive publication target empty or missing.
    _publish_file(path, lambda handle: np.savez_compressed(handle, **arrays))


def _candidate_names(value: Any) -> None:
    try:
        task_eval._candidate_names(value)
    except Exception as exc:
        raise ApplySSVEPDenoiserV5Error(str(exc)) from exc


def _assert_snapshot(reference: Path, catalog: dict, config_path: Path, config_sha: str,
                     checkpoint_path: Path, checkpoint_sha: str) -> None:
    task_eval._verify_unchanged(ROOT, catalog)
    if _sha(config_path) != config_sha or _stamp(config_path) != catalog["config_stamp"]:
        _fail("config changed during export")
    if _sha(checkpoint_path) != checkpoint_sha:
        _fail("checkpoint changed during export")
    manifest = reference / "manifest.json"
    if _sha(manifest) != catalog["manifest_hash"]:
        _fail("reference manifest changed during export")


def _load_reference_subject(catalog: dict, item: dict) -> dict[str, np.ndarray]:
    try:
        return task_eval._read_subject(catalog, item, "reference")
    except Exception as exc:
        raise ApplySSVEPDenoiserV5Error(str(exc)) from exc


def _subject_sidecar(item: dict, reference_side: dict, signal: np.ndarray, destination: Path,
                     config: dict, checkpoint_sha: str) -> dict[str, Any]:
    shape = list(signal.shape)
    side: dict[str, Any] = {
        "schema_version": 1,
        "reference_version": task_eval.REFERENCE_VERSION,
        "status": "formal",
        "dataset": reference_side["dataset"],
        "subject_id": reference_side["subject_id"],
        "group_key": reference_side["group_key"],
        "split": reference_side["split"],
        "native_sample_rate_hz": reference_side["native_sample_rate_hz"],
        "onset_sample_zero_based": reference_side["onset_sample_zero_based"],
        "frequency_order_hz": reference_side["frequency_order_hz"],
        "phase_pi_multiples": reference_side["phase_pi_multiples"],
        "channel_order": reference_side["channel_order"],
        "native_units": reference_side["native_units"],
        "output_axes": reference_side["output_axes"],
        "trial_order": reference_side["trial_order"],
        "electrode_order": reference_side["electrode_order"],
        "materialized_trials": int(signal.shape[0]),
        "signal_key": config["candidate_signal_key"],
        "signal_shape": shape,
        "npz_relative_path": destination.name,
        "reference_npz_sha256": item["npz_hash"],
        "reference_manifest_sha256": item["side_hash"],
        "model_version": core.MODEL_VERSION,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_model_version": core.MODEL_VERSION,
        "reference_signal_key": "reference",
        "reference_signal_shape": list(reference_side.get("reference_shape", shape)),
        "reference_trial_count": int(reference_side.get("materialized_trials", signal.shape[0])),
        "reference_window_samples": int(task_eval.WINDOW_LENGTHS[reference_side["dataset"]]),
        "reference_sample_rate_hz": reference_side["native_sample_rate_hz"],
        "reference_native_units": reference_side["native_units"],
        "npz_members": sorted(["manifest_json", config["candidate_signal_key"], *task_eval.LABEL_KEYS,
                               *( ["phase_pi_multiple"] if reference_side["phase_pi_multiples"] is not None else [] )]),
    }
    _candidate_names(side)
    return side


def _labels_from_reference(arrays: dict[str, np.ndarray], reference_side: dict) -> dict[str, np.ndarray]:
    labels = {key: np.asarray(arrays[key]).copy() for key in task_eval.LABEL_KEYS}
    if reference_side["phase_pi_multiples"] is not None:
        labels["phase_pi_multiple"] = np.asarray(arrays["phase_pi_multiple"]).copy()
    return labels


def _write_failure_record(output: Path, error: BaseException) -> None:
    if not output.exists() or not output.is_dir():
        return
    record = {
        "schema_version": 1,
        "status": "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "error_type": type(error).__name__,
        "error": str(error),
    }
    try:
        _publish_json(output / "failure_record.json", record)
    except Exception:
        # The original failure is the authoritative result. Never recurse or
        # delete the candidate directory while trying to record diagnostics.
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=core.REFERENCE_RELATIVE)
    parser.add_argument("--config", type=Path, default=core.CONFIG_RELATIVE)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cuda:0", "cpu"), default="cuda")
    args = parser.parse_args(argv)

    output: Path | None = None
    try:
        reference = _trusted_reference(args.reference_dir)
        config_path = _trusted_file(args.config)
        checkpoint_path = _trusted_file(args.checkpoint)
        output = _candidate_output(args.output_dir)
        report_path = _report_output(args.output_report)
        if output == report_path or output in report_path.parents:
            _fail("candidate and report artifacts must be independent")

        config_sha = _sha(config_path)
        config_stamp = _stamp(config_path)
        config = core.load_config(config_path, root=ROOT)
        if _sha(config_path) != config_sha or _stamp(config_path) != config_stamp:
            _fail("config changed while loading")
        if config.get("candidate_signal_key") != "processed_v5":
            _fail("processed_v5 candidate interface required")

        manifest_path = reference / "manifest.json"
        # Metadata-only reference preflight. _catalog validates every sidecar
        # and its declared SHA before any checkpoint or waveform is loaded.
        inventory = task_eval._inventory(ROOT, reference, False)
        catalog = task_eval._catalog(ROOT, reference, False, inventory)
        catalog["config_stamp"] = config_stamp
        if catalog["manifest_hash"] != core.REFERENCE_MANIFEST_SHA256:
            _fail("reference manifest hash mismatch")
        if _sha(manifest_path) != core.REFERENCE_MANIFEST_SHA256:
            _fail("reference manifest hash mismatch")

        checkpoint_sha = _sha(checkpoint_path)
        checkpoint_stamp = _stamp(checkpoint_path)
        checkpoint = core.load_checkpoint(checkpoint_path, config, args.device)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("loaded_models"), dict):
            _fail("checkpoint loader must return a dict containing loaded_models")
        loaded_models = checkpoint["loaded_models"]
        if _sha(checkpoint_path) != checkpoint_sha or _stamp(checkpoint_path) != checkpoint_stamp:
            _fail("checkpoint changed while loading")
        _assert_snapshot(reference, catalog, config_path, config_sha, checkpoint_path, checkpoint_sha)

        rows = sorted(catalog["subjects"].items())
        if not rows:
            _fail("reference contains no eligible subjects")
        output.mkdir(parents=False, exist_ok=False)
        created_subjects: list[dict[str, Any]] = []
        try:
            for key, item in rows:
                _assert_snapshot(reference, catalog, config_path, config_sha, checkpoint_path, checkpoint_sha)
                arrays = _load_reference_subject(catalog, item)
                reference_side = item["side"]
                signal = np.asarray(arrays["reference"])
                processed = core.denoise_array(
                    loaded_models[reference_side["dataset"]], signal,
                    int(reference_side["native_sample_rate_hz"]), config, args.device,
                )
                if not isinstance(processed, np.ndarray) or processed.shape != signal.shape or not np.isfinite(processed).all():
                    _fail("denoiser returned invalid processed_v5 array")
                relative = Path(reference_side["dataset"]) / f"subject_{int(reference_side['subject_id']):03d}.npz"
                destination = output / relative
                destination.parent.mkdir(parents=False, exist_ok=True)
                side = _subject_sidecar(item, reference_side, signal, destination, config, checkpoint_sha)
                embedded = {k: v for k, v in side.items() if k not in {"npz_sha256", "npz_relative_path"}}
                labels = _labels_from_reference(arrays, reference_side)
                _publish_npz(destination, {
                    config["candidate_signal_key"]: np.asarray(processed),
                    **labels,
                    "manifest_json": np.asarray(json.dumps(embedded, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
                })
                side["npz_sha256"] = _sha(destination)
                side_path = Path(str(destination) + ".manifest.json")
                _publish_json(side_path, side)
                created_subjects.append({
                    "dataset": key[0], "subject_id": key[1], "split": reference_side["split"],
                    "relative_path": relative.as_posix(), "npz_sha256": side["npz_sha256"],
                    "manifest_sha256": _sha(side_path), "trials": int(signal.shape[0]),
                })
                _assert_snapshot(reference, catalog, config_path, config_sha, checkpoint_path, checkpoint_sha)

            root_manifest = {
                "schema_version": 1,
                "reference_version": task_eval.REFERENCE_VERSION,
                "status": "formal",
                "build_completed": True,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "reference_manifest_sha256": core.REFERENCE_MANIFEST_SHA256,
                "model_version": core.MODEL_VERSION,
                "checkpoint_sha256": checkpoint_sha,
                "config_sha256": config_sha,
                "candidate_signal_key": config["candidate_signal_key"],
                "materialized_subjects": created_subjects,
                "runtime": {"enabled": False, "production_validated": False},
                "holdout_waveforms_loaded": False,
                "outer_validation_used_for_selection": False,
            }
            _candidate_names(root_manifest)
            _publish_json(output / "manifest.json", root_manifest)
            report = {
                "schema_version": 1,
                "status": "formal",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_dir": output.relative_to(ROOT).as_posix(),
                "candidate_manifest_sha256": _sha(output / "manifest.json"),
                "reference_manifest_sha256": core.REFERENCE_MANIFEST_SHA256,
                "checkpoint_sha256": checkpoint_sha,
                "config_sha256": config_sha,
                "subjects_materialized": len(created_subjects),
                "split_counts": {
                    "train": sum(row["split"] == "train" for row in created_subjects),
                    "validation": sum(row["split"] == "validation" for row in created_subjects),
                },
                "holdout_waveforms_loaded": False,
                "runtime_enabled": False,
                "production_validated": False,
                "task_metrics_computed": False,
            }
            _publish_json(report_path, report)
        except BaseException as exc:
            _write_failure_record(output, exc)
            raise

        print(json.dumps({"status": "completed", "candidate_dir": str(output),
                          "subjects": len(created_subjects), "holdout_loaded": False,
                          "runtime_enabled": False}, ensure_ascii=False))
        return 0
    except ApplySSVEPDenoiserV5Error:
        raise
    except Exception as exc:
        if output is not None:
            _write_failure_record(output, exc)
        raise ApplySSVEPDenoiserV5Error(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())

