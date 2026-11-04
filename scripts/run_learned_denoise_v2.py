"""Explicit offline EEG v2 diagnostics; never repairs or deploys automatically."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np

from eeg_tools.learned_dataset import read_session_eeg
from eeg_tools.workstation.learned_monitor import _assess_continuity, _load_raw
from scripts.evaluate_denoiser_v2 import EvaluationError, _dump, sha256

SAFETY_METADATA = {
    "diagnostic_only": True, "automatic_processing_enabled": False,
    "automatic": False, "release_gate_passed": False, "release": False,
    "production_validated": False, "raw_is_immutable": True,
    "original_quality_unchanged": True, "missing_data_repaired": False,
    "warning": "Offline model hypothesis only: no clean ground truth, no automatic repair claim, no release approval.",
}
RESULT_FIELDS = ("denoised", "model_denoised", "artifact_probability", "confidence",
                 "replacement_mask", "valid_mask", "rejected_mask")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path, nargs="?", help="legacy positional session directory")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--session", type=Path, help="read-only NeuroStation session directory")
    source.add_argument("--input", type=Path, help="read-only NPZ with signal[C,T], sample_rate_hz and continuity evidence")
    parser.add_argument("--model", type=Path, default=Path("models/denoiser_v2.pt"))
    parser.add_argument("--config", type=Path, default=Path("configs/learned_denoiser_v2.json"))
    parser.add_argument("--output-dir", type=Path, help="new independent directory, outside the source session")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2, help="positive optional PyTorch CPU thread count")
    return parser


def _snapshot(source: Path) -> dict[str, str]:
    files = sorted(p for p in source.rglob("*") if p.is_file()) if source.is_dir() else [source]
    return {str(p.resolve()): sha256(p) for p in files}


def _ensure_output_separate(output: Path, source: Path, protected_files: tuple[Path, ...] = ()) -> None:
    output = output.resolve()
    source = source.resolve()
    if source.is_dir() and (output == source or source in output.parents):
        raise EvaluationError("output must be independent of the source session; derived-in-session is not allowed in v2")
    for path in (source, *protected_files):
        if output == path.resolve() or output in path.resolve().parents:
            raise EvaluationError("output cannot contain a protected source, model, or config file")
    if output.exists():
        raise EvaluationError("output directory already exists; select a new diagnostic output directory")


def _rate(value: Any) -> float:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind == "b":
        raise EvaluationError("sample_rate_hz must be a scalar number")
    try:
        rate = float(array.item())
    except (ValueError, TypeError) as error:
        raise EvaluationError("sample rate is unknown") from error
    if not np.isfinite(rate) or rate != 250.0:
        raise EvaluationError(f"input sample rate {rate!r} must be exactly 250 Hz; no implicit resampling")
    return rate


def _time_checks(timestamps: np.ndarray, samples: int, *, uniform_required: bool) -> None:
    if timestamps.shape != (samples,) or not np.all(np.isfinite(timestamps)):
        raise EvaluationError("continuity requires finite timestamps for every sample")
    delta = np.diff(timestamps)
    if np.any(delta < 0) or np.any(delta > 1.0):
        raise EvaluationError("input has reversed timestamps or an obvious timestamp gap")
    if uniform_required and not np.allclose(delta, 1.0 / 250.0, rtol=0.05, atol=1e-6):
        raise EvaluationError("without packet evidence, timestamps must establish a contiguous 250 Hz sample grid")


def _read_session(session: Path) -> tuple[np.ndarray, float, np.ndarray, dict]:
    metadata = json.loads((session / "session.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or "sampling_rate_hz" not in metadata:
        raise EvaluationError("session.json must explicitly declare sampling_rate_hz; v1's default is not accepted")
    rate = _rate(metadata["sampling_rate_hz"])
    raw_path = session / "raw_brainflow.tsv"
    _, indexes, timestamps, fields = _load_raw(raw_path)
    if not fields or len(indexes) < 2:
        raise EvaluationError("session requires EEG columns and at least two samples")
    # Reuse v1's reader only after checking metadata: no channel-count or rate guess.
    eeg, observed_rate, valid = read_session_eeg(session, expected_channels=len(fields))
    if observed_rate != rate or eeg.shape[1] != len(indexes):
        raise EvaluationError("read-only session readers disagree on samples or sample rate")
    _time_checks(timestamps, len(indexes), uniform_required=False)
    audit = _assess_continuity(raw_path, timestamps)
    if audit["status"] != "no_discontinuity_detected":
        raise EvaluationError(f"session packet continuity was not established: {audit['status']}")
    # _load_raw's fallback export counters are not treated as loss evidence.
    return eeg, rate, valid, audit


def _read_npz(path: Path) -> tuple[np.ndarray, float, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        names = [name for name in ("signal", "eeg", "raw") if name in data]
        if len(names) != 1:
            raise EvaluationError("offline NPZ must contain exactly one signal/eeg/raw array [C,T]")
        if "sample_rate_hz" not in data:
            raise EvaluationError("NPZ must explicitly declare sample_rate_hz")
        eeg = np.asarray(data[names[0]], dtype=np.float32)
        rate = _rate(data["sample_rate_hz"])
        if eeg.ndim != 2 or eeg.shape[0] < 1 or eeg.shape[1] < 2:
            raise EvaluationError("NPZ EEG must have shape [C,T] with C>=1,T>=2")
        valid = np.isfinite(eeg)
        if "valid_mask" in data:
            mask = np.asarray(data["valid_mask"])
            if mask.shape != eeg.shape or mask.dtype.kind != "b":
                raise EvaluationError("valid_mask must be boolean with shape [C,T]")
            valid &= mask
        packet_checked = "package_num" in data
        if packet_checked:
            packets = np.asarray(data["package_num"], dtype=np.float64)
            if packets.shape != (eeg.shape[1],) or not np.all(np.isfinite(packets)):
                raise EvaluationError("package_num must contain one finite packet number per sample")
            if np.any(packets < 0) or np.any(packets >= 256) or np.any(packets != np.floor(packets)):
                raise EvaluationError("package_num must be integer Cyton counters in [0,255]")
            if np.any(np.diff(packets) % 256 != 1):
                raise EvaluationError("NPZ packet discontinuity: no diagnostic inference")
        if "timestamp_s" in data:
            _time_checks(np.asarray(data["timestamp_s"], dtype=np.float64), eeg.shape[1],
                         uniform_required=not packet_checked)
        elif not packet_checked:
            raise EvaluationError("NPZ requires package_num or a contiguous timestamp_s grid, not guessed continuity")
    return eeg, rate, valid, {"status": "no_discontinuity_detected",
                             "packet_continuity": "checked" if packet_checked else "not_assessed",
                             "uniform_timestamps_checked": not packet_checked,
                             "missing_data_repaired": False, "data_integrity_certified": False}


def _validate_result(result: Any, raw: np.ndarray, valid: np.ndarray) -> None:
    for field in RESULT_FIELDS:
        if np.asarray(getattr(result, field)).shape != raw.shape:
            raise EvaluationError(f"v2 result {field} must share input shape [C,T]")
    for field in ("replacement_mask", "valid_mask", "rejected_mask"):
        if np.asarray(getattr(result, field)).dtype.kind != "b":
            raise EvaluationError(f"v2 result {field} must be boolean")
    if np.any(result.valid_mask & ~valid) or np.any(result.replacement_mask & (~result.valid_mask | result.rejected_mask)):
        raise EvaluationError("model may not mark invalid/rejected samples as repaired or replaced")
    retained = ~result.replacement_mask
    if not np.array_equal(result.denoised[retained], raw[retained], equal_nan=True):
        raise EvaluationError("gated output changed samples not selected for replacement")
    if np.any(result.valid_mask & ~np.isfinite(result.denoised)):
        raise EvaluationError("gated output is non-finite on valid input")


def _write_csv(path: Path, raw: np.ndarray, result: Any, rate: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "time_s", "channel", "raw", *RESULT_FIELDS])
        for sample in range(raw.shape[1]):
            for channel in range(raw.shape[0]):
                writer.writerow([sample, sample / rate, channel + 1, float(raw[channel, sample]),
                                 *(getattr(result, field)[channel, sample].item() for field in RESULT_FIELDS)])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.threads < 1:
            raise EvaluationError("threads must be positive")
        if args.session_dir and (args.session or args.input):
            raise EvaluationError("choose one of positional session, --session, or --input")
        source = args.input or args.session or args.session_dir
        if source is None:
            raise EvaluationError("--input or --session is required")
        source = source.resolve()
        is_npz = args.input is not None
        if is_npz and source.suffix.casefold() != ".npz":
            raise EvaluationError("--input accepts NPZ only")
        if (is_npz and not source.is_file()) or (not is_npz and not source.is_dir()):
            raise EvaluationError("input source is missing or has the wrong type")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = (args.output_dir or ROOT / "analysis_outputs" / "offline_diagnostics" /
                  "learned_denoise_v2" / f"{source.stem}_{stamp}_{uuid.uuid4().hex[:8]}").resolve()
        _ensure_output_separate(output, source, (args.model, args.config))
        before = _snapshot(source)
        eeg, rate, valid, continuity = _read_npz(source) if is_npz else _read_session(source)
        # The only optional ML import occurs after all read-only input checks.
        from eeg_tools.learned_denoise_v2 import MODEL_VERSION, apply_model_to_array, load_config, load_model, torch
        if torch is not None:
            torch.set_num_threads(args.threads)
        config = load_config(args.config)
        model_hash = sha256(args.model)
        config_hash = sha256(args.config)
        bundle = load_model(args.model, config, device=args.device)
        result = apply_model_to_array(bundle, eeg, valid_mask=valid)
        _validate_result(result, eeg, valid)
        if _snapshot(source) != before:
            raise EvaluationError("source changed during diagnostic read/inference; refusing derived output")
        if sha256(args.model) != model_hash or sha256(args.config) != config_hash:
            raise EvaluationError("model or config changed during diagnostic inference")
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output / "denoised_signal.npz", raw=eeg,
                            **{field: getattr(result, field) for field in RESULT_FIELDS},
                            sample_rate_hz=np.asarray(rate), diagnostic_only=np.asarray(True),
                            automatic=np.asarray(False), release=np.asarray(False))
        _write_csv(output / "denoised_signal.csv", eeg, result, rate)
        report = {
            "schema_version": 2, "status": "completed", "threads": args.threads, **SAFETY_METADATA,
            "model_version": MODEL_VERSION, "model_path": str(args.model.resolve()),
            "model_sha256": model_hash, "config_path": str(args.config.resolve()), "config_sha256": config_hash,
            "calibrated_threshold": bundle.calibrated_threshold,
            "threshold_calibration": bundle.metadata["threshold_calibration"],
            "model_validation_status": config["runtime"]["validation_status"],
            "source": str(source), "source_sha256": before, "source_hashes_unchanged": True,
            "sample_rate_hz": rate, "channel_count": eeg.shape[0], "sample_count": eeg.shape[1],
            "replacement_fraction": float(np.mean(result.replacement_mask)),
            "invalid_fraction": float(np.mean(~result.valid_mask)),
            "rejected_fraction": float(np.mean(result.rejected_mask)),
            "data_integrity_assessment": continuity, "diagnostics": result.diagnostics,
            "generated_at": datetime.now(timezone.utc).isoformat(), "output_dir": str(output),
        }
        (output / "quality_metrics.json").write_text(_dump(report), encoding="utf-8")
        (output / "model_manifest.json").write_text(_dump({**report, "generated_files": [
            "denoised_signal.npz", "denoised_signal.csv", "quality_metrics.json", "model_manifest.json"]}), encoding="utf-8")
        if _snapshot(source) != before:
            raise EvaluationError("source changed while exporting; derived outputs must not be interpreted as stable diagnostics")
        print(_dump(report))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError, AttributeError) as error:
        print(_dump({"status": "error", **SAFETY_METADATA, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
