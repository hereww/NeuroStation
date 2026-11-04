"""Explicit offline v3 applicability diagnostics, never automatic signal repair."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib
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
from scripts import run_learned_denoise_v2 as readers
from scripts.evaluate_denoiser_v2 import EvaluationError, _dump, sha256

RESULT_FIELDS = (*readers.RESULT_FIELDS, "support_probability", "domain_rejected_mask")
SAFETY_METADATA = {
    **readers.SAFETY_METADATA, "domain_probability_is_artifact_probability": False,
    "domain_probability_is_repair_success_probability": False,
    "warning": "Explicit offline v3 applicability hypothesis only; support score is NOT artifact/repair quality, no production or clinical approval.",
}


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path, nargs="?", help="explicit legacy positional session")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--session", type=Path)
    source.add_argument("--input", type=Path, help="NPZ signal/eeg/raw[C,T] with explicit rate and continuity")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/domain_gate_v3.pt"))
    parser.add_argument("--denoiser-checkpoint", type=Path, default=Path("models/denoiser_v2.pt"))
    parser.add_argument("--config", type=Path, default=Path("configs/learned_denoiser_v3.json"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=positive_int, default=2)
    return parser


def resolve(root: Path, path: Path) -> Path:
    return (path if path.is_absolute() else root / path).resolve()


def load_bundle(checkpoint: Path, denoiser_checkpoint: Path, config: Path,
                device: str, threads: int) -> tuple[Any, Any]:
    """Lazy optional ML dependency; never installed or imported by a Qt entry point."""
    torch = importlib.import_module("torch")
    torch.set_num_threads(threads)
    if device == "cuda" and not torch.cuda.is_available():
        raise EvaluationError("CUDA explicitly requested but unavailable; no silent CPU fallback")
    core = importlib.import_module("eeg_tools.learned_denoise_v3")
    cfg = core.load_config(config)
    runtime = cfg.get("runtime", {})
    if runtime.get("enabled") is not False or runtime.get("production_validated") is not False:
        raise EvaluationError("v3 runtime must explicitly disable automatic processing and production validation")
    bundle = core.load_model(checkpoint, denoiser_checkpoint, config=cfg, device=device)
    threshold = bundle.support_threshold
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0.5 <= threshold <= 1:
        raise EvaluationError("frozen support threshold must be a finite checkpoint value in [0.5,1]")
    return core, bundle


def validate_result(result: Any, raw: np.ndarray, valid: np.ndarray, support_threshold: float) -> None:
    readers._validate_result(result, raw, valid)
    for field in ("support_probability", "domain_rejected_mask"):
        if np.asarray(getattr(result, field)).shape != raw.shape:
            raise EvaluationError(f"v3 {field} must share input [C,T] shape")
    if result.domain_rejected_mask.dtype.kind != "b":
        raise EvaluationError("domain_rejected_mask must be boolean")
    if not np.array_equal(result.valid_mask, valid):
        raise EvaluationError("v3 cannot exclude valid source/test samples from metric support")
    if np.any(result.replacement_mask & result.domain_rejected_mask):
        raise EvaluationError("unsupported samples cannot be replaced")
    refused = result.domain_rejected_mask | ~valid
    if not np.array_equal(result.denoised[refused], raw[refused], equal_nan=True):
        raise EvaluationError("unsupported/invalid fallback must retain raw values exactly in original units")
    probability = result.support_probability
    finite = np.isfinite(probability)
    if np.any(finite & ((probability < 0) | (probability > 1))):
        raise EvaluationError("support score must be in [0,1]")
    accepted = valid & ~result.domain_rejected_mask
    if np.any(accepted & (~finite | (probability <= support_threshold))):
        raise EvaluationError("domain acceptance must use the frozen strict > support threshold")
    if np.any(valid & ~np.isfinite(result.model_denoised)):
        raise EvaluationError("candidate must stay finite on valid source/test samples")


def read_source(source: Path, is_npz: bool) -> tuple[np.ndarray, float, np.ndarray, dict]:
    # Reuse v2's conservative continuity and explicit sampling-rate checks.
    # Recover the stored numeric precision rather than letting its float32 reader
    # quantize an unsupported float64 track before raw fallback is checked.
    checked, rate, checked_valid, continuity = readers._read_npz(source) if is_npz else readers._read_session(source)
    if is_npz:
        with np.load(source, allow_pickle=False) as archive:
            name = next(key for key in ("signal", "eeg", "raw") if key in archive)
            raw = archive[name].copy()
            if raw.dtype.kind not in "fiu":
                raise EvaluationError("input raw must be real numeric")
            valid = np.isfinite(raw)
            if "valid_mask" in archive:
                valid &= archive["valid_mask"]
    else:
        raw_path = source / "raw_brainflow.tsv"
        _, _, _, names = readers._load_raw(raw_path)
        rows = []
        with raw_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                values = []
                for name in names:
                    try:
                        value = float(row[name])
                    except (KeyError, ValueError, TypeError):
                        value = float("nan")
                    values.append(value if np.isfinite(value) else float("nan"))
                rows.append(values)
        raw = np.asarray(rows, dtype=np.float64).T
        valid = np.isfinite(raw)
    if raw.shape != checked.shape or not np.array_equal(valid, checked_valid):
        raise EvaluationError("stored-precision reader disagrees with v2 validity/shape audit")
    return raw, rate, valid, continuity


def window_counters(result: Any, samples_per_window: int = 500) -> dict:
    shape = result.valid_mask.shape
    counts = {"channel_count": shape[0], "full_channel_windows": 0, "domain_accepted_channel_windows": 0,
              "actual_replacement_channel_windows": 0, "changed_channel_windows": 0,
              "tail_samples_per_channel": shape[1] % samples_per_window}
    for start in range(0, shape[1] - samples_per_window + 1, samples_per_window):
        section = slice(start, start + samples_per_window)
        valid = result.valid_mask[:, section]
        counts["full_channel_windows"] += shape[0]
        counts["domain_accepted_channel_windows"] += int(np.count_nonzero(np.any(valid & ~result.domain_rejected_mask[:, section], axis=1)))
        counts["actual_replacement_channel_windows"] += int(np.count_nonzero(np.any(result.replacement_mask[:, section], axis=1)))
    return counts


def write_domain_csv(path: Path, result: Any, rate: float) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "time_s", "channel", "support_probability", "domain_rejected", "actual_replacement"])
        for sample in range(result.valid_mask.shape[1]):
            for channel in range(result.valid_mask.shape[0]):
                writer.writerow([sample, sample / rate, channel + 1,
                                 result.support_probability[channel, sample],
                                 bool(result.domain_rejected_mask[channel, sample]),
                                 bool(result.replacement_mask[channel, sample])])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = args.root.resolve()
        if args.session_dir and (args.session or args.input):
            raise EvaluationError("choose exactly one explicit source")
        selected = args.input or args.session or args.session_dir
        if selected is None:
            raise EvaluationError("explicit --session or --input is required; no automatic processing")
        source = resolve(root, selected)
        is_npz = args.input is not None
        if (is_npz and (source.suffix.lower() != ".npz" or not source.is_file())) or (not is_npz and not source.is_dir()):
            raise EvaluationError("explicit source missing or wrong type")
        checkpoint, denoiser, config = (resolve(root, path) for path in (args.checkpoint, args.denoiser_checkpoint, args.config))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = resolve(root, args.output) if args.output else root / "analysis_outputs" / "offline_diagnostics" / "learned_denoise_v3" / f"{source.stem}_{stamp}_{uuid.uuid4().hex[:8]}"
        readers._ensure_output_separate(output, source, (checkpoint, denoiser, config))
        if output.is_symlink():
            raise EvaluationError("output symlink already exists")
        before = readers._snapshot(source)
        raw, rate, valid, continuity = read_source(source, is_npz)
        protected = {str(path): sha256(path) for path in (checkpoint, denoiser, config)}
        core, bundle = load_bundle(checkpoint, denoiser, config, args.device, args.threads)
        result = core.apply_model_to_array(bundle, raw, valid_mask=valid)
        validate_result(result, raw, valid, bundle.support_threshold)
        if readers._snapshot(source) != before or any(sha256(Path(p)) != value for p, value in protected.items()):
            raise EvaluationError("immutable input/model/config changed during inference")
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output / "denoised_signal.npz", raw=raw,
                            **{field: getattr(result, field) for field in RESULT_FIELDS},
                            sample_rate_hz=np.asarray(rate), diagnostic_only=np.asarray(True),
                            automatic=np.asarray(False), release=np.asarray(False))
        readers._write_csv(output / "denoised_signal.csv", raw, result, rate)
        write_domain_csv(output / "domain_scores.csv", result, rate)
        counters = window_counters(result)
        counters["changed_channel_windows"] = sum(int(np.count_nonzero(np.any(
            result.valid_mask[:, start:start + 500] & (result.denoised[:, start:start + 500] != raw[:, start:start + 500]), axis=1)))
            for start in range(0, raw.shape[1] - 500 + 1, 500))
        report = {
            "schema_version": 3, "status": "completed", **SAFETY_METADATA,
            "model_version": core.MODEL_VERSION, "source": str(source), "source_sha256": before,
            "source_hashes_unchanged": True, "checkpoint_path": str(checkpoint), "denoiser_checkpoint_path": str(denoiser),
            "config_path": str(config), "protected_sha256": protected, "support_threshold": bundle.support_threshold,
            "threshold_changed": False, "threshold_comparison": ">", "device": args.device, "threads": args.threads,
            "support_score_interpretation": "uncalibrated applicability score; NOT artifact probability or repair quality",
            "sample_rate_hz": rate, "channel_count": raw.shape[0], "sample_count": raw.shape[1],
            "counters": counters, "replacement_sample_fraction": float(np.mean(result.replacement_mask)),
            "data_integrity_assessment": continuity, "diagnostics": result.diagnostics,
            "runtime": bundle.config["runtime"], "output_dir": str(output),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        generated = ["denoised_signal.npz", "denoised_signal.csv", "domain_scores.csv", "quality_metrics.json", "model_manifest.json"]
        (output / "quality_metrics.json").write_text(_dump(report), encoding="utf-8")
        (output / "model_manifest.json").write_text(_dump({**report, "generated_files": generated}), encoding="utf-8")
        if readers._snapshot(source) != before:
            raise EvaluationError("source changed during export; outputs are not stable diagnostics")
        print(_dump(report))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError, AttributeError) as error:
        print(_dump({"status": "error", **SAFETY_METADATA, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
