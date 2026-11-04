"""Explicit offline v4 diagnostics for [T,C] NPY/NPZ/numeric CSV.
No hardware/UI/session integration, raw overwrite, training or automatic repair.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.evaluate_denoiser_v4 import (
    ARRAY_FIELDS, V4EvaluationError, infer_windows, jsonable, load_bundle,
    positive_int, read_json, resolve, sha256, validate_interface, write_new_report,
)


def signal_array(value) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype.kind not in "iuf" or value.ndim not in (1, 2) or not all(value.shape):
        raise V4EvaluationError("signal must be nonempty real [T,C] or [T]; no implicit transpose")
    return value.astype(np.float64).reshape(-1, 1) if value.ndim == 1 else value.astype(np.float64)


def binary_mask(value, shape) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim == 1 and shape[1] == 1:
        value = value[:, None]
    if value.shape != shape or value.dtype.kind not in "biuf" or not np.all(np.isin(value, [0, 1])):
        raise V4EvaluationError("valid mask must match [T,C] exactly and contain only0/1; no broadcast")
    return value.astype(bool)


def load_input(path: Path, sample_rate: float, *, signal_key="signal", mask_path=None, all_finite_valid=False):
    if type(sample_rate) not in (int, float) or not np.isfinite(sample_rate) or sample_rate != 250:
        raise V4EvaluationError("explicit sample_rate must equal250Hz; no resampling")
    embedded_mask = None
    if path.suffix.lower() == ".npy":
        raw = signal_array(np.load(path, allow_pickle=False))
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if signal_key not in archive:
                raise V4EvaluationError(f"NPZ missing explicit signal key: {signal_key}")
            raw = signal_array(archive[signal_key])
            if "sample_rate_hz" in archive:
                rate = archive["sample_rate_hz"]
                if rate.shape != () or rate.dtype.kind not in "iuf" or float(rate) != sample_rate:
                    raise V4EvaluationError("NPZ sample_rate_hz must be scalar matching explicit250Hz")
            if "valid_mask" in archive:
                embedded_mask = binary_mask(archive["valid_mask"], raw.shape)
    elif path.suffix.lower() == ".csv":
        # No header, timestamps, delimiter guessing or channel-order inference.
        raw = signal_array(np.loadtxt(path, delimiter=",", ndmin=2))
    else:
        raise V4EvaluationError("only .npy/.npz/numeric headerless .csv are supported")
    if mask_path is not None and (embedded_mask is not None or all_finite_valid):
        raise V4EvaluationError("choose only one mask source; cannot override embedded mask")
    if all_finite_valid and embedded_mask is not None:
        raise V4EvaluationError("cannot override embedded valid_mask with all-finite declaration")
    if mask_path is not None:
        if mask_path.suffix.lower() != ".npy":
            raise V4EvaluationError("external valid_mask must be .npy")
        mask = binary_mask(np.load(mask_path, allow_pickle=False), raw.shape)
        policy = "explicit_external_mask"
    elif embedded_mask is not None:
        mask, policy = embedded_mask, "explicit_NPZ_valid_mask"
    elif all_finite_valid:
        if not np.all(np.isfinite(raw)):
            raise V4EvaluationError("all-finite-valid declaration incompatible with nonfinite source")
        mask, policy = np.ones(raw.shape, dtype=bool), "operator_declared_all_finite_valid"
    else:
        raise V4EvaluationError("explicit valid_mask or --all-finite-valid required")
    return raw, mask, {"sample_rate_hz": sample_rate, "layout": "T,C_no_transpose", "mask_policy": policy,
                       "original_units": "stored_units_not_recalibrated"}


def process_array(bundle, signal: np.ndarray, valid_mask: np.ndarray, *, chunk_size=128, apply=None) -> dict:
    signal = signal_array(signal)
    valid_mask = binary_mask(valid_mask, signal.shape)
    length, channels = signal.shape
    complete = length // 500 * 500
    result = {"denoised": signal.copy(), "model_denoised": signal.copy(),
              "artifact_probability": np.full(signal.shape, np.nan), "confidence": np.zeros(signal.shape),
              "replacement_mask": np.zeros(signal.shape, dtype=bool),
              "valid_mask": valid_mask & np.isfinite(signal), "rejected_mask": np.ones(signal.shape, dtype=bool)}
    chunks = []
    if complete:
        def flatten(values):
            return values[:complete].reshape(-1, 500, channels).transpose(2, 0, 1).reshape(-1, 500)
        computed = infer_windows(bundle, flatten(signal), valid_mask=flatten(valid_mask), chunk_size=chunk_size, apply=apply)
        for name in ARRAY_FIELDS:
            result[name][:complete] = computed[name].reshape(channels, -1, 500).transpose(1, 2, 0).reshape(complete, channels)
        chunks = computed["diagnostics"]
    if not np.array_equal(result["denoised"][complete:], signal[complete:], equal_nan=True):
        raise V4EvaluationError("incomplete tail changed")
    changed = ~((result["denoised"] == signal) | (np.isnan(result["denoised"]) & np.isnan(signal)))
    result["diagnostics"] = {"channel_count": channels, "time_samples": length,
                             "full_channel_window_count": complete // 500 * channels,
                             "tail_samples_per_channel_unchanged": length - complete,
                             "replacement_sample_fraction": float(result["replacement_mask"].mean()),
                             "actual_change_sample_fraction": float(changed.mean()),
                             "inference_chunks": chunks, "probability_calibrated": False,
                             "efficacy_assessed": False, "automatic_processing_enabled": False}
    return result


def validate_output(output: Path, root: Path, protected: list[Path]) -> None:
    if output.exists() or output.is_symlink():
        raise V4EvaluationError("output directory already exists; overwrite refused")
    if any(output == p or p.is_relative_to(output) for p in protected):
        raise V4EvaluationError("output directory contains protected source/model/config")
    if any(output.is_relative_to(root / d) for d in ("datasets", "models", "configs", "eeg_tools", "scripts", "tests")):
        raise V4EvaluationError("output cannot be inside raw/data/source directories")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--signal-key", default="signal")
    parser.add_argument("--sample-rate", "--sample-rate-hz", dest="sample_rate", type=float)
    masks = parser.add_mutually_exclusive_group()
    masks.add_argument("--valid-mask", type=Path)
    masks.add_argument("--all-finite-valid", action="store_true")
    parser.add_argument("--output", type=Path, help="new independent directory")
    parser.add_argument("--model", "--checkpoint", dest="model", type=Path, default=Path("models/denoiser_v4.pt"))
    parser.add_argument("--config", type=Path, default=Path("configs/learned_denoiser_v4.json"))
    parser.add_argument("--interface", type=Path, default=Path("analysis_outputs/learned_denoiser_v4_interface_contract.json"))
    parser.add_argument("--device", choices=["cpu", "cuda", "cuda:0"], default="cuda")
    parser.add_argument("--threads", type=positive_int, default=2)
    parser.add_argument("--chunk-size", type=positive_int, default=128)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--check-report", type=Path, help="optional exclusive preflight JSON")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    paths = {key: resolve(root, getattr(args, key)) for key in ("model", "config", "interface")}
    report = {"schema_version": 1, "status": "formal", "check_only": args.check_only,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "diagnostic_ready": False,
              "training_executed": False, "test_evaluation_executed": False,
              "inference_executed": False, "automatic_processing_enabled": False, "production_validated": False,
              "hardware_opened": False, "gui_started": False, "model_efficacy": "not_assessed",
              "python_executable": sys.executable, "requested_device": args.device, "threads": args.threads}
    output = None
    output_owned = False
    start = time.perf_counter()
    try:
        if args.check_report and not args.check_only:
            raise V4EvaluationError("--check-report is only for --check-only")
        check_path = resolve(root, args.check_report) if args.check_report else None
        if check_path and (check_path.exists() or any(check_path.is_relative_to(root / d) for d in ("datasets", "models", "configs", "scripts", "eeg_tools", "tests"))):
            raise V4EvaluationError("check-report exists or is in protected directories")
        if args.input:
            source = resolve(root, args.input)
            if not source.is_file() or source.suffix.lower() not in (".npz", ".npy", ".csv") or args.sample_rate != 250:
                raise V4EvaluationError("explicit source file and --sample-rate250 required")
            paths["input"] = source
            if args.valid_mask:
                paths["valid_mask"] = resolve(root, args.valid_mask)
                if not paths["valid_mask"].is_file() or paths["valid_mask"].suffix.lower() != ".npy":
                    raise V4EvaluationError("external valid_mask NPY file missing/wrong type")
            if args.output:
                output = resolve(root, args.output)
                validate_output(output, root, list(paths.values()))
            elif not args.check_only:
                raise V4EvaluationError("inference requires a new --output directory")
        elif not args.check_only or args.output or args.valid_mask or args.all_finite_valid or args.sample_rate is not None:
            raise V4EvaluationError("explicit --input required for inference/source/output options")
        # Snapshot exact code/config/input/model bytes before loading. Check-only
        # need not parse any source values, nor call inference.
        provenance = {str(p): sha256(p) for p in paths.values() if p.is_file()}
        for rel in ("scripts/run_learned_denoise_v4.py", "scripts/evaluate_denoiser_v4.py",
                    "scripts/verify_denoiser_v4_artifacts.py", "eeg_tools/learned_denoise_v4.py"):
            p = root / rel
            provenance[str(p)] = sha256(p)
        report["input_and_source_sha256"] = provenance
        cfg = read_json(paths["config"])
        validate_interface(read_json(paths["interface"]), cfg)
        if not paths["model"].is_file():
            report["model_load_status"] = "missing_checkpoint_training_not_confirmed"
            raise V4EvaluationError(f"checkpoint missing: {paths['model']}; NOT successful training")
        core, bundle = load_bundle(paths["model"], paths["config"], args.device, args.threads)
        report.update(diagnostic_ready=True, model_load_status="weights_only_validated", actual_device=bundle.device,
                      artifact_threshold=bundle.artifact_threshold, torch_version=str(core.torch.__version__))
        if not args.check_only:
            raw, mask, input_metadata = load_input(paths["input"], args.sample_rate, signal_key=args.signal_key,
                                    mask_path=paths.get("valid_mask"), all_finite_valid=args.all_finite_valid)
            # Reserve directory before inference; keep failure report in a new
            # directory rather than silently deleting or reusing failed runs.
            output.mkdir(parents=True, exist_ok=False)
            output_owned = True
            write_new_report(output / "started.json", report)
            result = process_array(bundle, raw, mask, chunk_size=args.chunk_size, apply=core.apply_model_to_array)
            report.update(inference_executed=True, input_metadata=input_metadata, diagnostics=result["diagnostics"])
            for path, digest in provenance.items():
                if sha256(Path(path)) != digest:
                    raise V4EvaluationError(f"source/checkpoint/config changed during inference: {path}")
            with (output / "diagnostics.npz").open("xb") as handle:
                np.savez_compressed(handle, **{k: result[k] for k in ARRAY_FIELDS}, sample_rate_hz=np.array(250))
            report["source_sha256_verified_after_inference"] = True
            report["output_directory"] = str(output)
        report["result"] = "diagnostic_ready_not_production_approval" if args.check_only else "offline_inference_only_not_efficacy_evaluation"
        return 0
    except Exception as error:
        report.update(diagnostic_ready=False, result="failed", error={"type": type(error).__name__, "message": str(error)})
        return 1
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        if output_owned:
            write_new_report(output / "report.json", report)
        if args.check_only and args.check_report:
            target = resolve(root, args.check_report)
            if not target.exists() and not any(target.is_relative_to(root / d) for d in ("datasets", "models", "configs", "scripts", "eeg_tools", "tests")):
                write_new_report(target, report)
        print(json.dumps(jsonable(report), ensure_ascii=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    raise SystemExit(main())
