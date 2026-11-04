"""Fixed-test, provenance-guarded semi-synthetic EEG v2 research evaluation.

No training, calibration, workstation deployment, or release approval occurs here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
from scipy import signal as scipy_signal

# Predeclared before independent test execution; never fitted on test scores.
RESEARCH_THRESHOLDS = {
    "all_test_gated_rmse_improvement_min": 0.10,
    "corrupted_test_gated_snr_gain_db_min": 1.0,
    "eog_gated_rmse_improvement_min": 0.0,
    "emg_gated_rmse_improvement_min": 0.0,
    "clean_replacement_window_fraction_max": 0.01,
    "clean_gated_change_rmse_ac_rms_units_max": 0.02,
    "corrupted_replacement_fraction_min": 0.10,
}
BASELINE_PARAMETERS = {
    "name": "fixed_bandpass_1_45_hz", "low_hz": 1.0, "high_hz": 45.0,
    "order": 4, "sample_rate_hz": 250.0, "implementation": "scipy.signal.sosfiltfilt",
    "axis": -1, "scale": "unchanged_dataset_signal_units", "test_tuned": False,
}
SPLITS = ("train", "validation", "test")
ARTIFACT_TYPES = ("clean", "eog", "emg")


class EvaluationError(ValueError):
    """Fail closed on missing or inconsistent benchmark evidence."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _dump(value: dict) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/processed/learned_denoiser_v2.npz"))
    parser.add_argument("--model", type=Path, default=Path("models/denoiser_v2.pt"))
    parser.add_argument("--config", type=Path, default=Path("configs/learned_denoiser_v2.json"))
    parser.add_argument("--output", type=Path, default=Path("analysis_outputs/learned_denoiser_v2_test.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2, help="positive optional PyTorch CPU thread count")
    parser.add_argument("--overwrite", action="store_true", help="explicitly authorize another fixed-test evaluation at this report path")
    parser.add_argument("--check-only", action="store_true", help="validate hashes/splits/checkpoint without model inference or writing a test report")
    return parser


def _finite_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _masked_rmse(values: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    return _finite_number(np.sqrt(np.mean(np.square(values[mask], dtype=np.float64))))


def _clean_ac_rms(clean: np.ndarray, mask: np.ndarray) -> float | None:
    counts = mask.sum(axis=-1, keepdims=True)
    means = np.divide(np.where(mask, clean, 0).sum(axis=-1, keepdims=True), counts,
                      out=np.zeros_like(counts, dtype=np.float64), where=counts > 0)
    return _masked_rmse(clean - means, mask)


def _metric_set(clean: np.ndarray, noisy: np.ndarray, prediction: np.ndarray,
                valid: np.ndarray, replacement: np.ndarray, artifact_mask: np.ndarray) -> dict:
    # A single common support is used for all methods; predictions must not drop
    # difficult valid samples to make metrics look better.
    if np.any(valid & ~np.isfinite(prediction)):
        raise EvaluationError("prediction is non-finite on valid benchmark samples")
    rmse = _masked_rmse(prediction - clean, valid)
    input_rmse = _masked_rmse(noisy - clean, valid)
    ac_rms = _clean_ac_rms(clean, valid)
    change = _masked_rmse(prediction - noisy, valid)
    correlation = None
    if np.count_nonzero(valid) >= 2:
        c = clean[valid].astype(np.float64)
        p = prediction[valid].astype(np.float64)
        c -= c.mean()
        p -= p.mean()
        denom = np.linalg.norm(c) * np.linalg.norm(p)
        if denom > 0:
            correlation = _finite_number(np.dot(c, p) / denom)
    snr_gain = None
    if input_rmse is not None and rmse is not None:
        if input_rmse == rmse == 0:
            snr_gain = 0.0
        elif input_rmse > 0 and rmse > 0:
            snr_gain = _finite_number(20 * np.log10(input_rmse / rmse))
    replacement = np.asarray(replacement, dtype=bool) & valid
    usable_windows = np.any(valid, axis=(1, 2))
    replaced_windows = np.any(replacement, axis=(1, 2))
    clean_samples = valid & ~artifact_mask
    return {
        "window_count": int(len(clean)), "valid_sample_count": int(np.count_nonzero(valid)),
        "RMSE": rmse, "relative_RMSE": rmse / ac_rms if rmse is not None and ac_rms else None,
        "correlation": correlation, "SNR_gain_db": snr_gain,
        "rmse_improvement_fraction": 1 - rmse / input_rmse if rmse is not None and input_rmse else None,
        "mean_replacement": float(replacement[valid].mean()) if np.any(valid) else None,
        "replacement_fraction": float(replacement[valid].mean()) if np.any(valid) else None,
        "replacement_window_fraction": float(replaced_windows[usable_windows].mean()) if np.any(usable_windows) else None,
        "change_RMSE": change,
        "change_RMSE_clean_ac_rms_units": change / ac_rms if change is not None and ac_rms else None,
        "clean_AC_RMS": ac_rms,
        "clean_false_replacement_fraction": float(replacement[clean_samples].mean()) if np.any(clean_samples) else None,
        "clean_signal_distortion_RMSE": _masked_rmse(prediction - clean, clean_samples),
        "segments": {
            "artifact": {"sample_count": int(np.count_nonzero(valid & artifact_mask)),
                         "RMSE": _masked_rmse(prediction - clean, valid & artifact_mask)},
            "nonartifact": {"sample_count": int(np.count_nonzero(clean_samples)),
                            "RMSE": _masked_rmse(prediction - clean, clean_samples)},
        },
    }


def fixed_bandpass(noisy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not np.all(valid):
        # Filter each contiguous valid run independently: do not interpolate gaps.
        filtered = noisy.astype(np.float64).copy()
        sos = scipy_signal.butter(4, [1.0, 45.0], btype="bandpass", fs=250.0, output="sos")
        for row, mask, target in zip(noisy.reshape(-1, 500), valid.reshape(-1, 500), filtered.reshape(-1, 500)):
            bounds = np.diff(np.r_[False, mask, False].astype(np.int8))
            for start, end in zip(np.flatnonzero(bounds == 1), np.flatnonzero(bounds == -1)):
                if end - start > 27:
                    target[start:end] = scipy_signal.sosfiltfilt(sos, row[start:end])
        return filtered
    sos = scipy_signal.butter(4, [1.0, 45.0], btype="bandpass", fs=250.0, output="sos")
    return scipy_signal.sosfiltfilt(sos, noisy, axis=-1)


def summarize_methods(clean: np.ndarray, noisy: np.ndarray, valid_mask: np.ndarray,
                      artifact_mask: np.ndarray, artifact_type: np.ndarray,
                      candidate: np.ndarray, gated: np.ndarray, replacement_mask: np.ndarray) -> dict:
    valid = np.asarray(valid_mask, dtype=bool)
    if any(np.asarray(x).shape != clean.shape for x in (noisy, valid, artifact_mask, candidate, gated, replacement_mask)):
        raise EvaluationError("metric tensors must share shape [N,1,500]")
    if clean.ndim != 3 or clean.shape[1:] != (1, 500) or len(artifact_type) != len(clean):
        raise EvaluationError("metric inputs require [N,1,500] and N artifact labels")
    if np.any(valid & (~np.isfinite(clean) | ~np.isfinite(noisy))):
        raise EvaluationError("valid benchmark samples must be finite")
    baseline = fixed_bandpass(noisy, valid)
    methods = {
        "candidate": (candidate, valid & (candidate != noisy)),
        "gated": (gated, replacement_mask),
        "noop": (noisy, np.zeros_like(valid)),
        "bandpass": (baseline, valid & (baseline != noisy)),
    }
    groups = {"all": np.ones(len(clean), dtype=bool),
              "corrupted": np.isin(artifact_type, ["eog", "emg"]),
              **{name: artifact_type == name for name in ARTIFACT_TYPES}}
    return {name: {group: _metric_set(clean[index], noisy[index], prediction[index], valid[index],
                                    replaced[index], artifact_mask[index])
                   for group, index in groups.items()}
            for name, (prediction, replaced) in methods.items()}


def research_gate(metrics: dict) -> dict:
    gated = metrics["gated"]
    specifications = [
        ("all_test_rmse", "all", "rmse_improvement_fraction", "min", "all_test_gated_rmse_improvement_min"),
        ("corrupted_snr", "corrupted", "SNR_gain_db", "min", "corrupted_test_gated_snr_gain_db_min"),
        ("eog_not_worse", "eog", "rmse_improvement_fraction", "min", "eog_gated_rmse_improvement_min"),
        ("emg_not_worse", "emg", "rmse_improvement_fraction", "min", "emg_gated_rmse_improvement_min"),
        ("clean_replacement_windows", "clean", "replacement_window_fraction", "max", "clean_replacement_window_fraction_max"),
        ("clean_distortion", "clean", "change_RMSE_clean_ac_rms_units", "max", "clean_gated_change_rmse_ac_rms_units_max"),
        ("non_noop_corrupted", "corrupted", "replacement_fraction", "min", "corrupted_replacement_fraction_min"),
    ]
    checks = {}
    for name, group, metric, direction, threshold_name in specifications:
        observed = gated[group][metric]
        threshold = RESEARCH_THRESHOLDS[threshold_name]
        passed = observed is not None and np.isfinite(observed) and gated[group]["valid_sample_count"] > 0
        passed = bool(passed and (observed >= threshold if direction == "min" else observed <= threshold))
        checks[name] = {"passed": passed, "observed": observed, "threshold": threshold, "direction": direction}
    return {
        "semi_synthetic_research_gate_passed": all(item["passed"] for item in checks.values()),
        "release_gate_passed": False, "production_validated": False,
        "automatic_processing_enabled": False, "scope": "semi_synthetic_epoch_holdout_only",
        "thresholds_predeclared": dict(RESEARCH_THRESHOLDS), "checks": checks,
        "baseline_win_required": False,
    }


def _reserve_report(path: Path, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_WRONLY | (os.O_TRUNC if overwrite else os.O_EXCL)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise EvaluationError(f"report already exists: {path}; use --check-only for diagnostics, or explicit --overwrite") from error
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(_dump({"status": "reserved_no_test_result_yet", "release_gate_passed": False,
                            "repeat_test_authorized": overwrite}))


REQUIRED_ARRAYS = ("clean", "noisy", "valid_mask", "artifact_mask", "split", "artifact_type",
                   "clean_epoch_id", "eog_epoch_id", "emg_epoch_id")
SOURCE_ID_KEYS = {"eeg": "clean_epoch_id", "eog": "eog_epoch_id", "emg": "emg_epoch_id"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def _index_list(values: Any, size: int, name: str) -> np.ndarray:
    _require(isinstance(values, list) and bool(values), f"{name} must be a nonempty plain integer list")
    _require(all(type(i) is int and 0 <= i < size for i in values), f"{name} contains invalid indices")
    _require(len(set(values)) == len(values), f"{name} contains duplicate indices")
    return np.asarray(values, dtype=np.int64)


def _validate_arrays(data: dict[str, np.ndarray], manifest: dict) -> None:
    _require(all(key in data for key in REQUIRED_ARRAYS), "NPZ is missing required v2 arrays")
    clean = data["clean"]
    _require(clean.ndim == 3 and clean.shape[1:] == (1, 500) and len(clean) > 0,
             "clean must have shape [N,1,500]")
    n = len(clean)
    for name in ("clean", "noisy", "valid_mask", "artifact_mask"):
        value = data[name]
        _require(value.shape == clean.shape, f"{name} has wrong shape")
        _require(value.dtype.kind in ("f", "i", "u") if name in ("clean", "noisy") else value.dtype.kind == "b",
                 f"{name} has wrong dtype")
    for name, allowed in (("split", SPLITS), ("artifact_type", ARTIFACT_TYPES)):
        _require(data[name].shape == (n,) and data[name].dtype.kind == "U", f"{name} must be an N-element unicode vector")
        _require(np.all(np.isin(data[name], allowed)), f"{name} contains unknown labels")
    for name in SOURCE_ID_KEYS.values():
        _require(data[name].shape == (n,) and data[name].dtype.kind in ("i", "u"), f"{name} must be an integer N-vector")
    valid = data["valid_mask"]
    _require(np.all(np.any(valid, axis=(1, 2))), "every benchmark window needs valid samples")
    _require(np.all(np.isfinite(clean[valid])) and np.all(np.isfinite(data["noisy"][valid])),
             "benchmark contains non-finite valid samples")
    _require(np.all(data["clean_epoch_id"] >= 0), "clean source epoch must be present for every sample")
    identity = data["artifact_type"] == "clean"
    _require(np.array_equal(data["clean"][identity], data["noisy"][identity]), "clean identity noisy must equal clean")
    _require(not np.any(data["artifact_mask"][identity]), "identity windows cannot have artifact_mask")
    for kind in ("eog", "emg"):
        used = data["artifact_type"] == kind
        ids = data[f"{kind}_epoch_id"]
        _require(np.all(ids[used] >= 0) and np.all(ids[~used] == -1), f"{kind}_epoch_id active/unused (-1) contract violated")
        _require(np.all(np.any(data["artifact_mask"][used], axis=(1, 2))), f"{kind} window lacks artifact mask")
    _require(manifest.get("schema_version") == 2 and manifest.get("dataset_version") == "learned_denoiser_v2",
             "unsupported dataset manifest schema/version")
    _require(manifest.get("requested_samples") == n, "manifest sample count disagrees with NPZ")
    preprocessing = manifest.get("preprocessing", {})
    _require(preprocessing.get("target_rate_hz") == 250 and preprocessing.get("target_samples") == 500,
             "manifest must establish 250 Hz / 500 samples")
    # Dataset scale is one unit of clean AC RMS. Do NOT normalize each method
    # separately, which could conceal a reconstruction scale bug.
    for row, mask in zip(clean, valid):
        _require(np.isclose(_clean_ac_rms(row[None], mask[None]), 1.0, rtol=2e-5, atol=2e-5),
                 "clean AC RMS must be one dataset unit")
    contract = manifest.get("npz_contract")
    _require(isinstance(contract, dict) and set(contract) == set(data), "manifest NPZ keys disagree")
    for name, value in data.items():
        _require(contract[name] == {"shape": list(value.shape), "dtype": str(value.dtype)},
                 f"manifest NPZ contract disagrees for {name}")
    for split in SPLITS:
        labels = data["artifact_type"][data["split"] == split]
        count = int(len(labels))
        _require(count > 0, f"{split} split is empty")
        _require(manifest.get("split_sample_targets", {}).get(split) == count, f"manifest {split} target count disagrees")
        stats = manifest.get("split_statistics", {}).get(split, {})
        expected = {"samples": count, "identity_samples": int(np.count_nonzero(labels == "clean")),
                    "eog_samples": int(np.count_nonzero(labels == "eog")),
                    "emg_samples": int(np.count_nonzero(labels == "emg"))}
        _require(all(stats.get(k) == v for k, v in expected.items()), f"manifest {split} statistics disagree")


def _validate_source_partition(data: dict[str, np.ndarray], manifest: dict) -> dict:
    # Reuse the actual training-side metadata-only guard. It neither trains nor
    # evaluates test signal content; this prevents a second guessed schema.
    from eeg_tools.learned_denoise_v2 import LearnedDenoiseV2Error
    from scripts.train_denoiser_v2 import validate_dataset_manifest
    try:
        validate_dataset_manifest(manifest, manifest["output"]["sha256"], data["split"],
                        {kind: data[key] for kind, key in SOURCE_ID_KEYS.items()})
    except LearnedDenoiseV2Error as error:
        raise EvaluationError(f"source epoch/fingerprint IDs not isolated: {error}") from error
    audit = manifest["epoch_split_manifest"]
    return {"source_epoch_ids_isolated": True, "global_fingerprints_isolated": True,
            "raw_epoch_pool_counts": {kind: audit["source_statistics"][kind]["raw_epochs"] for kind in SOURCE_ID_KEYS},
            "fingerprint_groups": len(audit["fingerprint_groups"]), "subject_disjoint_claim": False}


def _select_evaluation_indices(checkpoint: dict, data: dict[str, np.ndarray], manifest: dict,
                               *, dataset_sha256: str, manifest_sha256: str) -> tuple[np.ndarray, dict]:
    from eeg_tools.learned_denoise_v2 import LearnedDenoiseV2Error, validate_checkpoint_metadata
    try:
        checkpoint = validate_checkpoint_metadata(checkpoint, checkpoint.get("config"))
    except LearnedDenoiseV2Error as error:
        raise EvaluationError(f"checkpoint metadata guard: {error}") from error
    _require(checkpoint.get("dataset_sha256") == dataset_sha256 == manifest.get("output", {}).get("sha256"),
             "NPZ SHA256 does not match checkpoint and manifest")
    _require(checkpoint.get("dataset_manifest_sha256") == manifest_sha256,
             "manifest SHA256 does not match checkpoint")
    _require(checkpoint.get("dataset_manifest") == manifest, "checkpoint embedded manifest differs from sidecar")
    _validate_arrays(data, manifest)
    provenance = _validate_source_partition(data, manifest)
    n = len(data["split"])
    partitions = {split: np.flatnonzero(data["split"] == split) for split in SPLITS}
    for split in ("train", "validation"):
        indices = _index_list(checkpoint.get(f"{split}_indices"), n, f"checkpoint {split}_indices")
        _require(np.array_equal(np.sort(indices), partitions[split]), f"checkpoint {split}_indices disagree with the full NPZ split")
    if "test_indices" in checkpoint:
        indices = _index_list(checkpoint["test_indices"], n, "checkpoint test_indices")
        _require(np.array_equal(np.sort(indices), partitions["test"]), "checkpoint test_indices disagree")
    if "selected_train_indices" in checkpoint:
        selected = _index_list(checkpoint["selected_train_indices"], n, "selected_train_indices")
        _require(set(selected.tolist()) <= set(partitions["train"].tolist()), "selected training indices include held-out windows")
    threshold = checkpoint.get("calibrated_threshold")
    _require(type(threshold) in (int, float) and np.isfinite(threshold) and 0 <= threshold <= 1,
             "checkpoint calibrated_threshold must be finite and in [0,1]")
    _require(checkpoint.get("validation_threshold") == threshold, "threshold alias disagrees with calibrated_threshold")
    calibration = checkpoint.get("threshold_calibration", {})
    _require(isinstance(calibration, dict) and calibration.get("scope") == "validation_only"
             and calibration.get("comparison") == ">"
             and calibration.get("method") == "clean_score_order_statistic"
             and calibration.get("score_reduction") == "per_channel_window_mean"
             and calibration.get("max_clean_false_replacement_fraction") == 0.01
             and calibration.get("empirical_only") is True
             and calibration.get("production_calibrated") is False,
             "calibration must be the fixed validation-only clean-window order statistic")
    clean_validation = int(np.count_nonzero((data["split"] == "validation") & (data["artifact_type"] == "clean")))
    _require(clean_validation > 0 and calibration.get("clean_window_count") == clean_validation,
             "calibration clean-window count differs from fixed validation")
    observed = calibration.get("observed_clean_false_replacement_fraction")
    _require(type(observed) in (int, float) and np.isfinite(observed) and 0 <= observed <= 0.01,
             "calibration clean replacement evidence missing or exceeds the predeclared maximum")
    for kind in ARTIFACT_TYPES:
        _require(np.any((data["split"] == "test") & (data["artifact_type"] == kind)),
                 f"fixed test needs nonempty {kind} windows for predeclared gates")
    return partitions["test"], {
        **provenance, "dataset_sha256_verified": True, "dataset_manifest_sha256_verified": True,
        "checkpoint_split_indices_verified": True, "validation_used_as_test": False,
        "test_pre_fixed": True, "split_counts": {k: len(v) for k, v in partitions.items()},
        "evaluation_scope": "fixed_test_only_epoch_disjoint_semi_synthetic",
    }


def _load_dataset(path: Path) -> tuple[dict, dict, str, str]:
    manifest_path = Path(str(path) + ".manifest.json")
    digest = sha256(path)
    manifest_digest = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(isinstance(manifest, dict), "manifest must be a JSON object")
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    _require(sha256(path) == digest and sha256(manifest_path) == manifest_digest, "dataset/manifest changed during loading")
    return data, manifest, digest, manifest_digest


def _infer_fixed_test(bundle: Any, noisy: np.ndarray, valid: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    from eeg_tools.learned_denoise_v2 import apply_model_to_array
    from scripts.run_learned_denoise_v2 import _validate_result
    _require(type(batch_size) is int and batch_size > 0, "batch-size must be positive")
    candidate, gated = noisy.astype(np.float64).copy(), noisy.astype(np.float64).copy()
    replacement = np.zeros_like(valid, dtype=bool)
    batches = []
    rejected_samples = 0
    for start in range(0, len(noisy), batch_size):
        end = min(start + batch_size, len(noisy))
        x, mask = noisy[start:end, 0, :], valid[start:end, 0, :]
        # Independent single-channel model: C windows share weights, not statistics.
        result = apply_model_to_array(bundle, x, valid_mask=mask)
        _validate_result(result, x, mask)
        _require(np.array_equal(result.valid_mask, mask), "model may not exclude valid test samples from evaluation")
        candidate[start:end, 0, :] = result.model_denoised
        gated[start:end, 0, :] = result.denoised
        replacement[start:end, 0, :] = result.replacement_mask
        rejected_samples += int(np.count_nonzero(result.rejected_mask & mask))
        batches.append(result.diagnostics)
    return candidate, gated, replacement, {"batch_size": batch_size, "batch_count": len(batches),
                                          "rejected_valid_sample_count": rejected_samples, "batches": batches}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reserved = False
    try:
        _require(args.batch_size > 0 and args.threads > 0, "batch-size and threads must be positive")
        # Never allow --overwrite to destroy immutable inputs or a checkpoint.
        protected = (args.dataset, Path(str(args.dataset) + ".manifest.json"), args.model, args.config)
        _require(args.output.resolve() not in {p.resolve() for p in protected}, "report output is a protected input path")
        _require("v1" not in args.output.stem.casefold(), "v2 evaluation must not overwrite a v1 report")
        if not args.check_only and args.output.exists() and not args.overwrite:
            raise EvaluationError(f"report already exists: {args.output}; use --check-only (no test inference) or explicit --overwrite")
        from eeg_tools.learned_denoise_v2 import MODEL_VERSION, load_config, load_model, torch
        if torch is not None:
            torch.set_num_threads(args.threads)
        config = load_config(args.config)
        _require(config["sample_rate_hz"] == 250 and config["window_samples"] == 500 and config["hop_samples"] == 500,
                 "evaluation is fixed to 250 Hz / 500-sample non-overlapping windows")
        model_digest = sha256(args.model)
        config_digest = sha256(args.config)
        bundle = load_model(args.model, config, device=args.device)  # strict weights_only lives in core
        data, manifest, digest, manifest_digest = _load_dataset(args.dataset)
        indices, audit = _select_evaluation_indices(bundle.metadata, data, manifest,
                                                    dataset_sha256=digest, manifest_sha256=manifest_digest)
        _require(bundle.calibrated_threshold == bundle.metadata["calibrated_threshold"], "bundle threshold disagrees with checkpoint")
        common = {
            "schema_version": 2, "model_version": MODEL_VERSION, "threads": args.threads, **audit,
            "diagnostic_only": True, "automatic_processing_enabled": False,
            "automatic": False, "release_gate_passed": False, "release": False, "production_validated": False,
            "dataset_path": str(args.dataset.resolve()), "dataset_sha256": digest,
            "dataset_manifest_sha256": manifest_digest, "model_sha256": model_digest, "config_sha256": config_digest,
            "test_indices": indices.tolist(), "test_window_count": len(indices),
            "calibrated_threshold": bundle.calibrated_threshold,
            "threshold_comparison": ">", "threshold_calibration": bundle.metadata["threshold_calibration"],
            "research_thresholds_predeclared": dict(RESEARCH_THRESHOLDS), "baseline_parameters": dict(BASELINE_PARAMETERS),
            "architecture": bundle.metadata["architecture"], "sources": manifest.get("sources"),
            "metric_semantics": {"support": "same fixed dataset valid_mask for all methods",
                                 "RMSE": "pooled paired sample RMSE in unchanged dataset units",
                                 "relative_RMSE": "RMSE / pooled clean AC RMS",
                                 "correlation": "pooled Pearson correlation on common valid samples",
                                 "SNR_gain_db": "20 log10(noop RMSE / method RMSE); degenerate infinities are null",
                                 "mean_replacement": "fraction of valid samples selected (gated) or changed (other methods)",
                                 "clean_gate_units": "change RMSE / clean AC RMS; clean AC RMS=1 dataset unit",
                                 "confidence": "1-probability, NOT calibrated success probability"},
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        if args.check_only:
            print(_dump({**common, "status": "provenance_checked_no_test_inference",
                         "test_inference_executed": False, "existing_report": args.output.exists()}))
            return 0
        _reserve_report(args.output, args.overwrite)
        reserved = True
        clean, noisy, valid, artifact = (data[name][indices] for name in ("clean", "noisy", "valid_mask", "artifact_mask"))
        candidate, gated, replacement, diagnostics = _infer_fixed_test(bundle, noisy, valid, args.batch_size)
        metrics = summarize_methods(clean, noisy, valid, artifact, data["artifact_type"][indices], candidate, gated, replacement)
        gate = research_gate(metrics)
        _require(all(sha256(path) == expected for path, expected in ((args.dataset, digest),
                     (Path(str(args.dataset) + ".manifest.json"), manifest_digest), (args.model, model_digest),
                     (args.config, config_digest))), "protected inputs changed during evaluation")
        report = {**common, "status": "completed", "test_inference_executed": True,
                  "repeat_test_authorized": args.overwrite, "metrics": metrics, "gate": gate,
                  "semi_synthetic_research_gate_passed": gate["semi_synthetic_research_gate_passed"],
                  "inference_diagnostics": diagnostics}
        args.output.write_text(_dump(report), encoding="utf-8")
        print(_dump({"status": "completed", "output": str(args.output.resolve()),
                     "test_window_count": len(indices), **gate}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError, AttributeError) as error:
        failure = {"status": "error", "error": str(error), "release_gate_passed": False,
                   "automatic_processing_enabled": False, "research_thresholds_predeclared": dict(RESEARCH_THRESHOLDS)}
        if reserved:
            failure["test_attempt_reserved"] = True
            failure["warning"] = "A test attempt may have begun; preserved report blocks silent repeats. Inspect before explicit --overwrite."
            try:
                args.output.write_text(_dump(failure), encoding="utf-8")
            except OSError:
                pass
        print(_dump(failure))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
