"""Frozen v4 offline research evaluation; never train, tune or overwrite reports.
Only full evaluation reads test signals/GT; check-only verifies hashes/contract.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.verify_denoiser_v4_artifacts import BASE_FILES, EVALUATION_FILES, checked_path, verify_files, verify_lock

MODEL_VERSION = "learned_denoise_v4"
WINDOW = 500
DONG_SUBJECTS = ("S10.mat", "S11.mat")
DONG_DIR = "datasets/external/ssvep_datasets/dong2023_ssvep"
DONG_METADATA = "datasets/external/v2_metadata_audit/DONG2023_METADATA.json"
V1_SESSION = "session_20260924_131134_807"
ARRAY_FIELDS = ("denoised", "model_denoised", "artifact_probability", "confidence",
                "replacement_mask", "valid_mask", "rejected_mask")
LIMIT_KEYS = ("corrupted_rmse_improvement_min", "corrupted_snr_gain_db_min",
              "corrupted_actual_change_fraction_min", "clean_replacement_fraction_max",
              "clean_normalized_change_rmse_max", "protected_band_relative_discrepancy_max",
              "known_session_rmse_ratio_to_noisy_max", "new_dong_actual_change_fraction_max")


class V4EvaluationError(ValueError):
    """Ambiguous provenance, unsupported input or unsafe evaluation."""


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def read_json(path: Path, name: str = "JSON") -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise V4EvaluationError(f"{name} must be a JSON object")
    return value


def resolve(root: Path, path: str | Path) -> Path:
    return (root / Path(path)).resolve()


def relative(root: Path, path: Path) -> str:
    try:
        result = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise V4EvaluationError(f"artifact escapes project root: {path}") from error
    checked_path(root, result)
    return result


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise V4EvaluationError(f"{name} must be lowercase SHA-256")
    return value


def validate_interface(interface: dict, config: dict) -> None:
    if interface.get("status") != "design_interface_frozen_before_new_holdout_metrics":
        raise V4EvaluationError("interface is not the frozen v4 design contract")
    if interface.get("config") != config:
        raise V4EvaluationError("config differs from frozen interface")
    core = interface.get("core", {})
    functions = {str(f).split("(", 1)[0].strip() for f in core.get("functions", [])}
    if (core.get("module") != "eeg_tools.learned_denoise_v4" or not
            {"load_config", "validate_config", "load_model", "apply_model_to_array"}.issubset(functions)):
        raise V4EvaluationError("required core function signatures missing")
    result = str(core.get("result", ""))
    if "dataclass" not in result or "[T,C]" not in result or any(f not in result for f in ARRAY_FIELDS + ("diagnostics",)):
        raise V4EvaluationError("result must declare dataclass arrays [T,C] except diagnostics")
    if (config.get("status") != "formal" or config.get("model_version") != MODEL_VERSION
            or config.get("sample_rate_hz") != 250 or config.get("window_samples") != WINDOW
            or config.get("hop_samples") != WINDOW):
        raise V4EvaluationError("expected formal v4 250Hz/window500/hop500")
    if (config.get("runtime", {}).get("enabled") is not False
            or config.get("runtime", {}).get("production_validated") is not False):
        raise V4EvaluationError("automatic/production processing must stay disabled")


def validate_prereg(prereg: dict) -> dict:
    if prereg.get("status") != "formal" or prereg.get("experiment") != MODEL_VERSION:
        raise V4EvaluationError("expected formal v4 preregistration")
    limits = prereg.get("predeclared_acceptance", {})
    for key in LIMIT_KEYS:
        value = limits.get(key)
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise V4EvaluationError(f"invalid predeclared_acceptance.{key}")
    expected = {"new_dong_subjects": list(DONG_SUBJECTS), "dong_interval_samples": [125, 1125],
                "all_dong_targets_blocks_channels": True, "research_only": True,
                "subject_disjoint_claim": False, "production_validated": False, "automatic_enabled": False}
    if any(limits.get(k) != v for k, v in expected.items()):
        raise V4EvaluationError("preregistered offline/Dong scope differs")
    return limits


def lock_files(root: Path, lock: dict, label: str) -> dict[str, str]:
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise V4EvaluationError(f"{label} needs a nonempty flat files SHA mapping")
    resolved = {}
    for key, digest in files.items():
        if not isinstance(key, str) or Path(key).is_absolute():
            raise V4EvaluationError("lock paths must be repo-relative")
        path = checked_path(root, key)
        if str(path) in resolved:
            raise V4EvaluationError("duplicate resolved lock path")
        resolved[str(path)] = _sha(digest, key)
    differences = verify_files(root, files)
    if differences:
        raise V4EvaluationError(f"{label} SHA mismatch: {differences}")
    return resolved


def inspect_lock(root: Path, lock_path: Path, interface_path: Path, config_path: Path,
                 *, checkpoint: Path, dataset: Path, prereg_path: Path,
                 training_lock_path: Path) -> dict:
    root = root.resolve()
    lock = read_json(lock_path, "experiment lock")
    training = read_json(training_lock_path, "training source lock")
    for item, stage in ((lock, "evaluation"), (training, "training")):
        if (type(item.get("schema_version")) is not int or item["schema_version"] != 1
                or item.get("model_version") != MODEL_VERSION or item.get("stage") != stage
                or item.get("production_validated") is not False or item.get("automatic_enabled") is not False):
            raise V4EvaluationError(f"invalid {stage} lock schema/version/runtime flags")
        lock_files(root, item, stage)
        verify_lock(root, relative(root, lock_path if stage == "evaluation" else training_lock_path))
    config, interface, prereg = (read_json(p) for p in (config_path, interface_path, prereg_path))
    validate_interface(interface, config)
    validate_prereg(prereg)
    records = {str(checked_path(root, key)): digest for key, digest in lock["files"].items()}
    required_files = BASE_FILES + EVALUATION_FILES + [
        "datasets/processed/learned_denoiser_v1.npz", DONG_METADATA,
        f"{DONG_DIR}/DATASET_MANIFEST.json",
    ]
    required = {str(checked_path(root, key)) for key in required_files}
    required.update(str(p.resolve()) for p in (checkpoint, dataset, Path(str(dataset) + ".manifest.json"),
                    prereg_path, config_path, interface_path, training_lock_path))
    sources = prereg.get("data", {}).get("sources", {})
    if not isinstance(sources, dict) or not sources:
        raise V4EvaluationError("preregistration missing data.sources SHA evidence")
    for name, source in sources.items():
        path = checked_path(root, source["relative_path"])
        required.add(str(path))
        if records.get(str(path)) != _sha(source.get("sha256"), name):
            raise V4EvaluationError(f"author source not bound to preregistered SHA: {name}")
    required.update(str(checked_path(root, f"{DONG_DIR}/{s}")) for s in DONG_SUBJECTS)
    required.add(str(checkpoint.with_suffix(".training.json")))
    missing = sorted(required - records.keys())
    if missing:
        raise V4EvaluationError(f"experiment lock missing required source/payload hashes: {missing}")
    for key, path in (("preregistration_sha256", prereg_path), ("config_sha256", config_path),
                      ("dataset_sha256", dataset)):
        for item in (lock, training):
            if _sha(item.get(key), key) != sha256(path):
                raise V4EvaluationError(f"{key} binding mismatch")
    for key, path in (("checkpoint_sha256", checkpoint), ("training_source_lock_sha256", training_lock_path)):
        if _sha(lock.get(key), key) != sha256(path):
            raise V4EvaluationError(f"{key} binding mismatch")
    for key, digest in training["files"].items():
        if records.get(str(checked_path(root, key))) != digest:
            raise V4EvaluationError(f"evaluation and training source locks disagree: {key}")
    if "scripts/train_denoiser_v4.py" not in training["files"]:
        raise V4EvaluationError("training lock missing trainer source")
    records[str(lock_path)] = sha256(lock_path)
    return {"lock": lock, "config": config, "interface": interface, "preregistration": prereg,
            "records": records, "all_payloads_present": True}


def reverify(root: Path, records: dict[str, str]) -> None:
    differences = verify_files(root, {relative(root, Path(p)): digest for p, digest in records.items()})
    if differences:
        raise V4EvaluationError(f"frozen inputs changed during diagnostics: {differences}")


def validate_result(result: Any, signal: np.ndarray) -> dict[str, np.ndarray]:
    arrays = {}
    for name in ARRAY_FIELDS:
        value = getattr(result, name, None)
        if not isinstance(value, np.ndarray) or value.shape != signal.shape:
            raise V4EvaluationError(f"core.{name} must be array [T,C]={signal.shape}")
        if name.endswith("mask"):
            if value.dtype != np.dtype(bool):
                raise V4EvaluationError(f"core.{name} must be boolean")
        elif value.dtype.kind not in "iuf":
            raise V4EvaluationError(f"core.{name} must be real numeric")
        elif name in ("artifact_probability", "confidence"):
            if np.any(np.isinf(value)) or np.any((value < 0) | (value > 1)):
                raise V4EvaluationError(f"core.{name} must be in [0,1] (NaN only for unscored/rejected)")
        else:
            finite = np.isfinite(signal)
            if not np.all(np.isfinite(value[finite])) or not np.array_equal(value[~finite], signal[~finite], equal_nan=True):
                raise V4EvaluationError(f"core.{name} introduced invalid values")
        arrays[name] = value
    if not isinstance(getattr(result, "diagnostics", None), dict):
        raise V4EvaluationError("core.diagnostics must be dict")
    for name in ("artifact_probability", "confidence"):
        if np.any(np.isnan(arrays[name]) & ~arrays["rejected_mask"]):
            raise V4EvaluationError(f"core.{name} missing for non-rejected samples")
    mask = arrays["replacement_mask"]
    if np.any(mask & (~arrays["valid_mask"] | arrays["rejected_mask"])):
        raise V4EvaluationError("replacement overlaps invalid/rejected samples")
    if not np.array_equal(arrays["denoised"][~mask], signal[~mask], equal_nan=True):
        raise V4EvaluationError("gated output changed samples outside replacement")
    return arrays


def infer_windows(bundle: Any, windows: np.ndarray, *, chunk_size: int = 128,
                  valid_mask: np.ndarray | None = None, apply=None) -> dict:
    """Each independent500 window becomes one channel. No channel/window mixing."""
    if (not isinstance(windows, np.ndarray) or windows.ndim != 2 or windows.shape[1] != WINDOW
            or len(windows) == 0 or windows.dtype.kind not in "iuf" or chunk_size < 1):
        raise V4EvaluationError("inference expects nonempty real [N,500], positive chunk_size")
    if valid_mask is not None and (valid_mask.shape != windows.shape or valid_mask.dtype != np.dtype(bool)):
        raise V4EvaluationError("valid_mask must be boolean [N,500]")
    if apply is None:
        apply = importlib.import_module("eeg_tools.learned_denoise_v4").apply_model_to_array
    chunks = {k: [] for k in ARRAY_FIELDS}
    diagnostics = []
    for start in range(0, len(windows), chunk_size):
        signal = np.ascontiguousarray(windows[start:start + chunk_size].T, dtype=np.float64)
        mask = None if valid_mask is None else np.ascontiguousarray(valid_mask[start:start + chunk_size].T)
        result = apply(bundle, signal, valid_mask=mask)
        arrays = validate_result(result, signal)
        for key, array in arrays.items():
            chunks[key].append(array.T.copy())
        diagnostics.append({"first_window": start, "window_count": signal.shape[1], "core": result.diagnostics})
    return {**{key: np.concatenate(value) for key, value in chunks.items()}, "diagnostics": diagnostics}


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))


def normalized_rmse(error: np.ndarray, scale: np.ndarray) -> float | None:
    positive = scale > 0
    if np.any(~positive & np.any(error != 0, axis=1)):
        return None
    normalized = np.zeros_like(error, dtype=np.float64)
    np.divide(error, scale[:, None], out=normalized, where=positive[:, None])
    value = rms(normalized)
    return value if np.isfinite(value) else None


def stats(values: np.ndarray) -> dict:
    finite = np.asarray(values)[np.isfinite(values)]
    return {"count": int(np.size(values)), "undefined_count": int(np.size(values) - len(finite)),
            "min": float(np.min(finite)) if len(finite) else None,
            "mean": float(np.mean(finite)) if len(finite) else None,
            "max": float(np.max(finite)) if len(finite) else None,
            "quantiles_0_05_50_95_100": np.quantile(finite, [0, .05, .5, .95, 1]).tolist() if len(finite) else None}


def protected_discrepancy(raw: np.ndarray, output: np.ndarray) -> np.ndarray:
    """Local500 rFFT invariant with preregistered floor; not whole-record SSVEP."""
    if raw.shape != output.shape or raw.ndim != 2 or raw.shape[1] != WINDOW:
        raise V4EvaluationError("protected FFT expects matching [N,500]")
    result = np.zeros(len(raw), dtype=np.float64)
    frequency = np.fft.rfftfreq(WINDOW, d=1 / 250)
    band = (frequency >= 8) & (frequency <= 40)
    for i, (x, y) in enumerate(zip(raw, output)):
        if np.array_equal(x, y):
            continue
        ac = x - np.mean(x, dtype=np.float64)
        scale = rms(ac)
        if scale <= 0 or not np.isfinite(scale):
            result[i] = np.nan  # fail closed; unchanged zero reference handled above
            continue
        # Equivalent ratio; dividing before FFT avoids overflow/underflow.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            numerator = np.linalg.norm(np.fft.rfft((y - x) / scale)[band])
            denominator = max(np.linalg.norm(np.fft.rfft(ac / scale)[band]), np.sqrt(WINDOW) * 1e-12)
            result[i] = numerator / denominator
    return result


def coverage(raw: np.ndarray, output: np.ndarray, result: dict) -> dict:
    changed = output != raw  # actual changes, distinct from accepted no-op
    replace, valid, rejected = (result[k] for k in ("replacement_mask", "valid_mask", "rejected_mask"))
    return {"window_count": len(raw), "sample_count": raw.size,
            "replacement_window_count": int(np.count_nonzero(np.any(replace, axis=1))),
            "replacement_fraction": float(np.mean(replace)),
            "replacement_window_fraction": float(np.mean(np.any(replace, axis=1))),
            "actual_changed_window_count": int(np.count_nonzero(np.any(changed, axis=1))),
            "actual_change_fraction": float(np.mean(np.any(changed, axis=1))),
            "actual_change_sample_fraction": float(np.mean(changed)),
            "invalid_sample_count": int(np.count_nonzero(~valid)),
            "rejected_window_count": int(np.count_nonzero(np.any(rejected, axis=1))),
            "rejected_sample_fraction": float(np.mean(rejected)),
            "max_abs_change_original_units": float(np.max(np.abs(output - raw)))}


def metric_summary(raw: np.ndarray, clean: np.ndarray, result: dict) -> dict:
    if raw.shape != clean.shape or raw.ndim != 2 or raw.shape[1] != WINDOW or not len(raw):
        raise V4EvaluationError("metrics require nonempty matching [N,500]")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(clean)):
        raise V4EvaluationError("no nonfinite metric exclusion permitted")
    scales = np.sqrt(np.mean((raw - raw.mean(axis=1, keepdims=True)) ** 2, axis=1))
    raw_error = rms(raw - clean)
    metrics = {}
    for name, output in (("raw", raw), ("candidate", result["model_denoised"]), ("gated", result["denoised"])):
        error = rms(output - clean)  # primary true clean including ORIGINAL DC
        gain = 20 * np.log10(raw_error / error) if raw_error > 0 and error > 0 else None
        metrics[name] = {"global_rmse_original_units": error,
                         "rmse_improvement_fraction": 1 - error / raw_error if raw_error > 0 else None,
                         "snr_gain_db": float(gain) if gain is not None else None,
                         "ratio_to_raw_rmse": error / raw_error if raw_error > 0 else None,
                         "normalized_rmse_all_samples": normalized_rmse(output - clean, scales),
                         "normalized_change_rmse_all_samples": normalized_rmse(output - raw, scales),
                         "max_abs_change_original_units": float(np.max(np.abs(output - raw))),
                         "protected_8_40hz_local500": stats(protected_discrepancy(raw, output))}
    return {"metrics": metrics, "coverage": coverage(raw, result["denoised"], result),
            "raw_mean_distribution": stats(raw.mean(axis=1)), "raw_ac_rms_distribution": stats(scales),
            "artifact_score_distribution_not_calibrated": stats(result["artifact_probability"].mean(axis=1))}


def subset_result(result: dict, indices) -> dict:
    return {key: result[key][indices] for key in ARRAY_FIELDS}


def gate(value: float | None, limit: float, relation: str) -> dict:
    passed = value is not None and np.isfinite(value) and (value <= limit if relation == "<=" else value >= limit)
    return {"value": value, "limit": limit, "relation": relation, "passed": bool(passed),
            "status": "passed" if passed else "failed"}


def spectral_gate(raw: np.ndarray, result: dict, limit: float) -> dict:
    values = protected_discrepancy(raw, result["denoised"])
    value = float(np.max(values)) if np.all(np.isfinite(values)) else None
    return {**gate(value, limit, "<="), "per_window": jsonable(values),
            "scope": "local500_windows_only_not_whole_session_SSVEP"}


def evaluate_paired(data: dict, bundle: Any, limits: dict, *, chunk_size=128, apply=None) -> dict:
    raw, clean, identity = (data[k] for k in ("noisy", "clean", "is_identity"))
    if (identity.dtype != np.dtype(bool) or identity.shape != (len(raw),)
            or not np.any(identity) or not np.any(~identity) or not np.array_equal(raw[identity], clean[identity])):
        raise V4EvaluationError("paired data must contain original true-pure identity and corrupted samples")
    result = infer_windows(bundle, raw, chunk_size=chunk_size, apply=apply)
    summary = {name: metric_summary(raw[m], clean[m], subset_result(result, m))
               for name, m in (("identity", identity), ("corrupted", ~identity))}
    summary["all"] = metric_summary(raw, clean, result)
    grouped = {}
    for field in ("group_id", "channel_index"):
        grouped[field] = {}
        for key in np.unique(data[field]):
            mask = data[field] == key
            grouped[field][str(int(key))] = {
                name: metric_summary(raw[mask & kind], clean[mask & kind], subset_result(result, mask & kind))
                for name, kind in (("identity", identity), ("corrupted", ~identity)) if np.any(mask & kind)}
    rows = []
    for i in range(len(raw)):
        rows.append({**{k: int(data[k][i]) for k in ("segment_id", "channel_index", "window_start", "group_id")},
                     "is_identity": bool(identity[i]),
                     **metric_summary(raw[i:i+1], clean[i:i+1], subset_result(result, slice(i, i+1)))})
    clean_summary, corrupted = summary["identity"], summary["corrupted"]
    gates = {
        "clean_replacement": gate(clean_summary["coverage"]["replacement_fraction"], limits["clean_replacement_fraction_max"], "<="),
        "clean_all_sample_normalized_rmse": gate(clean_summary["metrics"]["gated"]["normalized_rmse_all_samples"], limits["clean_normalized_change_rmse_max"], "<="),
        "corrupted_improvement": gate(corrupted["metrics"]["gated"]["rmse_improvement_fraction"], limits["corrupted_rmse_improvement_min"], ">="),
        "corrupted_snr_gain": gate(corrupted["metrics"]["gated"]["snr_gain_db"], limits["corrupted_snr_gain_db_min"], ">="),
        "corrupted_actual_change_coverage": gate(corrupted["coverage"]["actual_change_fraction"], limits["corrupted_actual_change_fraction_min"], ">="),
        "protected_band": spectral_gate(raw, result, limits["protected_band_relative_discrepancy_max"])}
    return {"name": "paired_resampled_Klados_test", "reference": "real_resampled_pure_original_DC_not_DC_constrained",
            "split": "test", "subject_disjoint_claim": False, "sample_exclusion_count": 0,
            "summary": summary, "per_group_and_channel": grouped, "per_window": rows,
            "gates": gates, "inference_chunks": result["diagnostics"]}


def evaluate_known(data: dict, bundle: Any, limits: dict, *, chunk_size=128, apply=None) -> dict:
    raw, clean, artifacts = (data[k] for k in ("noisy", "clean", "artifact_mask"))
    if raw.shape != (26, 8, 1000) or clean.shape != raw.shape or artifacts.shape != raw.shape:
        raise V4EvaluationError("known v1 stress must include all26 [8,1000] windows")
    raw, clean, artifacts = (x.reshape(-1, WINDOW) for x in (raw, clean, artifacts))
    result = infer_windows(bundle, raw, chunk_size=chunk_size, apply=apply)
    summary = metric_summary(raw, clean, result)
    cleanish = ~artifacts.astype(bool)
    clean_changes = {}
    for name, output in (("candidate", result["model_denoised"]), ("gated", result["denoised"])):
        changes = (output - raw)[cleanish]
        clean_changes[name] = {"sample_count": len(changes), "change_rmse_original_units": rms(changes) if len(changes) else None,
                               "max_abs_change_original_units": float(np.max(np.abs(changes))) if len(changes) else None}
    return {"name": "known_v1_regression_stress_test", "session_id": V1_SESSION,
            "independent": False, "role": "known_development_stress_not_independent_holdout",
            "target_kind": "semi_synthetic_not_new_paired_GT", "original_window_count": 26,
            "single_channel_500_window_count": len(raw), "summary": summary,
            "cleanish_raw_changes": clean_changes,
            "cleanish_replacement_fraction": float(np.mean(result["replacement_mask"][cleanish])) if np.any(cleanish) else None,
            "gates": {"gated_to_raw_rmse_ratio": gate(summary["metrics"]["gated"]["ratio_to_raw_rmse"], limits["known_session_rmse_ratio_to_noisy_max"], "<=")},
            "inference_chunks": result["diagnostics"], "threshold_tuning_performed": False}


def validate_dong_metadata(meta: dict) -> None:
    # Exactly v3's audited axes/rate/stimulus interval; no gain changes.
    expected = {"dataset": "Dong2023", "sample_rate_hz": 250, "expected_shape": [8, 1250, 40, 4],
                "axis_order": ["channel", "time", "target", "block"], "array_key": "eegdata",
                "channel_count": 8, "target_count": 40, "stimulus_onset_status": "confirmed_primary_author_paper",
                "analysis_window_samples": {"start": 125, "stop_exclusive": 1125, "count": 1000, "indexing": "zero_based_half_open"}}
    if any(meta.get(k) != v for k, v in expected.items()) or meta.get("trial_structure", {}).get("blocks") != 4:
        raise V4EvaluationError("Dong requires audited axes/[8,1250,40,4]/250Hz/stimulus[125:1125]")
    if meta.get("target_axis_mapping_confirmed") is not False or meta.get("target_frequencies_hz") is not None:
        raise V4EvaluationError("v4 frozen Dong audit is mapping-pending; do not invent target labels")


def dong_windows(raw: np.ndarray) -> np.ndarray:
    if (not isinstance(raw, np.ndarray) or raw.shape != (8, 1250, 40, 4)
            or raw.dtype.kind not in "iuf" or not np.all(np.isfinite(raw))):
        raise V4EvaluationError("Dong must be finite real [8,1250,40,4]; no exclusions")
    return raw[:, 125:1125, :, :].transpose(2, 3, 0, 1).reshape(-1, WINDOW).astype(np.float64)


def load_dong(root: Path) -> tuple[np.ndarray, list[dict], dict]:
    from scipy.io import loadmat
    meta = read_json(checked_path(root, DONG_METADATA))
    validate_dong_metadata(meta)
    arrays, rows = [], []
    for subject in DONG_SUBJECTS:
        path = checked_path(root, f"{DONG_DIR}/{subject}")
        arrays.append(dong_windows(loadmat(path, variable_names=["eegdata"]).get("eegdata")))
        for target in range(40):
            for block in range(4):
                for channel in range(8):
                    for window in range(2):
                        rows.append({"subject": subject, "target_axis_index_not_frequency_label": target,
                                     "block_index": block, "channel_index": channel, "start_sample": 125 + WINDOW * window})
    return np.concatenate(arrays), rows, meta


def evaluate_dong(raw: np.ndarray, rows: list[dict], meta: dict, bundle: Any, limits: dict,
                  *, chunk_size=128, apply=None) -> dict:
    validate_dong_metadata(meta)
    if raw.shape != (5120, WINDOW) or len(rows) != len(raw):
        raise V4EvaluationError("Dong S10/S11 requires all5120 channel windows (320trials,2windows,8channels)")
    result = infer_windows(bundle, raw, chunk_size=chunk_size, apply=apply)
    def summarize(indices):
        outputs = {"candidate": result["model_denoised"][indices], "gated": result["denoised"][indices]}
        selected = raw[indices]
        return {"coverage": coverage(selected, outputs["gated"], subset_result(result, indices)),
                "local_8_40hz": {k: stats(protected_discrepancy(selected, v)) for k, v in outputs.items()}}
    summary = summarize(slice(None))
    subjects = np.array([row["subject"] for row in rows])
    gates = {"actual_change_false_acceptance_proxy": gate(summary["coverage"]["actual_change_fraction"], limits["new_dong_actual_change_fraction_max"], "<="),
             "protected_band": spectral_gate(raw, result, limits["protected_band_relative_discrepancy_max"])}
    return {"name": "new_Dong_S10_S11_transfer_safety", "true_clean_known": False,
            "clean_rmse": "not_assessed", "target_accuracy": "not_assessed", "target_specific_gate": "not_assessed",
            "mapping_confirmed": False, "units": "stored_original_units_physical_calibration_unknown",
            "false_acceptance_interpretation": "actual_change_safety_proxy_NOT_clean_ground_truth_error_rate",
            "whole_session_SSVEP_preservation": "not_assessed", "sample_exclusion_count": 0,
            "summary": summary, "per_subject": {s: summarize(subjects == s) for s in DONG_SUBJECTS},
            "per_channel": {str(c): summarize(np.array([r["channel_index"] == c for r in rows])) for c in range(8)},
            "per_window": [{**row, **summarize(slice(i, i+1))} for i, row in enumerate(rows)],
            "gates": gates, "inference_chunks": result["diagnostics"]}


def write_new_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(jsonable(report), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def make_check_report(lock_path, interface_path, config_path, result, error) -> dict:
    return {"schema_version": 1, "status": "formal", "check_only": True,
            "evaluation_executed": False, "holdout_accessed": False, "training_executed": False,
            "automatic_processing_enabled": False, "production_validated": False,
            "lock_path": str(lock_path), "interface_contract_path": str(interface_path), "config_path": str(config_path),
            "frozen_check_status": "passed" if error is None else "failed",
            "frozen_inputs": result["records"] if result else None,
            "payloads_present": bool(result), "error": str(error) if error else None,
            "not_assessed": ["training", "paired evaluation", "new Dong", "known v1 stress"]}


def load_bundle(checkpoint: Path, config: Path, device: str, threads: int):
    if threads < 1:
        raise V4EvaluationError("threads must be positive")
    core = importlib.import_module("eeg_tools.learned_denoise_v4")
    if core.torch is None:
        raise V4EvaluationError("selected runtime has no torch; no installation/fallback")
    core.torch.set_num_threads(threads)
    cfg = core.load_config(config)
    return core, core.load_model(checkpoint, config=cfg, device=device)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model", "--checkpoint", dest="model", type=Path, default=Path("models/denoiser_v4.pt"))
    parser.add_argument("--config", type=Path, default=Path("configs/learned_denoiser_v4.json"))
    parser.add_argument("--dataset", type=Path, default=Path("datasets/processed/learned_denoiser_v4.npz"))
    parser.add_argument("--preregistration", type=Path, default=Path("analysis_outputs/learned_denoiser_v4_preregistration.json"))
    parser.add_argument("--interface", type=Path, default=Path("analysis_outputs/learned_denoiser_v4_interface_contract.json"))
    parser.add_argument("--experiment-lock", type=Path, default=Path("analysis_outputs/learned_denoiser_v4_experiment_lock.json"))
    parser.add_argument("--training-source-lock", type=Path, default=Path("analysis_outputs/learned_denoiser_v4_training_source_lock.json"))
    parser.add_argument("--regression-dataset", type=Path, default=Path("datasets/processed/learned_denoiser_v1.npz"))
    parser.add_argument("--device", choices=["cpu", "cuda", "cuda:0"], default="cuda")
    parser.add_argument("--threads", type=positive_int, default=2)
    parser.add_argument("--chunk-size", type=positive_int, default=128)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    paths = {name: resolve(root, getattr(args, name)) for name in ("model", "config", "dataset", "preregistration",
                    "interface", "experiment_lock", "training_source_lock", "regression_dataset", "output")}
    output = paths.pop("output")
    # Reserve exclusive report BEFORE test access. Only complete this newly
    # created handle; never reopen or overwrite a pre-existing report.
    if any(output.is_relative_to(root / directory) for directory in ("datasets", "models", "configs", "eeg_tools", "scripts", "tests")):
        parser.error("report may not be written in protected source/data directories")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = output.open("x", encoding="utf-8")
    except FileExistsError:
        print(f"Refuse existing report: {output}", file=sys.stderr)
        return 1
    report = {"schema_version": 1, "status": "formal", "model_version": MODEL_VERSION,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "check_only": args.check_only,
              "training_executed": False, "evaluation_executed": False, "holdout_accessed": False,
              "research_only": True, "production_validated": False, "automatic_processing_enabled": False,
              "threshold_tuning_performed": False, "evaluation_status": "started", "sections": {}}
    def persist():
        handle.seek(0)
        handle.truncate()
        json.dump(jsonable(report), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
    persist()
    start = time.perf_counter()
    frozen = None
    try:
        frozen = inspect_lock(root, paths["experiment_lock"], paths["interface"], paths["config"],
                              checkpoint=paths["model"], dataset=paths["dataset"], prereg_path=paths["preregistration"],
                              training_lock_path=paths["training_source_lock"])
        report["frozen_inputs"] = frozen["records"]
        if args.check_only:
            report.update(make_check_report(paths["experiment_lock"], paths["interface"], paths["config"], frozen, None))
            report["evaluation_status"] = "not_assessed"
            return 0
        # These evaluation-only inputs must be bound by the experiment lock,
        # including any explicitly selected regression-dataset path.
        extra = [paths["regression_dataset"], checked_path(root, DONG_METADATA)]
        extra_records = {str(p): sha256(p) for p in extra}
        for path, digest in extra_records.items():
            if frozen["records"].get(path) != digest:
                raise V4EvaluationError(f"evaluation input missing experiment-lock SHA binding: {path}")
        report["supplementary_input_sha256"] = extra_records
        core, bundle = load_bundle(paths["model"], paths["config"], args.device, args.threads)
        if bundle.metadata.get("dataset_sha256") != frozen["lock"]["dataset_sha256"]:
            raise V4EvaluationError("checkpoint metadata dataset SHA differs from experiment lock")
        report["runtime"] = {"python": sys.version, "executable": sys.executable, "device": bundle.device,
                             "torch": str(core.torch.__version__), "threads": args.threads, "chunk_size": args.chunk_size,
                             "artifact_threshold": bundle.artifact_threshold, "probability_calibrated": False}
        limits = frozen["preregistration"]["predeclared_acceptance"]
        data_module = importlib.import_module("eeg_tools.learned_dataset_v4")
        reverify(root, {**frozen["records"], **extra_records})
        report["holdout_accessed"] = True
        persist()
        data = data_module.load_evaluation_split(paths["dataset"])
        if (data.get("dataset_sha256") != frozen["lock"]["dataset_sha256"]
                or data.get("manifest_sha256") != sha256(Path(str(paths["dataset"]) + ".manifest.json"))):
            raise V4EvaluationError("loaded dataset/manifest SHA mismatch")
        report["evaluation_executed"] = True
        report["sections"]["paired"] = evaluate_paired(data, bundle, limits, chunk_size=args.chunk_size, apply=core.apply_model_to_array)
        persist()
        known_module = importlib.import_module("scripts.evaluate_denoiser_v2_transfer")
        known = known_module.load_regression(paths["regression_dataset"], V1_SESSION, 250)
        report["sections"]["known_v1_stress"] = evaluate_known(known, bundle, limits, chunk_size=args.chunk_size, apply=core.apply_model_to_array)
        persist()
        raw, rows, meta = load_dong(root)
        report["sections"]["dong"] = evaluate_dong(raw, rows, meta, bundle, limits, chunk_size=args.chunk_size, apply=core.apply_model_to_array)
        reverify(root, {**frozen["records"], **extra_records})
        report["post_inference_immutability_verified"] = True
        report["research_gates_passed"] = all(g["passed"] for section in report["sections"].values() for g in section["gates"].values())
        report["evaluation_status"] = "research_gates_passed" if report["research_gates_passed"] else "research_gates_failed"
        report["notice"] = "Research gates are NOT production/clinical acceptance. Failed results retained; do not retune test thresholds."
        return 0 if report["research_gates_passed"] else 2
    except Exception as error:
        report["evaluation_status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        if args.check_only:
            report.update(make_check_report(paths["experiment_lock"], paths["interface"], paths["config"], frozen, error))
        print(f"v4 diagnostic refused/failed: {error}", file=sys.stderr)
        return 1
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        persist()
        handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
