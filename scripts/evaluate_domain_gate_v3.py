"""Frozen v3 applicability evaluation; research-only, exclusive new reports.

This CLI does not train or tune. It requires a source/checkpoint freeze proof
before reading S4/S5. Tests use synthetic/mocked inputs, never real holdout data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib
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
from scripts import evaluate_denoiser_v2 as benchmark_v2
from scripts.run_learned_denoise_v3 import RESULT_FIELDS, load_bundle, positive_int, resolve, validate_result
from scripts.evaluate_denoiser_v2 import EvaluationError, _dump, sha256

THRESHOLDS = {
    "new_unsupported_false_acceptance_max": 0.01,
    "supported_corrupted_accepted_fraction_min": 0.5,
    "known_benchmark_corrupted_rmse_improvement_min": 0.25,
    "known_benchmark_clean_replacement_fraction_max": 0.01,
    "known_session_rmse_ratio_to_noisy_max": 1.1,
}
SUBJECTS = ("S4.mat", "S5.mat")
REGRESSION_SESSION = "session_20260924_131134_807"
SAFETY = {
    "diagnostic_only": True, "automatic_processing_enabled": False, "automatic_enabled": False,
    "production_validated": False, "release_gate_passed": False,
    "use_unsupported_as_clean_ground_truth": False, "transfer_target_accuracy": "not_assessed_target_mapping",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/domain_gate_v3.pt"))
    parser.add_argument("--denoiser-checkpoint", type=Path, default=Path("models/denoiser_v2.pt"))
    parser.add_argument("--config", type=Path, default=Path("configs/learned_denoiser_v3.json"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("analysis_outputs/learned_denoiser_v3_domain_gate_evaluation.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=positive_int, default=2)
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument("--preregistration", type=Path, default=Path("analysis_outputs/learned_denoiser_v3_preregistration.json"))
    parser.add_argument("--freeze-lock", type=Path, default=Path("analysis_outputs/learned_denoiser_v3_experiment_lock.json"),
                        help="main-thread frozen-before-evaluation items(path,sha256) proof; required even for --check-only")
    parser.add_argument("--regression-dataset", type=Path, default=Path("datasets/processed/learned_denoiser_v1.npz"))
    parser.add_argument("--check-only", action="store_true", help="validate freeze/model only; never loads known benchmark, regression or new holdout")
    return parser


def load_preregistration(path: Path) -> dict:
    prereg = json.loads(path.read_text(encoding="utf-8-sig"))
    require(prereg.get("schema_version") == 1 and prereg.get("status") == "formal"
            and prereg.get("experiment") == "learned_denoiser_v3_domain_gate", "invalid v3 preregistration")
    declaration = prereg.get("predeclared_acceptance", {})
    require(all(declaration.get(k) == v for k, v in THRESHOLDS.items()), "preregistered gates differ from frozen evaluation constants")
    require(declaration.get("all_noop_cannot_pass") is True and declaration.get("production_validated") is False
            and declaration.get("automatic_enabled") is False and declaration.get("known_session_independent") is False,
            "preregistered safety/independence declarations missing")
    data = prereg.get("data", {})
    require(data.get("new_unsupported_holdout_subjects") == list(SUBJECTS)
            and data.get("known_benchmark_split") == "test" and data.get("known_benchmark_not_new_holdout") is True,
            "frozen subject/split identities differ")
    require(data.get("dong_interval_samples") == [125, 1125] and data.get("window_samples") == 500
            and data.get("sample_rate_hz") == 250 and data.get("max_window_exclusion") == 0
            and data.get("all_targets_blocks_channels") is True and data.get("use_unsupported_as_clean_ground_truth") is False,
            "frozen raw epoch coverage/units contract differs")
    training = prereg.get("training", {})
    require(training.get("comparison") == ">" and training.get("test_access_before_freeze") is False
            and training.get("denoiser_v2_frozen") is True, "missing freeze/strict threshold contract")
    return prereg


def validate_freeze(root: Path, lock_path: Path, prereg_path: Path, prereg: dict,
                    checkpoint: Path, denoiser: Path, config: Path) -> dict[str, str]:
    # Schema accepted here is an explicit contract for the main-thread freeze
    # artifact, not inferred from a score report. Nothing creates it here.
    require(lock_path.is_file(), "missing main-thread freeze lock/source proof; no real holdout evaluation allowed")
    lock = json.loads(lock_path.read_text(encoding="utf-8-sig"))
    require(lock.get("status") in ("frozen_before_evaluation", "frozen_before_first_test_after_training", "frozen_before_test"),
            "lock must explicitly declare frozen-before-evaluation state")
    before_access = lock.get("test_access_before_freeze") is False or lock.get("test_inference_executed") is False
    require(before_access, "lock must explicitly attest no holdout/test access before freeze")
    entries = lock.get("items")
    require(isinstance(entries, list) and bool(entries), "freeze lock has no source/checkpoint hash items")
    frozen = {}
    for entry in entries:
        require(isinstance(entry, dict) and isinstance(entry.get("path"), str), "invalid freeze item path")
        path = resolve(root, Path(entry["path"]))
        digest = entry.get("sha256")
        require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "invalid freeze SHA256")
        require(str(path) not in frozen, "duplicate freeze path")
        require(path.is_file() and sha256(path) == digest, f"frozen source/checkpoint changed: {path}")
        frozen[str(path)] = digest
    required_paths = [prereg_path, checkpoint, denoiser, config,
                      root / "scripts" / "evaluate_domain_gate_v3.py",
                      root / "eeg_tools" / "learned_denoise_v3.py",
                      root / "eeg_tools" / "learned_domain_dataset_v3.py"]
    require(all(str(p.resolve()) in frozen for p in required_paths), "freeze lock missing evaluator/core/dataset/config/checkpoint/preregistration sources")
    require(any("train" in Path(p).stem and "v3" in Path(p).stem for p in frozen), "freeze lock missing v3 training source")
    protected = prereg.get("protected_sha256")
    require(isinstance(protected, dict) and bool(protected), "preregistration missing frozen v2/source identities")
    for relative, digest in protected.items():
        path = resolve(root, Path(relative))
        require(sha256(path) == digest, f"preregistered protected v2/source changed: {path}")
        frozen[str(path)] = digest
    frozen[str(lock_path)] = sha256(lock_path)
    return frozen


def reserve_output(path: Path, protected: dict[str, str]) -> None:
    require(str(path.resolve()) not in protected, "report must not overwrite a protected source")
    require("v1" not in path.stem.lower() and "v2" not in path.stem.lower(), "v3 report must not overwrite v1/v2 reports")
    require(not path.exists() and not path.is_symlink(), "output already exists; no --overwrite, select a new report")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(_dump({"status": "reserved_before_any_holdout_read", **SAFETY,
                            "warning": "A reserved/failed report blocks silent repeats; no automatic retry."}))


def infer_arrays(core: Any, bundle: Any, noisy: np.ndarray, valid: np.ndarray, batch_size: int) -> dict:
    require(noisy.ndim == 3 and noisy.shape == valid.shape and valid.dtype.kind == "b", "arrays require matching [N,C,T] and bool mask")
    shape = noisy.shape
    x, mask = noisy.reshape(-1, shape[-1]), valid.reshape(-1, shape[-1])
    outputs = {name: [] for name in RESULT_FIELDS}
    diagnostics = []
    for start in range(0, len(x), batch_size):
        raw, good = x[start:start + batch_size], mask[start:start + batch_size]
        result = core.apply_model_to_array(bundle, raw, valid_mask=good)
        validate_result(result, raw, good, bundle.support_threshold)
        for name in RESULT_FIELDS:
            outputs[name].append(np.asarray(getattr(result, name)).copy())
        diagnostics.append(result.diagnostics)
    return {**{name: np.concatenate(parts, axis=0).reshape(shape) for name, parts in outputs.items()},
            "diagnostics": diagnostics}


def rmse(values: np.ndarray, valid: np.ndarray) -> float | None:
    if not np.any(valid):
        return None
    require(np.all(np.isfinite(values[valid])), "nonfinite prediction on shared valid metric support")
    return float(np.sqrt(np.mean(np.square(values[valid], dtype=np.float64))))


def masks_by_window(result: dict, raw: np.ndarray) -> dict:
    valid = result["valid_mask"]
    full = raw.shape[-1] // 500
    require(full > 0, "at least one 500-sample window required")
    dimensions = (-1, full, 500)
    def any_sample(mask):
        return np.any(mask[..., :full * 500].reshape(dimensions), axis=-1).reshape(-1)
    domain = valid & ~result["domain_rejected_mask"]
    return {"domain_accepted": any_sample(domain), "replaced": any_sample(result["replacement_mask"]),
            "changed": any_sample(valid & (result["denoised"] != raw)), "valid": any_sample(valid),
            "mother_epoch_domain_accepted": np.any(domain, axis=(1, 2)),
            "mother_epoch_replaced": np.any(result["replacement_mask"], axis=(1, 2))}


def paired_summary(clean: np.ndarray, raw: np.ndarray, result: dict) -> dict:
    valid = result["valid_mask"]
    windows = masks_by_window(result, raw)
    metrics = {name: {"RMSE": rmse(track - clean, valid), "change_RMSE_vs_raw": rmse(track - raw, valid)}
               for name, track in (("raw", raw), ("candidate", result["model_denoised"]), ("gated", result["denoised"]))}
    denominator = metrics["raw"]["RMSE"]
    improvement = 1 - metrics["gated"]["RMSE"] / denominator if denominator else None
    good = windows["valid"]
    return {"metrics": metrics, "gated_rmse_improvement_fraction": improvement,
            "mother_epoch_count": len(raw), "single_channel_500_window_count": len(good),
            "valid_single_channel_window_count": int(good.sum()),
            "domain_accepted_windows": int(np.count_nonzero(windows["domain_accepted"] & good)),
            "domain_accepted_fraction": float(windows["domain_accepted"][good].mean()) if np.any(good) else None,
            "actual_replacement_windows": int(np.count_nonzero(windows["replaced"] & good)),
            "replacement_window_fraction": float(windows["replaced"][good].mean()) if np.any(good) else None,
            "actually_changed_windows": int(np.count_nonzero(windows["changed"] & good)),
            "replacement_sample_fraction": float(result["replacement_mask"][valid].mean()) if np.any(valid) else None,
            "mother_epoch_any_domain_accepted": int(windows["mother_epoch_domain_accepted"].sum()),
            "mother_epoch_any_replacement": int(windows["mother_epoch_replaced"].sum())}


def evaluate_known_benchmark(core: Any, bundle: Any, path: Path, batch_size: int) -> dict:
    data, manifest, digest, manifest_digest = benchmark_v2._load_dataset(path)
    indices, provenance = benchmark_v2._select_evaluation_indices(bundle.denoiser.metadata, data, manifest,
                                                                  dataset_sha256=digest, manifest_sha256=manifest_digest)
    clean, noisy, valid = (data[k][indices] for k in ("clean", "noisy", "valid_mask"))
    result = infer_arrays(core, bundle, noisy, valid, batch_size)
    labels = data["artifact_type"][indices]
    groups = {"all": np.ones(len(indices), dtype=bool), "clean": labels == "clean", "corrupted": np.isin(labels, ["eog", "emg"]),
              "eog": labels == "eog", "emg": labels == "emg"}
    summaries = {}
    for name, selected in groups.items():
        require(np.any(selected), f"known benchmark {name} group must be nonempty")
        sliced = {k: value[selected] for k, value in result.items() if k != "diagnostics"}
        summaries[name] = paired_summary(clean[selected], noisy[selected], sliced)
    return {"scope": "known_v2_benchmark_not_new_holdout", "known_v2_benchmark_not_new_holdout": True,
            "new_independent_holdout": False, "split": "test", "test_indices": indices.tolist(),
            "dataset_sha256": digest, "dataset_manifest_sha256": manifest_digest,
            "provenance_guard": provenance, "summaries": summaries,
            "target_provenance": "published_EEG_epochs_semi_synthetic_not_real_noise_ground_truth",
            "support_probability_semantics": "applicability, distinct from artifact_probability", "threshold_tuned": False}


def _normalized_fingerprint(values: np.ndarray) -> str:
    import hashlib
    canonical = np.asarray(values, dtype="<f4").copy()
    canonical[canonical == 0] = 0
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def unsupported_raw_windows(root: Path, prereg: dict) -> tuple[np.ndarray, list[dict], dict]:
    from scripts import evaluate_denoiser_v2_transfer as transfer
    from eeg_tools.learned_denoise_v2 import normalize_window
    loader = importlib.import_module("eeg_tools.learned_domain_dataset_v3")
    # This is the only evaluation-only dataset entry point. Called only after freeze.
    negative = loader.load_unsupported_holdout(root)
    normalized, supported, supplied_provenance = (negative[k] for k in ("normalized", "supported", "provenance"))
    require(normalized.shape == (5120, 1, 500) and normalized.dtype == np.float32
            and np.all(np.isfinite(normalized)) and supported.shape == (5120,) and supported.dtype.kind == "b"
            and not np.any(supported) and isinstance(supplied_provenance, list) and len(supplied_provenance) == 5120,
            "unsupported holdout must include ALL 5120 finite normalized windows, false labels, and provenance")
    meta_path = resolve(root, Path(prereg["data"]["dong_metadata"]))
    meta, spec = transfer.load_metadata(meta_path, 250.0)
    require(spec["timing_verified"] and spec["spectral_slice"].start == 125 and spec["spectral_slice"].stop == 1125,
            "unsupported analysis requires the frozen audited [125:1125] interval")
    directory = resolve(root, Path(prereg["data"]["unsupported_dong_dir"]))
    windows, provenance, normalized_expected = [], [], []
    hashes = {str(meta_path): sha256(meta_path)}
    for subject in SUBJECTS:
        path = directory / subject
        hashes[str(path)] = sha256(path)
        epochs = transfer.load_dong_epochs(path, meta)  # does not mutate v2 SUBJECTS
        require(epochs.shape[0] == 8 and epochs.shape[2:] == (40, 4), "holdout must have all8 channels,40 targets,4 blocks")
        for target in range(40):
            for block in range(4):
                for channel in range(8):
                    for half in range(2):
                        start = 125 + half * 500
                        raw = epochs[channel, start:start + 500, target, block].astype(np.float64, copy=True)
                        require(raw.shape == (500,) and np.all(np.isfinite(raw)), "no holdout exclusions or nonfinite windows")
                        normalized_expected.append(normalize_window(raw)[0])
                        windows.append(raw)
                        provenance.append({"subject": subject, "stored_target_axis_index": target,
                                           "block_axis_index": block, "channel_index_zero_based": channel,
                                           "window_index_in_trial": half, "start_sample": start, "stop_sample_exclusive": start + 500})
        require(sha256(path) == hashes[str(path)], "holdout source changed during loading")
    # Compare multiset fingerprints instead of guessing the loader's row order.
    # Scores and raw fallback below are computed from ORIGINAL stored units.
    expected = Counter(_normalized_fingerprint(row) for row in normalized_expected)
    actual = Counter(_normalized_fingerprint(row[0]) for row in normalized)
    require(expected == actual, "normalized loader coverage differs from ALL original-unit holdout windows")
    # Dataset's now-confirmed provenance schema is explicit, not inferred IDs.
    for supplied, canonical, norm, original_norm in zip(supplied_provenance, provenance, normalized[:, 0], normalized_expected):
        expected_keys = {"subject": canonical["subject"], "target_axis_index": canonical["stored_target_axis_index"],
                         "block_index": canonical["block_axis_index"], "channel_index": canonical["channel_index_zero_based"],
                         "window_index": canonical["window_index_in_trial"], "start_sample": canonical["start_sample"],
                         "stop_sample_exclusive": canonical["stop_sample_exclusive"], "split": "unsupported_holdout",
                         "domain": "unsupported_real_EEG_not_clean", "window_samples": 500, "sample_rate_hz": 250}
        require(isinstance(supplied, dict) and all(supplied.get(k) == v for k, v in expected_keys.items()),
                "evaluation-only loader provenance differs from canonical raw subject/channel/window coverage")
        original_path = str((directory / canonical["subject"]).resolve())
        require(supplied.get("source_path") == original_path and supplied.get("source_sha256") == hashes[original_path]
                and supplied.get("metadata_sha256") == hashes[str(meta_path)] and np.array_equal(norm, original_norm),
                "loader/raw provenance or per-row normalized input differs; no unknown metadata may be guessed")
    return np.asarray(windows, dtype=np.float64)[:, None, :], provenance, hashes


def summarize_unsupported(raw: np.ndarray, result: dict, provenance: list[dict]) -> dict:
    require(raw.shape[1:] == (1, 500) and len(raw) == len(provenance), "negative windows/provenance mismatch")
    windows = masks_by_window(result, raw)
    subjects = {}
    for subject in sorted({row["subject"] for row in provenance}):
        selected = np.asarray([row["subject"] == subject for row in provenance])
        accepted = windows["domain_accepted"][selected]
        scored = np.all(np.isfinite(result["support_probability"][selected]), axis=(1, 2))
        replaced, changed = windows["replaced"][selected], windows["changed"][selected]
        mothers = {}
        for i in np.flatnonzero(selected):
            row = provenance[int(i)]
            key = (row["stored_target_axis_index"], row["block_axis_index"])
            state = mothers.setdefault(key, {"domain_accepted_any_channel_window": False, "replaced_any_channel_window": False})
            state["domain_accepted_any_channel_window"] |= bool(windows["domain_accepted"][i])
            state["replaced_any_channel_window"] |= bool(windows["replaced"][i])
        refused = result["domain_rejected_mask"][selected] | ~result["valid_mask"][selected]
        fallback_exact = bool(np.array_equal(result["denoised"][selected][refused], raw[selected][refused], equal_nan=True))
        subjects[subject] = {
            "single_channel_500_window_count": int(selected.sum()), "mother_trial_count": len(mothers),
            "false_accepted_windows": int(accepted.sum()), "false_acceptance_fraction": float(accepted.mean()),
            "domain_scored_window_count": int(scored.sum()), "excluded_window_count": 0,
            "actual_replacement_windows": int(replaced.sum()), "actual_replacement_window_fraction": float(replaced.mean()),
            "actually_changed_windows": int(changed.sum()),
            "mother_trials_with_any_domain_acceptance": sum(v["domain_accepted_any_channel_window"] for v in mothers.values()),
            "mother_trials_with_any_replacement": sum(v["replaced_any_channel_window"] for v in mothers.values()),
            "gated_change_RMSE_original_units": rmse(result["denoised"][selected] - raw[selected], result["valid_mask"][selected]),
            "candidate_change_RMSE_original_units": rmse(result["model_denoised"][selected] - raw[selected], result["valid_mask"][selected]),
            "domain_refused_raw_fallback_exact_original_units": fallback_exact,
        }
    return {"scope": "new_unsupported_S4_S5_holdout_safety_only", "subjects": subjects,
            "single_channel_500_window_count": len(raw), "false_accepted_windows": int(windows["domain_accepted"].sum()),
            "false_acceptance_fraction": float(windows["domain_accepted"].mean()),
            "domain_scored_window_count": int(np.count_nonzero(np.all(np.isfinite(result["support_probability"]), axis=(1, 2)))),
            "excluded_window_count": 0, "actual_replacement_windows": int(windows["replaced"].sum()),
            "actually_changed_windows": int(windows["changed"].sum()),
            "counting_unit": "one channel x one contiguous500sample window; domain accepted if ANY valid sample accepted, regardless artifact/no replacement",
            "mother_trial_counting_unit": "subject x stored target axis x block, all8 channels/all2 windows retained",
            "clean_ground_truth_available": False, "denoising_efficacy_assessed": False,
            "accuracy_status": "not_assessed_target_mapping", "raw_unit": "stored original numeric units; physical-unit metadata not inferred"}


def evaluate_unsupported(core: Any, bundle: Any, root: Path, prereg: dict, batch_size: int) -> dict:
    raw, provenance, hashes = unsupported_raw_windows(root, prereg)
    result = infer_arrays(core, bundle, raw, np.ones_like(raw, dtype=bool), batch_size)
    report = summarize_unsupported(raw, result, provenance)
    require(set(report["subjects"]) == set(SUBJECTS) and len(raw) == 5120
            and all(x["single_channel_500_window_count"] == 2560 and x["mother_trial_count"] == 160 for x in report["subjects"].values()),
            "new unsupported holdout coverage incomplete; no exclusion allowed")
    require(all(sha256(Path(path)) == digest for path, digest in hashes.items()), "unsupported raw/metadata changed during inference")
    report["source_sha256"] = hashes
    report["provenance"] = provenance
    report["inference_diagnostics"] = result["diagnostics"]
    return report


def evaluate_known_session(core: Any, bundle: Any, path: Path, batch_size: int) -> dict:
    from scripts import evaluate_denoiser_v2_transfer as transfer
    before = sha256(path)
    # Reuse read-only 26-window source/clean-ish validation, not its old metrics.
    data = transfer.load_regression(path, REGRESSION_SESSION, 250.0)
    raw, clean = data["noisy"], data["clean"]
    result = infer_arrays(core, bundle, raw, np.ones_like(raw, dtype=bool), batch_size)
    summary = paired_summary(clean, raw, result)
    denominator = summary["metrics"]["raw"]["RMSE"]
    ratio = summary["metrics"]["gated"]["RMSE"] / denominator if denominator else None
    cleanish = result["valid_mask"] & ~data["artifact_mask"]
    require(sha256(path) == before, "known session source changed during inference")
    return {"scope": "known_v1_stress_not_independent", "independent": False, "heldout": False,
            "seen_by_v1_experiments": True, "session_id": REGRESSION_SESSION, "original_source_window_count": len(raw),
            "source_window_indices": data["indices"].tolist(), "dataset_sha256": before,
            "target_provenance": "semi_synthetic_cleanish_NOT_absolute_clean_real_noise_ground_truth",
            "summary": summary, "gated_to_raw_RMSE_ratio": ratio,
            "cleanish_RMSE": {name: rmse(track - clean, cleanish) for name, track in (("raw", raw), ("candidate", result["model_denoised"]), ("gated", result["denoised"]))},
            "v3_fresh_inference": True, "old_v2_report_used_as_v3_measurement": False, "threshold_tuned": False}


def research_gate(benchmark: dict, unsupported: dict, known: dict) -> dict:
    corrupted, clean = benchmark["summaries"]["corrupted"], benchmark["summaries"]["clean"]
    checks = {}
    def check(name, observed, threshold, minimum=False):
        passed = isinstance(observed, (int, float)) and not isinstance(observed, bool) and np.isfinite(observed)
        checks[name] = {"observed": observed, "threshold": threshold, "direction": "min" if minimum else "max",
                        "passed": bool(passed and (observed >= threshold if minimum else observed <= threshold))}
    check("negative_false_acceptance", unsupported["false_acceptance_fraction"], THRESHOLDS["new_unsupported_false_acceptance_max"])
    for subject in SUBJECTS:
        evidence = unsupported.get("subjects", {}).get(subject, {})
        check(f"negative_false_acceptance_{subject}", evidence.get("false_acceptance_fraction"), THRESHOLDS["new_unsupported_false_acceptance_max"])
    check("supported_corrupted_acceptance", corrupted["domain_accepted_fraction"], THRESHOLDS["supported_corrupted_accepted_fraction_min"], True)
    check("known_benchmark_corrupted_improvement", corrupted["gated_rmse_improvement_fraction"], THRESHOLDS["known_benchmark_corrupted_rmse_improvement_min"], True)
    check("known_benchmark_clean_replacement", clean["replacement_window_fraction"], THRESHOLDS["known_benchmark_clean_replacement_fraction_max"])
    check("known_session_rmse_ratio", known["gated_to_raw_RMSE_ratio"], THRESHOLDS["known_session_rmse_ratio_to_noisy_max"])
    non_noop = corrupted["actual_replacement_windows"] > 0 and corrupted["actually_changed_windows"] > 0
    checks["not_all_noop"] = {"passed": bool(non_noop), "actual_replacement_windows": corrupted["actual_replacement_windows"],
                              "actually_changed_windows": corrupted["actually_changed_windows"]}
    complete = unsupported.get("single_channel_500_window_count") == 5120 and set(unsupported.get("subjects", {})) == set(SUBJECTS)
    complete &= unsupported.get("domain_scored_window_count") == 5120
    complete &= all(x.get("single_channel_500_window_count") == 2560 and x.get("mother_trial_count") == 160
                    and x.get("domain_scored_window_count") == 2560
                    and x.get("domain_refused_raw_fallback_exact_original_units") is True for x in unsupported.get("subjects", {}).values())
    checks["complete_negative_coverage_and_exact_fallback"] = {"passed": bool(complete), "expected_windows": 5120}
    return {"domain_gate_research_acceptance_passed": all(x["passed"] for x in checks.values()),
            "scope": "combined_applicability_safety_research_only_known_benchmark_and_new_negative_holdout",
            "thresholds_predeclared": dict(THRESHOLDS), "checks": checks, **SAFETY}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reserved = False
    output = resolve(args.root.resolve(), args.output)
    try:
        root = args.root.resolve()
        require(args.check_only or (not output.exists() and not output.is_symlink()), "output exists; evaluation refuses overwrite/repeated holdout")
        prereg_path, lock_path = (resolve(root, p) for p in (args.preregistration, args.freeze_lock))
        checkpoint, denoiser, config = (resolve(root, p) for p in (args.checkpoint, args.denoiser_checkpoint, args.config))
        prereg = load_preregistration(prereg_path)
        frozen = validate_freeze(root, lock_path, prereg_path, prereg, checkpoint, denoiser, config)
        core, bundle = load_bundle(checkpoint, denoiser, config, args.device, args.threads)
        common = {"schema_version": 3, "model_version": core.MODEL_VERSION, **SAFETY,
                  "support_threshold": bundle.support_threshold, "threshold_comparison": ">", "threshold_changed": False,
                  "checkpoint_sha256": sha256(checkpoint), "denoiser_sha256": sha256(denoiser), "config_sha256": sha256(config),
                  "freeze_lock_sha256": sha256(lock_path), "preregistration_sha256": sha256(prereg_path),
                  "freeze_proof_verified": True, "device": args.device, "threads": args.threads,
                  "thresholds_predeclared": dict(THRESHOLDS), "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "support_probability_semantics": "uncalibrated applicability, NOT artifact probability or repair quality",
                  "known_v2_benchmark_not_new_holdout": True, "known_v1_stress_independent": False}
        if args.check_only:
            print(_dump({**common, "status": "freeze_and_weights_checked_no_holdout_read", "holdout_loaded": False, "inference_executed": False}))
            return 0
        reserve_output(output, frozen)
        reserved = True
        benchmark = evaluate_known_benchmark(core, bundle, resolve(root, Path(prereg["data"]["support_npz"])), args.batch_size)
        unsupported = evaluate_unsupported(core, bundle, root, prereg, args.batch_size)
        known = evaluate_known_session(core, bundle, resolve(root, args.regression_dataset), args.batch_size)
        require(all(sha256(Path(path)) == value for path, value in frozen.items()), "frozen/protected sources changed during evaluation")
        gate = research_gate(benchmark, unsupported, known)
        report = {**common, "status": "completed", "holdout_loaded": True, "v3_fresh_inference": True,
                  "sections": {"known_v2_benchmark": benchmark, "new_unsupported_holdout": unsupported, "known_v1_stress": known},
                  "gate": gate, "domain_gate_research_acceptance_passed": gate["domain_gate_research_acceptance_passed"]}
        output.write_text(_dump(report), encoding="utf-8")
        print(_dump({"status": "completed", "output": str(output), **gate}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError, AttributeError) as error:
        failure = {"status": "error", "error": str(error), **SAFETY, "thresholds_predeclared": dict(THRESHOLDS),
                   "reserved_attempt": reserved, "warning": "No automatic repeated holdout run; inspect reserved attempt before any explicit new experiment."}
        if reserved:
            try:
                output.write_text(_dump(failure), encoding="utf-8")
            except OSError:
                pass
        print(_dump(failure))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
