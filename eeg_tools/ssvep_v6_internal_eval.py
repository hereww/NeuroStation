"""Frozen v6 internal-only diagnostic orchestration; never acceptance/deployment.

Import and --check-only do not import torch or read NPZ/PT bytes. The shared
contract owns schema/hash/path safety, corruption, metrics and research gates.
This module owns the fixed diagnostic cohort, inference and report assembly.
There is deliberately no CPU fallback, input override, training or task FFT.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
from typing import Any

ROOT = Path(__file__).absolute().parents[1]
DEFAULT_OUTPUT_PREFIX = "analysis_outputs/ssvep_v6_internal_diagnostic_20261007"
DATASETS = ("Nakanishi2015", "Wearable")
SUBJECTS = (5, 6)
CONDITIONS = ("identity", "transient", "low_frequency_drift", "high_frequency_burst", "common_mode")
TRIALS = {"Nakanishi2015": 180, "Wearable": 240}
SHAPES = {"Nakanishi2015": (180, 1114, 8), "Wearable": (240, 710, 8)}
RATES = {"Nakanishi2015": 256, "Wearable": 250}
STARTS = {"Nakanishi2015": 38, "Wearable": 125}
WINDOW, BATCH_SIZE, DIAGNOSTIC_SEED = 256, 64, 2026107603
CASE_COUNT, REFERENCE_TRIAL_COUNT = 4200, 840
METRIC_KEYS = frozenset((
    "input_normalized_mse", "output_normalized_mse", "relative_reduction",
    "identity_normalized_change", "outcome", "protected_complex_energy_ratio",
    "protected_weighted_abs_phase_radians", "undefined_phase_bin_count",
))
PROVENANCE_KEYS = frozenset(("dataset", "subject_id", "trial_index", "condition",
                           "seed", "window_samples", "window_start_sample", "sample_rate_hz"))
LIMITATIONS = (
    "INTERNAL engineering diagnostic only: subjects 5-6 were used for internal epoch selection; not independent acceptance.",
    "Recorded reference is not real clean ground truth; only fixed synthetic corruption is evaluated.",
    "Rectangular window-level protected-band phase metrics do not guarantee full streaming/Hann-OLA phase preservation.",
    "No decoding/task FFT validation, clinical validity, or production acceptance is tested.",
    "Known outer-validation subjects 7-8 and holdout subjects 9-10 are not loaded or used for selection.",
    "Passing internal gates does not deploy the model; runtime and production validation remain false.",
)


def _contract():
    return importlib.import_module("eeg_tools.ssvep_v6_contract")


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise _contract().V6Error(message)


def _finite_json(value: Any) -> bool:
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_finite_json(v) for v in value)
    if type(value) is dict:
        return all(type(k) is str and _finite_json(v) for k, v in value.items())
    return False


def _read_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        _require(False, f"nonfinite JSON token: {value}")
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                        parse_constant=invalid)
    _require(type(result) is dict and _finite_json(result), "finite JSON object required")
    return result


def output_paths(root: Path, prefix: str, *, check_only: bool = False) -> tuple[str, str]:
    """Only new, workspace-relative v6 stems; shared safety rejects reparse parents."""
    _require(type(prefix) is str and not PureWindowsPath(prefix).is_absolute(), "relative output prefix required")
    _require(re.fullmatch(r"analysis_outputs/ssvep_v6[A-Za-z0-9_-]*", prefix) is not None,
             "output prefix must be analysis_outputs/ssvep_v6... with no traversal")
    stem = prefix + ("_check_only" if check_only else "")
    paths = (stem + ".json", stem + ".md")
    c = _contract()
    for relative in paths:
        c.safe_path(root, relative)
    c.assert_new_outputs(root, paths)
    return paths


def diagnostic_seed(dataset: str, subject_id: int, trial_index: int, condition: str) -> int:
    _require(type(dataset) is str and dataset in DATASETS, "unknown diagnostic dataset")
    _require(type(subject_id) is int and subject_id in SUBJECTS, "only diagnostic subjects 5-6 permitted")
    _require(type(trial_index) is int and 0 <= trial_index < TRIALS[dataset], "invalid diagnostic trial")
    _require(type(condition) is str and condition in CONDITIONS, "invalid diagnostic condition")
    return (DIAGNOSTIC_SEED + DATASETS.index(dataset) * 10000000
            + subject_id * 100000 + trial_index * 10 + CONDITIONS.index(condition))


def validate_cases(rows: list[dict]) -> dict:
    """Exact 4,200-case Cartesian cohort, finite metrics, and phase-verifiable status."""
    _require(type(rows) is list, "cases must be a list")
    seen = set()
    identity_count = undefined_cases = undefined_bins = 0
    for row in rows:
        _require(type(row) is dict and set(row) == PROVENANCE_KEYS | METRIC_KEYS,
                 "case schema mismatch; waveforms/labels are forbidden")
        _require(_finite_json(row), "case has nonfinite or non-JSON fields")
        d, s, t, condition = (row[k] for k in ("dataset", "subject_id", "trial_index", "condition"))
        expected_seed = diagnostic_seed(d, s, t, condition)
        key = d, s, t, condition
        _require(key not in seen, "duplicate diagnostic case")
        seen.add(key)
        for name, expected in (("seed", expected_seed), ("window_samples", WINDOW),
                               ("window_start_sample", STARTS[d]), ("sample_rate_hz", RATES[d])):
            _require(type(row[name]) is int and row[name] == expected, f"case {name} mismatch")
        for name in ("input_normalized_mse", "output_normalized_mse"):
            _require(type(row[name]) in (int, float) and row[name] >= 0, f"invalid {name}")
        _require(type(row["outcome"]) is str and bool(row["outcome"]), "invalid case outcome")
        bins = row["undefined_phase_bin_count"]
        _require(type(bins) is int and bins >= 0, "invalid phase bin count")
        reduction = row["relative_reduction"]
        _require(reduction is None or type(reduction) in (int, float), "invalid relative reduction")
        phase_keys = ("protected_complex_energy_ratio", "protected_weighted_abs_phase_radians")
        for name in phase_keys:
            _require(row[name] is None or (type(row[name]) in (int, float) and row[name] >= 0),
                     f"invalid {name}")
        change = row["identity_normalized_change"]
        if condition == "identity":
            _require(type(change) in (int, float) and change >= 0, "invalid identity normalized change")
            _require(reduction is None, "identity reduction must be null")
            _require(row["protected_weighted_abs_phase_radians"] is None
                     or row["protected_weighted_abs_phase_radians"] <= math.pi, "phase exceeds pi")
            _require(bins == 0 or row["protected_weighted_abs_phase_radians"] is None,
                     "undefined phase must remain null")
            identity_count += 1
            undefined_bins += bins
            undefined_cases += int(bins > 0 or any(row[k] is None for k in phase_keys))
        else:
            _require(change is None and bins == 0 and all(row[k] is None for k in phase_keys),
                     "corrupted case identity/phase fields must be null/count zero")
    expected = {(d, s, t, k) for d in DATASETS for s in SUBJECTS
                for t in range(TRIALS[d]) for k in CONDITIONS}
    _require(seen == expected and len(rows) == CASE_COUNT, "missing diagnostic cases / incomplete cohort")
    return {"complete_cohort": True, "case_count": len(rows),
            "reference_trial_count": REFERENCE_TRIAL_COUNT, "phase_case_count": identity_count,
            "undefined_phase_case_count": undefined_cases, "undefined_phase_bin_count": undefined_bins,
            "phase_verifiable": identity_count == REFERENCE_TRIAL_COUNT and undefined_cases == 0}


def summarize_diagnostic(rows: list[dict], cfg: dict) -> dict:
    """Every grouping delegates to the ONE shared numerical summarizer/gate helper."""
    cohort = validate_cases(rows)
    c = _contract()
    groups = {name: defaultdict(list) for name in (
        "dataset", "condition", "dataset_condition", "subject", "subject_condition")}
    for row in rows:
        d, s, k = row["dataset"], row["subject_id"], row["condition"]
        for name, key in (("dataset", d), ("condition", k), ("dataset_condition", f"{d}/{k}"),
                          ("subject", f"{d}/{s}"), ("subject_condition", f"{d}/{s}/{k}")):
            groups[name][key].append(row)
    aggregates = {"global": c.summarize_cases(rows, allow_mixed_datasets=True)}
    for name, buckets in groups.items():
        if name == "condition":
            aggregates[name] = {key: c.summarize_cases(bucket, allow_mixed_datasets=True)
                                for key, bucket in sorted(buckets.items())}
        else:
            aggregates[name] = {key: c.summarize_cases(bucket)
                                for key, bucket in sorted(buckets.items())}
    dataset_gates = {d: c.gate_summary(aggregates["dataset"][d], cfg) for d in DATASETS}
    for gates in dataset_gates.values():
        _require(type(gates) is dict and type(gates.get("all_passed")) is bool,
                 "invalid shared gate result")
    return {**cohort, "aggregates": aggregates, "dataset_gates": dataset_gates,
            "all_internal_gates_passed": (all(g["all_passed"] for g in dataset_gates.values())
                                          and cohort["complete_cohort"] and cohort["phase_verifiable"])}


def _configure_cuda_environment(cfg: dict) -> None:
    expected = cfg["training"]["cublas_workspace_config"]
    _require(os.environ.get("CUBLAS_WORKSPACE_CONFIG", expected) == expected,
             "conflicting CUBLAS_WORKSPACE_CONFIG; frozen CUDA policy refused")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected


def cuda_preflight(torch: Any, cfg: dict) -> dict:
    """Execution-only: no fallback or autocast; fixed preregistered numerical policy."""
    _configure_cuda_environment(cfg)
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1,
             "CUDA:0 mandatory; CPU fallback refused")
    torch.set_num_threads(cfg["training"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    probe = torch.zeros((1,), device="cuda:0", dtype=torch.float32)
    _require(bool(torch.isfinite(probe).all()), "CUDA probe nonfinite")
    torch.cuda.synchronize(0)
    return {"device": "cuda:0", "device_name": torch.cuda.get_device_name(0),
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "amp": False, "allow_tf32": False, "deterministic_algorithms": True}


def _freeze_metadata(state: Any) -> dict:
    c = _contract()
    lock = _read_json(c.safe_path(state.root, c.EVAL_LOCK_RELATIVE))
    _require(lock.get("training_performed") is True, "frozen checkpoint must have prior training")
    training_sha = c.file_sha256(state.root, c.TRAIN_LOCK_RELATIVE)
    _require(lock.get("training_lock_sha256") == training_sha, "training lock SHA mismatch")
    return {"config_sha256": c.file_sha256(state.root, c.CONFIG_RELATIVE),
            "preregistration_sha256": c.file_sha256(state.root, c.PREREG_RELATIVE),
            "training_lock_sha256": training_sha, "evaluation_lock_sha256": state.lock_sha256,
            "checkpoint_sha256": lock.get("checkpoint_sha256"),
            "training_report_sha256": c.file_sha256(state.root, c.TRAIN_REPORT_RELATIVE)}


def _model_digest(models: dict) -> str:
    """In-memory tensor state, not a second binary load; checked around inference."""
    h = hashlib.sha256()
    for dataset in DATASETS:
        for name, tensor in sorted(models[dataset].state_dict().items()):
            cpu = tensor.detach().cpu().contiguous()
            h.update(json.dumps([dataset, name, str(cpu.dtype), list(cpu.shape)]).encode())
            h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def load_frozen_models(state: Any, torch: Any, validate_checkpoint: Any, metadata: dict) -> dict:
    c = _contract()
    c.verify_frozen(state)  # Before the ONLY PT deserialization.
    digest = c.file_sha256(state.root, c.CHECKPOINT_RELATIVE)
    _require(type(metadata["checkpoint_sha256"]) is str and digest == metadata["checkpoint_sha256"],
             "evaluation checkpoint byte SHA mismatch")
    ck = torch.load(c.safe_path(state.root, c.CHECKPOINT_RELATIVE),
                    map_location="cpu", weights_only=True)
    models = validate_checkpoint(ck, state.config, metadata["training_lock_sha256"])
    _require(type(models) is dict and set(models) == set(DATASETS), "CPU model domains mismatch")
    for model in models.values():
        _require(all(p.device.type == "cpu" for p in model.parameters()), "checkpoint validation must remain CPU")
        model.eval()
        model.requires_grad_(False)
        model.to(device="cuda:0", dtype=torch.float32)
        _require(not model.training and all(not p.requires_grad for p in model.parameters()),
                 "model must be eval with requires_grad=False")
    c.verify_frozen(state)
    return models


def _validate_records(records: list) -> list:
    import numpy as np
    _require(type(records) is list and len(records) == 4, "exactly four internal records required")
    expected = {(d, s) for d in DATASETS for s in SUBJECTS}
    seen = set()
    for record in records:
        _require(type(record.dataset) is str and type(record.subject_id) is int,
                 "illegal record identity type")
        key = record.dataset, record.subject_id
        _require(key in expected and key not in seen, "duplicate or prohibited record subject")
        seen.add(key)
        _require(type(record.sample_rate_hz) is int and record.sample_rate_hz == RATES[record.dataset],
                 "record native sample rate mismatch")
        _require(isinstance(record.signal, np.ndarray) and record.signal.shape == SHAPES[record.dataset]
                 and record.signal.dtype == np.float64 and np.isfinite(record.signal).all(),
                 "record must be complete finite float64 trial/time/channel tensor")
    _require(seen == expected, "incomplete internal records")
    return sorted(records, key=lambda r: (DATASETS.index(r.dataset), r.subject_id))


def infer_cases(state: Any, records: list, models: dict, torch: Any) -> list[dict]:
    """Fixed order, batch64 within domain/subject, diagnostic epoch0, no labels."""
    import numpy as np
    c = _contract()
    records = _validate_records(records)
    _require(state.config["seeds"]["diagnostic"] == DIAGNOSTIC_SEED,
             "fixed diagnostic seed mismatch")
    rows = []
    with torch.inference_mode():
        for record in records:
            pending = []
            def flush():
                if not pending:
                    return
                inputs = np.stack([item[2] for item in pending])
                pred = models[record.dataset](torch.from_numpy(inputs).to(device="cuda:0", dtype=torch.float32))
                _require(pred.dtype == torch.float32 and tuple(pred.shape) == tuple(inputs.shape)
                         and not pred.requires_grad and bool(torch.isfinite(pred).all()),
                         "invalid inference prediction shape/dtype/finite/gradient")
                predictions = pred.cpu().numpy()
                for (provenance, ref, norm, noisy, rms), prediction in zip(pending, predictions):
                    out = c.restore_window(noisy, norm, prediction, rms)
                    _require(out.dtype == np.float64 and out.shape == (8, WINDOW)
                             and np.isfinite(out).all(), "invalid float64 restoration")
                    metrics = c.case_metrics(ref, noisy, out, identity=provenance["condition"] == "identity",
                                             sample_rate_hz=record.sample_rate_hz)
                    _require(type(metrics) is dict and set(metrics) == METRIC_KEYS, "shared metrics schema mismatch")
                    rows.append({**provenance, **metrics})
                pending.clear()
            for trial in range(TRIALS[record.dataset]):
                start = STARTS[record.dataset]
                ref = np.ascontiguousarray(record.signal[trial, start:start + WINDOW, :].T)
                for condition in CONDITIONS:
                    noisy, seed = c.corrupt_case(ref, record.dataset, record.subject_id, trial,
                                                condition, state.config, purpose="diagnostic", epoch=0)
                    _require(type(seed) is int and seed == diagnostic_seed(
                        record.dataset, record.subject_id, trial, condition), "corruption diagnostic seed mismatch")
                    norm, mean, rms = c.normalize_window(noisy)
                    _require(noisy.dtype == np.float64 and noisy.shape == (8, WINDOW) and np.isfinite(noisy).all(),
                             "invalid noisy float64 window")
                    _require(norm.dtype == np.float32 and norm.shape == (8, WINDOW) and np.isfinite(norm).all(),
                             "invalid float32 normalized window")
                    _require(mean.shape == rms.shape == (8, 1) and mean.dtype == rms.dtype == np.float64
                             and np.isfinite(mean).all() and np.isfinite(rms).all() and (rms > 0).all(),
                             "invalid normalization mean/RMS")
                    provenance = {"dataset": record.dataset, "subject_id": record.subject_id,
                                  "trial_index": trial, "condition": condition, "seed": seed,
                                  "window_start_sample": start, "window_samples": WINDOW,
                                  "sample_rate_hz": RATES[record.dataset]}
                    pending.append((provenance, ref, norm, noisy, rms))
                    if len(pending) == BATCH_SIZE:
                        flush()
            flush()
    validate_cases(rows)
    return rows


def _base_report(state: Any, metadata: dict) -> dict:
    c = _contract()
    return {"schema_version": 1, "status": "formal", "model_version": c.MODEL_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "v6_internal_diagnostic", "diagnostic_seed": DIAGNOSTIC_SEED,
            "diagnostic_epoch": 0, "batch_size": BATCH_SIZE,
            "inference": "float32_forward_float64_window_residual_restore_no_hann_ola",
            "frozen": metadata, "preflight": state.report(), "scope": {
                "training_performed_this_evaluation": False, "checkpoint_prior_training": True,
                "epoch_selection_performed": False, "parameter_selection_performed": False,
                "outer_validation_waveforms_loaded": False, "holdout_waveforms_loaded": False,
                "outer_validation_used_for_training_or_selection": False,
                "clean_ground_truth_used": False, "independent_acceptance": False,
                "runtime_enabled": False, "production_validated": False},
            "deployment": {"status": "blocked", "runtime_enabled": False, "production_validated": False,
                           "pending": ["known_validation_one_time_locked_screen", "holdout_program"]},
            "aggregate_policy": {
                "mixed_aggregates": ["global", "condition"],
                "mixed_aggregates_descriptive_normalized_only": True,
                "native_raw_power_pooled": False,
                "gates_applied_only_to_dataset": True,
                "independent_acceptance_claimed": False},
            "limitations": list(LIMITATIONS)}


def build_check_only_report(state: Any) -> dict:
    """JSON/source/stat-only evidence; cannot claim PT/NPZ schema or phase checks."""
    c = _contract()
    c.verify_frozen(state)
    report = _base_report(state, _freeze_metadata(state))
    report.update({"check_only": True, "inference_performed": False, "checkpoint_loaded": False,
                   "waveforms_loaded": False, "cuda_initialized": False, "gpu": None,
                   "checks": {"json_schema": "validated", "frozen_nonbinary_hashes": "validated",
                              "binary_hashes": "not_executed", "checkpoint_schema": "not_executed",
                              "npz_schema": "not_executed", "phase": "not_executed"},
                   "case_count": 0, "reference_trial_count": 0, "phase_case_count": 0,
                   "complete_cohort": False, "phase_verifiable": False,
                   "all_internal_gates_passed": False, "dataset_gates": {}, "aggregates": {}, "cases": []})
    c.verify_frozen(state)
    _require(_freeze_metadata(state) == report["frozen"], "freeze provenance changed during check-only")
    return report


def run_diagnostic(state: Any) -> dict:
    """Call only after main-session checkpoint training and evaluation freeze."""
    c = _contract()
    _require(state.stage == "internal_evaluation" and state.check_only is False,
             "full internal evaluation preflight required")
    c.verify_frozen(state)
    metadata = _freeze_metadata(state)
    _configure_cuda_environment(state.config)  # Must precede torch import / CUDA initialization.
    torch = importlib.import_module("torch")
    model_module = importlib.import_module("eeg_tools.ssvep_denoise_v6")
    gpu = cuda_preflight(torch, state.config)
    models = load_frozen_models(state, torch, model_module.validate_checkpoint, metadata)
    before = _model_digest(models)
    c.verify_frozen(state)  # Immediately before the selected-four-record loader.
    records = c.load_training_records(state)
    rows = infer_cases(state, records, models, torch)
    torch.cuda.synchronize(0)
    _require(all(not model.training and all(not p.requires_grad for p in model.parameters())
                 for model in models.values()), "inference model eval/gradient flags changed")
    after = _model_digest(models)
    _require(before == after, "model state mutated during evaluation")
    c.verify_frozen(state)
    _require(_freeze_metadata(state) == metadata, "freeze provenance changed during evaluation")
    report = _base_report(state, metadata)
    report.update({"check_only": False, "inference_performed": True, "checkpoint_loaded": True,
                   "waveforms_loaded": True, "cuda_initialized": True, "gpu": gpu,
                   "checks": {"json_schema": "validated", "frozen_needed_hashes": "validated",
                              "fit_binary_hashes": "deferred_not_used", "checkpoint_schema": "validated",
                              "selected_npz_schema": "validated", "phase": "executed"},
                   "model_state_sha256_before": before, "model_state_sha256_after": after,
                   **summarize_diagnostic(rows, state.config), "cases": rows})
    return report


def render_markdown(report: dict) -> bytes:
    """Human evidence, not an acceptance certificate; JSON is the complete case log."""
    lines = ["# SSVEP v6 internal diagnostic", "", "**Formal protocol; INTERNAL ONLY; deployment BLOCKED.**", "",
             f"- Check-only: {report['check_only']}", f"- Inference performed: {report['inference_performed']}",
             f"- Cases: {report['case_count']}", f"- Phase cases: {report['phase_case_count']}",
             f"- Phase verifiable: {report['phase_verifiable']}",
             f"- All internal gates passed: {report['all_internal_gates_passed']}",
             "- No training/selection in this evaluation; checkpoint has prior training.",
             "- Runtime enabled: false; production validated: false.",
             "- Mixed global/condition aggregates are descriptive normalized-only; no native raw power pooling.",
             "- Gates apply only to per-dataset summaries; no independent acceptance claim.",
             "", "## Frozen evidence", ""]
    lines += [f"- {k}: `{v}`" for k, v in sorted(report["frozen"].items())]
    lines += ["", "## Checks", ""] + [f"- {k}: {v}" for k, v in report["checks"].items()]
    lines += ["", "## Per-dataset internal gates", "", "```json",
              json.dumps(report["dataset_gates"], ensure_ascii=False, indent=2, allow_nan=False), "```",
              "", "## Limitations", ""] + [f"- {x}" for x in report["limitations"]]
    lines += ["", "Pending: known_validation_one_time_locked_screen and separately controlled holdout program.", ""]
    return "\n".join(lines).encode("utf-8")


def evaluate(root: Path = ROOT, *, check_only: bool = False,
             output_prefix: str = DEFAULT_OUTPUT_PREFIX) -> dict:
    """Reject output collisions BEFORE preflight, torch import, GPU or binary reads."""
    paths = output_paths(root, output_prefix, check_only=check_only)
    c = _contract()
    state = c.preflight(root=root, stage="internal_evaluation", check_only=check_only)
    report = build_check_only_report(state) if check_only else run_diagnostic(state)
    _require(_finite_json(report), "report contains nonfinite/non-JSON values")
    c.verify_frozen(state)
    _require(_freeze_metadata(state) == report["frozen"], "freeze provenance changed before publication")
    c.assert_new_outputs(root, paths)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    c.publish_bundle(root, {paths[0]: payload, paths[1]: render_markdown(report)})
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen v6 INTERNAL diagnostic; no acceptance/deployment")
    parser.add_argument("--check-only", action="store_true", help="JSON/source/stat-only; never read PT/NPZ or initialize CUDA")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX,
                        help="workspace-relative analysis_outputs/ssvep_v6... output stem, no overwrite")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate(check_only=args.check_only, output_prefix=args.output_prefix)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"v6 internal diagnostic refused: {exc}")
        return 2
    print(json.dumps({k: report[k] for k in ("check_only", "case_count", "phase_case_count",
                                           "all_internal_gates_passed")}, allow_nan=False))
    return 0
