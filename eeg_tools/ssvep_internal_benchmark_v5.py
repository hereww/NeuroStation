"""Frozen v5 INTERNAL synthetic-corruption diagnostic, NOT acceptance or training.

Recorded reference is NOT clean truth. Window restoration is NOT Hann OLA.
Extra finite descriptive protocol/lock JSON fields are allowed; required values
and types are strict. All supplied freeze-lock boolean flags must be false.
bound_files/files contain EXACTLY the fixed 14/18 paths.
Protocol SHA is supplied by the main session; the main session writes the lock
AFTER reviewing these three new files. No freeze inputs are written here.
check_only requires the lock, hashes JSON/source/sidecars, and ONLY stats NPZ/PT;
it neither reads binary bytes nor initializes CUDA. Deferred checks are explicit.
The lock SHA is captured/rechecked before the legacy loader and after inference.
Atomic no-clobber publication uses os.link, never overwrite/replace. Each output
is atomic, not a claimed two-file transaction; second-link failure rolls back
only the first link created by that operation. No external source path is used.
"""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import stat
import tempfile
from typing import Any
import numpy as np
from eeg_tools import ssvep_denoise_v5 as core

ROOT = Path(__file__).absolute().parents[1]
PROTOCOL_PATH = "analysis_outputs/ssvep_internal_v5_benchmark_preregistration_20261007.json"
LOCK_PATH = "analysis_outputs/ssvep_internal_v5_benchmark_lock_20261007.json"
PROTOCOL_SHA256 = "e7d231ddb6a19ae9ccbee12407663a853c1a84f98a6910730b5d67fe5abccc97"
CODE_PATHS = ("eeg_tools/ssvep_internal_benchmark_v5.py", "scripts/benchmark_ssvep_internal_v5.py", "tests/test_ssvep_internal_benchmark_v5.py")
CORE_PATH = "eeg_tools/ssvep_denoise_v5.py"
SAFETY_PATH = "eeg_tools/ssvep_reference_v5.py"
CONFIG_PATH = "configs/learned_denoiser_v5_ssvep.json"
CHECKPOINT_PATH = "models/denoiser_v5_ssvep.pt"
REFERENCE_DIR = "datasets/processed/ssvep_reference_v5_20261007"
MANIFEST_PATH = REFERENCE_DIR + "/manifest.json"
PARTITION_PATH = "analysis_outputs/ssvep_reference_v5_partition_lock_20261007.json"
DATASETS = ("Nakanishi2015", "Wearable")
SUBJECTS = (5, 6)
CONDITIONS = ("identity", "transient", "low_frequency_drift", "high_frequency_burst", "common_mode")
STARTS = {"Nakanishi2015": 38, "Wearable": 125}
RATES = {"Nakanishi2015": 256, "Wearable": 250}
SHAPES = {"Nakanishi2015": (180, 1114, 8), "Wearable": (240, 710, 8)}
WINDOW, BATCH_SIZE, SEED, CASE_COUNT, TRIAL_COUNT = 256, 64, 2026100701, 4200, 840
SEED_FORMULA = "2026100701 + dataset_index*10000000 + subject_id*100000 + trial_index*10 + condition_index"
INFERENCE = "window_residual_restore_float64_no_ola"
SCOPE = dict.fromkeys(("training_performed", "epoch_selection_performed", "parameter_selection_performed", "outer_validation_waveforms_loaded", "holdout_waveforms_loaded", "raw_reference_is_clean_ground_truth", "runtime_enabled", "production_validated"), False)
NPZ_PATHS = tuple(f"{REFERENCE_DIR}/{d}/subject_{s:03d}.npz" for d in DATASETS for s in SUBJECTS)
SIDECAR_PATHS = tuple(p + ".manifest.json" for p in NPZ_PATHS)
INPUT_PATHS = (CORE_PATH, SAFETY_PATH, CONFIG_PATH, CHECKPOINT_PATH, MANIFEST_PATH, PARTITION_PATH) + NPZ_PATHS + SIDECAR_PATHS
FROZEN_PATHS = INPUT_PATHS + (PROTOCOL_PATH,) + CODE_PATHS
BINARY_PATHS = frozenset(NPZ_PATHS + (CHECKPOINT_PATH,))
LIMITATIONS = (
    "INTERNAL ONLY: subjects 5-6 previously participated in v5 epoch selection; not independent validation.",
    "Recorded reference is not clean ground truth; synthetic corruption is not a real artifact ground-truth dataset.",
    "WINDOW-LEVEL residual restoration, NO Hann OLA; NOT equivalent to the complete inference pipeline.",
    "No decoding/task preservation, true-clean EEG SNR/RMSE, clinical or production acceptance is tested.",
    "No training, new epoch selection, alpha tuning, or other parameter selection is performed.",
    "Does not overturn historical/frozen outer-validation conclusions; runtime remains disabled.",
    "Aggregates are dimensionless channel-normalized errors, never cross-domain native power.",
)

class InternalBenchmarkError(ValueError):
    """Fail closed before prohibited access or publication."""

def _require(ok: bool, message: str) -> None:
    if not ok:
        raise InternalBenchmarkError(message)

def _plain(value: Any) -> bool:
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_plain(x) for x in value)
    if type(value) is dict:
        return all(type(k) is str and _plain(v) for k, v in value.items())
    return False

def _exact(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(_exact(actual[k], v) for k, v in expected.items())
    if type(expected) is list:
        return len(actual) == len(expected) and all(_exact(a, b) for a, b in zip(actual, expected))
    return actual == expected

def _required(obj: dict, expected: dict, context: str) -> None:
    _require(type(obj) is dict and _plain(obj), f"{context}: finite JSON object required")
    for k, value in expected.items():
        _require(k in obj and _exact(obj[k], value), f"{context}: invalid {k}")

def safe_path(root: Path, value: str | Path) -> Path:
    """Reject lexical escapes/foreign drives BEFORE stat and all reparse chains."""
    text = str(value)
    _require(text != "" and ".." not in text.replace("\\", "/").split("/") and "\x00" not in text, "unsafe traversal/empty path")
    win = PureWindowsPath(text)
    _require(win.drive.casefold() != "d:", "D: access forbidden")
    _require(not win.drive or Path(text).is_absolute(), "foreign drive/UNC path forbidden")
    _require(not any(":" in x for x in win.parts[1:]), "alternate stream forbidden")
    if not win.drive:
        _require(":" not in text, "alternate stream/foreign drive forbidden")
    base = Path(os.path.abspath(root))
    _require(PureWindowsPath(str(base)).drive.casefold() != "d:", "D: root forbidden")
    _require(".." not in str(root).replace("\\", "/").split("/"), "unsafe root")
    path = Path(text) if Path(text).is_absolute() else base / text
    path = Path(os.path.abspath(path))
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise InternalBenchmarkError("external workspace path forbidden") from exc
    for entry in list(reversed(path.parents)) + [path]:
        try:
            info = entry.lstat()
        except FileNotFoundError:
            continue
        _require(not stat.S_ISLNK(info.st_mode) and not (getattr(info, "st_file_attributes", 0) & 0x400), "symlink/reparse path forbidden")
        if entry != path:
            _require(stat.S_ISDIR(info.st_mode), "path ancestor is not a directory")
    return path

def _stamp(root: Path, relative: str) -> tuple:
    st = safe_path(root, relative).lstat()
    _require(stat.S_ISREG(st.st_mode) and st.st_size > 0, f"regular nonempty file required: {relative}")
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns

def file_sha256(root: Path, relative: str) -> str:
    _require(relative in FROZEN_PATHS or relative == LOCK_PATH, "unapproved hash target")
    before = _stamp(root, relative)
    with safe_path(root, relative).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    _require(before == _stamp(root, relative), f"file changed during hashing: {relative}")
    return digest

def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result

def read_json(root: Path, relative: str) -> dict:
    _require(relative in FROZEN_PATHS or relative == LOCK_PATH, "unapproved metadata path")
    _require(relative.endswith(".json"), "metadata must be JSON")
    before = _stamp(root, relative)
    try:
        value = json.loads(safe_path(root, relative).read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InternalBenchmarkError(f"invalid JSON: {relative}") from exc
    _require(type(value) is dict and _plain(value), f"finite JSON object required: {relative}")
    _require(before == _stamp(root, relative), f"metadata changed during reading: {relative}")
    return value

def _bindings(value: Any, paths: tuple[str, ...], context: str) -> None:
    _require(type(value) is dict and set(value) == set(paths), f"{context}: exact fixed path bindings required")
    _require(all(type(v) is str and re.fullmatch(r"[0-9a-f]{64}", v) for v in value.values()), f"{context}: invalid SHA256")

def validate_preregistration(protocol: dict) -> None:
    """Extra descriptive JSON allowed; metrics_definition is nonempty dict/string.

    Computation is fixed by reviewed code, not parsed/executed from that field.
    """
    _required(protocol, {
        "schema_version": 1, "status": "formal", "experiment": "ssvep_v5_internal_synthetic_diagnostic",
        "subjects": list(SUBJECTS), "datasets": list(DATASETS), "conditions": list(CONDITIONS),
        "window_samples": WINDOW, "start_samples": STARTS, "sample_rates_hz": RATES,
        "seed": SEED, "seed_formula": SEED_FORMULA, "batch_size": BATCH_SIZE,
        "device": "cuda:0", "normalization_epsilon": 1e-8, "inference": INFERENCE, "scope": SCOPE,
    }, "preregistration")
    _require(core.EPSILON == 1e-8, "legacy normalization epsilon changed")
    metrics = protocol.get("metrics_definition")
    _require(type(metrics) in (str, dict) and bool(metrics), "metrics_definition required")
    for key, value in (("expected_case_count", CASE_COUNT), ("expected_reference_trials", TRIAL_COUNT)):
        if key in protocol:
            _require(_exact(protocol[key], value), f"invalid {key}")
    _bindings(protocol.get("bound_files"), INPUT_PATHS, "preregistration")
    _require(protocol["bound_files"][MANIFEST_PATH] == core.REFERENCE_MANIFEST_SHA256, "legacy reference manifest binding mismatch")
    _require(protocol["bound_files"][PARTITION_PATH] == core.PARTITION_LOCK_SHA256, "legacy partition binding mismatch")

def validate_lock(lock: dict, protocol: dict, protocol_sha256: str) -> None:
    _required(lock, {"schema_version": 1, "status": "formal", "stage": "frozen_before_internal_waveform_read", "expected_case_count": CASE_COUNT}, "freeze lock")
    _bindings(lock.get("files"), FROZEN_PATHS, "freeze lock")
    _require(lock["files"][PROTOCOL_PATH] == protocol_sha256 == PROTOCOL_SHA256, "frozen protocol SHA mismatch")
    for path, digest in protocol["bound_files"].items():
        _require(lock["files"][path] == digest, f"protocol/lock binding conflict: {path}")
    for key in set(SCOPE) | {"inference_performed", "waveforms_loaded", "checkpoint_loaded", "cuda_initialized", "clean_ground_truth_used", "outer_validation_used_for_selection"}:
        if key in lock:
            _require(lock[key] is False, f"freeze lock unsafe flag: {key}")
    if "scope" in lock:
        _require(_exact(lock["scope"], SCOPE), "freeze lock unsafe scope")
    def false_flags(value):
        if type(value) is bool:
            return value is False
        if type(value) is dict:
            return all(false_flags(v) for v in value.values())
        if type(value) is list:
            return all(false_flags(v) for v in value)
        return True
    _require(false_flags(lock), "freeze lock contains true boolean flag")

@dataclass(frozen=True)
class Preflight:
    root: Path
    protocol: dict
    lock: dict
    config: dict
    lock_sha256: str
    protocol_sha256: str
    stamps: dict
    verified_hashes: dict
    deferred_hashes: tuple[str, ...]
    selected: tuple[dict, ...]
    check_only: bool

    def report(self) -> dict:
        return {"schema_version": 1, "status": "formal", "mode": "check_only" if self.check_only else "preflight",
                "inference_performed": False, "checkpoint_loaded": False, "waveforms_loaded": False, "cuda_initialized": False,
                "protocol_sha256": self.protocol_sha256, "freeze_lock_sha256": self.lock_sha256,
                "verified_hashes": self.verified_hashes, "deferred_binary_hash_checks": list(self.deferred_hashes),
                "checkpoint_schema_state_dict_checks": "not_executed", "waveform_value_shape_checks": "not_executed", "cuda_preflight": "not_executed",
                "expected_reference_trials": TRIAL_COUNT, "expected_case_count": CASE_COUNT, "case_count": 0,
                "scope": dict(SCOPE), "limitations": list(LIMITATIONS)}

def _catalog(root: Path, protocol: dict) -> tuple[dict, ...]:
    manifest = read_json(root, MANIFEST_PATH)
    _required(manifest, {"schema_version": 1, "status": "formal", "reference_version": "ssvep_reference_v5", "build_completed": True,
                         "source_lock_file_sha256": core.PARTITION_LOCK_SHA256, "raw_reference_is_clean_ground_truth": False,
                         "runtime": {"enabled": False, "production_validated": False}}, "reference manifest")
    partition = read_json(root, PARTITION_PATH)
    _required(partition, {"schema_version": 1, "status": "formal", "stage": "reference_identity_freeze",
                          "partitions": {"train": [1, 2, 3, 4, 5, 6], "validation": [7, 8], "holdout": [9, 10]},
                          "raw_reference_is_clean_ground_truth": False,
                          "runtime": {"enabled": False, "production_validated": False}}, "partition lock")
    rows = manifest.get("materialized_subjects")
    _require(type(rows) is list, "reference catalog missing")
    identities, selected = set(), {}
    for row in rows:
        _require(type(row) is dict and row.get("dataset") in DATASETS and type(row.get("subject_id")) is int, "illegal catalog identity")
        domain, sid = row["dataset"], row["subject_id"]
        _require(sid in range(1, 9), "illegal materialized subject")
        key = domain, sid
        _require(key not in identities, "duplicate catalog identity")
        identities.add(key)
        if sid not in SUBJECTS:
            continue  # NO path/stat/sidecar/NPZ/source access for any other subject.
        relative = f"{REFERENCE_DIR}/{domain}/subject_{sid:03d}.npz"
        _required(row, {"split": "train", "relative_path": f"{domain}/subject_{sid:03d}.npz", "trials": SHAPES[domain][0],
                        "npz_sha256": protocol["bound_files"][relative], "manifest_sha256": protocol["bound_files"][relative + ".manifest.json"]}, "internal catalog row")
        side = read_json(root, relative + ".manifest.json")
        _required(side, {"schema_version": 1, "status": "formal", "reference_version": "ssvep_reference_v5", "dataset": domain, "subject_id": sid,
                         "split": "train", "reference_shape": list(SHAPES[domain]), "reference_dtype": "float64",
                         "native_sample_rate_hz": RATES[domain], "onset_sample_zero_based": STARTS[domain],
                         "source_lock_file_sha256": core.PARTITION_LOCK_SHA256, "raw_reference_is_clean_ground_truth": False,
                         "runtime_enabled": False, "npz_sha256": row["npz_sha256"], "materialized_trials": SHAPES[domain][0]}, "internal sidecar")
        selected[key] = {"dataset": domain, "subject_id": sid, "npz_sha256": row["npz_sha256"], "sidecar_sha256": row["manifest_sha256"]}
    _require(set(selected) == {(d, s) for d in DATASETS for s in SUBJECTS}, "internal four-subject cohort incomplete")
    return tuple(selected[(d, s)] for d in DATASETS for s in SUBJECTS)

def preflight(root: Path = ROOT, *, check_only: bool = True) -> Preflight:
    """No loader/CUDA/deserialization; check-only defers all binary hashes."""
    root = safe_path(root, ".")
    lock_sha, protocol_sha = file_sha256(root, LOCK_PATH), file_sha256(root, PROTOCOL_PATH)
    _require(protocol_sha == PROTOCOL_SHA256, "main-session preregistration SHA mismatch")
    protocol, lock = read_json(root, PROTOCOL_PATH), read_json(root, LOCK_PATH)
    validate_preregistration(protocol)
    validate_lock(lock, protocol, protocol_sha)
    verified, deferred, stamps = {}, [], {}
    for relative in FROZEN_PATHS:
        stamps[relative] = _stamp(root, relative)
        if check_only and relative in BINARY_PATHS:
            deferred.append(relative)
            continue
        digest = file_sha256(root, relative)
        _require(digest == lock["files"][relative], f"frozen file SHA mismatch: {relative}")
        verified[relative] = digest
    config = read_json(root, CONFIG_PATH)
    core.validate_config(config)
    selected = _catalog(root, protocol)
    state = Preflight(root, protocol, lock, config, lock_sha, protocol_sha, stamps, verified, tuple(deferred), selected, check_only)
    _verify_frozen(state)
    return state

def _verify_frozen(state: Preflight) -> None:
    _require(file_sha256(state.root, LOCK_PATH) == state.lock_sha256, "freeze lock changed")
    for path in FROZEN_PATHS:
        _require(_stamp(state.root, path) == state.stamps[path], f"frozen file stat changed: {path}")
        if path not in state.deferred_hashes:
            _require(file_sha256(state.root, path) == state.lock["files"][path], f"frozen file changed: {path}")

def case_seed(dataset: str, subject_id: int, trial_index: int, condition: str) -> int:
    _require(dataset in DATASETS and type(subject_id) is int and subject_id in SUBJECTS, "illegal dataset/subject")
    _require(type(trial_index) is int and 0 <= trial_index < SHAPES[dataset][0], "illegal trial index")
    _require(condition in CONDITIONS, "illegal condition")
    return SEED + DATASETS.index(dataset) * 10000000 + subject_id * 100000 + trial_index * 10 + CONDITIONS.index(condition)

def _window(value: Any, name: str, *, dtype=None) -> np.ndarray:
    arr = np.asarray(value)
    _require(arr.shape == (8, WINDOW) and arr.dtype.kind == "f", f"{name}: finite [8,256] float window required")
    _require(dtype is None or arr.dtype == dtype, f"{name}: dtype mismatch")
    _require(np.isfinite(arr).all(), f"{name}: nonfinite values")
    return arr

def synthetic_corruption(reference: np.ndarray, dataset: str, subject_id: int, trial_index: int,
                         condition: str, config: dict) -> tuple[np.ndarray, int]:
    ref = _window(reference, "reference", dtype=np.float64)
    seed = case_seed(dataset, subject_id, trial_index, condition)
    core.validate_config(config)  # Bound original, NEVER the benchmark-only override.
    if condition == "identity":
        return ref.copy(), seed
    override = deepcopy(config)
    override["corruption"].update(families=[condition], identity_probability=0, combine_probability=0)
    noisy, is_identity, families = core.corrupt_window(ref, RATES[dataset], override, seed)
    _require(is_identity is False and families == [condition], "corruption family/identity mismatch")
    return _window(noisy, "noisy", dtype=np.float64), seed

def restore_window(noisy: np.ndarray, normalized_noisy: np.ndarray, prediction: np.ndarray,
                   noisy_rms: np.ndarray) -> np.ndarray:
    x = _window(noisy, "noisy", dtype=np.float64)
    n = _window(normalized_noisy, "normalized noisy", dtype=np.float32)
    p = _window(prediction, "prediction", dtype=np.float32)
    rms = np.asarray(noisy_rms)
    _require(rms.shape == (8, 1) and rms.dtype == np.float64 and np.isfinite(rms).all() and (rms > 0).all(), "invalid noisy RMS")
    return _window(x + (p.astype(np.float64) - n.astype(np.float64)) * rms, "restored", dtype=np.float64)

def case_metrics(reference: np.ndarray, noisy: np.ndarray, output: np.ndarray, *, identity: bool = False) -> dict:
    _require(type(identity) is bool, "identity must be boolean")
    ref = _window(reference, "reference", dtype=np.float64)
    x, y = _window(noisy, "noisy", dtype=np.float64), _window(output, "output", dtype=np.float64)
    _require(not identity or np.array_equal(x, ref), "identity input must equal recorded reference")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        denominator = np.mean((ref - ref.mean(axis=1, keepdims=True)) ** 2, axis=1, keepdims=True) + core.EPSILON
        a = float(np.mean((x - ref) ** 2 / denominator))
        b = float(np.mean((y - ref) ** 2 / denominator))
    _require(np.isfinite(denominator).all() and math.isfinite(a) and math.isfinite(b), "nonfinite metric")
    reduction = (a - b) / a if a > 0 and not identity else None
    _require(reduction is None or math.isfinite(reduction), "nonfinite relative reduction")
    return {"input_normalized_mse": a, "output_normalized_mse": b, "relative_reduction": reduction,
            "identity_normalized_change": b if identity else None,
            "outcome": "win" if b < a else "loss" if b > a else "tie"}

def _validate_case(row: dict) -> tuple:
    _require(type(row) is dict and _plain(row), "case must be finite metadata")
    allowed = {"dataset", "subject_id", "trial_index", "condition", "seed", "window_start_sample", "window_samples", "sample_rate_hz",
               "input_normalized_mse", "output_normalized_mse", "relative_reduction", "identity_normalized_change", "outcome"}
    _require(set(row) == allowed, "case schema mismatch (waveforms/predictions/labels prohibited)")
    d, s, t, c = (row[k] for k in ("dataset", "subject_id", "trial_index", "condition"))
    seed = case_seed(d, s, t, c)
    _required(row, {"seed": seed, "window_start_sample": STARTS[d], "window_samples": WINDOW, "sample_rate_hz": RATES[d]}, "case provenance")
    a, b = row["input_normalized_mse"], row["output_normalized_mse"]
    _require(type(a) is float and type(b) is float and a >= 0 and b >= 0, "invalid normalized metrics")
    expected = (a - b) / a if a > 0 and c != "identity" else None
    _require(_exact(row["relative_reduction"], expected) and (expected is None or math.isfinite(expected)), "invalid case reduction")
    _require(_exact(row["identity_normalized_change"], b if c == "identity" else None), "invalid identity change")
    _require(c != "identity" or a == 0.0, "nonzero identity input error")
    _require(row["outcome"] == ("win" if b < a else "loss" if b > a else "tie"), "invalid outcome")
    return d, s, t, c

def _summary(rows: list[dict]) -> dict:
    a = float(np.mean([r["input_normalized_mse"] for r in rows], dtype=np.float64))
    b = float(np.mean([r["output_normalized_mse"] for r in rows], dtype=np.float64))
    identities = [r["identity_normalized_change"] for r in rows if r["condition"] == "identity"]
    result = {"case_count": len(rows), "mean_input_normalized_mse": a, "mean_output_normalized_mse": b,
              "ratio_of_mean_reduction": (a - b) / a if a > 0 else None,
              "identity_case_count": len(identities), "mean_identity_normalized_change": float(np.mean(identities)) if identities else None,
              "win_count": sum(r["outcome"] == "win" for r in rows), "loss_count": sum(r["outcome"] == "loss" for r in rows),
              "tie_count": sum(r["outcome"] == "tie" for r in rows)}
    _require(_plain(result), "nonfinite aggregate")
    return result

def aggregate_cases(cases: list[dict], *, require_complete: bool = True) -> dict:
    _require(type(cases) is list and bool(cases), "empty/malformed cases")
    identities, groups = set(), defaultdict(list)
    for row in cases:
        key = _validate_case(row)
        _require(key not in identities, "duplicate case identity")
        identities.add(key)
        d, s, _, c = key
        for group in (("subject", f"{d}:subject_{s:03d}"), ("dataset", d), ("condition", c),
                      ("dataset_condition", f"{d}:{c}"), ("subject_condition", f"{d}:subject_{s:03d}:{c}")):
            groups[group].append(row)
    if require_complete:
        expected = {(d, s, t, c) for d in DATASETS for s in SUBJECTS for t in range(SHAPES[d][0]) for c in CONDITIONS}
        _require(identities == expected and len(cases) == CASE_COUNT, "incomplete fixed 4200-case cohort")
    result = {"unit": "dimensionless channel-normalized recorded-reference error", "global": _summary(cases)}
    for level in ("subject", "dataset", "condition", "dataset_condition", "subject_condition"):
        result[level] = {key: _summary(rows) for (lev, key), rows in sorted(groups.items()) if lev == level}
    return result

def validate_checkpoint(checkpoint: dict, config: dict) -> dict:
    """CPU-only strict schema/state validation before any model reaches CUDA."""
    torch = core.torch
    _require(torch is not None, "PyTorch required")
    core.validate_config(config)
    metadata = core.checkpoint_metadata(config, {"reference_manifest_sha256": core.REFERENCE_MANIFEST_SHA256, "partition_lock_sha256": core.PARTITION_LOCK_SHA256})
    _require(type(checkpoint) is dict and set(checkpoint) == set(metadata) | {"models"}, "checkpoint complete key/schema mismatch")
    for key, value in metadata.items():
        if key != "training_report":
            _require(_exact(checkpoint[key], value), f"checkpoint metadata mismatch: {key}")
    _required(checkpoint["training_report"], {
        "schema_version": 1, "status": "formal", "model_version": core.MODEL_VERSION,
        "reference_manifest_sha256": core.REFERENCE_MANIFEST_SHA256, "partition_lock_sha256": core.PARTITION_LOCK_SHA256,
        "holdout_waveforms_loaded": False, "outer_validation_used_for_selection": False, "runtime_enabled": False,
        "production_validated": False, "amp": False}, "checkpoint training provenance")
    states = checkpoint["models"]
    _require(type(states) is dict and set(states) == set(DATASETS), "checkpoint domain mismatch")
    models = {}
    for domain in DATASETS:
        model = core.build_model(config)  # Original bound config, NOT corruption override.
        expected, actual = model.state_dict(), states[domain]
        _require(isinstance(actual, dict) and set(actual) == set(expected), "state_dict complete key mismatch")
        for key, tensor in actual.items():
            _require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and tensor.layout == torch.strided and not tensor.requires_grad, f"invalid CPU state tensor: {domain}:{key}")
            _require(tuple(tensor.shape) == tuple(expected[key].shape) and tensor.dtype == expected[key].dtype, f"state shape/dtype mismatch: {domain}:{key}")
            _require(bool(torch.isfinite(tensor).all()), f"nonfinite checkpoint tensor: {domain}:{key}")
        model.load_state_dict(actual, strict=True)
        model.requires_grad_(False)
        model.eval()
        models[domain] = model
    return models

def cuda_preflight() -> dict:
    torch = core.torch
    _require(torch is not None and torch.cuda.is_available(), "CUDA:0 mandatory; CPU fallback refused")
    _require(torch.cuda.device_count() >= 1, "CUDA:0 not present")
    torch.cuda.set_device(0)
    probe = torch.zeros((1,), dtype=torch.float32, device="cuda:0")
    _require(bool(torch.isfinite(probe).all()), "CUDA probe failed")
    torch.cuda.synchronize(0)
    return {"device": "cuda:0", "device_name": torch.cuda.get_device_name(0), "torch_version": str(torch.__version__), "amp": False}

def _validate_records(records: list, state: Preflight) -> list:
    _require(type(records) is list and len(records) == 4, "exactly four internal records required")
    expected = {(r["dataset"], r["subject_id"]): r for r in state.selected}
    seen = set()
    for record in records:
        _require(type(record.dataset) is str and type(record.subject_id) is int, "illegal record identity types")
        key = record.dataset, record.subject_id
        _require(type(record.subject_id) is int and key in expected and key not in seen, "duplicate/illegal record identity")
        seen.add(key)
        _require(type(record.sample_rate_hz) is int and record.sample_rate_hz == RATES[record.dataset], "record native rate mismatch")
        arr = record.signal
        _require(isinstance(arr, np.ndarray) and arr.dtype == np.float64 and arr.shape == SHAPES[record.dataset] and np.isfinite(arr).all(), "record reference shape/dtype/finite mismatch")
        binding = expected[key]
        _require(record.npz_sha256 == binding["npz_sha256"] and record.sidecar_sha256 == binding["sidecar_sha256"], "record hash mismatch")
    return sorted(records, key=lambda r: (DATASETS.index(r.dataset), r.subject_id))

def run_benchmark(root: Path = ROOT) -> dict:
    """Real-data entry point: invoke ONLY after main-session code freeze."""
    state = preflight(root, check_only=False)  # All 18 hashes; four selected NPZs ONLY.
    _require(not state.deferred_hashes, "incomplete frozen preflight")
    gpu = cuda_preflight()
    torch = core.torch
    _verify_frozen(state)
    checkpoint = torch.load(safe_path(state.root, CHECKPOINT_PATH), map_location="cpu", weights_only=True)
    models = validate_checkpoint(checkpoint, state.config)
    for model in models.values():
        model.to(device="cuda:0", dtype=torch.float32)
        _require(not model.training and all(not p.requires_grad for p in model.parameters()), "inference model flags unsafe")
    _verify_frozen(state)  # MUST precede the ONLY legacy waveform loader call.
    records = _validate_records(core.load_reference_records(state.root, subjects=(5, 6)), state)
    cases = []
    with torch.inference_mode():
        for record in records:
            pending = []
            def flush():
                if not pending:
                    return
                inputs = np.stack([item[2] for item in pending])
                predictions = models[record.dataset](torch.from_numpy(inputs).to(device="cuda:0", dtype=torch.float32))
                _require(predictions.dtype == torch.float32 and tuple(predictions.shape) == tuple(inputs.shape) and bool(torch.isfinite(predictions).all()), "invalid inference prediction")
                pred = predictions.cpu().numpy()
                for item, prediction in zip(pending, pred):
                    provenance, ref, normalized, noisy, rms = item
                    output = restore_window(noisy, normalized, prediction, rms)
                    cases.append({**provenance, **case_metrics(ref, noisy, output, identity=provenance["condition"] == "identity")})
                pending.clear()
            for trial_index in range(SHAPES[record.dataset][0]):
                start = STARTS[record.dataset]
                ref = np.ascontiguousarray(record.signal[trial_index, start:start + WINDOW, :].T)
                for condition in CONDITIONS:
                    noisy, seed = synthetic_corruption(ref, record.dataset, record.subject_id, trial_index, condition, state.config)
                    normalized, mean, rms = core.normalize_window(noisy)
                    _window(normalized, "normalized noisy", dtype=np.float32)
                    _require(mean.shape == (8, 1) and mean.dtype == np.float64 and np.isfinite(mean).all(), "normalization mean mismatch")
                    provenance = {"dataset": record.dataset, "subject_id": record.subject_id, "trial_index": trial_index, "condition": condition,
                                  "seed": seed, "window_start_sample": start, "window_samples": WINDOW, "sample_rate_hz": RATES[record.dataset]}
                    pending.append((provenance, ref, normalized, noisy, rms))
                    if len(pending) == BATCH_SIZE:
                        flush()
            flush()  # Fixed batch64; final remainder only, within native domain/subject.
    aggregates = aggregate_cases(cases)
    _verify_frozen(state)
    return {"schema_version": 1, "status": "formal", "experiment": state.protocol["experiment"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "internal_synthetic_diagnostic", "inference": INFERENCE, "inference_performed": True, "checkpoint_loaded": True,
            "waveforms_loaded": True, "cuda_initialized": True, "gpu": gpu, "batch_size": BATCH_SIZE, "case_count": len(cases),
            "reference_trial_count": TRIAL_COUNT, "protocol_sha256": state.protocol_sha256, "freeze_lock_sha256": state.lock_sha256,
            "bound_files": dict(state.lock["files"]), "scope": dict(SCOPE), "selected_reference": list(state.selected),
            "limitations": list(LIMITATIONS), "metrics_definition": state.protocol["metrics_definition"], "aggregates": aggregates, "cases": cases}

def output_directory(root: Path, value: str | Path) -> Path:
    path, base = safe_path(root, value), safe_path(root, "analysis_outputs")
    _require(path == base or base in path.parents, "reports must remain within workspace analysis_outputs")
    _require(not path.exists() or path.is_dir(), "output path is not a directory")
    return path

def report_paths(root: Path, output_dir: str | Path, *, check_only: bool) -> tuple[Path, Path]:
    directory = output_directory(root, output_dir)
    stem = "ssvep_internal_v5_benchmark_check_only_20261007" if check_only else "ssvep_internal_v5_benchmark_20261007"
    paths = tuple(safe_path(root, directory / (stem + suffix)) for suffix in (".json", ".md"))
    _require(all(not p.exists() for p in paths), "refuse overwrite existing report")
    return paths

def render_markdown(report: dict) -> str:
    lines = ["# Frozen v5 INTERNAL synthetic-corruption diagnostic", "", "**WINDOW forward only - NO Hann OLA; NOT a complete-pipeline or task-preservation acceptance.**", ""]
    lines.extend(f"- {limitation}" for limitation in LIMITATIONS)
    lines.extend(["", f"Mode: `{report['mode']}`; inference performed: `{report['inference_performed']}`.",
                  f"Protocol SHA256: `{report['protocol_sha256']}`", f"Freeze lock SHA256: `{report['freeze_lock_sha256']}`", ""])
    if not report["inference_performed"]:
        lines.extend(["Metadata/code/sidecar hashes and file stat only. No NPZ/PT bytes, checkpoint deserialization, or CUDA initialization.",
                      "Binary hash, checkpoint schema/state, waveform values and CUDA checks are **NOT EXECUTED**.", "", "Deferred:"])
        lines.extend(f"- `{path}`" for path in report["deferred_binary_hash_checks"])
    else:
        lines.extend([f"Cases: {report['case_count']}; recorded-reference trials: {report['reference_trial_count']}.", "",
                      "Errors/aggregates are dimensionless, normalized per channel by recorded-reference variance + 1e-8.",
                      "Reduction is the ratio of mean errors, not mean case reductions. Identity reduction is null.",
                      "Win/loss/tie counts are descriptive exact comparisons with no imposed acceptance threshold.", "",
                      "| Condition | Cases | Mean input | Mean output | Ratio-of-mean reduction | Identity change | Wins | Losses | Ties |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        def number(value):
            return "null" if value is None else f"{value:.8g}"
        for condition in CONDITIONS:
            r = report["aggregates"]["condition"][condition]
            lines.append(f"| {condition} | {r['case_count']} | {number(r['mean_input_normalized_mse'])} | {number(r['mean_output_normalized_mse'])} | {number(r['ratio_of_mean_reduction'])} | {number(r['mean_identity_normalized_change'])} | {r['win_count']} | {r['loss_count']} | {r['tie_count']} |")
        lines.extend(["", "JSON contains every case's metrics, deterministic seed and fixed window/cohort provenance; no waveforms, predictions or original labels.",
                      "Dataset, dataset-qualified subject, condition and global dimensionless aggregates are in JSON."])
    return "\n".join(lines) + "\n"

def publish_reports(root: Path, output_dir: str | Path, report: dict, *, check_only: bool) -> tuple[Path, Path]:
    """Atomic per-file publication, no overwrite even if a competing writer wins."""
    _require(_plain(report), "report contains nonfinite/nonmetadata values")
    paths = report_paths(root, output_dir, check_only=check_only)
    texts = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", render_markdown(report))
    directory = paths[0].parent
    directory.mkdir(parents=True, exist_ok=True)
    safe_path(root, directory)
    pending, published = [], []
    try:
        for target, text in zip(paths, texts):
            safe_path(root, target)
            fd, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".pending", dir=directory)
            temporary = Path(name)
            pending.append(temporary)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
        for temporary, target in zip(pending, paths):
            safe_path(root, target)
            os.link(temporary, target)  # EEXIST on collision, no replace fallback.
            published.append((target, temporary.stat()))
    except BaseException:
        for target, identity in published:
            if target.exists():
                current = safe_path(root, target).lstat()
                if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                    target.unlink()  # Only our successfully created hardlink.
        raise
    finally:
        for temporary in pending:
            safe_path(root, temporary).unlink(missing_ok=True)
    return paths
