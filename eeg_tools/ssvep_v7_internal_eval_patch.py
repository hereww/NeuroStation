"""Traceable compatibility patch for the frozen v7 internal diagnostic.

The original evaluator is intentionally preserved.  Its finite-tree validator
uses exact built-in types, while ``torch.__version__`` can be a
``torch.TorchVersion`` (a ``str`` subclass).  This wrapper reuses the frozen
inference path, converts only the GPU version metadata to plain strings, and
publishes a separately locked diagnostic bundle with the deviation recorded.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib
import json
import sys

from eeg_tools import ssvep_v7_internal_eval as base

ROOT = base.ROOT
PATCH_CODE_LOCK = "analysis_outputs/ssvep_v7_internal_evaluation_patch_code_lock_20261008.json"
PATCH_EVAL_LOCK = "analysis_outputs/ssvep_v7_internal_evaluation_patch_lock_20261008.json"
PATCH_PREFIX = "analysis_outputs/ssvep_v7_internal_diagnostic_patch_20261008"
PATCH_CODE_PATHS = (
    "eeg_tools/ssvep_v7_internal_eval_patch.py",
    "scripts/evaluate_ssvep_denoiser_v7_internal_patch.py",
    "tests/test_ssvep_v7_internal_eval_patch.py",
)
PATCH_BOUND_PATHS = (
    base.CONFIG_RELATIVE,
    base.PREREG_RELATIVE,
    base.TRAIN_LOCK,
    base.EVAL_LOCK,
    base.CODE_LOCK,
    base.CHECKPOINT,
    base.TRAIN_REPORT,
    PATCH_CODE_LOCK,
)
DEVIATION = {
    "type": "finite_tree_type_compatibility",
    "original_evaluator": "eeg_tools/ssvep_v7_internal_eval.py",
    "original_behavior": "exact-type JSON validation rejects torch.TorchVersion metadata",
    "patched_behavior": "GPU torch/cuda version metadata is converted to plain str before validation",
    "scope": "serialization metadata only; inference, weights, data, thresholds, and gates unchanged",
}


class PatchDiagnosticError(ValueError):
    pass


def require(ok, text):
    if not ok:
        raise PatchDiagnosticError(text)


def _c():
    return base._c()


@dataclass(frozen=True)
class PatchState:
    base_state: base.State
    patch_lock_sha256: str
    patch_code_lock_sha256: str
    patch_files: dict
    stamps: dict
    check_only: bool

    def report(self):
        return {
            "stage": "v7_internal_diagnostic_patch",
            "check_only": self.check_only,
            "verified_hashes": dict(self.patch_files),
            "patch_lock_sha256": self.patch_lock_sha256,
            "patch_code_lock_sha256": self.patch_code_lock_sha256,
            "waveforms_loaded_this_call": False,
            "checkpoint_loaded_this_call": False,
            "cuda_initialized_this_call": False,
            "deviation": DEVIATION,
        }


def _bound_hashes(state: base.State, patch_code_lock_sha256: str):
    return {
        base.CONFIG_RELATIVE: base.CONFIG_SHA256,
        base.PREREG_RELATIVE: base.PREREG_SHA256,
        base.TRAIN_LOCK: state.files[base.TRAIN_LOCK],
        base.EVAL_LOCK: state.lock_sha256,
        base.CODE_LOCK: state.code_sha256,
        base.CHECKPOINT: state.files[base.CHECKPOINT],
        base.TRAIN_REPORT: state.files[base.TRAIN_REPORT],
        PATCH_CODE_LOCK: patch_code_lock_sha256,
    }


def _validate_patch_locks(root: Path, state: base.State):
    c = _c()
    code = c.read_json(root, PATCH_CODE_LOCK)
    lock = c.read_json(root, PATCH_EVAL_LOCK)
    require(code.get("schema_version") == 1 and code.get("status") == "formal", "formal patch code lock required")
    require(code.get("stage") == "frozen_before_v7_internal_diagnostic_patch_code_execution", "patch code stage mismatch")
    require(set(code.get("files", {})) == set(PATCH_CODE_PATHS), "patch code freeze mismatch")
    code_files = {}
    for rel in PATCH_CODE_PATHS:
        digest = c.file_sha256(root, rel)
        require(digest == code["files"].get(rel), "patch code changed: " + rel)
        code_files[rel] = digest
    patch_code_lock_sha256 = c.file_sha256(root, PATCH_CODE_LOCK)

    require(lock.get("schema_version") == 1 and lock.get("status") == "formal", "formal patch lock required")
    require(lock.get("stage") == "frozen_before_v7_internal_diagnostic_patch_execution", "patch lock stage mismatch")
    require(lock.get("deviation") == DEVIATION, "patch deviation mismatch")
    require(all(lock.get("scope", {}).get(k) is False for k in base.FALSE_FLAGS), "unsafe/missing patch lock flags")
    expected = _bound_hashes(state, patch_code_lock_sha256)
    require(lock.get("files") == expected, "patch bound-file mismatch")
    require(lock.get("base_evaluation_lock_sha256") == state.lock_sha256, "base evaluation lock mismatch")
    require(lock.get("base_code_lock_sha256") == state.code_sha256, "base code lock mismatch")
    patch_files = {**code_files, PATCH_CODE_LOCK: patch_code_lock_sha256, PATCH_EVAL_LOCK: c.file_sha256(root, PATCH_EVAL_LOCK)}
    return patch_code_lock_sha256, patch_files


def preflight(root=ROOT, *, check_only=True):
    require(type(check_only) is bool, "check_only bool required")
    root = _c().safe_path(root, ".")
    state = base.preflight(root, check_only=check_only)
    patch_code_lock_sha256, patch_files = _validate_patch_locks(root, state)
    stamps = {rel: _c()._stamp(root, rel) for rel in patch_files}
    return PatchState(state, _c().file_sha256(root, PATCH_EVAL_LOCK), patch_code_lock_sha256, patch_files, stamps, check_only)


def verify_frozen(state: PatchState):
    require(type(state) is PatchState, "patch state required")
    base.verify_frozen(state.base_state)
    c = _c()
    require(c.file_sha256(state.base_state.root, PATCH_EVAL_LOCK) == state.patch_lock_sha256, "patch lock changed")
    require(c.file_sha256(state.base_state.root, PATCH_CODE_LOCK) == state.patch_code_lock_sha256, "patch code lock changed")
    for rel, digest in state.patch_files.items():
        require(c._stamp(state.base_state.root, rel) == state.stamps[rel], "patch file stat changed: " + rel)
        require(c.file_sha256(state.base_state.root, rel) == digest, "patch file changed: " + rel)


def normalize_gpu_versions(report):
    """Convert only str subclasses emitted by torch into JSON-native strings."""
    gpu = report.get("gpu") if isinstance(report, dict) else None
    if isinstance(gpu, dict):
        for key in ("torch_version", "cuda_version"):
            value = gpu.get(key)
            if isinstance(value, str) and type(value) is not str:
                gpu[key] = str(value)
    return report


def _decorate_report(report, state: PatchState):
    normalize_gpu_versions(report)
    report["stage"] = "v7_fixed_internal_diagnostic_patch"
    report["deviation"] = DEVIATION
    report["patch"] = {
        "applied": True,
        "base_stage": "v7_fixed_internal_diagnostic",
        "base_evaluation_lock_sha256": state.base_state.lock_sha256,
        "base_code_lock_sha256": state.base_state.code_sha256,
        "patch_evaluation_lock_sha256": state.patch_lock_sha256,
        "patch_code_lock_sha256": state.patch_code_lock_sha256,
    }
    report.setdefault("frozen", {}).update({
        "base_evaluation_lock_sha256": state.base_state.lock_sha256,
        "base_code_lock_sha256": state.base_state.code_sha256,
        "patch_evaluation_lock_sha256": state.patch_lock_sha256,
        "patch_code_lock_sha256": state.patch_code_lock_sha256,
    })
    report.setdefault("preflight", {})["patch"] = state.report()
    return report


def run_diagnostic(state: PatchState):
    require(type(state) is PatchState and not state.check_only, "cannot execute check-only")
    verify_frozen(state)
    report = base.run_diagnostic(state.base_state)
    report = _decorate_report(report, state)
    verify_frozen(state)
    require(base.finite_tree(report), "finite JSON required after patch")
    return report


def _check_only_report(state: PatchState):
    report = {
        "schema_version": 1,
        "status": "formal",
        "check_only": True,
        "case_count": 0,
        "phase_case_count": 0,
        "all_internal_gates_passed": False,
        "runtime_enabled": False,
        "production_validated": False,
        "waveforms_loaded": False,
        "checkpoint_loaded": False,
        "cuda_initialized": False,
        "preflight": state.base_state.report(),
    }
    return _decorate_report(report, state)


def evaluate(root=ROOT, *, check_only=False, prefix=PATCH_PREFIX):
    paths = base.output_paths(root, prefix, check_only=check_only)
    state = preflight(root, check_only=check_only)
    if check_only:
        report = _check_only_report(state)
    else:
        report = run_diagnostic(state)
    require(base.finite_tree(report), "finite JSON required")
    verify_frozen(state)
    runtime = importlib.import_module("eeg_tools.ssvep_v7_experiment")
    runtime.publish_bundle(
        state.base_state.root,
        {
            paths[0]: (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"),
            paths[1]: base.markdown(report),
        },
    )
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = evaluate(check_only=args.check_only)
    except (ValueError, OSError) as exc:
        print("v7 diagnostic patch refused: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps({k: result[k] for k in ("check_only", "case_count", "phase_case_count", "all_internal_gates_passed")}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
