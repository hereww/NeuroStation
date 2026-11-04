"""JSON-only audit for the v7 diagnostic compatibility-patch report.

This audit reads only the saved JSON report and frozen config.  It reuses the
standard v7 scalar reaggregation after removing the explicitly recorded patch
metadata in memory; it never reads NPZ/PT files, initializes CUDA, or changes
acceptance/deployment state.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import stat
import sys
from datetime import datetime, timezone

from eeg_tools import ssvep_saved_audit_v7 as base
from eeg_tools import ssvep_v7_internal_eval_patch as diagnostic_patch

ROOT = base.ROOT
_ORIGINAL_RENDER_MARKDOWN = base.render_markdown
REPORT_RELATIVE = "analysis_outputs/ssvep_v7_internal_diagnostic_patch_20261008.json"
DEFAULT_OUTPUT_PREFIX = "analysis_outputs/ssvep_v7_saved_internal_audit_patch_20261008"
AUDIT_VERSION = "ssvep_saved_internal_json_audit_v7_patch"
AUDIT_SOURCE_PATHS = (
    "eeg_tools/ssvep_saved_audit_v7_patch.py",
    "scripts/audit_ssvep_saved_internal_v7_patch.py",
    "tests/test_ssvep_saved_audit_v7_patch.py",
)
BASE_FIELDS = set(base.REPORT_FIELDS)
PATCH_FIELDS = {"deviation", "patch"}
BASE_FROZEN_FIELDS = {
    "configuration_sha256", "preregistration_sha256", "training_lock_sha256",
    "evaluation_lock_sha256", "checkpoint_sha256", "training_report_sha256",
    "code_lock_sha256",
}
BASE_PREFLIGHT_FIELDS = {
    "stage", "check_only", "verified_hashes", "deferred", "lock_sha256",
    "code_lock_sha256", "waveforms_loaded_this_call", "checkpoint_loaded_this_call",
    "cuda_initialized_this_call",
}
MAX_SOURCE_BYTES = 64 * 1024 * 1024


class PatchAuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise PatchAuditError(message)


def _digest(value, context):
    require(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value), context + " must be lowercase SHA256")


def _read_source(root: Path, rel: str):
    require(rel in AUDIT_SOURCE_PATHS, "unapproved audit source")
    path = base.safe_path(root, rel)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= MAX_SOURCE_BYTES, "regular bounded audit source required")
    stamp = base._stamp(info)
    data = path.read_bytes()
    require(base._stamp(path.lstat()) == stamp, "audit source changed during read")
    return data, stamp


def validate_patch_metadata(report):
    require(type(report) is dict and set(report) == BASE_FIELDS | PATCH_FIELDS, "patch diagnostic report schema mismatch")
    require(report.get("stage") == "v7_fixed_internal_diagnostic_patch", "patch report stage mismatch")
    require(report.get("deviation") == diagnostic_patch.DEVIATION, "patch deviation metadata mismatch")
    patch = report.get("patch")
    require(type(patch) is dict and set(patch) == {
        "applied", "base_stage", "base_evaluation_lock_sha256", "base_code_lock_sha256",
        "patch_evaluation_lock_sha256", "patch_code_lock_sha256",
    }, "patch provenance schema mismatch")
    require(patch["applied"] is True and patch["base_stage"] == "v7_fixed_internal_diagnostic", "patch provenance flags mismatch")
    for key in ("base_evaluation_lock_sha256", "base_code_lock_sha256", "patch_evaluation_lock_sha256", "patch_code_lock_sha256"):
        _digest(patch[key], "patch." + key)
    frozen = report.get("frozen")
    require(type(frozen) is dict and set(frozen) == BASE_FROZEN_FIELDS | {
        "base_evaluation_lock_sha256", "base_code_lock_sha256", "patch_evaluation_lock_sha256", "patch_code_lock_sha256",
    }, "patched frozen provenance schema mismatch")
    for key in frozen:
        _digest(frozen[key], "frozen." + key)
    require(frozen["base_evaluation_lock_sha256"] == patch["base_evaluation_lock_sha256"], "base evaluation provenance mismatch")
    require(frozen["base_code_lock_sha256"] == patch["base_code_lock_sha256"], "base code provenance mismatch")
    require(frozen["patch_evaluation_lock_sha256"] == patch["patch_evaluation_lock_sha256"], "patch evaluation provenance mismatch")
    require(frozen["patch_code_lock_sha256"] == patch["patch_code_lock_sha256"], "patch code provenance mismatch")
    preflight = report.get("preflight")
    require(type(preflight) is dict and set(preflight) == BASE_PREFLIGHT_FIELDS | {"patch"}, "patched preflight schema mismatch")
    patch_preflight = preflight["patch"]
    require(type(patch_preflight) is dict and patch_preflight.get("stage") == "v7_internal_diagnostic_patch" and patch_preflight.get("check_only") is False, "patched preflight scope mismatch")
    return report


def base_compatible_report(report):
    """Return a copy with only patch bookkeeping removed for the frozen base audit."""
    sanitized = dict(report)
    sanitized.pop("deviation")
    sanitized.pop("patch")
    sanitized["stage"] = "v7_fixed_internal_diagnostic"
    sanitized["frozen"] = {key: report["frozen"][key] for key in BASE_FROZEN_FIELDS}
    sanitized["preflight"] = {key: report["preflight"][key] for key in BASE_PREFLIGHT_FIELDS}
    return sanitized


def output_paths(root: Path, prefix: str = DEFAULT_OUTPUT_PREFIX):
    return base.output_paths(root, prefix)


def render_markdown(audit):
    text = _ORIGINAL_RENDER_MARKDOWN(audit).decode("utf-8")
    marker = "**Compatibility-patch report: patch metadata was validated separately; scalar reaggregation used the frozen base v7 schema.**\n\n"
    return (text.replace("**Scalar bookkeeping audit only; NOT independent acceptance or deployment.**\n\n", "**Scalar bookkeeping audit only; NOT independent acceptance or deployment.**\n\n" + marker, 1)).encode("utf-8")


def publish_audit(root: Path, prefix: str, audit: dict):
    # The base publisher is atomic and applies the same collision checks.  Its
    # renderer is replaced locally only for this patch audit call.
    original = base.render_markdown
    try:
        base.render_markdown = render_markdown
        return base.publish_audit(root, prefix, audit)
    finally:
        base.render_markdown = original


def audit_saved_report(root: Path = ROOT, *, report_relative: str = REPORT_RELATIVE,
                       output_prefix: str = DEFAULT_OUTPUT_PREFIX, check_only: bool = False) -> dict:
    require(type(check_only) is bool, "check_only must be bool")
    outputs = output_paths(root, output_prefix)
    report_path = base._json_path(root, report_relative)
    require(report_path not in tuple(base.safe_path(root, p) for p in outputs), "input/output collision")
    cfg_data, cfg_stamp = base._read_bytes(root, base.CONFIG_RELATIVE)
    cfg_sha = hashlib.sha256(cfg_data).hexdigest()
    require(cfg_sha == base.CONFIG_SHA256, "frozen config bytes SHA drift")
    cfg = base.strict_json_loads(cfg_data)
    base.validate_config(cfg)
    if check_only:
        return {
            "schema_version": 1, "status": "formal", "audit_version": AUDIT_VERSION,
            "check_only": True, "audit_performed": False, "audit_passed": False,
            "report_read": False, "report_read_deferred": report_relative, "would_publish": list(outputs),
            "config_sha256": cfg_sha, "research_acceptance": False, "runtime_enabled": False,
            "production_validated": False, "waveforms_loaded": False, "checkpoint_loaded": False,
            "cuda_initialized": False, "limitations": list(base.LIMITATIONS),
        }
    report_data, report_stamp = base._read_bytes(root, report_relative)
    report_sha = hashlib.sha256(report_data).hexdigest()
    report = base.strict_json_loads(report_data)
    validate_patch_metadata(report)
    result = base.audit_report(base_compatible_report(report), cfg)
    result["audit_version"] = AUDIT_VERSION
    result["patch_metadata_validated"] = True
    result["patch_deviation"] = dict(diagnostic_patch.DEVIATION)
    result["patch_provenance"] = dict(report["patch"])
    result["inputs"] = {
        "report_path": report_relative, "report_sha256": report_sha,
        "config_path": base.CONFIG_RELATIVE, "config_sha256": cfg_sha,
    }
    source_hashes = {}
    snapshots = {base.CONFIG_RELATIVE: (cfg_sha, cfg_stamp), report_relative: (report_sha, report_stamp)}
    for rel in AUDIT_SOURCE_PATHS:
        data, stamp = _read_source(root, rel)
        digest = hashlib.sha256(data).hexdigest()
        source_hashes[rel] = digest
        snapshots[rel] = (digest, stamp)
    result["audit_source_sha256"] = source_hashes
    result["audit_source_binding_scope"] = "separate_v7_saved_json_patch_audit_evidence_not_training_or_diagnostic_bindings"
    for rel, (digest, stamp) in snapshots.items():
        if rel in AUDIT_SOURCE_PATHS:
            data, current = _read_source(root, rel)
        else:
            data, current = base._read_bytes(root, rel)
        require(current == stamp and hashlib.sha256(data).hexdigest() == digest, "audit input/source changed: " + rel)
    publish_audit(root, output_prefix, result)
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Independent JSON-only audit of the patched v7 internal diagnostic")
    parser.add_argument("--report", default=REPORT_RELATIVE)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = audit_saved_report(report_relative=args.report, output_prefix=args.output_prefix, check_only=args.check_only)
    except (ValueError, OSError) as exc:
        print("v7 patched saved JSON audit refused: " + str(exc), file=sys.stderr)
        return 2
    keys = ("check_only", "audit_performed", "audit_passed", "research_acceptance", "runtime_enabled", "production_validated")
    summary = {key: result[key] for key in keys}
    if not args.check_only:
        summary["mismatch_count"] = result["mismatch_count"]
    print(json.dumps(summary, allow_nan=False))
    return 0 if args.check_only or result["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

