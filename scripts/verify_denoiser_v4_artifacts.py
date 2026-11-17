"""Freeze/verify v4 experiment artifacts without reading holdout model metrics.

This utility never edits a model, raw dataset, previous experiment or default
workstation configuration. Changed locks cannot be silently overwritten.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
TRAIN_LOCK = "analysis_outputs/learned_denoiser_v4_training_source_lock.json"
EVAL_LOCK = "analysis_outputs/learned_denoiser_v4_experiment_lock.json"
BASELINE = "analysis_outputs/learned_denoiser_v4_protected_baseline.json"
BASE_FILES = [
    "eeg_tools/learned_denoise_v4.py", "eeg_tools/learned_dataset_v4.py",
    "scripts/build_denoise_dataset_v4.py", "scripts/train_denoiser_v4.py",
    "scripts/verify_denoiser_v4_artifacts.py",
    "configs/learned_denoiser_v4.json", "analysis_outputs/learned_denoiser_v4_preregistration.json",
    "analysis_outputs/learned_denoiser_v4_interface_contract.json",
    "datasets/processed/learned_denoiser_v4.npz", "datasets/processed/learned_denoiser_v4.npz.manifest.json",
    "scripts/evaluate_denoiser_v2_transfer.py", "eeg_tools/learned_denoise_v2.py",
    "eeg_tools/learned_domain_dataset_v3.py",
]

EVALUATION_FILES = ["scripts/evaluate_denoiser_v4.py", "scripts/run_learned_denoise_v4.py",
                    "scripts/start_denoiser_v4_diagnostics.ps1"]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def checked_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("empty/non-string artifact path")
    # Validate both path syntaxes even on POSIX: Windows traversal/ADS must
    # not be treated as harmless filename characters by a Linux audit.
    from pathlib import PureWindowsPath
    windows = PureWindowsPath(relative)
    if windows.drive or windows.root or ".." in windows.parts or ":" in relative:
        raise ValueError(f"artifact must stay inside workspace: {relative}")
    root = root.resolve()
    p = (root / relative).resolve()
    if not p.is_relative_to(root) or p == root:
        raise ValueError(f"artifact must stay inside workspace: {relative}")
    return p


def verify_files(root: Path, files: dict[str, str]) -> list[dict]:
    if not isinstance(files, dict) or not files:
        raise ValueError("nonempty file SHA256 mapping required")
    changed = []
    for rel, expected in files.items():
        p = checked_path(root, rel)
        if (not isinstance(expected, str) or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError(f"invalid SHA256: {rel}")
        actual = file_sha256(p) if p.is_file() else None
        if actual != expected:
            changed.append({"path": rel, "expected": expected, "actual": actual})
    return changed


def verify_lock(root: Path, relative: str) -> dict:
    p = checked_path(root, relative)
    lock = json.loads(p.read_text(encoding="utf-8"))
    mismatch = verify_files(root, lock.get("files"))
    if mismatch:
        raise ValueError(f"artifact lock mismatch: {json.dumps(mismatch, ensure_ascii=True)}")
    return {"lock": relative, "lock_sha256": file_sha256(p),
            "verified_file_count": len(lock["files"]), "passed": True}


def freeze(root: Path, stage: str) -> dict:
    if stage not in {"training", "evaluation"}:
        raise ValueError("stage must be training or evaluation")
    rel_lock = TRAIN_LOCK if stage == "training" else EVAL_LOCK
    target = checked_path(root, rel_lock)
    if target.exists():
        return verify_lock(root, rel_lock)
    cfg_path = checked_path(root, "configs/learned_denoiser_v4.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    runtime = cfg.get("runtime", {})
    if cfg.get("status") != "formal" or runtime.get("enabled") is not False or runtime.get("production_validated") is not False:
        raise ValueError("v4 must stay formal/offline and unvalidated for production")
    prereg_path = checked_path(root, "analysis_outputs/learned_denoiser_v4_preregistration.json")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    paths = list(BASE_FILES)
    for source in prereg["data"]["sources"].values():
        p = checked_path(root, source["relative_path"])
        if file_sha256(p) != source["sha256"]:
            raise ValueError(f"raw source digest changed: {source['relative_path']}")
        paths.append(source["relative_path"])
    for name in prereg["predeclared_acceptance"]["new_dong_subjects"]:
        paths.append(f"datasets/external/ssvep_datasets/dong2023_ssvep/{name}")
    test_paths = [p.relative_to(root).as_posix() for p in (root / "tests").glob("*v4*.py")]
    if stage == "training":
        test_paths = [rel for rel in test_paths if not any(x in rel for x in ("evaluate_denoiser", "run_learned"))]
    paths += test_paths
    if stage == "evaluation":
        verify_lock(root, TRAIN_LOCK)
        paths += EVALUATION_FILES + [TRAIN_LOCK, "models/denoiser_v4.pt", "models/denoiser_v4.training.json"]
    files = {rel: file_sha256(checked_path(root, rel)) for rel in sorted(set(paths))}
    lock = {
        "schema_version": 1, "model_version": "learned_denoise_v4", "stage": stage,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "immutable offline research experiment, not clinical or automatic deployment approval",
        "files": files, "preregistration_sha256": file_sha256(prereg_path),
        "config_sha256": file_sha256(cfg_path),
        "dataset_sha256": files["datasets/processed/learned_denoiser_v4.npz"],
        "production_validated": False, "automatic_enabled": False,
    }
    if stage == "evaluation":
        lock["training_source_lock_sha256"] = file_sha256(checked_path(root, TRAIN_LOCK))
        lock["checkpoint_sha256"] = files["models/denoiser_v4.pt"]
    # Snapshots contain source/contract/config, never duplicate the raw or NPZ data.
    snap = checked_path(root, f"analysis_outputs/learned_denoiser_v4_{stage}_source_snapshot")
    for rel in paths:
        if rel.startswith(("eeg_tools/", "scripts/", "tests/", "configs/")) or rel.endswith(("preregistration.json", "interface_contract.json")):
            dest = snap / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and file_sha256(dest) != files[rel]:
                raise ValueError("refuse to overwrite an existing different source snapshot")
            if not dest.exists():
                shutil.copyfile(checked_path(root, rel), dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as f:
        json.dump(lock, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    return verify_lock(root, rel_lock)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["training", "evaluation", "verify"], default="verify")
    args = parser.parse_args(argv)
    if args.stage != "verify":
        result = freeze(ROOT, args.stage)
    else:
        results = [verify_lock(ROOT, rel) for rel in (TRAIN_LOCK, EVAL_LOCK) if (ROOT / rel).exists()]
        baseline = json.loads((ROOT / BASELINE).read_text(encoding="utf-8"))
        differences = verify_files(ROOT, baseline["files"])
        result = {"locks": results, "protected_file_count": len(baseline["files"]),
                  "protected_differences": differences, "passed": not differences and bool(results)}
        if differences:
            raise ValueError(f"protected pre-existing files changed: {json.dumps(differences, ensure_ascii=True)}")
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
    return 0 if result.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
