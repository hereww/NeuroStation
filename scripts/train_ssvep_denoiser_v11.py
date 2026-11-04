"""V11 CUDA training runner: metadata-only by default, explicit --run only.

Uses frozen v10 numerical helpers with a separate v11 configuration and lock.
Never loads evaluation subjects or personal recordings; no CPU fallback/resume.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# This frozen helper module imports only the standard library. Do not call its
# v10-only configuration, preflight, training entrypoint or CLI functions.
from scripts.train_ssvep_denoiser_v10 import (
    TrainingV10Error, initialize_cuda, load_training_references, condition_inputs,
    fit_epoch, evaluate_selection, preferred_candidate, _snapshot,
    publish_exclusive, canonical_sha, file_sha,
)

CHECKPOINT = "models/denoiser_v11_personal.pt"
REPORT = "analysis_outputs/ssvep_v11_training_20261013.json"
LOCK = "analysis_outputs/ssvep_v11_training_lock_20261013.json"
CONDITIONS = ("identity", "drift", "high_burst", "gaussian", "combined")
FIT_SUBJECTS = (1, 2, 3, 4)
SELECTION_SUBJECTS = (5, 6)


class TrainingV11Error(RuntimeError):
    """V11 contract/publication failure; never fall back to CPU."""


def require(ok, message):
    if not ok:
        raise TrainingV11Error("v11 trainer: " + message)


def assert_new_outputs(root=ROOT):
    # lexists also rejects broken links. Called BEFORE even importing the data API.
    for name in (CHECKPOINT, REPORT):
        require(not os.path.lexists(Path(root) / name), "output already exists: " + name)


def validate_training_config(cfg):
    require(cfg.get("model_version") == "learned_denoise_v11_personal"
            and cfg.get("status") == "formal", "wrong model/config version")
    require((cfg.get("channel_count"), cfg.get("window_samples"), cfg.get("sample_rate_hz"))
            == (8, 512, 250), "input contract must be native 250Hz [B,8,512]")
    require(cfg["data"]["fit_subjects"] == list(FIT_SUBJECTS)
            and cfg["data"]["selection_subjects"] == list(SELECTION_SUBJECTS)
            and cfg["data"]["evaluation_subjects"] == [7, 8], "subject roles changed")
    require(cfg["data"]["personal_sessions_used_for_training"] is False
            and cfg["data"]["task_labels_enter_forward"] is False
            and cfg["data"]["raw_reference_is_clean_ground_truth"] is False,
            "invalid training data/ground-truth claims")
    require(cfg.get("schema_version") == 1 and cfg.get("hop_samples") == 128,
            "wrong schema/hop contract")
    require(cfg["data"].get("evaluation_is_pristine_project_holdout") is False
            and cfg["data"].get("all_public_subjects_previously_available_in_project") is True,
            "public development history must be disclosed")
    require(isinstance(cfg.get("prior_experiment"), str) and "v10" in cfg["prior_experiment"]
            and "failed" in cfg["prior_experiment"], "prior v10 failed experiment missing")
    expected = {"seed": 20261014, "epochs": 24, "batch_size": 64,
        "learning_rate": 0.0003, "weight_decay": 0.0001, "optimizer": "AdamW",
        "device": "cuda:0", "require_cuda": True, "amp": False,
        "allow_tf32": False, "deterministic_algorithms": True,
        "cpu_threads": 4, "gradient_clip": 1.0, "identity_loss_weight": 100.0}
    for key, value in expected.items():
        actual = cfg["training"].get(key)
        require(type(actual) is type(value) and actual == value,
                "training setting changed: " + key)
    require(cfg["augmentation"]["conditions"] == list(CONDITIONS)
            and cfg["augmentation"]["synthetic_training_examples"] == 480,
            "augmentation condition/count changed")
    require(cfg["engineering_gates"]["identity_mean_normalized_change_max"] == 0.002,
            "identity gate changed")
    require(cfg["runtime"]["enabled"] is False
            and cfg["runtime"]["production_validated"] is False, "runtime must remain disabled")
    # Enforce JSON primitives (including finite numbers) before checkpoint creation.
    json.dumps(cfg, allow_nan=False)


def checked_metadata(root=ROOT):
    api = importlib.import_module("eeg_tools.ssvep_v11_runtime")
    cfg = api.load_config(root=root)
    validate_training_config(cfg)
    state = api.preflight(root=root)
    require(isinstance(state, dict) and state.get("status") == "ready", "preflight is not ready")
    require(state.get("config_sha256") == canonical_sha(cfg), "preflight config mismatch")
    require(state.get("training_lock_sha256") == file_sha(Path(root) / LOCK), "lock binding mismatch")
    require(state.get("waveforms_loaded") is False and state.get("cuda_initialized") is False,
            "metadata preflight loaded waveforms/CUDA")
    return api, cfg, state


def run_training(api, cfg, metadata, root=ROOT):
    assert_new_outputs(root)
    validate_training_config(cfg)
    require(metadata.get("config_sha256") == canonical_sha(cfg)
            and metadata.get("training_lock_sha256") == file_sha(Path(root) / LOCK),
            "training metadata/config/lock mismatch")
    require(api.preflight(root=root) == metadata, "training bindings changed before run")
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    torch, device = initialize_cuda(cfg)
    np = importlib.import_module("numpy")
    np.random.seed(cfg["training"]["seed"])
    model = api.build_model(cfg).to(device=device, dtype=torch.float32)
    require(all(p.device == device and p.dtype == torch.float32 for p in model.parameters()),
            "model is not float32 CUDA:0")
    data_api = importlib.import_module("eeg_tools.ssvep_v10_data")
    fit, fit_ids, synthetic_ids, selection, selection_ids = load_training_references(data_api, np, cfg, root)
    fixed_inputs = condition_inputs(data_api, np, selection, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                 weight_decay=cfg["training"]["weight_decay"])
    epochs = []
    best_eligible = best_fallback = None
    eligible_state = fallback_state = None
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        epoch_inputs = condition_inputs(data_api, np, fit, cfg, epoch=epoch)
        loss = fit_epoch(torch, np, model, optimizer, fit, epoch_inputs, cfg, device, epoch)
        del epoch_inputs
        candidate = evaluate_selection(torch, model, selection, fixed_inputs, cfg, device)
        candidate.update(epoch=epoch, fit_loss=loss,
                         fit_cases_per_condition=len(fit), selection_cases_per_condition=len(selection))
        candidate["became_eligible_best"] = preferred_candidate(candidate, best_eligible, eligible=True)
        candidate["became_fallback_best"] = preferred_candidate(candidate, best_fallback, eligible=False)
        if candidate["became_eligible_best"]:
            best_eligible, eligible_state = dict(candidate), _snapshot(model)
        if candidate["became_fallback_best"]:
            best_fallback, fallback_state = dict(candidate), _snapshot(model)
        epochs.append(candidate)
        print(json.dumps({"status": "epoch_completed", **candidate}, allow_nan=False), flush=True)
    selected = best_eligible if best_eligible is not None else best_fallback
    state = eligible_state if best_eligible is not None else fallback_state
    require(selected is not None and state is not None, "no completed checkpoint candidate")
    require(all(bool(torch.isfinite(value).all()) for value in state.values()), "nonfinite checkpoint")
    # Recheck source/config/data bindings before committing either output.
    final_metadata = api.preflight(root=root)
    require(canonical_sha(cfg) == metadata["config_sha256"]
            and final_metadata == metadata and file_sha(Path(root) / LOCK) == metadata["training_lock_sha256"],
            "training bindings changed during run")
    assert_new_outputs(root)
    checkpoint = {"schema_version": 1, "model_version": cfg["model_version"],
        "config": cfg, "config_sha256": canonical_sha(cfg),
        "training_lock_sha256": metadata["training_lock_sha256"], "state_dict": state,
        "selected_epoch": int(selected["epoch"]), "selection_gate_passed": best_eligible is not None,
        "runtime_enabled": False, "production_validated": False}
    report = {"schema_version": 1, "status": "training_completed", "model_version": cfg["model_version"],
        "config": cfg, "config_sha256": canonical_sha(cfg),
        "training_lock_sha256": metadata["training_lock_sha256"], "preflight": metadata,
        "started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "device": str(device), "device_name": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "amp": False, "allow_tf32": False, "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "conditions": list(CONDITIONS), "epochs": epochs, "selected_epoch": int(selected["epoch"]),
        "selected_selection": selected, "selection_gate_passed": best_eligible is not None,
        "selection_gate_scope": "identity_mean_normalized_change<=0.002 only; not all engineering gates",
        "selection_reason": "lowest_corrupted_nmse_among_identity_eligible" if best_eligible is not None
            else "no_identity_eligible_epoch_lowest_weighted_selection_loss_gate_failed",
        "fit_source_identities": fit_ids, "fit_source_identity_count": len(fit_ids),
        "synthetic_source_identities": synthetic_ids, "synthetic_source_identity_count": len(synthetic_ids),
        "selection_source_identities": selection_ids, "selection_source_identity_count": len(selection_ids),
        "fit_examples_per_epoch": len(fit) * len(CONDITIONS),
        "selection_examples_per_epoch": len(selection) * len(CONDITIONS),
        "seed_policy": {"fit": "seed+one_based_epoch*1000+condition_index",
            "selection": "seed+100000+condition_index", "synthetic": cfg["training"]["seed"],
            "shuffle": "seed+one_based_epoch"},
        "metric_definition": "mean squared prediction-reference difference over cases/channels/samples; references unit RMS per channel",
        "loss_definition": "mean per-case NMSE with identity weight 100 and other condition weights 1; equal cases per condition",
        "prior_experiment": cfg["prior_experiment"], "prior_v10_failed": True,
        "public_evaluation_not_pristine": True,
        "evaluation_is_pristine_project_holdout": False,
        "all_public_subjects_previously_available_in_project": True,
        "evaluation_subjects_loaded": [], "personal_data_loaded": False,
        "task_labels_enter_forward": False, "raw_reference_is_clean_ground_truth": False,
        "independent_personal_validation": False, "runtime_enabled": False, "production_validated": False,
        "checkpoint": CHECKPOINT}
    # Validate report serialization before publishing the checkpoint.
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    publish_exclusive(Path(root) / CHECKPOINT, lambda stream: torch.save(checkpoint, stream))
    report["checkpoint_sha256"] = file_sha(Path(root) / CHECKPOINT)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    publish_exclusive(Path(root) / REPORT, lambda stream: stream.write(payload))
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="metadata/hash checks only (default); no Torch/CUDA/NPZ load")
    mode.add_argument("--run", action="store_true", help="train on CUDA:0 after independent lock freeze; never overwrite")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        assert_new_outputs(ROOT)
        api, cfg, metadata = checked_metadata(ROOT)
        if not args.run:
            print(json.dumps({**metadata, "mode": "check_only", "training_started": False},
                             ensure_ascii=False, allow_nan=False))
            return 0
        report = run_training(api, cfg, metadata, ROOT)
        print(json.dumps({"status": report["status"], "selected_epoch": report["selected_epoch"],
            "selection_gate_passed": report["selection_gate_passed"],
            "runtime_enabled": False, "production_validated": False}, allow_nan=False), flush=True)
        return 0
    except (TrainingV11Error, TrainingV10Error, OSError, ImportError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc), "runtime_enabled": False,
                          "production_validated": False}, ensure_ascii=False, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
