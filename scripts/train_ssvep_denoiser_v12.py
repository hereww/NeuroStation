"""V12 CUDA training runner with noisy-window normalization consistency.

The default invocation performs metadata, configuration, lock and output checks only.
Training is explicit (``--run``), requires CUDA:0, and never falls back to CPU.
V12 owns its normalization contract: every corrupted window is normalized using
its own float64 noisy mean/RMS; the same statistics transform the reference, the
network sees only float32 normalized data, and selection restores predictions in
original units through the runtime API.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# This frozen module is stdlib-only at import time. It supplies only generic
# helpers; v10 fit/evaluation functions are deliberately not reused.
from scripts.train_ssvep_denoiser_v10 import (  # noqa: E402
    TrainingV10Error, canonical_sha, condition_inputs, file_sha,
    initialize_cuda, load_training_references, publish_exclusive,
    _snapshot,
)

CHECKPOINT = "models/denoiser_v12_personal.pt"
REPORT = "analysis_outputs/ssvep_v12_training_20261013.json"
LOCK = "analysis_outputs/ssvep_v12_training_lock_20261013.json"
CONDITIONS = ("identity", "drift", "high_burst", "gaussian", "combined")
FIT_SUBJECTS = (1, 2, 3, 4)
SELECTION_SUBJECTS = (5, 6)
WINDOW_SHAPE = (8, 512)


class TrainingV12Error(RuntimeError):
    """V12 contract, numerical or publication failure."""


def require(ok, message):
    if not ok:
        raise TrainingV12Error("v12 trainer: " + message)


def assert_new_outputs(root=ROOT):
    # This must remain before importing the runtime/data/model dependencies.
    for name in (CHECKPOINT, REPORT):
        require(not os.path.lexists(Path(root) / name), "output already exists: " + name)


def validate_training_config(cfg):
    require(isinstance(cfg, dict), "configuration must be an object")
    require(cfg.get("model_version") == "learned_denoise_v12_personal"
            and cfg.get("status") == "formal", "wrong model/config version")
    require((cfg.get("channel_count"), cfg.get("window_samples"), cfg.get("sample_rate_hz"))
            == (8, 512, 250), "input contract must be native 250Hz [B,8,512]")
    data = cfg.get("data")
    require(isinstance(data, dict), "missing data configuration")
    require(data.get("fit_subjects") == list(FIT_SUBJECTS)
            and data.get("selection_subjects") == list(SELECTION_SUBJECTS)
            and data.get("evaluation_subjects") == [7, 8], "subject roles changed")
    require(data.get("personal_sessions_used_for_training") is False
            and data.get("task_labels_enter_forward") is False
            and data.get("raw_reference_is_clean_ground_truth") is False,
            "invalid training data/ground-truth claims")
    require(cfg.get("schema_version") == 1 and cfg.get("hop_samples") == 128,
            "wrong schema/hop contract")
    require(data.get("evaluation_is_pristine_project_holdout") is False
            and data.get("all_public_subjects_previously_available_in_project") is True,
            "public development history must be disclosed")
    prior = cfg.get("prior_experiment")
    require(isinstance(prior, str) and "v11" in prior and "normalization" in prior,
            "v11 normalization failure disclosure missing")
    training = cfg.get("training")
    require(isinstance(training, dict), "missing training configuration")
    expected = {"seed": 20261015, "epochs": 24, "batch_size": 64,
        "learning_rate": 0.0003, "weight_decay": 0.0001, "optimizer": "AdamW",
        "device": "cuda:0", "require_cuda": True, "amp": False,
        "allow_tf32": False, "deterministic_algorithms": True,
        "cpu_threads": 4, "gradient_clip": 1.0, "identity_loss_weight": 100.0}
    for key, value in expected.items():
        actual = training.get(key)
        require(type(actual) is type(value) and actual == value,
                "training setting changed: " + key)
    augmentation = cfg.get("augmentation")
    require(isinstance(augmentation, dict)
            and augmentation.get("conditions") == list(CONDITIONS)
            and augmentation.get("synthetic_training_examples") == 480,
            "augmentation condition/count changed")
    gates = cfg.get("engineering_gates")
    require(isinstance(gates, dict)
            and gates.get("identity_mean_normalized_change_max") == 0.002,
            "identity gate changed")
    normalization = cfg.get("normalization")
    require(isinstance(normalization, dict)
            and normalization.get("input_statistics_only") is True
            and normalization.get("loss_coordinates") == "restored reference-normalized units; no mislabeled noisy-normalized MSE"
            and normalization.get("identity_restoration"),
            "noisy-stat normalization contract missing")
    runtime = cfg.get("runtime")
    require(isinstance(runtime, dict)
            and runtime.get("enabled") is False
            and runtime.get("production_validated") is False,
            "runtime must remain disabled")
    json.dumps(cfg, ensure_ascii=False, allow_nan=False)


def checked_metadata(root=ROOT):
    # Dynamic import is intentionally here, never at module import/check setup.
    api = importlib.import_module("eeg_tools.ssvep_v12_runtime")
    cfg = api.load_config(root=root)
    validate_training_config(cfg)
    state = api.preflight(root=root)
    require(isinstance(state, dict) and state.get("status") == "ready", "preflight is not ready")
    require(state.get("config_sha256") == canonical_sha(cfg), "preflight config mismatch")
    require(state.get("training_lock_sha256") == file_sha(Path(root) / LOCK),
            "lock binding mismatch")
    require(state.get("waveforms_loaded") is False and state.get("cuda_initialized") is False,
            "metadata preflight loaded waveforms/CUDA")
    return api, cfg, state


def augmentation_seed(seed, condition_index, *, epoch=None):
    require(type(condition_index) is int and 0 <= condition_index < len(CONDITIONS),
            "invalid condition index")
    require(epoch is None or (type(epoch) is int and 1 <= epoch <= 24),
            "invalid fit epoch")
    return seed + (100000 if epoch is None else epoch * 1000) + condition_index


def preferred_candidate(candidate, previous, *, eligible):
    if eligible and not candidate["identity_gate_passed"]:
        return False
    key = "corrupted_mean_sameunit_mse" if eligible else "selection_loss_sameunit_mse"
    return previous is None or candidate[key] < previous[key]


def _validate_batch_array(np, value, name, dtype, shape):
    require(isinstance(value, np.ndarray) and value.dtype == dtype
            and value.ndim == 3 and value.shape[1:] == shape
            and len(value) > 0 and np.isfinite(value).all(),
            "invalid " + name)


def _validate_stats(np, means, rms, count):
    require(isinstance(means, np.ndarray) and means.dtype == np.float64
            and means.shape in ((count, 8), (count, 8, 1)) and np.isfinite(means).all(),
            "invalid noisy means")
    require(isinstance(rms, np.ndarray) and rms.dtype == np.float64
            and rms.shape in ((count, 8), (count, 8, 1)) and np.isfinite(rms).all()
            and bool((rms > 0).all()), "invalid noisy RMS")


def normalize_batch(api, np, noisy, reference=None):
    """Normalize noisy windows and optionally transform references with same stats."""
    _validate_batch_array(np, noisy, "noisy batch", np.float32, WINDOW_SHAPE)
    normalized, means, rms = api.normalize_noisy(noisy)
    _validate_batch_array(np, normalized, "normalized noisy batch", np.float32, WINDOW_SHAPE)
    _validate_stats(np, means, rms, len(noisy))
    result = {"normalized": normalized, "means": means, "rms": rms}
    if reference is not None:
        _validate_batch_array(np, reference, "reference batch", np.float32, WINDOW_SHAPE)
        target = api.target_in_noisy_coordinates(reference, means, rms)
        _validate_batch_array(np, target, "normalized reference target", np.float32, WINDOW_SHAPE)
        result["target"] = target
    return result


def _rms_tensor(torch, rms, device):
    tensor = torch.from_numpy(rms).to(device=device, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    require(tensor.ndim == 3 and tensor.shape[1:] in ((8, 1), (8, 512)),
            "RMS tensor cannot broadcast across window")
    return tensor


def restored_mse_torch(torch, prediction, target_norm, rms):
    """Per-case MSE in original/reference units, not noisy-normalized units."""
    require(prediction.shape == target_norm.shape and prediction.ndim == 3,
            "prediction/target shape mismatch")
    scale = rms if isinstance(rms, torch.Tensor) else _rms_tensor(torch, rms, prediction.device)
    if scale.ndim == 2:
        scale = scale.unsqueeze(-1)
    restored_residual = (prediction - target_norm) * scale
    value = restored_residual.square().mean(dim=(1, 2))
    require(bool(torch.isfinite(value).all()), "nonfinite restored-unit loss")
    return value


def _prediction(torch, model, x):
    require(x.dtype == torch.float32 and x.ndim == 3 and x.shape[1:] == WINDOW_SHAPE,
            "model input must be float32 normalized [B,8,512]")
    prediction = model(x)
    require(isinstance(prediction, torch.Tensor) and prediction.shape == x.shape
            and prediction.dtype == torch.float32 and prediction.device == x.device,
            "model must return same-shape/device float32 prediction")
    require(bool(torch.isfinite(prediction).all()), "model produced nonfinite predictions")
    return prediction


def fit_epoch(torch, np, api, model, optimizer, refs, inputs, cfg, device, epoch):
    """Fit one epoch; every batch derives all target stats from its noisy input."""
    model.train()
    n = len(refs)
    rng = np.random.default_rng(cfg["training"]["seed"] + epoch)
    order = rng.permutation(n * len(CONDITIONS))
    total_loss = 0.0
    batch_size = cfg["training"]["batch_size"]
    for start in range(0, len(order), batch_size):
        ids = order[start:start + batch_size]
        batch = np.empty((len(ids), 8, 512), dtype=np.float32)
        for condition_index, values in enumerate(inputs):
            mask = ids // n == condition_index
            batch[mask] = values[ids[mask] % n]
        reference = refs[ids % n]
        prepared = normalize_batch(api, np, batch, reference)
        x = torch.from_numpy(prepared["normalized"]).to(device=device, dtype=torch.float32)
        target = torch.from_numpy(prepared["target"]).to(device=device, dtype=torch.float32)
        rms = _rms_tensor(torch, prepared["rms"], device)
        weights = np.where(ids // n == 0, cfg["training"]["identity_loss_weight"], 1.0).astype(np.float32)
        weight = torch.from_numpy(weights).to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        pred = _prediction(torch, model, x)
        per_case = restored_mse_torch(torch, pred, target, rms)
        loss = (per_case * weight).mean()
        require(bool(torch.isfinite(loss)), "nonfinite training loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
            cfg["training"]["gradient_clip"], error_if_nonfinite=True)
        require(bool(torch.isfinite(grad_norm)), "nonfinite gradients")
        optimizer.step()
        total_loss += float(loss.detach().item()) * len(ids)
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "nonfinite model state")
    return total_loss / len(order)


def _restore_numpy_batch(api, np, noisy, normalized, prediction, rms):
    restored = api.restore_residual(noisy, normalized, prediction, rms)
    require(isinstance(restored, np.ndarray) and restored.dtype == np.float64
            and restored.shape == noisy.shape and np.isfinite(restored).all(),
            "runtime restoration must return finite float64 same-shape output")
    return restored


def selection_summary(condition_sameunit_mse, identity_change, cfg):
    require(set(condition_sameunit_mse) == set(CONDITIONS),
            "selection condition coverage mismatch")
    require(all(type(v) in (float, int) and math.isfinite(v) and v >= 0
                for v in condition_sameunit_mse.values()), "invalid same-unit selection metric")
    require(type(identity_change) in (float, int) and math.isfinite(identity_change)
            and identity_change >= 0, "invalid identity normalized change")
    corrupted = sum(condition_sameunit_mse[c] for c in CONDITIONS[1:]) / 4.0
    identity_mse = float(condition_sameunit_mse["identity"])
    weighted = (cfg["training"]["identity_loss_weight"] * identity_mse + 4.0 * corrupted) / 5.0
    gate = identity_change <= cfg["engineering_gates"]["identity_mean_normalized_change_max"]
    return {"condition_sameunit_mse": dict(condition_sameunit_mse),
            "corrupted_mean_sameunit_mse": corrupted,
            "identity_mean_sameunit_mse": identity_mse,
            "identity_mean_normalized_change": float(identity_change),
            "selection_loss_sameunit_mse": weighted,
            "identity_gate_passed": bool(gate)}


def evaluate_selection(torch, np, api, model, references, inputs, cfg, device):
    """Evaluate in restored original units; normalized identity change is separate."""
    model.eval()
    metrics, changes, input_metrics = {}, {}, {}
    size = cfg["training"]["batch_size"]
    with torch.inference_mode():
        for condition, values in zip(CONDITIONS, inputs):
            error_sum = change_sum = input_sum = 0.0
            for start in range(0, len(references), size):
                noisy = values[start:start + size]
                ref = references[start:start + size]
                prepared = normalize_batch(api, np, noisy, ref)
                x = torch.from_numpy(prepared["normalized"]).to(device=device, dtype=torch.float32)
                pred = _prediction(torch, model, x)
                pred_np = pred.detach().to("cpu").numpy().astype(np.float32, copy=False)
                restored = _restore_numpy_batch(api, np, noisy, prepared["normalized"], pred_np, prepared["rms"])
                residual = restored - ref.astype(np.float64)
                error_sum += float(np.square(residual).mean(axis=(1, 2)).sum())
                rms = prepared["rms"]
                if rms.ndim == 2:
                    rms = rms[:, :, None]
                change_sum += float(np.square((restored - noisy.astype(np.float64)) / rms).mean(axis=(1, 2)).sum())
                input_sum += float(np.square(noisy.astype(np.float64) - ref.astype(np.float64)).mean(axis=(1, 2)).sum())
            metrics[condition] = error_sum / len(references)
            changes[condition] = change_sum / len(references)
            input_metrics[condition] = input_sum / len(references)
    summary = selection_summary(metrics, changes["identity"], cfg)
    summary["identity_mean_normalized_change_by_condition"] = changes
    summary["input_condition_sameunit_mse"] = input_metrics
    summary["metric_units"] = "sameunit_mse_in_restored_reference_units"
    return summary


def _validate_state(torch, state):
    require(isinstance(state, dict) and state, "empty model state")
    require(all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()),
            "checkpoint state must contain tensors only")
    require(all(v.device.type == "cpu" and bool(torch.isfinite(v).all()) for v in state.values()),
            "checkpoint state must be finite CPU tensors")


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
        loss = fit_epoch(torch, np, api, model, optimizer, fit, epoch_inputs, cfg, device, epoch)
        del epoch_inputs
        candidate = evaluate_selection(torch, np, api, model, selection, fixed_inputs, cfg, device)
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
    _validate_state(torch, state)
    final_metadata = api.preflight(root=root)
    require(final_metadata == metadata and file_sha(Path(root) / LOCK) == metadata["training_lock_sha256"],
            "training bindings changed during run")
    assert_new_outputs(root)
    checkpoint = {"schema_version": 1, "model_version": cfg["model_version"],
        "config": cfg, "config_sha256": canonical_sha(cfg),
        "training_lock_sha256": metadata["training_lock_sha256"], "state_dict": state,
        "selected_epoch": int(selected["epoch"]), "selection_gate_passed": best_eligible is not None,
        "selection_metric": "corrupted_mean_sameunit_mse",
        "runtime_enabled": False, "production_validated": False}
    report = {"schema_version": 1, "status": "training_completed", "model_version": cfg["model_version"],
        "config": cfg, "config_sha256": canonical_sha(cfg),
        "training_lock_sha256": metadata["training_lock_sha256"], "preflight": metadata,
        "started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started, "device": str(device),
        "device_name": torch.cuda.get_device_name(device), "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda, "amp": False, "allow_tf32": False,
        "deterministic_algorithms": True, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "conditions": list(CONDITIONS), "epochs": epochs, "selected_epoch": int(selected["epoch"]),
        "selected_selection": selected, "selection_gate_passed": best_eligible is not None,
        "selection_gate_scope": "identity_mean_normalized_change<=0.002; selection metric is same-unit MSE",
        "selection_reason": "lowest_corrupted_sameunit_mse_among_identity_eligible" if best_eligible is not None
            else "no_identity_eligible_epoch_lowest_sameunit_selection_loss_gate_failed",
        "fit_source_identities": fit_ids, "fit_source_identity_count": len(fit_ids),
        "synthetic_source_identities": synthetic_ids, "synthetic_source_identity_count": len(synthetic_ids),
        "selection_source_identities": selection_ids, "selection_source_identity_count": len(selection_ids),
        "fit_examples_per_epoch": len(fit) * len(CONDITIONS),
        "selection_examples_per_epoch": len(selection) * len(CONDITIONS),
        "seed_policy": {"fit": "seed+one_based_epoch*1000+condition_index",
            "selection": "seed+100000+condition_index", "synthetic": cfg["training"]["seed"],
            "shuffle": "seed+one_based_epoch"},
        "metric_definition": "same-unit MSE after runtime restoration against recorded reference",
        "loss_definition": "mean(((prediction-target_in_noisy_coordinates)*noisy_rms)^2) over channel/sample; identity weight 100, other conditions 1",
        "normalization_contract": "float64 per-noisy-window mean/RMS for input and target; float32 forward; float64 runtime restoration",
        "prior_experiment": cfg["prior_experiment"], "public_evaluation_not_pristine": True,
        "evaluation_is_pristine_project_holdout": False, "all_public_subjects_previously_available_in_project": True,
        "evaluation_subjects_loaded": [], "personal_data_loaded": False,
        "task_labels_enter_forward": False, "raw_reference_is_clean_ground_truth": False,
        "independent_personal_validation": False, "runtime_enabled": False,
        "production_validated": False, "checkpoint": CHECKPOINT}
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    publish_exclusive(Path(root) / CHECKPOINT, lambda stream: torch.save(checkpoint, stream))
    report["checkpoint_sha256"] = file_sha(Path(root) / CHECKPOINT)
    publish_exclusive(Path(root) / REPORT, lambda stream: stream.write(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")))
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="metadata/hash checks only (default)")
    mode.add_argument("--run", action="store_true", help="train on CUDA:0; never overwrite")
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
    except (TrainingV12Error, TrainingV10Error, OSError, ImportError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc), "runtime_enabled": False,
                          "production_validated": False}, ensure_ascii=False, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
