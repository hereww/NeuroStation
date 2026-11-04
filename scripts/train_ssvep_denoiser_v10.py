"""Frozen v10 CUDA training runner; default invocation checks metadata only.

Training is explicit (--run), requires an independently frozen source/data lock,
and never reads evaluation subjects or personal recordings. No resume/overrides.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHECKPOINT = "models/denoiser_v10_personal.pt"
REPORT = "analysis_outputs/ssvep_v10_training_20261013.json"
LOCK = "analysis_outputs/ssvep_v10_training_lock_20261013.json"
CONDITIONS = ("identity", "drift", "high_burst", "gaussian", "combined")
FIT_SUBJECTS = (1, 2, 3, 4)
SELECTION_SUBJECTS = (5, 6)


class TrainingV10Error(RuntimeError):
    """A contract, numerical or publication failure; never fall back to CPU."""


def require(ok, message):
    if not ok:
        raise TrainingV10Error("v10 trainer: " + message)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_new_outputs(root=ROOT):
    # lexists also rejects broken links. Called BEFORE even importing the data API.
    for name in (CHECKPOINT, REPORT):
        require(not os.path.lexists(Path(root) / name), "output already exists: " + name)


def validate_training_config(cfg):
    require(cfg.get("model_version") == "learned_denoise_v10_personal"
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
    expected = {"seed": 20261013, "epochs": 24, "batch_size": 64,
        "learning_rate": 0.001, "weight_decay": 0.0001, "optimizer": "AdamW",
        "device": "cuda:0", "require_cuda": True, "amp": False,
        "allow_tf32": False, "deterministic_algorithms": True,
        "cpu_threads": 4, "gradient_clip": 1.0, "identity_loss_weight": 20.0}
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
    api = importlib.import_module("eeg_tools.ssvep_v10_data")
    cfg = api.load_config(root=root)
    validate_training_config(cfg)
    state = api.preflight(root=root)
    require(isinstance(state, dict) and state.get("status") == "ready", "preflight is not ready")
    require(state.get("config_sha256") == canonical_sha(cfg), "preflight config mismatch")
    require(state.get("training_lock_sha256") == file_sha(Path(root) / LOCK), "lock binding mismatch")
    require(state.get("waveforms_loaded") is False and state.get("cuda_initialized") is False,
            "metadata preflight loaded waveforms/CUDA")
    return api, cfg, state


def augmentation_seed(seed, condition_index, *, epoch=None):
    """One-based fit epoch; selection cases use a fixed disjoint seed range."""
    require(type(condition_index) is int and 0 <= condition_index < len(CONDITIONS),
            "invalid condition index")
    require(epoch is None or (type(epoch) is int and 1 <= epoch <= 24), "invalid fit epoch")
    return seed + (100000 if epoch is None else epoch * 1000) + condition_index


def selection_summary(condition_nmse, cfg):
    require(set(condition_nmse) == set(CONDITIONS), "selection condition coverage mismatch")
    require(all(type(v) in (float, int) and math.isfinite(v) and v >= 0
                for v in condition_nmse.values()), "nonfinite/negative selection metric")
    corrupted = sum(condition_nmse[c] for c in CONDITIONS[1:]) / 4
    identity = float(condition_nmse["identity"])
    loss = (cfg["training"]["identity_loss_weight"] * identity + 4 * corrupted) / 5
    return {"condition_nmse": dict(condition_nmse), "corrupted_mean_nmse": corrupted,
            "identity_mean_normalized_change": identity, "selection_loss": loss,
            "identity_gate_passed": identity <= cfg["engineering_gates"]["identity_mean_normalized_change_max"]}


def preferred_candidate(candidate, previous, *, eligible):
    if eligible and not candidate["identity_gate_passed"]:
        return False
    key = "corrupted_mean_nmse" if eligible else "selection_loss"
    # Strict improvement preserves the earliest epoch in an exact tie.
    return previous is None or candidate[key] < previous[key]


def _validate_references(np, reference, identities, subjects):
    require(isinstance(reference, np.ndarray) and reference.dtype == np.float32
            and reference.ndim == 3 and reference.shape[1:] == (8, 512)
            and len(reference) > 0 and np.isfinite(reference).all(), "invalid reference tensor")
    require(isinstance(identities, list) and len(identities) == len(reference), "identity count mismatch")
    require(all(isinstance(item, dict) and type(item.get("subject_id")) is int
                and item["subject_id"] in subjects and item.get("dataset") == "Wearable"
                for item in identities), "unexpected source identities/subject")
    require({item["subject_id"] for item in identities} == set(subjects), "missing source subject")
    keys = [(item["subject_id"], item["trial_index"]) for item in identities]
    require(len(set(keys)) == len(keys), "duplicate source cases")
    require(np.max(np.abs(reference.mean(axis=-1))) < 1e-4
            and np.max(np.abs(np.mean(reference.astype(np.float64)**2, axis=-1) - 1)) < 1e-4,
            "references must already be centered and unit RMS; no implicit normalization")
    json.dumps(identities, allow_nan=False)


def load_training_references(api, np, cfg, root=ROOT):
    """Only two literal public cohorts can reach the data loader."""
    fit, fit_ids = api.load_reference_windows(list(FIT_SUBJECTS), root=root)
    _validate_references(np, fit, fit_ids, FIT_SUBJECTS)
    selection, selection_ids = api.load_reference_windows(list(SELECTION_SUBJECTS), root=root)
    _validate_references(np, selection, selection_ids, SELECTION_SUBJECTS)
    count = cfg["augmentation"]["synthetic_training_examples"]
    synthetic = api.synthetic_references(count, cfg["training"]["seed"])
    require(isinstance(synthetic, np.ndarray) and synthetic.dtype == np.float32
            and synthetic.shape == (count, 8, 512) and np.isfinite(synthetic).all(), "invalid synthetic reference")
    require(np.max(np.abs(synthetic.mean(axis=-1))) < 1e-4
            and np.max(np.abs(np.mean(synthetic.astype(np.float64)**2, axis=-1) - 1)) < 1e-4,
            "synthetic references must be centered and unit RMS")
    synthetic_ids = [{"dataset": "synthetic_known_sinusoid_v10", "synthetic_index": i,
                      "seed": cfg["training"]["seed"]} for i in range(count)]
    return (np.concatenate((fit, synthetic)), fit_ids, synthetic_ids, selection, selection_ids)


def condition_inputs(api, np, references, cfg, *, epoch=None):
    arrays = []
    for index, condition in enumerate(CONDITIONS):
        seed = augmentation_seed(cfg["training"]["seed"], index, epoch=epoch)
        corrupted = api.corrupt(references, condition, seed, cfg)
        require(isinstance(corrupted, np.ndarray) and corrupted.dtype == np.float32
                and corrupted.shape == references.shape and np.isfinite(corrupted).all(),
                "invalid corruption output: " + condition)
        if condition == "identity":
            require(np.array_equal(corrupted, references), "identity corruption must preserve reference")
        arrays.append(corrupted)
    return arrays


def initialize_cuda(cfg):
    # Set before Torch import/CUDA context creation. Never initialize CUDA in check-only.
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    require(workspace in (None, ":4096:8", ":16:8"), "incompatible CUBLAS_WORKSPACE_CONFIG")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = importlib.import_module("torch")
    require(torch.cuda.is_available() and torch.cuda.device_count() >= 1,
            "CUDA required; CPU fallback is forbidden")
    torch.set_num_threads(cfg["training"]["cpu_threads"])
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    seed = cfg["training"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(cfg["training"]["device"])
    require(device.type == "cuda" and device.index == 0, "only configured CUDA:0 is allowed")
    return torch, device


def _prediction(torch, model, x):
    # Condition IDs and task labels never enter model.forward.
    prediction = model(x)
    require(isinstance(prediction, torch.Tensor) and prediction.shape == x.shape
            and prediction.dtype == torch.float32 and prediction.device == x.device,
            "model must return same-shape/device float32 prediction")
    require(bool(torch.isfinite(prediction).all()), "model produced nonfinite predictions")
    return prediction


def fit_epoch(torch, np, model, optimizer, refs, inputs, cfg, device, epoch):
    model.train()
    n = len(refs)
    # Equal case count per condition, followed by a reproducible joint shuffle.
    rng = np.random.default_rng(cfg["training"]["seed"] + epoch)
    order = rng.permutation(n * len(CONDITIONS))
    total_loss = 0.0
    batch_size = cfg["training"]["batch_size"]
    for start in range(0, len(order), batch_size):
        ids = order[start:start + batch_size]
        # Gather one batch without duplicating the entire five-condition CPU pool.
        batch = np.empty((len(ids), 8, 512), dtype=np.float32)
        for condition_index, values in enumerate(inputs):
            mask = ids // n == condition_index
            batch[mask] = values[ids[mask] % n]
        x = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
        target = torch.from_numpy(refs[ids % n]).to(device=device, dtype=torch.float32)
        weights = np.where(ids // n == 0, cfg["training"]["identity_loss_weight"], 1.0).astype(np.float32)
        weight = torch.from_numpy(weights).to(device=device)
        optimizer.zero_grad(set_to_none=True)
        pred = _prediction(torch, model, x)
        per_case = (pred - target).square().mean(dim=(1, 2))
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


def evaluate_selection(torch, model, references, inputs, cfg, device):
    model.eval()
    metrics, input_metrics = {}, {}
    size = cfg["training"]["batch_size"]
    with torch.inference_mode():
        for condition, values in zip(CONDITIONS, inputs):
            error_sum = input_sum = 0.0
            for start in range(0, len(references), size):
                x = torch.from_numpy(values[start:start + size]).to(device=device, dtype=torch.float32)
                target = torch.from_numpy(references[start:start + size]).to(device=device, dtype=torch.float32)
                pred = _prediction(torch, model, x)
                error_sum += float((pred.double() - target.double()).square().mean((1, 2)).sum().item())
                input_sum += float((x.double() - target.double()).square().mean((1, 2)).sum().item())
            metrics[condition] = error_sum / len(references)
            input_metrics[condition] = input_sum / len(references)
    summary = selection_summary(metrics, cfg)
    summary["input_condition_nmse"] = input_metrics
    return summary


def _snapshot(model):
    return {key: value.detach().to("cpu").clone() for key, value in model.state_dict().items()}


def publish_exclusive(path, writer):
    """Publish a fully flushed same-volume temp file by an exclusive hard link.

    os.link fails if destination exists (including races); it never replaces it.
    A failed report publication intentionally leaves the checkpoint without a
    success report, so an incomplete run cannot masquerade as a complete bundle.
    """
    path = Path(path)
    require(not os.path.lexists(path), "output already exists: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_training(api, cfg, metadata, root=ROOT):
    assert_new_outputs(root)
    validate_training_config(cfg)
    require(metadata.get("config_sha256") == canonical_sha(cfg)
            and metadata.get("training_lock_sha256") == file_sha(Path(root) / LOCK),
            "training metadata/config/lock mismatch")
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    torch, device = initialize_cuda(cfg)
    np = importlib.import_module("numpy")
    np.random.seed(cfg["training"]["seed"])
    model_api = importlib.import_module("eeg_tools.ssvep_denoise_v10")
    model = model_api.build_model(cfg).to(device=device, dtype=torch.float32)
    require(all(p.device == device and p.dtype == torch.float32 for p in model.parameters()),
            "model is not float32 CUDA:0")
    fit, fit_ids, synthetic_ids, selection, selection_ids = load_training_references(api, np, cfg, root)
    fixed_inputs = condition_inputs(api, np, selection, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                 weight_decay=cfg["training"]["weight_decay"])
    epochs = []
    best_eligible = best_fallback = None
    eligible_state = fallback_state = None
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        epoch_inputs = condition_inputs(api, np, fit, cfg, epoch=epoch)
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
        "loss_definition": "mean per-case NMSE with identity weight 20 and other condition weights 1; equal cases per condition",
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
    except (TrainingV10Error, OSError, ImportError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc), "runtime_enabled": False,
                          "production_validated": False}, ensure_ascii=False, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
