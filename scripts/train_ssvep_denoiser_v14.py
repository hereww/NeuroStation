"""Independent v14 denoiser experiment driver.

Default and ``--check-only`` are metadata-only.  ``--freeze-only`` performs a
formal, non-writing dry freeze by default and returns the prospective lock
bindings.  A caller that explicitly chooses ``freeze_only(..., write_lock=True)``
may create the 20261014 lock, but this implementation pass does not do so.
``--run`` requires an existing lock, CUDA:0, deterministic no-AMP/no-TF32
settings, and exclusively empty checkpoint/report/manifest destinations.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eeg_tools import ssvep_v14_runtime as api


def exclusive_json(root, name, value):
    encoded = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    target = api.safe_path(root, name, new=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    target = api.safe_path(root, name, new=True)
    with target.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return target


def assert_new_outputs(root, *, include_lock=False):
    names = [api.MODEL, api.MANIFEST, api.REPORT]
    if include_lock:
        names.insert(0, api.LOCK)
    for name in names:
        api.safe_path(root, name, new=True)


def prospective_lock(root, cfg):
    files = {name: api.sha(api.safe_path(root, name)) for name in api.bindings(cfg)}
    return {
        "schema_version": 1,
        "status": "formal",
        "model_version": cfg["model_version"],
        "config_sha256": api.canonical_sha(cfg),
        "research_plan_path": api.PLAN,
        "research_plan_sha256": api.plan_sha(cfg),
        "files": files,
        "roles": {"fit": [1, 2, 3, 4], "select": [5, 6], "personal": [], "evaluation": []},
        "synthetic": cfg["synthetic"],
        "normalization_contract": api.NORMALIZATION,
        "runtime_enabled": False,
        "production_validated": False,
        "reference_disclosure": "public sources are recorded proxies, long sources are artificial numerical cases; no personal fit/selection",
    }


def check_only(root=ROOT):
    cfg = api.load_config(root)
    api.validate_plan(root, cfg)
    names = api.bindings(cfg)
    api.require(set(names).isdisjoint((api.LOCK, api.MODEL, api.MANIFEST, api.REPORT)),
                "source/output collision")
    for name in names:
        api.require(api.safe_path(root, name).is_file(), "missing source: " + name)
    if api.safe_path(root, api.LOCK).exists():
        result = api.preflight(root)
        result["status"] = "ready"
    else:
        result = {
            "status": "ready_to_freeze",
            "config_sha256": api.canonical_sha(cfg),
            "research_plan_sha256": api.plan_sha(cfg),
            "source_count": len(names),
            "waveforms_loaded": False,
            "torch_imported": False,
            "cuda_initialized": False,
            "runtime_enabled": False,
            "production_validated": False,
        }
    assert_new_outputs(root)
    return result


def freeze_only(root=ROOT, *, write_lock=False):
    """Validate and optionally write only a new formal source lock.

    The CLI calls this with ``write_lock=False`` so review/check runs cannot
    create artifacts.  Any explicit lock write is exclusive and hash-bound.
    """
    cfg = api.load_config(root)
    api.validate_plan(root, cfg)
    lock = prospective_lock(root, cfg)
    if write_lock:
        assert_new_outputs(root, include_lock=True)
        exclusive_json(root, api.LOCK, lock)
        return api.preflight(root)
    check_only(root)
    return {
        "status": "ready_to_freeze",
        "would_write": api.LOCK,
        "config_sha256": lock["config_sha256"],
        "research_plan_sha256": lock["research_plan_sha256"],
        "bound_source_count": len(lock["files"]),
        "roles": lock["roles"],
        "runtime_enabled": False,
        "production_validated": False,
        "wrote_files": [],
    }


def _read_locked_public(subjects, cfg, root):
    import numpy as np
    api.require(list(subjects) in (cfg["data"]["fit_subjects"], cfg["data"]["selection_subjects"]),
                "only declared fit/select cohorts may be read by trainer")
    lock = json.loads(api.safe_path(root, api.LOCK).read_text(encoding="utf-8"))
    refs, sources = [], []
    for subject in subjects:
        name = f"{cfg['data']['reference_directory']}/subject_{subject:03d}.npz"
        manifest_name = name + ".manifest.json"
        api.require(api.sha(api.safe_path(root, name)) == lock["files"][name]
                    and api.sha(api.safe_path(root, manifest_name)) == lock["files"][manifest_name],
                    "locked public source changed")
        manifest = json.loads(api.safe_path(root, manifest_name).read_text(encoding="utf-8"))
        api.require(manifest["subject_id"] == subject and manifest["dataset"] == "Wearable"
                    and manifest["native_sample_rate_hz"] == 250
                    and manifest["raw_reference_is_clean_ground_truth"] is False,
                    "public provenance mismatch")
        with np.load(api.safe_path(root, name), allow_pickle=False) as archive:
            raw = np.asarray(archive["reference"], dtype=np.float64)
        api.require(raw.shape == (240, 710, 8) and np.isfinite(raw).all(), "public layout mismatch")
        ref = api.condition_line(raw)[:, cfg["data"]["window_start_sample"]:cfg["data"]["window_end_sample_exclusive"], :]
        ref = ref.transpose(0, 2, 1).copy()
        ref -= ref.mean(axis=-1, keepdims=True)
        scale = np.sqrt(np.mean(ref ** 2, axis=-1, keepdims=True))
        api.require(np.isfinite(scale).all() and np.all(scale > api.RMS_FLOOR), "flat public reference")
        refs.append(ref / scale)
        sources.extend((name, manifest_name))
    return np.concatenate(refs), sources


def synthesize_reference(count, cfg, seed):
    """Generate continuous 4096-point artificial numerical reference records."""
    import numpy as np
    api.require(type(count) is int and count > 0 and type(seed) is int, "invalid synthetic request")
    n = cfg["synthetic"]["record_samples"]
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / cfg["sample_rate_hz"]
    values = []
    for _ in range(count):
        frequency = rng.uniform(*cfg["synthetic"]["fundamental_hz"], 8)
        phase = rng.uniform(0, 2 * np.pi, 8)
        reference = np.sin(2 * np.pi * frequency[:, None] * t + phase[:, None])
        reference += 0.25 * np.sin(2 * np.pi * (2 * frequency)[:, None] * t + phase[:, None] / 2)
        background = rng.normal(size=(8, n))
        background -= background.mean(axis=-1, keepdims=True)
        background /= np.sqrt(np.mean(background ** 2, axis=-1, keepdims=True))
        reference += 0.02 * background + rng.uniform(-0.3, 0.3, (8, 1))
        values.append(reference)
    return np.stack(values).astype(np.float64)


def corrupt(reference, condition, seed, cfg):
    """Create corruption outside forward; labels and conditions never enter model."""
    import numpy as np
    ref = api._windows(reference, "reference")
    api.require(condition in api.CONDITIONS and type(seed) is int and seed >= 0, "invalid corruption request")
    if condition == "identity":
        return ref.copy()
    rng = np.random.default_rng(seed)
    count, channels, n = ref.shape
    t = np.arange(n, dtype=np.float64) / cfg["sample_rate_hz"]
    rms = np.sqrt(np.mean((ref - ref.mean(-1, keepdims=True)) ** 2, axis=-1, keepdims=True))
    api.require(np.all(rms > api.RMS_FLOOR), "flat corruption reference")
    aug, noise = cfg["augmentation"], np.zeros_like(ref)

    def centered_scaled(wave, amplitude):
        wave = wave - wave.mean(-1, keepdims=True)
        scale = np.sqrt(np.mean(wave ** 2, axis=-1, keepdims=True))
        api.require(np.all(scale > 1e-12), "degenerate synthetic artifact")
        return wave / scale * rms * amplitude

    if condition in ("drift", "combined"):
        frequency = rng.uniform(*aug["drift_hz"], (count, channels, 1))
        drift = np.sin(2 * np.pi * frequency * t + rng.uniform(0, 2 * np.pi, (count, channels, 1)))
        noise += centered_scaled(drift, rng.uniform(*aug["drift_rms"], (count, channels, 1)))
        trend = np.linspace(-1, 1, n)[None, None, :]
        noise += centered_scaled(trend, rng.uniform(*aug["trend_rms"], (count, channels, 1)))
    if condition in ("high_burst", "combined"):
        for i in range(count):
            duration = max(4, int(n * rng.uniform(*aug["burst_duration_fraction"])))
            start = int(rng.integers(0, n - duration + 1))
            wave = np.sin(2 * np.pi * rng.uniform(*aug["burst_hz"], (channels, 1)) * t[start:start + duration]
                          + rng.uniform(0, 2 * np.pi, (channels, 1))) * np.hanning(duration)[None, :]
            wave -= wave.mean(-1, keepdims=True)
            wave /= np.sqrt(np.mean(wave ** 2, axis=-1, keepdims=True))
            noise[i, :, start:start + duration] += wave * rms[i] * rng.uniform(*aug["burst_rms"], (channels, 1))
    if condition in ("gaussian", "combined"):
        noise += centered_scaled(rng.normal(size=ref.shape), rng.uniform(*aug["gaussian_rms"], (count, channels, 1)))
    out = ref + noise
    api.require(np.isfinite(out).all(), "corruption overflow")
    return out


def cases_for_domain(reference, seed, cfg):
    import numpy as np
    ref = api._windows(reference, "reference")
    children = np.random.SeedSequence(seed).generate_state(len(api.CONDITIONS))
    cases = {}
    for condition, child in zip(api.CONDITIONS, children):
        raw = corrupt(ref, condition, int(child), cfg)
        notch = api.condition_line(raw.transpose(0, 2, 1)).transpose(0, 2, 1).copy()
        cases[condition] = (raw, notch)
    return cases


def _record_stats_batch(windows):
    import numpy as np
    means, rms = [], []
    for item in windows:
        mean, scale = api.record_statistics(item.T)
        means.append(mean)
        rms.append(scale)
    return np.stack(means), np.stack(rms)


def _normalize_domain(noisy, scope):
    import numpy as np
    if scope == "window_512":
        return api.normalize_noisy_window(noisy)
    means, rms = _record_stats_batch(noisy)
    return api.normalize_with_record_stats(noisy, means, rms), means, rms


def selection_summary(domains, cfg):
    api.require(set(domains) == {"public", "long"}, "both selection domains required")
    objectives, gates = [], {}
    for name, metrics in domains.items():
        api.require(set(metrics["conditions"]) == set(api.CONDITIONS), "missing selection condition")
        identity = metrics["identity_normalized_change"]
        api.require(math.isfinite(identity) and identity >= 0, "invalid identity metric")
        gates[name] = identity <= cfg["selection"]["identity_normalized_change_max"]
        errors = [metrics["conditions"][c]["model_mse"] for c in api.CONDITIONS[1:]]
        api.require(all(math.isfinite(v) and v >= 0 for v in errors), "invalid selection MSE")
        objectives.append(sum(errors) / len(errors))
    return {"balanced_corrupted_mse": sum(objectives) / 2,
            "identity_gates": gates, "selection_gate_passed": all(gates.values()),
            "domains": domains}


def preferred_candidate(metrics, previous_score):
    return metrics["selection_gate_passed"] and (previous_score is None or
            metrics["balanced_corrupted_mse"] < previous_score)


def evaluate_selection(model, references, cases, cfg):
    import numpy as np
    model.eval()
    domains = {}
    for name in ("public", "long"):
        ref = references[name]
        scope = "window_512" if ref.shape[-1] == 512 else "record_4096"
        conditions, identity_change = {}, None
        for condition in api.CONDITIONS:
            raw, notch = cases[name][condition]
            if scope == "window_512":
                output = api.process_windows(notch, model, scope, batch_size=cfg["training"]["batch_size"])
                _, _, rms = api.normalize_noisy_window(notch)
            else:
                means, rms = _record_stats_batch(notch)
                output = api.process_windows(notch, model, scope, record_stats=(means, rms),
                                             batch_size=cfg["training"]["batch_size"])
            conditions[condition] = {
                "raw_mse": float(np.mean((raw - ref) ** 2)),
                "notch_mse": float(np.mean((notch - ref) ** 2)),
                "model_mse": float(np.mean((output - ref) ** 2)),
                "fixed_notch_mse_reduction": float(np.mean((raw - ref) ** 2) - np.mean((notch - ref) ** 2)),
                "learned_mse_reduction": float(np.mean((notch - ref) ** 2) - np.mean((output - ref) ** 2)),
            }
            if condition == "identity":
                if scope == "window_512":
                    identity_change = float(np.mean(((output - ref) / rms) ** 2))
                else:
                    identity_change = float(np.mean(((output - ref) / rms) ** 2))
        domains[name] = {"count": len(ref), "window_samples": ref.shape[-1],
                         "normalization_scope": scope, "conditions": conditions,
                         "identity_normalized_change": identity_change}
    return selection_summary(domains, cfg)


def fit_epoch(model, optimizer, references, cases, cfg, epoch):
    """One deterministic equal-domain update pass; personal data is unreachable."""
    import numpy as np
    import torch
    model.train()
    batch_size = cfg["training"]["batch_size"]
    rng = np.random.default_rng(np.random.SeedSequence([cfg["training"]["seed"], epoch, 91]))
    orders = {name: rng.permutation(len(ref) * len(api.CONDITIONS)) for name, ref in references.items()}
    steps = max(math.ceil(len(v) / batch_size) for v in orders.values())
    cursors = {name: 0 for name in references}
    total, updates = 0.0, 0
    for _ in range(steps):
        for name in ("public", "long"):
            order = orders[name]
            if cursors[name] >= len(order):
                orders[name] = order = rng.permutation(len(order))
                cursors[name] = 0
            indices = order[cursors[name]:cursors[name] + batch_size]
            cursors[name] += len(indices)
            ref = references[name]
            which, rows = indices // len(ref), indices % len(ref)
            noisy = np.stack([cases[name][api.CONDITIONS[int(c)]][1][int(r)] for c, r in zip(which, rows)])
            target = ref[rows]
            if noisy.shape[-1] == 512:
                normalized, _, rms = api.normalize_noisy_window(noisy)
            else:
                means, rms = _record_stats_batch(noisy)
                normalized = api.normalize_with_record_stats(noisy, means, rms)
            x = torch.from_numpy(normalized).to("cuda:0")
            noisy_t = torch.from_numpy(noisy).to("cuda:0", dtype=torch.float64)
            target_t = torch.from_numpy(target).to("cuda:0", dtype=torch.float64)
            rms_t = torch.from_numpy(rms).to("cuda:0", dtype=torch.float64)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            losses = api.restored_mse_torch(prediction, x, noisy_t, rms_t, target_t)
            weights = torch.full_like(losses, 1.0)
            weights[which == 0] = cfg["training"]["identity_loss_weight"]
            loss = (losses * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())
            updates += 1
    return {"weighted_restored_mse": total / updates, "optimizer_updates": updates,
            "equal_domain_updates": True, "personal_data_used": False}


def run_training(root=ROOT):
    cfg, metadata = api.load_config(root), api.preflight(root)
    assert_new_outputs(root)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    api.require(torch.cuda.is_available(), "CUDA required; no implicit CPU training")
    torch.set_num_threads(cfg["training"]["cpu_threads"])
    random.seed(cfg["training"]["seed"]); np.random.seed(cfg["training"]["seed"])
    torch.manual_seed(cfg["training"]["seed"]); torch.cuda.manual_seed_all(cfg["training"]["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    model = api.build_model(cfg).to(device="cuda:0", dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])
    public_fit, fit_sources = _read_locked_public(cfg["data"]["fit_subjects"], cfg, root)
    public_select, select_sources = _read_locked_public(cfg["data"]["selection_subjects"], cfg, root)
    long_fit = synthesize_reference(cfg["synthetic"]["fit_count"], cfg, cfg["synthetic"]["fit_seed"])
    long_select = synthesize_reference(cfg["synthetic"]["selection_count"], cfg, cfg["synthetic"]["selection_seed"])
    references_fit = {"public": public_fit, "long": long_fit}
    references_select = {"public": public_select, "long": long_select}
    fit_cases = {name: cases_for_domain(ref, [cfg["training"]["seed"], 1, i], cfg)
                 for i, (name, ref) in enumerate(references_fit.items())}
    select_cases = {name: cases_for_domain(ref, [cfg["training"]["seed"], 2, i], cfg)
                    for i, (name, ref) in enumerate(references_select.items())}
    history, best_state, best_score, selected_epoch = [], None, None, None
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        fit_metrics = fit_epoch(model, optimizer, references_fit, fit_cases, cfg, epoch)
        metrics = evaluate_selection(model, references_select, select_cases, cfg)
        row = {"epoch": epoch, "fit": fit_metrics, "selection": metrics}
        print(json.dumps(row, allow_nan=False), flush=True); history.append(row)
        if preferred_candidate(metrics, best_score):
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_score, selected_epoch = metrics["balanced_corrupted_mse"], epoch
    api.require(api.preflight(root) == metadata, "frozen sources changed during training")
    eligible = best_state is not None
    if not eligible:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    common = {"schema_version": 1, "status": "formal", "model_version": cfg["model_version"],
              "config_sha256": metadata["config_sha256"],
              "training_lock_sha256": metadata["training_lock_sha256"],
              "research_plan_sha256": api.plan_sha(cfg), "selected_epoch": selected_epoch,
              "selection_gate_passed": eligible, "runtime_enabled": False,
              "production_validated": False}
    report = {**common, "status": "training_completed", "parameter_count": sum(p.numel() for p in model.parameters()),
              "training_device": "cuda:0", "device_name": torch.cuda.get_device_name(0),
              "torch_version": str(torch.__version__), "epochs": cfg["training"]["epochs"],
              "normalization_contract": api.NORMALIZATION, "history": history,
              "public_sources": {"fit": fit_sources, "select": select_sources},
              "personal_data_used": False, "public7_8_used": False,
              "checkpoint_role": "selected_research" if eligible else "diagnostic_last_epoch_no_eligible_checkpoint",
              "selection_gate_authorizes_default_deployment": False}
    model_path = api.safe_path(root, api.MODEL, new=True); model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = api.safe_path(root, api.MODEL, new=True)
    with model_path.open("xb") as stream:
        torch.save({**common, "config": cfg, "state_dict": best_state}, stream)
        stream.flush(); os.fsync(stream.fileno())
    report_path = exclusive_json(root, api.REPORT, report)
    exclusive_json(root, api.MANIFEST, {**common, "checkpoint_sha256": api.sha(model_path),
                                        "training_report_sha256": api.sha(report_path),
                                        "checkpoint_role": report["checkpoint_role"],
                                        "normalization_contract": api.NORMALIZATION})
    return {"status": "training_completed", "selected_epoch": selected_epoch,
            "selection_gate_passed": eligible, "checkpoint": api.MODEL,
            "manifest": api.MANIFEST, "report": api.REPORT,
            "runtime_enabled": False, "production_validated": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check-only", action="store_true")
    group.add_argument("--freeze-only", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_training() if args.run else freeze_only() if args.freeze_only else check_only()
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "rejected", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
