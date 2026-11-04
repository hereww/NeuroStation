"""Frozen v6 training entrypoint. Default: metadata-only; --run requires freeze.

No path/device/cohort/config overrides. Actual training is CUDA:0-only, never
loads v5 weights, and cannot select using outer/holdout waveforms. Internal
engineering gates are not runtime or production acceptance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import io
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _contract():
    return importlib.import_module("eeg_tools.ssvep_v6_contract")


def _core():
    return importlib.import_module("eeg_tools.ssvep_denoise_v6")


def _require_training_preflight(state):
    if getattr(state, "check_only", True) is not False or getattr(state, "stage", None) != "training":
        raise _contract().V6Error("full frozen training preflight required; check-only cannot train/publish")


def _cublas_environment():
    # Execute before importing torch or any module which might initialize CUDA.
    existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing is not None and existing != ":4096:8":
        raise RuntimeError("v6 refuses conflicting CUBLAS_WORKSPACE_CONFIG")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def configure_training(cfg):
    _cublas_environment()
    core = _core()
    torch = core.torch
    if not torch.cuda.is_available():
        raise _contract().V6Error("CUDA:0 required; actual training has no CPU fallback")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    set_seed(cfg["training"]["seed"])
    return "cuda:0"


def set_seed(seed):
    import numpy as np
    torch = _core().torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ReferenceWindowDataset:
    """One fixed native window/trial, all five conditions, and no combinations.

    Requires A-loaded records. Family/subject/trial information is metadata for
    summarization only and is never an argument to model.forward.
    """
    def __init__(self, records, domain, cfg, *, purpose, epoch):
        core, api = _core(), _contract()
        api.validate_config(cfg)
        if domain not in core.DOMAINS or purpose not in ("training", "internal_selection"):
            raise api.V6Error("invalid v6 dataset/purpose")
        self.cfg, self.domain, self.purpose = cfg, domain, purpose
        self.subjects = (1, 2, 3, 4) if purpose == "training" else (5, 6)
        self.records = sorted((r for r in records if r.dataset == domain and r.subject_id in self.subjects),
                              key=lambda r: r.subject_id)
        if [r.subject_id for r in self.records] != list(self.subjects):
            raise api.V6Error("missing/duplicate frozen subjects")
        self.entries = []
        start = cfg["start_samples"][domain]
        for ri, record in enumerate(self.records):
            signal = record.signal
            if (record.sample_rate_hz != cfg["sample_rates_hz"][domain]
                    or signal.ndim != 3 or signal.shape[2] != 8
                    or signal.shape[1] < start + 256):
                raise api.V6Error("invalid frozen record shape/rate")
            for trial in range(signal.shape[0]):
                for condition in core.CONDITIONS:
                    self.entries.append((ri, trial, condition))
        self.set_epoch(epoch)

    def set_epoch(self, epoch):
        if type(epoch) is not int or (self.purpose == "training" and not 1 <= epoch <= 18) or (
                self.purpose == "internal_selection" and epoch != 0):
            raise _contract().V6Error("invalid fixed-purpose corruption epoch")
        self.epoch = epoch

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        import numpy as np
        api = _contract()
        ri, trial, condition = self.entries[index]
        record = self.records[ri]
        start = self.cfg["start_samples"][self.domain]
        ref = np.ascontiguousarray(record.signal[trial, start:start + 256, :].T, dtype=np.float64)
        noisy, seed = api.corrupt_case(ref, self.domain, record.subject_id, trial, condition,
                                      self.cfg, purpose=self.purpose, epoch=self.epoch)
        norm, mean, rms = api.normalize_window(noisy)
        target = ((ref - mean) / rms).astype(np.float32)
        if norm.shape != (8, 256) or not np.isfinite(target).all():
            raise api.V6Error("invalid normalized training window")
        return {"noisy": norm, "target": target, "identity": condition == "identity",
                "ref64": ref, "noisy64": noisy, "rms64": rms,
                "dataset": self.domain, "subject_id": record.subject_id,
                "trial": trial, "condition": condition, "seed": seed, "sample_rate_hz": record.sample_rate_hz}


def run_epoch(model, loader, cfg, device, optimizer=None):
    """Synthetic tests may call on CPU; train_domain/main strictly refuse CPU."""
    core, api = _core(), _contract()
    torch = core.torch
    training = optimizer is not None
    model.train(training)
    accumulator = core.LossAccumulator()
    rows = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            noisy = batch["noisy"].to(device=device, dtype=torch.float32)
            target = batch["target"].to(device=device, dtype=torch.float32)
            identity = batch["identity"].to(device=device, dtype=torch.bool)
            if training:
                optimizer.zero_grad(set_to_none=True)
            pred = model(noisy)  # No condition/target IDs, clean target, or noise-family input.
            losses = core.compute_loss(pred, target, noisy, identity, cfg)
            accumulator.update(losses)
            if training:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise api.V6Error("nonfinite optimizer state/model output")
            else:
                predictions = pred.cpu().numpy()
                normalized = noisy.cpu().numpy()
                for i in range(pred.shape[0]):
                    ref = batch["ref64"][i].cpu().numpy()
                    corrupted = batch["noisy64"][i].cpu().numpy()
                    rms = batch["rms64"][i].cpu().numpy()
                    out = api.restore_window(corrupted, normalized[i], predictions[i], rms)
                    metrics = api.case_metrics(ref, corrupted, out, identity=bool(identity[i].item()),
                                               sample_rate_hz=int(batch["sample_rate_hz"][i]))
                    rows.append({**metrics, "dataset": batch["dataset"][i],
                                 "subject_id": int(batch["subject_id"][i]),
                                 "trial_index": int(batch["trial"][i]), "condition": batch["condition"][i],
                                 "seed": int(batch["seed"][i])})
    return accumulator.result(), rows


def train_domain(records, domain, cfg, device, *, frozen_state):
    api = _contract()
    if device != "cuda:0":
        raise api.V6Error("CUDA:0 required; actual domain training refuses CPU")
    _require_training_preflight(frozen_state)
    core = _core()
    torch = core.torch
    if not torch.cuda.is_available():
        raise api.V6Error("CUDA:0 required; actual domain training refuses CPU")
    api.verify_frozen(frozen_state)
    set_seed(cfg["training"]["seed"])
    fit = ReferenceWindowDataset(records, domain, cfg, purpose="training", epoch=1)
    internal = ReferenceWindowDataset(records, domain, cfg, purpose="internal_selection", epoch=0)
    expected_fit = 3600 if domain == core.DOMAINS[0] else 4800
    expected_internal = 1800 if domain == core.DOMAINS[0] else 2400
    if len(fit) != expected_fit or len(internal) != expected_internal:
        raise api.V6Error("frozen trial/case counts mismatch")
    generator = torch.Generator(device="cpu").manual_seed(cfg["training"]["seed"])
    fit_loader = torch.utils.data.DataLoader(fit, batch_size=64, shuffle=True, num_workers=0,
                                             drop_last=False, generator=generator)
    internal_loader = torch.utils.data.DataLoader(internal, batch_size=64, shuffle=False,
                                                  num_workers=0, drop_last=False)
    model = core.build_model(cfg, cfg["sample_rates_hz"][domain]).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0001,
                                   foreach=False, fused=False)
    history, best_key, best_state = [], None, None
    for epoch in range(1, 19):
        api.verify_frozen(frozen_state)
        fit.set_epoch(epoch)
        train_loss, _ = run_epoch(model, fit_loader, cfg, device, optimizer)
        internal_loss, rows = run_epoch(model, internal_loader, cfg, device)
        if len(rows) != expected_internal:
            raise api.V6Error("incomplete internal restoration")
        summary = api.summarize_cases(rows)
        gates = api.gate_summary(summary, cfg)
        key = core.selection_key(summary, gates, internal_loss["total"], epoch)
        if best_key is None or key < best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        row = {"epoch": epoch, "train_loss": train_loss, "internal_loss": internal_loss,
               "conditions": summary, "gates": gates, "selection_key": list(key),
               "best_epoch": best_key[-1]}
        core._finite_json(row)
        history.append(row)
        print(json.dumps({"dataset": domain, **row}, ensure_ascii=False, allow_nan=False), flush=True)
    # No retry, warmstart, alpha sweep, continuation, or epoch-zero candidate.
    return best_state, {"dataset": domain, "sample_rate_hz": cfg["sample_rates_hz"][domain],
                        "fit_subjects": [1, 2, 3, 4], "internal_validation_subjects": [5, 6],
                        "fit_case_count": len(fit), "internal_case_count": len(internal),
                        "best_epoch": best_key[-1], "best_selection_key": list(best_key),
                        "internal_gates_passed": best_key[0] == 0,
                        "selection_status": "internal_gates_passed" if best_key[0] == 0 else "gates_failed",
                        "history": history}


def make_training_report(cfg, state, records, domain_reports):
    core = _core()
    report = {"schema_version": 1, "status": "formal", "model_version": core.MODEL_VERSION,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "device": "cuda:0",
              "config_sha256": core.CONFIG_SHA256, "training_lock_sha256": state.lock_sha256,
              "preregistration_sha256": core.PREREG_SHA256,
              "reference_manifest_sha256": core.REFERENCE_MANIFEST_SHA256,
              "partition_lock_sha256": core.PARTITION_LOCK_SHA256,
              "source_binding": core.source_binding(records), "domain_reports": domain_reports,
              "sample_rates_hz": dict(core.RATES), "fit_subjects": [1, 2, 3, 4],
              "internal_validation_subjects": [5, 6], "target_definition": cfg["corruption"]["target_definition"],
              "amp": False, "outer_validation_waveforms_loaded": False, **core.PROVENANCE}
    core._finite_json(report)
    return report


def publish_training_bundle(state, models, report):
    """CPU BytesIO self-load with weights_only; A owns exclusive publication."""
    _require_training_preflight(state)
    api, core = _contract(), _core()
    api.verify_frozen(state)
    api.assert_new_outputs(state.root, [api.CHECKPOINT_RELATIVE, api.TRAIN_REPORT_RELATIVE])
    ck = core.checkpoint_metadata(state.config, report, state.lock_sha256)
    ck["models"] = models
    stream = io.BytesIO()
    core.torch.save(ck, stream)
    stream.seek(0)
    loaded = core.torch.load(stream, map_location="cpu", weights_only=True)
    core.validate_checkpoint(loaded, state.config, state.lock_sha256)
    report_bytes = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    api.verify_frozen(state)
    api.publish_bundle(state.root, {api.CHECKPOINT_RELATIVE: stream.getvalue(),
                                    api.TRAIN_REPORT_RELATIVE: report_bytes})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="metadata only; no torch/CUDA/binary/waveform read (default)")
    mode.add_argument("--run", action="store_true", help="run exactly the frozen CUDA experiment after review and freeze")
    args = parser.parse_args(argv)
    api = _contract()
    # Collision rejection precedes config/preflight, torch import, CUDA, or data reads.
    api.assert_new_outputs(ROOT, [api.CHECKPOINT_RELATIVE, api.TRAIN_REPORT_RELATIVE])
    state = api.preflight(root=ROOT, stage="training", check_only=not args.run)
    if not args.run:
        print(json.dumps(state.report(), ensure_ascii=False, allow_nan=False))
        return 0
    _require_training_preflight(state)
    api.verify_frozen(state)
    device = configure_training(state.config)
    api.verify_frozen(state)
    records = api.load_training_records(state)
    core = _core()
    if {(r.dataset, r.subject_id) for r in records} != {
            (domain, sid) for domain in core.DOMAINS for sid in range(1, 7)} or len(records) != 12:
        raise api.V6Error("training requires exactly the 12 frozen subjects1-6")
    states, reports = {}, {}
    for domain in core.DOMAINS:
        states[domain], reports[domain] = train_domain(records, domain, state.config, device,
                                                      frozen_state=state)
    report = make_training_report(state.config, state, records, reports)
    publish_training_bundle(state, states, report)
    print(json.dumps({"status": "training_completed", "best_epochs": {
        d: reports[d]["best_epoch"] for d in core.DOMAINS}, "internal_gates_passed": {
        d: reports[d]["internal_gates_passed"] for d in core.DOMAINS},
        "runtime_enabled": False, "production_validated": False}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


