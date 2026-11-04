"""Frozen v6 window model and checkpoint contract; no dataset/weight I/O here.

The 0.5 bound applies before FFT projection, not to the final pointwise
residual. Post-projection clipping is deliberately forbidden. This offline,
noncausal window model is not a streaming/task-preservation acceptance claim.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path

# Must precede torch import, including CPU-only import and checkpoint validation.
_CUBLAS = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG", _CUBLAS) != _CUBLAS:
    raise RuntimeError("v6 refuses conflicting CUBLAS_WORKSPACE_CONFIG")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = "learned_denoise_v6_ssvep"
MODEL_TYPE = "zero_initialized_band_projected_residual_tcn"
CONFIG_SHA256 = "f1deaf6f28add5f14f24719675fb071f1c27c62befc7bafafc47c3e8c66eb8f0"
PREREG_SHA256 = "6d98383db26f7afb92302fac4371eb40c3dd31539f282ef1a37fe7440d5c07ec"
REFERENCE_MANIFEST_SHA256 = "607617315889697a5bfcea986bf055bf4ed31182e8a69776842c736f481604b6"
PARTITION_LOCK_SHA256 = "cb5e05b9d33fb3034f178392f65817185226756cf4c609dcfec53b4c2d4b8a8f"
DOMAINS = ("Nakanishi2015", "Wearable")
RATES = {"Nakanishi2015": 256, "Wearable": 250}
CONDITIONS = ("identity", "transient", "low_frequency_drift", "high_frequency_burst", "common_mode")
PROVENANCE = {
    "clean_ground_truth_used": False,
    "holdout_waveforms_loaded": False,
    "outer_validation_used_for_selection": False,
    "runtime_enabled": False,
    "production_validated": False,
}
CHECKPOINT_KEYS = {
    "schema_version", "status", "model_version", "model_type", "config",
    "config_sha256", "training_lock_sha256", "reference_manifest_sha256",
    "partition_lock_sha256", "training_report", "models", *PROVENANCE,
}


def contract():
    """Import A's contract only when an operation actually requires it."""
    return importlib.import_module("eeg_tools.ssvep_v6_contract")


def fail(message):
    raise contract().V6Error(message)


def _finite_json(value):
    """Reject tensors, NumPy scalars, NaN, and unexpected metadata types."""
    if type(value) is dict:
        if any(type(k) is not str for k in value):
            raise ValueError("metadata keys must be strings")
        for item in value.values():
            _finite_json(item)
    elif type(value) in (list, tuple):
        for item in value:
            _finite_json(item)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite metadata")
    elif type(value) not in (str, int, bool, type(None)):
        raise ValueError("unexpected metadata type")


def _exact(left, right):
    # Python == alone incorrectly equates True/1 and False/0.
    return type(left) is type(right) and (
        set(left) == set(right) and all(_exact(left[k], right[k]) for k in left)
        if type(left) is dict else
        len(left) == len(right) and all(_exact(a, b) for a, b in zip(left, right))
        if type(left) in (list, tuple) else left == right
    )


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _model_config(cfg, sample_rate_hz):
    expected = {
        "hidden_channels": 32, "kernel_size": 7,
        "dilations": [1, 2, 4, 8, 16, 32], "group_norm_groups": 8,
        "dropout": 0.0, "pre_projection_residual_limit_normalized": 0.5,
        "zero_initialize_output": True, "protected_band_hz": [8.0, 40.0],
        "projection": "rfft_remove_inclusive_protected_band_then_irfft_n256",
        "post_projection_clip": False,
    }
    if (not _exact(cfg.get("model"), expected)
            or type(cfg.get("input_channels")) is not int or cfg["input_channels"] != 8
            or type(cfg.get("window_samples")) is not int or cfg["window_samples"] != 256
            or not _exact(cfg.get("sample_rates_hz"), RATES)
            or type(sample_rate_hz) is not int or sample_rate_hz not in RATES.values()):
        raise ValueError("v6 requires the frozen architecture and native sample rate")


class ResidualBlock(nn.Module):
    """Same two-convolution residual topology as v5; entirely new state."""
    def __init__(self, dilation):
        super().__init__()
        padding = dilation * 3
        self.net = nn.Sequential(
            nn.Conv1d(32, 32, 7, padding=padding, dilation=dilation),
            nn.GroupNorm(8, 32), nn.GELU(), nn.Dropout(0.0),
            nn.Conv1d(32, 32, 7, padding=padding, dilation=dilation),
            nn.GroupNorm(8, 32),
        )

    def forward(self, x):
        return x + 0.5 * self.net(x)


class SSVEPDenoiserV6(nn.Module):
    def __init__(self, cfg, sample_rate_hz):
        super().__init__()
        _model_config(cfg, sample_rate_hz)
        self.sample_rate_hz = sample_rate_hz
        self.input = nn.Conv1d(8, 32, 7, padding=3)
        self.blocks = nn.Sequential(*(ResidualBlock(d) for d in cfg["model"]["dilations"]))
        self.activation = nn.GELU()
        self.output = nn.Conv1d(32, 8, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        frequencies = torch.fft.rfftfreq(256, d=1.0 / sample_rate_hz)
        self.register_buffer("keep_mask", (frequencies < 8) | (frequencies > 40), persistent=False)

    def forward(self, x):
        if (not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[0] == 0
                or tuple(x.shape[1:]) != (8, 256) or x.dtype != torch.float32
                or not bool(torch.isfinite(x).all())):
            raise ValueError("v6 forward requires finite float32 [B,8,256]")
        delta_pre = 0.5 * torch.tanh(self.output(self.activation(self.blocks(self.input(x)))))
        delta = torch.fft.irfft(torch.fft.rfft(delta_pre, dim=-1) * self.keep_mask,
                               n=256, dim=-1)
        pred = x + delta
        if not bool(torch.isfinite(pred).all()):
            raise ValueError("nonfinite v6 prediction")
        return pred


def build_model(cfg, sample_rate_hz):
    """Return a fresh CPU float32 model; caller explicitly moves to CPU/GPU."""
    contract().validate_config(cfg)
    return SSVEPDenoiserV6(cfg, sample_rate_hz)


def compute_loss(pred, target, noisy, identity, cfg):
    """Group means plus sufficient statistics for case-weighted aggregation."""
    contract().validate_config(cfg)
    expected = (1.0, 0.01, 20.0, 0.001, 1e-8)
    keys = ("corrupted_case_relative_mse_weight", "relative_mse_input_floor",
            "identity_normalized_mse_weight", "normalized_change_penalty_weight",
            "target_channel_variance_epsilon")
    if not _exact(tuple(cfg["loss"][k] for k in keys), expected):
        raise ValueError("v6 loss parameters are frozen")
    if (pred.shape != target.shape or pred.shape != noisy.shape or pred.ndim != 3
            or tuple(pred.shape[1:]) != (8, 256) or pred.shape[0] == 0
            or identity.dtype != torch.bool or identity.shape != (pred.shape[0],)
            or any(t.device != pred.device for t in (target, noisy, identity))
            or any(t.dtype != torch.float32 or not bool(torch.isfinite(t).all())
                   for t in (pred, target, noisy))):
        raise ValueError("invalid v6 loss input")
    variance = (target - target.mean(-1, keepdim=True)).square().mean(-1, keepdim=True) + 1e-8
    out_error = ((pred - target).square() / variance).mean((1, 2))
    in_error = ((noisy - target).square() / variance).mean((1, 2))
    ratio = out_error / in_error.clamp(min=0.01)
    change_case = (pred - noisy).square().mean((1, 2))
    n_identity = int(identity.sum().item())
    n_corrupted = pred.shape[0] - n_identity
    corrupted_sum = ratio[~identity].sum()
    identity_sum = out_error[identity].sum()
    change_sum = change_case.sum()
    corrupted = corrupted_sum / max(n_corrupted, 1)
    identity_loss = identity_sum / max(n_identity, 1)
    change = change_sum / pred.shape[0]
    total = corrupted + 20.0 * identity_loss + 0.001 * change
    values = {"total": total, "corrupted": corrupted, "identity": identity_loss,
              "change": change, "corrupted_sum": corrupted_sum,
              "identity_sum": identity_sum, "change_sum": change_sum,
              "corrupted_count": n_corrupted, "identity_count": n_identity,
              "case_count": pred.shape[0]}
    if not all(bool(torch.isfinite(v)) for v in values.values() if isinstance(v, torch.Tensor)):
        raise ValueError("nonfinite v6 loss")
    return values


class LossAccumulator:
    """Never average batch means when identity/corrupted counts differ."""
    def __init__(self):
        self.sums = {k: 0.0 for k in ("corrupted", "identity", "change")}
        self.counts = {k: 0 for k in ("corrupted", "identity", "case")}

    def update(self, losses):
        for key in self.sums:
            self.sums[key] += float(losses[key + "_sum"].detach().cpu())
        for key in self.counts:
            self.counts[key] += losses[key + "_count"]

    def result(self):
        if self.counts["case"] <= 0:
            raise ValueError("empty v6 epoch")
        values = {"corrupted": self.sums["corrupted"] / max(self.counts["corrupted"], 1),
                  "identity": self.sums["identity"] / max(self.counts["identity"], 1),
                  "change": self.sums["change"] / self.counts["case"]}
        values["total"] = values["corrupted"] + 20.0 * values["identity"] + 0.001 * values["change"]
        values.update({k + "_count": v for k, v in self.counts.items()})
        _finite_json(values)
        return values


def selection_key(summary, gates, internal_total_loss, epoch):
    """Eligible first, worst ratio, identity MSE, total loss, earliest epoch."""
    if type(epoch) is not int or not 1 <= epoch <= 18:
        raise ValueError("only frozen epochs 1..18 are selection candidates")
    conditions = summary
    ratios = []
    for condition in CONDITIONS[1:]:
        entry = conditions[condition]
        baseline, output = float(entry["mean_input_normalized_mse"]), float(entry["mean_output_normalized_mse"])
        if not math.isfinite(baseline) or baseline <= 0 or not math.isfinite(output) or output < 0:
            raise ValueError("invalid selection errors")
        ratios.append(output / baseline)  # No mean-of-ratios and no alpha selection.
    identity = float(conditions["identity"]["mean_output_normalized_mse"])
    if type(gates["all_passed"]) is not bool:
        raise ValueError("gate result must be bool")
    key = (0 if gates["all_passed"] else 1, max(ratios), identity,
           float(internal_total_loss), epoch)
    _finite_json(key)
    if identity < 0 or key[3] < 0:
        raise ValueError("negative selection loss")
    return key


def _frozen_prereg():
    # Metadata JSON only; this never opens NPZ/PT or any data root.
    api = contract()
    raw = api.safe_path(ROOT, api.PREREG_RELATIVE).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PREREG_SHA256:
        fail("v6 preregistration SHA mismatch")
    return json.loads(raw)


def source_binding(records):
    return [{"dataset": r.dataset, "subject_id": r.subject_id,
             "sample_rate_hz": r.sample_rate_hz, "npz_sha256": r.npz_sha256,
             "sidecar_sha256": r.sidecar_sha256}
            for r in sorted(records, key=lambda r: (r.dataset, r.subject_id))]


def _validate_summary(summary, case_count):
    fields = {"case_count", "mean_input_normalized_mse", "mean_output_normalized_mse",
              "ratio_of_mean_reduction", "mean_identity_normalized_change",
              "mean_protected_complex_energy_ratio", "mean_protected_weighted_abs_phase_radians",
              "undefined_phase_case_count"}
    if type(summary) is not dict or set(summary) != set(CONDITIONS):
        fail("incomplete condition summary")
    for condition, entry in summary.items():
        if type(entry) is not dict or set(entry) != fields or not _exact(entry["case_count"], case_count // 5):
            fail("invalid condition summary keys/count")
        baseline, output = entry["mean_input_normalized_mse"], entry["mean_output_normalized_mse"]
        if any(type(v) is not float or not math.isfinite(v) or v < 0 for v in (baseline, output)):
            fail("invalid summary normalized MSE")
        undefined = entry["undefined_phase_case_count"]
        if type(undefined) is not int or not 0 <= undefined <= entry["case_count"]:
            fail("invalid undefined phase case count")
        if condition == "identity":
            if baseline != 0.0 or entry["ratio_of_mean_reduction"] is not None or not _exact(entry["mean_identity_normalized_change"], output):
                fail("invalid identity summary")
            for field in ("mean_protected_complex_energy_ratio", "mean_protected_weighted_abs_phase_radians"):
                value = entry[field]
                if value is not None and (type(value) is not float or not math.isfinite(value) or value < 0):
                    fail("invalid identity protected metric")
            phase = entry["mean_protected_weighted_abs_phase_radians"]
            if phase is not None and phase > math.pi:
                fail("invalid identity phase")
            if undefined and phase is not None:
                fail("undefined phase cannot be numeric")
        else:
            if baseline <= 0 or undefined != 0 or any(entry[k] is not None for k in (
                    "mean_identity_normalized_change", "mean_protected_complex_energy_ratio",
                    "mean_protected_weighted_abs_phase_radians")):
                fail("invalid corrupted summary")
            expected = (baseline - output) / baseline
            if type(entry["ratio_of_mean_reduction"]) is not float or not math.isclose(
                    entry["ratio_of_mean_reduction"], expected, rel_tol=1e-12, abs_tol=1e-12):
                fail("invalid ratio-of-mean reduction")


def _validate_report(report, cfg, lock_sha256):
    keys = {"schema_version", "status", "model_version", "created_at_utc", "device",
            "config_sha256", "training_lock_sha256", "preregistration_sha256",
            "reference_manifest_sha256", "partition_lock_sha256", "source_binding",
            "domain_reports", "sample_rates_hz", "fit_subjects", "internal_validation_subjects",
            "target_definition", "amp", "outer_validation_waveforms_loaded", *PROVENANCE}
    if type(report) is not dict or set(report) != keys:
        fail("invalid training_report keys")
    _finite_json(report)
    expected = {"schema_version": 1, "status": "formal", "model_version": MODEL_VERSION,
                "device": "cuda:0", "config_sha256": CONFIG_SHA256,
                "training_lock_sha256": lock_sha256, "preregistration_sha256": PREREG_SHA256,
                "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
                "partition_lock_sha256": PARTITION_LOCK_SHA256, "sample_rates_hz": RATES,
                "fit_subjects": [1, 2, 3, 4], "internal_validation_subjects": [5, 6],
                "amp": False, "outer_validation_waveforms_loaded": False,
                "target_definition": cfg["corruption"]["target_definition"], **PROVENANCE}
    for key, value in expected.items():
        if not _exact(report.get(key), value):
            fail("invalid training_report binding: " + key)
    try:
        created = datetime.fromisoformat(report["created_at_utc"])
        if created.tzinfo is None or created.utcoffset() != timezone.utc.utcoffset(created):
            fail("creation timestamp must be UTC")
    except (TypeError, ValueError) as exc:
        fail("invalid creation timestamp: " + str(exc))
    bound = _frozen_prereg()["bound_files"]
    expected_sources = []
    for domain in DOMAINS:
        for sid in range(1, 7):
            path = f"datasets/processed/ssvep_reference_v5_20261007/{domain}/subject_{sid:03d}.npz"
            expected_sources.append({"dataset": domain, "subject_id": sid, "sample_rate_hz": RATES[domain],
                                     "npz_sha256": bound[path], "sidecar_sha256": bound[path + ".manifest.json"]})
    if not _exact(report["source_binding"], expected_sources):
        fail("training source_binding is not exactly the 12 frozen subjects")
    if type(report["domain_reports"]) is not dict or set(report["domain_reports"]) != set(DOMAINS):
        fail("checkpoint requires exactly two domain reports")
    for domain, domain_report in report["domain_reports"].items():
        expected_keys = {"dataset", "sample_rate_hz", "fit_subjects", "internal_validation_subjects",
                         "fit_case_count", "internal_case_count", "best_epoch", "best_selection_key",
                         "internal_gates_passed", "selection_status", "history"}
        if type(domain_report) is not dict or set(domain_report) != expected_keys:
            fail("invalid domain_report keys")
        for key, val in {"dataset": domain, "sample_rate_hz": RATES[domain],
                         "fit_subjects": [1, 2, 3, 4], "internal_validation_subjects": [5, 6],
                         "fit_case_count": 3600 if domain == DOMAINS[0] else 4800,
                         "internal_case_count": 1800 if domain == DOMAINS[0] else 2400}.items():
            if not _exact(domain_report[key], val):
                fail("invalid domain binding: " + key)
        history = domain_report["history"]
        if type(history) is not list or len(history) != 18:
            fail("checkpoint must document exactly 18 epochs")
        candidates = []
        for epoch, row in enumerate(history, 1):
            if type(row) is not dict or set(row) != {"epoch", "train_loss", "internal_loss", "conditions", "gates", "selection_key", "best_epoch"}:
                fail("invalid epoch history")
            if not _exact(row["epoch"], epoch):
                fail("invalid epoch numbering")
            for loss_name, cases in (("train_loss", domain_report["fit_case_count"]),
                                     ("internal_loss", domain_report["internal_case_count"])):
                loss = row[loss_name]
                if type(loss) is not dict or set(loss) != {"total", "corrupted", "identity", "change", "corrupted_count", "identity_count", "case_count"}:
                    fail("invalid aggregated loss schema")
                for k, v in {"case_count": cases, "identity_count": cases // 5,
                             "corrupted_count": cases * 4 // 5}.items():
                    if not _exact(loss[k], v):
                        fail("invalid aggregated case counts")
                if any(type(loss[k]) is not float or loss[k] < 0 for k in ("total", "corrupted", "identity", "change")):
                    fail("invalid aggregated losses")
                recomputed = loss["corrupted"] + 20 * loss["identity"] + 0.001 * loss["change"]
                if not math.isclose(loss["total"], recomputed, rel_tol=1e-12, abs_tol=1e-12):
                    fail("invalid loss aggregation")
            _validate_summary(row["conditions"], domain_report["internal_case_count"])
            gates = contract().gate_summary(row["conditions"], cfg)
            if not _exact(row["gates"], gates):
                fail("invalid recorded gates")
            key = selection_key(row["conditions"], gates, row["internal_loss"]["total"], epoch)
            if not _exact(row["selection_key"], list(key)):
                fail("invalid selection key")
            candidates.append(key)
            if not _exact(row["best_epoch"], min(candidates)[-1]):
                fail("invalid best-so-far epoch")
        best = min(candidates)
        eligible = best[0] == 0
        if (not _exact(domain_report["best_epoch"], best[-1])
                or not _exact(domain_report["best_selection_key"], list(best))
                or not _exact(domain_report["internal_gates_passed"], eligible)
                or not _exact(domain_report["selection_status"], "internal_gates_passed" if eligible else "gates_failed")):
            fail("invalid selected epoch or gate status")


def checkpoint_metadata(cfg, training_report, training_lock_sha256):
    """Metadata only; caller subsequently adds exactly two model state_dicts."""
    contract().validate_config(cfg)
    if not _sha(training_lock_sha256):
        fail("invalid training lock SHA256")
    _validate_report(training_report, cfg, training_lock_sha256)
    return {"schema_version": 1, "status": "formal", "model_version": MODEL_VERSION,
            "model_type": MODEL_TYPE, "config": copy.deepcopy(cfg),
            "config_sha256": CONFIG_SHA256, "training_lock_sha256": training_lock_sha256,
            "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
            "partition_lock_sha256": PARTITION_LOCK_SHA256,
            "training_report": copy.deepcopy(training_report), **PROVENANCE}


def validate_checkpoint(ck, cfg, training_lock_sha256):
    """Validate already weights_only-loaded metadata/state; return CPU eval models.

    No torch.load/path arguments, no CUDA, and no loading of old v5 weights.
    """
    contract().validate_config(cfg)
    if type(ck) is not dict or set(ck) != CHECKPOINT_KEYS:
        fail("invalid v6 checkpoint keys")
    expected = checkpoint_metadata(cfg, ck["training_report"], training_lock_sha256)
    for key, value in expected.items():
        if not _exact(ck[key], value):
            fail("invalid checkpoint metadata: " + key)
    if type(ck["models"]) is not dict or set(ck["models"]) != set(DOMAINS):
        fail("checkpoint requires exactly two independent states")
    models = {}
    # Model construction consumes RNG; validation must not alter the training RNG.
    with torch.random.fork_rng(devices=[]):
        for domain in DOMAINS:
            model = build_model(cfg, RATES[domain])
            state = ck["models"][domain]
            expected_state = model.state_dict()
            if not isinstance(state, dict) or set(state) != set(expected_state):
                fail("invalid state keys for " + domain)
            for name, tensor in state.items():
                if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                        or tensor.device.type != "cpu" or tensor.dtype != torch.float32
                        or tensor.shape != expected_state[name].shape
                        or tensor.requires_grad or not bool(torch.isfinite(tensor).all())):
                    fail("invalid state tensor: " + domain + "/" + name)
            model.load_state_dict(state, strict=True)
            model.eval()
            models[domain] = model
    return models


