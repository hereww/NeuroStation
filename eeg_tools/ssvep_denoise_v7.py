"""Pure v7 model/loss APIs: no dataset, weight, lock, or publishing I/O.

Inputs are already noisy-window-normalized float32 [B,8,256]. A shared
NONCAUSAL residual TCN feeds two zero-initialized heads. The low branch is
0.5*tanh(low_head(features)), projected to rFFT bins strictly below 8 Hz.
The high branch implements the frozen config EXACTLY:
clamp(tanh(linear(mean_time(GELU(TCN_features)))), 0, 0.5).
It predicts one attenuation per channel/window, shared across bins >40 Hz.
High coefficients are only attenuated, never amplified; 8..40 Hz is protected.

Zero-initialized high attenuation and low output give bitwise initial identity.
Bounds apply BEFORE low projection; there is NO post-projection pointwise clip.
Conditions/identity/targets never enter forward. Window FFT projection and TCN
are not causal streaming or clinical/task-preservation acceptance claims.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

MODEL_VERSION = "learned_denoise_v7_ssvep"
MODEL_TYPE = "zero_initialized_low_residual_nonamplifying_high_attenuation_tcn"
DOMAINS = ("Nakanishi2015", "Wearable")
RATES = {"Nakanishi2015": 256, "Wearable": 250}
CONDITIONS = ("identity", "transient", "low_frequency_drift", "high_frequency_burst", "common_mode")
INPUT_CHANNELS = 8
WINDOW_SAMPLES = 256
PARAMETER_COUNT = 89520

_MODEL = {
    "hidden_channels": 32, "kernel_size": 7, "dilations": [1, 2, 4, 8, 16, 32],
    "group_norm_groups": 8, "dropout": 0.0, "protected_band_hz": [8.0, 40.0],
    "low_residual_limit_normalized": 0.5,
    "low_band_projection": "rfft_keep_frequency_lt8_then_irfft_n256",
    "high_band_projection": "rfft_keep_frequency_gt40_then_irfft_n256",
    "high_attenuation_max": 0.5,
    "high_attenuation_mapping": "clamp(tanh(linear(mean_time(GELU(TCN_features)))),0,0.5)",
    "high_attenuation_scope": "one_real_scalar_per_channel_per_window_shared_across_high_bins",
    "zero_initialize_low_output": True, "zero_initialize_high_gate": True,
    "post_projection_clip": False,
}
_LOSS = {
    "relative_mse_input_floor": 0.01, "high_frequency_case_weight": 4.0,
    "other_corrupted_case_weight": 1.0, "identity_normalized_mse_weight": 60.0,
    "high_frequency_low_branch_change_weight": 20.0,
    "normalized_change_penalty_weight": 0.001, "target_channel_variance_epsilon": 1e-8,
}


def _exact(left: Any, right: Any) -> bool:
    """Deep equality rejecting bool/int and list/tuple substitutions."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(_exact(left[k], right[k]) for k in left)
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(_exact(a, b) for a, b in zip(left, right))
    return left == right


def validate_config(cfg: Mapping[str, Any], sample_rate_hz: int | None = None) -> None:
    """Check the in-memory frozen architecture/loss subset, not data provenance.

    No file is read here and no report/checkpoint validity is implied. The runner
    remains responsible for checking the frozen config bytes and training locks.
    """
    if not isinstance(cfg, Mapping):
        raise ValueError("v7 cfg must be a mapping")
    expected = {
        "schema_version": 1, "status": "formal", "model_version": MODEL_VERSION,
        "model_type": MODEL_TYPE, "input_channels": INPUT_CHANNELS,
        "window_samples": WINDOW_SAMPLES, "sample_rates_hz": RATES,
        "conditions": list(CONDITIONS), "model": _MODEL,
    }
    for name, value in expected.items():
        if not _exact(cfg.get(name), value):
            raise ValueError("v7 frozen config mismatch: " + name)
    if not isinstance(cfg.get("loss"), Mapping):
        raise ValueError("v7 config requires loss settings")
    for name, value in _LOSS.items():
        if not _exact(cfg["loss"].get(name), value):
            raise ValueError("v7 frozen loss mismatch: " + name)
    if sample_rate_hz is not None and (
        type(sample_rate_hz) is not int or sample_rate_hz not in RATES.values()
    ):
        raise ValueError("v7 requires native sample_rate_hz 256 or 250")


def _inclusive_attenuation_clamp(raw: torch.Tensor) -> torch.Tensor:
    """Exact clamp(raw,0,.5) forward, with the INSIDE subgradient at boundaries.

    Strict comparisons make gradients one at raw=0 and raw=.5, zero strictly
    outside. This explicitly avoids version-dependent stock clamp boundary
    gradients that can strand the zero-initialized high head. It changes only
    the boundary subgradient, NOT the frozen tanh/clamp forward mapping.
    """
    return torch.where(raw < 0.0, torch.zeros_like(raw),
                       torch.where(raw > 0.5, torch.full_like(raw, 0.5), raw))


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all())


def _window(value: Any, *, shape=None, device=None, name="input") -> None:
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.ndim != 3 or value.shape[0] == 0
            or tuple(value.shape[1:]) != (INPUT_CHANNELS, WINDOW_SAMPLES)
            or value.dtype != torch.float32
            or (shape is not None and value.shape != shape)
            or (device is not None and value.device != device) or not _finite(value)):
        raise ValueError(name + " must be finite float32 [B,8,256] on the model/loss device")


class ResidualBlock(nn.Module):
    """Same-length two-convolution noncausal residual block, independent v7 state."""
    def __init__(self, dilation: int):
        super().__init__()
        kwargs = {"device": "cpu", "dtype": torch.float32}
        self.net = nn.Sequential(
            nn.Conv1d(32, 32, 7, padding=3*dilation, dilation=dilation, **kwargs),
            nn.GroupNorm(8, 32, **kwargs), nn.GELU(), nn.Dropout(0.0),
            nn.Conv1d(32, 32, 7, padding=3*dilation, dilation=dilation, **kwargs),
            nn.GroupNorm(8, 32, **kwargs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.5 * self.net(x)


class SSVEPDenoiserV7(nn.Module):
    def __init__(self, cfg: Mapping[str, Any], sample_rate_hz: int):
        super().__init__()
        validate_config(cfg, sample_rate_hz)
        if type(sample_rate_hz) is not int or sample_rate_hz not in RATES.values():
            raise ValueError("v7 requires native sample_rate_hz 256 or 250")
        self.sample_rate_hz = sample_rate_hz
        kwargs = {"device": "cpu", "dtype": torch.float32}
        self.input = nn.Conv1d(8, 32, 7, padding=3, **kwargs)
        self.blocks = nn.Sequential(*(ResidualBlock(d) for d in cfg["model"]["dilations"]))
        self.activation = nn.GELU()
        self.low_head = nn.Conv1d(32, 8, 1, **kwargs)
        self.high_head = nn.Linear(32, 8, **kwargs)
        for head in (self.low_head, self.high_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        frequencies = torch.fft.rfftfreq(256, d=1.0/sample_rate_hz, device="cpu", dtype=torch.float32)
        self.register_buffer("low_mask", frequencies < 8.0, persistent=False)
        self.register_buffer("high_mask", frequencies > 40.0, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return prediction only; no condition/identity/target/subject arguments."""
        return self.forward_with_aux(x)[0]

    def forward_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return pred and {'low_delta': [B,8,256], 'high_attenuation': [B,8,1]}.

        Using identical high-spectrum inverse transforms at zero attenuation avoids
        round-trip FFT error in the initial identity. After training, protected bins
        remain zero in the residual spectrum, up to float32 FFT roundoff.
        """
        _window(x, device=self.input.weight.device)
        if self.input.weight.dtype != torch.float32:
            raise ValueError("v7 model parameters must remain float32")
        features = self.activation(self.blocks(self.input(x)))
        low_delta_pre = 0.5 * torch.tanh(self.low_head(features))
        high_logits = self.high_head(features.mean(dim=-1))
        if not _finite(low_delta_pre) or not _finite(high_logits):
            raise ValueError("nonfinite v7 branch output")
        low_delta = torch.fft.irfft(
            torch.fft.rfft(low_delta_pre, dim=-1) * self.low_mask, n=256, dim=-1)
        high_attenuation = _inclusive_attenuation_clamp(torch.tanh(high_logits)).unsqueeze(-1)
        high_spectrum = torch.fft.rfft(x, dim=-1) * self.high_mask
        high_input = torch.fft.irfft(high_spectrum, n=256, dim=-1)
        attenuated_high = torch.fft.irfft(
            high_spectrum * (1.0-high_attenuation), n=256, dim=-1)
        pred = x + low_delta + (attenuated_high-high_input)
        if not _finite(pred) or not _finite(low_delta):
            raise ValueError("nonfinite v7 prediction")
        return pred, {"low_delta": low_delta, "high_attenuation": high_attenuation}


def build_model(cfg: Mapping[str, Any], sample_rate_hz: int) -> SSVEPDenoiserV7:
    """Return a fresh CPU float32 model; runner explicitly moves it if necessary."""
    return SSVEPDenoiserV7(cfg, sample_rate_hz)


def model_parameter_count(model: nn.Module) -> int:
    """The frozen eight-channel architecture has PARAMETER_COUNT=89,520."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _condition_codes(conditions: Any, batch: int, device: torch.device) -> torch.Tensor:
    """Loss-only labels: ordered strings, integer codes, or same-device code tensor."""
    if isinstance(conditions, torch.Tensor):
        if (conditions.layout != torch.strided or conditions.shape != (batch,)
                or conditions.device != device or conditions.dtype not in
                (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
            raise ValueError("conditions must be integer codes [B] on the loss device")
        codes = conditions.to(dtype=torch.long)
    elif isinstance(conditions, Sequence) and not isinstance(conditions, (str, bytes)):
        if len(conditions) != batch:
            raise ValueError("conditions length must match batch size")
        values = []
        for value in conditions:
            if type(value) is str and value in CONDITIONS:
                values.append(CONDITIONS.index(value))
            elif type(value) is int and 0 <= value < len(CONDITIONS):
                values.append(value)
            else:
                raise ValueError("unknown v7 condition string/code")
        codes = torch.tensor(values, dtype=torch.long, device=device)
    else:
        raise ValueError("conditions must be a string/code sequence or integer tensor")
    if bool(((codes < 0) | (codes >= len(CONDITIONS))).any()):
        raise ValueError("condition code out of range")
    return codes


def compute_loss(pred: torch.Tensor, target: torch.Tensor, noisy: torch.Tensor,
                 identity: torch.Tensor, aux: Mapping[str, Any], cfg: Mapping[str, Any],
                 *, conditions: Sequence[str] | Sequence[int] | torch.Tensor) -> dict[str, Any]:
    """Weighted group means and sufficient statistics, all in normalized coordinates.

    Error cases use demeaned target-channel variance +1e-8; corrupted relative MSE
    divides by input error floored at .01. HF cases have weight 4, other corrupted
    cases 1. Identity error is NOT divided by the input baseline. Low-branch HF
    change and all-case change are plain mean squares in ALREADY noisy-normalized
    coordinates, not re-divided by target variance. Missing groups contribute zero.
    """
    validate_config(cfg)
    _window(pred, name="pred")
    for name, value in (("target", target), ("noisy", noisy)):
        _window(value, shape=pred.shape, device=pred.device, name=name)
    batch = pred.shape[0]
    if (not isinstance(identity, torch.Tensor) or identity.layout != torch.strided
            or identity.dtype != torch.bool or identity.shape != (batch,)
            or identity.device != pred.device):
        raise ValueError("identity must be bool [B] on the loss device")
    if not isinstance(aux, Mapping) or not {"low_delta", "high_attenuation"} <= set(aux):
        raise ValueError("aux requires low_delta and high_attenuation")
    _window(aux["low_delta"], shape=pred.shape, device=pred.device, name="low_delta")
    gate = aux["high_attenuation"]
    if (not isinstance(gate, torch.Tensor) or gate.layout != torch.strided
            or gate.shape != (batch, 8, 1) or gate.dtype != torch.float32
            or gate.device != pred.device or not _finite(gate)
            or bool(((gate < 0) | (gate > 0.5)).any())):
        raise ValueError("high_attenuation must be finite float32 [B,8,1] in [0,0.5]")
    codes = _condition_codes(conditions, batch, pred.device)
    if not torch.equal(identity, codes == 0):
        raise ValueError("identity mask and condition labels disagree")
    loss_cfg = cfg["loss"]
    variance = (target-target.mean(-1, keepdim=True)).square().mean(-1, keepdim=True)
    variance = variance + loss_cfg["target_channel_variance_epsilon"]
    out_error = ((pred-target).square()/variance).mean((1, 2))
    in_error = ((noisy-target).square()/variance).mean((1, 2))
    ratio = out_error/in_error.clamp(min=loss_cfg["relative_mse_input_floor"])
    change_case = (pred-noisy).square().mean((1, 2))
    low_case = aux["low_delta"].square().mean((1, 2))
    if any(not _finite(v) for v in (variance, out_error, in_error, ratio, change_case, low_case)):
        raise ValueError("nonfinite v7 loss intermediates")
    corrupted_mask, hf_mask = ~identity, codes == 3
    weights = torch.where(hf_mask, loss_cfg["high_frequency_case_weight"],
                          loss_cfg["other_corrupted_case_weight"])
    corrupted_sum = (weights[corrupted_mask]*ratio[corrupted_mask]).sum()
    corrupted_weight_sum = weights[corrupted_mask].sum()
    identity_sum = out_error[identity].sum()
    hf_low_sum = low_case[hf_mask].sum()
    change_sum = change_case.sum()
    n_identity, n_hf = int(identity.sum().item()), int(hf_mask.sum().item())
    corrupted = corrupted_sum/corrupted_weight_sum.clamp(min=1.0)
    identity_loss = identity_sum/max(n_identity, 1)
    hf_low_change = hf_low_sum/max(n_hf, 1)
    change = change_sum/batch
    total = (corrupted + loss_cfg["identity_normalized_mse_weight"]*identity_loss
             + loss_cfg["high_frequency_low_branch_change_weight"]*hf_low_change
             + loss_cfg["normalized_change_penalty_weight"]*change)
    values = {
        "total": total, "corrupted": corrupted, "identity": identity_loss,
        "hf_low_branch_change": hf_low_change, "change": change,
        "corrupted_sum": corrupted_sum, "corrupted_weight_sum": corrupted_weight_sum,
        "identity_sum": identity_sum, "hf_low_branch_change_sum": hf_low_sum,
        "change_sum": change_sum, "corrupted_count": batch-n_identity,
        "identity_count": n_identity, "hf_count": n_hf, "case_count": batch,
    }
    if any(not _finite(v) for v in values.values() if isinstance(v, torch.Tensor)):
        raise ValueError("nonfinite v7 loss")
    return values


class LossAccumulator:
    """Case/weight-correct epoch aggregation; never average unequal batch means."""
    def __init__(self):
        self.sums = {name: 0.0 for name in (
            "corrupted", "corrupted_weight", "identity", "hf_low_branch_change", "change")}
        self.counts = {name: 0 for name in ("corrupted", "identity", "hf", "case")}

    def update(self, losses: Mapping[str, Any]) -> None:
        # Validate before mutating: an invalid batch cannot poison partial sums.
        sums, counts = {}, {}
        for name in self.sums:
            value = losses[name+"_sum"]
            if not isinstance(value, torch.Tensor) or value.ndim != 0 or not _finite(value):
                raise ValueError("invalid accumulated loss sum")
            sums[name] = float(value.detach().cpu())
            if sums[name] < 0:
                raise ValueError("negative accumulated loss sum")
        for name in self.counts:
            value = losses[name+"_count"]
            if type(value) is not int or value < 0:
                raise ValueError("invalid accumulated loss count")
            counts[name] = value
        if (counts["case"] <= 0 or counts["corrupted"]+counts["identity"] != counts["case"]
                or counts["hf"] > counts["corrupted"]):
            raise ValueError("inconsistent accumulated loss counts")
        if not math.isclose(sums["corrupted_weight"],
                            counts["corrupted"]+3*counts["hf"], rel_tol=0, abs_tol=1e-6):
            raise ValueError("inconsistent accumulated corrupted weights")
        for name, value in sums.items():
            self.sums[name] += value
        for name, value in counts.items():
            self.counts[name] += value

    def result(self) -> dict[str, Any]:
        if self.counts["case"] <= 0:
            raise ValueError("empty v7 epoch")
        values = {
            "corrupted": self.sums["corrupted"]/max(self.sums["corrupted_weight"], 1.0),
            "identity": self.sums["identity"]/max(self.counts["identity"], 1),
            "hf_low_branch_change": self.sums["hf_low_branch_change"]/max(self.counts["hf"], 1),
            "change": self.sums["change"]/self.counts["case"],
        }
        values["total"] = (values["corrupted"]+60.0*values["identity"]
                           +20.0*values["hf_low_branch_change"]+0.001*values["change"])
        values.update({name+"_count": count for name, count in self.counts.items()})
        values["corrupted_weight_sum"] = self.sums["corrupted_weight"]
        if any(type(value) is float and not math.isfinite(value) for value in values.values()):
            raise ValueError("nonfinite accumulated v7 loss")
        return values


def _number(value: Any, name: str, *, positive=False) -> float:
    if type(value) not in (float, int):
        raise ValueError(name+" must be a finite number")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(name+" must be a finite number") from exc
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("invalid "+name)
    return value


def selection_key(summary: Mapping[str, Any], gates: Mapping[str, Any], total_loss: float,
                  epoch: int) -> tuple[int, float, float, float, int]:
    """v6 ordering: eligible, worst ratio of means, identity MSE, loss, earliest epoch."""
    if type(epoch) is not int or not 1 <= epoch <= 18:
        raise ValueError("v7 selection candidates are epochs 1..18")
    if not isinstance(summary, Mapping) or set(summary) != set(CONDITIONS):
        raise ValueError("selection summary requires all five conditions")
    ratios = []
    for condition in CONDITIONS[1:]:
        entry = summary[condition]
        if not isinstance(entry, Mapping):
            raise ValueError("invalid selection entry")
        baseline = _number(entry.get("mean_input_normalized_mse"), "baseline", positive=True)
        output = _number(entry.get("mean_output_normalized_mse"), "output")
        ratios.append(_number(output/baseline, "ratio"))
    identity_entry = summary["identity"]
    if not isinstance(identity_entry, Mapping):
        raise ValueError("invalid identity selection entry")
    identity_error = _number(identity_entry.get("mean_output_normalized_mse"), "identity error")
    if not isinstance(gates, Mapping) or type(gates.get("all_passed")) is not bool:
        raise ValueError("gates require boolean all_passed")
    return (0 if gates["all_passed"] else 1, max(ratios), identity_error,
            _number(total_loss, "total loss"), epoch)


