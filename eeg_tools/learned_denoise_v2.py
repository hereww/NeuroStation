"""Independent single-channel v2 denoiser; derived offline diagnostics only.

Channels are flattened into a batch, never mixed. Normalization/restoration uses
float64 noisy-window mean and AC RMS. Confidence is NOT calibrated efficacy.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np

MODEL_VERSION = "learned_denoise_v2"
DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1, "status": "formal", "model_version": MODEL_VERSION,
    "model_type": "single_channel_shared_residual_tcn", "sample_rate_hz": 250,
    "input_channels": 1, "window_samples": 500, "hop_samples": 500,
    "normalization": "per_channel_window_noisy_mean_ac_rms",
    "model": {"hidden_channels": 24, "dilations": [1, 2, 4, 8, 16],
              "kernel_size": 5, "residual_bound": 2.0, "dropout": 0.0},
    "inference": {"mode": "shadow", "max_invalid_fraction": 0.0, "min_ac_rms": 1e-8,
                  "artifact_score_reduction": "per_channel_window_mean",
                  "threshold_comparison": ">",
                  "confidence_interpretation": "one_minus_probability_not_calibrated"},
    "loss": {"time_weight": 1.0, "spectral_weight": 0.25, "identity_weight": 5.0,
             "clean_window_identity_multiplier": 4.0, "artifact_weight": 0.5},
    "runtime": {"enabled": False, "validation_status": "not_assessed",
                "deployment_scope": "explicit_offline_diagnostics_only", "production_validated": False},
}


class LearnedDenoiseV2Error(ValueError):
    """Invalid v2 configuration, checkpoint, input or unsafe prediction."""


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise LearnedDenoiseV2Error(f"{name} must be a finite number")
    if positive and value <= 0:
        raise LearnedDenoiseV2Error(f"{name} must be positive")
    return float(value)


def validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict) or set(config) != set(DEFAULT_CONFIG):
        raise LearnedDenoiseV2Error("v2 config has missing/unknown top-level keys")
    for key in ("schema_version", "status", "model_version", "model_type", "input_channels", "window_samples", "hop_samples", "normalization"):
        if type(config[key]) is not type(DEFAULT_CONFIG[key]) or config[key] != DEFAULT_CONFIG[key]:
            raise LearnedDenoiseV2Error(f"unsupported v2 {key}; expected {DEFAULT_CONFIG[key]!r}")
    _number(config["sample_rate_hz"], "sample_rate_hz", positive=True)
    for section in ("model", "inference", "loss"):
        if not isinstance(config[section], dict) or set(config[section]) != set(DEFAULT_CONFIG[section]):
            raise LearnedDenoiseV2Error(f"invalid v2 {section} keys")
    # Audit fields are optional so frozen checkpoints with the original runtime
    # schema remain readable. No audit result can enable automatic processing.
    runtime = config["runtime"]
    audit_keys = {"validation_date", "validation_reason", "research_validation_status",
                  "transfer_validation_status", "validation_reports"}
    required_keys = set(DEFAULT_CONFIG["runtime"])
    if (not isinstance(runtime, dict) or not required_keys <= set(runtime)
            or set(runtime) - required_keys - audit_keys):
        raise LearnedDenoiseV2Error("invalid v2 runtime keys")
    model = config["model"]
    if type(model["hidden_channels"]) is not int or model["hidden_channels"] != 24:
        raise LearnedDenoiseV2Error("v2 hidden_channels must be 24")
    if model["dilations"] != [1, 2, 4, 8, 16] or any(type(x) is not int for x in model["dilations"]):
        raise LearnedDenoiseV2Error("v2 dilations must be [1, 2, 4, 8, 16]")
    if type(model["kernel_size"]) is not int or model["kernel_size"] not in (5, 7):
        raise LearnedDenoiseV2Error("kernel_size must be 5 or 7")
    _number(model["residual_bound"], "residual_bound", positive=True)
    if not 0 <= _number(model["dropout"], "dropout") < 1:
        raise LearnedDenoiseV2Error("dropout must be in [0, 1)")
    inference = config["inference"]
    for key in ("mode", "artifact_score_reduction", "threshold_comparison", "confidence_interpretation"):
        if inference[key] != DEFAULT_CONFIG["inference"][key]:
            raise LearnedDenoiseV2Error(f"unsupported inference {key}")
    if _number(inference["max_invalid_fraction"], "max_invalid_fraction") != 0:
        raise LearnedDenoiseV2Error("v2 rejects any invalid full window; max_invalid_fraction must be 0")
    _number(inference["min_ac_rms"], "min_ac_rms", positive=True)
    for key, value in config["loss"].items():
        _number(value, key, positive=True)
    runtime = config["runtime"]
    if runtime["enabled"] is not False or runtime["production_validated"] is not False:
        raise LearnedDenoiseV2Error("v2 is offline-only; automatic/production enablement is forbidden")
    if runtime["validation_status"] not in ("not_assessed", "failed_holdout", "failed_known_session_regression"):
        raise LearnedDenoiseV2Error("v2 validation_status must not claim production acceptance")
    if runtime["deployment_scope"] != "explicit_offline_diagnostics_only":
        raise LearnedDenoiseV2Error("v2 deployment_scope must remain explicit_offline_diagnostics_only")
    for key in audit_keys - {"validation_reports"}:
        if key in runtime and (type(runtime[key]) is not str or not runtime[key].strip()):
            raise LearnedDenoiseV2Error(f"v2 runtime {key} must be a nonempty string")
    if "validation_date" in runtime:
        try:
            parsed_date = date.fromisoformat(runtime["validation_date"])
        except ValueError as error:
            raise LearnedDenoiseV2Error("v2 validation_date must be YYYY-MM-DD") from error
        if parsed_date.isoformat() != runtime["validation_date"]:
            raise LearnedDenoiseV2Error("v2 validation_date must be YYYY-MM-DD")
    if ("research_validation_status" in runtime
            and runtime["research_validation_status"] != "passed_epoch_holdout"):
        raise LearnedDenoiseV2Error("unsupported v2 research_validation_status")
    if ("transfer_validation_status" in runtime
            and runtime["transfer_validation_status"] != "not_assessed_target_mapping"):
        raise LearnedDenoiseV2Error("unsupported v2 transfer_validation_status")
    if "validation_reports" in runtime:
        reports = runtime["validation_reports"]
        if (type(reports) is not dict
                or set(reports) != {"public_epoch_holdout", "transfer_and_known_v1_regression"}
                or any(type(path) is not str or not path.strip() for path in reports.values())):
            raise LearnedDenoiseV2Error("invalid v2 validation_reports")


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        try:
            supplied = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise LearnedDenoiseV2Error(f"cannot read v2 config: {error}") from error
        if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_CONFIG):
            raise LearnedDenoiseV2Error("v2 config must be an object with known keys")
        for key, value in supplied.items():
            if isinstance(config[key], dict):
                if not isinstance(value, dict):
                    raise LearnedDenoiseV2Error(f"{key} must be an object")
                config[key].update(value)
            else:
                config[key] = value
    validate_config(config)
    return config


def validate_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(mask)
    if value.shape != shape or value.dtype.kind not in "biuf" or not np.all((value == 0) | (value == 1)):
        raise LearnedDenoiseV2Error("mask must be finite binary values with exactly the signal shape; no broadcasting")
    return value.astype(bool)


def normalize_window(values: np.ndarray, valid_mask: np.ndarray | None = None,
                     min_ac_rms: float = 1e-8) -> tuple[np.ndarray, float, float]:
    """Normalize ONE channel window, excluding invalid values; reject zero AC RMS.

    Returns float32 network input, and float64/Python-double original mean/RMS.
    Clean targets MUST use this noisy mean/RMS, not their own normalization.
    """
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf" or not len(raw):
        raise LearnedDenoiseV2Error("normalize_window expects one nonempty real channel")
    x = raw.astype(np.float64)
    valid = np.isfinite(x)
    if valid_mask is not None:
        valid &= validate_mask(valid_mask, x.shape)
    if not np.any(valid):
        raise LearnedDenoiseV2Error("window has no valid samples")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        center = float(np.mean(x[valid], dtype=np.float64))
        ac = x[valid] - center
        peak = float(np.max(np.abs(ac)))
        scale = peak * float(np.sqrt(np.mean((ac / peak) ** 2))) if peak > 0 else 0.0
        normalized = np.zeros(x.shape, dtype=np.float32)
        normalized[valid] = ((x[valid] - center) / scale).astype(np.float32)
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= min_ac_rms or not np.all(np.isfinite(normalized)):
        raise LearnedDenoiseV2Error("constant or unsafe mean/AC RMS window")
    return normalized, center, scale


try:
    import torch
    from torch import nn
except ImportError:  # optional ML dependency
    torch = None
    nn = None


if nn is not None:
    class _ResidualBlock(nn.Module):
        def __init__(self, hidden: int, dilation: int, kernel: int, dropout: float):
            super().__init__()
            padding = dilation * (kernel - 1) // 2
            self.net = nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel, padding=padding, dilation=dilation),
                nn.GroupNorm(1, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(hidden, hidden, kernel, padding=padding, dilation=dilation),
                nn.GroupNorm(1, hidden),
            )
            self.activation = nn.GELU()

        def forward(self, value):
            return self.activation(value + self.net(value))

    class ResidualTCNDenoiserV2(nn.Module):
        def __init__(self, config: dict[str, Any]):
            super().__init__()
            architecture = config["model"]
            hidden, kernel = architecture["hidden_channels"], architecture["kernel_size"]
            self.window_samples = config["window_samples"]
            self.residual_bound = float(architecture["residual_bound"])
            self.input = nn.Sequential(nn.Conv1d(2, hidden, kernel, padding=kernel // 2), nn.GroupNorm(1, hidden), nn.GELU())
            self.blocks = nn.Sequential(*[_ResidualBlock(hidden, d, kernel, architecture["dropout"]) for d in architecture["dilations"]])
            self.noise_head = nn.Conv1d(hidden, 1, 1)
            self.probability_head = nn.Conv1d(hidden, 1, 1)
            nn.init.zeros_(self.noise_head.weight)
            nn.init.zeros_(self.noise_head.bias)

        def forward(self, signal, valid_mask=None):
            if signal.ndim != 3 or signal.shape[1:] != (1, self.window_samples):
                raise LearnedDenoiseV2Error("model input must have shape [batch, 1, 500]")
            if valid_mask is None:
                valid_mask = torch.ones_like(signal)
            if valid_mask.shape != signal.shape:
                raise LearnedDenoiseV2Error("model valid_mask shape mismatch")
            features = self.blocks(self.input(torch.cat((signal, valid_mask), dim=1)))
            noise = self.residual_bound * torch.tanh(self.noise_head(features))
            # Zero-mean AC residual preserves DC; rescale after centering to keep
            # the configured bound rather than allowing centering to double it.
            noise = noise - noise.mean(dim=-1, keepdim=True)
            divisor = torch.clamp(noise.abs().amax(dim=-1, keepdim=True) / self.residual_bound, min=1.0)
            noise = noise / divisor
            logits = self.probability_head(features)
            return {"denoised": signal - noise, "noise": noise,
                    "artifact_logits": logits, "artifact_probability": torch.sigmoid(logits)}
else:
    ResidualTCNDenoiserV2 = None


def build_model(config: dict[str, Any]):
    validate_config(config)
    if torch is None or ResidualTCNDenoiserV2 is None:
        raise LearnedDenoiseV2Error("optional PyTorch is required for v2 model construction")
    return ResidualTCNDenoiserV2(config)


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    config: dict[str, Any]
    checkpoint_path: Path
    calibrated_threshold: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def validation_threshold(self) -> float | None:
        """Compatibility alias: calibration is validation-only, not efficacy."""
        return self.calibrated_threshold

    @property
    def checkpoint_metadata(self) -> dict[str, Any]:
        return self.metadata


@dataclass(frozen=True)
class DenoiseResult:
    denoised: np.ndarray
    model_denoised: np.ndarray
    artifact_probability: np.ndarray
    confidence: np.ndarray
    replacement_mask: np.ndarray
    valid_mask: np.ndarray
    rejected_mask: np.ndarray
    diagnostics: dict[str, Any]


def primitive_metadata(value: Any) -> bool:
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is float:
        return bool(np.isfinite(value))
    if type(value) is list:
        return all(primitive_metadata(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and primitive_metadata(item) for key, item in value.items())
    return False


def validate_checkpoint_metadata(checkpoint: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Shared fail-closed schema/config/calibration guard; no inference or metrics.

    Return plain metadata excluding weights. Dataset identity/split alignment is
    additionally checked against the frozen NPZ/manifest by the evaluator.
    """
    validate_config(config)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_version") != MODEL_VERSION or (type(checkpoint.get("checkpoint_schema_version")) is not int or checkpoint["checkpoint_schema_version"] != 1):
        raise LearnedDenoiseV2Error("unsupported checkpoint model_version/schema")
    metadata = {key: value for key, value in checkpoint.items() if key != "model_state"}
    if not primitive_metadata(metadata):
        raise LearnedDenoiseV2Error("checkpoint metadata must be finite plain scalars/lists/dicts")
    saved = checkpoint.get("config")
    validate_config(saved)
    for key in ("sample_rate_hz", "input_channels", "window_samples", "hop_samples", "normalization", "model_type", "model"):
        if saved[key] != config[key]:
            raise LearnedDenoiseV2Error(f"checkpoint {key} does not match config")
    if checkpoint.get("architecture") != config["model"]:
        raise LearnedDenoiseV2Error("checkpoint architecture does not match config")
    threshold = checkpoint.get("calibrated_threshold")
    if "validation_threshold" in checkpoint and checkpoint["validation_threshold"] != threshold:
        raise LearnedDenoiseV2Error("inconsistent validation/calibrated threshold metadata")
    if not 0 <= _number(threshold, "validation_threshold") <= 1:
        raise LearnedDenoiseV2Error("validation_threshold must be in [0, 1]")
    calibration = checkpoint.get("threshold_calibration")
    if (not isinstance(calibration, dict) or calibration.get("scope") != "validation_only"
            or calibration.get("comparison") != ">" or calibration.get("production_calibrated") is not False
            or calibration.get("method") != "clean_score_order_statistic"
            or calibration.get("score_reduction") != "per_channel_window_mean"
            or calibration.get("empirical_only") is not True):
        raise LearnedDenoiseV2Error("missing validation-only empirical threshold provenance")
    clean_count = calibration.get("clean_window_count")
    fraction = calibration.get("observed_clean_false_replacement_fraction")
    if type(clean_count) is not int or clean_count < 1 or not 0 <= _number(fraction, "observed clean false replacement fraction") <= 0.01:
        raise LearnedDenoiseV2Error("invalid empirical clean false replacement calibration")
    allowance = calibration.get("max_clean_false_replacement_fraction")
    if not 0 <= _number(allowance, "max_clean_false_replacement_fraction") <= 0.01:
        raise LearnedDenoiseV2Error("empirical false replacement allowance exceeds 1%")
    if fraction > allowance:
        raise LearnedDenoiseV2Error("observed false replacement exceeds its calibration allowance")
    return metadata


def load_model(path: Path | str, config: dict[str, Any], device: str = "cpu") -> ModelBundle:
    validate_config(config)
    if torch is None:
        raise LearnedDenoiseV2Error("optional PyTorch is required for v2 checkpoint loading")
    try:
        checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    except (OSError, pickle.UnpicklingError, TypeError, RuntimeError, EOFError) as error:
        raise LearnedDenoiseV2Error(f"cannot safely load v2 checkpoint: {error}") from error
    metadata = validate_checkpoint_metadata(checkpoint, config)
    threshold = metadata["calibrated_threshold"]
    model = build_model(config)
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except (KeyError, TypeError, RuntimeError) as error:
        raise LearnedDenoiseV2Error(f"invalid v2 model_state: {error}") from error
    if not all(bool(torch.isfinite(tensor).all()) for tensor in model.state_dict().values()):
        raise LearnedDenoiseV2Error("checkpoint contains non-finite weights")
    model.to(device).eval()
    return ModelBundle(model, deepcopy(config), Path(path), float(threshold), metadata)


def apply_model_to_array(bundle: ModelBundle, signal: np.ndarray,
                         valid_mask: np.ndarray | None = None) -> DenoiseResult:
    """Non-overlapping per-channel windows; tail/invalid/unsafe windows keep raw.

    Candidate and gated tracks are distinct. Failed windows have probability NaN,
    confidence zero, rejected_mask true. No raw buffer or file is ever changed.
    """
    validate_config(bundle.config)
    if torch is None:
        raise LearnedDenoiseV2Error("optional PyTorch is required for v2 inference")
    source = np.asarray(signal)
    if source.ndim != 2 or source.dtype.kind not in "iuf" or not all(source.shape):
        raise LearnedDenoiseV2Error("signal must be a nonempty real [channels, samples] array")
    x = source.astype(np.float64, copy=True)
    valid = np.isfinite(x)
    if valid_mask is not None:
        valid &= validate_mask(valid_mask, x.shape)
    deployed, candidate = x.copy(), x.copy()
    probabilities = np.full(x.shape, np.nan, dtype=np.float64)
    confidence = np.zeros(x.shape, dtype=np.float64)
    replacement = np.zeros(x.shape, dtype=bool)
    rejected = np.ones(x.shape, dtype=bool)
    threshold = bundle.calibrated_threshold
    if threshold is not None and not 0 <= _number(threshold, "validation_threshold") <= 1:
        raise LearnedDenoiseV2Error("invalid bundle validation_threshold")
    window = bundle.config["window_samples"]
    device = next(bundle.model.parameters(), torch.empty(0)).device
    counts = {"accepted_windows": 0, "replaced_windows": 0, "invalid_windows": 0,
              "constant_or_normalization_rejected_windows": 0, "bad_prediction_windows": 0}
    bound = float(bundle.config["model"]["residual_bound"])
    for start in range(0, x.shape[1] - window + 1, window):
        end = start + window
        channels, normalized, scales = [], [], []
        for channel in range(x.shape[0]):
            window_valid = valid[channel, start:end]
            if np.mean(~window_valid) > bundle.config["inference"]["max_invalid_fraction"]:
                counts["invalid_windows"] += 1
                continue
            try:
                norm, _, scale = normalize_window(x[channel, start:end], window_valid, bundle.config["inference"]["min_ac_rms"])
            except LearnedDenoiseV2Error:
                counts["constant_or_normalization_rejected_windows"] += 1
                continue
            channels.append(channel)
            normalized.append(norm)
            scales.append(scale)
        if not channels:
            continue
        tensor = torch.from_numpy(np.stack(normalized)[:, None]).to(device)
        tensor_valid = torch.from_numpy(valid[channels, start:end].astype(np.float32)[:, None]).to(device)
        try:
            with torch.no_grad():
                prediction = bundle.model(tensor, tensor_valid)
            shape = (len(channels), 1, window)
            if not isinstance(prediction, dict) or any(not isinstance(prediction.get(key), torch.Tensor) or tuple(prediction[key].shape) != shape for key in ("denoised", "noise", "artifact_probability")):
                raise LearnedDenoiseV2Error("invalid prediction shape")
            noise = prediction["noise"].detach().cpu().numpy().astype(np.float64)[:, 0]
            model_output = prediction["denoised"].detach().cpu().numpy().astype(np.float64)[:, 0]
            probability = prediction["artifact_probability"].detach().cpu().numpy().astype(np.float64)[:, 0]
        except (LearnedDenoiseV2Error, RuntimeError, ValueError, TypeError, KeyError):
            counts["bad_prediction_windows"] += len(channels)
            continue
        for index, channel in enumerate(channels):
            p, residual = probability[index], noise[index]
            with np.errstate(over="ignore", invalid="ignore"):
                restored = x[channel, start:end] - residual * scales[index]
            window_valid = valid[channel, start:end]
            if (not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1))
                    or not np.all(np.isfinite(model_output[index])) or not np.all(np.isfinite(residual))
                    or np.any(np.abs(residual) > bound + 1e-6)
                    or abs(float(np.mean(residual))) > 1e-6 * max(1.0, bound)
                    or not np.all(np.isfinite(restored[window_valid]))
                    or not np.allclose(model_output[index], normalized[index].astype(np.float64) - residual, rtol=1e-5, atol=1e-5)):
                counts["bad_prediction_windows"] += 1
                continue
            # Restore via raw + float64 residual, not re-quantized normalized raw.
            restored[~window_valid] = x[channel, start:end][~window_valid]
            candidate[channel, start:end] = restored
            probabilities[channel, start:end] = p
            confidence[channel, start:end] = 1.0 - p  # explicitly NOT calibrated
            rejected[channel, start:end] = False
            rejected[channel, start:end][~window_valid] = True
            counts["accepted_windows"] += 1
            score = float(np.mean(p))
            if threshold is not None and score > threshold:
                replacement[channel, start:end] = window_valid
                deployed[channel, start:end][window_valid] = restored[window_valid]
                counts["replaced_windows"] += 1
    diagnostics = {
        "model_version": MODEL_VERSION, "window_samples": window,
        "full_windows_per_channel": x.shape[1] // window,
        "tail_samples_per_channel": x.shape[1] % window,
        "calibrated_threshold": threshold, "validation_threshold": threshold, "threshold_comparison": ">",
        "threshold_scope": "validation_only_empirical_not_production_calibrated" if threshold is not None else "missing_threshold_no_replacement",
        "artifact_score_reduction": "per_channel_window_mean",
        "confidence_interpretation": "one_minus_probability_NOT_calibrated_success_probability",
        "normalization": bundle.config["normalization"], "restoration_dtype": "float64",
        "automatic_processing_enabled": False, "diagnostic_only": True, "release_gate_passed": False,
        "candidate_and_gated_reported_separately": True, **counts,
    }
    return DenoiseResult(deployed, candidate, probabilities, confidence, replacement, valid, rejected, diagnostics)


__all__ = ["MODEL_VERSION", "DEFAULT_CONFIG", "LearnedDenoiseV2Error", "load_config", "validate_config",
           "build_model", "ResidualTCNDenoiserV2", "ModelBundle", "DenoiseResult", "load_model",
           "apply_model_to_array", "normalize_window", "validate_mask", "primitive_metadata", "validate_checkpoint_metadata"]
