"""Frozen v2 denoising plus an independent, normalized single-channel domain gate.

Only explicit offline diagnostics are supported. Domain scores are uncalibrated
applicability scores, NOT denoising efficacy or artifact probabilities. This core
never loads datasets, trains models, opens hardware or changes source files.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np

from eeg_tools import learned_denoise_v2 as v2

# Optional dependency, shared read-only with v2. No installation/CPU fallback.
torch = v2.torch
nn = torch.nn if torch is not None else None
MODEL_VERSION = "learned_denoise_v3"
DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1, "status": "formal", "model_version": MODEL_VERSION,
    "model_type": "normalized_single_channel_domain_cnn", "sample_rate_hz": 250,
    "input_channels": 1, "window_samples": 500, "hop_samples": 500,
    "normalization": "per_channel_window_noisy_mean_ac_rms",
    "model": {"channels": [16, 32, 32], "kernel_sizes": [9, 7, 5],
              "strides": [2, 2, 2], "hidden_features": 32, "dropout": 0.0},
    "denoiser": {"model_version": "learned_denoise_v2", "frozen": True, "device": "cpu"},
    "inference": {"mode": "shadow", "max_invalid_fraction": 0.0, "min_ac_rms": 1e-8,
                  "threshold_comparison": ">", "support_threshold_floor": 0.5,
                  "max_unsupported_false_acceptance_fraction": 0.01,
                  "support_interpretation": "uncalibrated_supported_domain_not_efficacy",
                  "confidence_interpretation": "one_minus_v2_artifact_probability_not_calibrated"},
    "runtime": {"enabled": False, "validation_status": "not_assessed",
                "deployment_scope": "explicit_offline_diagnostics_only", "production_validated": False},
}
CHECKPOINT_KEYS = frozenset({"checkpoint_schema_version", "model_version", "config", "gate_state",
                            "support_threshold", "threshold_calibration", "denoiser_sha256",
                            "dataset_sha256", "metadata"})


class LearnedDenoiseV3Error(ValueError):
    """Invalid v3 configuration/checkpoint or unsupported explicit device/input."""


def primitive_metadata(value: Any) -> bool:
    """Finite JSON-like plain metadata; tensor/numpy/custom/cyclic values fail closed."""
    try:
        return v2.primitive_metadata(value)
    except RecursionError:
        return False


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not np.isfinite(value):
        raise LearnedDenoiseV3Error(f"{name} must be a finite plain number")
    return float(value)


def validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict) or set(config) != set(DEFAULT_CONFIG) or not primitive_metadata(config):
        raise LearnedDenoiseV3Error("v3 config must have known keys and finite plain values")
    sections = {"model", "denoiser", "inference", "runtime"}
    for key in set(DEFAULT_CONFIG) - sections:
        if type(config[key]) is not type(DEFAULT_CONFIG[key]) or config[key] != DEFAULT_CONFIG[key]:
            raise LearnedDenoiseV3Error(f"unsupported v3 {key}; expected {DEFAULT_CONFIG[key]!r}")
    for section in sections:
        if not isinstance(config[section], dict) or set(config[section]) != set(DEFAULT_CONFIG[section]):
            raise LearnedDenoiseV3Error(f"invalid v3 {section} keys")
    model = config["model"]
    for key in ("channels", "kernel_sizes", "strides"):
        if (type(model[key]) is not list or model[key] != DEFAULT_CONFIG["model"][key]
                or any(type(value) is not int for value in model[key])):
            raise LearnedDenoiseV3Error(f"gate {key} must match the frozen preregistration")
    if type(model["hidden_features"]) is not int or model["hidden_features"] != 32:
        raise LearnedDenoiseV3Error("gate hidden_features must be 32")
    if _number(model["dropout"], "dropout") != 0.0:
        raise LearnedDenoiseV3Error("frozen gate dropout must be 0")
    denoiser = config["denoiser"]
    if (denoiser["model_version"] != v2.MODEL_VERSION or denoiser["frozen"] is not True
            or denoiser["device"] != "cpu"):
        raise LearnedDenoiseV3Error("v3 must use a frozen v2 denoiser explicitly on CPU")
    inference = config["inference"]
    for key in ("mode", "threshold_comparison", "support_interpretation", "confidence_interpretation"):
        if inference[key] != DEFAULT_CONFIG["inference"][key]:
            raise LearnedDenoiseV3Error(f"unsupported v3 inference {key}")
    for key in ("max_invalid_fraction", "support_threshold_floor", "max_unsupported_false_acceptance_fraction"):
        if _number(inference[key], key) != DEFAULT_CONFIG["inference"][key]:
            raise LearnedDenoiseV3Error(f"inference {key} must match the frozen contract")
    if _number(inference["min_ac_rms"], "min_ac_rms") <= 0:
        raise LearnedDenoiseV3Error("min_ac_rms must be positive")
    runtime = config["runtime"]
    if runtime["enabled"] is not False or runtime["production_validated"] is not False:
        raise LearnedDenoiseV3Error("v3 automatic/production enablement is forbidden")
    if runtime["deployment_scope"] != "explicit_offline_diagnostics_only":
        raise LearnedDenoiseV3Error("v3 is explicit offline diagnostics only")
    if runtime["validation_status"] not in ("not_assessed", "failed_holdout", "failed_known_session_regression"):
        raise LearnedDenoiseV3Error("v3 validation_status cannot claim production acceptance")


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        try:
            supplied = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as error:
            raise LearnedDenoiseV3Error(f"cannot read v3 config: {error}") from error
        if not isinstance(supplied, dict) or set(supplied) - set(config):
            raise LearnedDenoiseV3Error("v3 config must be an object with known keys")
        for key, value in supplied.items():
            if isinstance(config[key], dict):
                if not isinstance(value, dict):
                    raise LearnedDenoiseV3Error(f"{key} must be an object")
                config[key].update(value)
            else:
                config[key] = value
    validate_config(config)
    return config


if nn is not None:
    class DomainGateCNN(nn.Module):
        """Three Conv1d/ReLU stages, avg pool, linear32, logit; no channel mixing."""
        def __init__(self, config: dict[str, Any]):
            super().__init__()
            validate_config(config)
            architecture = config["model"]
            layers = []
            incoming = 1
            for outgoing, kernel, stride in zip(architecture["channels"], architecture["kernel_sizes"], architecture["strides"]):
                layers.extend([nn.Conv1d(incoming, outgoing, kernel, stride=stride, padding=kernel // 2), nn.ReLU()])
                incoming = outgoing
            self.features = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.hidden = nn.Linear(incoming, architecture["hidden_features"])
            self.logit = nn.Linear(architecture["hidden_features"], 1)

        def forward(self, normalized):
            if (not isinstance(normalized, torch.Tensor) or normalized.ndim != 3
                    or tuple(normalized.shape[1:]) != (1, 500) or normalized.shape[0] < 1
                    or normalized.dtype != torch.float32 or not bool(torch.isfinite(normalized).all())):
                raise LearnedDenoiseV3Error("gate input must be finite float32 normalized [B,1,500]")
            features = self.pool(self.features(normalized)).flatten(1)
            return self.logit(self.hidden(features)).squeeze(-1)
else:
    DomainGateCNN = None


def build_gate(config: dict[str, Any]):
    validate_config(config)
    if DomainGateCNN is None:
        raise LearnedDenoiseV3Error("optional PyTorch is required for v3 gate construction")
    return DomainGateCNN(config)


@dataclass(frozen=True)
class ModelBundle:
    gate: Any
    denoiser: v2.ModelBundle
    config: dict[str, Any]
    support_threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Path | None = None
    device: str = "cpu"


@dataclass(frozen=True)
class DenoiseResult:
    denoised: np.ndarray
    model_denoised: np.ndarray
    artifact_probability: np.ndarray
    confidence: np.ndarray
    replacement_mask: np.ndarray
    valid_mask: np.ndarray
    rejected_mask: np.ndarray
    support_probability: np.ndarray
    domain_rejected_mask: np.ndarray
    diagnostics: dict[str, Any]


def _device(device: Any) -> str:
    if torch is None:
        raise LearnedDenoiseV3Error("optional PyTorch is required for v3 inference")
    try:
        selected = torch.device(device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise LearnedDenoiseV3Error(f"invalid explicit gate device: {device!r}") from error
    if selected.type == "cpu" and selected.index is None:
        return "cpu"
    if selected.type != "cuda":
        raise LearnedDenoiseV3Error("gate device must explicitly be cpu or cuda[:index]")
    if not torch.cuda.is_available():
        raise LearnedDenoiseV3Error("CUDA gate device requested but CUDA is unavailable; no CPU fallback")
    index = selected.index if selected.index is not None else torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise LearnedDenoiseV3Error("requested CUDA gate device index is unavailable")
    return f"cuda:{index}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, name: str) -> None:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise LearnedDenoiseV3Error(f"{name} must be a lowercase SHA256 hex digest")


def _denoiser_config(config: dict[str, Any], denoiser_config: dict[str, Any]) -> None:
    try:
        v2.validate_config(denoiser_config)
    except (ValueError, TypeError, KeyError) as error:
        raise LearnedDenoiseV3Error(f"invalid frozen v2 config: {error}") from error
    for key in ("sample_rate_hz", "input_channels", "window_samples", "hop_samples", "normalization"):
        if denoiser_config[key] != config[key]:
            raise LearnedDenoiseV3Error(f"frozen v2 {key} does not match v3")
    for key in ("max_invalid_fraction", "min_ac_rms"):
        if denoiser_config["inference"][key] != config["inference"][key]:
            raise LearnedDenoiseV3Error(f"frozen v2 normalization/rejection {key} does not match v3")


def validate_checkpoint_metadata(checkpoint: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Only the nine preregistered top-level fields are mandatory, never extra audit fields."""
    validate_config(config)
    if (not isinstance(checkpoint, dict) or not CHECKPOINT_KEYS <= set(checkpoint)
            or checkpoint["model_version"] != MODEL_VERSION
            or type(checkpoint["checkpoint_schema_version"]) is not int or checkpoint["checkpoint_schema_version"] != 1):
        raise LearnedDenoiseV3Error("unsupported v3 checkpoint version/schema or missing frozen keys")
    metadata = {key: value for key, value in checkpoint.items() if key != "gate_state"}
    if not primitive_metadata(metadata) or not isinstance(checkpoint["metadata"], dict):
        raise LearnedDenoiseV3Error("checkpoint metadata must be finite plain scalars/lists/dicts")
    saved = checkpoint["config"]
    validate_config(saved)
    for key in set(DEFAULT_CONFIG) - {"runtime"}:
        if saved[key] != config[key]:
            raise LearnedDenoiseV3Error(f"v3 checkpoint {key} does not match config architecture/inference")
    if "architecture" in checkpoint and checkpoint["architecture"] != config["model"]:
        raise LearnedDenoiseV3Error("optional checkpoint architecture does not match config")
    _digest(checkpoint["denoiser_sha256"], "denoiser_sha256")
    _digest(checkpoint["dataset_sha256"], "dataset_sha256")
    threshold = _number(checkpoint["support_threshold"], "support_threshold")
    if not config["inference"]["support_threshold_floor"] <= threshold <= 1:
        raise LearnedDenoiseV3Error("support_threshold must be in [0.5,1]")
    calibration = checkpoint["threshold_calibration"]
    if (not isinstance(calibration, dict) or calibration.get("scope") != "validation_only"
            or calibration.get("comparison") != ">" or calibration.get("empirical_only") is not True
            or calibration.get("production_calibrated") is not False
            or calibration.get("method") != "unsupported_score_order_statistic"):
        raise LearnedDenoiseV3Error("missing validation-only unsupported order-statistic threshold provenance")
    count = calibration.get("unsupported_window_count")
    if type(count) is not int or count < 1:
        raise LearnedDenoiseV3Error("unsupported validation count must be a positive integer")
    observed = _number(calibration.get("observed_unsupported_false_acceptance_fraction"), "observed unsupported false acceptance")
    maximum = _number(calibration.get("max_unsupported_false_acceptance_fraction"), "maximum unsupported false acceptance")
    if not 0 <= observed <= maximum <= config["inference"]["max_unsupported_false_acceptance_fraction"]:
        raise LearnedDenoiseV3Error("empirical unsupported false acceptance must not exceed 1%")
    return deepcopy(metadata)


def load_model(checkpoint_path: Path | str, denoiser_checkpoint_path: Path | str,
               config: dict[str, Any] | Path | str | None = None, device: str = "cpu") -> ModelBundle:
    selected_device = _device(device)
    try:
        if not str(checkpoint_path).strip() or not str(denoiser_checkpoint_path).strip():
            raise LearnedDenoiseV3Error("checkpoint paths must not be blank")
        gate_path, denoiser_path = Path(checkpoint_path).resolve(), Path(denoiser_checkpoint_path).resolve()
        gate_hash, denoiser_hash = _sha256(gate_path), _sha256(denoiser_path)
        checkpoint = torch.load(gate_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise LearnedDenoiseV3Error("v3 checkpoint must be a mapping")
        chosen = deepcopy(checkpoint.get("config")) if config is None else (
            load_config(config) if isinstance(config, (str, Path)) else deepcopy(config))
        metadata = validate_checkpoint_metadata(checkpoint, chosen)
        if metadata["denoiser_sha256"] != denoiser_hash:
            raise LearnedDenoiseV3Error("frozen denoiser SHA256 does not match the bound checkpoint")
        # Read the frozen v2 config from its own safe checkpoint, not a mutable default file.
        denoiser_checkpoint = torch.load(denoiser_path, map_location="cpu", weights_only=True)
        if not isinstance(denoiser_checkpoint, dict):
            raise LearnedDenoiseV3Error("frozen v2 checkpoint must be a mapping")
        _denoiser_config(chosen, denoiser_checkpoint.get("config"))
        denoiser = v2.load_model(denoiser_path, denoiser_checkpoint["config"], device="cpu")
        gate = build_gate(chosen)
        state = checkpoint["gate_state"]
        if (not isinstance(state, Mapping) or any(type(key) is not str or not isinstance(value, torch.Tensor)
                or not value.is_floating_point() or not bool(torch.isfinite(value).all()) for key, value in state.items())):
            raise LearnedDenoiseV3Error("gate_state must contain finite real floating tensors")
        gate.load_state_dict(state, strict=True)
        if not all(bool(torch.isfinite(tensor).all()) for tensor in gate.state_dict().values()):
            raise LearnedDenoiseV3Error("loaded gate contains non-finite weights")
        if _sha256(gate_path) != gate_hash or _sha256(denoiser_path) != denoiser_hash:
            raise LearnedDenoiseV3Error("checkpoint changed during loading")
        gate.to(selected_device).eval().requires_grad_(False)
        denoiser.model.eval().requires_grad_(False)
    except (OSError, pickle.UnpicklingError, ValueError, TypeError, KeyError, RuntimeError, EOFError, AttributeError) as error:
        if isinstance(error, LearnedDenoiseV3Error):
            raise
        raise LearnedDenoiseV3Error(f"cannot safely load v3/frozen v2 checkpoint: {error}") from error
    return ModelBundle(gate, denoiser, deepcopy(chosen), float(metadata["support_threshold"]),
                       metadata, gate_path, selected_device)


def _raw_array(signal: np.ndarray) -> np.ndarray:
    source = np.asarray(signal)
    if source.ndim != 2 or source.dtype.kind not in "iuf" or not all(source.shape) or source.dtype.itemsize > 8:
        raise LearnedDenoiseV3Error("signal must be a nonempty real [channels,samples] array of at most 64-bit values")
    if source.dtype.kind in "iu" and np.any(np.abs(source.astype(np.longdouble)) > 2 ** 53):
        raise LearnedDenoiseV3Error("integer raw values exceed exact float64 restoration range")
    # Matches frozen v2 restoration precision. Unselected float64 bits (including
    # signed zero/NaN payloads) are copied, never reconstructed from normalized input.
    return source.astype(np.float64, copy=True)


def apply_model_to_array(bundle: ModelBundle, signal: np.ndarray,
                         valid_mask: np.ndarray | None = None) -> DenoiseResult:
    validate_config(bundle.config)
    _denoiser_config(bundle.config, bundle.denoiser.config)
    selected_device = _device(bundle.device)
    threshold = _number(bundle.support_threshold, "bundle support_threshold")
    if not 0.5 <= threshold <= 1:
        raise LearnedDenoiseV3Error("bundle support_threshold must be in [0.5,1]")
    x = _raw_array(signal)
    valid = np.isfinite(x)
    if valid_mask is not None:
        try:
            valid &= v2.validate_mask(valid_mask, x.shape)
        except ValueError as error:
            raise LearnedDenoiseV3Error(str(error)) from error
    # The bound v2 implementation remains on CPU, even for a CUDA gate.
    try:
        tensors = list(bundle.denoiser.model.parameters()) + list(bundle.denoiser.model.buffers())
    except AttributeError as error:
        raise LearnedDenoiseV3Error("bundle must contain a v2 model") from error
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise LearnedDenoiseV3Error("frozen v2 denoiser must remain explicitly on CPU")
    denoised, candidate = x.copy(), x.copy()
    artifact = np.full(x.shape, np.nan, dtype=np.float64)
    confidence = np.zeros(x.shape, dtype=np.float64)
    support = np.full(x.shape, np.nan, dtype=np.float64)
    replacement = np.zeros(x.shape, dtype=bool)
    rejected = np.ones(x.shape, dtype=bool)
    domain_rejected = np.ones(x.shape, dtype=bool)
    window = bundle.config["window_samples"]
    counts = {key: 0 for key in ("supported_windows", "unsupported_windows", "domain_prediction_rejected_windows",
              "invalid_windows", "constant_or_normalization_rejected_windows", "v2_rejected_windows", "replaced_windows")}
    failure = None
    base_diagnostics = {}
    try:
        base = v2.apply_model_to_array(bundle.denoiser, x, valid_mask=valid)
        fields = ("denoised", "model_denoised", "artifact_probability", "confidence",
                  "replacement_mask", "valid_mask", "rejected_mask")
        if any(not isinstance(getattr(base, key, None), np.ndarray) or getattr(base, key).shape != x.shape for key in fields):
            raise LearnedDenoiseV3Error("invalid frozen v2 result shapes")
        if any(getattr(base, key).dtype.kind != "b" for key in ("replacement_mask", "valid_mask", "rejected_mask")):
            raise LearnedDenoiseV3Error("invalid frozen v2 result masks")
        if any(getattr(base, key).dtype.kind not in "fiu" for key in fields[:4]) or not np.array_equal(base.valid_mask, valid):
            raise LearnedDenoiseV3Error("invalid frozen v2 output types or validity mask")
        if not primitive_metadata(base.diagnostics):
            raise LearnedDenoiseV3Error("invalid frozen v2 diagnostic metadata")
        base_diagnostics = deepcopy(base.diagnostics)
    except Exception as error:
        # No unsafe v2 result may accidentally be treated as a gate-supported window.
        base = None
        failure = str(error)
        counts["v2_rejected_windows"] = x.shape[0] * (x.shape[1] // window)
    if base is not None:
        for start in range(0, x.shape[1] - window + 1, window):
            end = start + window
            for channel in range(x.shape[0]):
                position = (channel, slice(start, end))
                if not valid[position].all():
                    counts["invalid_windows"] += 1
                    continue
                try:
                    norm, _, _ = v2.normalize_window(x[position], valid[position], bundle.config["inference"]["min_ac_rms"])
                except ValueError:
                    counts["constant_or_normalization_rejected_windows"] += 1
                    continue
                if base.rejected_mask[position].any():
                    counts["v2_rejected_windows"] += 1
                    continue
                p = base.artifact_probability[position]
                c = base.confidence[position]
                original_replacement = base.replacement_mask[position]
                # Validate all predictions, including unselected v2 candidates.
                if (not np.isfinite(base.model_denoised[position]).all() or not np.isfinite(base.denoised[position]).all()
                        or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.isfinite(c).all()
                        or not np.allclose(c, 1.0 - p, atol=1e-12, rtol=1e-12)
                        or not np.array_equal(base.denoised[position][original_replacement], base.model_denoised[position][original_replacement])
                        or not np.array_equal(base.denoised[position][~original_replacement], x[position][~original_replacement])):
                    counts["v2_rejected_windows"] += 1
                    continue
                candidate[position] = base.model_denoised[position]
                artifact[position] = p
                confidence[position] = c  # inherited 1-p, NOT domain/denoising efficacy confidence
                try:
                    tensor = torch.from_numpy(norm[None, None]).to(selected_device)
                    with torch.no_grad():
                        logits = bundle.gate(tensor)
                    if (not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1,)
                            or not logits.is_floating_point() or logits.device != tensor.device
                            or not bool(torch.isfinite(logits).all())):
                        raise LearnedDenoiseV3Error("invalid/non-finite domain logits")
                    score = float(torch.sigmoid(logits).detach().cpu().item())
                    if not np.isfinite(score) or not 0 <= score <= 1:
                        raise LearnedDenoiseV3Error("invalid domain score")
                except Exception:
                    counts["domain_prediction_rejected_windows"] += 1
                    continue
                support[position] = score
                if not score > threshold:  # Strict >; equality is unsupported, never accepted.
                    counts["unsupported_windows"] += 1
                    continue
                domain_rejected[position] = False
                rejected[position] = False
                counts["supported_windows"] += 1
                replacement[position] = original_replacement
                # Only a v2 replacement AND supported-domain score can alter output.
                denoised[position][original_replacement] = base.denoised[position][original_replacement]
                if original_replacement.any():
                    counts["replaced_windows"] += 1
    changed = denoised.view(np.uint64) != x.view(np.uint64)
    diagnostics = {
        "model_version": MODEL_VERSION, "window_samples": window, "hop_samples": window,
        "full_windows_per_channel": x.shape[1] // window, "tail_samples_per_channel": x.shape[1] % window,
        "support_threshold": threshold, "threshold_comparison": ">",
        "threshold_scope": "validation_only_unsupported_order_statistic_empirical_not_production_calibrated",
        "domain_probability_interpretation": "uncalibrated_supported_domain_NOT_artifact_or_denoising_efficacy",
        "confidence_interpretation": "inherited_v2_one_minus_artifact_probability_NOT_calibrated_success_probability",
        "normalization": bundle.config["normalization"], "restoration_dtype": "float64",
        "gate_device": selected_device, "denoiser_device": "cpu", "denoiser_frozen": True,
        "replaced_samples": int(np.count_nonzero(replacement)), "changed_samples": int(np.count_nonzero(changed)),
        "rejected_samples": int(np.count_nonzero(rejected)), "domain_rejected_samples": int(np.count_nonzero(domain_rejected)),
        "domain_rejected_fraction": float(np.mean(domain_rejected)), "replacement_fraction": float(np.mean(replacement)),
        "automatic_processing_enabled": False, "diagnostic_only": True, "release_gate_passed": False,
        "production_validated": False, "raw_is_immutable": True, "original_quality_unchanged": True,
        "missing_data_repaired": False, "runtime": deepcopy(bundle.config["runtime"]),
        "v2_diagnostics": base_diagnostics, **counts,
    }
    if failure is not None:
        diagnostics["v2_prediction_error"] = failure
    return DenoiseResult(denoised, candidate, artifact, confidence, replacement, valid, rejected,
                         support, domain_rejected, diagnostics)


__all__ = ["MODEL_VERSION", "DEFAULT_CONFIG", "LearnedDenoiseV3Error", "DomainGateCNN", "ModelBundle",
           "DenoiseResult", "validate_config", "load_config", "build_gate", "primitive_metadata",
           "validate_checkpoint_metadata", "load_model", "apply_model_to_array"]
