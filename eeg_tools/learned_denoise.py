"""Learned EEG denoising primitives with an optional PyTorch backend.

The module is intentionally non-destructive: raw session files are never changed.
The learned model emits a denoised derived track, an artifact probability and a
confidence signal.  If PyTorch is not installed, configuration and dataset
helpers remain usable and inference fails with an actionable error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

MODEL_VERSION = "learned_denoise_v1"

DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "status": "formal",
    "model_version": MODEL_VERSION,
    "model_type": "residual_tcn",
    "sample_rate_hz": 250,
    "channel_count": 8,
    "window_samples": 1000,
    "hop_samples": 250,
    "normalization": "per_channel_robust",
    "inference": {
        "mode": "shadow",
        "min_confidence_for_replacement": 0.65,
        "max_invalid_fraction": 0.2,
        "preserve_invalid_samples": True,
    },
    "loss": {
        "time_weight": 1.0,
        "spectral_weight": 0.25,
        "mask_weight": 0.5,
        "identity_weight": 0.2,
        "confidence_weight": 0.1,
        "ssvep_weight": 0.15,
    },
}


class LearnedDenoiseError(ValueError):
    """Raised for invalid learned-denoising configuration or model output."""


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LearnedDenoiseError(f"invalid learned denoiser config: {path}: {error}") from error
    if not isinstance(value, dict):
        raise LearnedDenoiseError("learned denoiser config must be an object")
    config = _merge_config(value)
    validate_config(config)
    return config


def _merge_config(value: dict[str, Any]) -> dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    for key, item in value.items():
        if key not in {"inference", "loss"}:
            config[key] = item
    for section in ("inference", "loss"):
        supplied = value.get(section)
        if isinstance(supplied, dict):
            config[section].update(supplied)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", 0)) != 1:
        raise LearnedDenoiseError("unsupported learned denoiser schema_version")
    if config.get("status") != "formal":
        raise LearnedDenoiseError("learned denoiser config status must be formal")
    if config.get("model_version") != MODEL_VERSION:
        raise LearnedDenoiseError("unsupported learned denoiser model_version")
    if config.get("model_type") != "residual_tcn":
        raise LearnedDenoiseError("only residual_tcn is supported")
    sample_rate = float(config.get("sample_rate_hz", 0))
    channels = int(config.get("channel_count", 0))
    window = int(config.get("window_samples", 0))
    hop = int(config.get("hop_samples", 0))
    if not np.isfinite(sample_rate) or sample_rate <= 0 or channels <= 0 or window <= 0 or hop <= 0:
        raise LearnedDenoiseError("sample_rate_hz, channel_count, window_samples and hop_samples must be positive")
    if hop > window:
        raise LearnedDenoiseError("hop_samples cannot exceed window_samples")
    inference = config.get("inference")
    if not isinstance(inference, dict):
        raise LearnedDenoiseError("inference must be an object")
    confidence = float(inference.get("min_confidence_for_replacement", -1))
    invalid = float(inference.get("max_invalid_fraction", -1))
    if not 0 <= confidence <= 1 or not 0 <= invalid <= 1:
        raise LearnedDenoiseError("inference confidence and invalid fraction must be in [0, 1]")


def robust_normalize(values: np.ndarray, valid_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize [channels, samples] without letting invalid samples define scale."""
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2:
        raise LearnedDenoiseError("signal must have shape [channels, samples]")
    finite = np.isfinite(x)
    if valid_mask is not None:
        finite &= _validated_mask(valid_mask, x.shape)
    safe = np.where(finite, x, np.nan)
    center = np.nanmedian(safe, axis=1)
    q25 = np.nanpercentile(safe, 25, axis=1)
    q75 = np.nanpercentile(safe, 75, axis=1)
    scale = (q75 - q25) / 1.349
    fallback = np.nanstd(safe, axis=1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    center = np.where(np.isfinite(center), center, 0.0)
    normalized = (np.where(finite, x, center[:, None]) - center[:, None]) / scale[:, None]
    return normalized.astype(np.float32), center.astype(np.float32), scale.astype(np.float32)


def robust_denormalize(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    return x * np.asarray(scale, dtype=np.float32)[:, None] + np.asarray(center, dtype=np.float32)[:, None]


try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised by environments without optional ML deps
    torch = None
    nn = None


if nn is not None:

    class _ResidualBlock(nn.Module):
        def __init__(self, channels: int, dilation: int, dropout: float = 0.05) -> None:
            super().__init__()
            padding = dilation
            self.net = nn.Sequential(
                nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation),
                nn.GroupNorm(1, channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation),
                nn.GroupNorm(1, channels),
            )
            self.activation = nn.GELU()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # Symmetric padding keeps the TCN implementation portable. The real-time
            # path uses short rolling windows and marks the model as causal=false.
            residual = self.net(x)
            if residual.shape[-1] != x.shape[-1]:
                residual = residual[..., : x.shape[-1]]
            return self.activation(x + residual)


    class ResidualTCNDenoiser(nn.Module):
        """Small multi-channel residual TCN for 250 Hz, 8-channel EEG."""

        def __init__(self, channel_count: int = 8, hidden_channels: int = 48, dropout: float = 0.05) -> None:
            super().__init__()
            self.channel_count = int(channel_count)
            self.input = nn.Sequential(
                nn.Conv1d(channel_count * 2, hidden_channels, 5, padding=2),
                nn.GroupNorm(1, hidden_channels),
                nn.GELU(),
            )
            self.blocks = nn.Sequential(
                _ResidualBlock(hidden_channels, 1, dropout),
                _ResidualBlock(hidden_channels, 2, dropout),
                _ResidualBlock(hidden_channels, 4, dropout),
                _ResidualBlock(hidden_channels, 8, dropout),
                _ResidualBlock(hidden_channels, 16, dropout),
            )
            self.noise_head = nn.Conv1d(hidden_channels, channel_count, 1)
            self.mask_head = nn.Conv1d(hidden_channels, channel_count, 1)
            self.confidence_head = nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels // 2, 1),
                nn.GELU(),
                nn.Conv1d(hidden_channels // 2, 1, 1),
                nn.Sigmoid(),
            )

        def forward(self, signal: "torch.Tensor", valid_mask: "torch.Tensor") -> dict[str, "torch.Tensor"]:
            if signal.ndim != 3 or valid_mask.shape != signal.shape:
                raise ValueError("signal and valid_mask must both have shape [batch, channels, samples]")
            features = self.blocks(self.input(torch.cat([signal, valid_mask], dim=1)))
            noise = self.noise_head(features)
            denoised = signal - noise
            return {
                "denoised": denoised,
                "noise": noise,
                "artifact_probability": torch.sigmoid(self.mask_head(features)),
                "confidence": self.confidence_head(features),
            }

else:

    ResidualTCNDenoiser = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    config: dict[str, Any]
    checkpoint_path: Path
    # Defaults keep legacy three-argument construction compatible. Never expose weights here.
    checkpoint_metadata: dict[str, Any] = field(default_factory=dict)


def load_model(checkpoint_path: Path, config: dict[str, Any], device: str = "cpu") -> ModelBundle:
    if torch is None or ResidualTCNDenoiser is None:
        raise LearnedDenoiseError("PyTorch is required for learned denoiser inference; install the optional ml dependencies")
    validate_config(config)
    try:
        # No unsafe pickle fallback, including for legacy checkpoints.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except (pickle.UnpicklingError, TypeError, RuntimeError) as error:
        raise LearnedDenoiseError(f"cannot safely load learned denoiser checkpoint: {error}") from error
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise LearnedDenoiseError("checkpoint must include model_state and training config")
    if checkpoint.get("checkpoint_schema_version", 1) != 1:
        raise LearnedDenoiseError("unsupported checkpoint_schema_version")
    saved_config = checkpoint["config"]
    version = checkpoint.get("model_version", saved_config.get("model_version"))
    if version != MODEL_VERSION or version != config["model_version"]:
        raise LearnedDenoiseError("checkpoint model_version does not match supported config")
    if saved_config.get("model_version", version) != version:
        raise LearnedDenoiseError("checkpoint model_version metadata is inconsistent")
    # Historical status does not change inference compatibility. Sampling rate and
    # channel count, however, cannot safely be guessed from a weight tensor.
    for key in ("sample_rate_hz", "channel_count"):
        if key not in saved_config or saved_config[key] != config[key]:
            raise LearnedDenoiseError(f"checkpoint {key} does not match config")
    for key in ("schema_version", "model_type", "normalization", "window_samples"):
        if key in saved_config and saved_config[key] != config.get(key):
            raise LearnedDenoiseError(f"checkpoint {key} does not match config")
    for key in ("channel_names", "channel_order", "channel_config", "channels"):
        if key in saved_config or key in config:
            if saved_config.get(key) != config.get(key):
                raise LearnedDenoiseError(f"checkpoint {key} does not match config")
    model = ResidualTCNDenoiser(
        channel_count=int(config["channel_count"]),
        hidden_channels=int(checkpoint.get("hidden_channels", 48)),
        dropout=0.0,
    )
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except (KeyError, RuntimeError, TypeError) as error:
        raise LearnedDenoiseError(f"invalid checkpoint model_state: {error}") from error
    model.to(device)
    model.eval()
    metadata = {key: value for key, value in checkpoint.items() if key != "model_state"}
    return ModelBundle(model=model, config=config, checkpoint_path=checkpoint_path, checkpoint_metadata=metadata)


def _validated_mask(valid_mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    mask = np.asarray(valid_mask)
    if mask.shape != shape:
        raise LearnedDenoiseError("valid_mask must have exactly the same shape as signal; broadcasting is forbidden")
    if mask.dtype.kind not in "biuf" or not np.all((mask == 0) | (mask == 1)):
        raise LearnedDenoiseError("valid_mask must contain only booleans or finite binary values")
    return mask.astype(bool)


def apply_model_to_array(
    bundle: ModelBundle,
    signal: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Produce derived tracks; reject unsafe windows and preserve original invalid values.

    Padding is not an acquisition dropout. Gating counts only real samples and
    uses the worst channel. A rejected window vetoes replacement over its entire
    real span, even where a neighbouring accepted window overlaps it.
    """
    if torch is None:
        raise LearnedDenoiseError("PyTorch is required for learned denoiser inference")
    validate_config(bundle.config)
    raw = np.asarray(signal)
    if raw.ndim != 2 or raw.dtype.kind not in "iuf":
        raise LearnedDenoiseError("signal must be a real numeric array with shape [channels, samples]")
    with np.errstate(over="ignore", invalid="ignore"):
        x = np.asarray(raw, dtype=np.float32)
    if np.any(np.isfinite(raw) & ~np.isfinite(x)):
        raise LearnedDenoiseError("signal contains values outside the finite float32 range")
    channels, samples = x.shape
    if channels != int(bundle.config["channel_count"]):
        raise LearnedDenoiseError(f"expected {bundle.config['channel_count']} channels, got {channels}")
    if samples == 0:
        raise LearnedDenoiseError("signal must contain at least one real sample")
    valid = np.isfinite(x)
    if valid_mask is not None:
        valid &= _validated_mask(valid_mask, x.shape)
    window = int(bundle.config["window_samples"])
    hop = int(bundle.config["hop_samples"])
    padded_samples = max(window, samples)
    if padded_samples > window:
        padded_samples = window + int(np.ceil((padded_samples - window) / hop)) * hop
    pad = padded_samples - samples
    padded = np.pad(x, ((0, 0), (0, pad)), mode="constant", constant_values=np.nan)
    padded_valid = np.pad(valid, ((0, 0), (0, pad)), mode="constant", constant_values=False)
    out = np.zeros_like(padded, dtype=np.float64)
    weights = np.zeros(padded_samples, dtype=np.float64)
    masks = np.zeros_like(padded, dtype=np.float64)
    confidences = np.zeros(padded_samples, dtype=np.float64)
    rejected = np.zeros(padded_samples, dtype=bool)
    analysis_window = np.maximum(np.hanning(window).astype(np.float64), 1e-3)
    device = next(bundle.model.parameters(), torch.empty(0)).device
    max_invalid = float(bundle.config["inference"]["max_invalid_fraction"])
    with torch.no_grad():
        for start in range(0, padded_samples - window + 1, hop):
            end = start + window
            real_samples = min(end, samples) - start
            segment = padded[:, start:end]
            segment_valid = padded_valid[:, start:end]
            if real_samples <= 0:
                continue
            # Per-channel gating also rejects a wholly absent channel hidden by
            # an otherwise small global missing fraction.
            invalid_fraction = 1.0 - segment_valid[:, :real_samples].mean(axis=1)
            if np.any(invalid_fraction > max_invalid) or not np.all(segment_valid[:, :real_samples].any(axis=1)):
                rejected[start:end] = True
                continue
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                normalized, center, scale = robust_normalize(segment, segment_valid)
            if not np.all(np.isfinite(normalized)):
                rejected[start:end] = True
                continue
            signal_tensor = torch.from_numpy(normalized[None]).to(device)
            valid_tensor = torch.from_numpy(segment_valid.astype(np.float32)[None]).to(device)
            prediction = bundle.model(signal_tensor, valid_tensor)
            expected_shapes = {"denoised": (1, channels, window), "artifact_probability": (1, channels, window), "confidence": (1, 1, window)}
            if not isinstance(prediction, dict) or any(
                not isinstance(prediction.get(key), torch.Tensor) or tuple(prediction[key].shape) != shape
                for key, shape in expected_shapes.items()
            ):
                rejected[start:end] = True
                continue
            denoised = prediction["denoised"][0].detach().cpu().numpy()
            artifact = prediction["artifact_probability"][0].detach().cpu().numpy()
            confidence = prediction["confidence"][0, 0].detach().cpu().numpy()
            with np.errstate(over="ignore", invalid="ignore"):
                denoised = robust_denormalize(denoised, center, scale)
            if (not all(np.all(np.isfinite(value)) for value in (denoised, artifact, confidence))
                    or np.any((artifact < 0) | (artifact > 1))
                    or np.any((confidence < 0) | (confidence > 1))):
                rejected[start:end] = True
                continue
            out[:, start:end] += denoised * analysis_window[None, :]
            masks[:, start:end] += artifact * analysis_window[None, :]
            confidences[start:end] += confidence * analysis_window
            weights[start:end] += analysis_window
    safe_weights = np.maximum(weights, 1e-6)
    with np.errstate(over="ignore", invalid="ignore"):
        candidate = (out[:, :samples] / safe_weights[None, :samples]).astype(np.float32)
    artifact = (masks[:, :samples] / safe_weights[None, :samples]).astype(np.float32)
    confidence = (confidences[:samples] / safe_weights[:samples]).astype(np.float32)
    unsafe = rejected[:samples] | (weights[:samples] == 0) | ~np.all(np.isfinite(candidate), axis=0)
    confidence[unsafe] = 0.0
    artifact[:, unsafe] = 0.0
    candidate[:, unsafe] = x[:, unsafe]
    candidate[~valid] = x[~valid]
    replacement_mask = (
        (confidence[None, :] >= float(bundle.config["inference"]["min_confidence_for_replacement"]))
        & valid & ~unsafe[None, :]
    )
    deployed = np.where(replacement_mask, candidate, x).astype(np.float32)
    return {
        "denoised": deployed,
        "model_denoised": candidate,
        "replacement_mask": replacement_mask,
        "artifact_probability": artifact,
        "confidence": confidence,
        "valid_mask": valid,
    }


__all__ = [
    "DEFAULT_CONFIG",
    "MODEL_VERSION",
    "LearnedDenoiseError",
    "ModelBundle",
    "ResidualTCNDenoiser",
    "apply_model_to_array",
    "load_config",
    "load_model",
    "robust_denormalize",
    "robust_normalize",
    "torch_available",
    "validate_config",
]
