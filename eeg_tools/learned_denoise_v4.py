"""Frozen, diagnostic-only EOG denoiser core with conservative safety gates."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # optional ML dependency
    torch = None
    nn = None

MODEL_VERSION = "learned_denoise_v4"
DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "status": "formal",
    "model_version": MODEL_VERSION,
    "model_type": "eog_only_bandlimited_single_channel_tcn",
    "sample_rate_hz": 250,
    "input_channels": 1,
    "window_samples": 500,
    "hop_samples": 500,
    "normalization": "per_channel_window_noisy_mean_ac_rms",
    "model": {
        "hidden_channels": 24,
        "dilations": [1, 2, 4, 8, 16],
        "kernel_size": 5,
        "dropout": 0.0,
        "residual_peak_bound": 2.0,
        "residual_band_hz": [0.5, 6.0],
    },
    "inference": {
        "mode": "shadow",
        "min_ac_rms": 1e-8,
        "max_invalid_fraction": 0.0,
        "threshold_comparison": ">",
        "artifact_threshold_floor": 0.5,
        "clean_false_replacement_max": 0.01,
        "max_high_frequency_fraction_above_45hz": 0.1,
        "max_abs_dc_to_ac_rms": 20.0,
        "max_change_ac_rms_ratio": 0.75,
        "protected_band_hz": [8.0, 40.0],
        "protected_band_relative_tolerance": 1e-6,
        "confidence_interpretation": "one_minus_artifact_score_not_calibrated_efficacy",
    },
    "loss": {
        "time_weight": 1.0,
        "clean_identity_multiplier": 5.0,
        "artifact_weight": 0.5,
    },
    "runtime": {
        "enabled": False,
        "validation_status": "not_assessed",
        "deployment_scope": "explicit_offline_diagnostics_only",
        "production_validated": False,
    },
}
CHECKPOINT_KEYS = frozenset({
    "checkpoint_schema_version", "model_version", "config", "model_state",
    "artifact_threshold", "threshold_calibration", "dataset_sha256", "metadata",
})


class LearnedDenoiseV4Error(ValueError):
    """Fail-closed v4 config, checkpoint, input, or prediction error."""


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        raise LearnedDenoiseV4Error(f"{name} must be a finite plain number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise LearnedDenoiseV4Error(f"{name} must be finite") from error
    if not np.isfinite(converted) or (positive and converted <= 0):
        raise LearnedDenoiseV4Error(f"{name} must be finite" + (" and positive" if positive else ""))
    return converted


def primitive_metadata(value: Any) -> bool:
    """Return true only for finite JSON-like primitive metadata."""
    try:
        if value is None or type(value) in (bool, int, str):
            return True
        if type(value) is float:
            return bool(np.isfinite(value))
        if type(value) is list:
            return all(primitive_metadata(item) for item in value)
        if type(value) is dict:
            return all(type(key) is str and primitive_metadata(item) for key, item in value.items())
    except RecursionError:
        return False
    return False


def validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict) or set(config) != set(DEFAULT_CONFIG) or not primitive_metadata(config):
        raise LearnedDenoiseV4Error("v4 config must have exactly the frozen keys and finite primitive values")
    scalar_keys = ("schema_version", "status", "model_version", "model_type", "sample_rate_hz",
                   "input_channels", "window_samples", "hop_samples", "normalization")
    for key in scalar_keys:
        if config[key] != DEFAULT_CONFIG[key] or type(config[key]) is not type(DEFAULT_CONFIG[key]):
            raise LearnedDenoiseV4Error(f"unsupported v4 {key}; expected {DEFAULT_CONFIG[key]!r}")
    for section in ("model", "inference", "loss", "runtime"):
        if not isinstance(config[section], dict) or set(config[section]) != set(DEFAULT_CONFIG[section]):
            raise LearnedDenoiseV4Error(f"invalid frozen v4 {section} keys")
    model = config["model"]
    if (type(model["hidden_channels"]) is not int or model["hidden_channels"] != 24
            or model["dilations"] != [1, 2, 4, 8, 16]
            or any(type(item) is not int for item in model["dilations"])
            or type(model["kernel_size"]) is not int or model["kernel_size"] != 5):
        raise LearnedDenoiseV4Error("v4 TCN architecture is not the frozen contract")
    if _finite_number(model["dropout"], "dropout") != 0.0:
        raise LearnedDenoiseV4Error("v4 dropout must be zero")
    if _finite_number(model["residual_peak_bound"], "residual_peak_bound", positive=True) != 2.0:
        raise LearnedDenoiseV4Error("v4 residual_peak_bound must be 2.0")
    if model["residual_band_hz"] != [0.5, 6.0]:
        raise LearnedDenoiseV4Error("v4 residual_band_hz must be [0.5, 6.0]")
    for section in ("inference", "loss"):
        for key, expected in DEFAULT_CONFIG[section].items():
            value = config[section][key]
            if isinstance(expected, (int, float)):
                if _finite_number(value, key) != expected:
                    raise LearnedDenoiseV4Error(f"frozen v4 {section}.{key} cannot be changed")
            elif type(value) is not type(expected) or value != expected:
                raise LearnedDenoiseV4Error(f"frozen v4 {section}.{key} cannot be changed")
    runtime = config["runtime"]
    if runtime["enabled"] is not False or runtime["production_validated"] is not False:
        raise LearnedDenoiseV4Error("v4 automatic/production enablement is forbidden")
    if runtime["validation_status"] not in ("not_assessed", "failed_holdout", "failed_known_session_regression"):
        raise LearnedDenoiseV4Error("v4 validation_status cannot claim production acceptance")
    if runtime["deployment_scope"] != "explicit_offline_diagnostics_only":
        raise LearnedDenoiseV4Error("v4 deployment_scope must remain offline-only")


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        try:
            config = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as error:
            raise LearnedDenoiseV4Error(f"cannot read v4 config: {error}") from error
    validate_config(config)
    return deepcopy(config)


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


    class EOGBandlimitedTCN(nn.Module):
        """One-channel TCN whose residual is differentiably band-limited and bounded."""
        def __init__(self, config: dict[str, Any]):
            super().__init__()
            validate_config(config)
            architecture = config["model"]
            hidden = architecture["hidden_channels"]
            kernel = architecture["kernel_size"]
            self.window_samples = config["window_samples"]
            self.sample_rate_hz = float(config["sample_rate_hz"])
            self.residual_peak_bound = float(architecture["residual_peak_bound"])
            self.band_hz = tuple(float(item) for item in architecture["residual_band_hz"])
            self.input = nn.Sequential(nn.Conv1d(1, hidden, kernel, padding=kernel // 2),
                                       nn.GroupNorm(1, hidden), nn.GELU())
            self.blocks = nn.Sequential(*[_ResidualBlock(hidden, dilation, kernel, architecture["dropout"])
                                          for dilation in architecture["dilations"]])
            self.residual_head = nn.Conv1d(hidden, 1, 1)
            self.artifact_head = nn.Conv1d(hidden, 1, 1)
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)

        def _project_residual(self, residual):
            if residual.dtype != torch.float32 or residual.shape[-1] != 500:
                raise LearnedDenoiseV4Error("length-500 torch FFT must use float32, never half")
            frequencies = torch.fft.rfftfreq(self.window_samples, d=1.0 / self.sample_rate_hz,
                                              device=residual.device)
            band = (frequencies >= self.band_hz[0]) & (frequencies <= self.band_hz[1])
            spectrum = torch.fft.rfft(residual, dim=-1)
            projected = torch.fft.irfft(spectrum * band.to(spectrum.dtype), n=self.window_samples, dim=-1)
            projected = projected - projected.mean(dim=-1, keepdim=True)
            peak = projected.abs().amax(dim=-1, keepdim=True)
            scale = torch.where(peak > self.residual_peak_bound,
                                self.residual_peak_bound / torch.clamp(peak, min=torch.finfo(projected.dtype).tiny),
                                torch.ones_like(peak))
            return projected * scale

        def forward(self, signal):
            if (not isinstance(signal, torch.Tensor) or signal.ndim != 3
                    or tuple(signal.shape[1:]) != (1, self.window_samples)
                    or signal.dtype != torch.float32 or signal.shape[0] < 1
                    or not bool(torch.isfinite(signal).all())):
                raise LearnedDenoiseV4Error("model input must be finite float32 [B,1,500]")
            with torch.autocast(device_type=signal.device.type, enabled=False):
                features = self.blocks(self.input(signal))
                residual = self.residual_peak_bound * torch.tanh(self.residual_head(features))
                residual = self._project_residual(residual)
                clean = signal - residual
                logits = self.artifact_head(features).mean(dim=(1, 2))
            return clean, logits

else:
    EOGBandlimitedTCN = None


def build_model(config: dict[str, Any]):
    validate_config(config)
    if EOGBandlimitedTCN is None:
        raise LearnedDenoiseV4Error("optional PyTorch is required for v4 model construction")
    return EOGBandlimitedTCN(config)


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    config: dict[str, Any]
    artifact_threshold: float
    device: str
    metadata: dict[str, Any] = field(default_factory=dict)


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


def select_artifact_threshold(scores: Any, false_replacement_max: float = 0.01,
                              floor: float = 0.5) -> float:
    values = np.asarray(scores)
    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "fiu":
        raise LearnedDenoiseV4Error("scores must be a nonempty one-dimensional numeric array")
    values = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise LearnedDenoiseV4Error("scores must be finite probabilities in [0,1]")
    maximum = _finite_number(false_replacement_max, "false_replacement_max")
    threshold_floor = _finite_number(floor, "floor")
    if not 0 <= maximum <= 0.01 or not 0.5 <= threshold_floor <= 1:
        raise LearnedDenoiseV4Error("false_replacement_max must be in [0,.01] and floor in [.5,1]")
    if int(np.count_nonzero(values > threshold_floor)) / values.size <= maximum:
        return threshold_floor
    allowed = int(np.floor(values.size * maximum))
    order = np.sort(values)
    candidate = float(order[max(0, values.size - allowed - 1)])
    return float(max(threshold_floor, candidate))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_hex(value: Any, name: str) -> None:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise LearnedDenoiseV4Error(f"{name} must be a lowercase64hex digest")


def _validate_calibration(calibration: Any, config: dict[str, Any]) -> None:
    if not isinstance(calibration, dict) or not primitive_metadata(calibration):
        raise LearnedDenoiseV4Error("threshold_calibration must be finite primitive metadata")
    required = {"scope", "comparison", "method", "clean_window_count",
                "observed_clean_false_replacement_fraction", "max_clean_false_replacement_fraction",
                "empirical_only", "production_calibrated"}
    if not required <= set(calibration):
        raise LearnedDenoiseV4Error("threshold_calibration keys are not the frozen contract")
    if (calibration["scope"] != "validation_only" or calibration["comparison"] != ">"
            or calibration["method"] != "clean_identity_order_statistic"
            or calibration["empirical_only"] is not True or calibration["production_calibrated"] is not False):
        raise LearnedDenoiseV4Error("threshold must be validation-only clean-identity empirical order statistic")
    count = calibration["clean_window_count"]
    if type(count) is not int or count < 1:
        raise LearnedDenoiseV4Error("clean_window_count must be a positive integer")
    observed = _finite_number(calibration["observed_clean_false_replacement_fraction"],
                              "observed_clean_false_replacement_fraction")
    maximum = _finite_number(calibration["max_clean_false_replacement_fraction"],
                             "max_clean_false_replacement_fraction")
    contract_max = config["inference"]["clean_false_replacement_max"]
    if not 0 <= observed <= maximum <= contract_max:
        raise LearnedDenoiseV4Error("clean empirical false replacement fraction exceeds 1%")


def validate_checkpoint_metadata(checkpoint: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    if not isinstance(checkpoint, dict) or set(checkpoint) != CHECKPOINT_KEYS:
        raise LearnedDenoiseV4Error("v4 checkpoint keys do not exactly match the frozen schema")
    if (type(checkpoint["checkpoint_schema_version"]) is not int
            or checkpoint["checkpoint_schema_version"] != 1
            or checkpoint["model_version"] != MODEL_VERSION):
        raise LearnedDenoiseV4Error("unsupported v4 checkpoint version or model")
    if not primitive_metadata({key: value for key, value in checkpoint.items() if key != "model_state"}):
        raise LearnedDenoiseV4Error("checkpoint metadata must be finite JSON primitives")
    saved = checkpoint["config"]
    validate_config(saved)
    for key in set(DEFAULT_CONFIG) - {"runtime"}:
        if saved[key] != config[key]:
            raise LearnedDenoiseV4Error(f"checkpoint {key} does not match selected frozen config")
    threshold = _finite_number(checkpoint["artifact_threshold"], "artifact_threshold")
    if not config["inference"]["artifact_threshold_floor"] <= threshold <= 1:
        raise LearnedDenoiseV4Error("artifact_threshold must be in [0.5,1]")
    _validate_calibration(checkpoint["threshold_calibration"], config)
    _sha256_hex(checkpoint["dataset_sha256"], "dataset_sha256")
    if not isinstance(checkpoint["metadata"], dict):
        raise LearnedDenoiseV4Error("checkpoint metadata field must be a dictionary")
    return deepcopy({key: value for key, value in checkpoint.items() if key != "model_state"})


def _resolve_device(device: Any) -> str:
    if torch is None:
        raise LearnedDenoiseV4Error("optional PyTorch is required for v4 checkpoint loading")
    try:
        selected = torch.device(device)
    except (TypeError, RuntimeError, ValueError) as error:
        raise LearnedDenoiseV4Error(f"invalid device {device!r}") from error
    if selected.type == "cpu" and selected.index is None:
        return "cpu"
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise LearnedDenoiseV4Error("requested CUDA is unavailable; no silent CPU fallback")
    index = selected.index if selected.index is not None else torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise LearnedDenoiseV4Error("requested CUDA device index is unavailable")
    return f"cuda:{index}"


def _validate_state(model: Any, state: Any) -> None:
    if not isinstance(state, dict) or set(state) != set(model.state_dict()):
        raise LearnedDenoiseV4Error("model_state keys do not exactly match v4 model")
    reference = model.state_dict()
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or not value.is_floating_point() or tuple(value.shape) != tuple(reference[key].shape):
            raise LearnedDenoiseV4Error(f"model_state tensor shape/type mismatch: {key}")
        if not bool(torch.isfinite(value).all()):
            raise LearnedDenoiseV4Error(f"model_state contains non-finite values: {key}")


def load_model(path: Path | str, config: dict[str, Any] | Path | str | None = None,
               device: str = "cpu") -> ModelBundle:
    selected_device = _resolve_device(device)
    if not str(path).strip():
        raise LearnedDenoiseV4Error("checkpoint path must not be blank")
    checkpoint_path = Path(path).resolve()
    try:
        checkpoint = torch.load(checkpoint_path, map_location=selected_device, weights_only=True)
    except (OSError, pickle.UnpicklingError, TypeError, RuntimeError, EOFError) as error:
        raise LearnedDenoiseV4Error(f"cannot safely load v4 checkpoint: {error}") from error
    if not isinstance(checkpoint, dict):
        raise LearnedDenoiseV4Error("v4 checkpoint must be a plain dictionary")
    chosen = (deepcopy(checkpoint.get("config")) if config is None else
              load_config(config) if isinstance(config, (str, Path)) else deepcopy(config))
    try:
        validate_checkpoint_metadata(checkpoint, chosen)
        model = build_model(chosen)
        _validate_state(model, checkpoint["model_state"])
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(selected_device).eval().requires_grad_(False)
        if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
            raise LearnedDenoiseV4Error("loaded v4 model has non-finite state")
    except LearnedDenoiseV4Error:
        raise
    except (ValueError, TypeError, KeyError, RuntimeError, AttributeError) as error:
        raise LearnedDenoiseV4Error(f"invalid v4 checkpoint: {error}") from error
    metadata = validate_checkpoint_metadata(checkpoint, chosen)
    return ModelBundle(model, deepcopy(chosen), float(checkpoint["artifact_threshold"]), selected_device, metadata)


def _prepare_signal(signal: Any) -> np.ndarray:
    source = np.asarray(signal)
    if source.ndim == 1:
        source = source[:, None]
    if source.ndim != 2 or source.shape[0] < 1 or source.shape[1] < 1 or source.dtype.kind not in "iuf":
        raise LearnedDenoiseV4Error("signal must be real numeric [T,C] or [T]")
    if source.dtype.itemsize > 8:
        raise LearnedDenoiseV4Error("signal precision exceeds float64 restoration")
    if source.dtype.kind in "iu" and np.any(np.abs(source.astype(np.longdouble)) > 2 ** 53):
        raise LearnedDenoiseV4Error("integer raw exceeds exact float64 representation")
    return source.astype(np.float64, copy=True)


def _prepare_valid(valid_mask: Any, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(valid_mask)
    if mask.ndim == 1 and shape[1] == 1:
        mask = mask[:, None]
    if mask.shape != shape or mask.dtype.kind != "b":
        raise LearnedDenoiseV4Error("valid_mask must be boolean with exactly [T,C] shape")
    return mask.copy()


def _center_rms(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Stable float64 mean/AC RMS without squaring acquisition-scale values."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        center = float(np.mean(values, dtype=np.float64))
        if not np.isfinite(center):
            magnitude = float(np.max(np.abs(values)))
            center = magnitude * float(np.mean(values / magnitude, dtype=np.float64))
        centered = values - center
        peak = float(np.max(np.abs(centered)))
        scale = peak * float(np.sqrt(np.mean((centered / peak) ** 2, dtype=np.float64))) if peak > 0 else 0.0
    if not np.isfinite(center) or not np.isfinite(scale) or not np.all(np.isfinite(centered)):
        raise LearnedDenoiseV4Error("unsafe float64 mean/AC RMS")
    return centered, center, scale


def _normalize(values: np.ndarray, min_ac_rms: float) -> tuple[np.ndarray, float, float]:
    centered, center, scale = _center_rms(values)
    if scale <= min_ac_rms:
        raise LearnedDenoiseV4Error("constant or unsafe AC RMS window")
    normalized = (centered / scale).astype(np.float32)
    if not np.all(np.isfinite(normalized)):
        raise LearnedDenoiseV4Error("normalized window is non-finite")
    return normalized, center, scale


def _guard_raw(values: np.ndarray, sample_rate: float, config: dict[str, Any]) -> tuple[bool, str]:
    inference = config["inference"]
    try:
        centered, center, scale = _center_rms(values)
    except LearnedDenoiseV4Error:
        return False, "constant_windows"
    if scale <= inference["min_ac_rms"]:
        return False, "constant_windows"
    dc_ratio = abs(center) / scale
    if not np.isfinite(dc_ratio) or dc_ratio > inference["max_abs_dc_to_ac_rms"]:
        return False, "dc_guard_windows"
    # AC spectral-power ratio, with Parseval one-sided weights (Nyquist singleton).
    spectrum = np.fft.rfft(centered / scale)
    frequency = np.fft.rfftfreq(values.size, d=1.0 / sample_rate)
    power = np.abs(spectrum) ** 2
    power[1:-1] *= 2.0
    total = float(np.sum(power[frequency > 0], dtype=np.float64))
    high = float(np.sum(power[frequency > 45.0], dtype=np.float64))
    high_fraction = high / total if total > 0 else float("nan")
    if not np.isfinite(high_fraction) or high_fraction > inference["max_high_frequency_fraction_above_45hz"]:
        return False, "high_frequency_guard_windows"
    return True, "safe_windows"


def _project_numpy(residual: np.ndarray, sample_rate: float, band_hz: tuple[float, float], peak_bound: float) -> np.ndarray:
    if residual.shape != (500,) or not np.all(np.isfinite(residual)):
        raise LearnedDenoiseV4Error("residual must be finite one-window float64")
    frequency = np.fft.rfftfreq(residual.size, d=1.0 / sample_rate)
    spectrum = np.fft.rfft(residual.astype(np.float64, copy=False))
    spectrum[(frequency < band_hz[0]) | (frequency > band_hz[1])] = 0.0
    projected = np.fft.irfft(spectrum, n=residual.size).astype(np.float64, copy=False)
    projected -= float(np.mean(projected, dtype=np.float64))
    peak = float(np.max(np.abs(projected)))
    if not np.isfinite(peak):
        raise LearnedDenoiseV4Error("projected residual is non-finite")
    if peak > peak_bound:
        projected *= peak_bound / peak
    return projected


def _protected_difference(raw: np.ndarray, candidate: np.ndarray, sample_rate: float,
                          band: list[float]) -> float:
    """Preregistered L2 FFT delta / max(L2 FFT rawAC, sqrt(500)*ACRMS*1e-12).

    Both numerator and denominator are divided by raw AC RMS before FFT for
    numerical safety; the ratio and its protected-band floor are unchanged.
    """
    raw_ac, _, raw_rms = _center_rms(raw)
    if raw_rms <= 0:
        return 0.0 if np.array_equal(candidate, raw) else float("inf")
    frequency = np.fft.rfftfreq(raw.size, d=1.0 / sample_rate)
    selected = (frequency >= band[0]) & (frequency <= band[1])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        numerator = float(np.linalg.norm(np.fft.rfft((candidate - raw) / raw_rms)[selected]))
        raw_l2 = float(np.linalg.norm(np.fft.rfft(raw_ac / raw_rms)[selected]))
        denominator = max(raw_l2, float(np.sqrt(raw.size)) * 1e-12)
        ratio = numerator / denominator
    return float(ratio)


def _forward_windows(model: Any, normalized: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    """One batch transfer/forward/CPU transfer; validate row values independently later."""
    tensor = torch.from_numpy(normalized[:, None]).to(device)
    # Length-500 FFT and convolutions remain float32, even under an outer AMP context.
    with torch.no_grad(), torch.autocast(device_type=tensor.device.type, enabled=False):
        output = model(tensor)
    batch = normalized.shape[0]
    if (not isinstance(output, tuple) or len(output) != 2
            or not isinstance(output[0], torch.Tensor) or not isinstance(output[1], torch.Tensor)
            or tuple(output[0].shape) != (batch, 1, 500) or tuple(output[1].shape) != (batch,)
            or output[0].dtype != torch.float32 or not output[1].is_floating_point()
            or output[0].device != tensor.device or output[1].device != tensor.device):
        raise LearnedDenoiseV4Error("model returned invalid clean/logit batch shapes/types/devices")
    return (output[0].detach().cpu().numpy()[:, 0].astype(np.float64),
            output[1].detach().cpu().to(torch.float64).numpy())


def apply_model_to_array(bundle: ModelBundle, signal: Any, valid_mask: Any | None = None) -> DenoiseResult:
    if not isinstance(bundle, ModelBundle):
        raise LearnedDenoiseV4Error("bundle must be a v4 ModelBundle")
    validate_config(bundle.config)
    if torch is None:
        raise LearnedDenoiseV4Error("optional PyTorch is required for v4 inference")
    selected_device = _resolve_device(bundle.device)
    threshold = _finite_number(bundle.artifact_threshold, "bundle artifact_threshold")
    if not bundle.config["inference"]["artifact_threshold_floor"] <= threshold <= 1:
        raise LearnedDenoiseV4Error("bundle artifact_threshold must be in [.5,1]")
    if not primitive_metadata(bundle.metadata):
        raise LearnedDenoiseV4Error("bundle metadata must be finite primitives")
    if not isinstance(bundle.model, nn.Module):
        raise LearnedDenoiseV4Error("bundle model must be a PyTorch module")
    state = list(bundle.model.parameters()) + list(bundle.model.buffers())
    if any(value.device != torch.device(selected_device) for value in state):
        raise LearnedDenoiseV4Error("model device does not match explicit bundle device")
    x = _prepare_signal(signal)
    valid = np.isfinite(x)
    if valid_mask is not None:
        valid &= _prepare_valid(valid_mask, x.shape)
    original = x.copy()
    samples, channels = x.shape
    denoised = original.copy()
    candidate = original.copy()
    probability = np.full((samples, channels), np.nan, dtype=np.float64)
    confidence = np.zeros((samples, channels), dtype=np.float64)
    replacement = np.zeros((samples, channels), dtype=bool)
    rejected = np.ones((samples, channels), dtype=bool)
    counts = {key: 0 for key in ("safe_windows", "artifact_windows", "clean_windows", "invalid_windows",
              "constant_windows", "dc_guard_windows", "high_frequency_guard_windows", "prediction_rejected_windows",
              "change_guard_windows", "protected_band_rejected_windows", "replaced_windows")}
    band = tuple(float(item) for item in bundle.config["model"]["residual_band_hz"])
    peak_bound = float(bundle.config["model"]["residual_peak_bound"])
    window = int(bundle.config["window_samples"])
    forward_counts = {"model_forward_calls": 0, "batch_forward_failures": 0, "single_window_retry_calls": 0}
    batch_size = 128

    def restore_one(record, predicted, logit):
        channel, start, normalized, center, scale = record
        end = start + window
        raw_window = original[start:end, channel]
        try:
            if not np.isfinite(logit) or not np.all(np.isfinite(predicted)):
                raise LearnedDenoiseV4Error("model returned non-finite prediction or artifact logit")
            residual = _project_numpy(normalized.astype(np.float64) - predicted,
                                      float(bundle.config["sample_rate_hz"]), band, peak_bound)
            with np.errstate(over="ignore", invalid="ignore"):
                restored = raw_window - residual * scale
            # A zero residual keeps exact bits, including signed zero. Otherwise
            # only one DC-roundoff correction, never samplewise clipping, is used.
            if np.any(residual != 0) and np.all(np.isfinite(restored)):
                correction = center - float(np.mean(restored, dtype=np.float64))
                if correction != 0:
                    restored += correction
            if not np.all(np.isfinite(restored)):
                raise LearnedDenoiseV4Error("restored model output is non-finite")
            change = restored - raw_window
            _, _, change_scale = _center_rms(change)
            change_ratio = float(change_scale / scale)
            protected_error = _protected_difference(raw_window, restored, float(bundle.config["sample_rate_hz"]),
                                                    bundle.config["inference"]["protected_band_hz"])
            if not np.isfinite(change_ratio) or change_ratio > bundle.config["inference"]["max_change_ac_rms_ratio"]:
                raise LearnedDenoiseV4Error("change_ac_rms_ratio_guard")
            if not np.isfinite(protected_error) or protected_error > bundle.config["inference"]["protected_band_relative_tolerance"]:
                raise LearnedDenoiseV4Error("protected_band_guard")
            # Stable scalar sigmoid; no non-finite/saturated-invalid logits accepted.
            if logit >= 0:
                artifact_score = float(1.0 / (1.0 + np.exp(-logit)))
            else:
                exp_logit = float(np.exp(logit))
                artifact_score = exp_logit / (1.0 + exp_logit)
            if not np.isfinite(artifact_score) or not 0 <= artifact_score <= 1:
                raise LearnedDenoiseV4Error("artifact probability is non-finite")
        except Exception as error:
            reason = str(error)
            if reason == "change_ac_rms_ratio_guard":
                counts["change_guard_windows"] += 1
            elif reason == "protected_band_guard":
                counts["protected_band_rejected_windows"] += 1
            else:
                counts["prediction_rejected_windows"] += 1
            return
        candidate[start:end, channel] = restored
        probability[start:end, channel] = artifact_score
        confidence[start:end, channel] = 1.0 - artifact_score
        rejected[start:end, channel] = False
        if artifact_score > threshold:
            replacement[start:end, channel] = True
            denoised[start:end, channel] = restored
            counts["artifact_windows"] += 1
            counts["replaced_windows"] += 1
        else:
            counts["clean_windows"] += 1
        counts["safe_windows"] += 1

    def infer_batch(records):
        forward_counts["model_forward_calls"] += 1
        try:
            predictions, logits = _forward_windows(bundle.model, np.stack([record[2] for record in records]), selected_device)
        except Exception:
            if len(records) == 1:
                counts["prediction_rejected_windows"] += 1
            else:
                # Bad structure/global exception: retry singly to prevent one
                # failing row/channel from disabling all other valid windows.
                forward_counts["batch_forward_failures"] += 1
                for record in records:
                    forward_counts["single_window_retry_calls"] += 1
                    infer_batch([record])
            return
        for record, predicted, logit in zip(records, predictions, logits):
            restore_one(record, predicted, float(logit))

    pending = []
    for channel in range(channels):
        for start in range(0, samples - window + 1, window):
            end = start + window
            raw_window = original[start:end, channel]
            if not np.all(valid[start:end, channel]):
                counts["invalid_windows"] += 1
                continue
            safe, reason = _guard_raw(raw_window, bundle.config["sample_rate_hz"], bundle.config)
            if not safe:
                counts[reason] += 1
                continue
            try:
                normalized, center, scale = _normalize(raw_window, bundle.config["inference"]["min_ac_rms"])
            except Exception:
                counts["constant_windows"] += 1
                continue
            pending.append((channel, start, normalized, center, scale))
            if len(pending) == batch_size:
                infer_batch(pending)
                pending = []
    if pending:
        infer_batch(pending)
    changed = original.view(np.uint64) != denoised.view(np.uint64)
    tail = samples % window
    diagnostics = {
        "model_version": MODEL_VERSION, "window_samples": window, "hop_samples": window,
        "full_windows_per_channel": samples // window, "tail_samples_per_channel": tail,
        "artifact_threshold": threshold, "threshold_comparison": ">",
        "threshold_scope": "validation_only_clean_identity_order_statistic_not_production_calibrated",
        "confidence_interpretation": bundle.config["inference"]["confidence_interpretation"],
        "model_device": selected_device, "normalization": bundle.config["normalization"],
        "restoration_dtype": "float64", "residual_band_hz": list(band), "residual_peak_bound": peak_bound,
        "changed_samples": int(np.count_nonzero(changed)), "replaced_samples": int(np.count_nonzero(replacement)),
        "replacement_fraction": float(np.mean(replacement)), "rejected_samples": int(np.count_nonzero(rejected)),
        "automatic_processing_enabled": False, "diagnostic_only": True, "release_gate_passed": False,
        "production_validated": False, "raw_is_immutable": True, "original_quality_unchanged": True,
        "missing_data_repaired": False, "runtime": deepcopy(bundle.config["runtime"]),
        "high_frequency_fraction_semantics": "AC_Parseval_power_above_45Hz_divided_by_all_positive_frequency_power",
        "projection_peak_limiting": "uniform_window_scaling_not_sample_clipping",
        "protected_band_metric": "L2_FFT_candidate_minus_raw_over_max_L2_FFT_rawAC_sqrt500_times_rawACRMS_times_1e-12",
        "inference_batch_size": batch_size, "fft_dtype": "float32_torch_float64_numpy",
        **forward_counts, **counts,
    }
    return DenoiseResult(denoised, candidate, probability, confidence, replacement, valid, rejected, diagnostics)


__all__ = ["MODEL_VERSION", "DEFAULT_CONFIG", "CHECKPOINT_KEYS", "LearnedDenoiseV4Error",
           "EOGBandlimitedTCN", "ModelBundle", "DenoiseResult", "primitive_metadata",
           "validate_config", "load_config", "build_model", "select_artifact_threshold",
           "validate_checkpoint_metadata", "load_model", "apply_model_to_array"]
