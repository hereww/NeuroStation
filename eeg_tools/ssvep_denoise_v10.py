"""Pure PyTorch v10 spectral shrinkage for normalized native 250 Hz windows.

No disk I/O, notch filtering, normalization, labels, or target conditioning.
Each channel uses the same 4->24->24->1 spectral Conv1d encoder. Sigmoid
attenuation is the amount removed, so gain=1-attenuation. DC and discrete
8..40 Hz bins have exactly unit gain. The real FFT uses 512 samples.
The local contrast feature subtracts a five-bin, replicate-padded mean.

Protection concerns individual FFT windows, not an overlap-add record or
physiological truth. Aux error is measured after the actual irFFT/rFFT round
trip. A negligible protected reference has no meaningful relative error;
its ratio is zero and protected_energy_defined is false.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

MODEL_VERSION = "learned_denoise_v10_personal"
MODEL_TYPE = "fixed_line_notch_then_learned_protected_spectral_shrinkage"
SAMPLE_RATE_HZ = 250
CHANNEL_COUNT = 8
WINDOW_SAMPLES = 512

_DEFAULT_CONFIG = {
    "schema_version": 1,
    "status": "formal",
    "model_version": MODEL_VERSION,
    "model_type": MODEL_TYPE,
    "sample_rate_hz": SAMPLE_RATE_HZ,
    "channel_count": CHANNEL_COUNT,
    "window_samples": WINDOW_SAMPLES,
    "model": {
        "hidden_channels": 24,
        "feature_channels": 4,
        "kernels": [7, 5, 1],
        "attenuation": "sigmoid per source channel and frequency bin",
        "initial_gate_bias": -4.0,
        "protected_band_hz": [8.0, 40.0],
        "dc_preserved": True,
        "allow_amplification": False,
    },
}


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return type(actual) is bool and actual == expected
    if isinstance(expected, (int, float)):
        return (type(actual) in (int, float) and math.isfinite(actual)
                and actual == expected)
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_matches(a, b) for a, b in zip(actual, expected)))
    return type(actual) is type(expected) and actual == expected


def validate_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate the fixed architecture in memory and return an independent copy.

    Full experiment configs are accepted; data/training/preprocessing metadata
    are retained but are the caller's responsibility. No config file is read.
    """
    if config is None:
        return copy.deepcopy(_DEFAULT_CONFIG)
    if not isinstance(config, Mapping):
        raise ValueError("v10 config must be a mapping or None")
    for name, expected in _DEFAULT_CONFIG.items():
        if name == "model":
            continue
        if not _matches(config.get(name), expected):
            raise ValueError(f"v10 config mismatch: {name}")
    settings = config.get("model")
    if not isinstance(settings, Mapping):
        raise ValueError("v10 config requires model settings")
    for name, expected in _DEFAULT_CONFIG["model"].items():
        if not _matches(settings.get(name), expected):
            raise ValueError(f"v10 model config mismatch: {name}")
    return copy.deepcopy(dict(config))


class SpectralDenoiserV10(nn.Module):
    """Accept finite float32 [B,8,512] signal only; return same-shaped signal."""

    sample_rate_hz = SAMPLE_RATE_HZ

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.config = validate_config(config)
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 24, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(24, 24, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(24, 1, kernel_size=1),
        )
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.constant_(self.encoder[-1].bias, -4.0)
        frequencies = torch.fft.rfftfreq(WINDOW_SAMPLES, d=1.0 / SAMPLE_RATE_HZ)
        protected = ((frequencies >= 8.0) & (frequencies <= 40.0))
        protected[0] = True
        # Derived constants are not trainable/checkpoint-overridable state.
        self.register_buffer("protected_mask", protected, persistent=False)
        self.register_buffer("normalized_frequency", frequencies / (SAMPLE_RATE_HZ / 2),
                             persistent=False)

    def _check_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise ValueError("v10 input must be a torch.Tensor")
        if x.layout != torch.strided:
            raise ValueError("v10 input must be a dense strided tensor")
        if x.ndim != 3 or x.shape[0] < 1 or tuple(x.shape[1:]) != (8, 512):
            raise ValueError("v10 input must have shape [B,8,512], B>=1")
        if x.dtype != torch.float32:
            raise ValueError("v10 input must have dtype float32")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("v10 input must be finite")
        weight = self.encoder[0].weight
        if x.device != weight.device or weight.dtype != torch.float32:
            raise ValueError("v10 input and model must share device and float32 dtype")

    def _features(self, spectrum: torch.Tensor) -> torch.Tensor:
        b, c, bins = spectrum.shape
        local = torch.log1p(spectrum.abs() / math.sqrt(WINDOW_SAMPLES))
        cross_channel = local.mean(dim=1, keepdim=True).expand_as(local)
        smooth = F.avg_pool1d(F.pad(local.reshape(b*c, 1, bins), (2, 2),
                                   mode="replicate"), kernel_size=5, stride=1)
        contrast = local - smooth.reshape(b, c, bins)
        frequency = self.normalized_frequency.reshape(1, 1, bins).expand_as(local)
        return torch.stack((local, cross_channel, contrast, frequency), dim=2).reshape(
            b*c, 4, bins)

    def forward_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._check_input(x)
        spectrum = torch.fft.rfft(x, n=WINDOW_SAMPLES, dim=-1)
        if not bool(torch.isfinite(spectrum).all()):
            raise ValueError("v10 input FFT overflowed")
        b, c, bins = spectrum.shape
        logits = self.encoder(self._features(spectrum)).reshape(b, c, bins)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("v10 encoder produced nonfinite values")
        attenuation = torch.sigmoid(logits).masked_fill(self.protected_mask, 0.0)
        gain = 1.0 - attenuation
        pred = torch.fft.irfft(spectrum * gain, n=WINDOW_SAMPLES, dim=-1)
        if not bool(torch.isfinite(pred).all()):
            raise ValueError("v10 output must be finite")
        reconstructed = torch.fft.rfft(pred, n=WINDOW_SAMPLES, dim=-1)
        protected_reference = spectrum[..., self.protected_mask]
        protected_error = reconstructed[..., self.protected_mask] - protected_reference
        # Divide before squaring to keep intermediate diagnostic energies bounded.
        scale = spectrum.abs().amax(dim=(1, 2)).clamp_min(torch.finfo(x.dtype).tiny)
        reference_energy = (protected_reference.abs() / scale[:, None, None]).square().sum((1, 2))
        error_energy = (protected_error.abs() / scale[:, None, None]).square().sum((1, 2))
        total_energy = (spectrum.abs() / scale[:, None, None]).square().sum((1, 2))
        floor = (total_energy * 1e-12).clamp_min(torch.finfo(x.dtype).tiny)
        defined = reference_energy > floor
        ratio = torch.where(defined, error_energy / reference_energy.clamp_min(floor),
                            torch.zeros_like(error_energy))
        return pred, {
            "attenuation": attenuation,
            "gain": gain,
            "protected_complex_energy_ratio": ratio,
            "protected_energy_defined": defined,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(x)[0]


def build_model(config: Mapping[str, Any] | None = None) -> SpectralDenoiserV10:
    """Build v10 using in-memory defaults or an already loaded experiment config."""
    return SpectralDenoiserV10(config)
