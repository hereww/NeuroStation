"""Frozen v8 learned core + deterministic, label-free v9 trust-region adapter.

This is an INFERENCE variant, not a new training run. One scalar per window
limits mean squared normalized residual to .00045 (below the unchanged .0005
identity engineering gate). Scaling is global across channels/time: it does not
mix frequencies or use labels/targets. Protection is for each 256-sample window,
not a claim about the FFT of concatenated windows or causal streaming.
"""
from __future__ import annotations
import torch
from torch import nn

MODEL_VERSION = "learned_denoise_v9_ssvep"
CHANGE_BUDGET = 0.00045

class SSVEPDenoiserV9(nn.Module):
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core
        self.sample_rate_hz = core.sample_rate_hz
        self.core.requires_grad_(False)
        self.eval()

    def forward_with_aux(self, x):
        pred = self.core(x)
        delta = pred - x
        # Float64 accumulation; a small margin handles float32 output rounding.
        energy = delta.to(torch.float64).square().mean(dim=(1, 2), keepdim=True)
        gain = torch.sqrt((CHANGE_BUDGET * (1.0 - 1e-5)) / energy.clamp_min(1e-30)).clamp(max=1.0)
        limited = x + delta * gain.to(torch.float32)
        if not bool(torch.isfinite(limited).all()):
            raise ValueError("nonfinite v9 prediction")
        return limited, {"gain": gain, "uncapped_normalized_change": energy,
                         "normalized_change": (limited-x).to(torch.float64).square().mean((1,2), keepdim=True)}

    def forward(self, x):
        return self.forward_with_aux(x)[0]
