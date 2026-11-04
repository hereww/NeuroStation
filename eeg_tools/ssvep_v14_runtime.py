"""V14 offline SSVEP denoiser runtime with explicit normalization scopes.

This module is intentionally independent of v1-v13.  Metadata/check-only paths
use only the standard library.  Scientific dependencies are imported lazily by
notch, model, and inference functions.

Normalization contract
----------------------
* ``window_512``: each public 512-sample window computes its own noisy
  float64 channel mean and RMS.  The mean/RMS are used only for that window.
* ``record_4096``: a continuous record computes one noisy float64 channel mean
  and RMS over the entire record after the fixed notch.  Every 4096-sample
  subwindow uses those same record statistics.  Restoration is explicitly
  ``record_mean + prediction * record_rms`` before whole-window OLA; window-
  local means are never reinserted.
* The learned model receives only normalized ``signal`` float32.  The runtime
  overlap-adds complete restored windows on the record path.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/learned_denoiser_v14_personal.json"
SOURCES = (
    "eeg_tools/ssvep_v14_runtime.py",
    "scripts/train_ssvep_denoiser_v14.py",
    "tests/test_ssvep_v14.py",
)
LOCK = "analysis_outputs/ssvep_v14_training_lock_20261014.json"
MODEL = "models/denoiser_v14_personal.pt"
MANIFEST = "models/denoiser_v14_personal.manifest.json"
REPORT = "analysis_outputs/ssvep_v14_training_20261014.json"
PLAN = "analysis_outputs/ssvep_v14_experiment_plan_20261014.json"
NORMALIZATION = "v14_explicit_scope_window512_per_window_record4096_full_record_stats"
CONDITIONS = ("identity", "drift", "high_burst", "gaussian", "combined")
RMS_FLOOR = 1e-8


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError("v14: " + message)


def sha(filename: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(filename).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def safe_path(root: str | Path, name: str, *, new: bool = False) -> Path:
    """Resolve a normalized relative workspace path without following links."""
    require(isinstance(name, str) and "\\" not in name and ":" not in name,
            "unsafe relative path")
    parts = PurePosixPath(name).parts
    require(parts and not name.startswith("/") and ".." not in parts
            and "/".join(parts) == name, "unsafe relative path")
    base = Path(root).absolute()
    for item in (base, *base.parents):
        if item.exists():
            stat = item.lstat()
            require(not item.is_symlink() and not (getattr(stat, "st_file_attributes", 0) & 0x400),
                    "linked/reparse workspace parent")
    current = base
    for part in parts:
        current = current / part
        if current.exists() or current.is_symlink():
            stat = current.lstat()
            require(not current.is_symlink() and not (getattr(stat, "st_file_attributes", 0) & 0x400),
                    "linked/reparse path: " + name)
            if current.is_file():
                require(stat.st_nlink == 1, "hard-linked file: " + name)
    require(current.resolve().is_relative_to(base.resolve()), "path escapes workspace")
    if new:
        require(not current.exists(), "refusing existing output: " + name)
    return current


def load_config(root: str | Path = ROOT) -> dict:
    cfg = json.loads(safe_path(root, CONFIG).read_text(encoding="utf-8"))
    require(cfg.get("schema_version") == 1 and cfg.get("status") == "formal"
            and cfg.get("model_version") == "learned_denoise_v14_personal",
            "invalid formal v14 config identity")
    require((cfg["sample_rate_hz"], cfg["channel_count"], cfg["window_samples"],
             cfg["hop_samples"], cfg["supported_window_samples"])
            == (250, 8, 4096, 1024, [512, 4096]), "native context contract changed")
    model = cfg["model"]
    require(model["channels"] == [4, 24, 24, 1] and model["kernels"] == [7, 5, 1]
            and model["protected_band_hz"] == [8, 40] and model["dc_preserved"] is True
            and model["gain_bounds"] == [0, 1] and model["forward_inputs"] == ["signal"],
            "architecture contract changed")
    normalization = cfg["normalization"]
    require(normalization["normalization_scope"] == {
                "window_512": "per_window_noisy_float64_channel_mean_rms",
                "record_4096": "full_record_noisy_float64_channel_mean_rms_reused_for_all_subwindows",
                "process_windows_scope_required": True,
            } and normalization["statistics_dtype"] == "float64"
            and normalization["forward_dtype"] == "float32"
            and normalization["restore_dtype"] == "float64"
            and normalization["record_restore"] == "global_record_mean_plus_prediction_times_global_record_rms_then_whole_window_OLA; identity_prediction_copies_original_window_exactly"
            and normalization["rms_floor"] == RMS_FLOOR,
            "normalization contract changed")
    require(cfg["preprocessing"] == {
                "notch_hz": 50, "notch_q": 30, "phase": "offline_zero_phase",
                "resample": False, "unit_conversion": False,
            }, "fixed notch contract changed")
    data = cfg["data"]
    require(data["fit_subjects"] == [1, 2, 3, 4]
            and data["selection_subjects"] == [5, 6]
            and data["evaluation_subjects"] == [7, 8]
            and data["personal_sessions_used"] is False
            and data["personal_data_used_for_fit_or_selection"] is False
            and data["public_evaluation_used_for_selection"] is False
            and data["reference_directory"] == "datasets/processed/ssvep_reference_v5_20261007/Wearable",
            "data role contract changed")
    synth = cfg["synthetic"]
    require((synth["fit_count"], synth["selection_count"], synth["record_samples"],
             synth["continuous"], synth["normalization_scope"])
            == (96, 32, 4096, True, "full_record"), "synthetic continuous contract changed")
    train = cfg["training"]
    require((train["epochs"], train["device"], train["require_cuda"], train["deterministic"],
             train["amp"], train["allow_tf32"]) == (16, "cuda:0", True, True, False, False),
            "GPU-only deterministic training contract changed")
    require(cfg["selection"]["identity_normalized_change_max"] == 0.002
            and cfg["selection"]["require_both_domain_identity_gates"] is True,
            "selection gate changed")
    require(cfg["runtime"] == {
                "enabled": False, "production_validated": False,
                "scope": "explicit_offline_research_only",
            }, "runtime must remain disabled and unvalidated")
    artifacts = cfg["artifacts"]
    require(artifacts == {"lock": LOCK, "checkpoint": MODEL, "manifest": MANIFEST,
                          "report": REPORT}, "20261014 artifact paths changed")
    require(cfg["research_plan_path"] == PLAN, "20261014 research plan path changed")
    return cfg


def validate_plan(root: str | Path, cfg: dict) -> dict:
    """Validate the embedded preregistration without reading prior v1-v13 plans."""
    plan = cfg.get("research_plan")
    require(isinstance(plan, dict) and plan.get("schema_version") == 1
            and plan.get("status") == "formal"
            and plan.get("runtime_enabled") is False
            and plan.get("production_validated") is False
            and plan.get("source_lock_required") is True
            and plan.get("no_personal_data_in_fit_or_selection") is True,
            "invalid embedded formal research plan")
    design = plan["training_design"]
    require(design["public_fit_subjects"] == cfg["data"]["fit_subjects"]
            and design["public_selection_subjects"] == cfg["data"]["selection_subjects"]
            and design["public_evaluation_subjects"] == cfg["data"]["evaluation_subjects"]
            and design["personal_sessions_for_fit_or_selection"] is False
            and design["long_synthetic_normalization"] == "one full-record noisy float64 channel mean/RMS per 4096 record; restore global_record_mean + prediction*global_record_rms and OLA complete restored windows"
            and design["short_public_normalization"] == "per-window noisy float64 channel mean/RMS for 512 windows"
            and design["labels_enter_model_forward"] is False,
            "embedded plan/config mismatch")
    checks = plan["fixed_research_checks"]
    require(checks["public_identity_normalized_change_max"] == cfg["selection"]["identity_normalized_change_max"]
            and checks["long_synthetic_identity_normalized_change_max"] == cfg["selection"]["identity_normalized_change_max"]
            and checks["research_checks_authorize_default_deployment"] is False,
            "embedded plan selection mismatch")
    return plan


def plan_sha(cfg: dict) -> str:
    return canonical_sha(cfg["research_plan"])


def bindings(cfg: dict) -> list[str]:
    files = [CONFIG, *SOURCES]
    directory = cfg["data"]["reference_directory"]
    for subject in cfg["data"]["fit_subjects"] + cfg["data"]["selection_subjects"]:
        name = f"{directory}/subject_{subject:03d}.npz"
        files.extend((name, name + ".manifest.json"))
    return files


def _validate_lock(root: str | Path, cfg: dict, lock: dict) -> None:
    require(lock.get("schema_version") == 1 and lock.get("status") == "formal"
            and lock.get("model_version") == cfg["model_version"]
            and lock.get("config_sha256") == canonical_sha(cfg)
            and lock.get("research_plan_path") == PLAN
            and lock.get("research_plan_sha256") == plan_sha(cfg)
            and lock.get("normalization_contract") == NORMALIZATION
            and lock.get("runtime_enabled") is False
            and lock.get("production_validated") is False,
            "configuration/plan lock mismatch")
    require(lock.get("roles") == {
                "fit": [1, 2, 3, 4], "select": [5, 6],
                "personal": [], "evaluation": [],
            }, "training lock role mismatch")
    require(lock.get("synthetic") == cfg["synthetic"], "training lock synthetic mismatch")
    require(set(lock.get("files", {})) == set(bindings(cfg)), "unexpected/missing lock bindings")
    for name, digest in lock["files"].items():
        require(isinstance(digest, str) and len(digest) == 64
                and sha(safe_path(root, name)) == digest, "locked file changed: " + name)


def preflight(root: str | Path = ROOT) -> dict:
    cfg = load_config(root)
    validate_plan(root, cfg)
    lock_path = safe_path(root, LOCK)
    require(lock_path.is_file(), "training lock is required")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    _validate_lock(root, cfg, lock)
    return {
        "status": "ready",
        "config_sha256": canonical_sha(cfg),
        "training_lock_sha256": sha(lock_path),
        "research_plan_sha256": plan_sha(cfg),
        "checked_files": len(lock["files"]),
        "waveforms_loaded": False,
        "torch_imported": False,
        "cuda_initialized": False,
        "runtime_enabled": False,
        "production_validated": False,
    }


def _windows(value, name="signal", *, window_samples=None):
    import numpy as np
    arr = np.asarray(value)
    require(arr.dtype.kind == "f" and arr.ndim == 3 and arr.shape[0] > 0
            and arr.shape[1] == 8 and arr.shape[2] in (512, 4096)
            and (window_samples is None or arr.shape[2] == window_samples)
            and np.isfinite(arr).all(),
            f"{name} must be finite floating [B,8,512|4096]")
    return arr.astype(np.float64, copy=False)


def _record(value, name="record"):
    import numpy as np
    arr = np.asarray(value)
    require(arr.dtype.kind == "f" and arr.ndim == 2 and arr.shape[1] == 8
            and arr.shape[0] >= 4096 and np.isfinite(arr).all(),
            f"{name} must be finite floating [T>=4096,8]")
    return arr.astype(np.float64, copy=False)


def _validate_stats(means, rms, batch: int | None = None):
    import numpy as np
    mean = np.asarray(means)
    scale = np.asarray(rms)
    expected = (8, 1) if batch is None else (batch, 8, 1)
    require(mean.dtype == scale.dtype == np.float64
            and mean.shape == scale.shape == expected
            and np.isfinite(mean).all() and np.isfinite(scale).all()
            and np.all(scale > RMS_FLOOR), "invalid float64 noisy statistics")
    return mean, scale


def record_statistics(record):
    """Return one full-record ``(float64 mean, float64 RMS)`` pair per channel."""
    import numpy as np
    x = _record(record)
    mean = x.mean(axis=0, keepdims=True, dtype=np.float64).T
    centered = x - mean.T
    rms = np.sqrt(np.mean(centered * centered, axis=0, keepdims=True, dtype=np.float64)).T
    _validate_stats(mean, rms)
    return mean, rms


def normalize_noisy_window(noisy):
    """Normalize independent public 512 windows with per-window noisy statistics."""
    import numpy as np
    x = _windows(noisy, "noisy window", window_samples=512)
    means = x.mean(axis=-1, keepdims=True, dtype=np.float64)
    rms = np.sqrt(np.mean((x - means) ** 2, axis=-1, keepdims=True, dtype=np.float64))
    _validate_stats(means, rms, len(x))
    normalized = np.asarray((x - means) / rms, dtype=np.float32)
    require(np.isfinite(normalized).all(), "window normalization overflow")
    return normalized, means, rms


def normalize_noisy_record(record, starts=None, *, window_samples=4096):
    """Normalize record windows with one full-record statistic reused by all windows.

    Returns ``(normalized_windows, record_mean, record_rms, starts)``.  The
    record mean/RMS have shape ``[8,1]`` and are never recomputed per window.
    """
    import numpy as np
    x = _record(record)
    require(window_samples == 4096, "record normalization uses 4096-sample windows")
    chosen = window_starts(len(x)) if starts is None else list(starts)
    require(chosen and all(type(v) is int for v in chosen), "invalid record window starts")
    require(all(0 <= s <= len(x) - window_samples for s in chosen), "record window start out of range")
    mean, rms = record_statistics(x)
    windows = np.stack([x[s:s + window_samples].T for s in chosen])
    normalized = np.asarray((windows - mean[None, :, :]) / rms[None, :, :], dtype=np.float32)
    require(np.isfinite(normalized).all(), "record normalization overflow")
    return normalized, mean, rms, chosen


def _stats_for_windows(means, rms, batch: int):
    import numpy as np
    mean, scale = np.asarray(means), np.asarray(rms)
    if mean.shape == (8, 1):
        mean = np.broadcast_to(mean, (batch, 8, 1))
    if scale.shape == (8, 1):
        scale = np.broadcast_to(scale, (batch, 8, 1))
    _validate_stats(mean, scale, batch)
    return mean, scale


def normalize_with_record_stats(noisy, means, rms):
    """Normalize a [B,8,4096] batch using explicit full-record statistics."""
    import numpy as np
    x = _windows(noisy, "record windows", window_samples=4096)
    mean, scale = _stats_for_windows(means, rms, len(x))
    normalized = np.asarray((x - mean) / scale, dtype=np.float32)
    require(np.isfinite(normalized).all(), "record-window normalization overflow")
    return normalized


def restore_residual(noisy, normalized, prediction, rms):
    """Restore only the learned residual using the supplied scope statistics."""
    import numpy as np
    x = _windows(noisy)
    a = np.asarray(normalized)
    b = np.asarray(prediction)
    require(a.dtype == b.dtype == np.float32 and a.shape == b.shape == x.shape
            and np.isfinite(a).all() and np.isfinite(b).all(),
            "invalid residual prediction shape/dtype")
    _, scale = _stats_for_windows(np.zeros((8, 1), dtype=np.float64), rms, len(x))
    result = x + (b.astype(np.float64) - a.astype(np.float64)) * scale
    require(np.isfinite(result).all(), "residual restoration overflow")
    return result


def restore_record(record, normalized_windows, predictions, starts, record_stats,
                   *, window_samples=4096, hop_samples=1024):
    """Restore and OLA complete record windows from the global record base.

    For each window the restored signal is explicitly
    ``record_mean + prediction * record_rms``.  It is then overlap-added as a
    complete signal, not as a residual on top of an immutable local-window
    baseline.  If prediction equals normalized input, the original window is
    copied exactly to preserve the identity contract despite float32 forward
    quantization.
    """
    import numpy as np
    x = _record(record)
    require(window_samples == 4096 and hop_samples == 1024, "fixed record OLA contract changed")
    chosen = list(starts)
    normalized = _windows(normalized_windows, "normalized record windows", window_samples=4096)
    predicted = np.asarray(predictions)
    require(chosen and len(chosen) == len(normalized), "record windows/starts mismatch")
    require(predicted.dtype == np.float32 and predicted.shape == normalized.shape
            and np.isfinite(predicted).all(), "invalid record prediction")
    mean, rms = record_stats
    _validate_stats(mean, rms)
    windows = np.stack([x[s:s + window_samples].T for s in chosen])
    _stats_for_windows(mean, rms, len(windows))
    restored = mean[None, :, :] + predicted.astype(np.float64) * rms[None, :, :]
    identity = np.all(predicted == normalized, axis=(1, 2))
    if np.all(identity):
        # Exact record identity is a hard contract, not an allclose claim.
        return x.copy()
    if np.any(identity):
        restored[identity] = windows[identity]
    hann = np.hanning(window_samples + 2)[1:-1]
    output = np.zeros_like(x, dtype=np.float64)
    weights = np.zeros(len(x), dtype=np.float64)
    for index, start in enumerate(chosen):
        require(0 <= start <= len(x) - window_samples, "record window start out of range")
        output[start:start + window_samples] += restored[index].T * hann[:, None]
        weights[start:start + window_samples] += hann
    require(np.all(weights > 0), "OLA coverage failure")
    output /= weights[:, None]
    require(np.isfinite(output).all() and output.shape == x.shape, "invalid restored record")
    return output


def restored_mse_torch(prediction, normalized, noisy, rms, reference):
    """Autograd loss in exact float64 restored reference units."""
    import torch
    require(prediction.dtype == normalized.dtype == torch.float32
            and prediction.shape == normalized.shape == noisy.shape == reference.shape
            and noisy.dtype == reference.dtype == rms.dtype == torch.float64
            and rms.shape == (*noisy.shape[:2], 1), "loss coordinate/shape mismatch")
    restored = noisy + (prediction.double() - normalized.double()) * rms
    loss = (restored - reference).square().mean(dim=(1, 2))
    require(bool(torch.isfinite(loss).all()), "nonfinite restored-unit loss")
    return loss


def condition_line(signal):
    """Fixed 50 Hz Q30 zero-phase notch; keep units, time axis, and channels."""
    import numpy as np
    from scipy.signal import filtfilt, iirnotch
    arr = np.asarray(signal)
    require(arr.dtype.kind == "f" and arr.ndim in (2, 3) and arr.shape[-1] == 8
            and arr.shape[-2] >= 512 and np.isfinite(arr).all(), "invalid native notch input")
    x = arr.astype(np.float64, copy=False)
    mean = x.mean(axis=-2, keepdims=True)
    b, a = iirnotch(50, 30, fs=250)
    result = filtfilt(b, a, x - mean, axis=-2) + mean
    require(np.isfinite(result).all(), "notch output overflow")
    return result


def build_model(config=None):
    """Lazy 4->24->24->1 protected spectral shrinkage model."""
    import torch
    from torch import nn
    from torch.nn import functional as F
    cfg = load_config() if config is None else config
    require(cfg["model_version"] == "learned_denoise_v14_personal"
            and cfg["model"]["channels"] == [4, 24, 24, 1], "v14 config required")

    class SpectralDenoiserV14(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(4, 24, 7, padding=3), nn.SiLU(),
                nn.Conv1d(24, 24, 5, padding=2), nn.SiLU(),
                nn.Conv1d(24, 1, 1),
            )
            nn.init.zeros_(self.encoder[-1].weight)
            nn.init.constant_(self.encoder[-1].bias, cfg["model"]["initial_gate_bias"])

        def forward_with_aux(self, signal):
            require(isinstance(signal, torch.Tensor) and signal.dtype == torch.float32
                    and signal.ndim == 3 and signal.shape[0] > 0 and signal.shape[1] == 8
                    and signal.shape[-1] in (512, 4096)
                    and bool(torch.isfinite(signal).all()),
                    "forward requires finite float32 [B,8,512|4096] signal only")
            n = signal.shape[-1]
            spectrum = torch.fft.rfft(signal, n=n, dim=-1)
            freq = torch.fft.rfftfreq(n, d=1 / 250, device=signal.device)
            protected = (freq == 0) | ((freq >= 8) & (freq <= 40))
            local = torch.log1p(spectrum.abs() / math.sqrt(n))
            across = local.mean(dim=1, keepdim=True).expand_as(local)
            b, c, bins = local.shape
            smooth = F.avg_pool1d(F.pad(local.reshape(b * c, 1, bins), (2, 2), mode="replicate"), 5, stride=1)
            feature = torch.stack((local, across, local - smooth.reshape(b, c, bins),
                                   (freq / 125).reshape(1, 1, bins).expand_as(local)), dim=2)
            logits = self.encoder(feature.reshape(b * c, 4, bins)).reshape(b, c, bins)
            gain = (1 - torch.sigmoid(logits)).masked_fill(protected, 1)
            output = torch.fft.irfft(spectrum * gain, n=n, dim=-1)
            require(bool(torch.isfinite(output).all()), "model output overflow")
            return output, {"gain": gain, "frequencies_hz": freq, "protected_mask": protected}

        def forward(self, signal):
            return self.forward_with_aux(signal)[0]

    return SpectralDenoiserV14()


def process_windows(noisy, model, normalization_scope, *, record_stats=None, batch_size=32):
    """Apply learned stage with an explicit ``window_512`` or ``record_4096`` scope."""
    import numpy as np
    import torch
    require(normalization_scope in ("window_512", "record_4096"),
            "normalization_scope must be explicit")
    require(type(batch_size) is int and batch_size > 0, "positive integer batch size required")
    x = _windows(noisy)
    if normalization_scope == "window_512":
        require(x.shape[-1] == 512, "window_512 requires 512 samples")
    else:
        require(x.shape[-1] == 4096 and record_stats is not None,
                "record_4096 requires explicit full-record statistics")
    device = next(model.parameters(), torch.empty(0)).device
    require(not model.training, "inference requires model.eval()")
    output = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            chunk = x[start:start + batch_size]
            if normalization_scope == "window_512":
                normalized, _, rms = normalize_noisy_window(chunk)
            else:
                means, rms = record_stats
                means = np.asarray(means)
                rms = np.asarray(rms)
                if means.shape == rms.shape == (len(x), 8, 1):
                    means = means[start:start + len(chunk)]
                    rms = rms[start:start + len(chunk)]
                normalized = normalize_with_record_stats(chunk, means, rms)
            prediction = model(torch.from_numpy(normalized).to(device))
            require(isinstance(prediction, torch.Tensor) and prediction.dtype == torch.float32
                    and tuple(prediction.shape) == normalized.shape and prediction.device == device,
                    "invalid model output contract")
            prediction_np = prediction.cpu().numpy()
            if normalization_scope == "window_512":
                output.append(restore_residual(chunk, normalized, prediction_np, rms))
            else:
                # Record scope restores from the same global record base for
                # every subwindow; no local-window mean is reinserted.
                record_mean, record_rms = record_stats
                identity = np.all(prediction_np == normalized, axis=(1, 2))
                restored = (record_mean[None, :, :] +
                            prediction_np.astype(np.float64) * record_rms[None, :, :])
                if np.any(identity):
                    restored[identity] = chunk[identity]
                output.append(restored)
    return np.concatenate(output)


def window_starts(length: int) -> list[int]:
    require(type(length) is int and length >= 4096, "recording requires >=4096 native samples")
    starts = list(range(0, length - 4096 + 1, 1024))
    if starts[-1] != length - 4096:
        starts.append(length - 4096)
    return starts


def process_recording(signal, model, batch_size=32):
    """Process a complete record with one full-record normalization statistic."""
    import numpy as np
    x = _record(signal)
    require(type(batch_size) is int and batch_size > 0, "positive integer batch size required")
    notch = condition_line(x)
    starts = window_starts(len(x))
    _, record_mean, record_rms, starts = normalize_noisy_record(notch, starts)
    restored_windows = []
    for offset in range(0, len(starts), batch_size):
        selected = starts[offset:offset + batch_size]
        windows = np.stack([notch[s:s + 4096].T for s in selected])
        restored_windows.append(process_windows(
            windows, model, "record_4096",
            record_stats=(record_mean, record_rms), batch_size=batch_size))
    restored = np.concatenate(restored_windows)
    hann = np.hanning(4098)[1:-1]
    output = np.zeros_like(notch, dtype=np.float64)
    weights = np.zeros(len(notch), dtype=np.float64)
    for index, start in enumerate(starts):
        output[start:start + 4096] += restored[index].T * hann[:, None]
        weights[start:start + 4096] += hann
    require(np.all(weights > 0), "OLA coverage failure")
    output /= weights[:, None]
    require(np.isfinite(output).all() and output.shape == x.shape, "invalid OLA output")
    return output, notch, {
        "window_samples": 4096,
        "hop_samples": 1024,
        "window_count": len(starts),
        "window_starts": starts,
        "final_window_start": starts[-1],
        "processed_samples": len(x),
        "normalization_scope": "record_4096",
        "normalization_contract": NORMALIZATION,
        "record_mean_shape": list(record_mean.shape),
        "record_rms_shape": list(record_rms.shape),
        "record_statistics_reused_for_all_subwindows": True,
        "window_local_mean_reinserted": False,
        "unit_conversion_performed": False,
        "resampled": False,
        "source_channel_order_preserved": True,
        "fixed_notch": "50Hz Q30 offline zero phase; report separately",
        "overlap_add": "positive Hann weighted complete restored windows including final window",
        "whole_record_protected_spectrum_guaranteed": False,
        "runtime_enabled": False,
        "production_validated": False,
    }


def load_model(root=ROOT, device="cpu"):
    """Load only a hash-bound, eligible research checkpoint; missing means reject."""
    require(device in ("cpu", "cuda:0"), "explicit cpu or cuda:0 required")
    state, cfg = preflight(root), load_config(root)
    manifest = json.loads(safe_path(root, MANIFEST).read_text(encoding="utf-8"))
    report = json.loads(safe_path(root, REPORT).read_text(encoding="utf-8"))
    for value in (manifest, report):
        require(value["model_version"] == cfg["model_version"]
                and value["config_sha256"] == state["config_sha256"]
                and value["training_lock_sha256"] == state["training_lock_sha256"]
                and value["research_plan_sha256"] == plan_sha(cfg)
                and value["runtime_enabled"] is False
                and value["production_validated"] is False
                and value["selection_gate_passed"] is True,
                "unbound or ineligible v14 artifact")
    require(manifest["checkpoint_sha256"] == sha(safe_path(root, MODEL))
            and manifest["training_report_sha256"] == sha(safe_path(root, REPORT)),
            "artifact hash mismatch")
    import torch
    require(device == "cpu" or torch.cuda.is_available(), "CUDA unavailable")
    checkpoint = torch.load(safe_path(root, MODEL), map_location="cpu", weights_only=True)
    require(checkpoint["config"] == cfg
            and checkpoint["config_sha256"] == state["config_sha256"]
            and checkpoint["training_lock_sha256"] == state["training_lock_sha256"]
            and checkpoint["research_plan_sha256"] == plan_sha(cfg)
            and checkpoint["model_version"] == cfg["model_version"]
            and checkpoint["selection_gate_passed"] is True
            and checkpoint["runtime_enabled"] is False
            and checkpoint["production_validated"] is False,
            "checkpoint binding mismatch")
    model = build_model(cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), manifest


__all__ = [
    "CONFIG", "SOURCES", "LOCK", "MODEL", "MANIFEST", "REPORT", "PLAN",
    "NORMALIZATION", "CONDITIONS", "ROOT", "require", "sha", "canonical_sha",
    "safe_path", "load_config", "validate_plan", "plan_sha", "bindings", "preflight",
    "record_statistics", "normalize_noisy_window", "normalize_noisy_record",
    "normalize_with_record_stats", "restore_residual", "restore_record",
    "condition_line", "build_model", "process_windows", "window_starts",
    "process_recording", "restored_mse_torch", "load_model",
]
