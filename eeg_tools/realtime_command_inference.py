"""Strict five-second EEG windows to experimental realtime command predictions.

This adapter temporarily uses a hash-verified Wang2016 checkpoint. It never
changes the original model/experiment flags, trains, reads personal sessions,
or issues hardware, OS, keyboard, or serial actions. The OOD rule is a fixed,
unvalidated heuristic, not evidence of personal-model validity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from eeg_tools import matched_ssvep_decoder as decoder
from eeg_tools.runtime_bindings import runtime_config, model_run
from scripts.train_matched_ssvep_decoder import load_selected_model, select_device

DEFAULT_RUN_DIR = Path(__file__).resolve().parents[1] / "analysis_outputs" / "wang2016_spectral_mlp_20261015"
CHANNEL_NAMES = ("Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2")
WINDOW_FIELDS = frozenset(("type", "request_id", "generation", "sample_rate_hz", "channel_names", "samples"))
TRAINING_SAMPLES = 1250
SAMPLE_RATE_HZ = 250
WINDOW_SECONDS = 5
OOD_MAXABS = 8.0
OOD_P95ABS = 4.0
FLAGS = {
    "status": "formal", "source": "realtime_eeg_model",
    "experiment_only": True, "personal_validated": False, "runtime_enabled": False,
    "production_validated": False, "default_deployment_authorized": False,
    "executed": False,
}


class WindowValidationError(ValueError):
    pass


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise WindowValidationError(reason)


def _correlation(request: Any) -> dict[str, int | None]:
    return {key: request.get(key) if isinstance(request, dict)
            and type(request.get(key)) is int and request[key] >= 0 else None
            for key in ("request_id", "generation")}


def unknown_response(request: Any = None, *, reason: str) -> dict[str, Any]:
    """Fail closed, retaining valid association IDs and never echoing samples."""
    return {
        **FLAGS, "type": "prediction", **_correlation(request),
        "model_loaded": True, "reason": reason, "sample_count": None,
        "training_sample_count": TRAINING_SAMPLES, "window_matches_training": False,
        "raw_prediction": None, "probabilities": None, "ood": None,
        "prediction": {"command": "UNKNOWN", "accepted": False, "executed": False},
    }


def validate_window(request: Any) -> np.ndarray:
    _require(isinstance(request, dict), "request_must_be_object")
    _require(set(request) == WINDOW_FIELDS, "unexpected_or_missing_fields_no_labels_allowed")
    _require(request["type"] == "window", "unsupported_request_type")
    _require(all(type(request[key]) is int and request[key] >= 0
                 for key in ("request_id", "generation")), "invalid_request_association")
    _require(type(request["sample_rate_hz"]) in (int, float)
             and request["sample_rate_hz"] == SAMPLE_RATE_HZ, "sample_rate_mismatch")
    _require(isinstance(request["channel_names"], list)
             and request["channel_names"] == list(CHANNEL_NAMES), "channel_order_mismatch")
    samples = request["samples"]
    _require(isinstance(samples, list) and len(samples) == 8, "invalid_sample_shape")
    _require(all(isinstance(row, list) for row in samples), "invalid_sample_shape")
    n = len(samples[0])
    _require(n == TRAINING_SAMPLES and all(len(row) == n for row in samples),
             "invalid_sample_shape")
    # Do not coerce numeric strings, booleans, complex values, or labels.
    _require(all(type(value) in (int, float) for row in samples for value in row),
             "samples_must_be_real_numbers")
    try:
        eeg = np.asarray(samples, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise WindowValidationError("samples_must_be_finite") from exc
    _require(np.isfinite(eeg).all(), "samples_must_be_finite")
    with np.errstate(over="raise", invalid="raise"):
        ranges = np.ptp(eeg, axis=1)
        # Precision-based flat detection; no assumed microvolt calibration.
        precision = np.finfo(np.float64).eps * np.maximum(np.max(np.abs(eeg), axis=1), 1) * 32
    _require(np.all(ranges > precision), "flat_channel")
    return eeg


class RealtimeCommandInference:
    """Hash-verified 72-feature MLP; returns commands without executing them."""

    def __init__(self, run_dir: Path | str = DEFAULT_RUN_DIR, *, device: str = "cpu"):
        self.device = select_device(device, require_cuda=device == "cuda")
        # Existing loader verifies source/code hashes, selection, state_dict and
        # weights_only=True. Do not rewrite training artifacts or their flags.
        mappings = runtime_config().get("root_mappings", [])
        if mappings:
            from eeg_tools.model_relocation import load_relocated_model
            loaded = load_relocated_model(model_run(run_dir), self.device, mappings)
        else:
            loaded = load_selected_model(Path(run_dir), self.device)
        self.model, self.preregistration, self.selection = loaded
        self.model.eval()
        self.feature_mean = self.model.feature_mean.detach().cpu().numpy().astype(np.float64, copy=True)
        self.feature_std = self.model.feature_std.detach().cpu().numpy().astype(np.float64, copy=True)
        _require(self.feature_mean.shape == (72,) and self.feature_std.shape == (72,)
                 and np.isfinite(self.feature_mean).all() and np.isfinite(self.feature_std).all()
                 and (self.feature_std >= 1e-6).all(), "invalid_model_standardizer")

    def ready(self) -> dict[str, Any]:
        return {
            **FLAGS, "type": "ready", "model_loaded": True, "device": str(self.device),
            "channel_names": list(CHANNEL_NAMES), "sample_rate_hz": SAMPLE_RATE_HZ,
            "sample_count": TRAINING_SAMPLES, "window_seconds": WINDOW_SECONDS,
            "sample_layout": "channels_by_samples", "input_features": 72,
            "command_gate": dict(decoder.GATE),
            "commands": {str(k): v for k, v in decoder.COMMANDS.items()},
            "training_sample_count": TRAINING_SAMPLES,
            "ood_gate": {"maxabs_limit": OOD_MAXABS, "p95abs_limit": OOD_P95ABS,
                         "heuristic": True, "validated": False},
        }

    def infer(self, request: Any) -> dict[str, Any]:
        response = unknown_response(request, reason="invalid_request")
        try:
            eeg = validate_window(request)
            n = eeg.shape[1]
            response.update(sample_count=n, window_matches_training=True)
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                features = decoder.spectral_features(eeg)
                _require(features.shape == (72,) and np.isfinite(features).all(), "nonfinite_features")
                standardized = (features.astype(np.float64) - self.feature_mean) / self.feature_std
            _require(features.shape == (72,) and np.isfinite(standardized).all(), "nonfinite_features")
            absolute = np.abs(standardized)
            maxabs, p95abs = float(absolute.max()), float(np.percentile(absolute, 95))
            ood_passed = maxabs <= OOD_MAXABS and p95abs <= OOD_P95ABS
            response["ood"] = {
                "maxabs": maxabs, "p95abs": p95abs, "maxabs_limit": OOD_MAXABS,
                "p95abs_limit": OOD_P95ABS, "passed": ood_passed,
                "heuristic": True, "validated": False,
            }
            with torch.inference_mode():
                tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits = self.model(tensor)
                _require(isinstance(logits, torch.Tensor) and logits.shape == (1, 3)
                         and bool(torch.isfinite(logits).all()), "nonfinite_or_invalid_logits")
                probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            raw = decoder.command_prediction(probabilities)
            response.update(raw_prediction=raw, probabilities=raw["probabilities"])
            if not ood_passed:
                response["reason"] = "ood_feature_gate"
            elif raw["accepted"] is not True or raw["command"] not in ("MOVE_LEFT", "MOVE_RIGHT", "SELECT"):
                response["reason"] = "model_confidence_gate"
            else:
                response["reason"] = "accepted"
                response["prediction"] = {"command": raw["command"], "accepted": True, "executed": False}
        except WindowValidationError as exc:
            response["reason"] = str(exc)
        except Exception:
            # Malformed windows/inference errors cannot crash the NDJSON worker
            # or accidentally accept a command. Do not expose the input EEG.
            response["reason"] = "inference_failed"
        return response
