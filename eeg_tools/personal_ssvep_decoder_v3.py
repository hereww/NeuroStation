"""Evidence-gated posterior-channel CCA decoder for personal SSVEP diagnostics.

The decoder is deliberately offline and fail-closed. It compares stimulus and
pre-stimulus rest evidence, and returns ``target_id=None`` when the fixed gates
are not met. It never consumes recorded presentation labels and does not claim
gaze accuracy, denoising validity, or production readiness.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Iterable

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "personal_ssvep_decoder_v3.json"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate config key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON config constant: {value}")


def _canonical_bytes(config: dict) -> bytes:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_config(config_path: str | Path | None = None) -> dict:
    path = CONFIG_PATH if config_path is None else Path(config_path)
    try:
        config = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid v3 evidence-gated configuration: {exc}") from exc
    required = {
        "schema_version": 1,
        "status": "formal",
        "model_version": "personal_ssvep_cca_v3_evidence_gated",
        "type": "transparent_evidence_gated_reference_decoder",
        "method": "regularized_first_canonical_correlation_with_stimulus_rest_evidence",
        "source_domain": "personal_Cyton",
        "source_domain_validated": False,
        "production_validated": False,
        "detection_validated": False,
        "confidence_is_probability": False,
        "confidence_definition": "(best_stimulus_score-second_stimulus_score)/best_stimulus_score; zero for a single candidate",
        "target_selection": "evidence_gated_unknown_allowed",
        "available_definition": "computation_succeeded_only; accepted is true only when every declared evidence gate passes",
        "sample_rate_hz": 250.0,
        "channel_count": 8,
        "selected_channel_indices": [4, 5, 6, 7],
        "selected_channel_names": ["CH5", "CH6", "CH7", "CH8"],
        "selected_electrode_positions_from_preset": ["P7", "P8", "O1", "O2"],
        "channel_selection_reason": "a_priori posterior visual-channel subset for SSVEP diagnostic; physical layout still requires verification",
        "minimum_samples": 250,
        "minimum_rest_samples": 250,
        "input_orientation": "channels_by_samples",
        "reference_harmonics": 1,
        "preprocessing": {"notch_frequency_hz": 50.0, "notch_q": 30.0, "bandpass_hz": [1.0, 45.0], "bandpass_order": 4, "phase": "offline_zero_phase"},
        "covariance_regularization": 0.0001,
        "evidence_gates": {"minimum_stimulus_canonical_score": 0.12, "minimum_stimulus_to_rest_ratio": 1.25, "minimum_top_two_margin": 0.15},
        "forward_uses_recorded_labels": False,
        "gaze_accuracy_claim": False,
        "intended_scope": "explicit_offline_personal_signal_evidence_diagnostic_only",
        "runtime_enabled": False,
        "default_deployment_authorized": False,
    }
    _require(_canonical_bytes(config) == _canonical_bytes(required), "configuration does not match the fixed v3 evidence-gated decoder")
    return config


def get_config() -> dict:
    return validate_config()


def config_sha256(config_path: str | Path | None = None) -> str:
    return hashlib.sha256(_canonical_bytes(validate_config(config_path))).hexdigest()


def config_summary(config_path: str | Path | None = None) -> dict:
    config = validate_config(config_path)
    return {"config": config, "config_sha256": hashlib.sha256(_canonical_bytes(config)).hexdigest(), "fingerprint_format": "sha256_canonical_json_utf8_sorted_keys_compact"}


def _targets(targets, sample_rate_hz):
    items = list(targets)
    _require(bool(items), "v3 CCA targets must be nonempty")
    result = []
    identifiers, frequencies = set(), set()
    for item in items:
        _require(isinstance(item, (tuple, list)) and len(item) == 2, "CCA target must be an (ID, frequency) pair")
        identifier, frequency = item
        _require(isinstance(identifier, str) and identifier.strip() == identifier and bool(identifier), "invalid target ID")
        _require(identifier not in identifiers, "duplicate target ID")
        _require(isinstance(frequency, Real) and not isinstance(frequency, bool), "invalid target frequency")
        frequency = float(frequency)
        _require(math.isfinite(frequency) and 0 < frequency < sample_rate_hz / 2, "target frequency outside Nyquist")
        _require(frequency not in frequencies, "duplicate target frequency")
        identifiers.add(identifier)
        frequencies.add(frequency)
        result.append((identifier, frequency))
    return result


def _prepare(signal, sample_rate_hz: float):
    import numpy as np
    from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

    config = validate_config()
    original = np.asarray(signal)
    _require(original.dtype.kind in "iuf", "CCA input must contain real numeric samples")
    x = np.asarray(original, dtype=np.float64)
    _require(x.ndim == 2 and x.shape[0] == config["channel_count"] and x.shape[1] >= config["minimum_samples"], "CCA input must be [8, samples]")
    _require(bool(np.isfinite(x).all()), "CCA input contains nonfinite values")
    _require(bool(np.all(np.max(x, axis=1) > np.min(x, axis=1))), "CCA input contains a flat channel")
    x = x[config["selected_channel_indices"], :]
    x = x / max(float(np.max(np.abs(x))), np.finfo(float).tiny)
    x = x - np.mean(x, axis=1, keepdims=True)
    b, a = iirnotch(config["preprocessing"]["notch_frequency_hz"], config["preprocessing"]["notch_q"], fs=sample_rate_hz)
    x = filtfilt(b, a, x, axis=1, padtype="odd", padlen=None)
    sos = butter(config["preprocessing"]["bandpass_order"], config["preprocessing"]["bandpass_hz"], btype="bandpass", fs=sample_rate_hz, output="sos")
    x = sosfiltfilt(sos, x, axis=1, padtype="odd")
    _require(bool(np.isfinite(x).all()), "CCA preprocessing produced nonfinite values")
    x = x - np.mean(x, axis=1, keepdims=True)
    peak = float(np.max(np.abs(x)))
    _require(math.isfinite(peak) and peak > 0, "CCA filtered input is degenerate")
    return x / peak


def _canonical_score(x, frequency_hz: float, sample_rate_hz: float):
    import numpy as np

    config = validate_config()
    t = np.arange(x.shape[1], dtype=np.float64) / sample_rate_hz
    y = np.asarray([np.sin(2 * np.pi * frequency_hz * t), np.cos(2 * np.pi * frequency_hz * t)])
    y -= np.mean(y, axis=1, keepdims=True)
    x = x - np.mean(x, axis=1, keepdims=True)
    n = x.shape[1] - 1
    cxx, cyy, cxy = x @ x.T / n, y @ y.T / n, x @ y.T / n
    reg = config["covariance_regularization"]

    def inverse_sqrt(matrix):
        trace_scale = float(np.trace(matrix) / matrix.shape[0])
        _require(math.isfinite(trace_scale) and trace_scale > 0, "CCA covariance is degenerate")
        values, vectors = np.linalg.eigh(matrix + reg * trace_scale * np.eye(matrix.shape[0]))
        _require(bool(np.all(values > 0)), "CCA covariance eigensystem is invalid")
        return (vectors * (1 / np.sqrt(values))) @ vectors.T

    score = float(np.linalg.svd(inverse_sqrt(cxx) @ cxy @ inverse_sqrt(cyy), compute_uv=False)[0])
    _require(math.isfinite(score), "CCA score is nonfinite")
    return min(1.0, max(0.0, score))


def _score(prepared, target_list, sample_rate_hz):
    return {identifier: _canonical_score(prepared, frequency, sample_rate_hz) for identifier, frequency in target_list}


def _ordered_scores(scores, target_list):
    return sorted(scores.items(), key=lambda item: (-item[1], target_list.index(next(pair for pair in target_list if pair[0] == item[0]))))


def decode_window(signal, rest_signal, sample_rate_hz: float, targets: Iterable[tuple[str, float]]) -> dict:
    config = validate_config()
    _require(isinstance(sample_rate_hz, Real) and not isinstance(sample_rate_hz, bool) and float(sample_rate_hz) == config["sample_rate_hz"], "v3 CCA requires native 250 Hz input")
    target_list = _targets(targets, float(sample_rate_hz))
    prepared_stimulus = _prepare(signal, float(sample_rate_hz))
    prepared_rest = _prepare(rest_signal, float(sample_rate_hz))
    stimulus_scores = _score(prepared_stimulus, target_list, float(sample_rate_hz))
    rest_scores = _score(prepared_rest, target_list, float(sample_rate_hz))
    ordered = _ordered_scores(stimulus_scores, target_list)
    best_id, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else best_score
    confidence = 0.0 if best_score <= 0 else max(0.0, (best_score - second_score) / best_score)
    eps = 1e-12
    ratios = {key: float(stimulus_scores[key] / max(rest_scores[key], eps)) for key in stimulus_scores}
    best_ratio = ratios[best_id]
    total = math.fsum(stimulus_scores.values())
    relative_score = best_score / total if total > 0 else 0.0
    gates = {
        "minimum_stimulus_canonical_score": {"threshold": config["evidence_gates"]["minimum_stimulus_canonical_score"], "value": float(best_score), "passed": bool(best_score >= config["evidence_gates"]["minimum_stimulus_canonical_score"])},
        "minimum_stimulus_to_rest_ratio": {"threshold": config["evidence_gates"]["minimum_stimulus_to_rest_ratio"], "value": float(best_ratio), "passed": bool(best_ratio >= config["evidence_gates"]["minimum_stimulus_to_rest_ratio"])},
        "minimum_top_two_margin": {"threshold": config["evidence_gates"]["minimum_top_two_margin"], "value": float(confidence), "passed": bool(confidence >= config["evidence_gates"]["minimum_top_two_margin"])},
    }
    gate_passed = all(item["passed"] for item in gates.values())
    if gate_passed:
        target_id, reason = best_id, "all_evidence_gates_passed"
    elif not gates["minimum_stimulus_canonical_score"]["passed"] or not gates["minimum_stimulus_to_rest_ratio"]["passed"]:
        target_id, reason = None, "insufficient_physiological_evidence"
    else:
        target_id, reason = None, "ambiguous_evidence"
    return {
        "available": True,
        "accepted": bool(gate_passed),
        "method": config["model_version"],
        "target_id": target_id,
        "rejection_reason": None if gate_passed else reason,
        "confidence": float(confidence),
        "canonical_score": float(best_score),
        "relative_score": float(relative_score),
        "scores": {key: float(value) for key, value in stimulus_scores.items()},
        "stimulus_scores": {key: float(value) for key, value in stimulus_scores.items()},
        "rest_scores": {key: float(value) for key, value in rest_scores.items()},
        "stimulus_to_rest_ratios": ratios,
        "gates": gates,
        "samples": int(prepared_stimulus.shape[1]),
        "rest_samples": int(prepared_rest.shape[1]),
        "sample_rate_hz": float(sample_rate_hz),
        "selected_channel_indices": list(config["selected_channel_indices"]),
        "selected_channel_names": list(config["selected_channel_names"]),
        "source_domain": config["source_domain"],
        "source_domain_validated": config["source_domain_validated"],
        "production_validated": config["production_validated"],
        "gaze_accuracy_claim": config["gaze_accuracy_claim"],
        "detection_validated": config["detection_validated"],
        "confidence_is_probability": config["confidence_is_probability"],
        "target_selection": config["target_selection"],
        "available_definition": config["available_definition"],
        "runtime_enabled": config["runtime_enabled"],
        "default_deployment_authorized": config["default_deployment_authorized"],
        "config_sha256": config_sha256(),
    }
