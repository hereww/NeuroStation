"""Transparent, calibration-adaptive SSVEP candidate decoder for offline diagnostics.

This module is intentionally disjoint from the v3/v14 paths.  It is a
candidate-only interface: calibration sessions fit target centroids and robust
feature normalization, while decoding uses only the signal, rest window,
frequencies, and the fitted calibration artifact.  It is fail-closed and may
return ``target_id=None`` when the evidence gates are not met.

Recorded presentation labels are acceptable as calibration metadata and as a
post-hoc evaluation proxy only.  They are not independent gaze truth, and this
module does not claim gaze accuracy, source-domain validation, or deployment
readiness.  No deep model is trained here.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Iterable, Mapping

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "personal_ssvep_decoder_v4.json"
_EPSILON = 1e-12


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
    """Load and lock the exact v4 candidate configuration."""
    path = CONFIG_PATH if config_path is None else Path(config_path)
    try:
        config = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid v4 adaptive candidate configuration: {exc}") from exc

    required = {
        "schema_version": 1,
        "status": "formal",
        "model_version": "personal_ssvep_cca_v4_adaptive_candidate",
        "type": "transparent_calibration_adaptive_evidence_gated_decoder",
        "method": "transparent_cca_with_calibration_session_target_centroids_and_robust_normalization",
        "source_domain": "personal_Cyton",
        "source_domain_validated": False,
        "production_validated": False,
        "detection_validated": False,
        "confidence_is_probability": False,
        "confidence_definition": "(second_centroid_distance-best_centroid_distance)/max(second_centroid_distance,epsilon); not a probability",
        "target_selection": "calibration_centroid_evidence_gated_unknown_allowed",
        "available_definition": "computation_succeeded_only; accepted is true only when calibration integrity and every declared evidence gate passes",
        "sample_rate_hz": 250.0,
        "channel_count": 8,
        "selected_channel_indices": [4, 5, 6, 7],
        "selected_channel_names": ["CH5", "CH6", "CH7", "CH8"],
        "selected_electrode_positions_from_preset": ["P7", "P8", "O1", "O2"],
        "channel_selection_reason": "a_priori posterior visual-channel subset for offline personal SSVEP diagnostics; physical layout still requires verification",
        "minimum_samples": 250,
        "minimum_rest_samples": 250,
        "input_orientation": "channels_by_samples",
        "reference_harmonics": 1,
        "preprocessing": {
            "notch_frequency_hz": 50.0,
            "notch_q": 30.0,
            "bandpass_hz": [1.0, 45.0],
            "bandpass_order": 4,
            "phase": "offline_zero_phase",
        },
        "covariance_regularization": 0.0001,
        "robust_normalization": {
            "location": "median",
            "scale": "1.4826_median_absolute_deviation",
            "minimum_scale": 1e-06,
            "feature_definition": "concatenated_CCA_stimulus_scores_and_log1p_stimulus_to_rest_ratios",
        },
        "calibration": {
            "minimum_repetitions_per_target": 2,
            "centroid_definition": "mean_of_robustly_normalized_calibration_feature_vectors_by_target",
            "label_source": "recorded_presentation_labels_proxy_only_not_independent_gaze_truth",
            "forward_uses_calibration_labels": False,
            "holdout_required": False,
        },
        "evidence_gates": {
            "minimum_stimulus_canonical_score": 0.12,
            "minimum_stimulus_to_rest_ratio": 1.25,
            "minimum_calibrated_evidence": 0.45,
            "minimum_centroid_margin": 0.15,
        },
        "forward_uses_recorded_labels": False,
        "recorded_presentation_labels_are_proxy": True,
        "presented_matches_are_gaze_accuracy": False,
        "gaze_accuracy_claim": False,
        "intended_scope": "explicit_offline_candidate_personal_signal_diagnostic_only",
        "runtime_enabled": False,
        "default_deployment_authorized": False,
    }
    _require(_canonical_bytes(config) == _canonical_bytes(required), "configuration does not match the fixed v4 adaptive candidate decoder")
    return config


def get_config() -> dict:
    return validate_config()


def config_sha256(config_path: str | Path | None = None) -> str:
    return hashlib.sha256(_canonical_bytes(validate_config(config_path))).hexdigest()


def config_summary(config_path: str | Path | None = None) -> dict:
    config = validate_config(config_path)
    return {
        "config": config,
        "config_sha256": hashlib.sha256(_canonical_bytes(config)).hexdigest(),
        "fingerprint_format": "sha256_canonical_json_utf8_sorted_keys_compact",
    }


def _targets(targets: Iterable[tuple[str, float]], sample_rate_hz: float) -> list[tuple[str, float]]:
    items = list(targets)
    _require(bool(items), "v4 CCA targets must be nonempty")
    result: list[tuple[str, float]] = []
    identifiers: set[str] = set()
    frequencies: set[float] = set()
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


def _channel_indices(selected_channel_indices) -> list[int]:
    config = validate_config()
    indices = config["selected_channel_indices"] if selected_channel_indices is None else selected_channel_indices
    _require(isinstance(indices, (tuple, list)) and bool(indices), "selected_channel_indices must be a nonempty sequence")
    result = []
    for index in indices:
        _require(isinstance(index, int) and not isinstance(index, bool), "selected channel indices must be integers")
        _require(0 <= index < config["channel_count"], "selected channel index is outside the configured channel range")
        _require(index not in result, "selected channel indices must be unique")
        result.append(index)
    _require(len(result) >= 2, "at least two selected channels are required for CCA")
    return result


def _prepare(signal, sample_rate_hz: float, selected_channel_indices=None):
    import numpy as np
    from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

    config = validate_config()
    original = np.asarray(signal)
    _require(original.dtype.kind in "iuf" and original.dtype.kind != "b", "CCA input must contain real numeric samples")
    x = np.asarray(original, dtype=np.float64)
    _require(x.ndim == 2 and x.shape[0] == config["channel_count"] and x.shape[1] >= config["minimum_samples"], "CCA input must be [8, samples]")
    _require(bool(np.isfinite(x).all()), "CCA input contains nonfinite values")
    indices = _channel_indices(selected_channel_indices)
    selected = x[indices, :]
    _require(bool(np.all(np.max(selected, axis=1) > np.min(selected, axis=1))), "CCA input contains a flat selected channel")

    selected = selected - np.mean(selected, axis=1, keepdims=True)
    b, a = iirnotch(config["preprocessing"]["notch_frequency_hz"], config["preprocessing"]["notch_q"], fs=sample_rate_hz)
    selected = filtfilt(b, a, selected, axis=1, padtype="odd")
    sos = butter(
        config["preprocessing"]["bandpass_order"],
        config["preprocessing"]["bandpass_hz"],
        btype="bandpass",
        fs=sample_rate_hz,
        output="sos",
    )
    selected = sosfiltfilt(sos, selected, axis=1, padtype="odd")
    _require(bool(np.isfinite(selected).all()), "CCA preprocessing produced nonfinite values")
    selected = selected - np.mean(selected, axis=1, keepdims=True)
    peak = float(np.max(np.abs(selected)))
    _require(math.isfinite(peak) and peak > 0, "CCA filtered input is degenerate")
    return selected / peak


def _canonical_score(x, frequency_hz: float, sample_rate_hz: float) -> float:
    import numpy as np

    config = validate_config()
    t = np.arange(x.shape[1], dtype=np.float64) / sample_rate_hz
    reference = np.asarray([np.sin(2 * np.pi * frequency_hz * t), np.cos(2 * np.pi * frequency_hz * t)])
    reference -= np.mean(reference, axis=1, keepdims=True)
    x = x - np.mean(x, axis=1, keepdims=True)
    n = x.shape[1] - 1
    _require(n > 0, "CCA input is too short")
    cxx, cyy, cxy = x @ x.T / n, reference @ reference.T / n, x @ reference.T / n
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


def _score_pair(stimulus, rest, target_list: list[tuple[str, float]], sample_rate_hz: float) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    stimulus_scores = {identifier: _canonical_score(stimulus, frequency, sample_rate_hz) for identifier, frequency in target_list}
    rest_scores = {identifier: _canonical_score(rest, frequency, sample_rate_hz) for identifier, frequency in target_list}
    ratios = {identifier: float(stimulus_scores[identifier] / max(rest_scores[identifier], _EPSILON)) for identifier, _ in target_list}
    return stimulus_scores, rest_scores, ratios


def _feature_vector(stimulus_scores: Mapping[str, float], ratios: Mapping[str, float], target_list: list[tuple[str, float]]):
    import numpy as np

    values = [float(stimulus_scores[identifier]) for identifier, _ in target_list]
    values.extend(float(np.log1p(max(ratios[identifier], 0.0))) for identifier, _ in target_list)
    vector = np.asarray(values, dtype=np.float64)
    _require(vector.ndim == 1 and bool(np.isfinite(vector).all()), "feature vector is invalid")
    return vector


def _parse_calibration_session(session):
    if isinstance(session, Mapping):
        target_id = session.get("target_id")
        signal = session.get("signal", session.get("stimulus"))
        rest_signal = session.get("rest_signal", session.get("rest"))
    elif isinstance(session, (tuple, list)) and len(session) == 3:
        target_id, signal, rest_signal = session
    else:
        raise ValueError("calibration session must be {target_id, signal, rest_signal} or a three-item tuple")
    _require(isinstance(target_id, str) and target_id.strip() == target_id and bool(target_id), "calibration target_id is invalid")
    return target_id, signal, rest_signal


def fit_calibration(
    calibration_sessions,
    sample_rate_hz: float,
    targets: Iterable[tuple[str, float]],
    *,
    selected_channel_indices=None,
) -> dict:
    """Fit transparent target centroids from labeled calibration sessions.

    The labels are used only to assign calibration examples to centroids.  The
    returned artifact contains no learned neural network and is intended for
    explicit offline candidate evaluation only.
    """
    import numpy as np

    config = validate_config()
    _require(isinstance(sample_rate_hz, Real) and not isinstance(sample_rate_hz, bool) and float(sample_rate_hz) == config["sample_rate_hz"], "v4 CCA requires native 250 Hz input")
    target_list = _targets(targets, float(sample_rate_hz))
    indices = _channel_indices(selected_channel_indices)
    sessions = list(calibration_sessions)
    _require(bool(sessions), "at least one calibration session is required")
    target_ids = {identifier for identifier, _ in target_list}
    rows: list[tuple[str, object]] = []
    counts = {identifier: 0 for identifier, _ in target_list}
    for session in sessions:
        target_id, signal, rest_signal = _parse_calibration_session(session)
        _require(target_id in target_ids, "calibration target_id is not present in targets")
        prepared_stimulus = _prepare(signal, float(sample_rate_hz), indices)
        prepared_rest = _prepare(rest_signal, float(sample_rate_hz), indices)
        stimulus_scores, rest_scores, ratios = _score_pair(prepared_stimulus, prepared_rest, target_list, float(sample_rate_hz))
        rows.append((target_id, _feature_vector(stimulus_scores, ratios, target_list)))
        counts[target_id] += 1

    matrix = np.vstack([vector for _, vector in rows])
    location = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - location), axis=0)
    minimum_scale = float(config["robust_normalization"]["minimum_scale"])
    scale = np.maximum(1.4826 * mad, minimum_scale)
    normalized = (matrix - location) / scale
    centroids = []
    for identifier, _ in target_list:
        target_rows = [normalized[index] for index, (target_id, _) in enumerate(rows) if target_id == identifier]
        centroids.append(np.mean(np.vstack(target_rows), axis=0) if target_rows else np.zeros(matrix.shape[1], dtype=np.float64))
    minimum = int(config["calibration"]["minimum_repetitions_per_target"])
    eligible = all(counts[identifier] >= minimum for identifier, _ in target_list)
    model = {
        "schema_version": 1,
        "model_version": config["model_version"],
        "config_sha256": config_sha256(),
        "sample_rate_hz": float(sample_rate_hz),
        "selected_channel_indices": indices,
        "selected_channel_names": [f"CH{index + 1}" for index in indices],
        "target_frequencies_hz": [float(frequency) for _, frequency in target_list],
        "calibration_target_ids": [identifier for identifier, _ in target_list],
        "counts_by_target": counts,
        "minimum_repetitions_per_target": minimum,
        "eligible": bool(eligible),
        "robust_location": location.tolist(),
        "robust_scale": scale.tolist(),
        "target_centroids": [centroid.tolist() for centroid in centroids],
        "feature_dimension": int(matrix.shape[1]),
        "feature_definition": config["robust_normalization"]["feature_definition"],
        "label_source": config["calibration"]["label_source"],
        "holdout_required": bool(config["calibration"]["holdout_required"]),
        "holdout_validation_passed": False,
        "source_domain": config["source_domain"],
        "source_domain_validated": config["source_domain_validated"],
        "production_validated": config["production_validated"],
        "runtime_enabled": config["runtime_enabled"],
        "default_deployment_authorized": config["default_deployment_authorized"],
        "gaze_accuracy_claim": config["gaze_accuracy_claim"],
    }
    _validate_calibration_model(model)
    return model


def _validate_calibration_model(model: Mapping) -> dict:
    import numpy as np

    config = validate_config()
    _require(isinstance(model, Mapping), "calibration must be a mapping")
    _require(model.get("model_version") == config["model_version"], "calibration model version is incompatible")
    _require(model.get("config_sha256") == config_sha256(), "calibration was fit with a different locked configuration")
    frequencies = model.get("target_frequencies_hz")
    target_ids = model.get("calibration_target_ids")
    _require(isinstance(frequencies, list) and isinstance(target_ids, list) and len(frequencies) == len(target_ids) and bool(frequencies), "calibration target metadata is invalid")
    _require(len(set(target_ids)) == len(target_ids) and all(isinstance(item, str) and item for item in target_ids), "calibration target IDs are invalid")
    _require(all(isinstance(item, Real) and math.isfinite(float(item)) for item in frequencies), "calibration target frequencies are invalid")
    _targets(list(zip(target_ids, frequencies)), float(model.get("sample_rate_hz", config["sample_rate_hz"])))
    raw_indices = model.get("selected_channel_indices")
    _require(isinstance(raw_indices, list), "calibration channel indices are invalid")
    indices = _channel_indices(raw_indices)
    _require(indices == raw_indices, "calibration channel indices are invalid")
    location = np.asarray(model.get("robust_location"), dtype=np.float64)
    scale = np.asarray(model.get("robust_scale"), dtype=np.float64)
    centroids = np.asarray(model.get("target_centroids"), dtype=np.float64)
    feature_dimension = int(model.get("feature_dimension", -1))
    _require(location.ndim == 1 and scale.ndim == 1 and location.shape == scale.shape and location.size == feature_dimension, "calibration normalization vectors are invalid")
    _require(centroids.ndim == 2 and centroids.shape == (len(frequencies), feature_dimension), "calibration centroids are invalid")
    _require(bool(np.isfinite(location).all() and np.isfinite(scale).all() and np.isfinite(centroids).all()), "calibration contains nonfinite values")
    _require(bool(np.all(scale > 0)), "calibration robust scales must be positive")
    counts = model.get("counts_by_target")
    _require(isinstance(counts, Mapping) and all(int(counts.get(identifier, -1)) >= 0 for identifier in target_ids), "calibration counts are invalid")
    _require(isinstance(model.get("eligible"), bool), "calibration eligibility is invalid")
    return dict(model)


def validate_calibration_model(model: Mapping) -> dict:
    """Validate a serialized calibration artifact and return a shallow copy."""
    return _validate_calibration_model(model)


def _match_target_frequencies(target_list: list[tuple[str, float]], model: Mapping) -> list[tuple[str, float]]:
    model_frequencies = [float(value) for value in model["target_frequencies_hz"]]
    _require(len(target_list) == len(model_frequencies), "decode targets do not match calibration target count")
    for (_, frequency), expected in zip(target_list, model_frequencies):
        _require(math.isclose(frequency, expected, rel_tol=0.0, abs_tol=1e-9), "decode target frequencies do not match calibration")
    return target_list


def decode_window(
    signal,
    rest_signal,
    sample_rate_hz: float,
    targets: Iterable[tuple[str, float]],
    calibration: Mapping,
    *,
    selected_channel_indices=None,
) -> dict:
    """Decode one stimulus/rest pair with calibration-derived evidence gates."""
    import numpy as np

    config = validate_config()
    _require(isinstance(sample_rate_hz, Real) and not isinstance(sample_rate_hz, bool) and float(sample_rate_hz) == config["sample_rate_hz"], "v4 CCA requires native 250 Hz input")
    model = _validate_calibration_model(calibration)
    target_list = _match_target_frequencies(_targets(targets, float(sample_rate_hz)), model)
    model_indices = list(model["selected_channel_indices"])
    if selected_channel_indices is not None:
        _require(_channel_indices(selected_channel_indices) == model_indices, "decode channel selection must match calibration")
    prepared_stimulus = _prepare(signal, float(sample_rate_hz), model_indices)
    prepared_rest = _prepare(rest_signal, float(sample_rate_hz), model_indices)
    stimulus_scores, rest_scores, ratios = _score_pair(prepared_stimulus, prepared_rest, target_list, float(sample_rate_hz))
    raw_features = _feature_vector(stimulus_scores, ratios, target_list)
    location = np.asarray(model["robust_location"], dtype=np.float64)
    scale = np.asarray(model["robust_scale"], dtype=np.float64)
    centroids = np.asarray(model["target_centroids"], dtype=np.float64)
    normalized_features = (raw_features - location) / scale
    distances_array = np.sqrt(np.mean((centroids - normalized_features[None, :]) ** 2, axis=1))
    _require(bool(np.isfinite(distances_array).all()), "calibrated distances are nonfinite")
    distances = {identifier: float(distance) for (identifier, _), distance in zip(target_list, distances_array)}
    ordered = sorted(enumerate(target_list), key=lambda item: (distances_array[item[0]], item[0]))
    best_index, (best_id, _) = ordered[0]
    second_distance = float(distances_array[ordered[1][0]]) if len(ordered) > 1 else float(distances_array[best_index])
    best_distance = float(distances_array[best_index])
    calibrated_evidence = float(1.0 / (1.0 + best_distance))
    margin = float((second_distance - best_distance) / max(second_distance, _EPSILON)) if len(ordered) > 1 else 1.0
    best_ratio = float(ratios[best_id])
    best_score = float(stimulus_scores[best_id])
    gates = {
        "calibration_eligible": {"passed": bool(model["eligible"]), "value": bool(model["eligible"])},
        "minimum_stimulus_canonical_score": {
            "threshold": config["evidence_gates"]["minimum_stimulus_canonical_score"],
            "value": best_score,
            "passed": bool(best_score >= config["evidence_gates"]["minimum_stimulus_canonical_score"]),
        },
        "minimum_stimulus_to_rest_ratio": {
            "threshold": config["evidence_gates"]["minimum_stimulus_to_rest_ratio"],
            "value": best_ratio,
            "passed": bool(best_ratio >= config["evidence_gates"]["minimum_stimulus_to_rest_ratio"]),
        },
        "minimum_calibrated_evidence": {
            "threshold": config["evidence_gates"]["minimum_calibrated_evidence"],
            "value": calibrated_evidence,
            "passed": bool(calibrated_evidence >= config["evidence_gates"]["minimum_calibrated_evidence"]),
        },
        "minimum_centroid_margin": {
            "threshold": config["evidence_gates"]["minimum_centroid_margin"],
            "value": margin,
            "passed": bool(margin >= config["evidence_gates"]["minimum_centroid_margin"]),
        },
    }
    accepted = bool(all(item["passed"] for item in gates.values()))
    if accepted:
        target_id, reason = best_id, "all_adaptive_evidence_gates_passed"
    elif not gates["calibration_eligible"]["passed"]:
        target_id, reason = None, "calibration_ineligible"
    elif not gates["minimum_stimulus_canonical_score"]["passed"] or not gates["minimum_stimulus_to_rest_ratio"]["passed"]:
        target_id, reason = None, "insufficient_physiological_evidence"
    else:
        target_id, reason = None, "ambiguous_calibrated_evidence"
    return {
        "available": True,
        "accepted": accepted,
        "method": config["model_version"],
        "target_id": target_id,
        "rejection_reason": None if accepted else reason,
        "confidence": margin,
        "confidence_is_probability": config["confidence_is_probability"],
        "canonical_score": best_score,
        "relative_score": float(best_score / max(math.fsum(stimulus_scores.values()), _EPSILON)),
        "calibrated_evidence": calibrated_evidence,
        "centroid_distances": distances,
        "scores": {key: float(value) for key, value in stimulus_scores.items()},
        "stimulus_scores": {key: float(value) for key, value in stimulus_scores.items()},
        "rest_scores": {key: float(value) for key, value in rest_scores.items()},
        "stimulus_to_rest_ratios": ratios,
        "gates": gates,
        "samples": int(prepared_stimulus.shape[1]),
        "rest_samples": int(prepared_rest.shape[1]),
        "sample_rate_hz": float(sample_rate_hz),
        "selected_channel_indices": model_indices,
        "selected_channel_names": list(model["selected_channel_names"]),
        "source_domain": config["source_domain"],
        "source_domain_validated": config["source_domain_validated"],
        "production_validated": config["production_validated"],
        "gaze_accuracy_claim": config["gaze_accuracy_claim"],
        "presented_matches_are_gaze_accuracy": config["presented_matches_are_gaze_accuracy"],
        "target_selection": config["target_selection"],
        "available_definition": config["available_definition"],
        "runtime_enabled": config["runtime_enabled"],
        "default_deployment_authorized": config["default_deployment_authorized"],
        "config_sha256": config_sha256(),
    }


__all__ = [
    "CONFIG_PATH",
    "config_sha256",
    "config_summary",
    "decode_window",
    "fit_calibration",
    "get_config",
    "validate_calibration_model",
    "validate_config",
]
