"""Personal SSVEP score calibration with explicit holdout and fail-closed gates.

The forward CCA scores are computed without labels. Recorded presentation labels
are consumed only by the calibration step and by post-hoc diagnostic evaluation.
This module does not claim gaze accuracy or production readiness.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "personal_ssvep_calibration_v1.json"


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
        raise ValueError(f"Invalid personal calibration configuration: {exc}") from exc
    required = {
        "schema_version": 1,
        "status": "formal",
        "model_version": "personal_ssvep_calibration_v1",
        "type": "transparent_personal_score_calibrator",
        "method": "target_specific_positive_negative_cca_score_calibration",
        "source_domain": "personal_Cyton",
        "source_domain_validated": False,
        "production_validated": False,
        "detection_validated": False,
        "confidence_is_probability": False,
        "label_source": "recorded_presentation_target_proxy_not_independent_gaze_truth",
        "forward_uses_recorded_labels": False,
        "sample_rate_hz": 250.0,
        "channel_count": 8,
        "selected_channel_indices": [4, 5, 6, 7],
        "selected_channel_names": ["CH5", "CH6", "CH7", "CH8"],
        "selected_electrode_positions_from_preset": ["P7", "P8", "O1", "O2"],
        "window_start_after_onset_samples": 125,
        "window_samples": 1000,
        "minimum_repetitions_per_target": 8,
        "holdout_required": True,
        "minimum_holdout_match_rate": 0.8,
        "minimum_holdout_trials": 16,
        "evidence_gates": {"minimum_calibrated_evidence": 0.5, "minimum_top_two_margin": 0.15},
        "target_selection": "calibrated_evidence_with_unknown_allowed",
        "intended_scope": "explicit_offline_personal_calibration_diagnostic_only",
        "runtime_enabled": False,
        "default_deployment_authorized": False,
    }
    _require(_canonical_bytes(config) == _canonical_bytes(required), "configuration does not match fixed personal calibration v1")
    return config


def get_config() -> dict:
    return validate_config()


def config_sha256(config_path: str | Path | None = None) -> str:
    return hashlib.sha256(_canonical_bytes(validate_config(config_path))).hexdigest()


def config_summary(config_path: str | Path | None = None) -> dict:
    config = validate_config(config_path)
    return {"config": config, "config_sha256": hashlib.sha256(_canonical_bytes(config)).hexdigest(), "fingerprint_format": "sha256_canonical_json_utf8_sorted_keys_compact"}


def _validate_inputs(score_matrix, labels, target_ids):
    import numpy as np
    scores = np.asarray(score_matrix, dtype=np.float64)
    _require(scores.ndim == 2 and scores.shape[0] > 0 and scores.shape[1] == len(target_ids), "score_matrix must be [trials, targets]")
    _require(bool(np.isfinite(scores).all()), "score_matrix contains nonfinite values")
    labels = list(labels)
    _require(len(labels) == scores.shape[0], "labels length does not match score_matrix")
    _require(len(set(target_ids)) == len(target_ids) and all(isinstance(x, str) and x for x in target_ids), "target_ids must be unique nonempty strings")
    _require(all(label in target_ids for label in labels), "labels contain unknown target IDs")
    return scores, labels, list(target_ids)


def fit_calibrator(score_matrix, labels, target_ids, minimum_repetitions: int | None = None, allow_ineligible: bool = False) -> dict:
    import numpy as np
    config = validate_config()
    scores, labels, target_ids = _validate_inputs(score_matrix, labels, target_ids)
    minimum = config["minimum_repetitions_per_target"] if minimum_repetitions is None else minimum_repetitions
    _require(isinstance(minimum, int) and not isinstance(minimum, bool) and minimum >= 2, "minimum_repetitions must be an integer >= 2")
    counts = {target: labels.count(target) for target in target_ids}
    missing = {target: count for target, count in counts.items() if count < minimum}
    if missing:
        return {
            "schema_version": 1,
            "status": "formal",
            "eligible": False,
            "runtime_enabled": False,
            "default_deployment_authorized": False,
            "reason": "minimum_repetitions_per_target_not_met",
            "minimum_repetitions_per_target": minimum,
            "counts_by_target": counts,
            "missing_targets": missing,
            "target_ids": target_ids,
            "config_sha256": config_sha256(),
        }
    parameters = {}
    for index, target in enumerate(target_ids):
        positive = scores[np.asarray([label == target for label in labels]), index]
        negative = scores[np.asarray([label != target for label in labels]), index]
        pos_mean = float(np.mean(positive))
        neg_mean = float(np.mean(negative))
        pos_var = float(np.var(positive, ddof=1)) if len(positive) > 1 else 0.0
        neg_var = float(np.var(negative, ddof=1)) if len(negative) > 1 else 0.0
        delta = pos_mean - neg_mean
        scale = max(math.sqrt(max(0.0, pos_var + neg_var)), 0.03)
        parameters[target] = {"positive_mean": pos_mean, "negative_mean": neg_mean, "delta": delta, "scale": scale, "positive_count": int(len(positive)), "negative_count": int(len(negative))}
    return {
        "schema_version": 1,
        "status": "formal",
        "eligible": not bool(missing),
        "runtime_enabled": False,
        "default_deployment_authorized": False,
        "reason": "eligible_for_holdout_only" if not missing else "diagnostic_fit_with_insufficient_repetitions",
        "minimum_repetitions_per_target": minimum,
        "counts_by_target": counts,
        "missing_targets": missing,
        "target_ids": target_ids,
        "parameters": parameters,
        "holdout_required": config["holdout_required"],
        "holdout_validation_passed": False,
        "production_validated": False,
        "detection_validated": False,
        "label_source": config["label_source"],
        "confidence_is_probability": False,
        "config_sha256": config_sha256(),
    }


def predict(model: dict, scores) -> dict:
    import numpy as np
    config = validate_config()
    _require(isinstance(model, dict), "model must be a dictionary")
    target_ids = list(model.get("target_ids", []))
    x = np.asarray(scores, dtype=np.float64)
    _require(x.ndim == 1 and x.shape[0] == len(target_ids) and bool(np.isfinite(x).all()), "scores must be one finite vector")
    parameters = model.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        return {"available": False, "accepted": False, "target_id": None, "reason": "calibrator_ineligible", "confidence_is_probability": False}
    evidence = {}
    for target in target_ids:
        p = parameters[target]
        delta = float(p["delta"])
        direction = 1.0 if delta >= 0 else -1.0
        evidence[target] = float(direction * (x[target_ids.index(target)] - float(p["negative_mean"])) / max(float(p["scale"]), 1e-12))
    ordered = sorted(evidence.items(), key=lambda item: (-item[1], target_ids.index(item[0])))
    best_id, best = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else best
    margin = float(best - second)
    gate = {
        "minimum_calibrated_evidence": {"threshold": config["evidence_gates"]["minimum_calibrated_evidence"], "value": float(best), "passed": bool(best >= config["evidence_gates"]["minimum_calibrated_evidence"])},
        "minimum_top_two_margin": {"threshold": config["evidence_gates"]["minimum_top_two_margin"], "value": margin, "passed": bool(margin >= config["evidence_gates"]["minimum_top_two_margin"])},
        "holdout_validation": {"passed": bool(model.get("holdout_validation_passed") is True)},
    }
    accepted = bool(model.get("eligible") and model.get("holdout_validation_passed") is True and all(item["passed"] for item in gate.values()))
    return {
        "available": True,
        "accepted": accepted,
        "target_id": best_id if accepted else None,
        "reason": "all_calibration_gates_passed" if accepted else ("calibrator_holdout_not_validated" if model.get("holdout_validation_passed") is not True else "insufficient_calibrated_evidence"),
        "evidence": evidence,
        "confidence": margin,
        "confidence_is_probability": False,
        "gates": gate,
        "runtime_enabled": False,
        "default_deployment_authorized": False,
    }

