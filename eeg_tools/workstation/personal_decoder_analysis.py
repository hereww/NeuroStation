"""Versioned personal CCA diagnostics; scientific imports stay in the worker.

The candidate is a forced reference argmax, not a validated SSVEP detection.
Recorded targets are used only after inference to describe development matches.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

TRACKS = ("raw", "traditional", "v9")
PLAN = {
    "schema_version": 1, "status": "formal", "method": "personal_ssvep_cca_v1",
    "window": "same recorded and trimmed intervals as the existing FFT comparison",
    "channels": "all 8 source channels; physical electrode mapping unverified",
    "preprocessing": "each input track is filtered per trial by the fixed CCA decoder",
    "traditional_track": "already full-record filtered; CCA applies its fixed preprocessing again",
    "development_data": True, "both_personal_sessions_previously_explored": True,
    "independent_validation": False, "forward_uses_recorded_labels": False,
    "rest_control": "nonoverlapping 250-sample raw windows outside every recorded stimulus interval",
    "noise_control": {"seed": 20261013, "windows": 32, "samples": 1000,
                      "distribution": "independent unit Gaussian; no target signal"},
    "label_permutation": {"seed": 20261013, "draws": 2000,
                          "interpretation": "exploratory label association only; not independent validation"},
}


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def analyze(tracks, targets, recorded_trials, unavailable_reason=None):
    import numpy as np
    from eeg_tools.personal_ssvep_decoder import decode_window, get_config

    config = get_config()
    payload = {
        "schema_version": 1, "status": "formal", "method": PLAN["method"],
        "plan": json.loads(json.dumps(PLAN)), "config": config,
        "config_sha256": _digest(config),
        "implementation_sha256": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).absolute(), Path(__file__).absolute().parents[1] / "personal_ssvep_decoder.py")},
        "source_domain": "personal_Cyton", "source_domain_validated": False,
        "production_validated": False, "independent_gaze_truth_available": False,
        "confidence_is_probability": False, "detection_validated": False,
        "development_data": True, "training_performed": False,
        "clean_ground_truth_available": False, "denoising_efficacy_validated": False,
        "prediction_definition": "forced reference argmax; available does not mean an SSVEP exists",
        "confidence_definition": "(best-second)/best score margin; not a calibrated probability",
        "label_definition": "presented target proxy; gaze annotations are not independent truth",
        "trials": [], "tracks": {},
    }
    valid_ids = {identifier for identifier, _ in targets}
    for original in recorded_trials:
        trial = {key: original.get(key) for key in (
            "trial_index", "presented_target_id", "gaze_target_id", "onset_sample_index",
            "offset_sample_index", "window_start_sample_index")}
        start, end = trial["window_start_sample_index"], trial["offset_sample_index"]
        trial["window_samples"] = end - start if isinstance(start, int) and isinstance(end, int) else 0
        trial["predictions"], trial["prediction_status"] = {}, {}
        for name in TRACKS:
            signal = tracks.get(name)
            reason = unavailable_reason or original.get("unavailable_reason")
            if signal is None:
                reason = reason or "track unavailable"
            prediction = {"available": False, "target_id": None, "confidence": None, "scores": {}}
            if not reason:
                try:
                    score = decode_window(signal[start:end].T, 250.0, targets)
                    prediction = {"available": True, "target_id": score["target_id"],
                                  "confidence": score["confidence"], "scores": score["scores"]}
                except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
                    reason = str(exc)
            trial["predictions"][name] = prediction
            trial["prediction_status"][name] = {"available": prediction["available"], "reason": reason}
        payload["trials"].append(trial)
    rng = np.random.default_rng(PLAN["label_permutation"]["seed"])
    for name in TRACKS:
        decoded = [row for row in payload["trials"] if row["predictions"][name]["available"]]
        comparable = [row for row in decoded if row["presented_target_id"] in valid_ids]
        predictions = [row["predictions"][name]["target_id"] for row in comparable]
        labels = [row["presented_target_id"] for row in comparable]
        matched = sum(a == b for a, b in zip(labels, predictions))
        confusion = {label: {candidate: 0 for candidate in sorted(valid_ids)} for label in sorted(valid_ids)}
        for label, candidate in zip(labels, predictions):
            confusion[label][candidate] += 1
        null_counts = [sum(a == b for a, b in zip(rng.permutation(labels), predictions))
                       for _ in range(PLAN["label_permutation"]["draws"])] if labels else []
        payload["tracks"][name] = {
            "available": bool(decoded), "trial_count": len(decoded),
            "recorded_trial_count": len(recorded_trials), "presented_comparable_count": len(comparable),
            "presented_match_count": matched, "confusion_matrix_presented_proxy": confusion,
            "reason": None if decoded else (unavailable_reason or "no valid recorded intervals"),
            "label_permutation": {"draws": len(null_counts), "observed_matches": matched,
                "tail_fraction_with_plus_one": (1 + sum(n >= matched for n in null_counts)) / (1 + len(null_counts))
                    if null_counts else None,
                "validation_performed": False, "parameter_selection_adjusted": False},
        }
    payload["available"] = payload["tracks"]["raw"]["available"]
    payload["unavailable_reason"] = None if payload["available"] else (
        unavailable_reason or "no valid recorded CCA intervals")
    payload["controls"] = _controls(tracks.get("raw"), targets, recorded_trials, unavailable_reason)
    validate_analysis(payload)
    return payload


def _controls(raw, targets, trials, reason):
    import numpy as np
    from eeg_tools.personal_ssvep_decoder import decode_window

    def summaries(windows):
        rows = []
        for start, signal in windows:
            try:
                result = decode_window(signal.T, 250.0, targets)
                rows.append({"start_sample": start, "samples": len(signal), "available": True,
                             "target_id": result["target_id"], "confidence": result["confidence"],
                             "maximum_score": max(result["scores"].values())})
            except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
                rows.append({"start_sample": start, "samples": len(signal), "available": False, "reason": str(exc)})
        return rows

    if not targets:
        return {"available": False, "reason": reason or "protocol targets unavailable", "noise": [], "rest": []}
    rng = np.random.default_rng(PLAN["noise_control"]["seed"])
    noise = summaries([(None, rng.standard_normal((1000, 8))) for _ in range(32)])
    rest_windows = []
    rest_reason = reason
    if raw is not None and trials and not reason:
        spans = [(row.get("onset_sample_index"), row.get("offset_sample_index")) for row in trials]
        if all(isinstance(a, int) and isinstance(b, int) and 0 <= a < b <= len(raw) for a, b in spans):
            occupied = np.zeros(len(raw), dtype=bool)
            for a, b in spans:
                occupied[a:b] = True
            rest_windows = [(start, raw[start:start + 250]) for start in range(0, len(raw) - 249, 250)
                            if not occupied[start:start + 250].any()]
        else:
            rest_reason = "incomplete stimulus intervals; cannot establish non-stimulus windows"
    return {"available": True, "noise": noise, "rest": summaries(rest_windows),
            "rest_unavailable_reason": rest_reason if not rest_windows else None,
            "rest_definition": "recorded non-stimulus interval; absence of physiological SSVEP is unverified",
            "no_target_rejection_calibrated": False,
            "interpretation": "even noise can produce a forced candidate; do not use as validated commands"}


def validate_analysis(payload):
    """Validate a saved extension using its own snapshot, keeping old bundles readable."""
    def require(ok, message):
        if not ok:
            raise ValueError("personal decoder: " + message)
    require(isinstance(payload, dict) and payload.get("schema_version") == 1
            and payload.get("status") == "formal" and payload.get("method") == PLAN["method"], "invalid schema")
    for field in ("source_domain_validated", "production_validated", "independent_gaze_truth_available",
                  "confidence_is_probability", "detection_validated", "training_performed",
                  "clean_ground_truth_available", "denoising_efficacy_validated"):
        require(payload.get(field) is False, "invalid validation claim: " + field)
    require(payload.get("source_domain") == "personal_Cyton" and payload.get("development_data") is True,
            "invalid domain/development status")
    require(isinstance(payload.get("plan"), dict) and payload["plan"].get("forward_uses_recorded_labels") is False
            and payload["plan"].get("independent_validation") is False, "invalid analysis plan")
    config = payload.get("config")
    require(isinstance(config, dict) and _digest(config) == payload.get("config_sha256"), "configuration fingerprint mismatch")
    require(config.get("status") == "formal" and config.get("production_validated") is False
            and config.get("sample_rate_hz") == 250 and config.get("channel_count") == 8, "invalid configuration snapshot")
    require(set(payload.get("tracks", {})) == set(TRACKS) and isinstance(payload.get("trials"), list), "missing tracks/trials")
    for row in payload["trials"]:
        require(isinstance(row, dict) and set(row.get("predictions", {})) == set(TRACKS), "invalid trial")
        for name in TRACKS:
            prediction = row["predictions"][name]
            require(type(prediction.get("available")) is bool, "invalid prediction availability")
            require(row.get("prediction_status", {}).get(name, {}).get("available") is prediction["available"], "availability mismatch")
            if prediction["available"]:
                scores = prediction.get("scores", {})
                require(prediction.get("target_id") in scores and scores, "candidate missing from scores")
                require(all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                            for value in scores.values()), "invalid canonical score")
                value = prediction.get("confidence")
                require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1, "invalid margin")
            else:
                require(prediction.get("target_id") is None and prediction.get("confidence") is None,
                        "unavailable prediction has a candidate")
    for name in TRACKS:
        track = payload["tracks"][name]
        count = sum(row["predictions"][name]["available"] for row in payload["trials"])
        require(track.get("trial_count") == count and track.get("recorded_trial_count") == len(payload["trials"]),
                "trial count mismatch")
        require(type(track.get("presented_match_count")) is int
                and 0 <= track["presented_match_count"] <= track.get("presented_comparable_count", -1) <= count,
                "proxy match count invalid")
    require(payload.get("available") is payload["tracks"]["raw"].get("available"), "raw availability mismatch")
    json.dumps(payload, allow_nan=False)
    return payload
