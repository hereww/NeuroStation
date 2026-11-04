"""Fixed offline personal-Cyton CCA diagnostics; no trained detection claim.

Importing this module uses only the standard library. Numerical dependencies are
loaded by decode_window only. Candidate stimulus frequencies are references,
never recorded labels or gaze truth. Even noise yields a forced argmax: available
means computation succeeded, not that an SSVEP or reliable target was detected.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Iterable

_CONFIG_JSON = """{
  "schema_version": 1,
  "status": "formal",
  "model_version": "personal_ssvep_cca_v1",
  "type": "transparent_reference_decoder",
  "method": "regularized_first_canonical_correlation",
  "source_domain": "personal_Cyton",
  "source_domain_validated": false,
  "production_validated": false,
  "detection_validated": false,
  "confidence_is_probability": false,
  "confidence_definition": "(best_score-second_score)/best_score; zero for a single candidate",
  "target_selection": "forced_argmax_diagnostic_only",
  "available_definition": "computation_succeeded_only_not_ssvep_detection",
  "sample_rate_hz": 250.0,
  "channel_count": 8,
  "minimum_samples": 250,
  "input_orientation": "channels_by_samples",
  "reference_harmonics": 1,
  "preprocessing": {
    "notch_frequency_hz": 50.0,
    "notch_q": 30.0,
    "bandpass_hz": [1.0, 45.0],
    "bandpass_order": 4,
    "phase": "offline_zero_phase"
  },
  "covariance_regularization": 0.0001,
  "forward_uses_recorded_labels": false,
  "gaze_accuracy_claim": false,
  "intended_scope": "explicit_offline_personal_signal_diagnostic_only"
}"""
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "personal_ssvep_decoder_v1.json"


def _fixed_config() -> dict:
    return json.loads(_CONFIG_JSON)


# Backward-compatible flat parameter access, with immutable nested values.
_flat_config = _fixed_config()
_flat_config.update(_flat_config.pop("preprocessing"))
_flat_config["bandpass_hz"] = tuple(_flat_config["bandpass_hz"])
CONFIG = MappingProxyType(_flat_config)
del _flat_config


def _canonical_bytes(config: dict) -> bytes:
    return json.dumps(config, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate config key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON config constant: {value}")


def validate_config(config_path: str | Path | None = None) -> dict:
    """Read and validate the exact fixed configuration; return a detached copy.

    Object key order and whitespace do not affect the semantic SHA256. Types,
    values, extra/missing fields and duplicate JSON keys are checked strictly.
    There is no configuration override of this decoder's fixed parameters.
    """
    try:
        config = json.loads(Path(CONFIG_PATH if config_path is None else config_path).read_text(encoding="utf-8"),
                            object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        # Canonical JSON distinguishes false from 0, and integers from floats.
        if _canonical_bytes(config) != _canonical_bytes(_fixed_config()):
            raise ValueError("CCA JSON configuration does not match the fixed decoder")
        return config
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid personal CCA configuration: {exc}") from exc


def get_config() -> dict:
    """Return a validated, detached JSON configuration for bundle snapshots."""
    return validate_config()


def config_sha256(config_path: str | Path | None = None) -> str:
    """SHA256 of the validated, canonical UTF-8 JSON configuration."""
    return hashlib.sha256(_canonical_bytes(validate_config(config_path))).hexdigest()


def config_summary(config_path: str | Path | None = None) -> dict:
    """Standard-library-only configuration and semantic fingerprint for the GUI."""
    config = validate_config(config_path)
    return {"config": config,
            "config_sha256": hashlib.sha256(_canonical_bytes(config)).hexdigest(),
            "fingerprint_format": "sha256_canonical_json_utf8_sorted_keys_compact"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _targets(targets, sample_rate_hz):
    try:
        items = list(targets)
    except TypeError as exc:
        raise ValueError("CCA targets must be an iterable of (ID, frequency) pairs") from exc
    _require(bool(items), "CCA targets must be nonempty")
    result = []
    ids, frequencies = set(), set()
    for item in items:
        _require(isinstance(item, (tuple, list)) and len(item) == 2,
                 "CCA target must be an (ID, frequency) pair")
        identifier, frequency = item
        _require(isinstance(identifier, str) and bool(identifier.strip())
                 and identifier == identifier.strip() and identifier.isprintable(),
                 "CCA target ID must be a nonempty printable string without edge whitespace")
        _require(identifier not in ids, "CCA target IDs must be unique")
        _require(isinstance(frequency, Real) and not isinstance(frequency, bool),
                 "CCA target frequency must be a real number")
        try:
            frequency = float(frequency)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("CCA target frequency cannot be represented numerically") from exc
        _require(math.isfinite(frequency) and 0 < frequency < sample_rate_hz / 2,
                 "CCA target frequency must be finite and within (0, Nyquist)")
        _require(frequency not in frequencies, "CCA target frequencies must be unique")
        ids.add(identifier)
        frequencies.add(frequency)
        result.append((identifier, frequency))
    return result


def _prepare(signal, sample_rate_hz: float):
    import numpy as np
    from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

    original = np.asarray(signal)
    _require(original.dtype.kind in "iuf", "CCA input must contain real numeric samples")
    x = np.asarray(original, dtype=np.float64)
    _require(x.ndim == 2 and x.shape[0] == CONFIG["channel_count"]
             and x.shape[1] >= CONFIG["minimum_samples"],
             "CCA input must be [8, samples] with at least 250 samples")
    _require(bool(np.isfinite(x).all()), "CCA input contains nonfinite values")
    _require(bool(np.all(np.max(x, axis=1) > np.min(x, axis=1))),
             "CCA input contains a flat channel")
    # Global scaling preserves channel covariance proportions and prevents
    # overflow/underflow. No absolute voltage cutoff or unit conversion is used.
    x = x / np.max(np.abs(x))
    x = x - np.mean(x, axis=1, keepdims=True)
    peak = np.max(np.abs(x))
    _require(bool(peak > 0), "CCA input has no resolvable variation")
    x = x / peak
    _require(bool(np.all(np.max(np.abs(x), axis=1) > 32 * np.finfo(float).eps)),
             "CCA input contains a numerically degenerate channel")
    b, a = iirnotch(CONFIG["notch_frequency_hz"], CONFIG["notch_q"], fs=sample_rate_hz)
    x = filtfilt(b, a, x, axis=1, padtype="odd", padlen=None)
    sos = butter(CONFIG["bandpass_order"], CONFIG["bandpass_hz"], btype="bandpass",
                 fs=sample_rate_hz, output="sos")
    x = sosfiltfilt(sos, x, axis=1, padtype="odd", padlen=None)
    _require(bool(np.isfinite(x).all()), "CCA preprocessing produced nonfinite samples")
    x = x - np.mean(x, axis=1, keepdims=True)
    peak = np.max(np.abs(x))
    _require(bool(peak > 0), "CCA filtered input is degenerate")
    x = x / peak
    _require(bool(np.all(np.max(np.abs(x), axis=1) > 32 * np.finfo(float).eps)),
             "CCA filtered input contains a numerically degenerate channel")
    return x


def _canonical_score(x, frequency_hz: float, sample_rate_hz: float):
    import numpy as np

    t = np.arange(x.shape[1], dtype=np.float64) / sample_rate_hz
    y = np.asarray([np.sin(2 * np.pi * frequency_hz * t),
                    np.cos(2 * np.pi * frequency_hz * t)])
    y -= np.mean(y, axis=1, keepdims=True)
    x = x - np.mean(x, axis=1, keepdims=True)
    n = x.shape[1] - 1
    cxx, cyy, cxy = x @ x.T / n, y @ y.T / n, x @ y.T / n
    reg = CONFIG["covariance_regularization"]

    def inverse_sqrt(matrix):
        _require(bool(np.isfinite(matrix).all()), "CCA covariance is nonfinite")
        trace_scale = float(np.trace(matrix) / matrix.shape[0])
        _require(math.isfinite(trace_scale) and trace_scale > 0,
                 "CCA reference or signal covariance is degenerate")
        matrix = matrix + reg * trace_scale * np.eye(matrix.shape[0])
        values, vectors = np.linalg.eigh(matrix)
        _require(bool(np.isfinite(values).all()) and bool(np.all(values > 0)),
                 "CCA covariance eigensystem is invalid")
        return (vectors * (1 / np.sqrt(values))) @ vectors.T

    whitened = inverse_sqrt(cxx) @ cxy @ inverse_sqrt(cyy)
    _require(bool(np.isfinite(whitened).all()), "CCA whitening is nonfinite")
    score = float(np.linalg.svd(whitened, compute_uv=False)[0])
    _require(math.isfinite(score) and -1e-10 <= score <= 1 + 1e-10,
             "CCA canonical score is outside numerical bounds")
    return min(1.0, max(0.0, score))


def decode_window(signal, sample_rate_hz: float, targets: Iterable[tuple[str, float]]) -> dict:
    """Compute diagnostic CCA scores from [8,T>=250] native-250-Hz samples.

    Raises ValueError for invalid inputs/configuration or numerical failures.
    available=True reports successful computation only. target_id is forced
    argmax (ties follow candidate order), including for noise and rest windows.
    confidence is the relative top-two margin, never a calibrated probability;
    one candidate has no competing score and therefore receives margin zero.
    No reliable-target/detection threshold has been established or applied.
    """
    fingerprint = config_sha256()
    _require(isinstance(sample_rate_hz, Real) and not isinstance(sample_rate_hz, bool)
             and sample_rate_hz == CONFIG["sample_rate_hz"],
             "CCA requires native 250 Hz input")
    target_list = _targets(targets, sample_rate_hz)
    import numpy as np

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            prepared = _prepare(signal, float(sample_rate_hz))
            scores = {identifier: _canonical_score(prepared, frequency, float(sample_rate_hz))
                      for identifier, frequency in target_list}
            _require(all(math.isfinite(value) and 0 <= value <= 1 for value in scores.values()),
                     "CCA canonical score is outside numerical bounds")
            ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            best_id, best = ordered[0]
            second = ordered[1][1] if len(ordered) > 1 else best
            confidence = (best - second) / best if best > 0 else 0.0
            total = sum(scores.values())
            relative = best / total if total > 0 else 0.0
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError, TypeError) as exc:
        raise ValueError(f"CCA numerical/input failure: {exc}") from exc
    return {
        "available": True,
        "method": CONFIG["model_version"],
        "target_id": best_id,
        "confidence": float(confidence),
        "relative_score": float(relative),
        "scores": {key: float(value) for key, value in scores.items()},
        "samples": int(prepared.shape[1]),
        "sample_rate_hz": float(sample_rate_hz),
        "source_domain": CONFIG["source_domain"],
        "source_domain_validated": False,
        "production_validated": False,
        "gaze_accuracy_claim": False,
        "detection_validated": False,
        "confidence_is_probability": False,
        "target_selection": CONFIG["target_selection"],
        "available_definition": CONFIG["available_definition"],
        "confidence_definition": CONFIG["confidence_definition"],
        "config_sha256": fingerprint,
    }
