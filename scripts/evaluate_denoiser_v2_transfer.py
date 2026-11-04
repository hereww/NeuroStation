"""Frozen offline v2 transfer preservation / KNOWN-v1 stress regression.

Uses separately audited Dong2023 metadata, never v1 frequency/onset constants.
No training, downloads, resampling, test tuning, raw writes or report overwrites.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Small 500-sample CPU TCN calls are badly oversubscribed on many-core hosts.
# Respect externally supplied BLAS limits; --threads controls PyTorch explicitly.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
from scipy.io import loadmat
from eeg_tools import learned_denoise_v2 as v2

SUBJECTS = ("S2.mat", "S3.mat")
REGRESSION_SESSION = "session_20260924_131134_807"
HARMONICS = 3
CCA_SECONDS = 4.0
NUMERICAL_TOLERANCE = 1e-12
PRESERVATION_THRESHOLDS = {
    "gated_mean_target_amplitude_ratio_min": 0.95,
    "gated_mean_target_amplitude_ratio_max": 1.05,
    "gated_cca_accuracy_drop_max": 0.025,
}


class TransferEvaluationError(ValueError):
    """Fail closed on missing metadata, incompatible shapes or invalid evidence."""


def required(obj: Any, keys: tuple[str, ...], name: str) -> dict:
    if not isinstance(obj, dict):
        raise TransferEvaluationError(f"{name} must be an object")
    missing = set(keys) - obj.keys()
    if missing:
        raise TransferEvaluationError(f"{name} missing fields: {sorted(missing)}")
    return obj


def number(value: Any, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise TransferEvaluationError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise TransferEvaluationError(f"{name} must be positive")
    return float(value)


def sample_index(seconds: Any, rate: float, name: str) -> int:
    samples = number(seconds, name) * rate
    if samples < 0 or not np.isclose(samples, round(samples), atol=1e-7, rtol=0):
        raise TransferEvaluationError(f"{name} must map to nonnegative integer samples")
    return int(round(samples))


def frequencies(value: Any, count: int, rate: float, name: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != count or count < 2:
        raise TransferEvaluationError(f"{name} must contain exactly target_count frequencies")
    result = np.array([number(f, name, True) for f in value])
    if len(np.unique(result)) != count or np.any(HARMONICS * result >= rate / 2):
        raise TransferEvaluationError(f"{name} must be unique with 3 harmonics below Nyquist")
    return result


def window_slice(window: dict, total: int, name: str) -> slice:
    required(window, ("start", "stop_exclusive", "count", "indexing"), name)
    if (window["indexing"] != "zero_based_half_open"
            or any(type(window[k]) is not int for k in ("start", "stop_exclusive", "count"))
            or not 0 <= window["start"] < window["stop_exclusive"] <= total
            or window["count"] != window["stop_exclusive"] - window["start"]):
        raise TransferEvaluationError(f"{name} must be a consistent zero-based half-open sample interval")
    return slice(window["start"], window["stop_exclusive"])


def validate_metadata(meta: dict, model_rate: float) -> dict:
    """Read DONG2023_METADATA.json audit schema v1.0, including pending mapping.

    Required: dataset, metadata_status, sources, sample_rate_hz, array_key,
    expected_shape, axis_order, target_count, confirmed_frequency_set_hz,
    target_frequencies_hz (nullable), target_axis_mapping_confirmed,
    analysis_start_s, analysis_duration_s, analysis_window_samples.
    Confirmed onset also requires trial_structure onset/offset within epoch.
    Unconfirmed onset permits ONLY explicitly authorized whole stored epoch.
    CCA requires a metadata-authorized 4s interval; absent permission => not_assessed.
    candidate_target_axis_frequencies_hz is NEVER used, even if populated.
    """
    required(meta, ("schema_version", "dataset", "metadata_status", "sources", "sample_rate_hz",
                   "array_key", "expected_shape", "axis_order", "target_count",
                   "confirmed_frequency_set_hz", "target_frequencies_hz", "target_axis_mapping_confirmed",
                   "analysis_start_s", "analysis_duration_s", "analysis_window_samples"), "metadata")
    if meta["schema_version"] != "1.0" or meta["dataset"] != "Dong2023":
        raise TransferEvaluationError("expected audited Dong2023 metadata schema_version '1.0'")
    if not isinstance(meta["metadata_status"], str) or not meta["metadata_status"].strip():
        raise TransferEvaluationError("metadata_status must identify audit state")
    sources = meta["sources"]
    if not isinstance(sources, list) or not sources:
        raise TransferEvaluationError("sources must contain audit evidence")
    for source in sources:
        required(source, ("role",), "source")
        if not isinstance(source["role"], str) or not source["role"].strip():
            raise TransferEvaluationError("source role must be a nonempty evidence description")
        references = [source.get(k) for k in ("url", "local_path")]
        if not any(isinstance(ref, str) and ref.strip() for ref in references):
            raise TransferEvaluationError("source must include a nonempty url or local_path audit reference")
    rate = number(meta["sample_rate_hz"], "sample_rate_hz", True)
    if rate != model_rate:
        raise TransferEvaluationError("metadata sample_rate_hz differs from model; no implicit resampling")
    shape, axes = meta["expected_shape"], meta["axis_order"]
    if not isinstance(shape, list) or len(shape) != 4 or any(type(x) is not int or x <= 0 for x in shape):
        raise TransferEvaluationError("expected_shape must contain four positive 4D sizes")
    if (not isinstance(axes, list) or len(axes) != 4 or not all(isinstance(a, str) for a in axes)
            or set(axes) != {"channel", "time", "target", "block"}):
        raise TransferEvaluationError("axis_order must name four unique channel/time/target/block axes")
    if not isinstance(meta["array_key"], str) or any(not x for x in meta["array_key"].split(".")):
        raise TransferEvaluationError("array_key must name an audited MAT variable or struct path")
    count = meta["target_count"]
    if type(count) is not int or count != shape[axes.index("target")]:
        raise TransferEvaluationError("target_count does not match audited target axis")
    frequency_set = frequencies(meta["confirmed_frequency_set_hz"], count, rate, "confirmed_frequency_set_hz")
    mapped = meta["target_axis_mapping_confirmed"]
    if type(mapped) is not bool:
        raise TransferEvaluationError("target_axis_mapping_confirmed must be boolean")
    targets = None
    if mapped:
        targets = frequencies(meta["target_frequencies_hz"], count, rate, "target_frequencies_hz")
        if not np.allclose(np.sort(targets), np.sort(frequency_set), rtol=0, atol=1e-10):
            raise TransferEvaluationError("confirmed target mapping must be a permutation of frequency set")
    elif meta["target_frequencies_hz"] is not None:
        raise TransferEvaluationError("unconfirmed target_frequencies_hz must be null; candidate mapping is not labels")
    total = shape[axes.index("time")]
    if total < 500:
        raise TransferEvaluationError("epoch is shorter than one 500-sample inference window")
    spectrum = window_slice(meta["analysis_window_samples"], total, "analysis_window_samples")
    start = sample_index(meta["analysis_start_s"], rate, "analysis_start_s")
    duration = sample_index(meta["analysis_duration_s"], rate, "analysis_duration_s")
    if start != spectrum.start or duration != spectrum.stop - spectrum.start:
        raise TransferEvaluationError("analysis seconds and sample window disagree")
    structure = meta.get("trial_structure", {})
    if not isinstance(structure, dict):
        raise TransferEvaluationError("trial_structure must be an object")
    onset, offset = structure.get("stimulus_onset_in_stored_epoch_s"), structure.get("stimulus_offset_in_stored_epoch_s")
    onset_status = meta.get("stimulus_onset_status", "")
    if not isinstance(onset_status, str):
        raise TransferEvaluationError("stimulus_onset_status must be an audit status string")
    timing = (meta["metadata_status"].startswith("onset_confirmed")
              or onset_status.startswith("confirmed_primary_author"))
    if timing:
        if onset is None or offset is None:
            raise TransferEvaluationError("onset_confirmed requires explicit stored onset/offset evidence")
        a = sample_index(onset, rate, "stimulus_onset_in_stored_epoch_s")
        b = sample_index(offset, rate, "stimulus_offset_in_stored_epoch_s")
        if not 0 <= a < b <= total:
            raise TransferEvaluationError("stimulus timing is outside stored epoch")
        if (spectrum.start, spectrum.stop) != (a, b) and (spectrum.start, spectrum.stop) != (0, total):
            raise TransferEvaluationError("analysis must be exact audited stimulus interval or explicit whole epoch")
    elif (spectrum.start, spectrum.stop) != (0, total):
        raise TransferEvaluationError("timing_not_verified allows only whole-stored-epoch spectral diagnostic")
    cca = None
    if "cca_window_samples" in meta:
        cca = window_slice(meta["cca_window_samples"], total, "cca_window_samples")
        if cca.start < spectrum.start or cca.stop > spectrum.stop:
            raise TransferEvaluationError("CCA interval must lie inside authorized analysis window")
        if cca.stop - cca.start != sample_index(CCA_SECONDS, rate, "CCA duration"):
            raise TransferEvaluationError("metadata CCA interval must explicitly authorize fixed 4 seconds")
    elif duration == sample_index(CCA_SECONDS, rate, "CCA duration"):
        cca = spectrum  # Exact 4s analysis permission, NOT an implicit crop from 5s.
    return {"rate": rate, "frequency_set": frequency_set, "targets": targets,
            "spectral_slice": spectrum, "cca_slice": cca, "timing_verified": timing}


def load_metadata(path: Path, model_rate: float) -> tuple[dict, dict]:
    try:
        meta = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TransferEvaluationError(f"cannot read --metadata-json: {error}") from error
    return meta, validate_metadata(meta, model_rate)


def canonical_epochs(data: np.ndarray, meta: dict) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim != 4 or tuple(data.shape) != tuple(meta["expected_shape"]):
        raise TransferEvaluationError("Dong MAT must match audited 4D shape exactly, no squeeze or guessing")
    if data.dtype.kind not in "iuf" or not np.all(np.isfinite(data)):
        raise TransferEvaluationError("Dong epochs must be finite real signals; no selective trial exclusion")
    return np.transpose(data, [meta["axis_order"].index(a) for a in ("channel", "time", "target", "block")])


def load_dong_epochs(path: Path, meta: dict) -> np.ndarray:
    try:
        value: Any = loadmat(path, squeeze_me=False, struct_as_record=True)
        for key in meta["array_key"].split("."):
            if isinstance(value, np.ndarray) and value.dtype.names and value.size == 1:
                value = value.reshape(-1)[0]  # Unwrap only scalar struct, never signal axes.
            value = value[key]
        return canonical_epochs(value, meta)
    except (OSError, KeyError, TypeError, IndexError, NotImplementedError) as error:
        raise TransferEvaluationError(f"cannot read audited MAT variable from {path}: {error}") from error


def signal_array(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 2 or value.dtype.kind not in "iuf" or not all(value.shape) or not np.all(np.isfinite(value)):
        raise TransferEvaluationError("signal must be finite real [channel,sample]")
    return value.astype(np.float64, copy=False)


@lru_cache(maxsize=512)
def reference_matrix(frequency: float, samples: int, rate: float) -> np.ndarray:
    time = np.arange(samples, dtype=np.float64) / rate
    return np.column_stack([fn(2 * np.pi * h * frequency * time)
                            for h in range(1, HARMONICS + 1) for fn in (np.sin, np.cos)])


@lru_cache(maxsize=512)
def projection(frequency: float, samples: int, rate: float) -> np.ndarray:
    basis = np.column_stack((np.ones(samples), reference_matrix(frequency, samples, rate)))
    if np.linalg.matrix_rank(basis) != 7:
        raise TransferEvaluationError("too few samples for 3-harmonic exact projection")
    return np.linalg.pinv(basis)[1:].T


def exact_frequency_amplitudes(signal: np.ndarray, frequency: float, sample_rate: float) -> np.ndarray:
    """Joint DC+3-harmonic exact sin/cos LS projection => [C,h1..h3]."""
    signal = signal_array(signal)
    rate, frequency = number(sample_rate, "sample_rate", True), number(frequency, "frequency", True)
    if HARMONICS * frequency >= rate / 2:
        raise TransferEvaluationError("projection harmonics must lie below Nyquist")
    coefficients = (signal @ projection(frequency, signal.shape[1], rate)).reshape(len(signal), HARMONICS, 2)
    return np.linalg.norm(coefficients, axis=-1)


def qr_basis(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    q, r = np.linalg.qr(matrix, mode="reduced")
    u, singular, _ = np.linalg.svd(r, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * (singular[0] if singular.size else 0)
    return q @ u[:, :int(np.count_nonzero(singular > tolerance))]


@lru_cache(maxsize=512)
def cca_reference(frequency: float, samples: int, rate: float) -> np.ndarray:
    return qr_basis(reference_matrix(frequency, samples, rate))


def cca_scores(signal: np.ndarray, frequency_values: np.ndarray, sample_rate: float) -> np.ndarray:
    """Fixed centered rank-aware numpy QR/SVD, 3 harmonics; no labels/filtering."""
    signal = signal_array(signal)
    rate = number(sample_rate, "sample_rate", True)
    f = np.asarray(frequency_values)
    if (f.ndim != 1 or f.dtype.kind not in "iuf" or len(f) < 2 or not np.all(np.isfinite(f))
            or np.any(f <= 0) or len(np.unique(f)) != len(f) or np.any(HARMONICS * f >= rate / 2)):
        raise TransferEvaluationError("CCA requires unique valid frequency vector, 3 harmonics below Nyquist")
    if signal.shape[1] < 8:
        raise TransferEvaluationError("CCA requires at least 8 samples")
    data_basis = qr_basis(signal.T)
    scores = []
    for frequency in f:
        refs = cca_reference(float(frequency), signal.shape[1], rate)
        singular = np.linalg.svd(data_basis.T @ refs, compute_uv=False)
        scores.append(float(np.clip(singular[0], 0, 1)) if singular.size else 0.0)
    return np.array(scores)


def cca_top1(epochs: np.ndarray, labels: np.ndarray, frequency_values: np.ndarray, sample_rate: float) -> dict:
    epochs, labels = np.asarray(epochs), np.asarray(labels)
    if epochs.ndim != 3 or not all(epochs.shape):
        raise TransferEvaluationError("CCA epochs must be nonempty [trial,channel,sample]")
    if (labels.shape != (len(epochs),) or labels.dtype.kind not in "iu"
            or np.any(labels < 0) or np.any(labels >= len(frequency_values))):
        raise TransferEvaluationError("CCA labels must be zero-based integer target-axis indices [trial]")
    scores = np.stack([cca_scores(x, frequency_values, sample_rate) for x in epochs])
    predictions = scores.argmax(axis=1)  # Fixed first-index tie; truth never enters scoring.
    return {"accuracy": float(np.mean(predictions == labels)), "predictions": predictions, "scores": scores}


def distribution(values: Any) -> dict:
    values = np.asarray(values, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    return {"count": int(values.size), "defined_count": int(finite.size), "undefined_count": int(values.size - finite.size),
            "mean": float(finite.mean()) if finite.size else None,
            "min": float(finite.min()) if finite.size else None, "max": float(finite.max()) if finite.size else None,
            "quantiles_p0_p5_p25_p50_p75_p95_p100": np.quantile(finite, [0, .05, .25, .5, .75, .95, 1]) if finite.size else None}


def ratios(prediction: np.ndarray, raw: np.ndarray) -> np.ndarray:
    # Undefined near-zero raw amplitudes are counted, never silently selected out.
    return np.divide(prediction, raw, out=np.full_like(raw, np.nan), where=raw > 1e-12)


def binary_mask(value: Any, shape: tuple, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in "biuf" or not np.all((value == 0) | (value == 1)):
        raise TransferEvaluationError(f"{name} must be a binary mask with signal shape")
    return value.astype(bool)


def infer(bundle: Any, raw: np.ndarray) -> Any:
    if bundle.config["window_samples"] != 500 or bundle.config["hop_samples"] != 500:
        raise TransferEvaluationError("requires independent-channel windows500 hop500")
    raw = signal_array(raw)
    support = np.isfinite(raw)
    result = v2.apply_model_to_array(bundle, raw.copy(), valid_mask=support.copy())
    for field in ("denoised", "model_denoised", "artifact_probability", "confidence",
                  "replacement_mask", "valid_mask", "rejected_mask"):
        if np.asarray(getattr(result, field)).shape != raw.shape:
            raise TransferEvaluationError(f"v2 dataclass {field} must have [C,T] input shape")
    if not np.array_equal(binary_mask(result.valid_mask, raw.shape, "valid_mask"), support):
        raise TransferEvaluationError("prediction cannot drop difficult input support")
    for field in ("valid_mask", "replacement_mask", "rejected_mask"):
        if np.asarray(getattr(result, field)).dtype.kind != "b":
            raise TransferEvaluationError(f"v2 dataclass {field} must be boolean for common-support metrics")
    replacement = binary_mask(result.replacement_mask, raw.shape, "replacement_mask")
    rejected = binary_mask(result.rejected_mask, raw.shape, "rejected_mask")
    if np.any(replacement & (~support | rejected)):
        raise TransferEvaluationError("replacement of invalid/rejected samples forbidden")
    for prediction in (result.denoised, result.model_denoised):
        if np.any(support & ~np.isfinite(prediction)):
            raise TransferEvaluationError("nonfinite prediction on common valid support")
    if not np.array_equal(result.denoised[~replacement], raw[~replacement]):
        raise TransferEvaluationError("gated track must retain raw outside replacement_mask")
    if not np.array_equal(result.denoised[replacement], result.model_denoised[replacement]):
        raise TransferEvaluationError("gated replacements must equal candidate")
    return result


def rmse(values: np.ndarray, support: np.ndarray) -> float | None:
    return float(np.sqrt(np.mean(np.square(values[support])))) if np.any(support) else None


def fraction(mask: np.ndarray, support: np.ndarray) -> float | None:
    return float(np.sum(mask & support) / support.sum()) if np.any(support) else None


def change_metrics(raw: np.ndarray, prediction: np.ndarray, support: np.ndarray) -> dict:
    counts = support.sum(axis=-1, keepdims=True)
    means = np.divide(np.where(support, raw, 0).sum(axis=-1, keepdims=True), counts,
                      out=np.zeros_like(counts, dtype=np.float64), where=counts > 0)
    ac_rms = np.sqrt(np.divide(np.where(support, (raw - means) ** 2, 0).sum(axis=-1, keepdims=True), counts,
                              out=np.zeros_like(counts, dtype=np.float64), where=counts > 0))
    change = prediction - raw
    normalized = np.divide(change, ac_rms, out=np.full_like(change, np.nan), where=ac_rms > 1e-12)
    return {"sample_count": int(support.sum()),
            "raw_ac_rms_scope": "per channel, evaluated interval, on the explicitly reported sample support",
            "change_rmse_original_units": rmse(change, support),
            "maximum_absolute_change_original_units": float(np.max(np.abs(change[support]))) if np.any(support) else None,
            "change_rmse_relative_raw_ac_rms": rmse(normalized, support & np.isfinite(normalized)),
            "undefined_ac_rms_sample_count": int(np.sum(support & ~np.isfinite(normalized))),
            "per_channel_change_rmse_relative_raw_ac_rms": [rmse(normalized[c], support[c] & np.isfinite(normalized[c])) for c in range(len(raw))],
            "per_channel_raw_ac_rms": ac_rms.ravel()}



def coverage_metrics(result: Any) -> dict:
    valid = result.valid_mask
    replacement = result.replacement_mask
    rejected = result.rejected_mask
    channels, samples = valid.shape
    full_windows = samples // 500
    blocks = replacement[:, :full_windows * 500].reshape(channels, full_windows, 500)
    return {"valid_sample_count": int(valid.sum()),
            "replaced_sample_count": int((replacement & valid).sum()),
            "rejected_sample_count": int((rejected & valid).sum()),
            "replacement_fraction": fraction(replacement, valid),
            "rejected_fraction": fraction(rejected, valid),
            "channel_full_window_count": channels * full_windows,
            "replacement_channel_window_count": int(blocks.any(axis=-1).sum()),
            "replacement_channel_window_fraction": float(blocks.any(axis=-1).mean()) if full_windows else None,
            "tail_samples_per_channel": samples % 500}


def amplitude_summary(raw: np.ndarray, candidate: np.ndarray, gated: np.ndarray, *, target_labelled: bool = False) -> dict:
    result = {"target_definition": "target == h1" if target_labelled else "frequency-set fundamental, NOT target labels", "harmonics": {}}
    for index, name in ((0, "fundamental"), (0, "h1"), (1, "h2"), (2, "h3")):
        r, c, g = raw[..., index], candidate[..., index], gated[..., index]
        result["harmonics"][name] = {
            "raw_amplitude": distribution(r), "candidate_amplitude": distribution(c), "gated_amplitude": distribution(g),
            "candidate_to_raw_ratio": distribution(ratios(c, r)), "gated_to_raw_ratio": distribution(ratios(g, r)),
            "candidate_ratio_of_means": float(c.mean() / r.mean()) if r.mean() > 1e-12 else None,
            "gated_ratio_of_means": float(g.mean() / r.mean()) if r.mean() > 1e-12 else None}
    if target_labelled:
        result["harmonics"]["target"] = result["harmonics"]["fundamental"]
    return result


def preservation_checks(amplitudes: dict | None, accuracy: dict | None, threshold: float | None) -> dict:
    checks = {"predeclared_thresholds": dict(PRESERVATION_THRESHOLDS),
              "floating_point_comparison_tolerance": NUMERICAL_TOLERANCE,
              "mean_ratio_definition": "arithmetic mean over ALL trial/channel target amplitude ratios; not ratio of means",
              "pass_meaning_only": "transfer preservation evidence; NOT paired denoising efficacy or production",
              "validation_threshold_present": threshold is not None}
    if amplitudes is None or accuracy is None:
        return {**checks, "status": "not_assessed", "passed": None,
                "gated_target_amplitude_pass": None, "gated_cca_drop_pass": None,
                "reason": "true target-axis mapping or authorized 4-second CCA interval not verified"}
    summary = amplitudes["harmonics"]["fundamental"]["gated_to_raw_ratio"]
    mean = summary["mean"]
    amplitude_pass = (mean is not None and summary["undefined_count"] == 0
                      and PRESERVATION_THRESHOLDS["gated_mean_target_amplitude_ratio_min"] - NUMERICAL_TOLERANCE <= mean
                      <= PRESERVATION_THRESHOLDS["gated_mean_target_amplitude_ratio_max"] + NUMERICAL_TOLERANCE)
    drop = accuracy["raw"] - accuracy["gated"]
    cca_pass = drop <= PRESERVATION_THRESHOLDS["gated_cca_accuracy_drop_max"] + NUMERICAL_TOLERANCE
    return {**checks, "status": "assessed", "gated_mean_target_amplitude_ratio": mean,
            "gated_cca_accuracy_drop": drop, "gated_target_amplitude_pass": amplitude_pass,
            "gated_cca_drop_pass": cca_pass, "passed": bool(amplitude_pass and cca_pass and threshold is not None)}


def evaluate_dong(bundle: Any, dong_dir: Path, metadata_path: Path, subjects: list[str]) -> dict:
    if tuple(subjects) != SUBJECTS:
        raise TransferEvaluationError("fixed transfer subjects must be --subjects S2.mat S3.mat, never score-selected")
    meta, spec = load_metadata(metadata_path, bundle.config["sample_rate_hz"])
    mapped = spec["targets"] is not None
    f = spec["targets"] if mapped else spec["frequency_set"]
    records, subject_reports = [], {}
    for subject in subjects:
        path = dong_dir / subject
        data = load_dong_epochs(path, meta)
        subject_rows = []
        for target in range(data.shape[2]):
            for block in range(data.shape[3]):
                # Inference is anchored to the explicitly authorized analysis interval,
                # not an assumed stimulus onset or a window from a previous v1 script.
                raw = data[:, spec["spectral_slice"], target, block].astype(np.float64)
                result = infer(bundle, raw)
                tracks = {"raw": raw, "candidate": result.model_denoised, "gated": result.denoised}
                amp = {name: np.stack([exact_frequency_amplitudes(x, hz, spec["rate"]) for hz in f], axis=1)
                       for name, x in tracks.items()}
                scores = None
                if spec["cca_slice"] is not None:
                    s = spec["cca_slice"]
                    crop = slice(s.start - spec["spectral_slice"].start, s.stop - spec["spectral_slice"].start)
                    scores = {name: cca_scores(x[:, crop], f, spec["rate"]) for name, x in tracks.items()}
                valid = result.valid_mask
                cleanish = valid & ~result.replacement_mask & ~result.rejected_mask
                row = {"subject": subject, "stored_target_axis_index": target, "block_axis_index": block,
                       "target_hz": float(f[target]) if mapped else None,
                       "true_target_label_status": "author_mapping_confirmed" if mapped else "not_assessed",
                       "frequency_set_amplitudes_by_channel_frequency_h1_h2_h3": amp,
                       "frequency_set_ratios_by_channel_frequency_h1_h2_h3": {name: ratios(amp[name], amp["raw"]) for name in ("candidate", "gated")},
                       "target_amplitudes_by_channel_h1_h2_h3": {name: x[:, target] for name, x in amp.items()} if mapped else None,
                       "cca_scores": scores,
                       "cca_predictions": {name: int(x.argmax()) for name, x in scores.items()} if scores is not None else None,
                       "cca_predicted_hz": {name: float(f[x.argmax()]) for name, x in scores.items()} if scores is not None else None,
                       "coverage": coverage_metrics(result),
                       "replacement_fraction": fraction(result.replacement_mask, valid),
                       "cleanish_proxy_sample_count": int(cleanish.sum()),
                       "rejected_fraction": fraction(result.rejected_mask, valid),
                       "raw_change": {name: {"all": change_metrics(raw, x, valid),
                                              "cleanish_gate_proxy": change_metrics(raw, x, cleanish)} for name, x in tracks.items() if name != "raw"},
                       "artifact_probability": distribution(result.artifact_probability),
                       "confidence_not_calibrated": distribution(result.confidence),
                       "inference_diagnostics": result.diagnostics}
                subject_rows.append(row)
        subject_reports[subject] = {"input_sha256": sha256(path), "canonical_shape": data.shape,
                                   "trials": len(subject_rows), "all_trials_included": True,
                                   **summarize_dong_rows(subject_rows, f, mapped)}
        records.extend(subject_rows)
    all_summary = summarize_dong_rows(records, f, mapped)
    return {"diagnostic_name": "dong2023_transfer_preservation", "subjects": subjects,
            "subject_selection": "fixed_S2_S3_before_scores_not_performance_selected",
            "metadata": meta, "metadata_sha256": sha256(metadata_path),
            "target_axis_mapping_confirmed": mapped,
            "timing_status": "timing_verified" if spec["timing_verified"] else "timing_not_verified",
            "analysis_scope": "whole_stored_epoch_spectral_diagnostic_includes_pre_post_stimulus" if spec["spectral_slice"].start == 0 and spec["spectral_slice"].stop == meta["expected_shape"][meta["axis_order"].index("time")] else "audited_stimulus_interval_preservation_diagnostic",
            "downstream_stimulus_recognition_acceptance": False, "paired_denoising_efficacy_demonstrated": False,
            "production_validated": False, "all_trials_and_channels_reported": True,
            "frequency_set_hz_in_score_order": f, "candidate_mapping_used_as_labels": False,
            "signal_units": meta.get("signal_unit"),
            "signal_units_status": meta.get("signal_unit_status", "not_verified; amplitudes in stored data units"),
            "cleanish_definition": "accepted nonreplaced samples; gate proxy, NOT known clean EEG",
            "inference_scope": "each authorized analysis epoch/channel independent; windows500 hop500; tail retained raw",
            "cca_method": {"implementation": "numpy centered rank-aware QR/SVD", "harmonics": 3,
                           "duration_seconds": 4 if spec["cca_slice"] is not None else None,
                           "filtering": "none", "test_tuned": False, "tie_rule": "first_frequency_in_report_order"},
            **all_summary, "subject_reports": subject_reports, "epoch_reports": records,
            "preservation_checks": preservation_checks(all_summary["target_labelled_amplitude"],
                                                       all_summary["cca_top1_accuracy"], bundle.validation_threshold)}


def summarize_dong_rows(rows: list[dict], f: np.ndarray, mapped: bool) -> dict:
    # Summary and detailed arrays cover ALL channels/trials, not replaced-only results.
    arrays = {name: np.stack([r["frequency_set_amplitudes_by_channel_frequency_h1_h2_h3"][name] for r in rows])
              for name in ("raw", "candidate", "gated")}
    spectrum = amplitude_summary(arrays["raw"], arrays["candidate"], arrays["gated"])
    spectrum["per_frequency"] = {str(float(hz)): amplitude_summary(arrays["raw"][:, :, i], arrays["candidate"][:, :, i], arrays["gated"][:, :, i]) for i, hz in enumerate(f)}
    target_summary = None
    if mapped:
        target_arrays = {name: np.stack([r["target_amplitudes_by_channel_h1_h2_h3"][name] for r in rows]) for name in arrays}
        target_summary = amplitude_summary(target_arrays["raw"], target_arrays["candidate"], target_arrays["gated"], target_labelled=True)
    accuracy, agreement = None, None
    if rows[0]["cca_predictions"] is not None:
        agreement = {name: float(np.mean([r["cca_predictions"][name] == r["cca_predictions"]["raw"] for r in rows])) for name in ("candidate", "gated")}
        if mapped:
            accuracy = {name: float(np.mean([r["cca_predictions"][name] == r["stored_target_axis_index"] for r in rows])) for name in arrays}
    return {"frequency_set_spectral_preservation": spectrum,
            "target_labelled_amplitude_status": "assessed" if mapped else "not_assessed",
            "target_labelled_amplitude": target_summary,
            "cca_accuracy_status": "assessed" if accuracy is not None else "not_assessed",
            "cca_top1_accuracy": accuracy,
            "cca_prediction_agreement_status": "assessed" if agreement is not None else "not_assessed",
            "cca_raw_prediction_agreement": agreement,
            "cca_agreement_interpretation": "prediction stability, NOT correctness, NOT target labels",
            "replacement_fraction_distribution": distribution([r["replacement_fraction"] for r in rows]),
            "replacement_channel_window_fraction_distribution": distribution([r["coverage"]["replacement_channel_window_fraction"] for r in rows]),
            "raw_change_all_relative_ac_rms": {name: distribution([r["raw_change"][name]["all"]["change_rmse_relative_raw_ac_rms"] for r in rows]) for name in ("candidate", "gated")},
            "raw_change_cleanish_relative_ac_rms": {name: distribution([r["raw_change"][name]["cleanish_gate_proxy"]["change_rmse_relative_raw_ac_rms"] for r in rows]) for name in ("candidate", "gated")}}


def load_regression(path: Path, session_id: str, model_rate: float) -> dict:
    if session_id != REGRESSION_SESSION:
        raise TransferEvaluationError(f"known-v1 regression session must be {REGRESSION_SESSION}")
    with np.load(path, allow_pickle=False) as archive:
        keys = ("noisy", "clean", "artifact_mask", "session_id", "sample_rate_hz")
        missing = set(keys) - set(archive.files)
        if missing:
            raise TransferEvaluationError(f"v1 NPZ missing fields: {sorted(missing)}")
        sessions = archive["session_id"]
        if sessions.ndim != 1 or sessions.dtype.kind not in "US":
            raise TransferEvaluationError("regression session_id must be a 1D string vector")
        indices = np.flatnonzero(sessions.astype(str) == session_id)
        if len(indices) != 26:
            raise TransferEvaluationError("known_v1_regression_stress_test requires all 26 known windows")
        noisy, clean, artifacts = (archive[k] for k in ("noisy", "clean", "artifact_mask"))
        if (noisy.ndim != 3 or noisy.shape[0] != len(sessions) or not all(noisy.shape)
                or noisy.shape[2] < 500 or clean.shape != noisy.shape or artifacts.shape != noisy.shape
                or noisy.dtype.kind not in "iuf" or clean.dtype.kind not in "iuf"):
            raise TransferEvaluationError("regression arrays require matching real [window,channel,sample] shapes")
        rates = archive["sample_rate_hz"]
        if rates.shape != (len(sessions),) or rates.dtype.kind not in "iuf" or not np.all(rates[indices] == model_rate):
            raise TransferEvaluationError("regression sample_rate_hz must match model on all 26 windows")
        noisy, clean = noisy[indices].astype(np.float64), clean[indices].astype(np.float64)
        if not np.all(np.isfinite(noisy)) or not np.all(np.isfinite(clean)):
            raise TransferEvaluationError("regression inputs/targets must be finite; cannot exclude stress samples")
        artifacts = binary_mask(artifacts[indices], noisy.shape, "artifact_mask")
    return {"noisy": noisy, "clean": clean, "artifact_mask": artifacts, "indices": indices}


def evaluate_regression(bundle: Any, path: Path, session_id: str) -> dict:
    data = load_regression(path, session_id, bundle.config["sample_rate_hz"])
    rows, paired = [], {name: [] for name in ("noisy", "candidate", "gated")}
    for index, noisy, clean, artifacts in zip(data["indices"], data["noisy"], data["clean"], data["artifact_mask"]):
        result = infer(bundle, noisy)
        valid, cleanish = result.valid_mask, result.valid_mask & ~artifacts
        tracks = {"noisy": noisy, "candidate": result.model_denoised, "gated": result.denoised}
        metrics = {}
        for name, prediction in tracks.items():
            error = prediction - clean
            paired[name].append(error)
            metrics[name] = {"rmse_vs_semisynthetic_target_original_units": rmse(error, valid),
                             "cleanish_rmse_vs_target_original_units": rmse(error, cleanish),
                             "raw_change_all": change_metrics(noisy, prediction, valid),
                             "raw_change_cleanish": change_metrics(noisy, prediction, cleanish)}
        rows.append({"dataset_window_index": int(index), "metrics": metrics,
                     "coverage": coverage_metrics(result),
                     "replacement_fraction": fraction(result.replacement_mask, valid),
                     "cleanish_replacement_fraction": fraction(result.replacement_mask, cleanish),
                     "rejected_fraction": fraction(result.rejected_mask, valid),
                     "inference_diagnostics": result.diagnostics})
    summary = {}
    for name, errors in paired.items():
        summary[name] = {"rmse_vs_semisynthetic_target_original_units": float(np.sqrt(np.mean(np.square(np.stack(errors))))),
                         "per_window_rmse_distribution": distribution([r["metrics"][name]["rmse_vs_semisynthetic_target_original_units"] for r in rows]),
                         "cleanish_rmse_vs_target_distribution": distribution([r["metrics"][name]["cleanish_rmse_vs_target_original_units"] for r in rows]),
                         "cleanish_raw_change_relative_ac_rms_distribution": distribution([r["metrics"][name]["raw_change_cleanish"]["change_rmse_relative_raw_ac_rms"] for r in rows]),
                         "maximum_absolute_change_original_units": max(r["metrics"][name]["raw_change_all"]["maximum_absolute_change_original_units"] for r in rows)}
    return {"diagnostic_name": "known_v1_regression_stress_test", "independent": False, "heldout": False,
            "seen_by_v1_experiments": True, "session_id": session_id, "dataset_sha256": sha256(path),
            "target_provenance": "semi_synthetic_clean_ish_NOT_real_noise_paired_ground_truth",
            "warning": "Known v1 stress regression only, NOT independent test or production acceptance. Do not tune v2 on these results.",
            "test_tuned": False, "training_performed": False, "window_count": 26, "window_indices": data["indices"],
            "input_shape": data["noisy"].shape, "sample_rate_hz": bundle.config["sample_rate_hz"],
            "cleanish_definition": "artifact_mask==0 in v1 semi-synthetic annotation",
            "summary": summary, "replacement_fraction_distribution": distribution([r["replacement_fraction"] for r in rows]),
            "cleanish_replacement_fraction_distribution": distribution([r["cleanish_replacement_fraction"] for r in rows]),
            "replacement_channel_window_fraction_distribution": distribution([r["coverage"]["replacement_channel_window_fraction"] for r in rows]),
            "window_reports": rows, "production_validated": False, "acceptance_passed": False}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_report(output: Path, report: dict) -> None:
    """Exclusive create: there is intentionally no --overwrite option."""
    payload = json.dumps(jsonable(report), ensure_ascii=False, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload + "\n")


def positive_threads(value: str) -> int:
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--threads must be a positive integer") from error
    if count <= 0:
        raise argparse.ArgumentTypeError("--threads must be a positive integer")
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dong-dir", type=Path)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    parser.add_argument("--regression-dataset", type=Path)
    parser.add_argument("--regression-session-id", default=REGRESSION_SESSION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=positive_threads, default=2,
                        help="PyTorch CPU intra-op threads (positive integer, default 2); no model/CCA change")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.dong_dir is None and args.regression_dataset is None:
            raise TransferEvaluationError("request --dong-dir and/or --regression-dataset")
        if (args.dong_dir is None) != (args.metadata_json is None):
            raise TransferEvaluationError("Dong evaluation requires both --dong-dir and --metadata-json")
        if args.output.exists():
            raise TransferEvaluationError("output already exists; choose a new path (old reports never overwritten)")
        inputs = [args.model, args.config, args.metadata_json, args.regression_dataset]
        if args.dong_dir is not None:
            inputs += [args.dong_dir / s for s in SUBJECTS]
            if tuple(args.subjects) != SUBJECTS:
                raise TransferEvaluationError("fixed transfer subjects must be --subjects S2.mat S3.mat")
        if any(p is not None and args.output.resolve() == p.resolve() for p in inputs):
            raise TransferEvaluationError("output must not alias raw/config/model input")
        config = v2.load_config(args.config)
        if config["window_samples"] != 500 or config["hop_samples"] != 500:
            raise TransferEvaluationError("required windows500 hop500")
        # Fail closed on evidence problems BEFORE loading model or running inference.
        if args.dong_dir is not None:
            load_metadata(args.metadata_json, config["sample_rate_hz"])
        if args.regression_dataset is not None:
            load_regression(args.regression_dataset, args.regression_session_id, config["sample_rate_hz"])
        if v2.torch is None:
            raise TransferEvaluationError("optional PyTorch is required for v2 inference")
        v2.torch.set_num_threads(args.threads)
        bundle = v2.load_model(args.model, config, device=args.device)
        report = {"schema_version": 1, "model_version": "learned_denoise_v2",
                  "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "model_sha256": sha256(args.model), "config_sha256": sha256(args.config),
                  "frozen_validation_threshold": bundle.validation_threshold,
                  "execution": {"torch_cpu_threads": args.threads, "device": args.device,
                                "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
                                "mkl_num_threads_env": os.environ.get("MKL_NUM_THREADS")},
                  "threshold_source": "loaded checkpoint validation only; never refitted on transfer/regression",
                  "training_performed": False, "test_tuned": False, "production_validated": False,
                  "paired_denoising_efficacy_demonstrated": False, "release_gate_passed": False,
                  "sections": {}}
        if args.dong_dir is not None:
            report["sections"]["dong2023_transfer_preservation"] = evaluate_dong(bundle, args.dong_dir, args.metadata_json, args.subjects)
        if args.regression_dataset is not None:
            report["sections"]["known_v1_regression_stress_test"] = evaluate_regression(bundle, args.regression_dataset, args.regression_session_id)
        write_report(args.output, report)
    except (TransferEvaluationError, v2.LearnedDenoiseV2Error, OSError, ValueError) as error:
        parser.exit(2, f"transfer/regression evaluation refused: {error}\n")
    print(f"Diagnostic report: {args.output.resolve()} (not production acceptance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

