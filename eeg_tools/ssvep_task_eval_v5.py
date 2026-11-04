"""Frozen-window SSVEP task reference evaluator (no training or denoising claims).

The only default reference is REFERENCE_RELATIVE with FROZEN_MANIFEST_SHA256.
A candidate directory has manifest.json/materialized_subjects just like the
reference. Its root manifest must bind reference_manifest_sha256 to the frozen
root hash. Each subject sidecar binds reference_npz_sha256 and
reference_manifest_sha256 (the reference subject sidecar hash), declares
signal_key and signal_shape, and copies the reference identity, timing, units,
channel/frequency/phase/trial metadata. Its own NPZ/sidecar hashes are in its
catalog row. NPZ labels must exactly match the reference arrays; processed is
the default signal key. Do not copy clean-ground-truth-named reference fields
into candidates: all candidate metadata, paths and member names are screened.

check_only reads JSON metadata and file identities only, never NPZ bytes or
arrays. Consequently NPZ hashes and per-value trial alignment are deferred and
explicitly reported as unverified. Full evaluation reads only the selected
train/validation split, one subject/candidate at a time, with allow_pickle=False.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
from typing import Any
import zipfile

import numpy as np

from eeg_tools import ssvep_reference_v5 as reference_safety

WORKSPACE = Path(__file__).resolve().parents[1]
REFERENCE_RELATIVE = Path("datasets/processed/ssvep_reference_v5_20261007")
REFERENCE_VERSION = "ssvep_reference_v5"
FROZEN_MANIFEST_SHA256 = "607617315889697a5bfcea986bf055bf4ed31182e8a69776842c736f481604b6"
WINDOW_LENGTHS = {"Nakanishi2015": 1024, "Wearable": 500}
LABEL_KEYS = ("trial_index", "target_id", "block_id", "electrode_id", "frequency_hz")
BINDING_FIELDS = (
    "dataset", "subject_id", "group_key", "split", "native_sample_rate_hz",
    "onset_sample_zero_based", "frequency_order_hz", "phase_pi_multiples",
    "channel_order", "native_units", "output_axes", "trial_order", "electrode_order",
)
EPSILON = 1e-12
MAX_ARRAY_BYTES = 256 * 1024 * 1024


class SSVEPTaskEvalV5Error(ValueError):
    """The input or report failed the strict task-evaluation contract."""


def _fail(message: str) -> None:
    raise SSVEPTaskEvalV5Error(message)


def _inside(workspace: Path, value: str | Path) -> Path:
    # Reject lexical traversal as well as resolved escapes, including Windows
    # drive/UNC syntax when tests run on another platform.
    text = str(value)
    if ".." in text.replace("\\", "/").split("/"):
        _fail("path traversal refused")
    win = PureWindowsPath(text)
    if win.drive.casefold() == "d:":
        _fail("D: source paths are forbidden")
    if win.drive and not Path(text).is_absolute():
        _fail("foreign drive/UNC path refused")
    try:
        return reference_safety._inside(workspace, value)
    except reference_safety.SSVEPReferenceV5Error as exc:
        raise SSVEPTaskEvalV5Error(str(exc)) from exc


def _json(path: Path) -> dict:
    try:
        return reference_safety._json(path)
    except (ValueError, OSError, UnicodeError) as exc:
        raise SSVEPTaskEvalV5Error(f"invalid metadata: {path.name}: {exc}") from exc


def _sha(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        _fail("lowercase SHA256 binding required")
    return value


def _stamp(path: Path) -> tuple:
    try:
        reference_safety._guard_chain(path)
        st = path.stat()
        if not path.is_file():
            _fail("input must be a regular file")
        return st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_dev, st.st_ino
    except (OSError, reference_safety.SSVEPReferenceV5Error) as exc:
        raise SSVEPTaskEvalV5Error(f"unsafe/missing input: {path}") from exc


def _hash(path: Path) -> str:
    before = _stamp(path)
    digest = reference_safety.sha256(path)
    if _stamp(path) != before:
        _fail("input changed while hashing")
    return digest


def _dangerous(text: str) -> bool:
    # Separator-insensitive: groundTruth, ground-truth and target_signal all
    # count as forbidden candidate provenance, including nested JSON keys.
    normalized = re.sub(r"[^a-z0-9]", "", text.casefold())
    return any(token in normalized for token in ("clean", "groundtruth", "targetsignal"))


def _candidate_names(value: Any) -> None:
    if isinstance(value, str) and _dangerous(value):
        _fail("candidate clean/ground_truth/target_signal naming refused")
    if isinstance(value, list):
        for item in value:
            _candidate_names(item)
    if isinstance(value, dict):
        for key, item in value.items():
            _candidate_names(key)
            _candidate_names(item)


def _inventory(workspace: Path, directory: Path, candidate: bool) -> set[Path]:
    """Inventory without following links; never opens NPZ waveform contents."""
    found = set()
    def walk_error(error: OSError) -> None:
        raise SSVEPTaskEvalV5Error("input inventory is unreadable") from error
    for current, dirs, files in os.walk(directory, followlinks=False, onerror=walk_error):
        for name in dirs + files:
            path = _inside(workspace, Path(current) / name)
            if candidate:
                _candidate_names(path.relative_to(directory).as_posix())
            if path.suffix.casefold() == ".npz":
                # Even an unlisted/renamed holdout artifact is rejected before
                # array loading. Unknown names are rejected later by catalog.
                match = re.fullmatch(r"subject_(\d+)\.npz", name, re.IGNORECASE)
                if "holdout" in path.relative_to(directory).as_posix().casefold() or (match and int(match[1]) >= 9):
                    _fail("holdout NPZ refused")
                found.add(path)
    return found


def _identity(row: dict) -> tuple[str, int]:
    dataset, sid, split = row.get("dataset"), row.get("subject_id"), row.get("split")
    if dataset not in WINDOW_LENGTHS or type(sid) is not int:
        _fail("unknown dataset/subject identity")
    if split == "holdout" or sid not in range(1, 9):
        _fail("holdout/unfrozen subject refused")
    expected_split = "train" if sid <= 6 else "validation"
    if split != expected_split:
        _fail("subject split differs from frozen partition")
    return dataset, sid


def _shape(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3 or any(type(x) is not int or x <= 0 for x in value):
        _fail("signal shape must contain three positive integer dimensions")
    if math.prod(value) * 8 > MAX_ARRAY_BYTES:
        _fail("subject exceeds bounded array budget")
    return tuple(value)


def _window_metadata(side: dict, shape: tuple) -> None:
    dataset = side["dataset"]
    spec = reference_safety.SPECS[dataset]
    if side.get("native_sample_rate_hz") != spec["sample_rate_hz"]:
        _fail("native sampling rate differs from frozen source")
    if type(side.get("onset_sample_zero_based")) is not int or side["onset_sample_zero_based"] != spec["onset_sample_zero_based"]:
        _fail("onset differs from frozen source")
    if side["onset_sample_zero_based"] + WINDOW_LENGTHS[dataset] > shape[1]:
        _fail("source stimulation window is out of bounds")
    if side.get("frequency_order_hz") != spec["frequency_order_hz"]:
        _fail("frequency order differs from frozen source")
    if side.get("phase_pi_multiples") != spec["phase_pi_multiples"]:
        _fail("phase order differs from frozen source")
    if side.get("output_axes") != ["trial", "time", "channel"]:
        _fail("output axes must be trial/time/channel")
    if "native_units" not in side:
        _fail("native_units must be declared, including null for unknown")
    if side["native_units"] is not None and not isinstance(side["native_units"], str):
        _fail("invalid native units metadata")
    if type(side.get("materialized_trials")) is not int or side["materialized_trials"] != shape[0]:
        _fail("trial count differs from subject shape")


def _catalog(workspace: Path, directory: Path, candidate: bool, inventory: set[Path]) -> dict:
    manifest_path = directory / "manifest.json"
    manifest_hash = _hash(manifest_path)
    if not candidate and manifest_hash != FROZEN_MANIFEST_SHA256:
        _fail("frozen reference manifest SHA mismatch")
    manifest = _json(manifest_path)
    if candidate:
        _candidate_names(manifest)
        if manifest.get("reference_manifest_sha256") != FROZEN_MANIFEST_SHA256:
            _fail("candidate root must bind frozen reference manifest SHA")
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or manifest.get("reference_version") != REFERENCE_VERSION
            or manifest.get("status") != "formal" or manifest.get("build_completed") is not True):
        _fail("unfrozen reference_version/schema/status/build")
    rows = manifest.get("materialized_subjects")
    if not isinstance(rows, list) or not rows:
        _fail("materialized_subjects must be a nonempty list")
    subjects, tracked = {}, {manifest_path: (_stamp(manifest_path), manifest_hash)}
    for row in rows:
        if not isinstance(row, dict):
            _fail("invalid subject row")
        key = _identity(row)
        if key in subjects:
            _fail("duplicate subject")
        relative = row.get("relative_path")
        canonical = f"{key[0]}/subject_{key[1]:03d}.npz"
        if relative != canonical:
            _fail("subject relative path differs from frozen identity")
        npz = _inside(workspace, directory / relative)
        sidepath = _inside(workspace, Path(str(npz) + ".manifest.json"))
        side_hash = _hash(sidepath)
        if side_hash != _sha(row.get("manifest_sha256")):
            _fail("subject sidecar SHA mismatch")
        side = _json(sidepath)
        if candidate:
            _candidate_names(side)
        if _identity(side) != key or side.get("split") != row["split"]:
            _fail("sidecar subject/split binding mismatch")
        if side.get("reference_version") != REFERENCE_VERSION or side.get("status") != "formal" or type(side.get("schema_version")) is not int or side["schema_version"] != 1:
            _fail("sidecar version/status mismatch")
        if side.get("group_key") != f"{key[0]}:subject_{key[1]:03d}":
            _fail("subject group key mismatch")
        if side.get("npz_relative_path") != npz.name:
            _fail("sidecar NPZ filename mismatch")
        if _sha(side.get("npz_sha256")) != _sha(row.get("npz_sha256")):
            _fail("NPZ SHA declarations disagree")
        shape = _shape(side.get("signal_shape") if candidate else side.get("reference_shape"))
        _window_metadata(side, shape)
        if type(row.get("trials")) is not int or row["trials"] != shape[0]:
            _fail("catalog trial count mismatch")
        if not candidate and (side.get("reference_dtype") != "float64" or side.get("raw_reference_is_clean_ground_truth") is not False):
            _fail("reference must be float64 task reference, not clean ground truth")
        tracked[sidepath] = (_stamp(sidepath), side_hash)
        tracked[npz] = (_stamp(npz), None)  # no NPZ byte reads in check-only
        subjects[key] = {"side": side, "shape": shape, "path": npz,
                         "side_hash": side_hash, "npz_hash": row["npz_sha256"]}
    if set(item["path"] for item in subjects.values()) != inventory:
        _fail("unlisted or missing subject NPZ; holdout/foreign artifacts refused")
    if not candidate:
        if (manifest.get("holdout_waveform_loaded") is not False or manifest.get("holdout_materialized") is not False
                or manifest.get("raw_reference_is_clean_ground_truth") is not False
                or manifest.get("model_metrics_computed") is not False
                or manifest.get("runtime") != {"enabled": False, "production_validated": False}):
            _fail("reference manifest safety flags are not false")
    return {"manifest_hash": manifest_hash, "subjects": subjects, "tracked": tracked,
            "directory": directory, "candidate": candidate}


def _bind_candidate(candidate: dict, baseline: dict, signal_key: str) -> None:
    if set(candidate["subjects"]) != set(baseline["subjects"]):
        _fail("candidate subject set differs from frozen reference")
    for key, item in candidate["subjects"].items():
        ref = baseline["subjects"][key]
        side = item["side"]
        if item["shape"] != ref["shape"]:
            _fail("candidate shape binding differs from reference")
        for field in BINDING_FIELDS:
            if field not in side or side[field] != ref["side"].get(field):
                _fail(f"candidate metadata binding mismatch: {field}")
        if (side.get("reference_npz_sha256") != ref["npz_hash"]
                or side.get("reference_manifest_sha256") != ref["side_hash"]):
            _fail("candidate reference subject SHA binding mismatch")
        if side.get("signal_key") != signal_key:
            _fail("candidate sidecar signal_key mismatch")


def _verify_unchanged(workspace: Path, catalog: dict) -> None:
    for file, (stamp, digest) in catalog["tracked"].items():
        if _stamp(file) != stamp or (digest is not None and _hash(file) != digest):
            _fail("input changed during evaluation")
    if _inventory(workspace, catalog["directory"], catalog["candidate"]) != set(item["path"] for item in catalog["subjects"].values()):
        _fail("NPZ inventory changed during evaluation")


def _read_subject(catalog: dict, item: dict, signal_key: str) -> dict[str, np.ndarray]:
    path = item["path"]
    if _stamp(path) != catalog["tracked"][path][0] or _hash(path) != item["npz_hash"]:
        _fail("subject NPZ SHA/stat mismatch")
    catalog["tracked"][path] = (_stamp(path), item["npz_hash"])
    required = set(LABEL_KEYS) | {signal_key}
    if item["side"]["phase_pi_multiples"] is not None:
        required.add("phase_pi_multiple")
    if not catalog["candidate"]:
        required.add("manifest_json")
    try:
        # Bound uncompressed NPY shapes before numpy allocates any array.
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or any("/" in n or "\\" in n or not n.endswith(".npy") for n in names):
                _fail("duplicate/unsafe NPZ members")
            keys = {n[:-4] for n in names}
            if catalog["candidate"]:
                _candidate_names(sorted(keys))
            if not required.issubset(keys) or keys - (required | {"manifest_json"}):
                _fail("NPZ member schema mismatch")
            declared = item["side"].get("npz_members")
            if declared is not None and sorted(keys) != sorted(declared):
                _fail("sidecar NPZ member declarations mismatch")
            total = 0
            for info in infos:
                with archive.open(info) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        _fail("unsupported NPY format version")
                    if dtype.hasobject or dtype.fields is not None:
                        _fail("object/pickle/structured NPZ arrays refused")
                    size = math.prod(shape) * dtype.itemsize
                    if size > MAX_ARRAY_BYTES or any(x < 0 for x in shape):
                        _fail("NPY header exceeds subject array budget")
                    if info.filename == "manifest_json.npy":
                        if shape != () or dtype.kind not in "US" or size > 256 * 1024:
                            _fail("embedded manifest must be a bounded scalar string")
                    elif info.filename == signal_key + ".npy":
                        if shape != item["shape"] or dtype.kind not in "fiu":
                            _fail("signal shape/real numeric dtype mismatch")
                        if not catalog["candidate"] and dtype != np.dtype("float64"):
                            _fail("reference dtype must be float64")
                    elif shape != (item["shape"][0],):
                        _fail("label shape mismatch")
                    if stream.tell() + size != info.file_size:
                        _fail("NPY byte length/header mismatch")
                    total += size
            if total > MAX_ARRAY_BYTES:
                _fail("NPZ exceeds subject allocation budget")
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in keys if key != "manifest_json"}
            if "manifest_json" in keys:
                embedded = json.loads(str(loaded["manifest_json"].item()))
                if catalog["candidate"]:
                    _candidate_names(embedded)
                expected = {k: v for k, v in item["side"].items() if k not in ("npz_sha256", "npz_relative_path")}
                if embedded != expected:
                    _fail("embedded manifest differs from bound sidecar")
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile, EOFError) as exc:
        if isinstance(exc, SSVEPTaskEvalV5Error):
            raise
        raise SSVEPTaskEvalV5Error(f"invalid NPZ: {path.name}: {exc}") from exc
    if _stamp(path) != catalog["tracked"][path][0]:
        _fail("NPZ changed while reading")
    for key, array in arrays.items():
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            _fail(f"non-finite or nonnumeric array: {key}")
    if not catalog["candidate"]:
        _validate_labels(arrays, item["side"])
        if reference_safety._array_sha(np.ascontiguousarray(arrays["reference"])) != _sha(item["side"].get("reference_sha256")):
            _fail("reference array SHA mismatch")
    return arrays


def _validate_labels(arrays: dict, side: dict) -> None:
    count = side["materialized_trials"]
    for key in LABEL_KEYS[:4]:
        if arrays[key].dtype.kind not in "iu":
            _fail("integer trial labels required")
    target = arrays["target_id"]
    order = np.asarray(side["frequency_order_hz"], dtype=np.float64)
    if (not np.array_equal(arrays["trial_index"], np.arange(count))
            or np.any(target < 0) or np.any(target >= len(order))):
        _fail("trial_index/target_id invalid")
    if not np.array_equal(arrays["frequency_hz"], order[target]):
        _fail("frequency labels differ from fixed target order")
    blocks = 15 if side["dataset"] == "Nakanishi2015" else 10
    if np.any(arrays["block_id"] < 0) or np.any(arrays["block_id"] >= blocks):
        _fail("block labels out of source bounds")
    if side["dataset"] == "Nakanishi2015":
        if not np.all(arrays["electrode_id"] == -1):
            _fail("Nakanishi electrode labels must remain unknown (-1)")
    else:
        if not np.isin(arrays["electrode_id"], [0, 1]).all():
            _fail("Wearable electrode domain labels invalid")
        expected = np.asarray(side["phase_pi_multiples"])[target]
        if not np.array_equal(arrays["phase_pi_multiple"], expected):
            _fail("phase labels differ from fixed target order")


def _aligned(candidate: dict, baseline: dict) -> None:
    keys = set(LABEL_KEYS) | ({"phase_pi_multiple"} if "phase_pi_multiple" in baseline else set())
    for key in keys:
        a, b = candidate[key], baseline[key]
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            _fail(f"candidate relabel/unaligned trial values: {key}")


def spectral_decode(signal: np.ndarray, side: dict) -> dict:
    """Fixed, label-independent predictions, one trial FFT at a time.

    Tied nearest bins and tied decoder scores choose the first (lowest index).
    Power is squared FFT magnitude / sum(Hann**2), averaged across channels.
    No frequencies, latency, window lengths or thresholds are fitted to data.
    """
    shape = _shape(list(signal.shape))
    _window_metadata(side, shape)
    if signal.dtype.kind not in "fiu" or not np.isfinite(signal).all():
        _fail("decoder input must be finite real trial/time/channel signal")
    length = WINDOW_LENGTHS[side["dataset"]]
    start = side["onset_sample_zero_based"]
    fft_hz = np.fft.rfftfreq(length, d=1.0 / side["native_sample_rate_hz"])
    frequencies = np.asarray(side["frequency_order_hz"], dtype=np.float64)
    fundamental = np.abs(fft_hz[:, None] - frequencies).argmin(axis=0)
    harmonic = np.abs(fft_hz[:, None] - 2 * frequencies).argmin(axis=0)
    if np.any(frequencies <= 0) or np.any(2 * frequencies > fft_hz[-1]):
        _fail("target/harmonic exceeds valid FFT frequency range")
    hann = np.hanning(length)
    scores = np.empty((shape[0], len(frequencies)), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for index in range(shape[0]):
            segment = signal[index, start:start + length, :].astype(np.float64, copy=True)
            segment -= segment.mean(axis=0, keepdims=True)
            segment *= hann[:, None]
            power = np.abs(np.fft.rfft(segment, axis=0)) ** 2 / np.sum(hann ** 2)
            scores[index] = power[fundamental].mean(axis=1) + .5 * power[harmonic].mean(axis=1)
    if not np.isfinite(scores).all():
        _fail("spectral score overflow/non-finite result")
    predictions = scores.argmax(axis=1)
    sorted_scores = np.sort(scores, axis=1)
    confidence = (sorted_scores[:, -1] - sorted_scores[:, -2]) / (sorted_scores[:, -1] + EPSILON)
    return {"scores": scores, "predictions": predictions, "confidence": confidence,
            "fundamental_bins": fundamental.tolist(), "second_harmonic_bins": harmonic.tolist()}


def _trial_metrics(decoded: dict, labels: np.ndarray) -> dict:
    scores = decoded["scores"]
    target_power = scores[np.arange(len(labels)), labels]
    correct = decoded["predictions"] == labels
    with np.errstate(over="ignore", invalid="ignore"):
        total_power = scores.sum(axis=1)
    if not np.isfinite(total_power).all():
        _fail("total spectral score overflow")
    relative = target_power / (total_power + EPSILON)
    if not np.isfinite(relative).all():
        _fail("relative spectral score overflow")
    return {"correct": correct, "confidence": decoded["confidence"], "relative": relative,
            "target_power": target_power, "labels": labels}


def _metric_summary(trials: dict, mask: np.ndarray | None = None) -> dict:
    selected = {key: value if mask is None else value[mask] for key, value in trials.items()}
    labels = selected["labels"]
    correct = selected["correct"]
    per_target = {str(int(t)): float(correct[labels == t].mean()) for t in np.unique(labels)}
    return {"trials": int(len(labels)), "correct_trials": int(correct.sum()),
            "accuracy": float(correct.mean()),
            "macro_target_accuracy": math.fsum(per_target.values()) / len(per_target),
            "per_target_accuracy": per_target,
            "mean_confidence": float(selected["confidence"].mean()),
            "mean_relative_score": float(selected["relative"].mean()),
            "mean_target_frequency_power": math.fsum(float(x) / len(labels) for x in selected["target_power"])}


def _ratio(candidate: np.ndarray, baseline: np.ndarray) -> dict:
    # Zero baseline energy has no defined multiplicative ratio. Do not silently
    # manufacture efficacy by dividing it by epsilon.
    valid = baseline > EPSILON
    with np.errstate(over="ignore", invalid="ignore"):
        values = candidate[valid] / baseline[valid]
    if not np.isfinite(values).all():
        _fail("spectral ratio overflow")
    per_trial = [None] * len(baseline)
    for index, value in zip(np.flatnonzero(valid), values):
        per_trial[int(index)] = float(value)
    return {"per_trial": per_trial, "defined_trials": int(valid.sum()),
            "undefined_trials": int((~valid).sum()),
            "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "minimum": float(values.min()) if len(values) else None,
            "maximum": float(values.max()) if len(values) else None}


def _accuracy_aggregate(metrics: list[dict]) -> dict:
    """Dimensionless accuracy only: safe to pool across datasets/domains."""
    total = sum(row["trials"] for row in metrics)
    if not metrics or total <= 0:
        _fail("cannot aggregate empty accuracy metrics")
    # Actual subject metrics contain integer correct counts. The fallback keeps
    # this helper usable with manually supplied summaries without rounding them.
    correct = math.fsum(row.get("correct_trials", row["accuracy"] * row["trials"]) for row in metrics)
    micro = correct / total
    return {"subjects": len(metrics), "trials": total,
            "macro_accuracy": math.fsum(row["accuracy"] / len(metrics) for row in metrics),
            "micro_accuracy": micro, "accuracy": micro}


def _aggregate(metrics: list[dict]) -> dict:
    # Called for a single dataset/domain, never to pool unknown native power
    # units across datasets. accuracy is the backward-compatible micro alias.
    result = _accuracy_aggregate(metrics)
    total = result["trials"]
    for field in ("mean_confidence", "mean_relative_score", "mean_target_frequency_power"):
        result[field] = math.fsum(row[field] * (row["trials"] / total) for row in metrics)
    if not all(math.isfinite(value) for value in result.values()):
        _fail("aggregate overflow/non-finite result")
    return result


def _summaries(rows: list[dict]) -> dict:
    baseline = _aggregate([row["baseline"] for row in rows])
    result = {"baseline": baseline, "candidates": []}
    for index in range(len(rows[0]["candidates"])):
        candidates = [row["candidates"][index] for row in rows]
        total = sum(row["metrics"]["trials"] for row in candidates)
        ratios = [v for row in candidates for v in row["spectral_score_ratio"]["per_trial"] if v is not None]
        candidate_metrics = _aggregate([row["metrics"] for row in candidates])
        result["candidates"].append({"candidate_id": candidates[0]["candidate_id"],
            "metrics": candidate_metrics,
            "macro_accuracy_delta": candidate_metrics["macro_accuracy"] - baseline["macro_accuracy"],
            "micro_accuracy_delta": candidate_metrics["micro_accuracy"] - baseline["micro_accuracy"],
            "prediction_agreement": math.fsum(row["prediction_agreement"] * row["metrics"]["trials"] / total for row in candidates),
            "mean_spectral_score_ratio": math.fsum(ratios) / len(ratios) if ratios else None,
            "defined_ratio_trials": len(ratios)})
    return result


def _output_paths(workspace: Path, output: str | Path | None, inputs: list[Path]) -> tuple[Path, Path] | None:
    if output is None:
        return None
    json_path = _inside(workspace, output)
    if json_path.suffix != ".json":
        _fail("output must have a .json suffix")
    markdown_path = json_path.with_suffix(".md")
    for path in (json_path, markdown_path):
        _inside(workspace, path)
        if path.exists() or path.is_symlink():
            _fail("JSON/Markdown output exists; overwrite refused")
        if any(path.is_relative_to(directory) for directory in inputs):
            _fail("reports must not be written inside input directories")
    return json_path, markdown_path


def render_markdown(report: dict) -> str:
    lines = ["# NeuroStation v5 SSVEP task-fidelity baseline", "",
        "Status: formal. This is not model, hardware or clinical acceptance.", "",
        f"Split: {report['split']}. Holdout loaded: false. Runtime enabled: false.",
        "Reference is task reference, not clean ground truth. No training, SNR, RMSE or denoising efficacy is evaluated.",
        "Validation is reporting-only; not for training or threshold selection.",
        "model_metrics_computed=false means no learned weights were loaded/executed; full runs compute baseline and supplied candidate-derived task metrics.",
        f"Task metrics computed: {report['task_metrics_computed']}. Learned model loaded: false.",
        "Accuracy deltas are fractions (0.02 = two percentage points). No acceptance gate is applied here.", ""]
    if report["check_only"]:
        lines += ["Manifest/sidecar checks only. NPZ content hashes and trial values are NOT verified.",
                  f"Selected subjects: {report['selected_subjects']}. Waveforms loaded: false."]
    else:
        lines += [f"Global baseline micro accuracy: {report['micro_accuracy']:.6f}", "",
                  "| Dataset | Subjects | Trials | Macro subject accuracy | Micro trial accuracy |",
                  "|---|---:|---:|---:|---:|"]
        for name, summary in report["dataset_summary"].items():
            metrics = summary["baseline"]
            lines.append(f"| {name} | {metrics['subjects']} | {metrics['trials']} | {metrics['macro_accuracy']:.6f} | {metrics['accuracy']:.6f} |")
        lines += ["", "| Dataset | Subject | Baseline accuracy | Confidence | Relative target score |",
                  "|---|---:|---:|---:|---:|"]
        for row in report["subjects"]:
            m = row["baseline"]
            lines.append(f"| {row['dataset']} | {row['subject_id']} | {m['accuracy']:.6f} | {m['mean_confidence']:.6f} | {m['mean_relative_score']:.6f} |")
        for summary in report["dataset_summary"].values():
            for candidate in summary["candidates"]:
                lines += ["", f"Candidate: {candidate['candidate_id']}",
                          f"- Macro subject accuracy: {candidate['metrics']['macro_accuracy']:.6f}",
                          f"- Micro accuracy: {candidate['metrics']['micro_accuracy']:.6f}",
                          f"- Macro accuracy delta: {candidate['macro_accuracy_delta']:+.6f}",
                          f"- Micro accuracy delta: {candidate['micro_accuracy_delta']:+.6f}",
                          f"- Prediction agreement: {candidate['prediction_agreement']:.6f}",
                          f"- Mean spectral score ratio: {candidate['mean_spectral_score_ratio']}"]
    lines += ["", "## Limitations"]
    lines += ["- " + item for item in report["limitations"]]
    return "\n".join(lines) + "\n"


def _publish(workspace: Path, paths: tuple[Path, Path], report: dict, markdown: bool) -> None:
    # Validate everything before creating reports. Exclusive-create both outputs
    # (no overwrite), roll back only files reserved by this invocation on error.
    json_path, md_path = paths
    _output_paths(workspace, json_path, [])
    json_text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    outputs = [(json_path, json_text)]
    if markdown:
        outputs.append((md_path, render_markdown(report)))
    handles, created = [], []
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        for path, text in outputs:
            _inside(workspace, path)
            handle = path.open("x", encoding="utf-8", newline="\n")
            handles.append((handle, text))
            created.append(path)
        for handle, text in handles:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        for handle, _ in handles:
            handle.close()
    except Exception:
        for handle, _ in handles:
            handle.close()
        for path in created:
            path.unlink()  # only our exclusive-created report files
        raise


def evaluate_task(
    reference_dir: str | Path = REFERENCE_RELATIVE, *,
    candidate_dirs: list[str | Path] | tuple[str | Path, ...] = (),
    candidate_signal_key: str = "processed", split: str = "validation",
    output: str | Path | None = None, markdown: bool = False,
    check_only: bool = False, workspace: str | Path | None = None,
) -> dict:
    """Evaluate the pre-registered task decoder; no holdout or training access.

    workspace is a host/test root, not an exemption: the fixed relative
    reference path and compiled manifest hash are always enforced. Synthetic
    tests patch the trusted hash; the CLI exposes no trust-bypass option.
    """
    if split not in ("train", "validation"):
        _fail("only train/validation allowed; holdout refused")
    try:
        root = reference_safety._guard_chain(Path(workspace) if workspace is not None else WORKSPACE).resolve()
    except reference_safety.SSVEPReferenceV5Error as exc:
        raise SSVEPTaskEvalV5Error(str(exc)) from exc
    if not root.is_dir():
        _fail("workspace root must exist")
    directory = _inside(root, reference_dir)
    if directory != (root / REFERENCE_RELATIVE).resolve() or not directory.is_dir():
        _fail("reference must come from the fixed frozen directory")
    if (not isinstance(candidate_signal_key, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", candidate_signal_key)
            or _dangerous(candidate_signal_key) or candidate_signal_key in set(LABEL_KEYS) | {"reference", "manifest_json", "phase_pi_multiple"}):
        _fail("candidate signal key is dangerous/reserved")
    if isinstance(candidate_dirs, (str, Path)):
        _fail("candidate_dirs must be a sequence of directories")
    directories = [_inside(root, value) for value in candidate_dirs]
    if len(set(directories)) != len(directories):
        _fail("duplicate candidate directories")
    for candidate in directories:
        _candidate_names(candidate.relative_to(root).as_posix())
        if not candidate.is_dir() or candidate == directory or candidate.is_relative_to(directory) or directory.is_relative_to(candidate):
            _fail("candidate directory must be separate from reference")
    paths = _output_paths(root, output, [directory] + directories)
    # Inventory all directories before touching NPZ bytes or numeric arrays.
    inventories = [_inventory(root, p, i > 0) for i, p in enumerate([directory] + directories)]
    baseline = _catalog(root, directory, False, inventories[0])
    candidates = [_catalog(root, p, True, inventories[i + 1]) for i, p in enumerate(directories)]
    for candidate in candidates:
        _bind_candidate(candidate, baseline, candidate_signal_key)
    selected = sorted(key for key, item in baseline["subjects"].items() if item["side"]["split"] == split)
    if not selected:
        _fail("no frozen subjects in requested split")
    report = {
        "schema_version": 1, "evaluation_version": "ssvep_task_eval_v5", "status": "formal",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "split": split,
        "reference_version": REFERENCE_VERSION,
        "reference_dir": directory.relative_to(root).as_posix(),
        "reference_manifest_sha256": baseline["manifest_hash"],
        "candidate_dirs": [p.relative_to(root).as_posix() for p in directories],
        "candidate_signal_key": candidate_signal_key if candidates else None,
        "check_only": check_only, "selected_subjects": len(selected),
        "waveforms_loaded": not check_only, "holdout_loaded": False,
        "clean_ground_truth": False, "model_metrics_computed": False,
        "task_metrics_computed": not check_only, "learned_model_loaded": False,
        "candidate_task_metrics_computed": bool(candidates) and not check_only,
        "model_metrics_scope": "model_metrics_computed=false means no learned weights were loaded or executed; full runs compute baseline and any supplied candidate-derived task metrics",
        "runtime_enabled": False, "production_validated": False,
        "training_performed": False, "threshold_selection_performed": False,
        "macro_accuracy": None, "micro_accuracy": None, "global_accuracy_summary": None,
        "validation_use": "reporting_only; not training or threshold_selection",
        "npz_content_hashes_verified": not check_only,
        "trial_alignment_values_verified": not check_only,
        "npz_verification_scope": "selected_split_only" if not check_only else "deferred; JSON/sidecar only",
        "native_units_assumed": False,
        "decoder": {"name": "fixed_frequency_power", "window_lengths_samples": WINDOW_LENGTHS.copy(),
            "demean": "per_trial_per_channel", "window": "numpy_hanning_symmetric",
            "bin_rule": "nearest_rfft_bin; ties choose lower bin",
            "power_normalization": "abs(rfft)**2 / sum(hann**2); channel_mean",
            "score": "fundamental + 0.5 * second_harmonic",
            "confidence": "(best - second)/(best + eps)",
            "relative_score": "labeled_target_score/(sum_target_scores + eps)",
            "ratio": "candidate_labeled_target_score/baseline_labeled_target_score; baseline<=eps undefined",
            "epsilon": EPSILON, "predictions_use_labels": False,
            "adaptive_frequency": False, "adaptive_window": False},
        "subjects": [], "dataset_summary": {}, "domain_summary": {},
        "limitations": ["formal status is not model/hardware/clinical acceptance",
            "FFT power reference is not clean signal ground truth",
            "native amplitude units remain as supplied; unknown units are never inferred",
            "no cross-dataset native power aggregation/comparison",
            "source stimulation windows have no learned latency adjustment",
            "ratio denominator<=EPS is an undefined numerical guard, not evidence of absent EEG response",
            "FFT-only task metrics do not demonstrate phase preservation or artifact removal",
            "this fixed FFT decoder is not a replication of published CCA/FBCCA results",
            "accuracy deltas are fractions (0.02 = two percentage points); no gate or threshold is selected here"],
    }
    domain_rows = defaultdict(list)
    if not check_only:
        for key in selected:
            item = baseline["subjects"][key]
            side = item["side"]
            arrays = _read_subject(baseline, item, "reference")
            decoded = spectral_decode(arrays["reference"], side)
            base_trials = _trial_metrics(decoded, arrays["target_id"])
            row = {"dataset": key[0], "subject_id": key[1], "split": split,
                "native_units": side["native_units"], "native_units_assumed": False,
                "native_sample_rate_hz": side["native_sample_rate_hz"],
                "source_stimulation_window": {"start_inclusive": side["onset_sample_zero_based"],
                    "end_exclusive": side["onset_sample_zero_based"] + WINDOW_LENGTHS[key[0]]},
                "fundamental_bins": decoded["fundamental_bins"], "second_harmonic_bins": decoded["second_harmonic_bins"],
                "baseline": _metric_summary(base_trials),
                "trials": {"trial_index": arrays["trial_index"].tolist(),
                    "target_id": arrays["target_id"].tolist(), "block_id": arrays["block_id"].tolist(),
                    "electrode_id": arrays["electrode_id"].tolist(), "frequency_hz": arrays["frequency_hz"].tolist(),
                    "baseline_prediction": decoded["predictions"].tolist(),
                    "baseline_target_frequency_power": base_trials["target_power"].tolist()},
                "candidates": []}
            domains = {}
            for electrode in np.unique(arrays["electrode_id"]):
                name = "electrode_unknown" if int(electrode) == -1 else side["electrode_order"][int(electrode)]
                mask = arrays["electrode_id"] == electrode
                domains[name] = {"mask": mask, "row": {"baseline": _metric_summary(base_trials, mask), "candidates": []}}
            for catalog in candidates:
                candidate_arrays = _read_subject(catalog, catalog["subjects"][key], candidate_signal_key)
                _aligned(candidate_arrays, arrays)
                candidate_decoded = spectral_decode(candidate_arrays[candidate_signal_key], side)
                candidate_trials = _trial_metrics(candidate_decoded, arrays["target_id"])
                agreement = candidate_decoded["predictions"] == decoded["predictions"]
                cid = catalog["directory"].relative_to(root).as_posix()
                row["candidates"].append({"candidate_id": cid, "metrics": _metric_summary(candidate_trials),
                    "accuracy_delta": float(candidate_trials["correct"].mean()) - row["baseline"]["accuracy"],
                    "prediction_agreement": float(agreement.mean()),
                    "predictions": candidate_decoded["predictions"].tolist(),
                    "target_frequency_power": candidate_trials["target_power"].tolist(),
                    "spectral_score_ratio": _ratio(candidate_trials["target_power"], base_trials["target_power"])})
                for domain in domains.values():
                    mask = domain["mask"]
                    domain["row"]["candidates"].append({"candidate_id": cid,
                        "metrics": _metric_summary(candidate_trials, mask),
                        "accuracy_delta": float(candidate_trials["correct"][mask].mean()) - domain["row"]["baseline"]["accuracy"],
                        "prediction_agreement": float(agreement[mask].mean()),
                        "spectral_score_ratio": _ratio(candidate_trials["target_power"][mask], base_trials["target_power"][mask])})
                del candidate_arrays, candidate_decoded, candidate_trials
            row["domains"] = {name: domain["row"] for name, domain in domains.items()}
            for name, domain in domains.items():
                domain_rows[f"{key[0]}/{name}"].append(domain["row"])
            report["subjects"].append(row)
            del arrays, decoded, base_trials
        datasets = defaultdict(list)
        for row in report["subjects"]:
            datasets[row["dataset"]].append(row)
        report["dataset_summary"] = {name: _summaries(rows) for name, rows in sorted(datasets.items())}
        report["domain_summary"] = {name: _summaries(rows) for name, rows in sorted(domain_rows.items())}
        global_baseline = _accuracy_aggregate([row["baseline"] for row in report["subjects"]])
        report["macro_accuracy"] = global_baseline["macro_accuracy"]
        report["micro_accuracy"] = global_baseline["micro_accuracy"]
        report["global_accuracy_summary"] = {"baseline": global_baseline, "candidates": []}
        for index in range(len(candidates)):
            candidate_metrics = _accuracy_aggregate([row["candidates"][index]["metrics"] for row in report["subjects"]])
            report["global_accuracy_summary"]["candidates"].append({
                "candidate_id": report["subjects"][0]["candidates"][index]["candidate_id"],
                "metrics": candidate_metrics,
                "macro_accuracy_delta": candidate_metrics["macro_accuracy"] - global_baseline["macro_accuracy"],
                "micro_accuracy_delta": candidate_metrics["micro_accuracy"] - global_baseline["micro_accuracy"],
            })
    for catalog in [baseline] + candidates:
        _verify_unchanged(root, catalog)
    # Output publication happens only after *all* subjects and candidates pass.
    json.dumps(report, allow_nan=False)
    if paths is not None:
        _output_paths(root, paths[0], [directory] + directories)
        _publish(root, paths, report, markdown)
    return report
