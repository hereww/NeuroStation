"""Dataset builder/loader contract for learned denoiser v4.

The source is a preregistered paired, semi-simulated EOG corpus.  The loader
never assigns groups or infers subjects.  It exposes train/validation through
``load_split`` and exposes test only through the explicit evaluation API.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

WINDOW_SAMPLES = 500
SOURCE_RATE_HZ = 200
TARGET_RATE_HZ = 250
RESAMPLE_UP = 5
RESAMPLE_DOWN = 4
CHANNEL_COUNT = 19
MODEL_VERSION = "learned_denoise_v4"
DEFAULT_DATASET_RELATIVE = "datasets/processed/learned_denoiser_v4.npz"
DEFAULT_PREREGISTRATION_RELATIVE = "analysis_outputs/learned_denoiser_v4_preregistration.json"
DEFAULT_INTERFACE_RELATIVE = "analysis_outputs/learned_denoiser_v4_interface_contract.json"
SPLITS = ("train", "validation", "test")
LOADABLE_SPLITS = ("train", "validation")
ARRAY_FIELDS = (
    "normalized", "target", "noisy", "clean", "is_identity", "segment_id",
    "channel_index", "window_start", "group_id", "noisy_mean", "noisy_ac_rms",
)


class LearnedDatasetV4Error(ValueError):
    """Invalid v4 preregistration, source, manifest, array, or split scope."""


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_bytes(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()


def _array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    header = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)}, separators=(",", ":")).encode()
    return _digest_bytes(header, value.tobytes(order="C"))


def _plain(value: Any) -> bool:
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is float:
        return bool(np.isfinite(value))
    if type(value) is list:
        return all(_plain(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _plain(item) for key, item in value.items())
    return False


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise LearnedDatasetV4Error(f"cannot read JSON {path}: {error}") from error
    if type(value) is not dict or not _plain(value):
        raise LearnedDatasetV4Error(f"JSON must be a finite plain object: {path}")
    return value


def _hex_digest(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64 or value != value.lower() or any(c not in "0123456789abcdef" for c in value):
        raise LearnedDatasetV4Error(f"{name} must be lowercase SHA256")
    return value


def _resolve_path(root: Path, value: Any, name: str) -> Path:
    if type(value) is not str or not value.strip():
        raise LearnedDatasetV4Error(f"{name} path is required in preregistration")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _source_entries(data: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    raw = data.get("sources")
    if type(raw) is not dict or set(raw) != {"clean", "noisy", "heog", "veog"}:
        raise LearnedDatasetV4Error("data.sources must declare clean/noisy/heog/veog")
    entries = {}
    for name, value in raw.items():
        if type(value) is not dict:
            raise LearnedDatasetV4Error(f"source {name} must be an object")
        entries[name] = {
            "path": _resolve_path(root, value.get("relative_path"), f"source {name}"),
            "sha256": _hex_digest(value.get("sha256"), f"source {name}.sha256"),
            "declared_name": name,
        }
        if not entries[name]["path"].is_file() or sha256(entries[name]["path"]) != entries[name]["sha256"]:
            raise LearnedDatasetV4Error(f"source SHA256/missing-file mismatch: {entries[name]['path']}")
    return entries


def _grouping(data: dict[str, Any]) -> tuple[dict[int, int], dict[str, list[int]], dict]:
    raw = data.get("group_assignments")
    if (type(raw) is not dict or not raw
            or any(type(key) is not str or not key.isdecimal() or str(int(key)) != key
                   or type(group) is not int or group < 1 for key, group in raw.items())):
        raise LearnedDatasetV4Error("group_assignments must explicitly map segment strings to positive integer group IDs")
    assignment = {int(segment): group for segment, group in raw.items()}
    partitions = {}
    for split in SPLITS:
        groups = data.get(f"{split}_groups")
        if type(groups) is not list or not groups or any(type(group) is not int or group < 1 for group in groups):
            raise LearnedDatasetV4Error(f"data.{split}_groups must contain positive integer groups")
        partitions[split] = groups.copy()
    declared = [group for split in SPLITS for group in partitions[split]]
    if len(set(declared)) != len(declared):
        raise LearnedDatasetV4Error("train/validation/test groups overlap or contain duplicates")
    if set(declared) != set(assignment.values()):
        raise LearnedDatasetV4Error("declared split groups must exactly cover group_assignments")
    return assignment, partitions, {str(group): group for group in declared}


def _interface_contract(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    config = value.get("config", {})
    if (config.get("model_version") != MODEL_VERSION
            or config.get("window_samples") != WINDOW_SAMPLES
            or config.get("hop_samples") != WINDOW_SAMPLES
            or config.get("sample_rate_hz") != TARGET_RATE_HZ
            or value.get("dataset", {}).get("module") != "eeg_tools.learned_dataset_v4"):
        raise LearnedDatasetV4Error("v4 interface contract mismatch")
    return value


def read_preregistration(root: Path | str, preregistration: Path | str | None = None,
                         interface_contract: Path | str | None = None) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[int, Any], dict[str, list[Any]], dict[Any, int]]:
    root = Path(root).resolve()
    prereg_path = (root / (preregistration or DEFAULT_PREREGISTRATION_RELATIVE)).resolve() if not Path(preregistration or DEFAULT_PREREGISTRATION_RELATIVE).is_absolute() else Path(preregistration or DEFAULT_PREREGISTRATION_RELATIVE).resolve()
    interface_path = root / (interface_contract or DEFAULT_INTERFACE_RELATIVE)
    _interface_contract(interface_path.resolve())
    prereg = _read_json(prereg_path)
    data = prereg.get("data")
    if type(data) is not dict:
        raise LearnedDatasetV4Error("v4 preregistration data object is required")
    expected = {"source_sample_rate_hz": 200, "target_sample_rate_hz": 250,
                "channels_per_segment": 19, "window_samples": 500, "hop_samples": 500,
                "full_segment_windows_only": True}
    if any(type(data.get(key)) is not type(value) or data[key] != value for key, value in expected.items()):
        raise LearnedDatasetV4Error("preregistered rate/channel/window contract mismatch")
    sources = _source_entries(data, root)
    assignment, partitions, labels = _grouping(data)
    if (type(data.get("source_segments")) is not int or data["source_segments"] < 3
            or set(assignment) != set(range(1, data["source_segments"] + 1))):
        raise LearnedDatasetV4Error("group_assignments must enumerate every segment from 1 through max segment")
    if prereg.get("status") not in ("formal", "design_interface_frozen_before_new_holdout_metrics", "preregistered"):
        raise LearnedDatasetV4Error("v4 preregistration status is not recognized")
    return prereg, sources, assignment, partitions, labels


def _load_pair(path: Path, variable: str, segment_id: int) -> np.ndarray:
    try:
        value = loadmat(path, variable_names=[variable]).get(variable)
    except (OSError, ValueError, TypeError) as error:
        raise LearnedDatasetV4Error(f"cannot load {variable} from {path}: {error}") from error
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[0] != CHANNEL_COUNT:
        raise LearnedDatasetV4Error(f"segment {segment_id} {variable} must be [19,N]")
    if value.shape[1] < 1 or value.dtype.kind not in "iuf" or not np.all(np.isfinite(value)):
        raise LearnedDatasetV4Error(f"segment {segment_id} {variable} must be finite real data")
    return np.asarray(value, dtype=np.float64)


def _normalize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1 or len(raw) != WINDOW_SAMPLES or not np.all(np.isfinite(raw)):
        raise LearnedDatasetV4Error("window must be finite float64 [500]")
    mean = float(np.mean(raw, dtype=np.float64))
    centered = raw - mean
    with np.errstate(over="ignore", invalid="ignore"):
        rms = float(np.sqrt(np.mean(centered * centered, dtype=np.float64)))
    if not np.isfinite(mean) or not np.isfinite(rms) or rms <= 1e-8:
        raise LearnedDatasetV4Error("constant/unsafe noisy window; no exclusion is allowed")
    normalized = ((raw - mean) / rms).astype(np.float32)
    if not np.all(np.isfinite(normalized)):
        raise LearnedDatasetV4Error("normalized window is nonfinite")
    return normalized, mean, rms


def _window_record(clean: np.ndarray, noisy: np.ndarray, segment_id: int, channel: int,
                   start: int, group_id: int, identity: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized, mean, rms = _normalize(noisy)
    target = ((clean - mean) / rms).astype(np.float32)
    if not np.all(np.isfinite(target)):
        raise LearnedDatasetV4Error("target window is nonfinite")
    metadata = {"segment_id": int(segment_id), "channel_index": int(channel), "window_start": int(start),
                "group_id": int(group_id), "is_identity": bool(identity)}
    raw_hash = _digest_bytes(_array_hash(clean).encode(), _array_hash(noisy).encode(), json.dumps(metadata, sort_keys=True).encode())
    row = {"normalized": normalized[None, :], "target": target[None, :], "noisy": noisy.astype(np.float64),
           "clean": clean.astype(np.float64), "is_identity": bool(identity), "segment_id": int(segment_id),
           "channel_index": int(channel), "window_start": int(start), "group_id": int(group_id),
           "noisy_mean": float(mean), "noisy_ac_rms": float(rms)}
    record = {**metadata, "clean_sha256": _array_hash(clean), "noisy_sha256": _array_hash(noisy), "window_sha256": raw_hash}
    return row, record


def _empty_or_stack(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not rows:
        raise LearnedDatasetV4Error("a declared split produced no full windows")
    return {
        "normalized": np.stack([row["normalized"] for row in rows]).astype(np.float32),
        "target": np.stack([row["target"] for row in rows]).astype(np.float32),
        "noisy": np.stack([row["noisy"] for row in rows]).astype(np.float64),
        "clean": np.stack([row["clean"] for row in rows]).astype(np.float64),
        "is_identity": np.asarray([row["is_identity"] for row in rows], dtype=bool),
        "segment_id": np.asarray([row["segment_id"] for row in rows], dtype=np.int64),
        "channel_index": np.asarray([row["channel_index"] for row in rows], dtype=np.int64),
        "window_start": np.asarray([row["window_start"] for row in rows], dtype=np.int64),
        "group_id": np.asarray([row["group_id"] for row in rows], dtype=np.int64),
        "noisy_mean": np.asarray([row["noisy_mean"] for row in rows], dtype=np.float64),
        "noisy_ac_rms": np.asarray([row["noisy_ac_rms"] for row in rows], dtype=np.float64),
    }


def _manifest_window_hash(records: list[dict[str, Any]]) -> str:
    return _digest_bytes(*[record["window_sha256"].encode("ascii") for record in records])


def build_dataset(root: Path | str, preregistration: Path | str | None = None,
                  output: Path | str | None = None, interface_contract: Path | str | None = None,
                  check_only: bool = False) -> dict[str, Any]:
    root = Path(root).resolve()
    output_path = (root / (output or DEFAULT_DATASET_RELATIVE)).resolve()
    if not check_only and (output_path.exists() or Path(str(output_path) + ".manifest.json").exists()):
        raise LearnedDatasetV4Error("dataset output or manifest already exists; refusing overwrite")
    prereg, sources, assignment, partitions, label_map = read_preregistration(root, preregistration, interface_contract)
    prereg_path = (root / (preregistration or DEFAULT_PREREGISTRATION_RELATIVE)).resolve()
    prereg_digest = sha256(prereg_path)
    interface_path = (root / (interface_contract or DEFAULT_INTERFACE_RELATIVE)).resolve()
    interface_digest = sha256(interface_path)
    source_before = {name: sha256(entry["path"]) for name, entry in sources.items()}
    loaded: dict[str, dict[int, np.ndarray]] = {name: {} for name in sources}
    for name, entry in sources.items():
        try:
            values = loadmat(entry["path"])
        except (OSError, ValueError, TypeError) as error:
            raise LearnedDatasetV4Error(f"cannot load source MAT {entry['path']}: {error}") from error
        patterns = {"clean": "sim{segment_id}_resampled", "noisy": "sim{segment_id}_con",
                    "heog": "heog_{segment_id}", "veog": "veog_{segment_id}"}
        expected_variables = {patterns[name].format(segment_id=i) for i in assignment}
        if {key for key in values if not key.startswith("__")} != expected_variables:
            raise LearnedDatasetV4Error(f"source {name} variables must exactly enumerate declared segments")
        for segment_id in assignment:
            variable = patterns[name].format(segment_id=segment_id)
            value = values.get(variable)
            rows = CHANNEL_COUNT if name in ("clean", "noisy") else 1
            if (not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[0] != rows
                    or value.shape[1] < 1 or value.dtype.kind not in "iuf" or not np.all(np.isfinite(value))):
                raise LearnedDatasetV4Error(f"source variable {variable} must be finite real [{rows},N]")
            loaded[name][segment_id] = np.asarray(value, dtype=np.float64)
        if sha256(entry["path"]) != source_before[name]:
            raise LearnedDatasetV4Error(f"source changed while reading: {entry['path']}")
    arrays: dict[str, dict[str, np.ndarray]] = {}
    records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    raw_segments: list[dict[str, Any]] = []
    tail_records: list[dict[str, Any]] = []
    for segment_id in sorted(assignment):
        lengths = {value[segment_id].shape[1] for value in loaded.values()}
        if len(lengths) != 1:
            raise LearnedDatasetV4Error(f"segment {segment_id} paired source lengths differ")
        pure = resample_poly(loaded["clean"][segment_id], RESAMPLE_UP, RESAMPLE_DOWN, axis=1).astype(np.float64, copy=False)
        contaminated = resample_poly(loaded["noisy"][segment_id], RESAMPLE_UP, RESAMPLE_DOWN, axis=1).astype(np.float64, copy=False)
        if pure.shape != contaminated.shape:
            raise LearnedDatasetV4Error(f"segment {segment_id} pure/contaminated resampled shapes differ")
        group_id = assignment[segment_id]
        split = next(name for name in SPLITS if group_id in partitions[name])
        full_count, tail = divmod(pure.shape[1], WINDOW_SAMPLES)
        if full_count < 1:
            raise LearnedDatasetV4Error(f"segment {segment_id} has no full 500-sample window")
        raw_segments.append({"segment_id": int(segment_id), "group_id": int(group_id), "source_samples": int(loaded["clean"][segment_id].shape[1]),
                             "resampled_samples": int(pure.shape[1]), "full_window_count": int(full_count), "tail_samples": int(tail),
                             "pure_source_sha256": _array_hash(loaded["clean"][segment_id]), "contaminated_source_sha256": _array_hash(loaded["noisy"][segment_id]),
                             "pure_resampled_sha256": _array_hash(pure), "contaminated_resampled_sha256": _array_hash(contaminated),
                             "source_segment_hashes": {name: _array_hash(value[segment_id]) for name, value in loaded.items()},
                             "split": split})
        tail_records.append({"segment_id": int(segment_id), "group_id": int(group_id), "tail_samples": int(tail),
                             "pure_tail_sha256": _array_hash(pure[:, full_count * WINDOW_SAMPLES:]),
                             "contaminated_tail_sha256": _array_hash(contaminated[:, full_count * WINDOW_SAMPLES:])})
        for channel in range(CHANNEL_COUNT):
            for start in range(0, full_count * WINDOW_SAMPLES, WINDOW_SAMPLES):
                pure_window = pure[channel, start:start + WINDOW_SAMPLES]
                noisy_window = contaminated[channel, start:start + WINDOW_SAMPLES]
                # Polluted paired sample first, then matched true-pure identity.
                row, record = _window_record(pure_window, noisy_window, segment_id, channel, start, group_id, False)
                arrays.setdefault(split, {}).setdefault("rows", []).append(row)
                records[split].append(record)
                row, record = _window_record(pure_window, pure_window, segment_id, channel, start, group_id, True)
                arrays[split]["rows"].append(row)
                records[split].append(record)
    for split in SPLITS:
        arrays[split] = _empty_or_stack(arrays[split]["rows"])
    manifest: dict[str, Any] = {
        "schema_version": 1, "model_version": MODEL_VERSION, "status": "formal",
        "source_sample_rate_hz": SOURCE_RATE_HZ, "sample_rate_hz": TARGET_RATE_HZ,
        "resample_poly": {"up": RESAMPLE_UP, "down": RESAMPLE_DOWN, "axis": 1},
        "window_samples": WINDOW_SAMPLES, "hop_samples": WINDOW_SAMPLES, "channel_count": CHANNEL_COUNT,
        "target_definition": "(true_clean-noisy_mean)/noisy_ac_rms; original resampled pure clean remains the ground truth; DC is not redefined",
        "identity_definition": "one matched true-pure identity row for every polluted channel/window; no other fake clean",
        "preregistration_sha256": prereg_digest,
        "interface_contract_sha256": interface_digest,
        "source_lock": {name: {"path": str(entry["path"]), "sha256": source_before[name], "declared_name": entry["declared_name"]} for name, entry in sources.items()},
        "group_assignments": {str(key): int(value) for key, value in assignment.items()},
        "group_partitions": {split: [int(group) for group in partitions[split]] for split in SPLITS},
        "group_label_encoding": label_map,
        "split_counts": {split: int(len(arrays[split]["normalized"])) for split in SPLITS},
        "identity_counts": {split: int(np.count_nonzero(arrays[split]["is_identity"])) for split in SPLITS},
        "raw_segments": raw_segments, "tails": tail_records,
        "window_hashes": {split: {"count": len(records[split]), "sha256": _manifest_window_hash(records[split]), "rows": records[split]} for split in SPLITS},
        "test_access_semantics": "Builder materializes the preregistered test partition, but no model, metrics, threshold, or selection is accessed here.",
        "no_subject_or_electrode_inference": True,
        "physical_units": None,
    }
    for name, entry in sources.items():
        if sha256(entry["path"]) != source_before[name]:
            raise LearnedDatasetV4Error(f"source changed during build: {entry['path']}")
    if sha256(prereg_path) != prereg_digest or sha256(interface_path) != interface_digest:
        raise LearnedDatasetV4Error("preregistration/interface changed while building")
    if not _plain(manifest):
        raise LearnedDatasetV4Error("manifest contains non-primitive metadata")
    manifest_text = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    arrays_to_save = {f"{split}_{field}": arrays[split][field] for split in SPLITS for field in ARRAY_FIELDS}
    arrays_to_save["manifest_json"] = np.asarray(manifest_text)
    result = {"manifest": manifest, "arrays": arrays, "dataset_sha256": None, "output": None, "check_only": bool(check_only)}
    if check_only:
        return result
    output_path = (root / (output or DEFAULT_DATASET_RELATIVE)).resolve()
    manifest_path = Path(str(output_path) + ".manifest.json")
    sidecar = dict(manifest)
    _publish_dataset(output_path, manifest_path, arrays_to_save, sidecar)
    result["manifest"] = sidecar
    result["dataset_sha256"] = sha256(output_path)
    result["output"] = str(output_path)
    return result


def _publish_dataset(output: Path, sidecar_path: Path, arrays: dict[str, np.ndarray], manifest: dict) -> None:
    import os
    import tempfile
    if output.exists() or sidecar_path.exists():
        raise LearnedDatasetV4Error("dataset output or manifest already exists; refusing overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    staged, published = [], []
    try:
        for target in (output, sidecar_path):
            descriptor, filename = tempfile.mkstemp(prefix=target.name + ".", suffix=".pending", dir=target.parent)
            os.close(descriptor)
            staged.append((Path(filename), target))
        with staged[0][0].open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        manifest["dataset_file_sha256"] = sha256(staged[0][0])
        staged[1][0].write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        for temporary, target in staged:
            os.link(temporary, target)
            published.append((temporary, target))
    except Exception:
        for temporary, target in published:
            if target.exists() and temporary.samefile(target):
                target.unlink()
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)



def _validate_manifest_contents(manifest: dict[str, Any]) -> None:
    fixed = {"schema_version": 1, "model_version": MODEL_VERSION, "status": "formal",
             "source_sample_rate_hz": 200, "sample_rate_hz": 250,
             "window_samples": 500, "hop_samples": 500, "channel_count": 19}
    if any(type(manifest.get(key)) is not type(value) or manifest[key] != value for key, value in fixed.items()):
        raise LearnedDatasetV4Error("dataset manifest schema/rate/channel contract mismatch")
    _hex_digest(manifest.get("preregistration_sha256"), "preregistration_sha256")
    groups = manifest.get("group_partitions")
    if (type(groups) is not dict or set(groups) != set(SPLITS)
            or any(type(v) is not list or not v or any(type(g) is not int for g in v) for v in groups.values())
            or len({g for v in groups.values() for g in v}) != sum(map(len, groups.values()))):
        raise LearnedDatasetV4Error("manifest train/validation/test groups overlap or are missing")
    records = manifest.get("raw_segments")
    if type(records) is not list or not records:
        raise LearnedDatasetV4Error("manifest raw segment records missing")
    if len({record["segment_id"] for record in records}) != len(records):
        raise LearnedDatasetV4Error("duplicate manifest raw segment")
    if set(manifest.get("window_hashes", {})) != set(SPLITS):
        raise LearnedDatasetV4Error("complete window hashes missing")
    for split in SPLITS:
        count = manifest["split_counts"][split]
        expected = sum(record["full_window_count"] * 19 * 2 for record in records if record["split"] == split)
        if type(count) is not int or count < 1 or count != expected or manifest["identity_counts"][split] * 2 != count:
            raise LearnedDatasetV4Error("manifest full-window/identity counts are inconsistent")
        if len(manifest["window_hashes"][split]["rows"]) != count:
            raise LearnedDatasetV4Error("manifest per-window fingerprints incomplete")
    for record in records:
        if (record["split"] not in SPLITS or record["group_id"] not in groups[record["split"]]
                or manifest["group_assignments"].get(str(record["segment_id"])) != record["group_id"]
                or record["resampled_samples"] != (record["source_samples"] * 5 + 3) // 4
                or record["full_window_count"] * 500 + record["tail_samples"] != record["resampled_samples"]):
            raise LearnedDatasetV4Error("manifest raw segment/resample/group relation mismatch")

def _validate_manifest(manifest: dict[str, Any]) -> None:
    try:
        _validate_manifest_contents(manifest)
        if manifest.get("no_subject_or_electrode_inference") is not True:
            raise LearnedDatasetV4Error("subject/electrode identities are not established")
        _hex_digest(manifest.get("interface_contract_sha256"), "interface_contract_sha256")
        if set(manifest["source_lock"]) != {"clean", "noisy", "heog", "veog"}:
            raise LearnedDatasetV4Error("all four source identities required")
        for source in manifest["source_lock"].values():
            _hex_digest(source["sha256"], "source SHA256")
        records = manifest["raw_segments"]
        if set(manifest["group_assignments"]) != {str(r["segment_id"]) for r in records}:
            raise LearnedDatasetV4Error("group assignments do not enumerate raw segments")
        if set(manifest["group_assignments"].values()) != {g for v in manifest["group_partitions"].values() for g in v}:
            raise LearnedDatasetV4Error("group partitions do not cover all assignments")
        if len(manifest["tails"]) != len(records):
            raise LearnedDatasetV4Error("tail record coverage mismatch")
        for record, tail in zip(records, manifest["tails"]):
            for key in ("segment_id", "group_id", "tail_samples"):
                if tail[key] != record[key]:
                    raise LearnedDatasetV4Error("tail record disagrees with segment")
            for key in ("pure_tail_sha256", "contaminated_tail_sha256"):
                _hex_digest(tail[key], key)
            for key in ("segment_id", "group_id", "source_samples", "resampled_samples", "full_window_count", "tail_samples"):
                if type(record[key]) is not int or record[key] < 0:
                    raise LearnedDatasetV4Error("invalid raw segment integer")
            if record["source_samples"] < 1 or record["full_window_count"] < 1:
                raise LearnedDatasetV4Error("raw segment has no full windows")
            for key in ("pure_source_sha256", "contaminated_source_sha256", "pure_resampled_sha256", "contaminated_resampled_sha256"):
                _hex_digest(record[key], key)
        for split in SPLITS:
            entry = manifest["window_hashes"][split]
            if entry["count"] != manifest["split_counts"][split] or _manifest_window_hash(entry["rows"]) != entry["sha256"]:
                raise LearnedDatasetV4Error("manifest window fingerprints inconsistent")
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise LearnedDatasetV4Error("malformed dataset manifest") from error


def _manifest_for(path: Path) -> tuple[dict[str, Any], str, str]:
    manifest_path = Path(str(path) + ".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise LearnedDatasetV4Error("dataset NPZ and sidecar manifest are both required")
    dataset_digest, manifest_digest = sha256(path), sha256(manifest_path)
    manifest = _read_json(manifest_path)
    if manifest.get("dataset_file_sha256") != dataset_digest:
        raise LearnedDatasetV4Error("dataset file SHA256 does not match sidecar manifest")
    return manifest, dataset_digest, manifest_digest


def _load_split(path: Path, split: str) -> dict[str, Any]:
    if split not in SPLITS:
        raise LearnedDatasetV4Error(f"unknown dataset split: {split}")
    manifest, dataset_digest, manifest_digest = _manifest_for(path)
    if manifest.get("split_counts", {}).get(split, 0) < 1:
        raise LearnedDatasetV4Error(f"manifest has no rows for {split}")
    prefix = f"{split}_"
    required = {prefix + field for field in ARRAY_FIELDS}
    try:
        with np.load(path, allow_pickle=False) as archive:
            expected_members = {f"{s}_{field}" for s in SPLITS for field in ARRAY_FIELDS} | {"manifest_json"}
            if set(archive.files) != expected_members:
                raise LearnedDatasetV4Error(f"dataset is missing {split} arrays")
            values = {field: np.array(archive[prefix + field], copy=True) for field in ARRAY_FIELDS}
            embedded = archive.get("manifest_json")
            if embedded is None or embedded.ndim != 0 or embedded.dtype.kind != "U":
                raise LearnedDatasetV4Error("manifest_json must be a Unicode scalar")
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, LearnedDatasetV4Error):
            raise
        raise LearnedDatasetV4Error(f"cannot read v4 NPZ: {error}") from error
    base_manifest = {key: value for key, value in manifest.items() if key != "dataset_file_sha256"}
    try:
        embedded_value = json.loads(str(embedded.item()))
    except (ValueError, TypeError) as error:
        raise LearnedDatasetV4Error("invalid embedded manifest JSON") from error
    if embedded_value != base_manifest:
        raise LearnedDatasetV4Error("embedded/sidecar manifests disagree")
    _validate_manifest(manifest)
    _validate_arrays(values, split)
    expected_count = int(manifest["split_counts"][split])
    if expected_count != len(values["normalized"]):
        raise LearnedDatasetV4Error("manifest split count mismatch")
    actual_rows = [_record_from_arrays(values, i) for i in range(len(values["normalized"]))]
    if (actual_rows != manifest["window_hashes"][split]["rows"]
            or _manifest_window_hash(actual_rows) != manifest["window_hashes"][split]["sha256"]):
        raise LearnedDatasetV4Error("window hash mismatch")
    groups = manifest["group_partitions"][split]
    if set(values["group_id"].tolist()) != set(groups):
        raise LearnedDatasetV4Error("loaded group IDs disagree with preregistration")
    segment_records = {item["segment_id"]: item for item in manifest["raw_segments"] if item["split"] == split}
    if set(values["segment_id"].tolist()) != set(segment_records):
        raise LearnedDatasetV4Error("split segment IDs disagree with manifest")
    for segment_id, record in segment_records.items():
        indices = np.flatnonzero(values["segment_id"] == segment_id)
        identities = {(int(values["channel_index"][i]), int(values["window_start"][i]), bool(values["is_identity"][i])) for i in indices}
        expected_rows = {(channel, start, identity) for channel in range(19)
                         for start in range(0, record["full_window_count"] * 500, 500)
                         for identity in (False, True)}
        for channel in range(19):
            for start in range(0, record["full_window_count"] * 500, 500):
                pair = indices[(values["channel_index"][indices] == channel) & (values["window_start"][indices] == start)]
                if len(pair) != 2 or not np.array_equal(values["clean"][pair[0]], values["clean"][pair[1]]):
                    raise LearnedDatasetV4Error("identity and contaminated rows must share original pure GT")
        if identities != expected_rows or len(indices) != len(expected_rows) or not np.all(values["group_id"][indices] == record["group_id"]):
            raise LearnedDatasetV4Error("segment windows were duplicated, excluded, or regrouped")
    if sha256(path) != dataset_digest or sha256(Path(str(path) + ".manifest.json")) != manifest_digest:
        raise LearnedDatasetV4Error("dataset changed while loading")
    result = dict(values)
    result.update({"split": split, "manifest": manifest, "dataset_sha256": dataset_digest,
                   "manifest_sha256": manifest_digest, "evaluation_only": split == "test"})
    return result


def _record_from_arrays(values: dict[str, np.ndarray], index: int) -> dict[str, Any]:
    metadata = {"segment_id": int(values["segment_id"][index]), "channel_index": int(values["channel_index"][index]),
                "window_start": int(values["window_start"][index]), "group_id": int(values["group_id"][index]),
                "is_identity": bool(values["is_identity"][index])}
    clean, noisy = values["clean"][index], values["noisy"][index]
    return {**metadata, "clean_sha256": _array_hash(clean), "noisy_sha256": _array_hash(noisy),
            "window_sha256": _digest_bytes(_array_hash(clean).encode(), _array_hash(noisy).encode(), json.dumps(metadata, sort_keys=True).encode())}


def _validate_arrays(values: dict[str, np.ndarray], split: str) -> None:
    if values["normalized"].ndim != 3:
        raise LearnedDatasetV4Error(f"{split}.normalized must be float32 [N,1,500]")
    n = len(values["normalized"])
    if values["normalized"].dtype != np.dtype("float32") or values["normalized"].shape != (n, 1, WINDOW_SAMPLES):
        raise LearnedDatasetV4Error(f"{split}.normalized must be float32 [N,1,500]")
    if values["target"].dtype != np.dtype("float32") or values["target"].shape != (n, 1, WINDOW_SAMPLES):
        raise LearnedDatasetV4Error(f"{split}.target must be float32 [N,1,500]")
    for name in ("noisy", "clean"):
        if values[name].dtype != np.dtype("float64") or values[name].shape != (n, WINDOW_SAMPLES):
            raise LearnedDatasetV4Error(f"{split}.{name} must be float64 [N,500]")
    if values["is_identity"].dtype != np.dtype("bool") or values["is_identity"].shape != (n,):
        raise LearnedDatasetV4Error(f"{split}.is_identity must be bool [N]")
    for name in ("segment_id", "channel_index", "window_start", "group_id"):
        if values[name].dtype.kind not in "iu" or values[name].shape != (n,):
            raise LearnedDatasetV4Error(f"{split}.{name} must be integer [N]")
    for name in ("noisy_mean", "noisy_ac_rms"):
        if values[name].dtype != np.dtype("float64") or values[name].shape != (n,) or not np.all(np.isfinite(values[name])):
            raise LearnedDatasetV4Error(f"{split}.{name} must be finite float64 [N]")
    if n < 1 or n % 2 or np.count_nonzero(values["is_identity"]) != n // 2:
        raise LearnedDatasetV4Error("split must contain matched 50% true-pure identity windows")
    for name in ("normalized", "target", "noisy", "clean"):
        if not np.all(np.isfinite(values[name])):
            raise LearnedDatasetV4Error(f"{split}.{name} contains nonfinite data")
    if not np.array_equal(values["noisy"][values["is_identity"]], values["clean"][values["is_identity"]]):
        raise LearnedDatasetV4Error("identity sample is not true pure")
    for index in range(n):
        normalized, mean, rms = _normalize(values["noisy"][index])
        expected_target = ((values["clean"][index] - mean) / rms).astype(np.float32)
        if (mean != values["noisy_mean"][index] or rms != values["noisy_ac_rms"][index]
                or not np.array_equal(normalized, values["normalized"][index, 0])
                or not np.array_equal(expected_target, values["target"][index, 0])):
            raise LearnedDatasetV4Error("normalization/target changed; original clean GT and DC must not be redefined")
    if np.any(values["channel_index"] < 0) or np.any(values["channel_index"] >= CHANNEL_COUNT) or np.any(values["window_start"] < 0):
        raise LearnedDatasetV4Error(f"{split} metadata indices are invalid")


def load_split(path: Path | str, split: str) -> dict[str, Any]:
    """Load only train or validation; test is intentionally not a valid selector."""
    if split not in LOADABLE_SPLITS:
        raise LearnedDatasetV4Error("load_split accepts only train or validation; use load_evaluation_split for test")
    return _load_split(Path(path).resolve(), split)


def load_evaluation_split(path: Path | str) -> dict[str, Any]:
    """Explicit test-only API for evaluation callers."""
    return _load_split(Path(path).resolve(), "test")


__all__ = ["LearnedDatasetV4Error", "build_dataset", "read_preregistration", "load_split",
           "load_evaluation_split", "sha256", "ARRAY_FIELDS", "SPLITS"]
