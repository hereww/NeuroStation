"""Prepare exploratory SSVEP NPZ+manifest from an explicitly verified JSON spec.

No network access, model loading, training, deployment, filtering, resampling,
channel repair, axis inference, frequency rounding, or default Gu metadata.
Only numeric MATLAB v4/v5 files supported by scipy.io are accepted; v7.3/HDF5,
structs and other schemas are rejected, never guessed.

Required source JSON (schema_version=1, status="formal"):
  sample_rate_hz: 250
  mat_variable: exact numeric MAT variable name
  expected_shape: four positive integers IN ORIGINAL MAT AXIS ORDER
  axes: {"channel": int, "sample": int, "condition": int, "block": int}
  channel_names: 64 unique, exact electrode names in original channel order
  subject_id: nonempty string or finite number
  condition_table: one row per original condition, with exactly
    {"condition_index": zero-based int, "frequency_hz": number,
     "depth": nonempty string or finite number,
     "session": nonempty string or finite number}
Optional sample_window: {"start": inclusive int, "stop": exclusive int}.
Without an explicit window all source samples are preserved: a 1285-sample
record MUST NOT be described as 5 s at 250 Hz. A 5 s crop needs an officially
verified start and stop, exactly 1250 samples apart, in sample_window.

The spec is a caller assertion, NOT automatic verification against the paper.
All four exact targets and all eight actual electrode names are required.
Missing targets are available in PreparationError.audit; no partial output.
Subject/depth/session labels use JSON scalar strings to retain their types.
Subject/block/depth/session groups and condition identities are retained;
there are no train/validation/test splits or automatic training entry points.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.io import loadmat, whosmat
from scipy.io.matlab import MatReadError

TARGET_FREQUENCIES_HZ = (10.0, 12.0, 15.0, 20.0)
TARGET_CHANNEL_NAMES = ("Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2")
AXIS_NAMES = ("channel", "sample", "condition", "block")
MAX_SPEC_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_ARRAY_BYTES = 512 * 1024 * 1024
FLAGS = {
    "status": "formal", "exploratory_only": True, "runtime_enabled": False,
    "production_validated": False, "default_deployment_authorized": False,
    "training_executed": False, "deployment_executed": False,
}
MAT_ITEM_BYTES = {
    "double": 8, "single": 4, "int8": 1, "uint8": 1, "int16": 2,
    "uint16": 2, "int32": 4, "uint32": 4, "int64": 8, "uint64": 8,
}


class MatchedSSVEPPreparationError(ValueError):
    """Validation failure; audit includes missing targets when applicable."""

    def __init__(self, message: str, *, audit: dict[str, Any] | None = None):
        super().__init__(message)
        self.audit = audit


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatchedSSVEPPreparationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _number(value: Any, label: str) -> float:
    _require(type(value) in (int, float), f"{label} must be a JSON number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise MatchedSSVEPPreparationError(f"{label} must be finite") from exc
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _identity(value: Any, label: str) -> None:
    if isinstance(value, str):
        _require(bool(value.strip()), f"{label} must be nonempty")
    else:
        _number(value, label)


def validate_source_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate schema strictly and detach from caller-owned nested objects."""
    required = {"schema_version", "status", "sample_rate_hz", "mat_variable",
                "expected_shape", "axes", "channel_names", "subject_id", "condition_table"}
    _require(isinstance(spec, Mapping), "source spec must be a JSON object")
    _require(required <= set(spec), f"missing source-spec fields: {sorted(required - set(spec))}")
    _require(set(spec) <= required | {"sample_window"}, "unknown source-spec fields")
    _require(type(spec["schema_version"]) is int and spec["schema_version"] == 1,
             "unknown schema_version; only 1 is supported")
    _require(spec["status"] == "formal", "source spec status must be formal")
    _require(_number(spec["sample_rate_hz"], "sample_rate_hz") == 250,
             "sample_rate_hz must be exactly 250; resampling is forbidden")
    name = spec["mat_variable"]
    _require(isinstance(name, str) and bool(name.strip()) and not name.startswith("__"),
             "mat_variable must be an explicit numeric MAT variable name")
    axes = spec["axes"]
    _require(isinstance(axes, Mapping) and set(axes) == set(AXIS_NAMES),
             "axes must contain exactly channel/sample/condition/block")
    _require(all(type(axis) is int for axis in axes.values())
             and sorted(axes.values()) == [0, 1, 2, 3],
             "axes must be a permutation of integer axes 0..3")
    shape = spec["expected_shape"]
    _require(isinstance(shape, list) and len(shape) == 4
             and all(type(size) is int and size > 0 for size in shape),
             "expected_shape must contain four positive integers in source axis order")
    _require(shape[axes["channel"]] == 64, "source channel axis must have exactly 64 channels")
    names = spec["channel_names"]
    _require(isinstance(names, list) and len(names) == 64
             and all(isinstance(n, str) and bool(n.strip()) and n == n.strip() for n in names),
             "channel_names must contain 64 exact nonempty electrode names")
    _require(len(set(n.casefold() for n in names)) == 64, "duplicate electrode names")
    _identity(spec["subject_id"], "subject_id")
    rows = spec["condition_table"]
    _require(isinstance(rows, list) and len(rows) == shape[axes["condition"]],
             "condition_table must cover the entire original condition axis")
    indices = []
    for row in rows:
        _require(isinstance(row, Mapping) and set(row) == {
            "condition_index", "frequency_hz", "depth", "session"},
            "condition rows must contain exactly condition_index/frequency_hz/depth/session")
        index = row["condition_index"]
        _require(type(index) is int, "condition_index must be an integer")
        indices.append(index)
        frequency = _number(row["frequency_hz"], "frequency_hz")
        _require(0 < frequency < 125, "frequency_hz must be positive and below Nyquist")
        _identity(row["depth"], "depth")
        _identity(row["session"], "session")
    _require(sorted(indices) == list(range(len(rows))),
             "condition_index must cover the complete zero-based range without duplicates")
    if "sample_window" in spec:
        window = spec["sample_window"]
        _require(isinstance(window, Mapping) and set(window) == {"start", "stop"},
                 "sample_window must contain exactly start/stop")
        _require(all(type(value) is int for value in window.values()),
                 "sample_window start/stop must be integers")
        _require(0 <= window["start"] < window["stop"] <= shape[axes["sample"]],
                 "sample_window must be within source sample bounds")
    result = json.loads(_json(spec))
    result["condition_table"].sort(key=lambda row: row["condition_index"])
    return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_source_spec(path: Path | str) -> dict[str, Any]:
    """Read only the JSON supplied by the caller, with no fallback metadata."""
    path = Path(path)
    with path.open("rb") as handle:
        raw = handle.read(MAX_SPEC_BYTES + 1)
    _require(len(raw) <= MAX_SPEC_BYTES, "source spec exceeds size bound")
    return validate_source_spec(json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object))


def _match(spec: Mapping[str, Any]) -> tuple[list[int], list[dict[str, Any]], dict[str, Any]]:
    names = spec["channel_names"]
    rows = [row for row in spec["condition_table"] if row["frequency_hz"] in TARGET_FREQUENCIES_HZ]
    available = sorted(set(row["frequency_hz"] for row in spec["condition_table"]))
    audit = {
        "requested_channel_names": list(TARGET_CHANNEL_NAMES),
        "missing_channel_names": [name for name in TARGET_CHANNEL_NAMES if name not in names],
        "requested_frequencies_hz": list(TARGET_FREQUENCIES_HZ),
        "available_frequencies_hz": available,
        "missing_target_frequencies_hz": [f for f in TARGET_FREQUENCIES_HZ if f not in available],
        "frequency_match": "exact; no rounding or nearest-frequency substitution",
    }
    if audit["missing_channel_names"] or audit["missing_target_frequencies_hz"]:
        raise MatchedSSVEPPreparationError("missing exact target channels or frequencies", audit=audit)
    return [names.index(name) for name in TARGET_CHANNEL_NAMES], rows, audit


def _stamp(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def _load_mat_variable(path: Path, spec: Mapping[str, Any], max_array_bytes: int) -> np.ndarray:
    try:
        entries = [entry for entry in whosmat(path) if entry[0] == spec["mat_variable"]]
        _require(len(entries) == 1, "explicit MAT variable is missing or ambiguous")
        _, shape, matlab_class = entries[0]
        _require(list(shape) == spec["expected_shape"], "MAT header shape does not match expected_shape")
        _require(matlab_class in MAT_ITEM_BYTES, "unsupported MAT variable class; numeric dense data required")
        # A factor of two also bounds a complex source before it is rejected.
        _require(math.prod(shape) * MAT_ITEM_BYTES[matlab_class] * 2 <= max_array_bytes,
                 "MAT variable exceeds max_array_bytes before loading")
        loaded = loadmat(path, variable_names=[spec["mat_variable"]],
                         squeeze_me=False, verify_compressed_data_integrity=True)
    except (NotImplementedError, OSError, ValueError, TypeError, MatReadError,
            EOFError, IndexError, struct.error, zlib.error) as exc:
        if isinstance(exc, MatchedSSVEPPreparationError):
            raise
        raise MatchedSSVEPPreparationError(f"unsupported or unreadable MAT schema: {exc}") from exc
    data = loaded.get(spec["mat_variable"])
    _require(isinstance(data, np.ndarray) and data.dtype.kind in "fiu" and data.ndim == 4,
             "MAT variable must be a real numeric dense 4D array")
    _require(list(data.shape) == spec["expected_shape"], "loaded MAT shape differs from formal spec")
    _require(data.nbytes <= max_array_bytes, "loaded MAT variable exceeds max_array_bytes")
    # Chunked validation avoids a second full-size boolean allocation.
    iterator = np.nditer(data, flags=["external_loop", "buffered"], buffersize=65536, order="K")
    _require(all(np.isfinite(chunk).all() for chunk in iterator), "MAT contains non-finite data")
    return data


def prepare_matched_ssvep_training(
    mat_path: Path | str,
    source_spec: Mapping[str, Any],
    output_dir: Path | str,
    *,
    mat_variable: str | None = None,
    axis_spec: Mapping[str, int] | None = None,
    max_array_bytes: int = DEFAULT_MAX_ARRAY_BYTES,
) -> dict[str, Any]:
    """Materialize one source subject into an exclusively NEW output directory.

    Explicit mat_variable/axis_spec API arguments must agree with the formal
    JSON. They cannot override its meaning. All indices in outputs are zero
    based; samples and numeric dtype are preserved exactly. max_array_bytes
    bounds decoded source AND selected EEG payloads, independently.
    """
    spec = validate_source_spec(source_spec)
    _require(mat_variable is None or mat_variable == spec["mat_variable"], "mat_variable does not match spec")
    _require(axis_spec is None or axis_spec == spec["axes"], "axis_spec does not match spec")
    _require(type(max_array_bytes) is int and max_array_bytes > 0, "max_array_bytes must be a positive integer")
    mat_path, output_dir = Path(mat_path).resolve(), Path(output_dir)
    _require(mat_path.is_file(), "MAT source must be an existing file")
    # Check the supplied path before resolving: a dangling symlink is also an
    # existing caller-owned output, even when its destination does not exist.
    _require(not output_dir.exists() and not output_dir.is_symlink(),
             "refusing to overwrite: output directory must be new")
    output_dir = output_dir.resolve()
    project_root = Path(__file__).resolve().parents[1]
    _require(not any(output_dir.is_relative_to(project_root / name) for name in ("configs", "models")),
             "output must not be inside project configs or models")
    channels, selected_rows, audit = _match(spec)
    initial_stamp = _stamp(mat_path)
    source_sha = _sha256(mat_path)
    data = _load_mat_variable(mat_path, spec, max_array_bytes)
    canonical = np.transpose(data, tuple(spec["axes"][name] for name in AXIS_NAMES))
    window = spec.get("sample_window", {"start": 0, "stop": canonical.shape[1]})
    start, stop = window["start"], window["stop"]
    blocks = canonical.shape[3]
    trial_count = len(selected_rows) * blocks
    _require(trial_count * 8 * (stop - start) * data.dtype.itemsize <= max_array_bytes,
             "selected EEG exceeds max_array_bytes")
    eeg = np.empty((trial_count, 8, stop - start), dtype=data.dtype)
    labels: dict[str, list[Any]] = {key: [] for key in (
        "subject", "block", "depth", "session", "frequency_hz", "condition_index", "group_id")}
    for row in selected_rows:
        for block in range(blocks):
            trial = len(labels["block"])
            # No intermediate copy of all conditions/channels is necessary.
            eeg[trial] = canonical[channels, start:stop, row["condition_index"], block]
            group = [spec["subject_id"], block, row["depth"], row["session"]]
            labels["subject"].append(_json(spec["subject_id"]))
            labels["block"].append(block)
            labels["depth"].append(_json(row["depth"]))
            labels["session"].append(_json(row["session"]))
            labels["frequency_hz"].append(row["frequency_hz"])
            labels["condition_index"].append(row["condition_index"])
            # JSON tuple preserves identity even for numeric 1 vs string "1".
            labels["group_id"].append(_json(group))
    _require(_stamp(mat_path) == initial_stamp and _sha256(mat_path) == source_sha,
             "MAT source changed during preparation")
    spec_sha = hashlib.sha256(_json(spec).encode("utf-8")).hexdigest()
    manifest = {
        **FLAGS, "schema_version": 1,
        "source_file": str(mat_path), "source_sha256": source_sha,
        "source_spec_sha256": spec_sha, "source_spec": spec,
        "metadata_authority": "caller-provided formal spec; no official metadata inferred",
        "original_mat_shape": list(data.shape), "mat_variable": spec["mat_variable"],
        "axis_spec": spec["axes"], "canonical_axis_order": list(AXIS_NAMES),
        "sample_rate_hz": 250, "source_sample_count": canonical.shape[1],
        "sample_window": window, "sample_count": stop - start,
        "duration_seconds": (stop - start) / 250,
        "selected_channel_names": list(TARGET_CHANNEL_NAMES),
        "source_channel_indices": channels, "index_base": 0,
        "true_frequencies_hz": sorted(set(row["frequency_hz"] for row in selected_rows)),
        "missing_target_audit": audit, "selected_conditions": selected_rows,
        "subject_id": spec["subject_id"], "block_count": blocks,
        "eeg_shape": list(eeg.shape), "eeg_axes": ["trial", "channel", "sample"],
        "dtype": str(eeg.dtype), "trial_order": "condition-major_then_block-major",
        "group_fields": ["subject", "block", "depth", "session"],
        "identity_label_encoding": "subject/depth/session are JSON scalars preserving original types",
        "group_id_encoding": "JSON array of original typed subject/block/depth/session labels",
        "split_assigned": False, "raw_reference_is_clean_ground_truth": False,
        "signal_processing_applied": False, "training_authorized": False,
    }
    # Exclusive creation is the no-overwrite gate, including concurrent runs.
    # Never recursively delete or rename a caller-owned output directory.
    output_dir.mkdir(parents=False, exist_ok=False)
    created: list[Path] = []
    npz_path, manifest_path = output_dir / "matched_ssvep_training.npz", output_dir / "manifest.json"
    try:
        with npz_path.open("xb") as handle:
            created.append(npz_path)
            np.savez_compressed(handle, eeg=eeg,
                **{key: np.asarray(value) for key, value in labels.items()},
                sample_rate_hz=np.asarray(250), channel_names=np.asarray(TARGET_CHANNEL_NAMES),
                source_channel_indices=np.asarray(channels, dtype=np.int64),
                source_sha256=np.asarray(source_sha), source_spec_sha256=np.asarray(spec_sha),
                source_sample_indices=np.arange(start, stop, dtype=np.int64),
                missing_target_frequencies_hz=np.asarray(audit["missing_target_frequencies_hz"], dtype=np.float64),
                **{key: np.asarray(value) for key, value in FLAGS.items()})
        manifest["npz_file"] = npz_path.name
        manifest["npz_sha256"] = _sha256(npz_path)
        with manifest_path.open("x", encoding="utf-8") as handle:
            created.append(manifest_path)
            json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        try:
            output_dir.rmdir()
        except OSError:
            pass
        raise
    return manifest


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--source-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="must not exist; parent must exist")
    parser.add_argument("--max-array-bytes", type=int, default=DEFAULT_MAX_ARRAY_BYTES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        manifest = prepare_matched_ssvep_training(args.mat, load_source_spec(args.source_spec),
            args.output_dir, max_array_bytes=args.max_array_bytes)
    except (ValueError, OSError, TypeError) as exc:
        print(json.dumps({**FLAGS, "error": str(exc),
                          "audit": getattr(exc, "audit", None)}, ensure_ascii=False, allow_nan=False))
        return 2
    print(json.dumps({**FLAGS, "output_dir": str(args.output_dir.resolve()),
        "eeg_shape": manifest["eeg_shape"], "source_sha256": manifest["source_sha256"],
        "missing_target_audit": manifest["missing_target_audit"]}, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
