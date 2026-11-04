"""Auditable, epoch-disjoint single-channel EEGdenoiseNet v2 dataset builder.

This module is independent of PyTorch, v1, and the v2 inference/training core.
Only published clean EEG epochs are targets; EOG/EMG are independent noise pools.
The source release has no subject IDs, so its split is NOT subject-disjoint.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "datasets" / "external" / "eegdenoisenet"
DEFAULT_OUTPUT = ROOT / "datasets" / "processed" / "learned_denoiser_v2.npz"
DEFAULT_SEED = 20261005
SPLITS = ("train", "validation", "test")
MODALITIES = ("eeg", "eog", "emg")
SOURCE_RATE_HZ = 256
TARGET_RATE_HZ = 250
SOURCE_SAMPLES = 512
TARGET_SAMPLES = 500


class DatasetV2Error(ValueError):
    """Invalid source epochs, build parameters, or an unsafe output path."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allocate_counts(total: int, fractions: Sequence[float], *, nonempty: bool = False) -> list[int]:
    """Largest-remainder apportionment; small pools may need a minimum of one."""
    quotas = np.asarray(fractions, dtype=np.float64) * total
    counts = np.floor(quotas).astype(np.int64)
    for index in sorted(range(3), key=lambda i: (-(quotas[i] - counts[i]), i))[:total - int(counts.sum())]:
        counts[index] += 1
    if nonempty and total >= 3:
        for index in range(3):
            if counts[index] == 0:
                donor = max(range(3), key=lambda i: (counts[i], -i))
                counts[donor] -= 1
                counts[index] += 1
    return [int(value) for value in counts]


def _validate_parameters(samples: int, seed: int, identity_fraction: float,
                         snr_min_db: float, snr_max_db: float,
                         fractions: Sequence[float]) -> tuple[float, ...]:
    if type(samples) is not int or samples < 3:
        raise DatasetV2Error("samples must be an integer >= 3 (all three sample splits must be nonempty)")
    if type(seed) is not int or seed < 0:
        raise DatasetV2Error("seed must be a non-negative integer")
    if isinstance(identity_fraction, bool) or not isinstance(identity_fraction, Real) or not math.isfinite(identity_fraction) or not 0 <= identity_fraction <= 1:
        raise DatasetV2Error("identity_fraction must be finite and in [0, 1]")
    if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in (snr_min_db, snr_max_db)):
        raise DatasetV2Error("SNR bounds must be finite numbers")
    if snr_min_db > snr_max_db:
        raise DatasetV2Error("snr_min_db cannot exceed snr_max_db")
    try:
        endpoint_scales = [10.0 ** (-float(value) / 20.0) for value in (snr_min_db, snr_max_db)]
    except OverflowError as error:
        raise DatasetV2Error("SNR bounds produce unrepresentable noise amplitude") from error
    if any(not math.isfinite(scale) or scale <= 0 for scale in endpoint_scales):
        raise DatasetV2Error("SNR bounds produce zero or nonfinite noise amplitude")
    if not isinstance(fractions, (tuple, list, np.ndarray)) or len(fractions) != 3 or any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0 for value in fractions):
        raise DatasetV2Error("fractions must contain three finite, positive numbers")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise DatasetV2Error("fractions must sum to 1")
    return tuple(float(value) / sum(fractions) for value in fractions)


def _center_and_rms(epochs: np.ndarray, description: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            centered = epochs - np.mean(epochs, axis=1, keepdims=True)
            rms = np.sqrt(np.mean(centered ** 2, axis=1))
    except FloatingPointError as error:
        raise DatasetV2Error(f"{description}: invalid/overflowing AC RMS") from error
    bad = ~np.isfinite(rms) | (rms <= 0)
    if np.any(bad):
        ids = np.flatnonzero(bad).tolist()
        raise DatasetV2Error(f"{description}: zero or invalid AC RMS at epoch IDs {ids[:10]}")
    return centered, rms


def _load_source(source_dir: Path, modality: str) -> tuple[np.ndarray, dict[str, Any]]:
    variable = f"{modality.upper()}_all_epochs"
    npy_path = source_dir / f"{variable}.npy"
    mat_path = source_dir / f"{variable}.mat"
    rate_evidence: dict[str, Any]
    if npy_path.is_file():
        source_path = npy_path
        try:
            raw = np.load(npy_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise DatasetV2Error(f"cannot read NPY source {npy_path}: {error}") from error
        if mat_path.is_file():
            try:
                metadata = loadmat(mat_path, variable_names=["fs"])
                fs = np.asarray(metadata.get("fs", [])).reshape(-1)
            except (OSError, ValueError, NotImplementedError) as error:
                raise DatasetV2Error(f"cannot read MAT rate metadata {mat_path}: {error}") from error
            if fs.size != 1 or not np.isfinite(fs[0]) or float(fs[0]) != SOURCE_RATE_HZ:
                raise DatasetV2Error(f"{mat_path}: fs must be {SOURCE_RATE_HZ} Hz")
            rate_evidence = {"kind": "companion_mat_fs", "path": str(mat_path), "sha256": _sha256(mat_path), "fs_hz": int(fs[0])}
        else:
            # NPY stores no rate. State the contract assumption, never pretend
            # it was measured or independently read from nonexistent metadata.
            rate_evidence = {"kind": "EEGdenoiseNet_512_sample_2_second_contract", "assumed": True, "fs_hz": SOURCE_RATE_HZ}
    elif mat_path.is_file():
        source_path = mat_path
        try:
            content = loadmat(mat_path, variable_names=[variable, "fs"])
        except (OSError, ValueError, NotImplementedError) as error:
            raise DatasetV2Error(f"cannot read MAT source {mat_path}: {error}") from error
        if variable not in content:
            raise DatasetV2Error(f"{mat_path}: missing variable {variable}")
        raw = content[variable]
        fs = np.asarray(content.get("fs", [])).reshape(-1)
        if fs.size != 1 or not np.isfinite(fs[0]) or float(fs[0]) != SOURCE_RATE_HZ:
            raise DatasetV2Error(f"{mat_path}: fs must be {SOURCE_RATE_HZ} Hz")
        rate_evidence = {"kind": "source_mat_fs", "fs_hz": int(fs[0])}
    else:
        raise DatasetV2Error(f"source missing: expected {npy_path} or {mat_path}")
    if raw.ndim != 2 or raw.shape[0] < 3 or raw.shape[1] != SOURCE_SAMPLES:
        raise DatasetV2Error(f"{source_path}: expected [epochs >= 3, {SOURCE_SAMPLES}], got {raw.shape}")
    if raw.dtype.kind not in "fiu":
        raise DatasetV2Error(f"{source_path}: epochs must be real numeric values, not {raw.dtype}")
    epochs = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(epochs)):
        invalid_ids = np.flatnonzero(~np.all(np.isfinite(epochs), axis=1)).tolist()
        raise DatasetV2Error(f"{source_path}: nonfinite input at epoch IDs {invalid_ids[:10]}")
    _center_and_rms(epochs, str(source_path))
    metadata = {
        "path": str(source_path), "sha256": _sha256(source_path),
        "format": source_path.suffix[1:], "shape": [int(value) for value in raw.shape],
        "dtype": str(raw.dtype), "sample_rate_hz": SOURCE_RATE_HZ,
        "sample_rate_evidence": rate_evidence, "epoch_duration_s": 2.0,
        "epoch_id_semantics": "zero-based original row index in the source matrix",
    }
    return epochs, metadata


def _fingerprint(epoch: np.ndarray) -> str:
    canonical = np.array(epoch, dtype="<f8", copy=True)
    canonical[canonical == 0] = 0.0  # +0 and -0 are numerically identical.
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _partition_sources(raw_pools: dict[str, np.ndarray], fractions: Sequence[float],
                       rng: np.random.Generator) -> tuple[dict[str, dict[str, list[int]]], dict[str, Any]]:
    # The namespace is GLOBAL across EEG/EOG/EMG. Even a waveform copied
    # between different modalities must have a single split assignment.
    grouped: dict[str, dict[str, list[int]]] = {}
    for modality in MODALITIES:
        for epoch_id, epoch in enumerate(raw_pools[modality]):
            grouped.setdefault(_fingerprint(epoch), {}).setdefault(modality, []).append(epoch_id)
    groups = [{"fingerprint": fp, "epoch_ids_by_source": grouped[fp]} for fp in sorted(grouped)]
    buckets: dict[tuple[str, ...], list[int]] = {}
    for index, group in enumerate(groups):
        signature = tuple(kind for kind in MODALITIES if kind in group["epoch_ids_by_source"])
        buckets.setdefault(signature, []).append(index)
    for signature in sorted(buckets):
        bucket = np.asarray(buckets[signature], dtype=np.int64)
        rng.shuffle(bucket)
        counts = _allocate_counts(len(bucket), fractions, nonempty=True)
        cursor = 0
        for split, count in zip(SPLITS, counts):
            for group_id in bucket[cursor:cursor + count]:
                groups[int(group_id)]["split"] = split
            cursor += count

    def coverage() -> dict[str, dict[str, int]]:
        counts = {split: {kind: 0 for kind in MODALITIES} for split in SPLITS}
        for group in groups:
            for kind in group["epoch_ids_by_source"]:
                counts[group["split"]][kind] += 1
        return counts

    # Small mixed-modality fingerprint strata can need a coverage repair.
    # Move a whole group only if every affected donor pool stays nonempty.
    for split in SPLITS:
        for modality in MODALITIES:
            counts = coverage()
            if counts[split][modality]:
                continue
            candidates = [group for group in groups if modality in group["epoch_ids_by_source"]
                          and group["split"] != split
                          and all(counts[group["split"]][kind] > 1 for kind in group["epoch_ids_by_source"])]
            if not candidates:
                raise DatasetV2Error(f"cannot allocate nonempty {modality} pools to all splits; need >= 3 independent fingerprint groups")
            candidates[0]["split"] = split
    pools = {split: {kind: [] for kind in MODALITIES} for split in SPLITS}
    for group in groups:
        for modality, ids in group["epoch_ids_by_source"].items():
            pools[group["split"]][modality].extend(ids)
    for split in SPLITS:
        for modality in MODALITIES:
            pools[split][modality].sort()
    modality_stats = {}
    for modality in MODALITIES:
        group_count = sum(modality in group["epoch_ids_by_source"] for group in groups)
        modality_stats[modality] = {
            "raw_epochs": int(raw_pools[modality].shape[0]), "unique_fingerprint_groups": group_count,
            "duplicate_epochs": int(raw_pools[modality].shape[0]) - group_count,
            "raw_epoch_ids_by_split": {split: pools[split][modality] for split in SPLITS},
        }
    audit = {
        "fingerprint_algorithm": "sha256 of original 512 samples as little-endian float64; signed zero canonicalized",
        "fingerprint_namespace": "global_across_EEG_EOG_EMG",
        "split_before_resampling_or_mixing": True,
        "partition_method": "seeded membership-stratified fingerprint groups; largest remainder plus nonempty coverage repair",
        "fingerprint_groups": groups, "source_statistics": modality_stats,
        "unique_global_groups": len(groups),
        "duplicate_global_epochs": sum(len(pool) for pool in raw_pools.values()) - len(groups),
    }
    return pools, audit


def _prepare_epochs(raw: np.ndarray, modality: str) -> np.ndarray:
    centered, _ = _center_and_rms(raw, modality)
    # Exactly one source epoch -> one output epoch. Never concatenate epochs,
    # tile samples, or broadcast single-channel data into an 8-channel montage.
    resampled = resample_poly(centered, up=125, down=128, axis=1)
    if resampled.shape != (raw.shape[0], TARGET_SAMPLES) or not np.all(np.isfinite(resampled)):
        raise DatasetV2Error(f"{modality}: invalid resampling result")
    centered, rms = _center_and_rms(resampled, f"resampled {modality}")
    normalized = centered / rms[:, None]
    if not np.all(np.isfinite(normalized)):
        raise DatasetV2Error(f"{modality}: invalid normalized epochs")
    return normalized


def _identity_counts(sample_counts: Sequence[int], fraction: float) -> list[int]:
    total_identity = int(math.floor(sum(sample_counts) * fraction + 0.5))
    quotas = np.asarray(sample_counts, dtype=np.float64) * fraction
    counts = np.floor(quotas).astype(np.int64)
    for index in sorted(range(3), key=lambda i: (-(quotas[i] - counts[i]), i))[:total_identity - int(counts.sum())]:
        counts[index] += 1
    return [int(value) for value in counts]


def _write_outputs(output: Path, arrays: dict[str, np.ndarray], manifest: dict[str, Any],
                   overwrite: bool) -> tuple[Path, str]:
    manifest_path = Path(str(output) + ".manifest.json")
    if not overwrite and (output.exists() or manifest_path.exists()):
        raise DatasetV2Error(f"output already exists (use overwrite explicitly): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pending: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=output.parent, prefix=output.name + ".", suffix=".pending", delete=False) as handle:
            pending = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        digest = _sha256(pending)
        manifest["output"] = {"path": str(output), "sha256": digest, "manifest_path": str(manifest_path)}
        # Check JSON compatibility BEFORE replacing either output; no NumPy
        # objects, Infinity, or NaN are allowed in audit metadata.
        content = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        pending.replace(output)
        pending = None
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, prefix=manifest_path.name + ".", suffix=".pending", delete=False) as handle:
            pending = Path(handle.name)
            handle.write(content)
        pending.replace(manifest_path)
        pending = None
        return manifest_path, digest
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def build_dataset_v2(
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
    output: Path | str = DEFAULT_OUTPUT,
    *, samples: int = 12000, seed: int = DEFAULT_SEED,
    identity_fraction: float = 0.4, snr_min_db: float = -7.0, snr_max_db: float = 20.0,
    fractions: Sequence[float] = (0.70, 0.15, 0.15), overwrite: bool = False,
) -> dict[str, Any]:
    """Write NPZ + ``<output>.manifest.json`` and return a JSON-safe report.

    Targets are [N,1,500] float32 clean EEG (zero-mean, AC RMS=1).
    Sample counts are allocated FIRST by largest remainder; within each split,
    independent seeded draws mix only raw IDs from that split. Clean examples
    are identities; artifact examples use exactly one EOG OR EMG whole epoch.
    No recording/session/subject-disjoint generalization is asserted.
    """
    fractions = _validate_parameters(samples, seed, identity_fraction, snr_min_db, snr_max_db, fractions)
    identity_fraction, snr_min_db, snr_max_db = map(float, (identity_fraction, snr_min_db, snr_max_db))
    sample_counts = _allocate_counts(samples, fractions, nonempty=True)
    identities = _identity_counts(sample_counts, identity_fraction)
    source_dir = Path(source_dir).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not source_dir.is_dir():
        raise DatasetV2Error(f"source directory does not exist: {source_dir}")
    if output.suffix.lower() != ".npz":
        raise DatasetV2Error("output must be an .npz path")
    manifest_path = Path(str(output) + ".manifest.json")
    if output.is_relative_to(source_dir) or manifest_path.is_relative_to(source_dir):
        raise DatasetV2Error("output must be outside the source directory; sources are immutable")
    if not overwrite and (output.exists() or manifest_path.exists()):
        raise DatasetV2Error(f"output already exists (use overwrite explicitly): {output}")
    raw_pools: dict[str, np.ndarray] = {}
    sources = {}
    for modality in MODALITIES:
        raw_pools[modality], sources[modality] = _load_source(source_dir, modality)
    children = np.random.SeedSequence(seed).spawn(4)
    split_pools, epoch_audit = _partition_sources(raw_pools, fractions, np.random.default_rng(children[0]))
    prepared = {kind: _prepare_epochs(raw_pools[kind], kind) for kind in MODALITIES}
    arrays = {
        "clean": np.empty((samples, 1, TARGET_SAMPLES), dtype=np.float32),
        "noisy": np.empty((samples, 1, TARGET_SAMPLES), dtype=np.float32),
        "valid_mask": np.ones((samples, 1, TARGET_SAMPLES), dtype=bool),
        "artifact_mask": np.zeros((samples, 1, TARGET_SAMPLES), dtype=bool),
        "split": np.empty(samples, dtype="U10"),
        "clean_epoch_id": np.full(samples, -1, dtype=np.int64),
        "eog_epoch_id": np.full(samples, -1, dtype=np.int64),
        "emg_epoch_id": np.full(samples, -1, dtype=np.int64),
        "artifact_type": np.empty(samples, dtype="U5"),
    }
    sample_snr: list[float | None] = [None] * samples
    statistics: dict[str, Any] = {}
    offset = 0
    for split_index, (split, count, identity_count) in enumerate(zip(SPLITS, sample_counts, identities)):
        rng = np.random.default_rng(children[split_index + 1])
        artifact_count = count - identity_count
        eog_count = artifact_count // 2
        emg_count = artifact_count - eog_count
        labels = np.asarray(["clean"] * identity_count + ["eog"] * eog_count + ["emg"] * emg_count, dtype="U5")
        rng.shuffle(labels)
        for within, artifact_type in enumerate(labels):
            index = offset + within
            clean_id = int(rng.choice(split_pools[split]["eeg"]))
            clean = prepared["eeg"][clean_id]
            arrays["clean"][index, 0] = clean
            arrays["split"][index] = split
            arrays["clean_epoch_id"][index] = clean_id
            arrays["artifact_type"][index] = artifact_type
            if artifact_type == "clean":
                arrays["noisy"][index, 0] = arrays["clean"][index, 0]
            else:
                kind = str(artifact_type)
                artifact_id = int(rng.choice(split_pools[split][kind]))
                snr_db = float(rng.uniform(snr_min_db, snr_max_db))
                amplitude = float(10.0 ** (-snr_db / 20.0))
                noisy = clean + prepared[kind][artifact_id] * amplitude
                arrays["noisy"][index, 0] = noisy
                arrays[f"{kind}_epoch_id"][index] = artifact_id
                arrays["artifact_mask"][index, 0] = True
                sample_snr[index] = snr_db
        values = [value for value in sample_snr[offset:offset + count] if value is not None]
        statistics[split] = {
            "samples": count, "target_fraction": fractions[split_index],
            "identity_samples": identity_count, "identity_fraction_actual": identity_count / count,
            "eog_samples": eog_count, "emg_samples": emg_count,
            "snr_draw_min_db": min(values) if values else None,
            "snr_draw_max_db": max(values) if values else None,
            "snr_draw_mean_db": sum(values) / len(values) if values else None,
            "original_pool_sizes": {kind: len(split_pools[split][kind]) for kind in MODALITIES},
        }
        offset += count
    if not np.all(np.isfinite(arrays["clean"])) or not np.all(np.isfinite(arrays["noisy"])):
        raise DatasetV2Error("nonfinite generated signal (check SNR bounds and source amplitudes)")
    manifest = {
        "schema_version": 2, "status": "formal", "dataset_version": "learned_denoiser_v2",
        "source_dir": str(source_dir), "sources": sources, "seed": seed,
        "requested_samples": samples, "split_fractions": dict(zip(SPLITS, fractions)),
        "split_sample_targets": dict(zip(SPLITS, sample_counts)),
        "sampling_semantics": {
            "generalization_claim": "epoch-disjoint semi-synthetic",
            "subject_identifiers_available": False, "subject_disjoint_claim": False,
            "clean_target": "published EEGdenoiseNet clean EEG epoch; not EOG/EMG and not automatically accepted local recordings",
            "original_epochs_are_synchronous_pairs": False,
            "epoch_draws": "uniform with replacement within each raw fingerprint split; independent per-split RNG streams",
            "epoch_concatenation": False, "channel_repetition": False,
            "channel_count": 1, "epoch_duration_s": 2.0,
            "identity_fraction_requested": identity_fraction,
            "identity_count": sum(identities), "identity_count_rounding": "nearest integer (half up), apportioned by largest remainder",
            "artifact_types": ["eog", "emg"], "artifact_combination": "one EOG OR EMG, never both",
            "artifact_mask": "identity: all false; synthesized artifact epoch: all true",
        },
        "preprocessing": {
            "source_rate_hz": SOURCE_RATE_HZ, "target_rate_hz": TARGET_RATE_HZ,
            "source_samples": SOURCE_SAMPLES, "target_samples": TARGET_SAMPLES,
            "resampling": "scipy.signal.resample_poly(up=125, down=128)",
            "demean": "each raw epoch before resampling and each resampled epoch afterwards",
            "clean_scale": "unit AC RMS after resampling, shared across identity and artifact targets",
            "noise_scale": "zero-mean unit AC RMS artifact scaled by 10**(-SNR_db/20)",
            "snr_definition": "20*log10(AC_RMS(clean)/AC_RMS(noisy-clean))",
            "snr_distribution": "independent uniform per artifact sample", "snr_min_db": snr_min_db, "snr_max_db": snr_max_db,
        },
        "epoch_split_manifest": epoch_audit, "split_statistics": statistics,
        "sample_snr_db": sample_snr,
        "npz_contract": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
    }
    manifest_path, digest = _write_outputs(output, arrays, manifest, overwrite)
    return {
        "status": "formal", "result": "completed", "output": str(output),
        "manifest": str(manifest_path), "npz_sha256": digest,
        "samples": samples, "shape": [samples, 1, TARGET_SAMPLES],
        "sampling_rate_hz": TARGET_RATE_HZ, "split_sample_targets": dict(zip(SPLITS, sample_counts)),
        "generalization_claim": "epoch-disjoint semi-synthetic",
    }


__all__ = ["DatasetV2Error", "build_dataset_v2", "DEFAULT_SOURCE_DIR", "DEFAULT_OUTPUT", "DEFAULT_SEED"]
