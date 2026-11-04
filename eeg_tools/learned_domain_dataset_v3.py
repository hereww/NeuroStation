"""Frozen v3 domain data: supported semisynthetic EEG versus unsupported real EEG.

Only normalized single-channel signals are features. Provenance never enters the
network. Training entry points cannot select test or S4/S5. No window exclusion,
interpolation, resampling, denoising, clean-target inference or raw-file writes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from eeg_tools.learned_denoise_v2 import LearnedDenoiseV2Error, normalize_window

SUPPORT_NPZ = "datasets/processed/learned_denoiser_v2.npz"
DONG_DIR = "datasets/external/ssvep_datasets/dong2023_ssvep"
DONG_METADATA = "datasets/external/v2_metadata_audit/DONG2023_METADATA.json"
PREREGISTRATION = "analysis_outputs/learned_denoiser_v3_preregistration.json"
TRAIN_SUBJECTS = ("S6.mat", "S7.mat")
VALIDATION_SUBJECTS = ("S8.mat", "S9.mat")
HOLDOUT_SUBJECTS = ("S4.mat", "S5.mat")
WINDOW_SAMPLES = 500
DONG_SHAPE = (8, 1250, 40, 4)


class DomainDatasetV3Error(ValueError):
    """Incompatible sources/audits or a window that cannot be safely normalized."""


def sha256(path: Path | str) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DomainDatasetV3Error(f"cannot read audited metadata {path}: {error}") from error
    if type(value) is not dict:
        raise DomainDatasetV3Error(f"metadata must be an object: {path}")
    return value


def _contract(root: Path) -> dict[str, Any]:
    contract = _json(root / PREREGISTRATION)
    data = contract.get("data", {})
    expected = {
        "support_npz": SUPPORT_NPZ, "supported_train_split": "train",
        "supported_validation_split": "validation", "known_benchmark_split": "test",
        "unsupported_dong_dir": DONG_DIR, "dong_metadata": DONG_METADATA,
        "unsupported_train_subjects": list(TRAIN_SUBJECTS),
        "unsupported_validation_subjects": list(VALIDATION_SUBJECTS),
        "new_unsupported_holdout_subjects": list(HOLDOUT_SUBJECTS),
        "dong_interval_samples": [125, 1125], "window_samples": 500,
        "sample_rate_hz": 250, "all_targets_blocks_channels": True,
        "max_window_exclusion": 0, "use_unsupported_as_clean_ground_truth": False,
    }
    if type(data) is not dict or any(data.get(k) != v for k, v in expected.items()):
        raise DomainDatasetV3Error("v3 source/split/window contract differs from preregistration")
    return contract


def _normalize(windows: np.ndarray, source: str) -> np.ndarray:
    if (windows.ndim != 3 or windows.shape[1:] != (1, WINDOW_SAMPLES)
            or windows.dtype.kind not in "iuf" or len(windows) == 0
            or not np.all(np.isfinite(windows))):
        raise DomainDatasetV3Error(f"{source}: expected finite nonempty real [N,1,500]")
    result = np.empty(windows.shape, dtype=np.float32)
    for index, window in enumerate(windows):
        try:
            # v2 computes original mean and AC RMS in float64 before casting
            # the normalized feature to float32. DC/scale are NOT features.
            result[index, 0], _, _ = normalize_window(window[0])
        except LearnedDenoiseV2Error as error:
            raise DomainDatasetV3Error(f"{source}: window {index} rejected; no exclusion allowed: {error}") from error
    return result


def _support(root: Path, split_name: str, contract: dict) -> tuple[np.ndarray, list[dict], dict]:
    path = root / SUPPORT_NPZ
    manifest_path = Path(str(path) + ".manifest.json")
    digest, manifest_digest = sha256(path), sha256(manifest_path)
    protected = contract.get("protected_sha256", {})
    if (protected.get(SUPPORT_NPZ) != digest
            or protected.get(SUPPORT_NPZ + ".manifest.json") != manifest_digest):
        raise DomainDatasetV3Error("supported NPZ/manifest SHA256 differs from frozen preregistration")
    manifest = _json(manifest_path)
    preprocessing = manifest.get("preprocessing", {})
    if (manifest.get("output", {}).get("sha256") != digest
            or preprocessing.get("target_rate_hz") != 250
            or preprocessing.get("target_samples") != 500):
        raise DomainDatasetV3Error("supported source identity/sample-rate/window audit mismatch")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not {"noisy", "split", "valid_mask"} <= set(archive.files):
                raise DomainDatasetV3Error("supported NPZ missing noisy/split/valid_mask")
            partitions = archive["split"]
            if (partitions.ndim != 1 or partitions.dtype.kind not in "US"
                    or set(partitions.tolist()) != {"train", "validation", "test"}):
                raise DomainDatasetV3Error("supported split labels must be train/validation/test")
            indices = np.flatnonzero(partitions == split_name)
            member = archive["noisy"]
            if member.shape != (len(partitions), 1, 500) or member.dtype != np.dtype("float32"):
                raise DomainDatasetV3Error("supported noisy must be float32 [N,1,500]")
            # Compressed NPZ members decompress as a whole. Only the requested
            # train/validation rows are selected, inspected or normalized. The
            # decompressed member can contain test bytes physically; test rows
            # are never selected/inspected/used for training or calibration.
            # Clean/artifact targets are not loaded.
            windows = np.array(member[indices], copy=True)
            mask_member = archive["valid_mask"]
            if mask_member.shape != member.shape or mask_member.dtype != np.dtype("bool"):
                raise DomainDatasetV3Error("supported valid_mask must be bool with noisy shape")
            if not np.all(mask_member[indices]):
                raise DomainDatasetV3Error("supported windows contain missing samples; no exclusions allowed")
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, DomainDatasetV3Error):
            raise
        raise DomainDatasetV3Error(f"invalid supported NPZ: {error}") from error
    features = _normalize(windows, str(path))
    if sha256(path) != digest or sha256(manifest_path) != manifest_digest:
        raise DomainDatasetV3Error("supported source changed while loading")
    sources = manifest.get("sources")
    if type(sources) is not dict or set(sources) != {"eeg", "eog", "emg"}:
        raise DomainDatasetV3Error("supported source provenance missing")
    for entry in sources.values():
        if (type(entry) is not dict or type(entry.get("path")) is not str
                or not _sha_string(entry.get("sha256"))):
            raise DomainDatasetV3Error("supported source provenance requires path and SHA256")
    provenance = [{"domain": "supported_semi_synthetic", "split": split_name,
                   "source_path": str(path.resolve()), "source_sha256": digest,
                   "manifest_sha256": manifest_digest, "source_window_index": int(index),
                   "channel_index": 0, "window_samples": 500, "sample_rate_hz": 250}
                  for index in indices]
    source = {"path": str(path.resolve()), "sha256": digest,
              "manifest_path": str(manifest_path.resolve()), "manifest_sha256": manifest_digest,
              "original_sources": sources}
    return features, provenance, source


def _sha_string(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _dong(root: Path, split: str, subjects: tuple[str, ...]) -> tuple[np.ndarray, list[dict], dict]:
    metadata_path = root / DONG_METADATA
    metadata_digest = sha256(metadata_path)
    meta = _json(metadata_path)
    interval = meta.get("analysis_window_samples", {})
    if (meta.get("dataset") != "Dong2023" or meta.get("sample_rate_hz") != 250
            or meta.get("expected_shape") != list(DONG_SHAPE)
            or meta.get("axis_order") != ["channel", "time", "target", "block"]
            or meta.get("array_key") != "eegdata"
            or interval != {"start": 125, "stop_exclusive": 1125, "count": 1000, "indexing": "zero_based_half_open"}
            or meta.get("stimulus_onset_status") != "confirmed_primary_author_paper"
            or meta.get("channel_count") != 8 or meta.get("target_count") != 40
            or meta.get("trial_structure", {}).get("blocks") != 4):
        raise DomainDatasetV3Error("Dong metadata must audit 250Hz, axes/shape, and exact [125:1125] stimulus interval")
    features, provenance, identities = [], [], []
    for subject in subjects:
        path = root / DONG_DIR / subject
        digest = sha256(path)
        try:
            data = loadmat(path, variable_names=["eegdata"])
        except (OSError, ValueError, TypeError) as error:
            raise DomainDatasetV3Error(f"cannot load Dong source {path}: {error}") from error
        raw = data.get("eegdata")
        if (not isinstance(raw, np.ndarray) or raw.shape != DONG_SHAPE
                or raw.dtype.kind not in "iuf" or not np.all(np.isfinite(raw))):
            raise DomainDatasetV3Error(f"{subject}: expected finite real Dong [8,1250,40,4]")
        # Stable ordering: target, block, channel, then each of two consecutive
        # windows. 40*4*8*2 = 2560 windows per subject, never performance-selected.
        windows = raw[:, 125:1125, :, :].transpose(2, 3, 0, 1).reshape(-1, 1, 500)
        features.append(_normalize(windows, str(path)))
        for target in range(40):
            for block in range(4):
                for channel in range(8):
                    for window_index in range(2):
                        provenance.append({"domain": "unsupported_real_EEG_not_clean",
                            "split": split, "source_path": str(path.resolve()), "source_sha256": digest,
                            "metadata_sha256": metadata_digest, "subject": subject,
                            "target_axis_index": target, "block_index": block,
                            "channel_index": channel, "window_index": window_index,
                            "start_sample": 125 + 500 * window_index,
                            "stop_sample_exclusive": 625 + 500 * window_index,
                            "window_samples": 500, "sample_rate_hz": 250})
        if sha256(path) != digest:
            raise DomainDatasetV3Error(f"{subject}: source changed while loading")
        identities.append({"path": str(path.resolve()), "sha256": digest, "subject": subject,
                           "window_count": int(len(windows))})
    if sha256(metadata_path) != metadata_digest:
        raise DomainDatasetV3Error("Dong audit metadata changed while loading")
    return np.concatenate(features), provenance, {
        "metadata_path": str(metadata_path.resolve()), "metadata_sha256": metadata_digest,
        "subjects": identities, "target_mapping_required_for_domain_labels": False,
        "unsupported_is_clean_ground_truth": False}


def load_domain_split(root: Path | str, split: str) -> dict[str, Any]:
    """Train/validation only. No test selector, holdout reads or exclusions."""
    if split not in ("train", "validation"):
        raise DomainDatasetV3Error("training loader allows only train or validation; test/holdout forbidden")
    root = Path(root).resolve()
    contract = _contract(root)
    supported, support_provenance, support_sources = _support(root, split, contract)
    unsupported, dong_provenance, dong_sources = _dong(
        root, split, TRAIN_SUBJECTS if split == "train" else VALIDATION_SUBJECTS)
    return {"normalized": np.concatenate((supported, unsupported)),
            "supported": np.concatenate((np.ones(len(supported), dtype=bool), np.zeros(len(unsupported), dtype=bool))),
            "provenance": support_provenance + dong_provenance,
            "sources": {"supported": support_sources, "unsupported": dong_sources},
            "dataset_sha256": support_sources["sha256"],
            "preregistration_sha256": sha256(root / PREREGISTRATION),
            "split": split, "test_values_accessed": False,
            "new_unsupported_holdout_accessed": False,
            "npz_io_semantics": "Compressed noisy/valid_mask members may physically decompress test bytes; test rows are never selected, inspected, normalized, model inputs, training/metric/calibration data."}


def load_unsupported_holdout(root: Path | str) -> dict[str, Any]:
    """Evaluation ONLY: independent S4/S5; never called by gate training."""
    root = Path(root).resolve()
    _contract(root)
    normalized, provenance, sources = _dong(root, "unsupported_holdout", HOLDOUT_SUBJECTS)
    return {"normalized": normalized, "supported": np.zeros(len(normalized), dtype=bool),
            "provenance": provenance, "sources": {"unsupported": sources},
            "preregistration_sha256": sha256(root / PREREGISTRATION),
            "split": "unsupported_holdout"}


__all__ = ["DomainDatasetV3Error", "load_domain_split", "load_unsupported_holdout"]
