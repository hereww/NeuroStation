"""Train offline shared-single-channel EEG v2 without evaluating test windows."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Main also sets these to --threads; default environment before optional torch import.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
from eeg_tools.learned_denoise_v2 import (
    MODEL_VERSION, LearnedDenoiseV2Error, build_model, load_config,
    normalize_window, primitive_metadata, torch, validate_mask,
)

SPLITS = ("train", "validation", "test")
EPOCH_KEYS = {"eeg": "clean_epoch_id", "eog": "eog_epoch_id", "emg": "emg_epoch_id"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "learned_denoiser_v2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "denoiser_v2.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-train-windows", type=int)
    return parser


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset_manifest(manifest: dict, dataset_sha256: str, split: np.ndarray,
                    epoch_ids: dict[str, np.ndarray]) -> None:
    """Guard source-ID/fingerprint provenance, NOT test signal content or metrics."""
    if (not primitive_metadata(manifest) or manifest.get("schema_version") != 2
            or manifest.get("status") != "formal" or manifest.get("dataset_version") != "learned_denoiser_v2"):
        raise LearnedDenoiseV2Error("invalid v2 dataset manifest version/status/plain metadata")
    if manifest.get("output", {}).get("sha256") != dataset_sha256:
        raise LearnedDenoiseV2Error("dataset SHA256 does not match manifest")
    if manifest.get("requested_samples") != len(split):
        raise LearnedDenoiseV2Error("manifest sample count does not match split")
    semantics = manifest.get("sampling_semantics", {})
    if semantics.get("subject_disjoint_claim") is not False or semantics.get("subject_identifiers_available") is not False:
        raise LearnedDenoiseV2Error("EEGdenoiseNet v2 cannot claim subject-disjoint validation")
    audit = manifest.get("epoch_split_manifest", {})
    if audit.get("split_before_resampling_or_mixing") is not True:
        raise LearnedDenoiseV2Error("raw epochs must be split before resampling/mixing")
    if audit.get("fingerprint_namespace") != "global_across_EEG_EOG_EMG":
        raise LearnedDenoiseV2Error("manifest requires global fingerprint namespace")
    groups = audit.get("fingerprint_groups")
    stats = audit.get("source_statistics")
    if not isinstance(groups, list) or not groups or not isinstance(stats, dict):
        raise LearnedDenoiseV2Error("manifest requires source-ID and fingerprint groups")
    mapping: dict[str, dict[int, str]] = {name: {} for name in EPOCH_KEYS}
    fingerprints = set()
    for group in groups:
        fingerprint = group.get("fingerprint")
        partition = group.get("split")
        sources = group.get("epoch_ids_by_source")
        if (not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(x not in "0123456789abcdef" for x in fingerprint)
                or fingerprint in fingerprints or partition not in SPLITS
                or not isinstance(sources, dict) or not sources or set(sources) - set(EPOCH_KEYS)):
            raise LearnedDenoiseV2Error("invalid/duplicate/cross-split fingerprint provenance")
        fingerprints.add(fingerprint)
        for name, ids in sources.items():
            if not isinstance(ids, list) or not ids:
                raise LearnedDenoiseV2Error("fingerprint group needs original epoch IDs")
            for epoch_id in ids:
                if type(epoch_id) is not int or epoch_id < 0 or epoch_id in mapping[name]:
                    raise LearnedDenoiseV2Error("invalid/duplicate original epoch ID")
                mapping[name][epoch_id] = partition
    for name, values in epoch_ids.items():
        if values.shape != split.shape or values.dtype.kind not in "iu" or np.any(values < -1):
            raise LearnedDenoiseV2Error(f"invalid original {name} epoch IDs")
        raw_epochs = stats.get(name, {}).get("raw_epochs")
        if type(raw_epochs) is not int or raw_epochs < 3 or set(mapping[name]) != set(range(raw_epochs)):
            raise LearnedDenoiseV2Error(f"manifest must completely enumerate original {name} epochs")
        pools = stats[name].get("raw_epoch_ids_by_split")
        if not isinstance(pools, dict) or set(pools) != set(SPLITS):
            raise LearnedDenoiseV2Error("manifest original epoch split pools missing")
        for partition in SPLITS:
            expected = sorted(epoch_id for epoch_id, group in mapping[name].items() if group == partition)
            if not expected or pools[partition] != expected:
                raise LearnedDenoiseV2Error("source pools disagree with global fingerprint groups")
            ids = set(values[split == partition].tolist()) - {-1}
            if not ids.issubset(expected):
                raise LearnedDenoiseV2Error("dataset uses original epoch ID outside its assigned split")
    if np.any(epoch_ids["eeg"] < 0):
        raise LearnedDenoiseV2Error("all windows require an original clean_epoch_id")


# Backward-compatible guard name for the parallel evaluation agent.
_manifest_guard = validate_dataset_manifest


def _load_training_data(path: Path, config: dict, seed: int, max_train_windows: int | None) -> dict:
    manifest_path = Path(str(path) + ".manifest.json")
    dataset_sha256, manifest_sha256 = sha256(path), sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as archive:
        required = {"clean", "noisy", "valid_mask", "artifact_mask", "split", "artifact_type", *EPOCH_KEYS.values()}
        if not required.issubset(archive.files):
            raise LearnedDenoiseV2Error("v2 NPZ contract missing keys")
        split = np.asarray(archive["split"])
        if split.ndim != 1 or split.dtype.kind not in "US" or set(split.tolist()) != set(SPLITS):
            raise LearnedDenoiseV2Error("split must contain nonempty train/validation/test groups")
        epoch_ids = {name: np.asarray(archive[key]) for name, key in EPOCH_KEYS.items()}
        validate_dataset_manifest(manifest, dataset_sha256, split, epoch_ids)
        train_indices = np.flatnonzero(split == "train")
        validation_indices = np.flatnonzero(split == "validation")
        selected_train = train_indices.copy()
        if max_train_windows is not None and max_train_windows < len(train_indices):
            selected_train = np.sort(np.random.default_rng(seed).choice(train_indices, max_train_windows, replace=False))
        # Read only train/validation ROWS for learning/metrics. NPZ compressed
        # members may be decompressed as a whole; no test signal row is inspected,
        # normalized, fed to a model, used for selection, or assigned a metric.
        partitions = {}
        for name, indices in (("train", selected_train), ("validation", validation_indices)):
            values = {}
            for key in ("clean", "noisy", "valid_mask", "artifact_mask"):
                member = archive[key]
                if member.shape != (len(split), 1, config["window_samples"]):
                    raise LearnedDenoiseV2Error(f"{key} must have shape [N,1,500]")
                if key in ("clean", "noisy") and member.dtype != np.dtype("float32"):
                    raise LearnedDenoiseV2Error("clean/noisy NPZ arrays must be float32")
                values[key] = np.array(member[indices], copy=True)
            types = archive["artifact_type"]
            if types.shape != split.shape:
                raise LearnedDenoiseV2Error("artifact_type shape must match split")
            values["artifact_type"] = np.asarray(types[indices]).astype(str)
            if set(values["artifact_type"].tolist()) - {"clean", "eog", "emg"}:
                raise LearnedDenoiseV2Error("training/validation artifact_type must be clean/eog/emg")
            valid = validate_mask(values["valid_mask"], values["noisy"].shape)
            artifact = validate_mask(values["artifact_mask"], values["noisy"].shape)
            if not np.all(valid) or not np.all(np.isfinite(values["noisy"])) or not np.all(np.isfinite(values["clean"])):
                raise LearnedDenoiseV2Error("training/validation pairs must be fully valid finite windows; no interpolation")
            clean_windows = values["artifact_type"] == "clean"
            if not np.any(clean_windows):
                raise LearnedDenoiseV2Error(f"{name} requires clean identity windows")
            if not np.any(~clean_windows):
                raise LearnedDenoiseV2Error(f"{name} requires corrupted windows to diagnose all-noop models")
            if np.any(artifact[clean_windows]) or not np.all(artifact[~clean_windows]):
                raise LearnedDenoiseV2Error("artifact masks must follow clean/corrupted whole-window contract")
            if not np.array_equal(values["clean"][clean_windows], values["noisy"][clean_windows]):
                raise LearnedDenoiseV2Error("clean identity windows must have clean == noisy")
            for modality in ("eog", "emg"):
                used = epoch_ids[modality][indices]
                expected = values["artifact_type"] == modality
                if np.any(used[expected] < 0) or np.any(used[~expected] != -1):
                    raise LearnedDenoiseV2Error(f"{modality}_epoch_id must be -1 when unused")
            noisy_norm, clean_norm, centers, scales = [], [], [], []
            for noisy, clean in zip(values["noisy"][:, 0], values["clean"][:, 0]):
                normalized, center, scale = normalize_window(noisy, min_ac_rms=config["inference"]["min_ac_rms"])
                target = ((clean.astype(np.float64) - center) / scale).astype(np.float32)
                if not np.all(np.isfinite(target)):
                    raise LearnedDenoiseV2Error("unsafe normalized clean training/validation target")
                noisy_norm.append(normalized)
                clean_norm.append(target)
                centers.append(center)
                scales.append(scale)
            values.update(noisy_normalized=np.asarray(noisy_norm)[:, None], clean_normalized=np.asarray(clean_norm)[:, None],
                          centers=np.asarray(centers, dtype=np.float64), scales=np.asarray(scales, dtype=np.float64),
                          valid_mask=valid, artifact_mask=artifact)
            partitions[name] = values
    if sha256(path) != dataset_sha256 or sha256(manifest_path) != manifest_sha256:
        raise LearnedDenoiseV2Error("dataset/manifest changed during loading")
    return {"partitions": partitions, "train_indices": train_indices.tolist(),
            "validation_indices": validation_indices.tolist(), "selected_train_indices": selected_train.tolist(),
            "dataset_sha256": dataset_sha256, "dataset_manifest": manifest,
            "dataset_manifest_sha256": manifest_sha256, "dataset_manifest_path": str(manifest_path.resolve())}


def calibrate_validation_threshold(clean_scores: np.ndarray, max_false_fraction: float = 0.01) -> tuple[float, dict]:
    scores = np.asarray(clean_scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise LearnedDenoiseV2Error("calibration requires finite validation clean-window scores in [0,1]")
    if not 0 <= max_false_fraction <= 0.01:
        raise LearnedDenoiseV2Error("clean false replacement allowance cannot exceed 1%")
    allowed = math.floor(len(scores) * max_false_fraction)
    threshold = float(np.sort(scores)[len(scores) - allowed - 1])
    observed = float(np.mean(scores > threshold))
    return threshold, {
        "scope": "validation_only", "comparison": ">", "method": "clean_score_order_statistic",
        "score_reduction": "per_channel_window_mean", "max_clean_false_replacement_fraction": float(max_false_fraction),
        "clean_window_count": len(scores), "allowed_false_replacement_windows": allowed,
        "observed_clean_false_replacement_fraction": observed,
        "empirical_only": True, "production_calibrated": False,
        "warning": "Validation-only empirical clean false replacement control, NOT clinical or production probability calibration.",
    }


def _loss_terms(output: dict, noisy: Any, clean: Any, artifact: Any, clean_windows: Any, config: dict) -> dict:
    functional = torch.nn.functional
    denoised = output["denoised"]
    time_loss = functional.mse_loss(denoised, clean)
    # Non-DC clean-spectrum preservation in noisy normalized coordinates.
    pred_spectrum = torch.log1p(torch.abs(torch.fft.rfft(denoised, dim=-1)[..., 1:]))
    clean_spectrum = torch.log1p(torch.abs(torch.fft.rfft(clean, dim=-1)[..., 1:]))
    spectral_loss = functional.l1_loss(pred_spectrum, clean_spectrum)
    identity_weights = (1.0 - artifact) * (1.0 + clean_windows[:, None, None] * (config["loss"]["clean_window_identity_multiplier"] - 1.0))
    identity_loss = torch.mean((denoised - noisy) ** 2 * identity_weights)
    artifact_loss = functional.binary_cross_entropy_with_logits(output["artifact_logits"], artifact)
    terms = {"time": time_loss, "spectral": spectral_loss, "identity": identity_loss, "artifact": artifact_loss}
    terms["total"] = (config["loss"]["time_weight"] * time_loss + config["loss"]["spectral_weight"] * spectral_loss
                      + config["loss"]["identity_weight"] * identity_loss + config["loss"]["artifact_weight"] * artifact_loss)
    return terms


def _run_epoch(model: Any, loader: Any, optimizer: Any, device: Any, config: dict, train: bool) -> float:
    model.train(train)
    total, count = 0.0, 0
    with nullcontext() if train else torch.no_grad():
        for noisy, clean, artifact, is_clean in loader:
            noisy, clean, artifact, is_clean = (value.to(device) for value in (noisy, clean, artifact, is_clean))
            if train:
                optimizer.zero_grad(set_to_none=True)
            output = model(noisy)
            loss = _loss_terms(output, noisy, clean, artifact, is_clean, config)["total"]
            if not bool(torch.isfinite(loss)):
                raise LearnedDenoiseV2Error("non-finite training/validation loss")
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            total += float(loss.detach().cpu()) * len(noisy)
            count += len(noisy)
    return total / count


def _validation_report(model: Any, values: dict, config: dict, device: Any, batch_size: int) -> tuple[float, dict, dict]:
    model.eval()
    probabilities, residuals = [], []
    with torch.no_grad():
        for start in range(0, len(values["noisy"]), batch_size):
            tensor = torch.from_numpy(values["noisy_normalized"][start:start + batch_size]).to(device)
            prediction = model(tensor)
            probabilities.append(prediction["artifact_probability"].cpu().numpy()[:, 0].astype(np.float64))
            residuals.append(prediction["noise"].cpu().numpy()[:, 0].astype(np.float64))
    probability, residual = np.concatenate(probabilities), np.concatenate(residuals)
    if not np.all(np.isfinite(probability)) or not np.all(np.isfinite(residual)):
        raise LearnedDenoiseV2Error("non-finite validation prediction")
    scores = probability.mean(axis=1)
    is_clean = values["artifact_type"] == "clean"
    threshold, calibration = calibrate_validation_threshold(scores[is_clean])
    noisy, clean = values["noisy"][:, 0].astype(np.float64), values["clean"][:, 0].astype(np.float64)
    candidate = noisy - residual * values["scales"][:, None]
    replace = scores > threshold
    gated = np.where(replace[:, None], candidate, noisy)
    if not np.all(np.isfinite(candidate)):
        raise LearnedDenoiseV2Error("non-finite restored validation prediction")

    def metrics(track):
        corrupted = ~is_clean
        base_mse = float(np.mean((noisy[corrupted] - clean[corrupted]) ** 2))
        mse = float(np.mean((track[corrupted] - clean[corrupted]) ** 2))
        return {
            "all_rmse": float(np.sqrt(np.mean((track - clean) ** 2))),
            "corrupted_rmse": float(np.sqrt(mse)),
            "corrupted_rmse_improvement_fraction": float((np.sqrt(base_mse) - np.sqrt(mse)) / max(np.sqrt(base_mse), 1e-12)),
            "corrupted_snr_gain_db": float(10 * np.log10(max(base_mse, 1e-24) / max(mse, 1e-24))),
            "clean_change_rmse": float(np.sqrt(np.mean((track[is_clean] - noisy[is_clean]) ** 2))),
        }
    report = {
        "scope": "validation_only", "clean_windows": int(np.sum(is_clean)), "corrupted_windows": int(np.sum(~is_clean)),
        "clean_false_replacement_fraction": float(np.mean(replace[is_clean])),
        "corrupted_replacement_fraction": float(np.mean(replace[~is_clean])),
        "candidate": metrics(candidate), "gated": metrics(gated),
        "gated_is_all_noop": bool(not np.any(replace)),
        "confidence_interpretation": "one_minus_probability_NOT_calibrated_success_probability",
    }
    return threshold, calibration, report


def _atomic_checkpoint(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".pending", delete=False) as stream:
        pending = Path(stream.name)
    try:
        torch.save(checkpoint, pending)
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".pending", delete=False) as stream:
        pending = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if torch is None:
            raise LearnedDenoiseV2Error("optional PyTorch required for v2 training")
        if args.epochs < 1 or args.batch_size < 1 or args.threads < 1 or (args.max_train_windows is not None and args.max_train_windows < 1):
            raise LearnedDenoiseV2Error("epochs/batch-size/threads/max-train-windows must be positive")
        if "v2" not in args.output.stem or args.output.resolve() == (ROOT / "models" / "denoiser_v1.pt").resolve():
            raise LearnedDenoiseV2Error("output name must contain v2; refusing to overwrite v1")
        if args.output.resolve() in {args.dataset.resolve(), Path(str(args.dataset) + ".manifest.json").resolve(), args.config.resolve()}:
            raise LearnedDenoiseV2Error("checkpoint output must not overwrite source data/config/manifest")
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(args.threads)
        except RuntimeError:
            pass  # already initialized by an embedding caller
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        config = load_config(args.config)
        data = _load_training_data(args.dataset, config, args.seed, args.max_train_windows)
        device = torch.device(args.device)
        model = build_model(config).to(device)
        from torch.utils.data import DataLoader, TensorDataset
        loaders = {}
        for name, values in data["partitions"].items():
            dataset = TensorDataset(torch.from_numpy(values["noisy_normalized"]), torch.from_numpy(values["clean_normalized"]),
                                    torch.from_numpy(values["artifact_mask"].astype(np.float32)),
                                    torch.from_numpy((values["artifact_type"] == "clean").astype(np.float32)))
            loaders[name] = DataLoader(dataset, batch_size=args.batch_size, shuffle=name == "train")
        best_loss, best_epoch, history = float("inf"), 0, []
        report_path = args.output.with_suffix(".training.json")
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        for epoch in range(1, args.epochs + 1):
            train_loss = _run_epoch(model, loaders["train"], optimizer, device, config, True)
            validation_loss = _run_epoch(model, loaders["validation"], optimizer, device, config, False)
            threshold, calibration, val_report = _validation_report(model, data["partitions"]["validation"], config, device, args.batch_size)
            item = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
                    "calibrated_threshold": threshold, "validation": val_report}
            history.append(item)
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                checkpoint = {
                    "checkpoint_schema_version": 1, "model_version": MODEL_VERSION,
                    "config": config, "architecture": config["model"], "model_state": model.state_dict(),
                    "dataset_sha256": data["dataset_sha256"], "dataset_manifest": data["dataset_manifest"],
                    "dataset_manifest_sha256": data["dataset_manifest_sha256"], "dataset_manifest_path": data["dataset_manifest_path"],
                    "train_indices": data["train_indices"], "validation_indices": data["validation_indices"],
                    "selected_train_indices": data["selected_train_indices"],
                    "calibrated_threshold": threshold, "validation_threshold": threshold, "threshold_calibration": calibration,
                    "best_epoch": epoch, "best_validation_loss": validation_loss, "history": history.copy(),
                    "training_seed": args.seed, "batch_size": args.batch_size, "threads": args.threads,
                    "optimizer": {"name": "AdamW", "learning_rate": 2e-4, "weight_decay": 1e-4},
                    "evaluation_scope": "train_validation_only_no_test_metrics", "validation_report": val_report,
                    "target_provenance": "epoch_disjoint_single_channel_semi_synthetic_not_subject_disjoint",
                    "production_validated": False, "automatic_processing_enabled": False,
                    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                if not primitive_metadata({key: value for key, value in checkpoint.items() if key != "model_state"}):
                    raise LearnedDenoiseV2Error("non-primitive checkpoint metadata")
                _atomic_checkpoint(args.output, checkpoint)
            print(json.dumps(item, ensure_ascii=True, allow_nan=False), flush=True)
        report = {
            "status": "completed", "model_version": MODEL_VERSION, "checkpoint": str(args.output.resolve()),
            "best_epoch": best_epoch, "best_validation_loss": best_loss, "epochs": args.epochs,
            "train_windows": len(data["train_indices"]), "selected_train_windows": len(data["selected_train_indices"]),
            "validation_windows": len(data["validation_indices"]), "dataset_sha256": data["dataset_sha256"],
            "dataset_manifest_sha256": data["dataset_manifest_sha256"], "history": history,
            "evaluation_scope": "train_validation_only_no_test_metrics", "automatic_processing_enabled": False,
            "production_validated": False, "warning": "Training/validation complete is not heldout efficacy or release acceptance.",
        }
        _atomic_json(report_path, report)
        print(json.dumps({key: value for key, value in report.items() if key != "history"}, ensure_ascii=True, indent=2))
    except (LearnedDenoiseV2Error, OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
