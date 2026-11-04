"""Train the optional PyTorch residual TCN denoiser."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import hashlib
import json
import math
import random
from typing import Any

import numpy as np

from eeg_tools.learned_denoise import (
    LearnedDenoiseError,
    ResidualTCNDenoiser,
    load_config,
    robust_normalize,
    torch_available,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="paired NPZ from build_denoise_dataset.py")
    parser.add_argument("--config", type=Path, default=Path("configs/learned_denoiser_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("models/denoiser_v1.pt"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--device", default="cpu")
    return parser


def _require_torch() -> Any:
    if not torch_available():
        raise LearnedDenoiseError("PyTorch is not installed. Install the optional ML dependencies before training.")
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    return torch, nn, DataLoader, TensorDataset


def _dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_session_ids(session_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(session_ids)
    if values.ndim != 1 or not len(values):
        raise LearnedDenoiseError("session_id must be a nonempty one-dimensional array of real session groups")
    labels = np.asarray([str(value) for value in values])
    if any(not label.strip() or label.strip().lower() in {"nan", "none", "null"} for label in labels):
        raise LearnedDenoiseError("session_id contains a missing session group")
    return labels


def _seeded_session_groups(labels: np.ndarray, seed: int) -> tuple[list[str], list[str], list[str]]:
    unique = sorted(set(labels.tolist()))
    if len(unique) < 3:
        raise LearnedDenoiseError("session split requires at least 3 real session groups; window-level fallback is forbidden")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    # This seeded selection is frozen for test, independent of window balancing.
    train_count = min(len(unique) - 2, max(1, math.ceil(len(unique) * 0.7)))
    valid_count = min(len(unique) - train_count - 1, max(1, math.ceil(len(unique) * 0.15)))
    return unique[:train_count], unique[train_count:train_count + valid_count], unique[train_count + valid_count:]


def _split_by_session(session_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = _validated_session_ids(session_ids)
    train_groups, valid_groups, test_groups = _seeded_session_groups(labels, seed)
    names, sizes = np.unique(labels, return_counts=True)
    counts = {str(name): int(size) for name, size in zip(names, sizes)}
    largest = {name for name, count in counts.items() if count == max(counts.values())}
    if not largest.intersection(train_groups + valid_groups):
        # Never silently move a frozen held-out group into training.
        raise LearnedDenoiseError("largest session group is in the fixed test split; cannot ensure largest-in-train without changing test selection")

    def score(train: list[str], validation: list[str]) -> float:
        total = len(labels)
        fractions = (sum(counts[name] for name in train) / total,
                     sum(counts[name] for name in validation) / total,
                     sum(counts[name] for name in test_groups) / total)
        return sum((actual - target) ** 2 for actual, target in zip(fractions, (0.7, 0.15, 0.15)))

    # Mandatory largest-in-train, followed by strictly improving train/validation
    # swaps. Group counts and the exact seeded test selection remain unchanged.
    # Seeded iteration order breaks equal-size/equal-score ties reproducibly.
    while True:
        must_move_largest = not largest.intersection(train_groups)
        best_score = float("inf") if must_move_largest else score(train_groups, valid_groups)
        best_swap = None
        for train_index, train_group in enumerate(train_groups):
            for valid_index, valid_group in enumerate(valid_groups):
                candidate_train = train_groups.copy()
                candidate_validation = valid_groups.copy()
                candidate_train[train_index] = valid_group
                candidate_validation[valid_index] = train_group
                if not largest.intersection(candidate_train):
                    continue
                candidate_score = score(candidate_train, candidate_validation)
                if candidate_score < best_score - 1e-12:
                    best_score = candidate_score
                    best_swap = candidate_train, candidate_validation
        if best_swap is None:
            break
        train_groups, valid_groups = best_swap
    partitions = train_groups, valid_groups, test_groups
    return tuple(np.flatnonzero(np.isin(labels, groups)) for groups in partitions)


def _split_metadata(session_ids: np.ndarray, indices: tuple[np.ndarray, np.ndarray, np.ndarray], seed: int) -> dict[str, Any]:
    labels = _validated_session_ids(session_ids)
    partitions = {}
    for name, selected in zip(("train", "validation", "test"), indices):
        groups = sorted(set(labels[selected].tolist()))
        partitions[name] = {
            "sessions": groups,
            "session_count": len(groups),
            "window_indices": selected.tolist(),
            "window_count": len(selected),
            "window_fraction": len(selected) / len(labels),
        }
    return {
        "schema_version": 1,
        "strategy": "session_group_window_aware_fixed_test",
        "allocation_method": "largest_in_train_then_greedy_train_validation_swaps_minimizing_squared_window_fraction_error",
        "test_selection": "unchanged_seeded_session_groups",
        "initial_session_groups": {name: groups for name, groups in zip(("train", "validation", "test"), _seeded_session_groups(labels, seed))},
        "actual_window_fractions": {name: partition["window_fraction"] for name, partition in partitions.items()},
        "group_window_counts": {str(name): int(count) for name, count in zip(*np.unique(labels, return_counts=True))},
        "seed": seed,
        "target_fractions": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "rounding": "ceil_with_one_group_reserved_for_validation_and_test",
        "total_windows": len(labels),
        "all_sessions": sorted(set(labels.tolist())),
        "session_id_by_window": labels.tolist(),
        "partitions": partitions,
    }


def _split_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep per-window provenance on disk, not in multi-thousand-line CLI logs."""
    summary = {key: value for key, value in metadata.items() if key not in {"session_id_by_window", "partitions"}}
    summary["partitions"] = {
        name: {key: value for key, value in partition.items() if key != "window_indices"}
        for name, partition in metadata["partitions"].items()
    }
    return summary


def _save_split_metadata(checkpoint_path: Path, metadata: dict[str, Any]) -> Path:
    path = checkpoint_path.with_suffix(".split.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _prepare_tensor(torch: Any, values: np.ndarray, masks: np.ndarray) -> Any:
    normalized = []
    for value, mask in zip(values, masks):
        item, _, _ = robust_normalize(value, mask.astype(bool))
        normalized.append(item)
    return torch.from_numpy(np.stack(normalized).astype(np.float32))


def _prepare_pair_tensors(torch: Any, noisy: np.ndarray, clean: np.ndarray) -> tuple[Any, Any]:
    noisy_norm = []
    clean_norm = []
    for noisy_item, clean_item in zip(noisy, clean):
        valid = np.isfinite(noisy_item)
        normalized, center, scale = robust_normalize(noisy_item, valid)
        clean_scaled = (np.where(np.isfinite(clean_item), clean_item, center[:, None]) - center[:, None]) / scale[:, None]
        noisy_norm.append(normalized)
        clean_norm.append(clean_scaled.astype(np.float32))
    return (
        torch.from_numpy(np.stack(noisy_norm).astype(np.float32)),
        torch.from_numpy(np.stack(clean_norm).astype(np.float32)),
    )


def _spectral_loss(torch: Any, prediction: Any, target: Any) -> Any:
    prediction_fft = torch.fft.rfft(prediction, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)
    pred_power = torch.log1p(torch.abs(prediction_fft))
    target_power = torch.log1p(torch.abs(target_fft))
    return torch.mean(torch.abs(pred_power - target_power))


def _run_epoch(model: Any, loader: Any, optimizer: Any, device: Any, config: dict[str, Any], train: bool) -> float:
    import torch
    from torch.nn import functional as F

    model.train(train)
    totals: list[float] = []
    for noisy, clean, artifact_mask in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        artifact_mask = artifact_mask.to(device)
        valid_mask = torch.ones_like(noisy)
        if train:
            optimizer.zero_grad(set_to_none=True)
        output = model(noisy, valid_mask)
        time_loss = F.smooth_l1_loss(output["denoised"], clean)
        spectral_loss = _spectral_loss(torch, output["denoised"], clean)
        mask_loss = F.binary_cross_entropy(output["artifact_probability"], artifact_mask)
        clean_weight = 1.0 - artifact_mask
        identity_loss = torch.sum(torch.abs((output["denoised"] - noisy) * clean_weight)) / torch.clamp(torch.sum(clean_weight), min=1.0)
        confidence_target = torch.exp(-0.5 * torch.mean(torch.abs(output["denoised"].detach() - clean), dim=1, keepdim=True)).clamp(0.0, 1.0)
        confidence_loss = F.smooth_l1_loss(output["confidence"], confidence_target)
        losses = config["loss"]
        total = (
            float(losses["time_weight"]) * time_loss
            + float(losses["spectral_weight"]) * spectral_loss
            + float(losses["mask_weight"]) * mask_loss
            + float(losses["identity_weight"]) * identity_loss
            + float(losses.get("confidence_weight", 0.1)) * confidence_loss
        )
        if train:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        totals.append(float(total.detach().cpu()))
    return float(np.mean(totals)) if totals else float("nan")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        torch, _, DataLoader, TensorDataset = _require_torch()
        config = load_config(args.config)
        dataset_sha256 = _dataset_sha256(args.dataset)
        with np.load(args.dataset, allow_pickle=False) as arrays:
            noisy = np.asarray(arrays["noisy"], dtype=np.float32)
            clean = np.asarray(arrays["clean"], dtype=np.float32)
            artifact_mask = np.asarray(arrays["artifact_mask"], dtype=np.float32)
            session_ids = _validated_session_ids(arrays["session_id"])
            if "sample_rate_hz" in arrays:
                rates = np.asarray(arrays["sample_rate_hz"], dtype=np.float64)
                if not np.all(np.isfinite(rates)) or not np.all(rates == float(config["sample_rate_hz"])):
                    raise LearnedDenoiseError("dataset sample_rate_hz does not match config")
        if _dataset_sha256(args.dataset) != dataset_sha256:
            raise LearnedDenoiseError("dataset changed while loading")
        if noisy.ndim != 3 or clean.shape != noisy.shape:
            raise LearnedDenoiseError("dataset noisy and clean arrays must have shape [windows, channels, samples]")
        if artifact_mask.shape != noisy.shape or len(session_ids) != len(noisy):
            raise LearnedDenoiseError("artifact_mask and session_id must match dataset windows")
        if noisy.shape[1:] != (int(config["channel_count"]), int(config["window_samples"])):
            raise LearnedDenoiseError("dataset channels/window_samples do not match config")
        if not np.all(np.isfinite(noisy)) or not np.all(np.isfinite(clean)) or not np.all(np.isfinite(artifact_mask)):
            raise LearnedDenoiseError("training requires finite paired data and artifact masks")
        if np.any((artifact_mask < 0) | (artifact_mask > 1)):
            raise LearnedDenoiseError("artifact masks must be in [0, 1]")
        if args.epochs < 1 or args.batch_size < 1:
            raise LearnedDenoiseError("epochs and batch-size must be positive")
        train_idx, valid_idx, test_idx = _split_by_session(session_ids, args.seed)
        split_metadata = _split_metadata(session_ids, (train_idx, valid_idx, test_idx), args.seed)
        split_metadata["dataset_sha256"] = dataset_sha256
        split_metadata["dataset_shape"] = list(noisy.shape)
        split_path = _save_split_metadata(args.output, split_metadata)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        train_noisy, train_clean = _prepare_pair_tensors(torch, noisy[train_idx], clean[train_idx])
        train_mask = torch.from_numpy(artifact_mask[train_idx])
        valid_noisy, valid_clean = _prepare_pair_tensors(torch, noisy[valid_idx], clean[valid_idx])
        valid_mask = torch.from_numpy(artifact_mask[valid_idx])
        train_loader = DataLoader(TensorDataset(train_noisy, train_clean, train_mask), batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(valid_noisy, valid_clean, valid_mask), batch_size=args.batch_size, shuffle=False)
        device = torch.device(args.device)
        model = ResidualTCNDenoiser(channel_count=int(config["channel_count"])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        best_loss = float("inf")
        history: list[dict[str, float]] = []
        for epoch in range(1, args.epochs + 1):
            train_loss = _run_epoch(model, train_loader, optimizer, device, config, True)
            valid_loss = _run_epoch(model, valid_loader, optimizer, device, config, False)
            history.append({"epoch": float(epoch), "train_loss": train_loss, "valid_loss": valid_loss})
            if valid_loss < best_loss:
                best_loss = valid_loss
                args.output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "checkpoint_schema_version": 1,
                        "dataset_sha256": dataset_sha256,
                        "split_metadata": split_metadata,
                        "target_provenance": "semi_synthetic_clean_ish_not_real_noise_clean_ground_truth",
                        "model_state": model.state_dict(),
                        "hidden_channels": 48,
                        "model_version": config["model_version"],
                        "config": config,
                        "train_sessions": sorted({str(item) for item in session_ids[train_idx]}),
                        "validation_sessions": sorted({str(item) for item in session_ids[valid_idx]}),
                        "test_sessions": sorted({str(item) for item in session_ids[test_idx]}),
                        "history": history,
                    },
                    args.output,
                )
            print(json.dumps({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss}, ensure_ascii=False))
        report = {
            "status": "completed",
            "model": str(args.output.resolve()),
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": dataset_sha256,
            "split_metadata": _split_summary(split_metadata),
            "split_metadata_path": str(split_path.resolve()),
            "best_validation_loss": best_loss,
            "train_windows": int(len(train_idx)),
            "validation_windows": int(len(valid_idx)),
            "test_windows": int(len(test_idx)),
            "test_sessions": sorted({str(item) for item in session_ids[test_idx]}),
            "history": history,
            "warning": "This initial trainer uses clean-ish targets and must be followed by real-session and downstream SSVEP validation.",
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError, LearnedDenoiseError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
