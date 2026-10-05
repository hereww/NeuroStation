"""Train the optional PyTorch residual TCN denoiser."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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


def _split_by_session(session_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = sorted({str(value) for value in session_ids})
    if len(unique) < 2:
        indices = np.arange(len(session_ids))
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        first = max(1, int(len(indices) * 0.8))
        second = max(first + 1, int(len(indices) * 0.9))
        return indices[:first], indices[first:second], indices[second:]
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    train_count = max(1, int(math.ceil(len(unique) * 0.7)))
    valid_count = max(1, int(math.ceil(len(unique) * 0.15))) if len(unique) > 2 else 0
    train_sessions = set(unique[:train_count])
    valid_sessions = set(unique[train_count : train_count + valid_count])
    train = np.asarray([i for i, value in enumerate(session_ids) if str(value) in train_sessions], dtype=np.int64)
    valid = np.asarray([i for i, value in enumerate(session_ids) if str(value) in valid_sessions], dtype=np.int64)
    test = np.asarray([i for i, value in enumerate(session_ids) if str(value) not in train_sessions and str(value) not in valid_sessions], dtype=np.int64)
    if not len(valid):
        valid = train[-max(1, len(train) // 10) :]
    if not len(test):
        test = valid.copy()
    return train, valid, test


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
        arrays = np.load(args.dataset, allow_pickle=False)
        noisy = np.asarray(arrays["noisy"], dtype=np.float32)
        clean = np.asarray(arrays["clean"], dtype=np.float32)
        artifact_mask = np.asarray(arrays["artifact_mask"], dtype=np.float32)
        session_ids = np.asarray(arrays["session_id"])
        if noisy.ndim != 3 or clean.shape != noisy.shape:
            raise LearnedDenoiseError("dataset noisy and clean arrays must have shape [windows, channels, samples]")
        train_idx, valid_idx, test_idx = _split_by_session(session_ids, args.seed)
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
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        for epoch in range(1, args.epochs + 1):
            train_loss = _run_epoch(model, train_loader, optimizer, device, config, True)
            valid_loss = _run_epoch(model, valid_loader, optimizer, device, config, False)
            history.append({"epoch": float(epoch), "train_loss": train_loss, "valid_loss": valid_loss})
            if valid_loss < best_loss:
                best_loss = valid_loss
                args.output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
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
