"""Evaluate denoiser reconstruction, spectral and artifact-mask metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from eeg_tools.learned_denoise import LearnedDenoiseError, load_config, load_model, apply_model_to_array
from scripts.train_denoiser import _dataset_sha256, _validated_session_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="paired NPZ dataset")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/learned_denoiser_v1.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--diagnostic-all", action="store_true", help="explicitly evaluate all windows for diagnostics only; never a held-out evaluation")
    return parser


def _select_evaluation_indices(
    checkpoint: dict,
    session_ids: np.ndarray,
    *,
    diagnostic_all: bool = False,
    dataset_sha256: str | None = None,
) -> np.ndarray:
    """Validate provenance before selecting whole held-out session groups.

    Old checkpoints without a hash remain readable, but missing split lists or
    historical overlapping validation/test groups cannot claim held-out metrics.
    """
    labels = _validated_session_ids(session_ids)
    groups = {}
    for name in ("train", "validation", "test"):
        values = checkpoint.get(f"{name}_sessions")
        if values is None and diagnostic_all:
            values = []
        if not isinstance(values, (list, tuple)) or (not values and not diagnostic_all):
            raise LearnedDenoiseError(f"checkpoint requires nonempty {name}_sessions for held-out evaluation")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise LearnedDenoiseError(f"invalid checkpoint {name}_sessions")
        groups[name] = set(values)
        if len(groups[name]) != len(values):
            raise LearnedDenoiseError(f"duplicate checkpoint {name}_sessions")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = groups[left] & groups[right]
        if overlap:
            raise LearnedDenoiseError(f"checkpoint session overlap between {left} and {right}: {sorted(overlap)}")
    metadata = checkpoint.get("split_metadata")
    hashes = []
    if "dataset_sha256" in checkpoint:
        hashes.append(checkpoint["dataset_sha256"])
    if metadata is not None:
        if not isinstance(metadata, dict) or metadata.get("schema_version") != 1 or metadata.get("strategy") not in {"session_group", "session_group_window_aware_fixed_test"}:
            raise LearnedDenoiseError("invalid checkpoint split_metadata schema/strategy")
        if "dataset_sha256" in metadata:
            hashes.append(metadata["dataset_sha256"])
    for expected in hashes:
        if not isinstance(expected, str) or len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected.lower()):
            raise LearnedDenoiseError("invalid checkpoint dataset SHA256")
        if dataset_sha256 is None or expected.lower() != dataset_sha256.lower():
            raise LearnedDenoiseError("dataset SHA256 does not match checkpoint training dataset")
    available = set(labels.tolist())
    known = set().union(*groups.values())
    if known - available:
        raise LearnedDenoiseError(f"checkpoint session groups missing from dataset: {sorted(known - available)}")
    if not diagnostic_all and known != available:
        raise LearnedDenoiseError("dataset has session groups not present in checkpoint split")
    if metadata is not None:
        if metadata.get("total_windows") != len(labels) or metadata.get("session_id_by_window") != labels.tolist():
            raise LearnedDenoiseError("dataset windows/session order do not match checkpoint split_metadata")
        if metadata.get("all_sessions") != sorted(available):
            raise LearnedDenoiseError("checkpoint all_sessions does not match dataset")
        partitions = metadata.get("partitions")
        if not isinstance(partitions, dict):
            raise LearnedDenoiseError("checkpoint split_metadata requires partitions")
        for name, sessions in groups.items():
            partition = partitions.get(name)
            expected_indices = np.flatnonzero(np.isin(labels, sorted(sessions))).tolist()
            if not isinstance(partition, dict) or partition.get("sessions") != sorted(sessions):
                raise LearnedDenoiseError(f"inconsistent {name} sessions in split_metadata")
            if (partition.get("window_indices") != expected_indices or partition.get("window_count") != len(expected_indices)
                    or partition.get("session_count") != len(sessions)):
                raise LearnedDenoiseError(f"inconsistent {name} windows in split_metadata")
    selected = np.arange(len(labels)) if diagnostic_all else np.flatnonzero(np.isin(labels, sorted(groups["test"])))
    if not len(selected):
        raise LearnedDenoiseError("evaluation selection contains no windows")
    return selected


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _report_metrics(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray, artifact_mask: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(clean) & np.isfinite(noisy) & np.isfinite(denoised)
    if not np.any(finite):
        raise LearnedDenoiseError("evaluation has no finite paired samples")
    clean, noisy, denoised, artifact_mask = (value[finite].astype(np.float64) for value in (clean, noisy, denoised, artifact_mask))
    error_noisy = noisy - clean
    error_denoised = denoised - clean
    noisy_rmse = float(np.sqrt(np.mean(error_noisy**2)))
    denoised_rmse = float(np.sqrt(np.mean(error_denoised**2)))
    artifact = artifact_mask.astype(bool)
    clean_region = ~artifact
    return {
        "noisy_rmse": noisy_rmse,
        "denoised_rmse": denoised_rmse,
        "rmse_improvement_fraction": float((noisy_rmse - denoised_rmse) / max(noisy_rmse, 1e-9)),
        "noisy_correlation": _safe_corr(noisy, clean),
        "denoised_correlation": _safe_corr(denoised, clean),
        "clean_region_change_rmse": float(np.sqrt(np.mean((denoised[clean_region] - noisy[clean_region]) ** 2))) if np.any(clean_region) else 0.0,
        "artifact_region_rmse_before": float(np.sqrt(np.mean((noisy[artifact] - clean[artifact]) ** 2))) if np.any(artifact) else 0.0,
        "artifact_region_rmse_after": float(np.sqrt(np.mean((denoised[artifact] - clean[artifact]) ** 2))) if np.any(artifact) else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        dataset_sha256 = _dataset_sha256(args.dataset)
        with np.load(args.dataset, allow_pickle=False) as arrays:
            clean = np.asarray(arrays["clean"], dtype=np.float32)
            noisy = np.asarray(arrays["noisy"], dtype=np.float32)
            artifact_mask = np.asarray(arrays["artifact_mask"])
            session_ids = _validated_session_ids(arrays["session_id"])
            if "sample_rate_hz" in arrays:
                rates = np.asarray(arrays["sample_rate_hz"], dtype=np.float64)
                if not np.all(np.isfinite(rates)) or not np.all(rates == float(config["sample_rate_hz"])):
                    raise LearnedDenoiseError("dataset sample_rate_hz does not match config")
        if _dataset_sha256(args.dataset) != dataset_sha256:
            raise LearnedDenoiseError("dataset changed while loading")
        if (noisy.ndim != 3 or noisy.shape != clean.shape or artifact_mask.shape != noisy.shape
                or len(session_ids) != len(noisy)):
            raise LearnedDenoiseError("paired arrays/session_id must match [windows, channels, samples]")
        if noisy.shape[1:] != (int(config["channel_count"]), int(config["window_samples"])):
            raise LearnedDenoiseError("dataset channels/window_samples do not match config")
        if artifact_mask.dtype.kind not in "biuf" or not np.all((artifact_mask == 0) | (artifact_mask == 1)):
            raise LearnedDenoiseError("artifact_mask must contain finite binary values")
        artifact_mask = artifact_mask.astype(bool)
        bundle = load_model(args.model, config, args.device)
        checkpoint = bundle.checkpoint_metadata
        selected = _select_evaluation_indices(checkpoint, session_ids, diagnostic_all=args.diagnostic_all, dataset_sha256=dataset_sha256)
        metadata = checkpoint.get("split_metadata", {})
        if "dataset_shape" in metadata and metadata["dataset_shape"] != list(noisy.shape):
            raise LearnedDenoiseError("dataset shape does not match checkpoint split_metadata")
        clean, noisy, artifact_mask = clean[selected], noisy[selected], artifact_mask[selected]
        denoised = []
        model_denoised = []
        artifact_probability = []
        confidence = []
        for item in noisy:
            result = apply_model_to_array(bundle, item, np.isfinite(item))
            denoised.append(result["denoised"])
            model_denoised.append(result["model_denoised"])
            artifact_probability.append(result["artifact_probability"])
            confidence.append(result["confidence"])
        denoised_array = np.stack(denoised)
        model_denoised_array = np.stack(model_denoised)
        metrics = _report_metrics(clean, noisy, denoised_array, artifact_mask)
        metrics["candidate_model_metrics"] = _report_metrics(clean, noisy, model_denoised_array, artifact_mask)
        threshold = 0.5
        finite = np.isfinite(clean) & np.isfinite(noisy) & np.isfinite(denoised_array)
        predicted = (np.stack(artifact_probability) >= threshold)[finite]
        evaluated_mask = artifact_mask[finite]
        true_positive = float(np.sum(predicted & evaluated_mask))
        false_positive = float(np.sum(predicted & ~evaluated_mask))
        false_negative = float(np.sum(~predicted & evaluated_mask))
        metrics.update(
            {
                "artifact_precision": true_positive / max(true_positive + false_positive, 1.0),
                "artifact_recall": true_positive / max(true_positive + false_negative, 1.0),
                "mean_confidence": float(np.mean(np.stack(confidence))),
                "windows": int(clean.shape[0]),
            }
        )
        metrics["artifact_f1"] = 2 * metrics["artifact_precision"] * metrics["artifact_recall"] / max(metrics["artifact_precision"] + metrics["artifact_recall"], 1e-9)
        report = {
            "status": "completed",
            "dataset": str(args.dataset.resolve()),
            "model": str(args.model.resolve()),
            "dataset_sha256": dataset_sha256,
            "dataset_hash_verified": "dataset_sha256" in checkpoint or "dataset_sha256" in metadata,
            "evaluation_scope": "all_diagnostic_not_heldout" if args.diagnostic_all else "heldout_test",
            "heldout": not args.diagnostic_all,
            "session_groups": sorted(set(session_ids[selected].tolist())),
            "checkpoint_session_groups": {name: checkpoint.get(f"{name}_sessions", []) for name in ("train", "validation", "test")},
            "windows": len(selected),
            "total_dataset_windows": len(session_ids),
            "window_indices": selected.tolist(),
            "target_provenance": "semi_synthetic_clean_ish_not_real_noise_clean_ground_truth",
            "warning": "Semi-synthetic evaluation against clean-ish engineering targets, NOT real-noise paired clean ground truth or clinical/downstream SSVEP validation.",
            "provenance_warning": None if "dataset_sha256" in checkpoint or "dataset_sha256" in metadata else "Legacy checkpoint has no dataset hash; dataset identity is unverified.",
            "metrics": metrics,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        console_report = {key: value for key, value in report.items() if key != "window_indices"}
        print(json.dumps(console_report, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, LearnedDenoiseError, RuntimeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
