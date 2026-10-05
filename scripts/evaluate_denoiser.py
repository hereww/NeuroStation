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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="paired NPZ dataset")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/learned_denoiser_v1.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _report_metrics(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray, artifact_mask: np.ndarray) -> dict[str, float]:
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
        arrays = np.load(args.dataset, allow_pickle=False)
        clean = np.asarray(arrays["clean"], dtype=np.float32)
        noisy = np.asarray(arrays["noisy"], dtype=np.float32)
        artifact_mask = np.asarray(arrays["artifact_mask"], dtype=bool)
        bundle = load_model(args.model, config, args.device)
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
        predicted = np.stack(artifact_probability) >= threshold
        true_positive = float(np.sum(predicted & artifact_mask))
        false_positive = float(np.sum(predicted & ~artifact_mask))
        false_negative = float(np.sum(~predicted & artifact_mask))
        metrics.update(
            {
                "artifact_precision": true_positive / max(true_positive + false_positive, 1.0),
                "artifact_recall": true_positive / max(true_positive + false_negative, 1.0),
                "mean_confidence": float(np.mean(np.stack(confidence))),
                "windows": int(clean.shape[0]),
            }
        )
        metrics["artifact_f1"] = 2 * metrics["artifact_precision"] * metrics["artifact_recall"] / max(metrics["artifact_precision"] + metrics["artifact_recall"], 1e-9)
        report = {"status": "completed", "dataset": str(args.dataset.resolve()), "model": str(args.model.resolve()), "metrics": metrics}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, LearnedDenoiseError, RuntimeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
