"""Run a trained learned denoiser on one NeuroStation session."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from eeg_tools.learned_dataset import read_session_eeg
from eeg_tools.learned_denoise import LearnedDenoiseError, apply_model_to_array, load_config, load_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/learned_denoiser_v1.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    ends = np.flatnonzero(~padded[1:] & padded[:-1])
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _write_segments(path: Path, artifact: np.ndarray, confidence: np.ndarray, sample_rate: float) -> None:
    fieldnames = [
        "channel",
        "start_sample",
        "end_sample_exclusive",
        "start_time_s",
        "end_time_s",
        "mean_artifact_probability",
        "mean_confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for channel_index in range(artifact.shape[0]):
            mask = artifact[channel_index] >= 0.5
            for start, end in _segments(mask):
                writer.writerow(
                    {
                        "channel": f"CH{channel_index + 1}",
                        "start_sample": start,
                        "end_sample_exclusive": end,
                        "start_time_s": start / sample_rate,
                        "end_time_s": max(start, end - 1) / sample_rate,
                        "mean_artifact_probability": float(np.mean(artifact[channel_index, start:end])),
                        "mean_confidence": float(np.mean(confidence[start:end])),
                    }
                )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        eeg, sample_rate, valid = read_session_eeg(args.session_dir, int(config["channel_count"]))
        if abs(sample_rate - float(config["sample_rate_hz"])) > 1e-6:
            raise LearnedDenoiseError(f"session sample rate {sample_rate} does not match model config {config['sample_rate_hz']}")
        bundle = load_model(args.model, config, args.device)
        result = apply_model_to_array(bundle, eeg, valid)
        output_dir = args.output_dir or args.session_dir / "derived" / config["model_version"]
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / "denoised_signal.npz",
            denoised=result["denoised"],
            artifact_probability=result["artifact_probability"],
            confidence=result["confidence"],
            valid_mask=result["valid_mask"],
            sample_rate_hz=np.asarray(sample_rate, dtype=np.float32),
        )
        _write_segments(output_dir / "artifact_segments.tsv", result["artifact_probability"], result["confidence"], sample_rate)
        metrics = {
            "schema_version": 1,
            "status": "completed",
            "model_version": config["model_version"],
            "model_sha256": _sha256(args.model),
            "session_id": args.session_dir.name,
            "sample_rate_hz": sample_rate,
            "channel_count": int(eeg.shape[0]),
            "sample_count": int(eeg.shape[1]),
            "invalid_fraction": float(1.0 - np.mean(result["valid_mask"])),
            "artifact_fraction": float(np.mean(result["artifact_probability"] >= 0.5)),
            "mean_confidence": float(np.mean(result["confidence"])),
            "replacement_fraction": float(np.mean(~np.isclose(result["denoised"], eeg, equal_nan=True))),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "raw_session": str(args.session_dir.resolve()),
            "config": str(args.config.resolve()),
        }
        (output_dir / "quality_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "model_version": config["model_version"],
            "raw_is_immutable": True,
            "source_session_dir": str(args.session_dir.resolve()),
            "model_path": str(args.model.resolve()),
            "model_sha256": metrics["model_sha256"],
            "config_path": str(args.config.resolve()),
            "output_dir": str(output_dir.resolve()),
            "generated_files": ["denoised_signal.npz", "artifact_segments.tsv", "quality_metrics.json"],
        }
        (output_dir / "model_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({**metrics, "output_dir": str(output_dir.resolve())}, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, LearnedDenoiseError, RuntimeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
