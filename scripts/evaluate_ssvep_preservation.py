"""Measure SSVEP target-frequency preservation on downloaded Dong2023 data."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eeg_tools.learned_denoise import LearnedDenoiseError, apply_model_to_array, load_config, load_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="Dong2023 SSVEP dataset directory")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/learned_denoiser_v1.json"))
    parser.add_argument("--max-subjects", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("analysis_outputs/ssvep_preservation_v1.json"))
    parser.add_argument("--device", default="cpu")
    return parser


def _target_amplitude(values: np.ndarray, sample_rate: float, frequency: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values - np.mean(values, axis=-1, keepdims=True)
    sample_count = values.shape[-1]
    spectrum = np.abs(np.fft.rfft(values, axis=-1)) / max(sample_count, 1)
    frequencies = np.fft.rfftfreq(sample_count, d=1.0 / sample_rate)
    index = int(np.argmin(np.abs(frequencies - frequency)))
    return float(np.mean(spectrum[..., index]))


def _one_subject(path: Path, bundle: object, frequencies: list[float], sample_rate: float) -> dict[str, object]:
    loaded = loadmat(path, squeeze_me=True, struct_as_record=False)
    eeg = np.asarray(loaded.get("eegdata"), dtype=np.float32)
    if eeg.ndim != 4 or eeg.shape[0] != 8:
        raise LearnedDenoiseError(f"unexpected Dong2023 shape in {path}: {eeg.shape}")
    _, samples, targets, blocks = eeg.shape
    if samples < int(bundle.config["window_samples"]):
        raise LearnedDenoiseError(f"subject file is shorter than model window: {path}")
    raw_scores: list[float] = []
    denoised_scores: list[float] = []
    for target_index in range(min(targets, len(frequencies))):
        frequency = frequencies[target_index]
        for block_index in range(blocks):
            raw_trial = eeg[:, : int(bundle.config["window_samples"]), target_index, block_index]
            result = apply_model_to_array(bundle, raw_trial, np.isfinite(raw_trial))
            raw_scores.append(_target_amplitude(raw_trial, sample_rate, frequency))
            denoised_scores.append(_target_amplitude(result["denoised"], sample_rate, frequency))
    raw_mean = float(np.mean(raw_scores)) if raw_scores else 0.0
    denoised_mean = float(np.mean(denoised_scores)) if denoised_scores else 0.0
    return {
        "subject_file": path.name,
        "target_count": min(targets, len(frequencies)),
        "block_count": blocks,
        "raw_target_amplitude": raw_mean,
        "denoised_target_amplitude": denoised_mean,
        "ratio_denoised_to_raw": denoised_mean / max(raw_mean, 1e-12),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        bundle = load_model(args.model, config, args.device)
        subject_files = sorted(args.dataset_dir.glob("S*.mat"))[: max(1, args.max_subjects)]
        if not subject_files:
            raise LearnedDenoiseError(f"no S*.mat subject files found under {args.dataset_dir}")
        frequencies = [8.0 + 0.2 * index for index in range(40)]
        per_subject = [_one_subject(path, bundle, frequencies, float(config["sample_rate_hz"])) for path in subject_files]
        report = {
            "status": "completed",
            "dataset": str(args.dataset_dir.resolve()),
            "model": str(args.model.resolve()),
            "subjects": per_subject,
            "note": "This is a target-frequency preservation smoke evaluation, not a decoded-target accuracy result. Validate with the project protocol and independent held-out subjects before formal release.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, LearnedDenoiseError, RuntimeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
