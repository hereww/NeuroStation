"""Build a paired training archive from local NeuroStation sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eeg_tools.learned_dataset import (
    build_eegdenoisenet_arrays,
    build_training_arrays,
    concatenate_training_arrays,
    save_training_arrays,
)
from eeg_tools.learned_denoise import LearnedDenoiseError, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_root", type=Path, help="one session directory or a directory containing sessions")
    parser.add_argument("--output", type=Path, default=Path("datasets/processed/learned_denoiser_v1.npz"))
    parser.add_argument("--config", type=Path, default=Path("configs/learned_denoiser_v1.json"))
    parser.add_argument("--max-windows", type=int, default=20000)
    parser.add_argument("--eegdenoisenet-dir", type=Path, help="optional downloaded EEGdenoiseNet directory")
    parser.add_argument("--external-windows", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20261005)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        local_arrays = build_training_arrays(
            args.session_root,
            window_samples=int(config["window_samples"]),
            hop_samples=max(int(config["hop_samples"]), int(config["window_samples"]) // 2),
            expected_channels=int(config["channel_count"]),
            max_windows=args.max_windows,
            seed=args.seed,
        )
        arrays = local_arrays
        if args.eegdenoisenet_dir:
            external_arrays = build_eegdenoisenet_arrays(
                args.eegdenoisenet_dir,
                window_samples=int(config["window_samples"]),
                target_rate_hz=float(config["sample_rate_hz"]),
                max_windows=args.external_windows,
                seed=args.seed + 1,
            )
            arrays = concatenate_training_arrays(local_arrays, external_arrays)
        save_training_arrays(args.output, arrays)
        report = {
            "status": "completed",
            "output": str(args.output.resolve()),
            "windows": int(arrays["clean"].shape[0]),
            "shape": list(arrays["clean"].shape),
            "sessions": sorted({str(item) for item in arrays["session_id"]}),
            "seed": args.seed,
            "config": str(args.config.resolve()),
            "external_dataset": str(args.eegdenoisenet_dir.resolve()) if args.eegdenoisenet_dir else None,
            "warning": "clean is an engineering target created from recorded sessions and/or synthetic mixtures; it is not clinical ground truth",
        }
    except (OSError, ValueError, LearnedDenoiseError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
