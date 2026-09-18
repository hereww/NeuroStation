"""Run the reproducible, non-destructive EEG denoising pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_tools.denoise import run_denoise_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument(
        "--pipeline",
        type=Path,
        default=Path("configs/denoise_pipeline_v1.json"),
        help="JSON pipeline configuration",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        result = run_denoise_pipeline(
            arguments.session_dir,
            arguments.pipeline,
            arguments.output_dir,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
