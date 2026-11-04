"""Build single-channel 2-second EEGdenoiseNet v2 semi-synthetic pairs.

Fingerprint-disjoint epochs, not subject-disjoint: no source subject IDs exist.
The default output has 12000 samples; no acquisition or training is triggered.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eeg_tools.learned_dataset_v2 import (
    DEFAULT_OUTPUT, DEFAULT_SEED, DEFAULT_SOURCE_DIR, DatasetV2Error, build_dataset_v2,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR,
                        help="EEGdenoiseNet directory with EEG/EOG/EMG_all_epochs.npy (preferred) or .mat")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="NPZ output; audit manifest is <output>.manifest.json")
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--identity-fraction", type=float, default=0.4)
    parser.add_argument("--snr-min-db", type=float, default=-7.0)
    parser.add_argument("--snr-max-db", type=float, default=20.0)
    parser.add_argument("--fractions", type=float, nargs=3, default=(0.70, 0.15, 0.15),
                        metavar=("TRAIN", "VALIDATION", "TEST"), help="positive train/validation/test fractions summing to 1")
    parser.add_argument("--overwrite", action="store_true", help="explicitly replace existing v2 output and audit manifest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_dataset_v2(args.source_dir, args.output, samples=args.samples, seed=args.seed,
                                  identity_fraction=args.identity_fraction, snr_min_db=args.snr_min_db,
                                  snr_max_db=args.snr_max_db, fractions=args.fractions, overwrite=args.overwrite)
    except (DatasetV2Error, OSError, OverflowError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
