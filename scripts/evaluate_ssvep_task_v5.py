"""Report the fixed NeuroStation v5 SSVEP task decoder (no model training)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eeg_tools.ssvep_task_eval_v5 import (  # noqa: E402
    REFERENCE_RELATIVE, SSVEPTaskEvalV5Error, evaluate_task,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_RELATIVE)
    parser.add_argument("--output", type=Path, required=True, help="new workspace JSON report")
    parser.add_argument("--markdown", action="store_true", help="also create same-prefix .md report")
    parser.add_argument("--candidate-dir", type=Path, action="append", default=[])
    parser.add_argument("--candidate-signal-key", default="processed")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--check-only", action="store_true", help="JSON/sidecar checks; never open NPZ arrays")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        report = evaluate_task(args.reference_dir, output=args.output, markdown=args.markdown,
            candidate_dirs=args.candidate_dir, candidate_signal_key=args.candidate_signal_key,
            split=args.split, check_only=args.check_only)
        print(json.dumps({"status": report["status"], "output": str(args.output),
            "check_only": args.check_only, "holdout_loaded": False, "runtime_enabled": False,
            "task_metrics_computed": report["task_metrics_computed"], "learned_model_loaded": False,
            "candidate_task_metrics_computed": report["candidate_task_metrics_computed"],
            "production_validated": False, "selected_subjects": report["selected_subjects"]}, allow_nan=False))
        return 0
    except (SSVEPTaskEvalV5Error, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"SSVEP task evaluation refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
