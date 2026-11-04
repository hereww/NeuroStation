"""Build all frozen v4 source groups, or validate them without publishing."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eeg_tools.learned_dataset_v4 import LearnedDatasetV4Error, build_dataset


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--interface-contract", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        result = build_dataset(args.root, preregistration=args.preregistration,
                               interface_contract=args.interface_contract,
                               output=args.output, check_only=args.check_only)
        manifest = result["manifest"]
        print(json.dumps({"status": "integrity_ready" if args.check_only else "completed",
              "output": result["output"], "dataset_sha256": result["dataset_sha256"],
              "split_counts": manifest["split_counts"], "identity_counts": manifest["identity_counts"],
              "source_lock": manifest["source_lock"], "source_segments": len(manifest["raw_segments"]),
              "subject_disjoint_claim": False, "electrode_mapping_claim": False,
              "model_inference_or_metrics_executed": False,
              "automatic_processing_enabled": False, "production_validated": False},
              ensure_ascii=True, allow_nan=False, indent=2))
    except (LearnedDatasetV4Error, OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
