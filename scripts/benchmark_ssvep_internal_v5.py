"""Fixed-path v5 internal synthetic diagnostic. Freeze/review first; no training.

--check-only reads JSON/source/sidecar bytes and binary stat only: NO PT/NPZ
bytes, CUDA, or inference. A main-session freeze lock is required in both modes.
No reference/config/checkpoint/device/cohort/window redirection is supported.
Reports must be new artifacts under workspace analysis_outputs; no overwrite.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eeg_tools import ssvep_internal_benchmark_v5 as benchmark

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", default="analysis_outputs/ssvep_internal_v5_benchmark_20261007")
    args = parser.parse_args(argv)
    try:
        # Reject unsafe/colliding reports BEFORE hashes, CUDA or waveform reads.
        benchmark.report_paths(ROOT, args.output_dir, check_only=args.check_only)
        if args.check_only:
            report = benchmark.preflight(ROOT, check_only=True).report()
        else:
            report = benchmark.run_benchmark(ROOT)
        paths = benchmark.publish_reports(ROOT, args.output_dir, report, check_only=args.check_only)
    except (benchmark.InternalBenchmarkError, benchmark.core.SSVEPDenoiseV5Error, OSError, RuntimeError) as exc:
        parser.exit(2, f"internal v5 benchmark refused: {exc}\n")
    print(json.dumps({"status": "check_only_completed" if args.check_only else "internal_diagnostic_completed",
                      "inference_performed": report["inference_performed"], "case_count": report["case_count"],
                      "json": str(paths[0]), "markdown": str(paths[1]), "runtime_enabled": False,
                      "production_validated": False}, ensure_ascii=False, allow_nan=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
