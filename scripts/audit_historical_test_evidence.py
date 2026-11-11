"""Audit an explicit read-only evidence manifest; never run training or inference.

python -B scripts/audit_historical_test_evidence.py --manifest <local.json> --output <new.json>
Local inventory identity is NOT a historical validation result.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eeg_tools.evidence_bindings import EvidenceBindings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("output exists; evidence is never overwritten")
    bindings = EvidenceBindings.load(args.manifest)
    output = bindings.validate_output(args.output)
    report = bindings.audit()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, ensure_ascii=False))
    return 0 if report["all_declared_bytes_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
