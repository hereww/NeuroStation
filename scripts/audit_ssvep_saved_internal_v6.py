"""JSON-only independent v6 saved internal diagnostic audit CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from eeg_tools import ssvep_saved_audit_v6 as audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Independent saved v6 INTERNAL JSON arithmetic audit; no NPZ/PT/GPU, no acceptance')
    parser.add_argument('--report', default=audit.REPORT_RELATIVE, help='workspace JSON path; never NPZ/PT or external data')
    parser.add_argument('--output-prefix', default=audit.DEFAULT_OUTPUT_PREFIX, help='new ssvep_v6_saved_internal_audit output stem; no clobber')
    parser.add_argument('--check-only', action='store_true', help='verify config/path/output plan only; do not read the report or publish')
    return parser


def main(argv: list[str] | None = None, *, root: Path = audit.ROOT) -> int:
    args = build_parser().parse_args(argv)  # --help exits before any input read.
    try:
        report = audit.audit_saved_report(root, report_relative=args.report, output_prefix=args.output_prefix,
                                         check_only=args.check_only)
    except (ValueError, OSError) as exc:
        print(f'v6 saved JSON audit refused: {exc}', file=sys.stderr)
        return 2
    keys = ('check_only', 'audit_performed', 'audit_passed', 'research_acceptance', 'runtime_enabled', 'production_validated')
    summary = {key: report[key] for key in keys}
    if not args.check_only:
        summary['mismatch_count'] = report['mismatch_count']
    print(json.dumps(summary, allow_nan=False))
    return 0 if args.check_only or report['audit_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
