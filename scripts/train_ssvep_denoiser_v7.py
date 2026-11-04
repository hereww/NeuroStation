"""Frozen CUDA:0 v7 training CLI. Default --check-only reads metadata ONLY.

No config/cohort/device/output overrides, no lock creation, resume, runtime,
production, outer or holdout modes. --run requires main-session 50-entry lock.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _experiment():
    return importlib.import_module('eeg_tools.ssvep_v7_experiment')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true', help='config/prereg/lock metadata only (default); no torch/CUDA/data')
    mode.add_argument('--run', action='store_true', help='exact frozen training, only after independent 50-entry freeze')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    api = _experiment()
    try:
        # Reject output collision before CUDA, model imports or true data reads.
        api.assert_new_outputs(ROOT)
        state = api.preflight(root=ROOT, check_only=not args.run)
        if not args.run:
            print(json.dumps(state.report(), ensure_ascii=False, allow_nan=False))
            return 0
        report = api.run_training(state)
        print(json.dumps({'status': 'training_completed',
            'best_epochs': {d: report['domain_reports'][d]['best_epoch'] for d in api.DOMAINS},
            'internal_gates_passed': {d: report['domain_reports'][d]['internal_gates_passed'] for d in api.DOMAINS},
            'runtime_enabled': False, 'production_validated': False}, ensure_ascii=False, allow_nan=False))
        return 0
    except (api.V7Error, OSError, ImportError) as exc:
        print(json.dumps({'status': 'blocked', 'error': str(exc), 'runtime_enabled': False,
                          'production_validated': False}, ensure_ascii=False, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
