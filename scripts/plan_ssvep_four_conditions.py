"""Inspect/generate an offline plan only; no device, Qt, or stimulus imports."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eeg_tools.workstation.four_condition import FourConditionProtocol
from eeg_tools.workstation.ssvep import SSVEPProtocolError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/protocols/ssvep_four_condition_v1.json')
    parser.add_argument('--output', type=Path, help='New JSON file; existing files are never overwritten')
    args = parser.parse_args(argv)
    try:
        protocol = FourConditionProtocol.load(args.protocol)
        report = {
            'mode': 'offline_plan_only', 'parameter_status': 'pending' if protocol.pending else 'locked',
            'pending': list(protocol.pending), 'processing_identity': protocol.processing_identity,
            'hardware_started': False, 'stimulus_presented': False,
            'time_source': 'planned_relative_seconds_not_observed',
            'valid_trial_count': None, 'optical_timing_verified': False,
            'label_source': 'planned_condition_not_verified_gaze',
            'live_execution_supported': False,
        }
        if not protocol.pending:
            trials = protocol.build_trials()
            report['planned_trial_count'] = len(trials)
            report['planned_duration_s'] = trials[-1].rest_end_s
            report['trials'] = [asdict(trial) for trial in trials]
        text = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            # Exclusive creation protects prior session/derived records.
            with args.output.open('x', encoding='utf-8') as output:
                output.write(text + '\n')
        print(text)
        return 2 if protocol.pending else 0
    except (SSVEPProtocolError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
