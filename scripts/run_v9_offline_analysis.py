"""JSON-lines entry point for explicit offline v9 and fixed-filter comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from eeg_tools.workstation.v9_offline import run_job


class _JsonParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv=None):
    parser = _JsonParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    parser.add_argument('--acknowledge-out-of-domain', action='store_true')
    try:
        args = parser.parse_args(argv)
        run_job(args.session, args.output, device=args.device,
                acknowledge_out_of_domain=args.acknowledge_out_of_domain)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps({'type': 'error', 'status': 'error', 'stage': 'failed',
                          'message': str(exc) or type(exc).__name__, 'complete': False},
                         ensure_ascii=False), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
