"""Verify the live prediction writer against a replay, using sealed acquisition blocks."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from eeg_tools.continuous_ssvep import Gate, METHOD, evidence_settings
from eeg_tools.live_ssvep import LiveSSVEP
from eeg_tools.session_files import sha256
from eeg_tools.streaming import recover_blocks


def verify(root, output):
    directory = root / 'raw_blocks'
    audit = recover_blocks(directory)
    if not audit['complete'] or not audit['records']:
        raise ValueError('A complete verified raw-block journal is required')
    lock = json.loads((output/'frozen_models.json').read_text(encoding='utf-8'))
    if lock.get('method') != METHOD or lock.get('evidence') != evidence_settings():
        raise ValueError('Replay was produced by another detector. Re-run the replay before verification.')
    first = audit['records'][0]
    if first['session_id'] != lock['test_session']:
        raise ValueError('Wrong recording for this replay')
    results = {}
    with TemporaryDirectory() as tmp, ExitStack() as stack:
        sinks = {}
        gates = lock['gates']
        if set(gates) not in ({'baseline'}, {'baseline', 'personal'}):
            raise ValueError('Unexpected replay model set')
        for key in gates:
            dst = Path(tmp)/key
            dst.mkdir()
            sinks[key] = LiveSSVEP(dst, first['eeg_rows'], first['package_row'], first['timestamp_row'], Gate(**lock['gates'][key]))
            stack.callback(sinks[key].file.close)
        for record in audit['records']:
            block = np.load(directory/record['file'], allow_pickle=False)
            for sink in sinks.values():
                sink.consume(block, record)
        for key, sink in sinks.items():
            sink.file.close()
            results[key] = dict(prediction_sha256=sha256(sink.path), rows=sink.count,
                                identical=sha256(sink.path) == sha256(output/f'{key}_predictions.tsv'))
    return dict(blocks=len(audit['records']), samples=audit['verified_samples'],
                journal_sha256=sha256(directory/'blocks.jsonl'), arms=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session_dir', type=Path)
    parser.add_argument('replay_dir', type=Path)
    args = parser.parse_args()
    result = verify(args.session_dir,args.replay_dir)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if all(arm['identical'] for arm in result['arms'].values()) else 1)
