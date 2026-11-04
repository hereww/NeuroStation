"""Verified block replay: compare stable quality fields, never re-label original data."""
from pathlib import Path
import json
import numpy as np
from eeg_tools.quality.engine import QualityEngine,safe_json
from eeg_tools.streaming import recover_blocks
from eeg_tools.runtime_bindings import sha256


STABLE_FIELDS=('schema_version','processing_id','stream_id','generation','start','stop','dimensions',
               'reasons','channels','artifacts','display_sha256','initial_window_seconds')


def replay_quality(directory):
    directory=Path(directory)
    recovery=recover_blocks(directory)
    if recovery['issues']:raise ValueError('block integrity failure: '+repr(recovery['issues']))
    reference=directory/'quality-v2.jsonl'
    observed=[json.loads(line) for line in reference.read_text(encoding='utf-8').splitlines()] if reference.is_file() else []
    before={r['file']:sha256(directory/r['file']) for r in recovery['records']}
    engine=None; mismatches=[];results=[]
    for index,record in enumerate(recovery['records']):
        if engine is None:
            engine=QualityEngine(record['channel_names'],int(record['sample_rate_hz']))
        raw=np.load(directory/record['file'],allow_pickle=False)
        result=safe_json(engine.process(raw[record['eeg_rows']],record))
        stable={k:result.get(k) for k in STABLE_FIELDS}
        results.append(stable)
        if index>=len(observed):mismatches.append({'block':index,'reason':'missing_online_quality'})
        else:
            fields=[k for k in STABLE_FIELDS if stable[k]!=observed[index].get(k)]
            if fields:mismatches.append({'block':index,'fields':fields})
    if len(observed)!=len(results):mismatches.append({'reason':'quality_count_mismatch'})
    unchanged=all(sha256(directory/name)==digest for name,digest in before.items())
    return {'schema_version':2,'verified_blocks':len(results),'verified_samples':recovery['verified_samples'],
            'recording_complete':recovery['complete'],'online_replay_match':not mismatches and bool(results),
            'raw_hashes_unchanged':unchanged,'mismatches':mismatches,'hardware_verified':False,
            'processing_id':engine.processing_id if engine else None}
