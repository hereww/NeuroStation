"""Persist optional quality evidence without changing raw acquisition success."""
import json
import os
from pathlib import Path
from eeg_tools.quality.engine import QualityEngine, safe_json
from eeg_tools.streaming import atomic_json
from eeg_tools.runtime_bindings import sha256


class QualitySink:
    def __init__(self, directory, eeg_rows, channels, sample_rate=250, live_path=None):
        self.directory=Path(directory);self.eeg_rows=tuple(eeg_rows)
        self.engine=QualityEngine(channels,sample_rate)
        self.live_path=Path(live_path) if live_path else self.directory/'quality-live.json'
        self.handle=(self.directory/'quality-v2.jsonl').open('x',encoding='utf-8')
        self.manifests=set()
        self.latest=None;self.updates=0;self.counts={'pass':0,'fail':0,'unknown':0}

    def __call__(self, block, record):
        try:
            return self._process(block, record)
        except Exception:
            self.latest = None
            self.engine = QualityEngine(self.engine.channels, self.engine.fs)
            raise

    def _process(self, block, record):
        result=self.engine.process(block[list(self.eeg_rows)],record)
        self.latest=safe_json(result)
        if self.engine.processing_id not in self.manifests:
            path=self.directory/('processing-'+self.engine.processing_id+'.json')
            manifest=safe_json(self.engine.manifest())
            if path.exists():
                if json.loads(path.read_text(encoding='utf-8'))!=json.loads(json.dumps(manifest)):
                    raise ValueError('existing processing manifest content mismatch')
            else:atomic_json(path,manifest)
            self.manifests.add(self.engine.processing_id)
        compact={k:v for k,v in self.latest.items() if k not in ('raw_samples','display_samples','display_valid','psd')}
        self.handle.write(json.dumps(compact,allow_nan=False,sort_keys=True)+'\n');self.handle.flush()
        atomic_json(self.live_path,self.latest)
        self.updates+=1;self.counts[result['dimensions']['signal_quality']]+=1

    def close(self):
        if self.handle.closed:return
        try:
            self.handle.flush();os.fsync(self.handle.fileno())
        finally:
            self.handle.close()
        report={'schema_version':2,'processing_id':self.engine.processing_id,'update_counts':self.counts,
                'all_updates':self.updates,'denominator':'quality update records, not trials or accuracy',
                'signal_thresholds_validated':False,'hardware_validation':'not_run',
                'quality_journal_sha256':sha256(self.directory/'quality-v2.jsonl')}
        atomic_json(self.directory/'quality-summary-v2.json',report)
        return report
