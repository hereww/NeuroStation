"""Outer quality/freshness gate. Model worker schema stays unchanged."""
from collections import deque
from copy import deepcopy
import math
import time


class QualityCommandGate:
    def __init__(self, clock=time.monotonic):
        self.clock=clock;self.reset_session()

    def reset_session(self):
        self.records=deque(maxlen=4096)
        self.current_identity=None;self.current_stream=None;self.select_armed=True

    def observe(self, result):
        if not isinstance(result,dict) or result.get('schema_version')!=2:
            self.records.clear();return False
        new_identity=result.get('processing_id');new_stream=result.get('stream_id')
        changed=self.current_identity is not None and (new_identity!=self.current_identity or new_stream!=self.current_stream)
        if changed:self.records.clear()
        self.current_identity,self.current_stream=new_identity,new_stream
        self.records.append(deepcopy({k:result.get(k) for k in ("start","stop","stream_id","processing_id","dimensions","updated_monotonic_s")}))
        return changed

    def assess(self, window, *, processing_id=None, now=None):
        now=self.clock() if now is None else now
        def reject(reason):return {'accepted':False,'reason':reason,'processing_id':self.current_identity}
        if not self.records:return reject('quality_unavailable')
        latest=self.records[-1]
        updated=latest.get('updated_monotonic_s')
        if not isinstance(updated,(int,float)) or not math.isfinite(updated) or not 0<=now-updated<=2:
            return reject('quality_stale')
        if not self.current_identity or (processing_id is not None and processing_id!=self.current_identity):
            return reject('processing_identity_mismatch')
        try:
            start=int(window['sample_start']);stop=int(window['sample_end'])+1 # legacy inclusive adapter
        except (ValueError,TypeError,KeyError):return reject('unknown_window_interval')
        if stop-start!=1250:return reject('quality_window_mismatch')
        cursor=start
        for record in self.records:
            if record.get('stream_id')!=self.current_stream or record.get('processing_id')!=self.current_identity:
                continue
            if record.get('stop',-1)<=cursor:continue
            if record.get('start',stop)>cursor:return reject('quality_coverage_gap')
            if any(record.get('dimensions',{}).get(key)!='pass' for key in ('integrity','signal_quality','coverage','stimulus_timing')):
                return reject('quality_rejected_or_unassessable')
            cursor=min(stop,record['stop'])
            if cursor>=stop:
                return {'accepted':True,'reason':'accepted','processing_id':self.current_identity,
                        'stream_id':self.current_stream,'start':start,'stop':stop}
        return reject('quality_coverage_incomplete')

    def latch(self, command):
        if command=='SELECT':
            if not self.select_armed:return 'UNKNOWN','select_latched'
            self.select_armed=False
        elif command in ('MOVE_LEFT','MOVE_RIGHT'):
            self.select_armed=True
        return command,'accepted' if command!='UNKNOWN' else 'rejected'
