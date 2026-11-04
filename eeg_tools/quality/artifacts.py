"""Streaming artifact rules with explicit coverage, exact >= boundaries and tails."""
from collections import deque
from dataclasses import dataclass, asdict
import math
import numpy as np
from eeg_tools.quality.contracts import identity


@dataclass(frozen=True)
class ArtifactConfig:
    sample_rate_hz: int = 250
    amplitude_window_s: float = 2
    absolute_uv: float | None = None
    peak_to_peak_uv: float | None = None
    step_uv: float | None = None
    hardware_authority: str | None = None

    def __post_init__(self):
        if self.sample_rate_hz<=0 or self.amplitude_window_s<=0:
            raise ValueError('invalid artifact window')
        for v in (self.absolute_uv,self.peak_to_peak_uv,self.step_uv):
            if v is not None and (not math.isfinite(v) or v<=0):
                raise ValueError('threshold must be positive finite or unavailable')


def merge_intervals(intervals):
    grouped={}
    for row in intervals:
        key=(row['channel'],row['rule'],row.get('stream_id'))
        grouped.setdefault(key,[]).append(dict(row))
    out=[]
    for values in grouped.values():
        merged=[]
        for row in sorted(values,key=lambda x:(x['start'],x['stop'])):
            if merged and row['start']<=merged[-1]['stop']:
                merged[-1]['stop']=max(merged[-1]['stop'],row['stop'])
            else:merged.append(row)
        out.extend(merged)
    return sorted(out,key=lambda x:(x['start'],x['channel'],x['rule']))


class ArtifactDetector:
    def __init__(self, channels, config=ArtifactConfig()):
        self.channels=tuple(channels);self.config=config
        self.window=max(1,round(config.sample_rate_hz*config.amplitude_window_s))
        self.stream=None;self.expected=None
        self.reset()

    def reset(self):
        self.last=[None]*len(self.channels)
        self.minimum=[deque() for _ in self.channels];self.maximum=[deque() for _ in self.channels]
        self.finite_run=[0]*len(self.channels)
        self.expected=None

    def process(self,samples,*,stream_id,start):
        x=np.asarray(samples,dtype=float)
        if x.ndim!=2 or x.shape[0]!=len(self.channels):raise ValueError('artifact channel mismatch')
        if self.stream!=stream_id or (self.expected is not None and start!=self.expected):self.reset()
        self.stream=stream_id;self.expected=start+x.shape[1]
        intervals=[];counts={r:{'evaluated':0,'anomalous':0,'unassessable':0} for r in ('finite','absolute','peak_to_peak','step')}
        def mark(c,rule,left,right):
            intervals.append({'channel':self.channels[c],'rule':rule,'start':int(left),'stop':int(right),
                              'stream_id':stream_id,'status':'fail','rule_version':1})
        for c,row in enumerate(x):
            for j,v in enumerate(row):
                i=start+j
                counts['finite']['evaluated']+=1
                if not np.isfinite(v):
                    counts['finite']['anomalous']+=1;mark(c,'finite',i,i+1)
                    self.last[c]=None;self.minimum[c].clear();self.maximum[c].clear();self.finite_run[c]=0
                    for r in ('absolute','peak_to_peak','step'):counts[r]['unassessable']+=1
                    continue
                for q,low in ((self.minimum[c],True),(self.maximum[c],False)):
                    while q and q[0][0]<=i-self.window:q.popleft()
                    while q and ((q[-1][1]>=v) if low else (q[-1][1]<=v)):q.pop()
                    q.append((i,v))
                self.finite_run[c]+=1
                for rule,threshold,value,left,available in (
                    ('absolute',self.config.absolute_uv,abs(v),i,True),
                    ('step',self.config.step_uv,abs(v-self.last[c]) if self.last[c] is not None else 0,i-1,self.last[c] is not None),
                    ('peak_to_peak',self.config.peak_to_peak_uv,self.maximum[c][0][1]-self.minimum[c][0][1],i-self.window+1,self.finite_run[c]>=self.window)):
                    if threshold is None or not self.config.hardware_authority or not available:
                        counts[rule]['unassessable']+=1
                    else:
                        counts[rule]['evaluated']+=1
                        if value>=threshold:
                            counts[rule]['anomalous']+=1;mark(c,rule,left,i+1)
                self.last[c]=float(v)
        for v in counts.values():
            v['status']='fail' if v['anomalous'] else 'unknown' if v['unassessable'] else 'pass'
        return {'rules':counts,'intervals':merge_intervals(intervals),'config':asdict(self.config),
                'rule_identity':identity(asdict(self.config)),'interval':'zero_based_half_open'}
