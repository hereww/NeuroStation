"""Stateful causal filters for display/quality, separate from locked model input."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy import signal
from eeg_tools.quality.contracts import identity, IMPLEMENTATION_VERSION


@dataclass(frozen=True)
class ProcessingConfig:
    sample_rate_hz: float = 250
    channels: tuple[str, ...] = ('Fp1','Fp2','C3','C4','P7','P8','O1','O2')
    purpose: str = 'display'
    highpass_hz: float = 5
    lowpass_hz: float = 50
    notch_hz: tuple[float, ...] = (50,60)
    order: int = 4
    warmup_samples: int = 250
    reference: str = 'unchanged'
    normalization: str = 'none'
    unit: str = 'uV'
    phase: str = 'causal'

    def __post_init__(self):
        if self.purpose not in ('display','online_quality') or self.phase!='causal':
            raise ValueError('causal core cannot replace offline or model-locked processing')
        if not 0 < self.highpass_hz < self.lowpass_hz < self.sample_rate_hz/2:
            raise ValueError('invalid bandpass')
        if any(not 0<f<self.sample_rate_hz/2 for f in self.notch_hz):
            raise ValueError('invalid notch')
        if self.order<1 or self.warmup_samples<0 or not self.channels or len(set(self.channels))!=len(self.channels):
            raise ValueError('invalid filter contract')
        if self.reference!='unchanged' or self.normalization!='none' or self.unit!='uV':
            raise ValueError('unsupported reference, normalization or unit')

    @property
    def processing_id(self):
        return identity({'config':asdict(self),'implementation':IMPLEMENTATION_VERSION,'engine':'causal-sos-v1'})


class CausalProcessor:
    def __init__(self, config=ProcessingConfig()):
        self.config=config
        fs=config.sample_rate_hz
        parts=[signal.butter(config.order,config.highpass_hz,'highpass',fs=fs,output='sos'),
               signal.butter(config.order,config.lowpass_hz,'lowpass',fs=fs,output='sos')]
        for freq in config.notch_hz:
            b,a=signal.iirnotch(freq,25 if freq==50 else 30,fs=fs)
            parts.append(signal.tf2sos(b,a))
        self.sos=np.concatenate(parts)
        self.stream_id=None
        self.expected=None
        self.reset()

    def reset(self):
        self.state=np.zeros((len(self.config.channels),len(self.sos),2))
        self.age=np.zeros(len(self.config.channels),dtype=np.int64)
        self.expected=None

    def process(self, samples, *, stream_id, start):
        x=np.asarray(samples,dtype=float)
        if x.ndim!=2 or x.shape[0]!=len(self.config.channels):
            raise ValueError('channel order/shape mismatch')
        if stream_id!=self.stream_id or (self.expected is not None and start!=self.expected):
            self.reset()
        self.stream_id=stream_id
        self.expected=start+x.shape[1]
        out=np.full_like(x,np.nan); valid=np.zeros(x.shape,dtype=bool)
        for c,row in enumerate(x):
            finite=np.isfinite(row)
            cuts=np.flatnonzero(np.diff(np.r_[False,finite,False])).reshape(-1,2)
            previous=0
            for left,right in cuts:
                if left>previous:
                    self.state[c].fill(0); self.age[c]=0
                out[c,left:right],self.state[c]=signal.sosfilt(self.sos,row[left:right],zi=self.state[c])
                valid[c,left:right]=self.age[c]+np.arange(1,right-left+1)>self.config.warmup_samples
                self.age[c]+=right-left;previous=right
            if not len(cuts) or cuts[-1,1]<len(row):
                self.state[c].fill(0);self.age[c]=0
        return {'samples':out,'valid':valid,'processing_id':self.config.processing_id,
                'stream_id':stream_id,'start':start,'stop':self.expected,'warmup_samples':self.config.warmup_samples}
