"""Raw metrics, independent quality dimensions and display-only causal track."""
from collections import deque
from dataclasses import asdict
import time
import numpy as np
from scipy import signal
from eeg_tools.quality.contracts import identity, IMPLEMENTATION_VERSION
from eeg_tools.quality.filtering import CausalProcessor, ProcessingConfig
from eeg_tools.quality.artifacts import ArtifactConfig, ArtifactDetector


class QualityEngine:
    def __init__(self, channels, sample_rate=250, *, artifact_config=None, clock=time.monotonic):
        self.channels=tuple(channels);self.fs=sample_rate;self.clock=clock
        self.artifact_config=artifact_config or ArtifactConfig(sample_rate_hz=sample_rate)
        self.config=ProcessingConfig(sample_rate_hz=sample_rate,channels=self.channels)
        self.processing_id=identity({'filter':asdict(self.config),'artifact':asdict(self.artifact_config),
            'windows':{'amplitude_s':2,'psd_s':8,'welch_s':4,'overlap':.5,'drift_s':10,'trend_s':30},
            'version':IMPLEMENTATION_VERSION})
        self.filter=CausalProcessor(self.config)
        self.detector=ArtifactDetector(self.channels,self.artifact_config)
        self.stream=None;self.expected=None;self.latest=None
        self.buffers=[deque(maxlen=30*self.fs) for _ in self.channels]
        self.last_psd_stop=None;self.psd_cache=None

    def process(self, raw, record):
        began=self.clock()
        x=np.asarray(raw,dtype=float)
        start,stop,stream=record['start'],record['stop'],record['stream_id']
        if x.shape!=(len(self.channels),stop-start):raise ValueError('quality block shape mismatch')
        reset=stream!=self.stream or (self.expected is not None and start!=self.expected)
        if reset:
            self.buffers=[deque(maxlen=30*self.fs) for _ in self.channels]
            self.last_psd_stop=None;self.psd_cache=None
        self.stream,self.expected=stream,stop
        display=self.filter.process(x,stream_id=stream,start=start)
        artifacts=self.detector.process(x,stream_id=stream,start=start)
        for buffer,row in zip(self.buffers,x):buffer.extend(row)
        values=np.asarray(self.buffers,dtype=float)
        n=values.shape[1];recent=values[:,-2*self.fs:]
        finite=np.isfinite(recent).all(axis=1)
        metrics=[]
        for c,name in enumerate(self.channels):
            valid=recent[c,np.isfinite(recent[c])]
            metric={'channel':name,'finite_samples':int(len(valid)),'sample_count':int(recent.shape[1]),
                    'mean_uv':float(valid.mean()) if len(valid) else None,
                    'ac_rms_uv':float(valid.std()) if len(valid) else None,
                    'peak_to_peak_uv':float(np.ptp(valid)) if len(valid) else None,
                    'absolute_uv':float(np.max(np.abs(valid))) if len(valid) else None,
                    'amplitude_coverage':'pass' if n>=2*self.fs and finite[c] else 'unknown',
                    'drift_range_uv':None,'drift_threshold_status':'unvalidated',
                    'line_noise_status':'unknown_absolute_floor_unvalidated',
                    'trend_coverage':'pass' if n>=30*self.fs else 'unknown'}
            if n>=10*self.fs and np.isfinite(values[c,-10*self.fs:]).all():
                means=values[c,-10*self.fs:].reshape(10,self.fs).mean(axis=1)
                metric['drift_range_uv']=float(np.ptp(means))
            metrics.append(metric)
        if n>=8*self.fs and np.isfinite(values[:,-8*self.fs:]).all() and (self.last_psd_stop is None or stop-self.last_psd_stop>=self.fs):
            freq,power=signal.welch(values[:,-8*self.fs:],fs=self.fs,nperseg=4*self.fs,noverlap=2*self.fs,axis=1)
            self.psd_cache={'frequency_hz':freq.tolist(),'power_uv2_hz':power.tolist(),'sample_stop':stop}
            self.last_psd_stop=stop
            for c,metric in enumerate(metrics):
                total=float(power[c,(freq>=1)&(freq<=90)].sum())
                metric['line_ratio']={str(hz):float(power[c,abs(freq-hz)<=1].sum()/total) if total>0 else None for hz in (50,60)}
        flat=any(m['amplitude_coverage']=='pass' and m['peak_to_peak_uv']==0 for m in metrics)
        states=[r['status'] for r in artifacts['rules'].values()]
        status='fail' if 'fail' in states or flat else 'unknown' if 'unknown' in states or n<2*self.fs else 'pass'
        # Spectral/drift hardware thresholds are not validated: expose candidate metrics, not a global pass.
        if status=='pass':status='unknown'
        result={'schema_version':2,'processing_id':self.processing_id,'stream_id':stream,
                'generation':record['generation'],'start':start,'stop':stop,
                'updated_monotonic_s':self.clock(),'stale_after_s':2.0,'channels':metrics,
                'dimensions':{'acquisition':'recording','integrity':'unknown' if record.get('gaps') else 'pass',
                              'signal_quality':status,'stimulus_timing':'unknown',
                              'coverage':'pass' if n>=10*self.fs else 'unknown'},
                'reasons':['hardware_thresholds_unvalidated'] + (['flat_channel'] if flat else []) + list(record.get('gaps',[])),
                'artifacts':artifacts,'psd':self.psd_cache,
                'raw_samples':x.tolist(),'display_samples':display['samples'].tolist(),
                'display_valid':display['valid'].tolist(),
                'watermarks':record.get('watermarks',{}),
                'initial_window_seconds':{'amplitude':2,'psd':8,'drift':10,'trend':30},
                'compute_ms':(self.clock()-began)*1000}
        self.latest=result
        return result


def safe_json(value):
    if isinstance(value,dict):return {k:safe_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe_json(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value
