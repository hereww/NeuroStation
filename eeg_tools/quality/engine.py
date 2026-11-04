"""Raw metrics, independent quality dimensions and display-only causal track."""
from collections import deque
from dataclasses import asdict
import time
import hashlib
from pathlib import Path
from functools import lru_cache
import scipy
import numpy as np
from scipy import signal
from eeg_tools.quality.contracts import identity, IMPLEMENTATION_VERSION
from eeg_tools.quality.filtering import CausalProcessor, ProcessingConfig
from eeg_tools.quality.artifacts import ArtifactConfig, ArtifactDetector


@lru_cache(maxsize=1)
def implementation_id():
    from eeg_tools.runtime_bindings import sha256
    root=Path(__file__).resolve().parents[2]
    files=[root/'neurostation_signal.py',Path(__file__).resolve(),Path(__file__).with_name('artifacts.py')]
    return identity({'source_sha256':{p.name:sha256(p) for p in files if p.is_file()},
                     'packaged_version':IMPLEMENTATION_VERSION,'numpy':np.__version__,'scipy':scipy.__version__})


class QualityEngine:
    def __init__(self, channels, sample_rate=250, *, artifact_config=None, clock=time.monotonic):
        self.channels=tuple(channels);self.fs=sample_rate;self.clock=clock
        self.artifact_config=artifact_config or ArtifactConfig(sample_rate_hz=sample_rate)
        self.config=ProcessingConfig(sample_rate_hz=sample_rate,channels=self.channels)
        self.processing_id=identity({'filter':asdict(self.config),'artifact':asdict(self.artifact_config),
            'windows':{'amplitude_s':2,'psd_s':8,'welch_s':4,'overlap':.5,'drift_s':10,'trend_s':30},
            'version':IMPLEMENTATION_VERSION, 'implementation_id':implementation_id()})
        self.base_processing_id=self.processing_id
        self.acquisition_identity=None
        self.filter=CausalProcessor(self.config)
        self.detector=ArtifactDetector(self.channels,self.artifact_config)
        self.stream=None;self.expected=None;self.latest=None
        self.buffers=[deque(maxlen=30*self.fs) for _ in self.channels]
        self.last_psd_stop=None;self.psd_cache=None
        self.line_runs=[{50:0,60:0} for _ in self.channels]
        self.line_metrics=[{} for _ in self.channels]

    def manifest(self):
        return {'schema_version':2,'processing_id':self.processing_id,
                'implementation_id':implementation_id(),'acquisition_configuration_id':self.acquisition_identity,
                'raw':{'unit':'uV','filter':'none','sample_mutation':'none'},
                'display':{'config':asdict(self.config),
                           'sos_sha256':hashlib.sha256(self.filter.sos.astype('<f8').tobytes()).hexdigest(),
                           'sos_sections':len(self.filter.sos),
                           'order_definition':'4th-order high-pass + 4th-order low-pass + two 2nd-order notches',
                           'warmup_authority':'engineering display default; hardware settling not validated'},
                'online_quality':{'input':'raw','artifacts':asdict(self.artifact_config),
                                  'window_seconds':{'amplitude':2,'psd':8,'welch':4,'drift':10,'trend':30},
                                  'welch_overlap':.5,'threshold_authority':'unvalidated'},
                'offline_zero_phase':'separate denoise preprocessing.json; not equivalent to causal output',
                'model_locked':'original model preprocessing unchanged; raw window only'}

    def process(self, raw, record):
        began=self.clock()
        x=np.asarray(raw,dtype=float)
        start,stop,stream=record['start'],record['stop'],record['stream_id']
        if x.shape!=(len(self.channels),stop-start):raise ValueError('quality block shape mismatch')
        acquired=record.get('configuration_id')
        config_changed=acquired!=self.acquisition_identity
        if config_changed:
            self.acquisition_identity=acquired
            self.processing_id=identity({'processing':self.base_processing_id,'acquisition_configuration':acquired})
            self.filter.reset();self.detector.reset()
        reset=config_changed or stream!=self.stream or (self.expected is not None and start!=self.expected)
        if reset:
            self.buffers=[deque(maxlen=30*self.fs) for _ in self.channels]
            self.last_psd_stop=None;self.psd_cache=None
            self.line_runs=[{50:0,60:0} for _ in self.channels]
            self.line_metrics=[{} for _ in self.channels]
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
                metric['drift_slope_uv_s']=float(np.polyfit(np.arange(10),means,1)[0])
            if n>=30*self.fs and np.isfinite(values[c]).all():
                metric['trend_30s_range_uv']=float(np.ptp(values[c].reshape(30,self.fs).mean(axis=1)))
            metrics.append(metric)
        if n>=8*self.fs and np.isfinite(values[:,-8*self.fs:]).all() and (self.last_psd_stop is None or stop-self.last_psd_stop>=self.fs):
            freq,power=signal.welch(values[:,-8*self.fs:],fs=self.fs,nperseg=4*self.fs,noverlap=2*self.fs,axis=1)
            self.psd_cache={'frequency_hz':freq.tolist(),'power_uv2_hz':power.tolist(),'sample_stop':stop}
            self.last_psd_stop=stop
            for c,metric in enumerate(metrics):
                total=float(power[c,(freq>=1)&(freq<=90)].sum())
                lines={}
                for hz in (50,60):
                    band=abs(freq-hz)<=1
                    side=(abs(freq-hz)>=2)&(abs(freq-hz)<=5)
                    peak=float(np.max(power[c,band]));floor=float(np.median(power[c,side]))
                    ratio=float(power[c,band].sum()/total) if total>0 else None
                    prominence=float(10*np.log10(max(peak,1e-20)/max(floor,1e-20)))
                    warning=ratio is not None and ratio>=.20 and prominence>=6
                    severe=ratio is not None and ratio>=.50 and prominence>=10
                    self.line_runs[c][hz]=self.line_runs[c][hz]+1 if warning else 0
                    lines[str(hz)]={'ratio':ratio,'prominence_db':prominence,'power_uv2_hz':peak,
                                    'consecutive_updates':self.line_runs[c][hz],
                                    'candidate':'severe_review' if severe and self.line_runs[c][hz]>=3 else 'warning_review' if self.line_runs[c][hz]>=3 else 'none',
                                    'status':'unknown_absolute_floor_unvalidated'}
                self.line_metrics[c]=lines
        for c,metric in enumerate(metrics):
            metric['line_noise']=self.line_metrics[c]
        for metric in metrics:
            channel_rules=sorted({r['rule'] for r in artifacts['intervals'] if r['channel']==metric['channel']})
            if metric['amplitude_coverage']=='pass' and metric['peak_to_peak_uv']==0:
                channel_rules.append('flat_channel')
            metric['quality_status']='fail' if channel_rules else 'unknown'
            metric['quality_reasons']=channel_rules or ['hardware_thresholds_unvalidated']
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
                'display_sha256':hashlib.sha256(display['samples'].astype('<f8').tobytes()).hexdigest(),
                'compute_ms':(self.clock()-began)*1000}
        self.latest=result
        return result


def safe_json(value):
    if isinstance(value,dict):return {k:safe_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe_json(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value
