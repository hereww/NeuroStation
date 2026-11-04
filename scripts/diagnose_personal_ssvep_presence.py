"""Measure target-frequency evidence and stimulus-frame integrity in personal SSVEP data.

This read-only diagnostic does not train, decode, choose a threshold, or claim
gaze accuracy. It separates stimulus metadata validity from physiological response
evidence so a forced decoder argmax is not mistaken for a correct signal.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
FALSE_FLAGS={"runtime_enabled":False,"production_validated":False,"source_domain_validated":False,"independent_personal_validation":False,"clean_ground_truth_available":False,"presented_matches_are_gaze_accuracy":False,"physiological_clean_truth":False,"clinical_validation":False,"training_performed":False,"checkpoint_selection_performed":False,"default_deployment_authorized":False}

def require(ok,msg):
 if not ok: raise ValueError('personal SSVEP evidence: '+msg)
def sha(path):
 d=hashlib.sha256();
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): d.update(b)
 return d.hexdigest()
def read_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def atomic_json(p,v):
 require(not p.exists(),'report already exists: '+str(p)); q=p.with_name(p.name+'.pending')
 with q.open('x',encoding='utf-8',newline='\n') as f: json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def create_output(p):
 base=(ROOT/'analysis_outputs').resolve(); p=Path(p).resolve();require(p.is_relative_to(base),'output must be under analysis_outputs');require(not p.exists(),'output directory already exists');p.mkdir(parents=True);return p
def parse_raw(path):
 import numpy as np
 with Path(path).open(encoding='utf-8-sig',newline='') as f:
  rows=list(csv.DictReader(f,delimiter='\t'))
 require(rows,'raw file is empty'); names=[f'eeg_ch{i}' for i in range(1,9)]; return np.asarray([[float(r[n]) for n in names] for r in rows],dtype=np.float64), np.asarray([float(r['timestamp_s']) for r in rows],dtype=np.float64)
def preprocess(x):
 from scipy.signal import butter,filtfilt,iirnotch,sosfiltfilt
 import numpy as np
 x=np.asarray(x,dtype=np.float64).T; b,a=iirnotch(50,30,fs=250);x=filtfilt(b,a,x,axis=1);sos=butter(4,[6,45],btype='bandpass',fs=250,output='sos');x=sosfiltfilt(sos,x,axis=1);x=x[[4,5,6,7]];return x-x.mean(axis=1,keepdims=True)
def amplitude(x,f):
 import numpy as np
 n=x.shape[1];t=np.arange(n,dtype=np.float64)/250;A=np.column_stack((np.sin(2*np.pi*f*t),np.cos(2*np.pi*f*t),np.ones(n)));coef=np.linalg.lstsq(A,x.T,rcond=None)[0][:2].T;return float(np.sqrt(np.sum(coef*coef,axis=1)).mean())
def frame_peak(rows, expected_frequency=None):
 import numpy as np
 if not rows:return None
 y=np.asarray([float(r['lit']) for r in rows]); f,p=np.fft.rfftfreq(len(y),1/60),np.abs(np.fft.rfft(y-y.mean()))**2; i=int(np.argmax(p[1:])+1) if len(p)>1 else 0;return {"frame_count":len(rows),"peak_hz":float(f[i]),"target_frequency_hz": None if expected_frequency is None else float(expected_frequency),"peak_error_hz": None if expected_frequency is None else float(abs(f[i]-float(expected_frequency)))}
def evaluate_session(source,output):
 import numpy as np
 from eeg_tools.workstation import v9_offline as baseline
 from scripts.diagnose_neurostation_v9_output import validate_metadata
 raw,timestamps=parse_raw(source/'raw_brainflow.tsv'); before=baseline._source_hashes(source); validate_metadata(read_json(source/'session.json'),read_json(source/'channel_config.json'));targets,trials,reason=baseline._classification_inputs(source,raw);require(not reason,reason or 'classification metadata unavailable'); target_map=dict(targets); frames=list(csv.DictReader((source/'frame_timing.tsv').open(encoding='utf-8-sig'),delimiter='\t'))
 rows=[]
 for trial in trials:
  s,e=trial['onset_sample_index'],trial['offset_sample_index']; stim=preprocess(raw[trial['window_start_sample_index']:e]); rest_start=max(0,s-750);rest=preprocess(raw[rest_start:s]) if s-rest_start>=250 else None; amps={str(int(freq) if float(freq).is_integer() else freq):amplitude(stim,freq) for _,freq in targets}; rest_amps={str(int(freq) if float(freq).is_integer() else freq):amplitude(rest,freq) for _,freq in targets} if rest is not None else {}; target=target_map.get(trial.get('presented_target_id','')); key=str(int(target) if target is not None and float(target).is_integer() else target); fr=[r for r in frames if r.get('trial_index')==trial.get('trial_index')]; rows.append({"trial_index":trial['trial_index'],"presented_target_id":trial.get('presented_target_id',''),"target_frequency_hz":target,"stimulus_amplitudes":amps,"rest_amplitudes":rest_amps,"target_stimulus_amplitude":amps.get(key),"target_rest_amplitude":rest_amps.get(key),"target_to_rest_ratio":None if not rest_amps or not rest_amps.get(key) else amps.get(key)/(rest_amps[key]+1e-12),"stimulus_frequency_rank":1+sum(value>amps.get(key,0) for value in amps.values()),"frame_timing":frame_peak(fr,target)})
 after=baseline._source_hashes(source);require(before==after,'source data changed');return {"status":"formal","completion_status":"complete","source_session":str(source),"source_sha256_before":before,"source_sha256_after":after,"source_hashes_match":True,"targets":targets,"trials":rows,"interpretation":["Target/rest ratios are descriptive and not detection thresholds.","Frame peaks validate scheduled software stimulus timing, not retinal or gaze engagement.","No clean physiological truth or independent gaze truth is available."],**FALSE_FLAGS}
def evaluate(sessions,output):
 p=create_output(output); reports=[]
 for i,s in enumerate(sessions):
  d=evaluate_session(Path(s),p/f'personal_{i:02d}');(p/f'personal_{i:02d}').mkdir();atomic_json(p/f'personal_{i:02d}'/'report.json',d);reports.append(d)
 r={"schema_version":1,"status":"formal","completion_status":"complete","created_at_utc":datetime.now(timezone.utc).isoformat(),"personal":[{"source_session":x['source_session'],"report":f'personal_{i:02d}/report.json'} for i,x in enumerate(reports)],"source_files_unchanged":all(x['source_hashes_match'] for x in reports),"limitations":["This is evidence characterization, not a decoder or denoiser."],**FALSE_FLAGS};atomic_json(p/'summary.json',r);return r
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',action='append',required=True);p.add_argument('--output-directory',required=True);a=p.parse_args(argv)
 try:r=evaluate(a.session,a.output_directory)
 except (OSError,ValueError,KeyError,TypeError) as e:print(json.dumps({"status":"formal","completion_status":"incomplete","error":str(e),**FALSE_FLAGS},ensure_ascii=False),file=sys.stderr);return 1
 print(json.dumps({"status":"formal","completion_status":"complete","report":str(Path(a.output_directory).resolve()/'summary.json'),**FALSE_FLAGS},ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())

