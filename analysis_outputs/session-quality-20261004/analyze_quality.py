from pathlib import Path
import csv,json,hashlib
import numpy as np
from scipy import signal,stats
from eeg_tools.offline_analysis import analyze_session
ROOT=Path.cwd()
OUT=ROOT/'analysis_outputs'/'session-quality-20261004'
BASE=Path('C:/Users/Administrator/Documents/NeuroStation/Datasets')
IDS=['session_20261004_004555_413','session_20261004_004101_105']
FS=250.0
TARGETS=[10.,12.,15.,20.]
NAMES=['Fp1','Fp2','C3','C4','P7','P8','O1','O2']

def load(p):
 return np.genfromtxt(p,delimiter='\t',names=True,encoding='utf-8-sig')
def tsv(p):
 with p.open(encoding='utf-8-sig',newline='') as h:return list(csv.DictReader(h,delimiter='\t'))
def cca(x,freq,harmonics=2):
 # Largest canonical correlation using rank-revealing SVD; no fitted training data.
 t=np.arange(x.shape[1])/FS
 y=np.array([func(2*np.pi*h*freq*t) for h in range(1,harmonics+1) for func in (np.sin,np.cos)])
 a=x.T-x.mean(axis=1); b=y.T-y.mean(axis=1)
 ua,sa,_=np.linalg.svd(a,full_matrices=False);ub,sb,_=np.linalg.svd(b,full_matrices=False)
 ua=ua[:,sa>sa[0]*1e-10];ub=ub[:,sb>sb[0]*1e-10]
 return float(np.linalg.svd(ua.T@ub,compute_uv=False)[0])
def snr(x,freq):
 # Periodogram, mean posterior power; neighborhood excludes main-lobe +/-0.5 Hz.
 f,p=signal.periodogram(x,FS,window='hann',detrend='linear',axis=-1)
 p=np.mean(p,axis=0)
 idx=np.argmin(abs(f-freq));near=(abs(f-freq)>=.5)&(abs(f-freq)<=1.5)
 return float(10*np.log10((p[idx]+1e-24)/(np.mean(p[near])+1e-24)))
def ps_power(f,p,lo,hi):
 use=(f>=lo)&(f<=hi)
 return np.trapezoid(p[:,use],f[use],axis=-1)
def notch(x):
 for freq in [50,100]:
  b,a=signal.iirnotch(freq,30,FS);x=signal.filtfilt(b,a,x,axis=-1)
 return x

def analyze(sid):
 src=BASE/sid;dst=OUT/sid;dst.mkdir(exist_ok=True)
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir() if p.is_file()}
 session=json.loads((src/'session.json').read_text(encoding='utf-8-sig'))
 quality=json.loads((src/'quality.json').read_text());preflight=json.loads((src/'preflight.json').read_text())
 raw=load(src/'raw_brainflow.tsv');events=tsv(src/'events.tsv');frames=load(src/'frame_timing.tsv')
 x=np.vstack([raw['eeg_ch'+str(i)] for i in range(1,9)])
 clean=notch(x)
 eeg=signal.sosfiltfilt(signal.butter(4,[1,40],fs=FS,btype='bandpass',output='sos'),clean,axis=-1)
 ccaeeg=signal.sosfiltfilt(signal.butter(4,[6,45],fs=FS,btype='bandpass',output='sos'),clean,axis=-1)
 # exclude first/last 2 s for global filtering edge transients; keep trials by markers.
 interior=slice(500,-500)
 f,power=signal.welch(x[:,interior],FS,nperseg=1000,noverlap=500,detrend='linear',axis=-1)
 line50=ps_power(f,power,49,51);line60=ps_power(f,power,59,61);total=ps_power(f,power,1,100)
 delta=np.mod(np.diff(raw['package_num'].astype(int)),256)
 diffs=np.diff(raw['timestamp_s']); non1=delta[delta!=1]
 marker_errors=[]
 for e in events:
  i=int(e['sample_index']);m=int(float(e['marker_code']))
  if raw['marker'][i]!=m:marker_errors.append({'event':e['event_name'],'index':i,'wanted':m,'actual':float(raw['marker'][i])})
 channels=[]
 raw_corr=np.corrcoef(x[:,interior]);band_corr=np.corrcoef(eeg[:,interior])
 for i in range(8):
  z=x[i,interior];filtered=eeg[i,interior]
  epochs=np.array([np.std(eeg[i,j:j+250]) for j in range(500,x.shape[1]-500-249,250)])
  line_peak=f[(f>=45)&(f<=65)][np.argmax(power[i,(f>=45)&(f<=65)])]
  channels.append({'channel':i+1,'configured_position':NAMES[i],'median_dc_uv':float(np.median(z)),
  'raw_min_uv':float(z.min()),'raw_max_uv':float(z.max()),'raw_std_uv':float(z.std()),
  'drift_1s_medians_p95_p05_uv':float(np.ptp(np.quantile(np.array([np.median(x[i,j:j+250]) for j in range(500,x.shape[1]-500-249,250)]),[.05,.95]))),
  'band_1_40_notched_rms_uv':float(np.sqrt(np.mean(filtered**2))),
  'band_1_40_notched_1s_rms_median_uv':float(np.median(epochs)),
  'band_1_40_notched_1s_rms_p95_uv':float(np.quantile(epochs,.95)),
  'band_1_40_notched_abs_p99_uv':float(np.quantile(abs(filtered),.99)),
  'line50_rms_uv':float(np.sqrt(line50[i])), 'line50_fraction_1_100':float(line50[i]/total[i]),
  'line60_rms_uv':float(np.sqrt(line60[i])), 'line_peak_45_65_hz':float(line_peak),
  'near_nominal_adc_rail_fraction':float(np.mean(abs(z)>185625))})
 baseline=analyze_session(src,dst/'project_fft_baseline')
 onset=[e for e in events if e['event_name']=='stimulus_onset'];trialrows=[]
 for e in onset:
  off=next(a for a in events if a['event_name']=='stimulus_offset' and a['trial_index']==e['trial_index'])
  start=int(e['sample_index']);end=int(off['sample_index']);a=start+125; freq=float(e['frequency_hz']);wanted=TARGETS.index(freq)
  row={'trial':int(e['trial_index'])+1,'frequency_hz':freq,'start':start,'end':end,'sample_duration_s':(end-start)/FS,
  'event_wall_duration_s':float(off['monotonic_s'])-float(e['monotonic_s']),
  'posterior_fundamental_snr_db':snr(clean[4:8,a:end],freq),
  'posterior_harmonic_snr_db':snr(clean[4:8,a:end],2*freq),
  'posterior_1_40_rms_uv':float(np.sqrt(np.mean(eeg[4:8,a:end]**2))),
  'o1o2_fundamental_snr_db':snr(clean[6:8,a:end],freq)}
  for label,sel in [('all8',slice(0,8)),('posterior4',slice(4,8)),('o1o2',slice(6,8))]:
   scores=np.array([cca(ccaeeg[sel,a:end],ff) for ff in TARGETS]);pred=int(np.argmax(scores))
   row[label+'_cca_scores']=scores.tolist();row[label+'_cca_prediction_hz']=TARGETS[pred];row[label+'_cca_correct']=pred==wanted
  for label,sel in [('posterior4',slice(4,8)),('o1o2',slice(6,8))]:
   scores=np.array([cca(ccaeeg[sel,a:end],ff,1) for ff in TARGETS]);pred=int(np.argmax(scores))
   row[label+'_fundamental_cca_prediction_hz']=TARGETS[pred];row[label+'_fundamental_cca_correct']=pred==wanted
  trialrows.append(row)
 methods={}
 truth=np.array([int(r['frequency_hz']) for r in trialrows]);rng=np.random.default_rng(20261004)
 perms=np.array([rng.permutation(truth) for _ in range(20000)])
 for method in ['all8_cca','posterior4_cca','o1o2_cca','posterior4_fundamental_cca','o1o2_fundamental_cca']:
  pred=np.array([int(r[method+'_prediction_hz']) for r in trialrows]);correct=int(sum(pred==truth));sur=(perms==pred).sum(axis=1)
  methods[method]={'correct':correct,'total':12,'accuracy':correct/12,'balanced_label_permutation_p_one_sided':float((1+sum(sur>=correct))/(len(sur)+1)),
   'wilson_95_interval':list(stats.binomtest(correct,12).proportion_ci(.95,method='wilson')),
   'per_target_correct':{str(int(ff)):int(sum((truth==ff)&(pred==truth))) for ff in TARGETS},
   'confusion_rows_true_columns_pred':[[int(sum((truth==a)&(pred==b))) for b in TARGETS] for a in TARGETS]}
 timing=[]
 for ti in range(12):
  mask=frames['trial_index']==ti;ff=frames[mask];interval=ff['frame_interval_ms'];interval=interval[np.isfinite(interval)]
  lit=ff['lit'];actual=ff['actual_s'];trans=np.where(np.diff(lit)>0)[0]+1
  if len(trans)>1:
   per=np.diff(actual[trans]); med=float(1/np.median(per));avg=float((len(trans)-1)/(actual[trans[-1]]-actual[trans[0]]))
  else:med=avg=None
  timing.append({'trial':ti+1,'nominal_frequency_hz':trialrows[ti]['frequency_hz'],'frames':int(mask.sum()),
  'frame_interval_p95_ms':float(np.quantile(interval,.95)), 'frame_interval_max_ms':float(max(interval)),
  'software_dropped_count':int(ff['dropped_since_previous'].sum()),'lit_rising_period_median_frequency_hz':med,
  'lit_rising_long_run_frequency_hz':avg})
 session_mark=next(e for e in events if e['event_name']=='session_start');end_mark=next(e for e in events if e['event_name']=='session_end')
 res={'session_id':sid,'source_dir':str(src),'eye_side_file_label':session['eye_side'],'started_at':session['started_at'],
 'channel_position_caveat':'Configured automatic montage; not independently verified physical electrode positions.',
 'unit_caveat':'Expected BrainFlow Cyton microvolts with its standard conversion; no gain readback, impedance measurement or calibration certificate in files.',
 'nominal_sampling_rate_hz':FS,'samples':len(raw),'saved_timestamp_span_s':float(raw['timestamp_s'][-1]-raw['timestamp_s'][0]),
 'protocol_sample_duration_s':(int(end_mark['sample_index'])-int(session_mark['sample_index']))/FS,
 'declared_protocol_duration_s':session['protocol_duration_s'],
 'finite_fraction':float(np.mean(np.isfinite(x))), 'all_zero_sample_indexes':np.where(np.all(x==0,axis=0))[0].tolist(),
 'packet_missing_estimate':int(np.sum(non1[non1>1]-1)), 'packet_duplicate_count':int(sum(delta==0)),
 'packet_nonunit_steps':int(sum(delta!=1)), 'timestamp_nonpositive_count':int(sum(diffs<=0)),
 'timestamp_gap_count_gt_6ms':int(sum(diffs>.006)), 'timestamp_gap_max_ms':float(max(diffs)*1000),
 'timestamp_interval_median_us':float(np.median(diffs)*1e6),'markers_total':len(events),'marker_mismatch':marker_errors,
 'gaze_target_label_count':int(sum(bool(e['gaze_target_id']) for e in events)),
 'event_sources':sorted(set(e['source'] for e in events)),'preflight':preflight,
 'display_quality':quality['frame_timing'], 'channels':channels,
 'raw_pairwise_corr_median':float(np.median(raw_corr[np.triu_indices(8,1)])),
 'notched_band_pairwise_corr_median':float(np.median(band_corr[np.triu_indices(8,1)])),
 'notched_band_correlation_matrix':band_corr.tolist(),
 'fft_baseline_accuracy':baseline['presented_target_accuracy'],'cca_methods':methods,'trials':trialrows,'timing_trials':timing,
 'posterior_fundamental_snr_median_db':float(np.median([r['posterior_fundamental_snr_db'] for r in trialrows])),
 'posterior_fundamental_snr_positive_count':int(sum(r['posterior_fundamental_snr_db']>0 for r in trialrows)),
 'input_sha256':hashes,'source_files_unchanged':all(hashlib.sha256((src/f).read_bytes()).hexdigest()==h for f,h in hashes.items())}
 (dst/'analysis.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
 np.savez_compressed(dst/'signals_for_plot.npz',raw_uv=x,notched_band_uv=eeg,frequencies=f,psd_uv2_per_hz=power,fs=FS)
 with (dst/'channels.csv').open('w',encoding='utf-8-sig',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(channels[0]));w.writeheader();w.writerows(channels)
 with (dst/'trials.csv').open('w',encoding='utf-8-sig',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(trialrows[0]));w.writeheader();w.writerows(trialrows)
 return res


def supplement(res):
    dst=OUT/res['session_id']; x=notch(np.load(dst/'signals_for_plot.npz')['raw_uv'])
    raw=load(BASE/res['session_id']/'raw_brainflow.tsv');idx=np.arange(len(raw));cut=slice(500,-500)
    slope=np.polyfit(idx[cut],raw['timestamp_s'][cut]-raw['timestamp_s'][500],1)[0]
    res['host_timestamp_linear_slope_effective_fs_hz']=float(1/slope)
    selectivity=[]
    for ff in TARGETS:
        matched=[];unmatched=[];rests=[]
        for tr in res['trials']:
            a=tr['start']+125;b=tr['end'];duration=min(625,b-a)
            stim=x[4:8,b-duration:b];rest=x[4:8,b+125:b+125+duration]
            score=snr(stim,ff)
            (matched if tr['frequency_hz']==ff else unmatched).append(score)
            if tr['frequency_hz']==ff and rest.shape[-1]==duration:rests.append(snr(rest,ff))
        selectivity.append({'frequency_hz':ff,'matched_snr_median_db_2p5s':float(np.median(matched)),
            'nonmatched_snr_median_db_2p5s':float(np.median(unmatched)),
            'following_rest_snr_median_db_2p5s':float(np.median(rests)) if rests else None,
            'following_rest_trials':len(rests)})
    res['selectivity_exploratory_2p5s']=selectivity
    sensitivity=[]
    for lo,hi in [(6,45),(3,40),(8,45)]:
        filtered=signal.sosfiltfilt(signal.butter(4,[lo,hi],fs=FS,btype='bandpass',output='sos'),x,axis=-1)
        for trim in [.25,.5,1.]:
            preds=[TARGETS[int(np.argmax([cca(filtered[4:8,tr['start']+int(trim*FS):tr['end']],ff,2) for ff in TARGETS]))] for tr in res['trials']]
            sensitivity.append({'band_hz':[lo,hi],'onset_trim_s':trim,'correct':int(sum(a==tr['frequency_hz'] for a,tr in zip(preds,res['trials']))),'predictions_hz':preds})
    res['cca_sensitivity_posterior4']=sensitivity
    # These are clock-sensitivity diagnostics, not sampling-rate calibration or primary results.
    filtered=signal.sosfiltfilt(signal.butter(4,[6,45],fs=FS,btype='bandpass',output='sos'),x,axis=-1)
    diagnostics=[]
    for estimated_fs in [FS,1/slope,FS*50/50.25]:
        # Scale the sinusoidal reference frequencies without changing the nominal sample coordinate.
        preds=[TARGETS[int(np.argmax([cca(filtered[4:8,tr['start']+125:tr['end']],ff*FS/estimated_fs,2) for ff in TARGETS]))] for tr in res['trials']]
        diagnostics.append({'reference_sampling_rate_hz':float(estimated_fs),'correct':int(sum(a==tr['frequency_hz'] for a,tr in zip(preds,res['trials']))),'predictions_hz':preds})
    res['cca_sampling_rate_sensitivity_diagnostic_not_calibration']=diagnostics
    res['source_files_unchanged']=all(hashlib.sha256((BASE/res['session_id']/f).read_bytes()).hexdigest()==h for f,h in res['input_sha256'].items())
    (dst/'analysis.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
    return res

if __name__=='__main__':
 results=[supplement(analyze(s)) for s in IDS]
 (OUT/'comparison.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
 for r in results:
  print(r['session_id'],r['eye_side_file_label'],'n',r['samples'],'packet_loss',r['packet_missing_estimate'],'marker_errors',r['marker_mismatch'])
  print('channels',[(a['channel'],round(a['drift_1s_medians_p95_p05_uv']),round(a['line50_rms_uv'],1),round(a['line50_fraction_1_100']*100,1),round(a['band_1_40_notched_rms_uv'],1)) for a in r['channels']])
  print('corr',round(r['raw_pairwise_corr_median'],3),round(r['notched_band_pairwise_corr_median'],3),'SNR',round(r['posterior_fundamental_snr_median_db'],2),r['posterior_fundamental_snr_positive_count'])
  print('FFT',r['fft_baseline_accuracy'],'CCA',{m:(v['correct'],round(v['balanced_label_permutation_p_one_sided'],4)) for m,v in r['cca_methods'].items()})
  print('trials',[(a['trial'],a['frequency_hz'],round(a['posterior_fundamental_snr_db'],1),a['posterior4_cca_prediction_hz'],round(a['posterior_1_40_rms_uv'],1)) for a in r['trials']])
  print('timing',[(a['trial'],a['nominal_frequency_hz'],round(a['lit_rising_long_run_frequency_hz'],2),a['software_dropped_count']) for a in r['timing_trials']])
