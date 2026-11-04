"""Read-only audit of explicit NeuroStation/OpenBCI dataset root.

Excludes Trash/Users, never trains/denoises/opens hardware. Canonical BrainFlow
files counted once; alternate GUI exports hashed but not double counted. ADC
rail comparison is a DECLARED default-gain24 numerical heuristic, not calibrated
hardware/clinical acceptance. All source hashes rechecked before publication.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
sys.path.insert(0,str(Path(__file__).absolute().parents[1]))
import numpy as np
from scipy.signal import welch
from eeg_tools import ssvep_test_runtime as safety

RAIL = (4.5/24)*1e6*(2**23)/(2**23-1)
POSITIVE_RAIL = (4.5/24)*1e6

def rail_match(value):
    # Signed24bit endpoints are asymmetric: -8388608 vs +8388607.
    return np.isclose(value,-RAIL,atol=1e-4,rtol=0)|np.isclose(value,POSITIVE_RAIL,atol=1e-4,rtol=0)

class AuditError(ValueError):pass

def need(ok,message):
    if not ok:raise AuditError(message)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()

def checked(root,path):
    root=safety.safe_local_path(root);path=safety.safe_local_path(path)
    need(path.is_relative_to(root),'source outside explicit dataset root')
    need(path.is_file() and path.stat().st_size<=1024**3,'source regular file / size limit')
    return path

def metadata(path):
    return safety.json_read(path.read_bytes().lstrip(b'\xef\xbb\xbf'))

def read_tsv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))

def packet_summary(packet):
    valid=np.isfinite(packet)&(packet==np.rint(packet))&(packet>=0)&(packet<=255)
    pairs=valid[:-1]&valid[1:]
    # Do not bridge malformed/invalid rows to invent continuity.
    left=np.where(valid,packet,0.).astype(np.int64)
    step=np.mod(np.diff(left)[pairs],256)
    return {'transitions':len(step),'invalid_packet_rows':int((~valid).sum()),'expected_step_one':int((step==1).sum()),'duplicate_step_zero':int((step==0).sum()),
            'nonunit_nonzero_transitions':int((step>1).sum()),
            'missing_samples_modulo256_estimate':int(np.maximum(step-1,0).sum()),
            'limitation':'sequence continuity only; cannot detect loss of exact multiples of 256 or resolve resets/segmented file boundaries'}

def window_summary(x,window=256):
    n=len(x)//window
    a=x[:n*window].reshape(n,window,8)
    if not n:return {'full_windows':0,'incomplete_tail_samples':len(x),'nonfinite_windows':0,'flat_channel_windows':0,'windows_with_default_gain24_rail':0}
    finite=np.isfinite(a).all(axis=(1,2))
    safe=np.where(np.isfinite(a),a,0.)
    # Exclude nonfinite blocks from all interpretations of flat channel.
    flat=finite & np.any(np.ptp(safe,axis=1)<=1e-8,axis=1)
    rail=rail_match(safe).any(axis=(1,2))
    return {'full_windows':n,'incomplete_tail_samples':len(x)%window,'nonfinite_windows':int((~finite).sum()),
            'flat_channel_windows':int(flat.sum()),'windows_with_default_gain24_rail':int(rail.sum()),
            'screening_only_not_an_inference_valid_mask':True}

def signal_summary(data,rate=250):
    need(data.ndim==2 and data.shape[1]==11 and len(data)>0,'expected packet + 8 EEG + timestamp + marker')
    x=data[:,1:9];time=data[:,9];channels=[]
    for i in range(8):
        y=x[:,i];finite=y[np.isfinite(y)]
        need(len(finite)>0,'no finite samples in channel')
        rail=rail_match(finite)
        pair=np.isfinite(y[:-1])&np.isfinite(y[1:])
        equal=(y[:-1]==y[1:])&pair
        # PSD is descriptive: first <=60 seconds, explicitly not whole recording.
        segment=y[:min(len(y),rate*60)]
        line=None;peak=None
        if np.isfinite(segment).all() and len(segment)>=rate*2 and np.std(segment)>1e-8:
            f,p=welch(segment,fs=rate,nperseg=min(2048,len(segment)),detrend='constant')
            band=(f>=.5)&(f<=100);total=float(p[band].sum())
            if total>0:
                line=float(p[(f>=49)&(f<=51)].sum()/total)
                peak=float(f[band][np.argmax(p[band])])
        channels.append({'channel_index_one_based':i+1,'finite_fraction':float(np.isfinite(y).mean()),
              'min_input_units':float(finite.min()),'max_input_units':float(finite.max()),'mean_input_units':float(finite.mean()),
              'centered_std_input_units':float(finite.std()),'exact_adjacent_equal_fraction':float(equal.sum()/max(1,pair.sum())),
              'default_gain24_rail_fraction':float(rail.mean()),'default_gain24_near_rail_fraction':float((np.abs(finite)>=.99*RAIL).mean()),
              'first_60s_49_to_51Hz_power_fraction_of_0p5_to_100Hz':line,'first_60s_peak_0p5_to_100Hz':peak})
    dt=np.diff(time);finite_dt=dt[np.isfinite(dt)]
    return {'actual_samples':len(x),'sample_count_duration_s':len(x)/rate,'channels':channels,
            'all_eeg_zero_rows':int(np.all(x==0,axis=1).sum()), 'packet_sequence':packet_summary(data[:,0]),
            'timestamps':{'finite_fraction':float(np.isfinite(time).mean()),'first':float(time[0]) if np.isfinite(time[0]) else None,
                'last':float(time[-1]) if np.isfinite(time[-1]) else None,
                'last_minus_first_s':float(time[-1]-time[0]) if np.isfinite(time[[0,-1]]).all() else None,
                'median_step_s':float(np.median(finite_dt)) if len(finite_dt) else None,
                'nonpositive_steps':int((finite_dt<=0).sum()),'steps_gt_2p5_sample_periods':int((finite_dt>2.5/rate).sum()),
                'warning':'host timestamp batching/gaps are not by themselves proof of EEG packet loss'},
            'marker_values':{str(float(v)):int(n) for v,n in zip(*np.unique(data[:,10],return_counts=True))},
            'fixed_window_screen':window_summary(x)}

def raw_data(path,kind):
    if kind=='neurostation':
        with path.open(encoding='utf-8-sig') as f:head=f.readline().rstrip('\r\n').split('\t')
        names=['package_num',*[f'eeg_ch{i}' for i in range(1,9)],'timestamp_s','marker']
        need(all(name in head for name in names),'raw TSV missing named channels')
        return np.loadtxt(path,delimiter='\t',skiprows=1,usecols=[head.index(n) for n in names],ndmin=2)
    with path.open(encoding='utf-8-sig') as f:first=f.readline().strip().split('\t')
    need(len(first)==24,'imported BrainFlow Cyton format requires 24 tab-separated columns')
    # Cyton board0 metadata: EEG1..8, packet0, timestamp22, marker23.
    return np.loadtxt(path,delimiter='\t',usecols=[0,*range(1,9),22,23],ndmin=2)

def parse_with_review(path,kind):
    try:
        data=raw_data(path,kind)
        return data,{'malformed_row_count':0,'malformed_rows_one_based':[], 'preserved_invalid_placeholders':False}
    except ValueError as exc:
        if kind=='neurostation':raise  # Named formal raw export must parse exactly.
        rows=[];bad=[]
        with path.open(encoding='utf-8-sig') as f:
            for number,line in enumerate(f,1):
                values=line.rstrip('\r\n').split('\t')
                try:
                    need(len(values)==24,'wrong Cyton row width')
                    row=[float(values[i]) for i in [0,*range(1,9),22,23]]
                except ValueError:
                    row=[float('nan')]*11;bad.append(number)
                rows.append(row)
        return np.asarray(rows,dtype=np.float64),{'malformed_row_count':len(bad),'malformed_rows_one_based':bad,
             'preserved_invalid_placeholders':True,'strict_parse_error':str(exc),
             'warning':'No repair performed: invalid rows are represented ONLY in audit memory as NaN; raw file row count is not a verified sample count'}


def classify(item):
    result=item['signal'];suspect=[c['channel_index_one_based'] for c in result['channels'] if c['default_gain24_rail_fraction']>=.01]
    flat=result['fixed_window_screen']['flat_channel_windows']
    issues=[]
    if suspect:issues.append('default_gain24_rail_suspected_CH'+','.join(map(str,suspect)))
    if flat:issues.append('flat_channel_windows_present')
    if item.get('malformed_source_row_count'):issues.append('malformed_raw_rows_no_repair_performed')
    if result['all_eeg_zero_rows']:issues.append('all_zero_startup_or_other_rows_present')
    if result['packet_sequence']['nonunit_nonzero_transitions'] or result['packet_sequence']['duplicate_step_zero']:issues.append('packet_sequence_discontinuity_review')
    if item.get('actual_count_matches_metadata') is False:issues.append('sample_count_metadata_mismatch')
    if item['kind']=='neurostation':
        if item['historical_quality'].get('frame_timing',{}).get('cadence_status')=='review':issues.append('historical_display_cadence_review')
        issues.append('presented_target_not_independently_observed_gaze')
        issues.append('configured_electrode_positions_not_physical_verification')
    else:issues.append('no_stimulus_labels_in_imported_session')
    return {'issues':issues,'suspect_rail_channels':suspect,
       'offline_signal_comparison':'review_candidate' if not suspect else 'artifact_characterization_only_until_hardware_review',
       'supervised_ssvep_training_ready':False,'clean_noisy_pair_available':False,
       'task_evaluation':'conditional_presented_target_proxy_only' if item.get('events',{}).get('paired_trial_count',0)>0 else 'not_available_without_external_labels'}

def audit(root):
    root=safety.safe_local_path(root);need(root.is_dir(),'dataset root missing')
    sessions=sorted(p for p in root.glob('session_*') if p.is_dir())
    imports=root/'imports'
    if imports.is_dir():sessions+=sorted(p for p in imports.iterdir() if p.is_dir())
    # Take receipts of all files in active session directories only. Do not read
    # source_path/file_path pointers from metadata, Trash, Users or secure files.
    receipts={}
    for folder in sessions:
        for p in sorted(folder.iterdir()):
            if not p.is_file():continue
            p=checked(root,p);receipts[str(p)]={'size_bytes':p.stat().st_size,'sha256':digest(p)}
    items=[]
    for folder in sessions:
        s=metadata(checked(root,folder/'session.json'))
        kind='neurostation' if folder.parent==root else 'imported_openbci'
        rate=s.get('sampling_rate_hz');need(rate==250 and s.get('channel_count')==8,'only known250Hz8channel Cyton audit')
        paths=[folder/'raw_brainflow.tsv'] if kind=='neurostation' else sorted(folder.glob('BrainFlow-RAW*.csv'))
        need(bool(paths),'canonical raw files missing')
        parsed=[parse_with_review(checked(root,p),kind) for p in paths]
        parts=[value[0] for value in parsed]
        data=np.concatenate(parts,axis=0)
        item={'session_name':folder.name,'path':str(folder),'kind':kind,
              'recorded_at':s.get('started_at',s.get('source_recorded_at',s.get('recorded_at'))),
              'imported_at':s.get('created_at') if kind!='neurostation' else None,
              'metadata_status':s.get('status'),'metadata_simulated':s.get('simulated'),'board':s.get('board','Cyton format inferred from canonical schema'),
              'sampling_rate_hz':rate,'channel_count':8,'metadata_samples':s.get('recorded_samples_per_channel'),
              'canonical_source_files':[str(p) for p in paths],
              'source_part_samples':{p.name:len(a) for p,a in zip(paths,parts)},'signal':signal_summary(data,rate),
              'source_parse_review':{p.name:value[1] for p,value in zip(paths,parsed)},
              'malformed_source_row_count':sum(value[1]['malformed_row_count'] for value in parsed),
              'fully_parsed_finite_canonical_rows':int(np.isfinite(data).all(axis=1).sum()),
              'file_boundaries_are_not_automatically_continuous':True,
              'part_boundary_sample_indices':np.cumsum([len(a) for a in parts])[:-1].tolist(),
              'numerical_unit':'board-unit in NeuroStation metadata; microvolt convention expected by software docs but acquisition calibration/gain not independently verified'}
        item['actual_count_matches_metadata']=item['signal']['actual_samples']==item['metadata_samples']
        if kind=='neurostation':
            events=read_tsv(checked(root,folder/'events.tsv'));on=[r for r in events if r.get('event_name')=='stimulus_onset'];off=[r for r in events if r.get('event_name')=='stimulus_offset']
            on_ids={r['trial_index'] for r in on};off_ids={r['trial_index'] for r in off}
            item['events']={'event_count':len(events),'onset_count':len(on),'offset_count':len(off),'paired_trial_count':len(on_ids&off_ids),
             'target_counts':{t:sum(r.get('target_id')==t for r in on) for t in sorted({r.get('target_id') for r in on})},
             'frequencies_hz':sorted({float(r['frequency_hz']) for r in on if r.get('frequency_hz')}),
             'gaze_labeled_trials':sum(bool(r.get('gaze_target_id')) for r in on),'label_source_counts':dict(__import__('collections').Counter(r.get('label_source') or '<missing>' for r in on)),
             'onset_sample_indices':[int(r['sample_index']) for r in on],
             'onset_offset_indices_within_raw':all(0<=int(r['sample_index'])<len(data) for r in on+off)}
            item['historical_quality']=metadata(checked(root,folder/'quality.json'))
            channels=metadata(checked(root,folder/'channel_config.json'))
            item['configured_channel_order']=channels.get('channel_order')
            item['configured_electrode_positions']=[c.get('electrode_position') for c in channels.get('channels',[])]
            item['physical_electrode_positions_verified_this_audit']=False
            item['historical_validation_mode']=s.get('validation_mode')
            item['manifest_check']={'missing':[],'mismatched':[],'verified_count':0}
            with (folder/'manifest.csv').open(encoding='utf-8-sig',newline='') as f:
                for row in csv.DictReader(f):
                    name=row.get('file_name')
                    need(name and Path(name).name==name,'unsafe manifest filename')
                    p=checked(root,folder/name)
                    got=receipts.get(str(p))
                    if got is None:item['manifest_check']['missing'].append(name)
                    elif row.get('sha256')!=got['sha256']:item['manifest_check']['mismatched'].append(name)
                    else:item['manifest_check']['verified_count']+=1
        else:
            item['events']={'event_count':s.get('event_count',0),'paired_trial_count':0,'labels_available':False}
            recorded=s.get('raw_files',[])
            item['import_hash_check']={'verified_count':0,'mismatched':[]}
            for row in recorded:
                name=row.get('name');need(name and Path(name).name==name,'unsafe imported filename')
                got=receipts.get(str(folder/name))
                if not got or got['sha256']!=row.get('content_sha256'):item['import_hash_check']['mismatched'].append(name)
                else:item['import_hash_check']['verified_count']+=1
        item['assessment']=classify(item);items.append(item)
        print(folder.name,'samples=',item['signal']['actual_samples'],'railCH=',item['assessment']['suspect_rail_channels'],flush=True)
    for path,receipt in receipts.items():need(digest(checked(root,Path(path)))==receipt['sha256'],'source changed during audit: '+path)
    return {'schema_version':1,'status':'formal','audit_stage':'read_only_personal_dataset_inventory_and_numeric_screen',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_root':str(root),'source_immutable_verified':True,
        'neurostation_session_count':sum(i['kind']=='neurostation' for i in items),'imported_session_count':sum(i['kind']=='imported_openbci' for i in items),
        'total_canonical_samples':sum(i['signal']['actual_samples'] for i in items),'metadata_count_mismatch_sessions':[i['session_name'] for i in items if not i['actual_count_matches_metadata']],
        'excluded_from_waveform_and_metadata_read':['Trash','Users'],'gui_exports_double_counted':False,
        'heuristic':{'default_gain':24,'vref_assumed_volts':4.5,'negative_rail_input_convention':-RAIL,'positive_rail_input_convention':POSITIVE_RAIL,'rail_tolerance_absolute':1e-4,'rail_flag_channel_fraction':.01,
                     'hardware_gain_and_unit_calibration_verified':False,'clinical_quality_or_device_failure_diagnosis':False},
        'training_performed':False,'model_inference_performed':False,'raw_sources_modified':False,'hardware_opened':False,
        'historical_quality_or_validation_status_rewritten':False,'runtime_enabled':False,'production_validated':False,
        'sessions':items,'source_receipts':receipts,'auditor_sha256':digest(Path(__file__).absolute())}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args(argv)
    out=safety.safe_local_path(a.output);need(out.parent==safety.ROOT/'analysis_outputs','output must be under workspace analysis_outputs')
    need(not out.exists(),'output exists; no-clobber')
    report=audit(a.source);safety.exclusive_write(out,safety.json_bytes(report));print('REPORT:',out);return 0
if __name__=='__main__':raise SystemExit(main())
