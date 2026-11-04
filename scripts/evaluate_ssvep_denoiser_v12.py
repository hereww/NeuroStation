"""Evaluate selected v12 with the same noisy-stat normalization as runtime.

No training or checkpoint selection is performed. Outputs are explicit offline
research artifacts; personal data are read-only and never used for selection.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
import sys
ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT))
PUBLIC_CONDITIONS=('identity','drift','high_burst','gaussian','combined')
SYNTHETIC_CONDITIONS=('identity','line50','drift.3','gaussian','combined')
FREQUENCIES=(10,12,15,20);SEED=20261015

def require(ok,msg):
    if not ok: raise ValueError('v12 evaluation: '+msg)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def ratio(a,b):return float(a/b) if b>0 else None

def validate_paths(output,personal=(),root=ROOT):
    from eeg_tools.workstation import v9_offline as b
    output=b._path(output,root);analysis=b._path(Path(root)/'analysis_outputs')
    require(output != analysis and output.is_relative_to(analysis) and not output.exists(),'output must be new under analysis_outputs')
    sources=[]
    for value in personal:
        source=b._path(value,root);require(source.is_dir(),'personal source missing')
        require(not source.is_relative_to(output) and not output.is_relative_to(source),'source/output overlap')
        require(all(not source.is_relative_to(other) and not other.is_relative_to(source) for other in sources),'duplicate sources')
        require(all((source/name).is_file() for name in b.NATIVE_FILES),'incomplete native source')
        sources.append(source)
    return output,sources

def validate_windows(reference,noisy,pred,ratios,defined):
    import numpy as np
    arrays=[np.asarray(x) for x in (reference,noisy,pred)]
    require(arrays[0].ndim==3 and arrays[0].shape[1:]==(8,512) and all(x.shape==arrays[0].shape and np.isfinite(x).all() for x in arrays),'invalid window arrays')
    r,n,p=[x.astype(np.float64) for x in arrays]; ratios=np.asarray(ratios);defined=np.asarray(defined)
    require(ratios.shape==defined.shape==(len(r),) and defined.dtype==np.bool_,'invalid protection arrays')
    require(np.isfinite(ratios[defined]).all() and np.all(ratios[defined]>=0),'invalid defined protection ratio')
    incoming=np.mean((n-r)**2,axis=(1,2));outgoing=np.mean((p-r)**2,axis=(1,2));change=np.mean((p-n)**2,axis=(1,2))
    rows=[{'input_mse':float(a),'output_mse':float(b),'output_to_input_mse_ratio':ratio(b,a),'normalized_change_from_input':float(c),'protected_energy_defined':bool(d),'protected_complex_energy_ratio':float(q) if d else None} for a,b,c,q,d in zip(incoming,outgoing,change,ratios,defined)]
    return {'window_count':len(rows),'input_mse':float(incoming.mean()),'output_mse':float(outgoing.mean()),'output_to_input_mse_ratio':ratio(float(outgoing.mean()),float(incoming.mean())),'mean_normalized_change_from_input':float(change.mean()),'max_normalized_change_from_input':float(change.max()),'protected_defined_window_count':int(defined.sum()),'protected_undefined_window_count':int((~defined).sum()),'max_defined_protected_complex_energy_ratio':float(ratios[defined].max()) if defined.any() else None,'windows':rows}

def evaluate_public(model,cfg,device,root=ROOT):
    import numpy as np,torch
    from eeg_tools import ssvep_v10_data as data
    reference,identities=data.load_reference_windows([7,8],root=root);conditions={}
    for index,condition in enumerate(PUBLIC_CONDITIONS):
        noisy=data.corrupt(reference.copy(),condition,SEED+100+index,cfg);outputs=[];ratios=[];defined=[]
        for start in range(0,len(noisy),32):
            out,aux=__import__('eeg_tools.ssvep_v12_runtime',fromlist=['process_windows']).process_windows(noisy[start:start+32],model,32)
            outputs.append(out);ratios.append(aux['protected_complex_energy_ratio']);defined.append(aux['protected_energy_defined'])
        metrics=validate_windows(reference,noisy,np.concatenate(outputs),np.concatenate(ratios),np.concatenate(defined))
        for row,identity in zip(metrics['windows'],identities):row['identity']=identity
        conditions[condition]={'corruption_seed':SEED+100+index,**metrics}
    return {'stage':'neural_and_restored_v12_same_noisy_statistics','subjects':[7,8],'reference_shape':list(reference.shape),'reference_is_clean_truth':False,'evaluation_is_pristine_project_holdout':False,'all_public_subjects_previously_available_in_project':True,'normalization_contract':'same noisy mean/RMS for input and target; restore residual in original numeric coordinates','conditions':conditions,'checkpoint_selection_performed':False,'parameter_search_performed':False}

def centered_errors(clean,tracks):
    import numpy as np
    result={}
    for name,signal in tracks.items():
        result[name]={}
        for region,section in (('full',slice(None)),('interior',slice(512,-512))):
            truth,value=clean[section],signal[section];tc=truth-truth.mean(axis=0,keepdims=True);vc=value-value.mean(axis=0,keepdims=True);m=float(np.mean((vc-tc)**2));e=float(np.mean(tc**2))
            result[name][region]={'samples':len(truth),'clean_centered_mse':m,'clean_centered_normalized_mse':ratio(m,e),'mean_offset_mse':float(np.mean((value.mean(axis=0)-truth.mean(axis=0))**2))}
    return result

def synthetic_eval(model):
    import numpy as np
    from scripts import evaluate_ssvep_denoiser_v10 as common
    from eeg_tools.workstation import v9_offline as b
    from eeg_tools import ssvep_v12_runtime as runtime
    targets=[(str(x),float(x)) for x in FREQUENCIES];cases=[]
    for index,(hz,condition) in enumerate((h,c) for h in FREQUENCIES for c in SYNTHETIC_CONDITIONS):
        clean,raw=common.synthetic_case(hz,condition,SEED+200+index);before=raw.copy();out,notch,receipt=runtime.process_recording(raw.copy(),model);require(np.array_equal(raw,before),'synthetic input changed')
        tracks={'raw':raw,'notch':notch,'traditional':b._traditional(raw),'v12':out};cca={}
        for name,signal in tracks.items():
            score=common._decode(signal[512:-512],targets);cca[name]={**score,'matches_known_frequency':bool(score.get('available') and score.get('target_id')==str(hz))}
        cases.append({'frequency_hz':hz,'condition':condition,'seed':SEED+200+index,'errors':centered_errors(clean,tracks),'cca':cca,'receipt':receipt})
    return {'case_count':len(cases),'frequencies':list(FREQUENCIES),'conditions':list(SYNTHETIC_CONDITIONS),'cases':cases,'known_clean_is_artificial_only':True}

def personal_eval(source,folder,model):
    import numpy as np,json
    from eeg_tools import ssvep_v12_runtime as runtime
    from scripts.diagnose_neurostation_v9_output import parse_raw,validate_metadata
    from eeg_tools.workstation import v9_offline as b
    before=b._source_hashes(source);validate_metadata(json.loads((source/'session.json').read_text(encoding='utf-8-sig')),json.loads((source/'channel_config.json').read_text(encoding='utf-8-sig')));raw,header=parse_raw((source/'raw_brainflow.tsv').read_bytes());out,notch,receipt=runtime.process_recording(raw,model)
    targets,trials,reason=b._classification_inputs(source,raw);tracks={'raw':raw,'notch':notch,'v12':out};rows=[]
    from scripts import evaluate_ssvep_denoiser_v10 as common
    for trial in trials:
        row={k:trial.get(k) for k in ('trial_index','presented_target_id','gaze_target_id','window_start_sample_index','offset_sample_index')};row['predictions']={}
        for name,signal in tracks.items():
            unavailable=reason or trial.get('unavailable_reason')
            row['predictions'][name]=({'available':False,'target_id':None,'reason':unavailable} if unavailable else common._decode(signal[trial['window_start_sample_index']:trial['offset_sample_index']],targets))
        rows.append(row)
    summaries={name:{'trial_count':sum(r['predictions'][name].get('available',False) for r in rows),'recorded_trial_count':len(rows),'presented_match_count':sum(r['predictions'][name].get('target_id')==r.get('presented_target_id') for r in rows if r['predictions'][name].get('available'))} for name in tracks}
    folder.mkdir();artifacts={}
    for name,signal in tracks.items():
        with (folder/(name+'.npy')).open('xb') as f:np.save(f,signal,allow_pickle=False)
        artifacts[name+'.npy']={'sha256':sha(folder/(name+'.npy')),'shape':list(signal.shape),'dtype':str(signal.dtype)}
    after=b._source_hashes(source);require(before==after,'personal source changed')
    report={'status':'formal','completion_status':'complete','source_session':str(source),'source_header':header,'source_sha256_before':before,'source_sha256_after':after,'source_hashes_match':True,'targets':targets,'tracks':summaries,'trials':rows,'receipt':receipt,'artifacts':artifacts,'normalization_contract':'shared v12 noisy mean/RMS and residual restoration','source_domain':'personal_Cyton','independent_personal_validation':False,'clean_ground_truth_available':False,'production_validated':False,'runtime_enabled':False,'presented_matches_are_gaze_accuracy':False,'denoising_efficacy_validated':False}
    write_json(folder/'report.json',report);return {'source_session':str(source),'report':folder.name+'/report.json','tracks':summaries,'source_sha256':before}

def research_gate(public,synthetic,personal,cfg):
    conditions=public['conditions'];identity=conditions['identity']['mean_normalized_change_from_input'];bad=[conditions[x] for x in PUBLIC_CONDITIONS if x!='identity'];incoming=sum(x['input_mse']*x['window_count'] for x in bad)/sum(x['window_count'] for x in bad);outgoing=sum(x['output_mse']*x['window_count'] for x in bad)/sum(x['window_count'] for x in bad);reduction=1-outgoing/incoming
    checks={'identity_sameunit_change':identity<=cfg['engineering_gates']['identity_mean_normalized_change_max'],'corrupted_sameunit_reduction':reduction>=cfg['engineering_gates']['corrupted_mean_error_reduction_min'],'all_conditions_nonworse':all(x['output_mse']<=x['input_mse'] for x in bad),'protection_defined':all(x['protected_defined_window_count']>0 and (x['max_defined_protected_complex_energy_ratio'] or 0)<=cfg['engineering_gates']['protected_window_complex_energy_ratio_max'] for x in conditions.values()),'normalization_contract':True,'personal_proxy_not_lower':all(x['tracks']['v12']['presented_match_count']>=x['tracks']['raw']['presented_match_count'] for x in personal)}
    return {'passed':all(checks.values()),'checks':checks,'identity_mean_sameunit_change':identity,'corrupted_pooled_sameunit_error_reduction':reduction,'runtime_enabled':False,'production_validated':False,'independent_personal_validation':False,'clean_ground_truth_available':False}

def evaluate(output,personal=(),device='cuda:0',root=ROOT):
    output,sources=validate_paths(output,personal,root=root)
    from eeg_tools import ssvep_v12_runtime as runtime
    cfg=runtime.load_config(root);runtime.preflight(root);model,metadata=runtime.load_model(root,device)
    output.mkdir();public=evaluate_public(model,cfg,device,root);synthetic=synthetic_eval(model);write_json(output/'public.json',public);write_json(output/'synthetic.json',synthetic)
    personal_results=[personal_eval(source,output/f'personal_{i:02d}',model) for i,source in enumerate(sources)]
    for item,source in zip(personal_results,sources):require(__import__('eeg_tools.workstation.v9_offline',fromlist=['_source_hashes'])._source_hashes(source)==item['source_sha256'],'source changed before summary')
    contract={'max_absolute_difference':0.0,'definition':'process_windows and explicit normalize_noisy->model->restore_residual must be identical'}
    # Verify exact runtime contract on a deterministic synthetic batch.
    import numpy as np
    from eeg_tools import ssvep_v12_runtime as rt
    probe=np.random.default_rng(20261015).normal(size=(3,8,512)).astype(np.float32);a,_=rt.process_windows(probe,model);n,m,r=rt.normalize_noisy(probe);import torch
    with torch.inference_mode():p=model(torch.as_tensor(n,device=next(model.parameters()).device))
    b=rt.restore_residual(probe,n,p.detach().cpu().numpy(),r);contract['max_absolute_difference']=float(np.max(np.abs(a-b)));require(contract['max_absolute_difference']<=1e-10,'runtime normalization contract mismatch')
    conditions=public['conditions'];bad=[conditions[x] for x in PUBLIC_CONDITIONS if x!='identity'];incoming=sum(x['input_mse']*x['window_count'] for x in bad)/sum(x['window_count'] for x in bad);outgoing=sum(x['output_mse']*x['window_count'] for x in bad)/sum(x['window_count'] for x in bad)
    summary={'schema_version':1,'status':'formal','completion_status':'complete','created_at_utc':datetime.now(timezone.utc).isoformat(),'model':metadata,'config_sha256':runtime.canonical_sha(cfg),'normalization_contract':contract,'public':{'report':'public.json','stage':public['stage']},'synthetic':{'report':'synthetic.json','case_count':synthetic['case_count']},'personal':personal_results,'research_gate':research_gate(public,synthetic,personal_results,cfg),'all_public_subjects_previously_available_in_project':True,'evaluation_is_pristine_project_holdout':False,'independent_personal_validation':False,'clean_ground_truth_available':False,'training_performed':False,'checkpoint_selection_performed':False,'runtime_enabled':False,'production_validated':False,'limitations':['Reference is a conditioned recorded surrogate, not clean EEG truth.','Fixed notch and neural residual contributions are not biological noise-removal proof.','Personal presented-target matches are not independent gaze accuracy.','No runtime or default workstation enablement follows from research gates.'],'artifacts':{'public.json':{'sha256':sha(output/'public.json')},'synthetic.json':{'sha256':sha(output/'synthetic.json')}}}
    write_json(output/'summary.json',summary);return summary

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--output-directory',required=True);p.add_argument('--personal-session',action='append',default=[]);p.add_argument('--device',choices=('cuda:0','cpu'),default='cuda:0');a=p.parse_args(argv);result=evaluate(a.output_directory,a.personal_session,a.device);print(json.dumps({'summary':str(Path(a.output_directory)/'summary.json'),'research_gate_passed':result['research_gate']['passed'],'runtime_enabled':False},ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())

