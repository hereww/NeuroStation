"""Selected-v11 evaluation; no training or epoch selection, explicit offline only."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT))


def renamed(value):
    if isinstance(value,dict):
        return {('v11' if key=='v10' else key):renamed(item) for key,item in value.items()}
    if isinstance(value,list):return [renamed(item) for item in value]
    return value


def waveform_preservation(synthetic,model):
    import numpy as np
    from eeg_tools import ssvep_v11_runtime as runtime
    from scripts import evaluate_ssvep_denoiser_v10 as common
    rows=[]
    for index,hz in enumerate(common.FREQUENCIES):
        clean,noisy=common.synthetic_case(hz,'identity',common.SEED+200+index*5)
        output,notch,receipt=runtime.process_recording(noisy,model)
        t=np.arange(512,len(clean)-512)/250
        design=np.column_stack([np.sin(2*np.pi*f*t) for f in (hz,2*hz,7)]+
                               [np.cos(2*np.pi*f*t) for f in (hz,2*hz,7)]+[np.ones(len(t))])
        before=np.linalg.lstsq(design,clean[512:-512],rcond=None)[0]
        after=np.linalg.lstsq(design,output[512:-512],rcond=None)[0]
        ratios=[];phases=[]
        for harmonic in (0,1):
            a=before[harmonic]+1j*before[harmonic+3]
            b=after[harmonic]+1j*after[harmonic+3]
            ratios.extend((np.abs(b)/np.abs(a)).tolist())
            phases.extend(np.angle(b/a).tolist())
        rows.append({'frequency_hz':hz,'maximum_absolute_amplitude_ratio_change':float(np.max(np.abs(np.asarray(ratios)-1))),
            'maximum_absolute_phase_difference_radians':float(np.max(np.abs(phases))),
            'ground_truth_used_for_evaluation_only':True})
    return rows


def personal_report(source,folder,model):
    import numpy as np
    from scripts import evaluate_ssvep_denoiser_v10 as common
    from scripts.diagnose_neurostation_v9_output import parse_raw,validate_metadata
    from eeg_tools.workstation import v9_offline as backend
    from eeg_tools import ssvep_v11_runtime as runtime
    before=backend._source_hashes(source)
    validate_metadata(json.loads((source/'session.json').read_text(encoding='utf-8-sig')),
                      json.loads((source/'channel_config.json').read_text(encoding='utf-8-sig')))
    raw,header=parse_raw((source/'raw_brainflow.tsv').read_bytes())
    output,notch,receipt=runtime.process_recording(raw,model)
    targets,trials,reason=backend._classification_inputs(source,raw)
    tracks={'raw':raw,'notch':notch,'v11':output}
    predictions=[]
    for trial in trials:
        row={key:trial.get(key) for key in ('trial_index','presented_target_id','window_start_sample_index','offset_sample_index')}
        row['predictions']={}
        for name,signal in tracks.items():
            unavailable=reason or trial.get('unavailable_reason')
            row['predictions'][name]=({'available':False,'reason':unavailable,'target_id':None} if unavailable else
                common._decode(signal[trial['window_start_sample_index']:trial['offset_sample_index']],targets))
        predictions.append(row)
    summaries={}
    for name in tracks:
        decoded=[row for row in predictions if row['predictions'][name]['available']]
        summaries[name]={'trial_count':len(decoded),'presented_match_count':sum(row['predictions'][name]['target_id']==row['presented_target_id'] for row in decoded)}
    folder.mkdir()
    artifacts={}
    for name,signal in tracks.items():
        path=folder/(name+'.npy')
        with path.open('xb') as stream:np.save(stream,signal,allow_pickle=False)
        artifacts[path.name]={'sha256':common._sha(path),'shape':list(signal.shape)}
    backend._require(before==backend._source_hashes(source),'source changed in v11 analysis')
    report={'status':'formal','source_session':str(source),'source_sha256_before':before,
        'source_sha256_after':before,'source_hashes_match':True,'tracks':summaries,'trials':predictions,
        'receipt':receipt,'artifacts':artifacts,'source_domain_validated':False,'production_validated':False,
        'raw_sources_modified':False,'independent_gaze_truth_available':False,'clean_ground_truth_available':False,
        'development_data':True,'training_performed':False}
    common._write_json(folder/'report.json',report)
    return {'source_session':str(source),'report':folder.name+'/report.json','tracks':summaries,'source_sha256':before}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory',required=True)
    parser.add_argument('--personal-session',action='append',default=[])
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    args=parser.parse_args(argv)
    from scripts import evaluate_ssvep_denoiser_v10 as common
    from eeg_tools import ssvep_v11_runtime as runtime
    from eeg_tools.ssvep_v10_data import canonical_sha,sha
    from eeg_tools.workstation import v9_offline as backend
    output,sources=common.validate_paths(args.output_directory,args.personal_session)
    cfg=runtime.load_config();runtime.preflight()
    model,metadata=runtime.load_model(device=args.device)
    output.mkdir(parents=True)
    public=common.evaluate_public(model,cfg,args.device)
    synthetic=renamed(common.evaluate_synthetic(model))
    preservation=waveform_preservation(synthetic,model)
    common._write_json(output/'public.json',public)
    common._write_json(output/'synthetic.json',synthetic)
    common._write_json(output/'waveform_preservation.json',preservation)
    personal=[personal_report(source,output/f'personal_{index:02d}',model) for index,source in enumerate(sources)]
    for item,source in zip(personal,sources):backend._require(backend._source_hashes(source)==item['source_sha256'],'source changed before summary')
    conditions=public['conditions'];corrupted=[conditions[name] for name in common.PUBLIC_CONDITIONS if name!='identity']
    reduction=1-sum(row['output_mse'] for row in corrupted)/sum(row['input_mse'] for row in corrupted)
    identity=conditions['identity']['output_mse']
    protected=max(row['max_defined_protected_complex_energy_ratio'] or 0 for row in conditions.values())
    checks={'identity_reference_change':identity<=0.002,'corrupted_error_reduction':reduction>=0.05,
        'all_corruptions_nonworse':all(row['output_mse']<=row['input_mse'] for row in corrupted),
        'per_window_protection':protected<=1e-6 and all(row['protected_defined_window_count']>0 for row in conditions.values()),
        'synthetic_amplitude_phase_preserved':all(row['maximum_absolute_amplitude_ratio_change']<=0.02 and row['maximum_absolute_phase_difference_radians']<=0.02 for row in preservation),
        'personal_development_proxy_not_lower':all(row['tracks']['v11']['presented_match_count']>=row['tracks']['raw']['presented_match_count'] for row in personal)}
    baseline_errors=sum(not case['cca']['raw']['matches_known_frequency'] for case in synthetic['cases'])
    changed_decisions=sum(case['cca']['raw'].get('target_id')!=case['cca']['v11'].get('target_id') for case in synthetic['cases'])
    report={'schema_version':1,'status':'formal','completion_status':'complete','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'model':metadata,'config_sha256':canonical_sha(cfg),'personal':personal,
        'research_gate':{'passed':all(checks.values()),'checks':checks,'identity_mean_normalized_change':identity,
            'corrupted_pooled_error_reduction':reduction,'max_window_protected_ratio':protected},
        'synthetic_cca':{'raw_known_frequency_mismatches':baseline_errors,'candidate_changes_after_processing':changed_decisions,
            'harmonic_aliasing_disclosed':True,'not_used_as_biological_recovery_gate':True},
        'all_public_subjects_previously_available_in_project':True,'outer_evaluation_reused_after_v10':True,
        'independent_personal_validation':False,'clean_ground_truth_available':False,'production_validated':False,
        'runtime_enabled':False,'training_performed':False,'checkpoint_selection_performed':False,
        'artifacts':{name:{'sha256':sha(output/name)} for name in ('public.json','synthetic.json','waveform_preservation.json')},
        'limitations':['Identity and error gates concern conditioned recorded surrogates, not biological clean EEG.',
            'V11 is a development iteration after V10; public 7/8 results are reused development evaluation.',
            'Synthetic amplitude/phase truth is numerical only; CCA can confuse a fundamental with a candidate harmonic.',
            'Personal counts reference presented labels, not independent gaze truth.']}
    backend._require(metadata['checkpoint_sha256']==sha(ROOT/runtime.MODEL),'checkpoint changed during evaluation')
    common._write_json(output/'summary.json',report)
    print(json.dumps({'summary':str(output/'summary.json'),'research_gate_passed':report['research_gate']['passed'],'runtime_enabled':False}),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
