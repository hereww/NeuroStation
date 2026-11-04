"""Read-only personal diagnostic evaluator for the failed v14 checkpoint."""
from __future__ import annotations
import argparse, contextlib, hashlib, json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
SCRIPT='scripts/evaluate_ssvep_denoiser_v14.py'; TEST='tests/test_evaluate_ssvep_denoiser_v14.py'
SESSIONS=(r'C:\Users\Administrator\Documents\NeuroStation\Datasets\session_20261004_004101_105',r'C:\Users\Administrator\Documents\NeuroStation\Datasets\session_20261004_004555_413')
FALSE_FLAGS=dict(runtime_enabled=False,production_validated=False,source_domain_validated=False,independent_personal_validation=False,clean_ground_truth_available=False,presented_matches_are_gaze_accuracy=False,physiological_clean_truth=False,clinical_validation=False,training_performed=False,checkpoint_selection_performed=False,default_deployment_authorized=False)

def require(ok,msg):
 if not ok: raise ValueError('v14 evaluator: '+msg)
def sha(path):
 d=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): d.update(block)
 return d.hexdigest()
def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def atomic_json(path,value):
 path=Path(path); require(not path.exists(),'report already exists: '+str(path)); tmp=path.with_name(path.name+'.pending')
 with tmp.open('x',encoding='utf-8',newline='\n') as f:
  json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def configure():
 os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'; import torch
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 return torch
def model_context(meta): return {k:meta.get(k,False) for k in FALSE_FLAGS}
def load_diagnostic_model(root=ROOT,device='cpu',*,allow_failed_selection_diagnostic=False):
 require(allow_failed_selection_diagnostic is True,'explicit --allow-failed-selection-diagnostic required')
 from eeg_tools import ssvep_v14_runtime as api
 require(device in ('cpu','cuda:0'),'invalid device')
 state,cfg=api.preflight(root),api.load_config(root); paths={n:api.safe_path(root,getattr(api,n)) for n in ('MODEL','MANIFEST','REPORT','CONFIG','LOCK')}; before={n:sha(p) for n,p in paths.items()}
 manifest,report=read_json(paths['MANIFEST']),read_json(paths['REPORT']); common={'schema_version':1,'model_version':cfg['model_version'],'config_sha256':api.canonical_sha(cfg),'training_lock_sha256':before['LOCK'],'research_plan_sha256':api.plan_sha(cfg),'selected_epoch':None,'selection_gate_passed':False,'runtime_enabled':False,'production_validated':False}
 for artifact in (manifest,report):
  require(all(artifact.get(k)==v for k,v in common.items()),'artifact binding mismatch');require(artifact.get('checkpoint_role')=='diagnostic_last_epoch_no_eligible_checkpoint','not failed diagnostic role')
 require(report.get('status')=='training_completed' and report.get('epochs')==cfg['training']['epochs']==16 and len(report.get('history',[]))==16,'incomplete training report')
 require(all(row.get('selection',{}).get('selection_gate_passed') is False for row in report['history']),'history contains eligible epoch')
 require(manifest.get('checkpoint_sha256')==before['MODEL'] and manifest.get('training_report_sha256')==before['REPORT'],'artifact SHA mismatch')
 torch=configure();require(device=='cpu' or torch.cuda.is_available(),'CUDA unavailable');checkpoint=torch.load(paths['MODEL'],map_location='cpu',weights_only=True);require(checkpoint.get('config')==cfg and checkpoint.get('selection_gate_passed') is False,'checkpoint is not failed diagnostic')
 model=api.build_model(cfg);model.load_state_dict(checkpoint['state_dict'],strict=True);model=model.to(device).eval();require({n:sha(p) for n,p in paths.items()}==before and api.preflight(root)==state,'inputs changed while loading')
 meta={**manifest,'diagnostic_epoch':16,'selected_epoch':None,'selection_gate_passed':False,'checkpoint_role':'diagnostic_last_epoch_no_eligible_checkpoint','allow_failed_selection_diagnostic':True,**FALSE_FLAGS};return model,meta

def create_output(path,root=ROOT):
 base=(Path(root)/'analysis_outputs').resolve();candidate=Path(path).resolve();require(candidate.is_relative_to(base),'output must be under analysis_outputs');require(not candidate.exists(),'output directory already exists');candidate.mkdir(parents=True);return candidate

def checked(value,shape,name):
 import numpy as np
 a=np.asarray(value);require(a.dtype.kind=='f' and a.shape==shape and np.isfinite(a).all(),'invalid '+name);return a.astype(np.float64,copy=False)
def personal_eval(source,folder,model,meta):
 import numpy as np
 from eeg_tools import ssvep_v14_runtime as api
 from eeg_tools.personal_ssvep_decoder import decode_window,config_summary
 from eeg_tools.workstation import v9_offline as baseline
 from scripts.diagnose_neurostation_v9_output import validate_metadata,parse_raw
 before=baseline._source_hashes(source);validate_metadata(read_json(source/'session.json'),read_json(source/'channel_config.json'));raw,header=parse_raw((source/'raw_brainflow.tsv').read_bytes());raw_before=raw.copy();output,notch,receipt=api.process_recording(raw,model);require(np.array_equal(raw,raw_before),'raw input mutated');tracks={'raw':raw,'notch':checked(notch,raw.shape,'notch'),'traditional':checked(baseline._traditional(raw),raw.shape,'traditional'),'v14':checked(output,raw.shape,'v14')};targets,trials,metadata_reason=baseline._classification_inputs(source,raw);target_ids={x[0] for x in targets};rows=[]
 for trial in trials:
  row=dict(trial);row['predictions']={}
  for name,value in tracks.items():
   reason=metadata_reason or trial.get('unavailable_reason')
   if not reason:
    try: prediction=decode_window(value[trial['window_start_sample_index']:trial['offset_sample_index']].T,250.0,targets)
    except (ValueError,ArithmeticError) as exc: prediction={'available':False,'target_id':None,'reason':str(exc)}
   else: prediction={'available':False,'target_id':None,'reason':reason}
   row['predictions'][name]=prediction
  rows.append(row)
 summaries={}
 for name in tracks:
  avail=[row for row in rows if row['predictions'][name].get('available') and row.get('presented_target_id') in target_ids]; margins=[row['predictions'][name]['confidence'] for row in avail];summaries[name]={'recorded_trial_count':len(rows),'available_presented_trial_count':len(avail),'presented_match_count':sum(row['predictions'][name]['target_id']==row['presented_target_id'] for row in avail),'mean_top_two_margin':math.fsum(margins)/len(margins) if margins else None,'margin_is_probability':False}
 folder.mkdir();artifacts={}
 for name,value in tracks.items():
  p=folder/(name+'.npy');
  with p.open('xb') as f:np.save(f,value,allow_pickle=False);f.flush();os.fsync(f.fileno())
  artifacts[p.name]={'sha256':sha(p),'shape':list(value.shape),'dtype':str(value.dtype)}
 after=baseline._source_hashes(source);require(before==after,'personal source files changed');report={'status':'formal','completion_status':'complete','source_session':str(source),'source_header':header,'source_sha256_before':before,'source_sha256_after':after,'source_hashes_match':True,'targets':targets,'trials':rows,'classification_unavailable_reason':metadata_reason,'tracks':summaries,'decoder':config_summary(),'presented_matching_definition':'matches to recorded presentation, not independent gaze accuracy','artifacts':artifacts,'receipt':receipt,'checkpoint_role':meta['checkpoint_role'],**model_context(meta)};atomic_json(folder/'report.json',report);return report

def evaluate(output_directory,personal=SESSIONS,device='cpu',*,root=ROOT,allow_failed_selection_diagnostic=False):
 import numpy as np
 from eeg_tools import ssvep_v14_runtime as api
 from eeg_tools.workstation.v9_offline import _source_hashes
 require(allow_failed_selection_diagnostic is True,'explicit diagnostic flag required');sources=[Path(x).resolve() for x in personal];require(len(sources)==2 and all(p.is_dir() for p in sources),'exactly two personal sessions required');output=create_output(output_directory,root);preflight_before=api.preflight(root);source_before={str(s):_source_hashes(s) for s in sources};model,meta=load_diagnostic_model(root,device,allow_failed_selection_diagnostic=True);reports=[personal_eval(s,output/f'personal_{i:02d}',model,meta) for i,s in enumerate(sources)];require(all(source_before[r['source_session']]==r['source_sha256_after'] for r in reports),'source SHA mismatch');artifacts={}
 for p in sorted(output.rglob('*')):
  if p.is_file():artifacts[p.relative_to(output).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
 require(api.preflight(root)==preflight_before,'preflight changed')
 summary={'schema_version':1,'status':'formal','completion_status':'complete','created_at_utc':datetime.now(timezone.utc).isoformat(),'device':device,'model':meta,'allow_failed_selection_diagnostic':True,'personal':[{'source_session':r['source_session'],'report':f'personal_{i:02d}/report.json','tracks':r['tracks']} for i,r in enumerate(reports)],'personal_sources_sha256_before':source_before,'personal_sources_sha256_after':{str(s):_source_hashes(s) for s in sources},'preflight_before':preflight_before,'preflight_after':api.preflight(root),'artifacts':artifacts,'training_performed':False,'checkpoint_selection_performed':False,'clinical_validation':False,'clean_physiological_validation':False,'limitations':['Diagnostic failed-selection checkpoint cannot authorize deployment.','Presentation matches are recorded-label proxies, not gaze accuracy.','No clean physiological reference, independent gaze truth, physical layout or calibration verification.'],**model_context(meta)}
 atomic_json(output/'summary.json',summary);return summary

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-directory',required=True);p.add_argument('--personal-session',action='append',default=None);p.add_argument('--device',choices=('cpu','cuda:0'),default='cpu');p.add_argument('--allow-failed-selection-diagnostic',action='store_true');a=p.parse_args(argv);sessions=SESSIONS if a.personal_session is None else tuple(a.personal_session)
 try:
  result=evaluate(a.output_directory,sessions,a.device,allow_failed_selection_diagnostic=a.allow_failed_selection_diagnostic)
 except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:
  print(json.dumps({'status':'formal','completion_status':'incomplete','error':str(exc),**FALSE_FLAGS},ensure_ascii=False),file=sys.stderr);return 1
 print(json.dumps({'status':'formal','completion_status':'complete','report':str(Path(a.output_directory).resolve()/'summary.json'),**FALSE_FLAGS},ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
