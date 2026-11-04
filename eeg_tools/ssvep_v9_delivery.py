"""One preregistered v9 GPU diagnostic + JSON audit + exclusive packaging.

Development subjects5-6 only; SAME v8 corruption seeds for a paired engineering
comparison, NOT new participants/independent evidence. No retraining/sweep.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib
import io
import json
from pathlib import Path
from . import ssvep_test_runtime as rt

CODE_PATHS = (
 'eeg_tools/ssvep_denoise_v9.py','eeg_tools/ssvep_test_runtime.py','eeg_tools/ssvep_v9_delivery.py',
 'scripts/run_ssvep_test_model.py','scripts/build_ssvep_test_model.py','scripts/start_ssvep_test_model.ps1',
 'scripts/create_ssvep_test_demo.py',
 'tests/test_ssvep_test_model.py','tests/test_ssvep_v9_delivery.py',
)

def diagnostic(root=rt.ROOT):
    cfg,lock,_=rt.preflight(root,packaged=False)
    prereg=rt.json_read(rt.read_bytes(rt.safe_local_path(rt.PREREG,root)))
    rt.require(prereg['change_budget_normalized_mse']==0.00045 and prereg['parameter_search']=='none; exactly one fixed adapter', 'preregistration mismatch')
    ev=importlib.import_module('eeg_tools.ssvep_v8_internal_eval')
    state=ev.preflight(root,check_only=False)
    models,gpu=rt.load_models(root,packaged=False,device='cuda:0')
    before=ev._model_digest(models)
    records=importlib.import_module('eeg_tools.ssvep_denoise_v5').load_reference_records(root,subjects=(5,6))
    rows=ev.infer_cases(state,records,models,importlib.import_module('torch'))
    after=ev._model_digest(models)
    rt.require(before==after,'core state changed during inference')
    ev.verify_frozen(state);rt.verify_files(lock['files'],root)
    report={
      'schema_version':1,'status':'formal','model_version':cfg['model_version'], 'created_at_utc':rt.now(),
      'stage':'v9_conservative_inference_internal_diagnostic','inference_performed':True,'gpu':gpu,
      'config_sha256':rt.sha(rt.read_bytes(rt.safe_local_path(rt.CONFIG,root))),
      'inference_lock_sha256':rt.sha(rt.read_bytes(rt.safe_local_path(rt.LOCK,root))),
      'base_checkpoint_sha256':rt.BASE_SHA,'weights_retrained_this_version':False,
      'diagnostic_seed':state.config['seeds']['diagnostic'], 'cases':rows,
      'model_state_sha256_before':before,'model_state_sha256_after':after,
      'scope':{'subjects':[5,6], 'outer_validation_waveforms_loaded':False,'holdout_waveforms_loaded':False,
               'D_drive_accessed_this_run':False,'recorded_reference_is_clean_ground_truth':False,
               'same_seeds_as_v8_paired_comparison':True,'independent_validation':False},
      'deployment':{'runtime_enabled':False,'production_validated':False,'scope':'explicit_offline_test_only'},
      'limitations':['No causal or overlap-add validation','No task accuracy or clinical validity evidence',
                     'Frozen manifests have unknown anatomical channel order and native units',
                     'Identity change budget is in noisy-normalized coordinates; no truth-based safety guarantee on arbitrary signals'],
      **ev.summarize_diagnostic(rows,state.config),
    }
    return report

def audit_report(report, root=rt.ROOT):
    ev=importlib.import_module('eeg_tools.ssvep_v8_internal_eval')
    cfg=rt.json_read(rt.read_bytes(rt.safe_local_path(rt.BASE_CONFIG,root)))
    rt.require(report['model_version']=='learned_denoise_v9_ssvep' and report['weights_retrained_this_version'] is False,'wrong model provenance')
    rt.require(report['base_checkpoint_sha256']==rt.BASE_SHA,'base hash mismatch')
    rt.require(report['inference_lock_sha256']==rt.sha(rt.read_bytes(rt.safe_local_path(rt.LOCK,root))),'inference lock mismatch')
    rt.require(report['config_sha256']==rt.sha(rt.read_bytes(rt.safe_local_path(rt.CONFIG,root))),'config hash mismatch')
    rt.require(report['model_state_sha256_before']==report['model_state_sha256_after'],'weights mutated')
    rt.require(report['scope']=={'subjects':[5,6], 'outer_validation_waveforms_loaded':False,'holdout_waveforms_loaded':False,
               'D_drive_accessed_this_run':False,'recorded_reference_is_clean_ground_truth':False,
               'same_seeds_as_v8_paired_comparison':True,'independent_validation':False},'scope mismatch')
    rt.require(report['deployment']=={'runtime_enabled':False,'production_validated':False,'scope':'explicit_offline_test_only'},'deployment mismatch')
    recomputed=ev.summarize_diagnostic(report['cases'],cfg)
    for key,value in recomputed.items():rt.require(report.get(key)==value,'saved aggregation mismatch: '+key)
    return {'schema_version':1,'status':'formal','created_at_utc':rt.now(), 'audit_passed':True,
            'mismatch_count':0,'cases_checked':len(report['cases']), 'method':'JSON_only_arithmetic_and_provenance',
            'independent_reinference':False,'production_acceptance':False,
            'report_sha256':rt.sha(rt.json_bytes(report))}

def package(root=rt.ROOT):
    import torch
    cfg,lock,_=rt.preflight(root,packaged=False)
    # All files are exclusive. A partial crash is kept, never silently overwritten.
    paths=[rt.safe_local_path(rel,root) for rel in (rt.MODEL,rt.MANIFEST)]
    rt.require(all(not p.exists() for p in paths),'package exists; no-clobber')
    report=rt.json_read(rt.read_bytes(rt.safe_local_path(rt.REPORT,root)))
    audit=rt.json_read(rt.read_bytes(rt.safe_local_path(rt.AUDIT,root)))
    audit_report(report,root)
    rt.require(audit['audit_passed'] is True and audit['report_sha256']==rt.sha(rt.read_bytes(rt.safe_local_path(rt.REPORT,root))), 'audit/report binding failed')
    base=rt.read_bytes(rt.safe_local_path(rt.BASE_MODEL,root));rt.require(rt.sha(base)==rt.BASE_SHA,'base changed')
    ck={'schema_version':1,'status':'formal','model_version':cfg['model_version'],'config':cfg,
        'base_checkpoint_bytes':base,'base_checkpoint_sha256':rt.BASE_SHA,
        'inference_lock_sha256':rt.sha(rt.read_bytes(rt.safe_local_path(rt.LOCK,root)))}
    stream=io.BytesIO();torch.save(ck,stream)
    files={rt.CONFIG:rt.sha(rt.read_bytes(rt.safe_local_path(rt.CONFIG,root))),
           rt.PREREG:rt.sha(rt.read_bytes(rt.safe_local_path(rt.PREREG,root))),
           rt.LOCK:ck['inference_lock_sha256'],rt.REPORT:rt.sha(rt.read_bytes(rt.safe_local_path(rt.REPORT,root))),
           rt.AUDIT:rt.sha(rt.read_bytes(rt.safe_local_path(rt.AUDIT,root))),rt.MODEL:rt.sha(stream.getvalue())}
    manifest={'schema_version':1,'status':'formal','model_version':cfg['model_version'],'created_at_utc':rt.now(),
              'files':files,'base_checkpoint_sha256':rt.BASE_SHA,'weights_retrained_this_version':False,
              'domain_internal_gates':{d:report['dataset_gates'][d]['all_passed'] for d in rt.DOMAINS},
              'all_internal_gates_passed':report['all_internal_gates_passed'],
              'runtime_enabled':False,'production_validated':False,'usage':'explicit offline testing, not production validation'}
    rt.verify_files(lock['files'],root)
    rt.exclusive_write(paths[0],stream.getvalue());rt.exclusive_write(paths[1],rt.json_bytes(manifest))
    rt.load_models(root,packaged=True,device='cpu')
    return manifest

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--package-existing',action='store_true')
    a=p.parse_args(argv)
    try:
        rt.require(sum((a.check_only,a.run,a.package_existing))==1,'choose one explicit action')
        if a.check_only:
            cfg,lock=rt.locked_config();print(json.dumps({'status':'ready','check_only':True,'frozen_file_count':len(lock['files']),
               'waveforms_loaded':False,'checkpoint_loaded':False,'cuda_initialized':False},indent=2));return 0
        if a.run:
            rt.require(all(not rt.safe_local_path(rel).exists() for rel in (rt.REPORT,rt.AUDIT,rt.MODEL,rt.MANIFEST)), 'output exists; no-clobber')
            report=diagnostic();audit=audit_report(report)
            rt.exclusive_write(rt.safe_local_path(rt.REPORT),rt.json_bytes(report))
            rt.exclusive_write(rt.safe_local_path(rt.AUDIT),rt.json_bytes(audit))
        manifest=package()
        print(json.dumps(manifest,ensure_ascii=False,indent=2));return 0
    except (ValueError,OSError,RuntimeError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'error','error':str(exc)},ensure_ascii=False));return 1
