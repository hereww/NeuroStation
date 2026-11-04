"""Verify the ACTUAL saved v9 package/examples and metadata-only preflight.

No mocks, no retraining, no additional participants. Does not edit frozen code.
This verifier is separately hashed in its evidence JSON after package freezing.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).absolute().parents[1]))
import numpy as np
from eeg_tools import ssvep_test_runtime as rt
from eeg_tools.ssvep_v9_delivery import audit_report

def main():
    cfg,lock,manifest=rt.preflight()
    saved=rt.json_read(rt.read_bytes(rt.safe_local_path(rt.REPORT)))
    audit=audit_report(saved)
    rt.require(audit['audit_passed'] and saved['case_count']==4200,'saved diagnostic audit')
    ev=rt.safe_local_path('analysis_outputs/ssvep_v9_examples_20261008')
    example_manifest=rt.json_read(rt.read_bytes(ev/'manifest.json'))
    rt.require(example_manifest['D_drive_accessed'] is False and example_manifest['holdout_waveforms_loaded'] is False,'example scope')
    for name,digest in example_manifest['files'].items():rt.require(rt.sha(rt.read_bytes(ev/name))==digest,'example changed')
    checks=[]
    for domain,label in [('Wearable','wearable'),('Nakanishi2015','nakanishi')]:
        inp=ev/(domain+'_input.npy');x=np.load(inp,allow_pickle=False)
        results={}
        for device in ('cpu','gpu'):
            folder=rt.safe_local_path(f'analysis_outputs/ssvep_v9_{label}_{device}_demo_20261008')
            report=rt.json_read(rt.read_bytes(folder/'report.json'))
            for name,digest in report['output_sha256'].items():rt.require(rt.sha(rt.read_bytes(folder/name))==digest,'output changed')
            rt.require(report['input_sha256']==rt.sha(rt.read_bytes(inp)),'raw source mutated')
            rt.require(report['inference_executed'] is True and report['hardware_opened'] is False and report['production_validated'] is False,'inference safety flags')
            rt.require(report['processing']['processed_samples']==1536 and report['processing']['preserved_samples']==543,'bad skip/tail counts')
            y=np.load(folder/'denoised.npy',allow_pickle=False);mask=np.load(folder/'processed_mask.npy',allow_pickle=False)
            rt.require(mask.dtype==np.bool_ and mask.shape==(len(x),) and y.shape==x.shape,'output shape')
            np.testing.assert_array_equal(y[~mask],x[~mask])
            changes=[];protected=[]
            for start in range(0,len(x)-255,256):
                if not mask[start:start+256].all():continue
                before=x[start:start+256].T;after=y[start:start+256].T
                rms=np.sqrt(np.mean((before-before.mean(axis=1,keepdims=True))**2,axis=1,keepdims=True)+1e-8)
                delta=(after-before)/rms
                change=float(np.mean(delta**2));rt.require(change<=.00045*1.0001,'residual budget')
                f=np.fft.rfftfreq(256,1/rt.RATES[domain]);band=(f>=8)&(f<=40)
                native=np.fft.rfft(before-before.mean(axis=1,keepdims=True))
                ratio=float(np.sum(np.abs(np.fft.rfft(after-before)[...,band])**2)/(np.sum(np.abs(native[...,band])**2)+1e-30))
                rt.require(ratio<=1e-6,'fixed-window protected energy leak')
                changes.append(change);protected.append(ratio)
            checks.append({'domain':domain,'device':device,'passed':True,'processed_samples':int(mask.sum()),'preserved_samples':int((~mask).sum()),
                'max_normalized_window_change':max(changes),'max_fixed_window_protected_complex_energy_ratio':max(protected),
                'input_sha256':rt.sha(rt.read_bytes(inp)),'report_sha256':rt.sha(rt.read_bytes(folder/'report.json'))})
            results[device]=y
        # Normalize per original window/channel; no fictitious microvolt unit assumption.
        cpu,gpu=results['cpu'],results['gpu'];max_difference=0.
        for start in range(0,len(x)-255,256):
            raw=x[start:start+256]
            rms=np.sqrt(np.var(raw,axis=0,keepdims=True)+1e-8)
            max_difference=max(max_difference,float(np.max(np.abs(cpu[start:start+256]-gpu[start:start+256])/rms)))
        rt.require(max_difference<1e-5,'CPU/GPU normalized disagreement')
        checks.append({'domain':domain,'comparison':'cpu_vs_cuda','passed':True,'max_abs_difference_noisy_rms_units':max_difference})
    # Deny torch import: proving --check-only does not deserialize weights/init CUDA.
    guarded="""import builtins,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
original=builtins.__import__
def guarded(name,*a,**k):
 if name=='torch' or name.startswith('torch.'):raise RuntimeError('unexpected torch import')
 return original(name,*a,**k)
builtins.__import__=guarded
import runpy
sys.argv=['scripts/run_ssvep_test_model.py','--check-only']
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    check=subprocess.run([sys.executable,'-c',guarded],cwd=rt.ROOT,capture_output=True,text=True)
    rt.require(check.returncode==0,'preflight imported torch or failed: '+check.stderr)
    checks.append({'comparison':'check_only_with_torch_import_forbidden','passed':True})
    rt.verify_files(lock['files'])  # v8 historical hashes still intact
    proof={'schema_version':1,'status':'formal','created_at_utc':rt.now(),'verification_passed':True,
           'checks':checks,'case_count':saved['case_count'],'all_internal_gates_passed':saved['all_internal_gates_passed'],
           'saved_report_sha256':rt.sha(rt.read_bytes(rt.safe_local_path(rt.REPORT))),
           'model_sha256':manifest['files'][rt.MODEL], 'frozen_bindings_verified':len(lock['files']),
           'verifier_sha256':rt.sha(rt.read_bytes(Path(__file__).absolute())),
           'runtime_enabled':False,'production_validated':False,'independent_validation':False,
           'weights_retrained_this_version':False,'D_drive_accessed_this_verification':False,
           'holdout_waveforms_loaded_this_verification':False}
    out=rt.safe_local_path('analysis_outputs/ssvep_v9_end_to_end_verification_20261008.json')
    rt.exclusive_write(out,rt.json_bytes(proof));print(json.dumps(proof,ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
