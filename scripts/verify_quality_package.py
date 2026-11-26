"""Bounded packaged startup checks; no device open, stimulus, or model loading."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--entry-name',default='NeuroStationQualityUpgrade.exe')
    a=p.parse_args();artifact=a.artifact.resolve();output=a.output.resolve()
    exe=artifact/a.entry_name;output.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'QT_QPA_PLATFORM':'offscreen','NEUROSTATION_DATASETS':str(output/'smoke-data'),
         'NEUROSTATION_SETTINGS':str(output/'smoke-settings'),'NEUROSTATION_DIAGNOSTICS':str(output/'smoke-diagnostics'),
         'PATH':os.environ['SystemRoot']+'/System32;'+os.environ['SystemRoot']}
    env.pop('NEUROSTATION_RUNTIME_CONFIG',None);env.pop('PYTHONPATH',None);env.pop('NEUROSTATION_QUALITY_ROOT',None)
    checks=[]
    for name,args in [('workstation',['--smoke-test']),('eeg_processing',['--processing-page','--smoke-test']),
                      ('realtime',['--realtime-commands','--smoke-test']),
                      ('capture_help',['--realtime-capture-worker','--help']),('main_worker_help',['--acquisition-worker','--help']),
                      ('diagnostics',['--diagnostics']),
                      ('quality_workflow',['--quality-workflow-smoke',str(output/'quality-workflow')])]:
        try:
            run=subprocess.run([str(exe),*args],cwd=artifact,env=env,capture_output=True,timeout=120 if name=='quality_workflow' else 45)
            (output/(name+'.stdout.txt')).write_bytes(run.stdout);(output/(name+'.stderr.txt')).write_bytes(run.stderr)
            checks.append({'check':name,'arguments':args,'exit_code':run.returncode,'passed':run.returncode==0})
        except subprocess.TimeoutExpired:
            checks.append({'check':name,'arguments':args,'passed':False,'error':'owned subprocess exceeded its startup/workflow deadline'})
    report={'schema_version':1,'artifact':str(artifact),'checks':checks,'all_passed':all(c['passed'] for c in checks),
            'hardware_started':False,'stimulus_started':False,'model_included':False,
            'ssvep_model_included': (artifact / '_internal/assets/ssvep_models/cca-v3.onnx').is_file(), 'torch_included':False,
            'fresh_windows_machine_validation':'not_run','smoke_path':env['PATH']}
    (output/'build-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['all_passed'] else 2

if __name__=='__main__':raise SystemExit(main())
