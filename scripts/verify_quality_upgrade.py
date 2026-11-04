"""Reproducible software-only acceptance runs, with failures and skips kept explicit.

Example: python scripts/verify_quality_upgrade.py --mode focused --output quality-output/acceptance-focused
No hardware capture, stimulus or training is started by this runner.
"""
from __future__ import annotations
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
FOCUSED={
 'test_acquisition_worker','test_config','test_denoise','test_offline_analysis',
 'test_session_files','test_formal_acquisition_defaults','test_formal_ssvep_protocol',
 'test_release_config_policy','test_release_metadata','test_workstation_cli',
 'test_workstation_ssvep','test_desktop_gateway','test_device_discovery',
}

class RecordedResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.passed=[];self.started=[]
    def startTest(self,test):
        self.started.append(test.id());super().startTest(test)
    def addSuccess(self,test):
        self.passed.append(test.id());super().addSuccess(test)


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT).decode('utf-8',errors='replace')


def collect_ids(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):yield from collect_ids(item)
        else:yield item.id()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('focused','full','ui'),default='focused')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    os.environ.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',QT_QPA_PLATFORM='offscreen',
                      NEUROSTATION_DATASETS=str(output/'datasets'),
                      NEUROSTATION_SETTINGS=str(output/'settings'),
                      NEUROSTATION_DIAGNOSTICS=str(output/'diagnostics'))
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(ROOT));os.chdir(ROOT)
    files=git('ls-files','--cached','--others','--exclude-standard','-z').split('\0')
    source={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files if name and
            (ROOT/name).is_file() and ((name.endswith(('.py','.ps1','.toml')) or name.startswith('configs/')))}
    identity=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
    (output/'source-manifest.json').write_text(json.dumps(source,indent=2,ensure_ascii=False),encoding='utf-8')
    loader=unittest.TestLoader()
    if args.mode=='ui':
        suite=loader.discover(str(ROOT/'apps/workstation_ui/tests'),top_level_dir=str(ROOT))
    else:
        sys.path.insert(0,str(ROOT/'tests'))
        paths=sorted((ROOT/'tests').glob('test*.py'))
        if args.mode=='focused':
            paths=[p for p in paths if p.stem in FOCUSED or p.stem.startswith(('test_quality','test_realtime'))]
        suite=loader.discover(str(ROOT/'tests')) if args.mode=='full' else unittest.TestSuite(loader.loadTestsFromName(p.stem) for p in paths)
    loaded=list(collect_ids(suite));began=time.monotonic()
    with (output/'tests.log').open('w',encoding='utf-8') as log:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=RecordedResult).run(suite)
    elapsed=time.monotonic()-began
    issues=[{'kind':kind,'id':test.id(),'test':str(test),'detail':text}
            for kind,rows in (('failure',result.failures),('error',result.errors)) for test,text in rows]
    checks={}
    with (output/'formal-config.log').open('w',encoding='utf-8') as log:
        config=subprocess.run([sys.executable,'-B','scripts/validate_release_configs.py'],cwd=ROOT,
                              stdout=log,stderr=subprocess.STDOUT,timeout=60)
    checks['formal_configuration']=config.returncode==0
    syntax=[]
    for name in source:
        if name.endswith('.py'):
            try:compile((ROOT/name).read_bytes(),name,'exec')
            except (SyntaxError,ValueError) as exc:syntax.append({'path':name,'error':str(exc)})
    checks['syntax']=not syntax
    changed=[name for name,digest in source.items() if not (ROOT/name).is_file() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    checks['source_unchanged_during_run']=not changed
    packages={}
    for package in ('brainflow','numpy','scipy','PySide6','torch'):
        try:packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:packages[package]=None
    report={'schema_version':1,'mode':args.mode,'finished_at':datetime.now(timezone.utc).isoformat(),
            'git_head':git('rev-parse','HEAD').strip(),'source_identity':identity,
            'python':sys.executable,'python_version':sys.version,'packages':packages,
            'loaded_tests':len(loaded),'started_tests':result.testsRun,'passed_tests':len(result.passed),
            'failure_events':len(result.failures),'error_events':len(result.errors),
            'skipped_tests':len(result.skipped),'expected_failures':len(result.expectedFailures),
            'unexpected_successes':len(result.unexpectedSuccesses),
            'not_started_ids':sorted(set(loaded)-set(result.started)),
            'elapsed_s':elapsed,'issues':issues,
            'skips':[{'id':test.id(),'reason':reason} for test,reason in result.skipped],
            'checks':checks,'syntax_errors':syntax,'changed_source_files':changed,
            'passed_ids':result.passed,'hardware_validation':'not_run','physical_stimulus':'not_run',
            'successful':result.wasSuccessful() and all(checks.values())}
    (output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('issues','skips','passed_ids','not_started_ids')},ensure_ascii=False,indent=2))
    return 0 if report['successful'] else 1

if __name__=='__main__':raise SystemExit(main())
