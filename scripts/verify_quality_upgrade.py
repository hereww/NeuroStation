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
 'test_ssvep_four_conditions', 'test_ssvep_four_condition_presenter',
}

# Portable contracts are deliberately distinct from historical-evidence checks.
# Full mode still discovers everything, including the explicitly excluded probe.
CONTRACT_MODULES = frozenset({
 'test_learned_domain_dataset_v3', 'test_train_domain_gate_v3',
 'test_domain_gate_v3_evaluation', 'test_learned_denoise_v3',
 'test_learned_dataset_v4', 'test_evaluate_denoiser_v4',
 'test_train_denoiser_v4', 'test_learned_denoise_v4', 'test_run_learned_denoise_v4',
 'test_ssvep_v13', 'test_evaluate_ssvep_denoiser_v13',
 'test_evaluate_ssvep_denoiser_v14', 'test_ssvep_v7_experiment', 'test_ssvep_v8_experiment',
 'test_ssvep_v7_internal_eval', 'test_ssvep_v8_internal_eval',
 'test_analyze_ssvep_temporal_confirmation', 'test_quality_evidence_bindings',
 'test_quality_acceptance_runner', 'test_quality_research_fixtures',
 'test_ssvep_reference_v5', 'test_ssvep_denoise_v5', 'test_ssvep_denoise_v6',
})
CONTRACT_EXCLUSIONS = {
 'test_evaluate_ssvep_denoiser_v14.EvaluatorContractTests.test_personal_sessions_are_fixed_and_present':
     'requires original personal-session directories; retained in full mode, NOT portable acceptance',
}
SOURCE_SUFFIXES = ('.py', '.ps1', '.toml', '.spec', '.bat', '.cmd')
SOURCE_DIRECTORIES = ('configs/', 'apps/workstation_ui/locales/', 'tests/fixtures/')


def source_manifest(root, files):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in sorted(set(files)) if name and (root / name).is_file()
            and (name.endswith(SOURCE_SUFFIXES) or name.startswith(SOURCE_DIRECTORIES))}


def manifest_changes(before, after):
    return sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))


def test_cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from test_cases(item)
        else:
            yield item


def select_contract_tests(suite):
    selected, excluded = [], []
    for test in test_cases(suite):
        reason = CONTRACT_EXCLUSIONS.get(test.id().removeprefix('tests.'))
        if reason:
            excluded.append({'id': test.id(), 'reason': reason, 'status': 'not_run_in_contracts'})
        else:
            selected.append(test)
    return unittest.TestSuite(selected), excluded


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
    parser.add_argument('--mode',choices=('focused','full','ui','contracts'),default='focused')
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
    source=source_manifest(ROOT, files)
    identity=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
    (output/'source-manifest.json').write_text(json.dumps(source,indent=2,ensure_ascii=False),encoding='utf-8')
    loader=unittest.TestLoader()
    selection={'policy_version':1,'scope':args.mode,'excluded_tests':[], 'unselected_modules':[]}
    if args.mode=='ui':
        suite=loader.discover(str(ROOT/'apps/workstation_ui/tests'),top_level_dir=str(ROOT))
    else:
        sys.path.insert(0,str(ROOT/'tests'))
        paths=sorted((ROOT/'tests').glob('test*.py'))
        if args.mode=='focused':
            paths=[p for p in paths if p.stem in FOCUSED or p.stem.startswith(('test_quality','test_realtime'))]
        modules=sorted(CONTRACT_MODULES) if args.mode=='contracts' else [p.stem for p in paths]
        selection['selected_modules']=modules
        selection['unselected_modules']=sorted(p.stem for p in (ROOT/'tests').glob('test*.py') if p.stem not in modules)
        suite=loader.discover(str(ROOT/'tests')) if args.mode=='full' else unittest.TestSuite(loader.loadTestsFromName(name) for name in modules)
        if args.mode=='contracts':
            suite, selection['excluded_tests']=select_contract_tests(suite)
    loaded=list(collect_ids(suite));selection["selected_test_ids"]=loaded;began=time.monotonic()
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
    current=source_manifest(ROOT, git('ls-files','--cached','--others','--exclude-standard','-z').split('\0'))
    changed=manifest_changes(source,current)
    checks['source_unchanged_during_run']=not changed
    packages={}
    for package in ('brainflow','numpy','scipy','PySide6','torch'):
        try:packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:packages[package]=None
    report={'schema_version':2,'mode':args.mode,'finished_at':datetime.now(timezone.utc).isoformat(),
            'git_head':git('rev-parse','HEAD').strip(),'git_status':git('status','--short'),'source_identity':identity,
            'source_manifest_scope':{'suffixes':SOURCE_SUFFIXES,'directories':SOURCE_DIRECTORIES},
            'selection':selection,
            'count_semantics':'success callbacks and failure/error events; subtests and class setup may overlap method counts',
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
            'successful':result.wasSuccessful() and all(checks.values()),
            'fully_passed_selection':result.wasSuccessful() and all(checks.values()) and not result.skipped
                and not result.expectedFailures and not (set(loaded)-set(result.started))}
    (output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('issues','skips','passed_ids','not_started_ids','selection','git_status')},ensure_ascii=False,indent=2))
    return 0 if report['successful'] else 1

if __name__=='__main__':raise SystemExit(main())
