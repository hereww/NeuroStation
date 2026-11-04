"""Create a NEW local runtime binding. Never rewrite training manifests or enable models."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eeg_tools.runtime_bindings import RootResolver,sha256


def inspect_python(path,packages):
    path=path.expanduser().resolve(strict=True)
    probe='import sys,json,importlib.metadata as m; print(json.dumps({"executable":sys.executable,"version":sys.version,"packages":{p:m.version(p) for p in sys.argv[1:]}}))'
    result=subprocess.run([str(path),'-B','-c',probe,*packages],capture_output=True,text=True,encoding='utf-8',
                          env={**os.environ,'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'},timeout=60,check=True)
    return {'path':str(path),'sha256':sha256(path)},json.loads(result.stdout)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-python',type=Path,required=True)
    p.add_argument('--model-python',type=Path)
    p.add_argument('--model-run',type=Path)
    p.add_argument('--root-mappings',type=Path,help='JSON array of exact absolute from/to roots')
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True,help='New binding file; existing files are refused')
    a=p.parse_args(argv)
    if bool(a.model_python)!=bool(a.model_run):p.error('--model-python and --model-run must be supplied together')
    if a.root_mappings and not a.model_run:p.error('root mappings require an explicit model run')
    if not a.output_root.is_absolute() or not a.config.is_absolute():p.error('output/config paths must be absolute')
    if a.config.exists():p.error('binding already exists; choose a new file and review differences')
    capture,environment=inspect_python(a.capture_python,['brainflow','numpy','scipy','PySide6'])
    mappings=json.loads(a.root_mappings.read_text(encoding='utf-8-sig')) if a.root_mappings else []
    RootResolver(mappings)
    config={'schema_version':1,'files':{'capture_python':capture},'root_mappings':mappings,
            'output_root':str(a.output_root.resolve()),'environment_observations':{'capture':environment}}
    if a.model_run:
        run=a.model_run.expanduser().resolve(strict=True)
        if not run.is_dir():p.error('model run must be a directory')
        model,environment=inspect_python(a.model_python,['torch','numpy','scipy'])
        config['files']['model_python']=model;config['model_run']=str(run)
        config['environment_observations']['model']=environment
    a.config.parent.mkdir(parents=True,exist_ok=True)
    with a.config.open('x',encoding='utf-8') as f:json.dump(config,f,ensure_ascii=False,indent=2)
    print(json.dumps({'config':str(a.config),'model_loaded':False,'experimental_model_enabled':False,
                      'note':'Runtime metadata only. Model artifacts are independently checked at explicit load.'},ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
