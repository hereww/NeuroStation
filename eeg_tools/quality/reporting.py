"""Offline, read-only session reporting with explicit denominators and provenance."""
from pathlib import Path
import csv
import html
import json
import uuid
from eeg_tools.runtime_bindings import sha256
from eeg_tools.quality.statistics import build_trial_ledger,summarize_trials,ratio
from eeg_tools.streaming import recover_blocks


def build_session_report(session_dir, output_dir=None):
    session=Path(session_dir).resolve()
    output=Path(output_dir) if output_dir else session/'derived'/('quality-report-v2-'+uuid.uuid4().hex[:12])
    if output.resolve()==session:raise ValueError('report must not replace source session')
    output.mkdir(parents=True,exist_ok=False)
    inputs={p.name:sha256(p) for p in session.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.tsv','.csv')}
    meta={}
    if (session/'session.json').is_file():meta=json.loads((session/'session.json').read_text(encoding='utf-8'))
    events=[]
    if (session/'events.tsv').is_file():
        with (session/'events.tsv').open(encoding='utf-8',newline='') as f:events=list(csv.DictReader(f,delimiter='\t'))
    # A standalone continuous stream has no implicit trial denominator.
    events=[r for r in events if str(r.get('trial_index','')) not in ('','-1','None')]
    recovered=recover_blocks(session/'raw_blocks') if (session/'raw_blocks/blocks.jsonl').is_file() else None
    count=recovered['verified_samples'] if recovered else int(meta.get('sample_count') or meta.get('recorded_samples_per_channel') or 0)
    ledger=build_trial_ledger(events,[],count)
    commands=[]
    for p in session.glob('*command*.jsonl'):
        with p.open(encoding='utf-8') as f:
            for line in f:
                try:
                    row=json.loads(line)
                    if row.get('type')=='command':commands.append(row)
                except ValueError:commands.append({'accepted':False,'reason':'unreadable_command_record'})
    accepted=sum(r.get('accepted') is True for r in commands)
    report={'schema_version':2,'session_id':meta.get('session_id',session.name),'source_hashes':inputs,
            'recording_status':meta.get('recording_status',meta.get('status','unknown')),
            'raw_blocks':{k:v for k,v in recovered.items() if k!='records'} if recovered else {'status':'legacy_no_block_journal'},
            'trials':summarize_trials(ledger),'trial_ledger':ledger,
            'commands':{'all':len(commands),'accepted':accepted,'rejected_or_unassessable':len(commands)-accepted,
                        'acceptance':ratio(accepted,len(commands),'emitted command records, including UNKNOWN; not accuracy')},
            'limitations':['This report does not rescore EEG or infer missing labels.',
                           'No trial scoring supplied: scoring coverage remains unknown; use derived analysis_* statistics_v2 for scored evidence.',
                           'Software tests are not hardware, signal or personal-model validation.'],
            'hardware_verification':'not_established_by_this_report','personal_model_validated':False}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    text=html.escape(json.dumps(report,ensure_ascii=False,indent=2))
    (output/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>NeuroStation 支线会话质量报告</title><style>body{font:15px system-ui;margin:40px;max-width:1200px;background:#101a29;color:#e2ebf6}pre{white-space:pre-wrap;overflow-wrap:anywhere}h1{color:#72dfc3}</style><h1>NeuroStation 会话质量报告 v2</h1><p>每个比例附分母。未知不等于合格；软件状态不代表实机或个人模型验收。</p><pre>'+text+'</pre></html>',encoding='utf-8')
    return {'output_dir':str(output),'report':report}
