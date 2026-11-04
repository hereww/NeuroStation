"""Offline, read-only session reporting with explicit denominators and provenance."""
from pathlib import Path
import csv
from collections import Counter
import json
import uuid
from eeg_tools.runtime_bindings import sha256
from eeg_tools.quality.statistics import build_trial_ledger,summarize_trials,ratio
from eeg_tools.streaming import recover_blocks
from neurostation_quality_report import render_report_html


def _signal_updates(session, cancelled=None):
    """Summarize recorded updates, not trials or a fresh signal-quality verdict."""
    path = session/'raw_blocks/quality-v2.jsonl'
    if not path.is_file():
        return {'status':'unavailable', 'all':0, 'seal_evidence_match':None}
    counts = Counter(pass_=0, fail=0, unknown=0, unassessable=0)
    channels, reasons = Counter(), Counter()
    total = 0
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if cancelled and cancelled():raise InterruptedError('quality.cancelled')
            if not line.strip():continue
            total += 1
            try:
                row = json.loads(line)
                if not isinstance(row,dict) or row.get('schema_version') != 2:
                    raise ValueError('unknown quality schema')
                state = row.get('dimensions',{}).get('signal_quality')
                if state not in ('pass','fail','unknown'):raise ValueError('unknown disposition')
                channel_names = {r['channel'] for r in row.get('channels',[]) if r.get('quality_status')=='fail'}
                why = set(row.get('reasons',[])) | {r['rule'] for r in row.get('artifacts',{}).get('intervals',[])}
                counts['pass_' if state=='pass' else state] += 1
                channels.update(channel_names);reasons.update(why)
            except (ValueError,TypeError,KeyError,AttributeError):
                counts['unassessable'] += 1
    seal_match = None
    seal_path = session/'raw_blocks/seal.json'
    digest = sha256(path)
    if seal_path.is_file():
        try:
            seal = json.loads(seal_path.read_text(encoding='utf-8'))
            expected = seal.get('quality_evidence',{}).get('quality-v2.jsonl')
            seal_match = expected == digest if expected else None
        except (ValueError,AttributeError):
            seal_match = False
    return {'status':'observed', 'all':total,
            'counts':{'pass':counts['pass_'],'fail':counts['fail'],'unknown':counts['unknown'],'unassessable':counts['unassessable']},
            'abnormal_channel_updates':dict(channels), 'reason_updates':dict(reasons),
            'seal_evidence_match':seal_match, 'journal_sha256':digest,
            'denominator':'recorded quality update lines, including malformed records; not trials, duration or accuracy'}


def build_session_report(session_dir, output_dir=None, *, cancelled=None):
    if cancelled and cancelled():raise InterruptedError("quality.cancelled")
    session=Path(session_dir).resolve()
    output=Path(output_dir) if output_dir else session/'derived'/('quality-report-v2-'+uuid.uuid4().hex[:12])
    if output.resolve()==session:raise ValueError('report must not replace source session')
    output.mkdir(parents=True,exist_ok=False)
    inputs={p.name:sha256(p) for p in session.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.ndjson','.tsv','.csv')}
    for name in ('raw_blocks/blocks.jsonl', 'raw_blocks/seal.json', 'raw_blocks/quality-v2.jsonl'):
        if (session/name).is_file():inputs[name]=sha256(session/name)
    meta={}
    if (session/'session.json').is_file():meta=json.loads((session/'session.json').read_text(encoding='utf-8'))
    events=[]
    if (session/'events.tsv').is_file():
        with (session/'events.tsv').open(encoding='utf-8',newline='') as f:events=list(csv.DictReader(f,delimiter='\t'))
    # A standalone continuous stream has no implicit trial denominator.
    events=[r for r in events if str(r.get('trial_index','')) not in ('','-1','None')]
    recovered=recover_blocks(session/'raw_blocks', cancelled=cancelled) if (session/'raw_blocks/blocks.jsonl').is_file() else None
    count=recovered['verified_samples'] if recovered else int(meta.get('sample_count') or meta.get('recorded_samples_per_channel') or 0)
    ledger=build_trial_ledger(events,[],count,planned_trial_count=meta.get('trial_count'))
    commands=[]
    for p in sorted({*session.glob('*command*.jsonl'), *session.glob('*command*.ndjson')}):
        with p.open(encoding='utf-8') as f:
            for line in f:
                if not line.strip():continue
                try:
                    row=json.loads(line)
                    if not isinstance(row,dict):raise ValueError('not a command object')
                    if row.get('type')=='command':commands.append(row)
                except ValueError:commands.append({'accepted':False,'reason':'unreadable_command_record'})
    if cancelled and cancelled():raise InterruptedError('quality.cancelled')
    signal_updates=_signal_updates(session, cancelled)
    accepted=sum(r.get('accepted') is True for r in commands)
    report={'schema_version':2,'session_id':meta.get('session_id',session.name),'source_hashes':inputs,
            'source_session':str(session),
            'data_origin':meta.get('data_origin', 'legacy_simulated' if meta.get('simulated') is True else 'not_declared'),
            'source_unchanged':all((session/name).is_file() and sha256(session/name)==digest for name,digest in inputs.items()),
            'recording_status':meta.get('recording_status',meta.get('status','unknown')),
            'raw_blocks':{k:v for k,v in recovered.items() if k!='records'} if recovered else {'status':'legacy_no_block_journal'},
            'trials':summarize_trials(ledger),'trial_ledger':ledger,
            'planned_trial_count':meta.get('trial_count') if type(meta.get('trial_count')) is int else None,
            'signal_quality_updates':signal_updates,
            'commands':{'all':len(commands),'accepted':accepted,'rejected_or_unassessable':len(commands)-accepted,
                        'acceptance':ratio(accepted,len(commands),'emitted command records, including UNKNOWN; not accuracy')},
            'limitations':['This report does not rescore EEG or infer missing labels.',
                           'No trial scoring supplied: scoring coverage remains unknown; use derived analysis_* statistics_v2 for scored evidence.',
                           'Software tests are not hardware, signal or personal-model validation.'],
            'hardware_verification':'not_established_by_this_report','personal_model_validated':False}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'report.html').write_text(render_report_html(report),encoding='utf-8')
    return {'output_dir':str(output),'report':report}
