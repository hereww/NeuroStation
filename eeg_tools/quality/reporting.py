"""Offline, read-only session reporting with explicit denominators and provenance."""
from pathlib import Path
import csv
from collections import Counter
import io
import json
import uuid

from eeg_tools.quality.evidence import (capture_snapshot, read_snapshot, snapshot_lines,
                                        snapshot_matches, check_cancelled, publish_bundle)
from eeg_tools.quality.statistics import build_trial_ledger, summarize_trials, ratio
from eeg_tools.streaming import recover_blocks
from neurostation_quality_report import render_report_html


def _input_names(session):
    names = {p.name for p in session.iterdir() if p.is_file()
             and p.suffix in ('.json', '.jsonl', '.ndjson', '.tsv', '.csv')}
    for folder in ('raw_blocks', 'signal_processing'):
        directory = session / folder
        if directory.is_dir():
            names.update(folder + '/' + p.name for p in directory.iterdir() if p.is_file())
    return names


def _signal_updates(session, cancelled=None, snapshot=None):
    """Summarize recorded updates, not trials or a fresh signal-quality verdict."""
    snapshot = snapshot if snapshot is not None else capture_snapshot(session, _input_names(session), cancelled=cancelled)
    folder = 'signal_processing' if 'signal_processing/quality-v2.jsonl' in snapshot else 'raw_blocks'
    name = folder + '/quality-v2.jsonl'
    if name not in snapshot:
        return {'status': 'unavailable', 'all': 0, 'seal_evidence_match': None}
    counts = Counter(pass_=0, fail=0, unknown=0, unassessable=0)
    channels, reasons = Counter(), Counter()
    total = 0
    for line in snapshot_lines(session, name, snapshot, cancelled=cancelled):
        if not line.strip():
            continue
        total += 1
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or row.get('schema_version') != 2:
                raise ValueError('unknown quality schema')
            state = row.get('dimensions', {}).get('signal_quality')
            if state not in ('pass', 'fail', 'unknown'):
                raise ValueError('unknown disposition')
            channel_names = {r['channel'] for r in row.get('channels', []) if r.get('quality_status') == 'fail'}
            why = set(row.get('reasons', [])) | {r['rule'] for r in row.get('artifacts', {}).get('intervals', [])}
            counts['pass_' if state == 'pass' else state] += 1
            channels.update(channel_names)
            reasons.update(why)
        except (ValueError, TypeError, KeyError, AttributeError):
            counts['unassessable'] += 1
    seal_match = None
    if folder + '/seal.json' in snapshot:
        try:
            seal = json.loads(read_snapshot(session, folder + '/seal.json', snapshot))
            expected = seal.get('quality_evidence', {}).get('quality-v2.jsonl')
            seal_match = expected == snapshot[name] if expected else None
        except (ValueError, AttributeError):
            seal_match = False
    return {'status': 'observed', 'all': total,
            'counts': {'pass': counts['pass_'], 'fail': counts['fail'], 'unknown': counts['unknown'], 'unassessable': counts['unassessable']},
            'abnormal_channel_updates': dict(channels), 'reason_updates': dict(reasons),
            'seal_evidence_match': seal_match, 'journal_sha256': snapshot[name],
            'denominator': 'recorded quality update lines, including malformed records; not trials, duration or accuracy'}


def build_session_report(session_dir, output_dir=None, *, cancelled=None):
    check_cancelled(cancelled)
    session = Path(session_dir).resolve()
    output = (Path(output_dir) if output_dir else session / 'derived' / ('quality-report-v2-' + uuid.uuid4().hex[:12])).resolve()
    if output == session or any(output.is_relative_to(session / name) for name in ('raw_blocks', 'signal_processing')):
        raise ValueError('report must not replace source session or raw blocks')
    if output.exists():
        raise FileExistsError(str(output))
    inputs = capture_snapshot(session, _input_names(session), cancelled=cancelled)
    meta = json.loads(read_snapshot(session, 'session.json', inputs)) if 'session.json' in inputs else {}
    events = []
    if 'events.tsv' in inputs:
        events = list(csv.DictReader(io.StringIO(read_snapshot(session, 'events.tsv', inputs).decode('utf-8')), delimiter='\t'))
    # A standalone continuous stream has no implicit trial denominator.
    events = [r for r in events if str(r.get('trial_index', '')) not in ('', '-1', 'None')]
    recovered = recover_blocks(session / 'raw_blocks', cancelled=cancelled) if 'raw_blocks/blocks.jsonl' in inputs else None
    count = recovered['verified_samples'] if recovered else int(meta.get('sample_count') or meta.get('recorded_samples_per_channel') or 0)
    ledger = build_trial_ledger(events, [], count, planned_trial_count=meta.get('trial_count'))
    commands = []
    command_names = sorted(name for name in inputs if '/' not in name and 'command' in name
                           and Path(name).suffix in ('.jsonl', '.ndjson'))
    for name in command_names:
        for line in snapshot_lines(session, name, inputs, cancelled=cancelled):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('not a command object')
                if row.get('type') == 'command':
                    commands.append(row)
            except ValueError:
                commands.append({'accepted': False, 'reason': 'unreadable_command_record'})
    check_cancelled(cancelled)
    signal_updates = _signal_updates(session, cancelled, inputs)
    accepted = sum(r.get('accepted') is True for r in commands)
    report = {'schema_version': 2, 'session_id': meta.get('session_id', session.name), 'source_hashes': inputs,
              'source_session': str(session), 'source_inventory_version': 1,
              'data_origin': meta.get('data_origin', 'legacy_simulated' if meta.get('simulated') is True else 'not_declared'),
              'source_unchanged': True,  # Publication below is conditional on a full inventory/content recheck.
              'recording_status': meta.get('recording_status', meta.get('status', 'unknown')),
              'raw_blocks': {k: v for k, v in recovered.items() if k != 'records'} if recovered else {'status': 'not_requested' if meta.get('save_raw_eeg') is False else 'legacy_no_block_journal'},
              'trials': summarize_trials(ledger), 'trial_ledger': ledger,
              'planned_trial_count': meta.get('trial_count') if type(meta.get('trial_count')) is int else None,
              'signal_quality_updates': signal_updates,
              'commands': {'all': len(commands), 'accepted': accepted, 'rejected_or_unassessable': len(commands) - accepted,
                           'acceptance': ratio(accepted, len(commands), 'emitted command records, including UNKNOWN; not accuracy')},
              'limitations': ['This report does not rescore EEG or infer missing labels.',
                              'Source hashes cover session metadata and raw-block files; they are not an external authenticity signature.',
                              'Read-only content checks are not a filesystem-wide atomic snapshot. Use quiescent source files.',
                              'No trial scoring supplied: scoring coverage remains unknown; use derived analysis_* statistics_v2 for scored evidence.',
                              'Software tests are not hardware, signal or personal-model validation.'],
              'hardware_verification': 'not_established_by_this_report', 'personal_model_validated': False}

    def verify_source():
        if not snapshot_matches(session, inputs, names=lambda: _input_names(session), cancelled=cancelled):
            raise ValueError('quality.error.source_changed')

    check_cancelled(cancelled)
    publish_bundle(output, {'report.json': json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'),
                            'report.html': render_report_html(report).encode('utf-8')},
                   cancelled=cancelled, validate=verify_source)
    return {'output_dir': str(output), 'report': report}
