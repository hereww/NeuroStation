"""Verified block replay; numerical agreement alone is not sealed evidence."""
from contextlib import nullcontext
import hashlib
import io
from pathlib import Path
import json
import re

import numpy as np
from eeg_tools.quality.engine import QualityEngine, safe_json
from eeg_tools.quality.evidence import capture_snapshot, snapshot_matches, check_cancelled, read_snapshot
from eeg_tools.streaming import recover_blocks


STABLE_FIELDS = ('schema_version', 'processing_id', 'stream_id', 'generation', 'start', 'stop', 'dimensions',
                 'reasons', 'channels', 'artifacts', 'display_sha256', 'initial_window_seconds')


def _inputs(directory):
    return {p.name for p in directory.iterdir() if p.is_file()}


def _sealed_evidence(directory, snapshot, recovery):
    issues = []
    try:
        seal = json.loads(read_snapshot(directory, 'seal.json', snapshot))
        if not isinstance(seal, dict) or seal.get('schema_version') != 2:
            raise ValueError('invalid_seal_schema')
        expected = seal.get('quality_evidence')
        if seal.get('quality_complete') is not True or not isinstance(expected, dict) or not expected:
            raise ValueError('quality_not_sealed')
        last_hash = recovery['records'][-1]['record_hash'] if recovery['records'] else '0' * 64
        if (seal.get('last_hash') != last_hash or not isinstance(seal.get('watermarks'), dict)
                or any(seal['watermarks'].get(key) != recovery['verified_samples']
                       for key in ('received', 'written', 'durable'))):
            issues.append('seal_raw_binding_mismatch')
        if recovery['uncommitted_files'] or recovery['temporary_files']:
            issues.append('uncommitted_raw_tail')
        if 'quality-v2.jsonl' not in expected:
            issues.append('quality_journal_not_sealed')
        for name, digest in expected.items():
            if (not re.fullmatch(r'(?:quality-[\w-]+\.jsonl?|processing-[0-9a-f]{64}\.json)', name)
                    or not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)):
                issues.append('invalid_quality_evidence_entry')
            elif snapshot.get(name) != digest:
                issues.append('sealed_content_mismatch:' + name)
        for name in snapshot:
            if name.startswith('processing-') and name not in expected:
                issues.append('processing_manifest_not_sealed:' + name)
    except (ValueError, TypeError, AttributeError) as exc:
        issues.append('unverified_seal:' + str(exc))
    return issues


def replay_quality(directory, *, on_result=None, cancelled=None):
    directory = Path(directory)
    check_cancelled(cancelled)
    snapshot = capture_snapshot(directory, _inputs(directory), cancelled=cancelled)
    recovery = recover_blocks(directory, cancelled=cancelled)
    if recovery['issues']:
        raise ValueError('block integrity failure: ' + repr(recovery['issues']))
    evidence_issues = _sealed_evidence(directory, snapshot, recovery)
    engine = None
    mismatches, manifests = [], set()
    observed_count = 0
    reference_digest = hashlib.sha256()
    reference = directory / 'quality-v2.jsonl'
    source = reference.open('rb') if 'quality-v2.jsonl' in snapshot else nullcontext(None)
    with source as handle:
        for index, record in enumerate(recovery['records']):
            check_cancelled(cancelled)
            if engine is None:
                engine = QualityEngine(record['channel_names'], int(record['sample_rate_hz']))
            # Verify exactly the bytes about to be processed, against the journal.
            data = (directory / record['file']).read_bytes()
            if hashlib.sha256(data).hexdigest() != record['sha256']:
                raise ValueError('block integrity failure at consumption: ' + record['file'])
            raw = np.load(io.BytesIO(data), allow_pickle=False)
            result = safe_json(engine.process(raw[record['eeg_rows']], record))
            stable = {key: result.get(key) for key in STABLE_FIELDS}
            if engine.processing_id not in manifests:
                name = 'processing-' + engine.processing_id + '.json'
                try:
                    manifest = json.loads(read_snapshot(directory, name, snapshot))
                    if manifest != json.loads(json.dumps(safe_json(engine.manifest()))):
                        evidence_issues.append('processing_manifest_mismatch:' + name)
                except (ValueError, TypeError):
                    evidence_issues.append('processing_manifest_unavailable:' + name)
                manifests.add(engine.processing_id)
            line = handle.readline() if handle else b''
            if not line:
                mismatches.append({'block': index, 'reason': 'missing_online_quality'})
            else:
                reference_digest.update(line)
                observed_count += 1
                try:
                    observed = json.loads(line)
                    if not isinstance(observed, dict):
                        raise ValueError('not a quality object')
                    fields = [key for key in STABLE_FIELDS if stable[key] != observed.get(key)]
                    if fields:
                        mismatches.append({'block': index, 'fields': fields})
                except (ValueError, UnicodeError):
                    mismatches.append({'block': index, 'reason': 'invalid_online_quality'})
            if on_result is not None:
                on_result(result, index + 1, len(recovery['records']))
        if handle:
            for line in handle:
                check_cancelled(cancelled)
                reference_digest.update(line)
                observed_count += 1
    if observed_count != len(recovery['records']):
        mismatches.append({'reason': 'quality_count_mismatch'})
    if 'quality-v2.jsonl' in snapshot and reference_digest.hexdigest() != snapshot['quality-v2.jsonl']:
        evidence_issues.append('quality_changed_during_consumption')
    unchanged = snapshot_matches(directory, snapshot, names=lambda: _inputs(directory), cancelled=cancelled)
    raw_hashes = {row['file']: row['sha256'] for row in recovery['records']}
    raw_unchanged = snapshot_matches(directory, raw_hashes, cancelled=cancelled)
    if not unchanged:
        evidence_issues.append('source_changed_during_replay')
    numerical_match = not mismatches and bool(recovery['records'])
    mismatches.extend({'reason': reason} for reason in evidence_issues)
    check_cancelled(cancelled)
    return {'schema_version': 2, 'verified_blocks': len(recovery['records']),
            'verified_samples': recovery['verified_samples'], 'recording_complete': recovery['complete'],
            'online_replay_match': numerical_match and not evidence_issues and raw_unchanged,
            'numerical_replay_match': numerical_match, 'quality_evidence_verified': not evidence_issues,
            'evidence_authority': 'local_seal_hashes_not_external_signature',
            'source_unchanged': unchanged, 'raw_hashes_unchanged': raw_unchanged,
            'mismatches': mismatches, 'hardware_verified': False,
            'processing_id': engine.processing_id if engine else None}
