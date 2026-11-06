"""Explicit software-only preview jobs. No board, stimulus or model is imported here."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import hashlib
from copy import deepcopy
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from eeg_tools.quality.evidence import publish_bundle, snapshot_matches, check_cancelled
from eeg_tools.quality.recording import QualitySink
from eeg_tools.quality.replay import replay_quality
from eeg_tools.quality.reporting import build_session_report
from eeg_tools.streaming import BlockRecorder, atomic_json


class PreviewCancelled(RuntimeError):
    pass


def session_path(selected):
    path = Path(selected).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError('quality.error.not_session')
    if path.name == 'raw_blocks' and (path / 'blocks.jsonl').is_file():
        return path.parent
    if not any((path / name).is_file() for name in
               ('session.json', 'events.tsv', 'raw_blocks/blocks.jsonl')):
        raise ValueError('quality.error.not_session')
    return path


def new_output(root, kind, source=None):
    root = Path(root).expanduser().resolve()
    if source is not None:
        source = Path(source).resolve()
        if root == source or source in root.parents:
            raise ValueError('quality.error.output_in_source')
    parent = (root / kind).resolve()
    if source is not None and (parent == source or source in parent.parents):
        raise ValueError('quality.error.output_in_source')
    parent.mkdir(parents=True, exist_ok=True)
    if source is not None and (parent.resolve() == source or source in parent.resolve().parents):
        raise ValueError('quality.error.output_in_source')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return parent / f'{stamp}-{uuid.uuid4().hex[:12]}'


def run_demo(root, publish, cancel: threading.Event, *, seconds=30, realtime=True):
    """Generate a clearly identified synthetic recording through the production sinks.

    Defaults are not a hardware calibration. Flat lines, a synthetic missing value
    and a large spike exercise UI explanations without inventing threshold authority.
    Cancellation seals the known prefix as aborted, never as completed.
    """
    if not 1 <= seconds <= 120:
        raise ValueError('demo length must be between 1 and 120 seconds')
    session = new_output(root, 'demos')
    session.mkdir()
    metadata = {
        'schema_version': 2, 'session_id': 'synthetic-' + session.name,
        'data_origin': 'synthetic_demo', 'simulated': True, 'status': 'recording',
        'sampling_rate_hz': 250, 'trial_count': 0, 'physical_electrodes_verified': False,
        'hardware_opened': False, 'stimulus_started': False, 'model_used': False,
        'scenario': 'CH2 flat at 5-9s and 24-27s; CH3 spike at 16s; CH4 missing at 18s',
    }
    atomic_json(session / 'session.json', metadata)
    names = tuple(f'CH{i + 1}' for i in range(8))
    recorder = BlockRecorder(session / 'raw_blocks', session_id=metadata['session_id'],
                             channel_names=names, eeg_rows=list(range(1, 9)),
                             package_row=0, timestamp_row=9,
                             configuration_id='synthetic-demo-not-hardware-v1')
    sink = None
    state = 'aborted'
    total = int(seconds * 250)
    began = time.monotonic()
    try:
        sink = QualitySink(recorder.directory, list(range(1, 9)), names)
        for start in range(0, total, 50):
            if cancel.is_set():
                break
            count = min(50, total - start)
            index = start + np.arange(count)
            t = index / 250
            eeg = np.array([25 * np.sin(2 * np.pi * (9 + c * .5) * t + c * .2)
                            + 3 * np.sin(2 * np.pi * 50 * t) for c in range(8)])
            eeg[1, ((t >= 5) & (t < 9)) | ((t >= 24) & (t < 27))] = 0
            eeg[2, index == 4000] = 650
            eeg[3, index == 4500] = np.nan
            block = np.vstack((index % 256, eeg, 1000 + t))
            recorder.append(block, sink)
            if recorder.consumer_errors:
                raise RuntimeError('quality.error.demo_processing')
            publish(sink.latest, start + count, total)
            if realtime and cancel.wait(max(0, began + (start + count) / 250 - time.monotonic())):
                break
        state = 'aborted' if cancel.is_set() else 'completed'
    except Exception:
        state = 'error'
        raise
    finally:
        try:
            if sink is not None:
                sink.close()
        finally:
            recorder.close(state, quality_complete=sink is not None and state != 'error')
            metadata.update(status=state, recording_status=state,
                            sample_count=recorder.durable)
            atomic_json(session / 'session.json', metadata)
    return {'kind': 'demo', 'session_dir': str(session), 'status': state,
            'sample_count': recorder.durable, 'data_origin': 'synthetic_demo',
            'hardware_opened': False, 'stimulus_started': False, 'model_used': False}


def run_replay(selected, root, publish, cancel, *, realtime_preview=True):
    source = session_path(selected)
    directory = source / 'raw_blocks'
    if not (directory / 'blocks.jsonl').is_file():
        raise ValueError('quality.error.legacy_no_blocks')

    def update(result, done, total):
        if cancel.is_set():
            raise PreviewCancelled('quality.cancelled')
        publish(result, done, total)
        if realtime_preview:
            # Accelerated playback; no timestamps in the source are rewritten.
            cancel.wait(min(.05, max(.001, (result['stop'] - result['start']) / 1000)))

    result = replay_quality(directory, on_result=update, cancelled=cancel.is_set)
    if cancel.is_set():
        raise PreviewCancelled('quality.cancelled')
    output = new_output(root, 'replays', source)
    publish_bundle(output, {'replay.json': json.dumps(result, ensure_ascii=False, indent=2).encode('utf-8')},
                   cancelled=cancel.is_set)
    return {'kind': 'replay', 'session_dir': str(source), 'output_dir': str(output),
            'verification': result, 'status': 'completed'}


def run_report(selected, root, cancel):
    source = session_path(selected)
    if cancel.is_set():
        raise PreviewCancelled('quality.cancelled')
    output = new_output(root, 'reports', source)
    result = build_session_report(source, output, cancelled=cancel.is_set)
    return {**result, 'kind': 'report', 'session_dir': str(source), 'status': 'completed'}


def export_report(selected, generated, destination, cancel):
    """Export into a fresh child folder; never overwrite source or existing reports."""
    from neurostation_quality_report import render_report_html
    source = session_path(selected)
    if cancel.is_set():
        raise PreviewCancelled('quality.cancelled')
    origin = Path(generated['output_dir'])
    report = deepcopy(generated['report'])
    # Read exactly once for validation/copy; final recheck detects persistent mutation.
    files = {name: (origin / name).read_bytes() for name in ('report.json', 'report.html')}
    expected = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    try:
        if json.loads(files['report.json']) != report or files['report.html'].decode('utf-8') != render_report_html(report):
            raise ValueError('report content mismatch')
    except (ValueError, UnicodeError) as exc:
        raise ValueError('quality.error.report_changed') from exc

    def verify_report():
        if not snapshot_matches(origin, expected, cancelled=cancel.is_set):
            raise ValueError('quality.error.report_changed')

    check_cancelled(cancel.is_set)
    output = new_output(destination, 'NeuroStation-reports', source)
    publish_bundle(output, files, cancelled=cancel.is_set, validate=verify_report)
    return {'kind': 'export', 'export_dir': str(output), 'status': 'completed'}
