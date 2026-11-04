"""Explicit offline desktop v9 analysis; GUI-facing APIs use only stdlib.

Scientific imports and the frozen native adapter are lazy worker-only imports.
A bundle is readable only after result.json is written last and all receipts pass.
"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import stat

ROOT = Path(__file__).absolute().parents[2]
CHANNEL_LABELS = [f'CH{i}' for i in range(1, 9)]
NATIVE_FILES = ('session.json', 'channel_config.json', 'raw_brainflow.tsv')
FIXED_PLAN = {
    'schema_version': 1, 'status': 'formal', 'sample_rate_hz': 250,
    'traditional': {
        'notch_frequency_hz': 50.0, 'notch_q': 30.0,
        'bandpass_hz': [1.0, 45.0], 'butterworth_order': 4,
        'implementation': 'iirnotch -> tf2sos -> concatenate butter SOS -> sosfiltfilt',
        'phase': 'offline_zero_phase', 'padtype': 'odd', 'padlen': None,
        'invalid_policy': 'unavailable_on_any_nonfinite_or_flat_input; no interpolation',
    },
    'fft': {
        'implementation': 'eeg_tools.offline_analysis._ssvep_score',
        'tracks': ['raw', 'traditional', 'v9'],
        'onset_offset': 'shared recorded sample indexes; marker fallback as existing baseline',
        'trim_seconds': 0.5, 'trim_rule': 'min(125, (end-start)//5); existing end-16 bound',
        'minimum_window_samples': 250,
        'missing_offset': 'unavailable; no invented stimulus interval',
    },
    'spectral': {'implementation': 'scipy.signal.welch', 'maximum_samples': 15000,
                 'nperseg_max': 2000, 'bands_hz': {'line_49_51_hz': [49.0, 51.0],
                 'signal_8_40_hz': [8.0, 40.0], 'total_0_5_100_hz': [0.5, 100.0]}},
    'preview_max_samples': 750, 'parameter_search': False,
    'source_domain': 'personal_Cyton', 'source_domain_validated': False,
    'selected_checkpoint_branch': 'Wearable', 'production_validated': False,
}


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _path(value, root=ROOT):
    text = str(value)
    win = PureWindowsPath(text)
    _require(bool(text) and '\x00' not in text and '..' not in text.replace('\\', '/').split('/'),
             'unsafe/traversal path')
    _require(not text.startswith(('\\\\', '//')) and not win.drive.startswith('\\\\'),
             'UNC/device path refused')
    _require(not win.drive or Path(text).is_absolute(), 'relative drive path refused')
    _require(not any(':' in p for p in win.parts[1:]) and (win.drive or ':' not in text),
             'alternate stream refused')
    for part in win.parts:
        _require(not part.endswith((' ', '.')) or part in ('.', '\\', '/'), 'ambiguous path')
        _require(part.rstrip(' .').split('.')[0].upper() not in
                 {'CON', 'NUL', 'PRN', 'AUX', *(f'COM{i}' for i in range(1, 10)),
                  *(f'LPT{i}' for i in range(1, 10))}, 'device path refused')
    path = Path(text)
    if not path.is_absolute():
        path = Path(root) / path
    path = Path(os.path.abspath(path))
    for item in (*reversed(path.parents), path):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        _require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
                 'symlink/reparse path refused')
        if item != path:
            _require(stat.S_ISDIR(info.st_mode), 'path ancestor is not a directory')
    return path


def _same(left, right):
    return os.path.normcase(str(left)) == os.path.normcase(str(right))


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def _json(path):
    _require(path.is_file() and 0 < path.stat().st_size <= 16 * 1024 * 1024,
             f'JSON file missing or oversized: {path.name}')
    def reject(value):
        raise ValueError(f'nonfinite JSON value: {value}')
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=_unique,
                      parse_constant=reject)


def _sha(path):
    path = _path(path)
    _require(path.is_file(), f'regular file required: {path.name}')
    info = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    _require(info.st_size == after.st_size and info.st_mtime_ns == after.st_mtime_ns,
             'file changed while hashing')
    return digest.hexdigest()


def _validate_metadata(session, channels):
    _require(type(session) is dict and type(channels) is dict, 'native metadata must be objects')
    for obj, name in ((session, 'session'), (channels, 'channel_config')):
        board = obj.get('board')
        _require(isinstance(board, str) and board.casefold() in ('cyton', 'openbci cyton'),
                 f'{name}: board must be Cyton')
        rate = obj.get('sampling_rate_hz')
        _require(type(rate) in (int, float) and rate == 250, f'{name}: sampling rate must be 250 Hz')
    _require(type(session.get('channel_count')) is int and session['channel_count'] == 8,
             'native session requires 8 channels')
    _require(session.get('simulated') is False, 'simulated must be false')
    order = channels.get('channel_order')
    _require(type(order) is list and len(order) == 8 and all(type(x) is str for x in order)
             and set(order) == set(CHANNEL_LABELS), 'unique CH1..CH8 channel order required')
    entries = channels.get('channels')
    _require(type(entries) is list and len(entries) == 8
             and all(type(x) is dict and type(x.get('name')) is str for x in entries)
             and {x['name'] for x in entries} == set(CHANNEL_LABELS), 'unique CH1..CH8 entries required')
    for entry in entries:
        if 'gui_index' in entry:
            _require(type(entry['gui_index']) is int
                     and entry['gui_index'] == CHANNEL_LABELS.index(entry['name']), 'inconsistent gui_index')


def eligible_native_session(session_dir):
    """Cheap metadata-only eligibility check; raw rows are validated by the worker."""
    try:
        path = _path(session_dir)
        _require(path.is_dir(), 'native session directory missing')
        files = [_path(path / name) for name in NATIVE_FILES]
        _require(all(p.is_file() for p in files), 'requires session.json, channel_config.json and raw_brainflow.tsv')
        _validate_metadata(_json(files[0]), _json(files[1]))
        return True, ''
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return False, str(exc)


def default_ml_python(root=ROOT):
    path = _path(Path(root) / '.venv-ml-gpu' / 'Scripts' / 'python.exe', root)
    return path if path.is_file() else None


def _prefix(session_dir):
    canonical = os.path.normcase(str(_path(session_dir)))
    return 'v9_offline_' + hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:12] + '_'


def new_output_dir(session_dir, root=ROOT):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    return _path(Path(root) / 'analysis_outputs' / (_prefix(_path(session_dir, root)) + stamp), root)


def _output_path(output_dir, session_dir=None, root=ROOT):
    path = _path(output_dir, root)
    analysis = _path(Path(root) / 'analysis_outputs', root)
    _require(path.parent == analysis, 'output must be a direct bundle under analysis_outputs')
    _require(re.fullmatch(r'v9_offline_[a-f0-9]{12}_[0-9]{8}T[0-9]{12}Z', path.name),
             'output bundle name must contain source path hash and UTC timestamp')
    if session_dir is not None:
        source = _path(session_dir, root)
        _require(path.name.startswith(_prefix(source)), 'output/source path hash mismatch')
        _require(not path.is_relative_to(source) and not source.is_relative_to(path),
                 'output overlaps source session')
    return path


def build_command(session_dir, output_dir, *, device='cuda:0', python=None, root=ROOT):
    root = _path(root)
    source = _path(session_dir, root)
    ok, reason = eligible_native_session(source)
    _require(ok, reason)
    output = _output_path(output_dir, source, root)
    _require(not os.path.lexists(output), 'output already exists; no overwrite')
    _require(device in ('cpu', 'cuda:0'), 'device must be cpu or cuda:0')
    executable = _path(python, root) if python is not None else default_ml_python(root)
    _require(executable is not None and executable.is_file(), 'ML Python executable missing')
    worker = _path(root / 'scripts' / 'run_v9_offline_analysis.py', root)
    _require(worker.is_file(), 'offline analysis worker missing')
    return ([str(executable), str(worker), '--session', str(source), '--output', str(output),
             '--device', device, '--acknowledge-out-of-domain'], str(root))


def _artifact(output, name):
    _require(type(name) is str and name and not Path(name).is_absolute()
             and '\\' not in name and ':' not in name and '..' not in name.split('/'),
             'unsafe artifact path')
    path = _path(output / name)
    _require(path.is_relative_to(output), 'artifact escapes bundle')
    return path


def _preview_valid(preview):
    _require(type(preview) is dict and preview.get('sample_rate_hz') == 250
             and preview.get('channel_labels') == CHANNEL_LABELS
             and type(preview.get('start_sample')) is int and preview['start_sample'] >= 0,
             'invalid preview metadata')
    time_s = preview.get('time_s')
    _require(type(time_s) is list and 0 < len(time_s) <= 750, 'invalid preview length')
    for index, value in enumerate(time_s):
        _require(type(value) in (int, float) and math.isfinite(value)
                 and abs(value - (preview['start_sample'] + index) / 250) < 1e-9,
                 'preview time grid mismatch')
    for track in ('raw', 'traditional', 'v9'):
        data = preview.get(track)
        if track == 'traditional' and data is None:
            continue
        _require(type(data) is list and len(data) == 8, f'invalid {track} preview channels')
        for row in data:
            _require(type(row) is list and len(row) == len(time_s), 'preview shape mismatch')
            _require(all(x is None or (type(x) in (int, float) and math.isfinite(x)) for x in row),
                     'invalid preview value')


def read_result(output_dir, session_dir=None, root=ROOT):
    """Validate a completed bundle and current source using only stdlib."""
    output = _output_path(output_dir, session_dir, root)
    result = _json(_path(output / 'result.json'))
    try:
        return _validate_result(output, result, session_dir, root)
    except (TypeError, KeyError, AttributeError) as exc:
        raise ValueError(f'invalid complete result structure: {exc}') from exc


def _validate_result(output, result, session_dir=None, root=ROOT):
    _require(type(result) is dict and result.get('schema_version') == 1
             and result.get('status') == 'formal' and result.get('completion_status') == 'complete',
             'result has no valid completion marker')
    source = _path(result.get('source_session', ''), root)
    _require(Path(result.get('source_session', '')).is_absolute(), 'source_session must be absolute')
    _output_path(output, source, root)
    if session_dir is not None:
        _require(_same(source, _path(session_dir, root)), 'result source session mismatch')
    _require(_same(_path(result.get('output_directory', ''), root), output), 'result output path mismatch')
    _require(result.get('source_domain') == 'personal_Cyton'
             and result.get('source_domain_validated') is False
             and result.get('production_validated') is False
             and result.get('selected_checkpoint_branch') == 'Wearable'
             and result.get('out_of_domain_acknowledged') is True, 'invalid source-domain flags')
    artifacts = result.get('artifacts')
    _require(type(artifacts) is dict, 'artifact SHA dictionary missing')
    required = {'plan.json', 'comparison.json', 'trials.json', 'preview.json', 'model/report.json',
                'model/original_signal.npy', 'model/denoised.npy', 'model/processed_mask.npy'}
    tracks = result.get('comparison', {}).get('tracks', {})
    _require(set(tracks) == {'raw', 'traditional', 'v9'}, 'comparison tracks missing')
    traditional = tracks['traditional'].get('available') is True
    if traditional:
        required.add('traditional_signal.npy')
    _require(required.issubset(artifacts), 'required artifact receipts missing')
    for name, receipt in artifacts.items():
        _require(type(receipt) is dict and re.fullmatch(r'[a-f0-9]{64}', str(receipt.get('sha256', ''))),
                 'invalid artifact SHA receipt')
        _require(_sha(_artifact(output, name)) == receipt['sha256'], f'artifact SHA mismatch: {name}')
    _require(_json(output / 'plan.json') == FIXED_PLAN, 'fixed analysis plan changed')
    _require(_json(output / 'comparison.json') == result['comparison'], 'comparison receipt mismatch')
    _require(_json(output / 'trials.json') == result.get('trials'), 'trial receipt mismatch')
    _require(_json(output / 'preview.json') == result.get('preview'), 'preview receipt mismatch')
    extension_files = {'personal_decoder/analysis.json', 'personal_decoder/config.json', 'personal_decoder/plan.json'}
    if 'personal_decoder' in result:
        from eeg_tools.workstation.personal_decoder_analysis import validate_analysis
        _require(extension_files.issubset(artifacts), 'personal decoder artifact receipts missing')
        decoder = validate_analysis(result['personal_decoder'])
        _require(_json(output / 'personal_decoder' / 'analysis.json') == decoder, 'personal decoder receipt mismatch')
        _require(_json(output / 'personal_decoder' / 'config.json') == decoder['config'], 'personal decoder config receipt mismatch')
        _require(_json(output / 'personal_decoder' / 'plan.json') == decoder['plan'], 'personal decoder plan receipt mismatch')
    else:
        _require(not extension_files.intersection(artifacts), 'personal decoder metadata missing')
    model = _json(output / 'model' / 'report.json')
    _require(model.get('completion_status') == 'complete' and model.get('source_domain') == 'personal_Cyton'
             and model.get('source_domain_validated') is False and model.get('production_validated') is False
             and model.get('selected_checkpoint_branch') == 'Wearable'
             and _same(_path(model.get('source_session', ''), root), source), 'model/source receipt mismatch')
    _require(model.get('processing') == result.get('model_processing')
             and model.get('model_version') == result.get('model_version')
             and model.get('checkpoint_sha256') == result.get('checkpoint_sha256')
             and model.get('gpu') == result.get('gpu'), 'model metadata receipt mismatch')
    before, after = result.get('source_sha256_before'), result.get('source_sha256_after')
    _require(type(before) is dict and set(NATIVE_FILES).issubset(before) and before == after
             and result.get('source_hashes_match') is True, 'source hash receipt mismatch')
    for name, digest in before.items():
        _artifact(source, name)
        _require(re.fullmatch(r'[a-f0-9]{64}', str(digest)), 'invalid source SHA receipt')
    model_before, model_after = model.get('source_sha256_before'), model.get('source_sha256_after')
    _require(type(model_before) is dict and set(NATIVE_FILES).issubset(model_before)
             and model_before == model_after, 'model source SHA receipt mismatch')
    for name, digest in model_before.items():
        _require(before.get(name) == digest, 'model source SHA receipt mismatch')
    try:
        _require(source.is_dir(), 'source changed: source directory missing')
        _require(_source_hashes(source) == before, 'source changed: current source SHA mismatch')
    except (OSError, ValueError) as exc:
        raise ValueError(f'source changed: {exc}') from exc
    _preview_valid(result.get('preview'))
    _require((result['preview']['traditional'] is not None) == traditional, 'traditional preview availability mismatch')
    expected_processing = {'processed_samples': model['processing']['processed_samples'],
                           'preserved_samples': model['processing']['preserved_samples'],
                           'windows': model['processing'].get('windows', []),
                           'window_status_counts': dict(Counter(w.get('status', 'unknown')
                                    for w in model['processing'].get('windows', [])))}
    _require(result.get('processing') == expected_processing, 'processing summary receipt mismatch')
    _require(result.get('classification_available') == result['comparison'].get('classification_available')
             and result.get('classification_unavailable_reason')
             == result['comparison'].get('classification_unavailable_reason'), 'classification metadata mismatch')
    result['output_directory'] = str(output)
    return result


def find_latest_result(session_dir, root=ROOT):
    source = _path(session_dir, root)
    analysis = _path(Path(root) / 'analysis_outputs', root)
    if not analysis.is_dir():
        return None
    candidates = sorted(analysis.glob(_prefix(source) + '*'), key=lambda p: p.name, reverse=True)
    for output in candidates:
        try:
            read_result(output, source, root)
            return output
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    return None


def _write_json(path, value):
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _progress(stage, message, status='running'):
    print(json.dumps({'type': 'progress', 'status': status, 'stage': stage, 'message': message},
                     ensure_ascii=False, allow_nan=False), flush=True)


def _source_hashes(source):
    result = {}
    for directory, dirs, files in os.walk(source, followlinks=False):
        for name in dirs:
            _path(Path(directory) / name)
        for name in sorted(files):
            path = _path(Path(directory) / name)
            result[path.relative_to(source).as_posix()] = _sha(path)
    return dict(sorted(result.items()))


def _diagnose(source, output, device):
    from scripts.diagnose_neurostation_v9_output import diagnose
    return diagnose(source, output, device=device, acknowledge_out_of_domain=True)


def _preflight():
    from eeg_tools import ssvep_test_runtime
    return ssvep_test_runtime.preflight()


def _invalid_signal(signal):
    import numpy as np
    if not np.isfinite(signal).all():
        return 'nonfinite_input; no interpolation'
    for start in range(0, len(signal), 256):
        window = signal[start:start + 256]
        if len(window) >= 2 and np.any(np.sqrt(np.mean((window - window.mean(axis=0)) ** 2, axis=0)) <= 1e-8):
            return 'flat_input_channel_or_window; no interpolation'
    return None


def _traditional(signal):
    import numpy as np
    from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos
    b, a = iirnotch(50.0, 30.0, fs=250.0)
    sos = np.concatenate((tf2sos(b, a), butter(4, [1.0, 45.0], btype='bandpass', fs=250.0, output='sos')))
    return sosfiltfilt(sos, signal, axis=0, padtype='odd', padlen=None)


def _spectral(signal):
    import numpy as np
    from scipy.signal import welch
    if signal is None:
        return {'available': False, 'reason': 'track unavailable'}
    invalid = _invalid_signal(signal)
    if invalid:
        return {'available': False, 'reason': invalid}
    segment = signal[:FIXED_PLAN['spectral']['maximum_samples']]
    freq, psd = welch(segment, fs=250.0, axis=0, nperseg=min(2000, len(segment)))
    bands = {}
    for name, (lo, hi) in FIXED_PLAN['spectral']['bands_hz'].items():
        chosen = (freq >= lo) & (freq <= hi)
        y, x = psd[chosen], freq[chosen]
        integral = np.sum((y[1:] + y[:-1]) * (np.diff(x)[:, None] / 2), axis=0)
        bands[name] = [float(value) for value in integral]
    total = bands['total_0_5_100_hz']
    return {'available': True, 'start_sample': 0, 'sample_count': len(segment),
            'units': 'input_numeric_units_squared; physical calibration unverified', 'power_by_channel': bands,
            'line_power_fraction_by_channel': [line / full if full > 0 else None
                                               for line, full in zip(bands['line_49_51_hz'], total)]}


def _classification_inputs(source, signal):
    """Read recorded intervals once; never invent missing offsets or gaze truth."""
    import numpy as np
    from eeg_tools.offline_analysis import _marker_sample_indexes, _int
    protocol = next((source / name for name in ('protocol.json', 'ssvep_config.json')
                     if (source / name).is_file()), None)
    if protocol is None or not (source / 'events.tsv').is_file():
        return [], [], 'missing events.tsv or protocol.json/ssvep_config.json'
    try:
        payload = _json(_path(protocol))
        targets = []
        for item in payload.get('targets', []):
            if isinstance(item, dict):
                targets.append((str(item['id']), float(item['frequency_hz'])))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                targets.append((str(item[0]), float(item[1])))
            else:
                raise ValueError('invalid protocol target')
        _require(targets and len({name for name, _ in targets}) == len(targets)
                 and all(name and math.isfinite(hz) and 0 < hz < 125 for name, hz in targets),
                 'protocol targets invalid or empty')
        with _path(source / 'events.tsv').open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle, delimiter='\t', strict=True)
            _require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames),
                     'events header missing or duplicate')
            events = list(reader)
            _require(all(None not in event and all(v is not None for v in event.values()) for event in events),
                     'events malformed row')
        # Read marker only for the existing baseline's recorded marker fallback.
        with (source / 'raw_brainflow.tsv').open(encoding='utf-8-sig', newline='') as handle:
            raw = csv.DictReader(handle, delimiter='\t')
            marker = np.asarray([float(row['marker']) for row in raw]) if 'marker' in (raw.fieldnames or []) else None
        _marker_sample_indexes(events, marker)
        onsets = [e for e in events if e.get('event_name') == 'stimulus_onset']
        _require(onsets, 'no stimulus_onset events')
        keys = [e.get('trial_index', '') for e in onsets]
        _require(len(set(keys)) == len(keys), 'duplicate trial onset indexes')
        trials = []
        for onset in onsets:
            index = onset.get('trial_index', '')
            offsets = [e for e in events if e.get('event_name') == 'stimulus_offset' and e.get('trial_index') == index]
            start = _int(onset.get('sample_index'))
            end = _int(offsets[0].get('sample_index')) if len(offsets) == 1 else None
            reason = None
            if start is None or end is None or start < 0 or end <= start or end > len(signal):
                reason = 'missing, ambiguous or out-of-range recorded onset/offset'
                window_start = None
            else:
                trim = min(125, max(0, (end - start) // 5))
                window_start = min(start + trim, max(start, end - 16))
                if end - window_start < 250:
                    reason = 'stimulus window too short for existing FFT baseline'
            presented = onset.get('presented_target_id') or onset.get('target_id') or ''
            if not presented:
                try:
                    hz = float(onset.get('frequency_hz', ''))
                    presented = next((name for name, freq in targets if round(freq, 6) == round(hz, 6)), '')
                except ValueError:
                    pass
            gaze = [e.get('gaze_target_id') for e in events if e.get('event_name') == 'gaze_target_mark'
                    and e.get('trial_index') == index and e.get('gaze_target_id')]
            gaze_id = gaze[-1] if gaze else onset.get('gaze_target_id', '')
            trials.append({'trial_index': index, 'presented_target_id': presented,
                           'gaze_target_id': gaze_id, 'gaze_truth_available': False,
                           'gaze_label_definition': 'recorded annotation; independent gaze truth unverified',
                           'onset_sample_index': start, 'offset_sample_index': end,
                           'window_start_sample_index': window_start, 'unavailable_reason': reason})
        return targets, trials, None
    except (OSError, ValueError, TypeError, KeyError, csv.Error) as exc:
        return [], [], f'classification metadata unavailable: {exc}'


def _compare(source, tracks, invalid_reason):
    from eeg_tools.offline_analysis import _ssvep_score
    targets, trials, metadata_reason = _classification_inputs(source, tracks['raw'])
    reason = invalid_reason or metadata_reason
    for trial in trials:
        predictions, availability = {}, {}
        for name, signal in tracks.items():
            unavailable = reason or trial['unavailable_reason']
            if signal is None:
                unavailable = invalid_reason or 'traditional track unavailable'
            if not unavailable:
                start, end = trial['window_start_sample_index'], trial['offset_sample_index']
                window = signal[start:end].T
                unavailable = _invalid_signal(window.T)
                if not unavailable:
                    score = _ssvep_score(window, 250.0, targets)
                    _require(all(math.isfinite(score[key]) for key in ('confidence', 'relative_score')),
                             'nonfinite FFT result')
                    predictions[name] = {'target_id': score['predicted_target_id'],
                                         'confidence': float(score['confidence'])}
                    availability[name] = {'available': True, 'window_samples': score['samples']}
                    continue
            predictions[name] = {'target_id': None, 'confidence': None}
            availability[name] = {'available': False, 'reason': unavailable}
        trial['predictions'] = predictions
        trial['prediction_status'] = availability
    summaries = {}
    for name, signal in tracks.items():
        predicted = [t for t in trials if t['prediction_status'][name]['available']]
        summaries[name] = {
            'available': signal is not None,
            'reason': invalid_reason if signal is None else None,
            'trial_count': len(predicted), 'recorded_trial_count': len(trials),
            'presented_match_count': sum(bool(t['presented_target_id'])
                                         and t['presented_target_id'] == t['predictions'][name]['target_id']
                                         for t in predicted),
            'presented_label_count': sum(bool(t['presented_target_id']) for t in predicted),
            'gaze_truth_count': 0, 'gaze_truth_match_count': 0,
            'gaze_annotation_count': sum(bool(t['gaze_target_id']) for t in predicted),
            'frequency_band_power': _spectral(signal),
        }
    available = any(t['prediction_status']['raw']['available'] for t in trials)
    comparison = {
        'classification_available': available,
        'classification_unavailable_reason': None if available else (reason or 'no valid recorded FFT intervals'),
        'tracks': summaries, 'independent_gaze_truth_available': False,
        'accuracy_acceptance_performed': False, 'clean_ground_truth_available': False,
        'presented_label_definition': 'rendered/plan target; proxy match count only, not gaze accuracy',
        'gaze_label_definition': 'recorded/manual annotation; independent gaze truth not established',
        'denoising_efficacy_validated': False,
        'frequency_power_definition': 'descriptive numerical power; not noise removal rate',
    }
    raw_power = summaries['raw']['frequency_band_power']
    for name in ('traditional', 'v9'):
        power = summaries[name]['frequency_band_power']
        if raw_power['available'] and power['available']:
            power['relative_change_from_raw'] = {
                band: [(b - a) / a if a > 0 else None for a, b in zip(values, power['power_by_channel'][band])]
                for band, values in raw_power['power_by_channel'].items()}
    return comparison, trials


def _preview(tracks):
    import numpy as np
    length = min(750, len(tracks['raw']))
    def values(signal):
        if signal is None:
            return None
        return [[float(x) if np.isfinite(x) else None for x in row] for row in signal[:length].T]
    return {'sample_rate_hz': 250, 'start_sample': 0,
            'time_s': [i / 250 for i in range(length)], 'channel_labels': CHANNEL_LABELS,
            **{name: values(signal) for name, signal in tracks.items()}}


def _write_completion(output, result):
    # A killed writer leaves a pending JSON, never a partially written result.json.
    pending = output / '.result.pending.json'
    _write_json(pending, result)
    os.link(pending, output / 'result.json')  # Atomic exclusive publication.
    pending.unlink()


def run_job(session_dir, output_dir, *, device='cuda:0', acknowledge_out_of_domain=False, root=ROOT):
    """Write a fresh complete bundle; failures/cancellation leave no result.json."""
    _require(acknowledge_out_of_domain is True, 'explicit out-of-domain acknowledgement required')
    _require(device in ('cpu', 'cuda:0'), 'device must be cpu or cuda:0')
    source = _path(session_dir, root)
    ok, reason = eligible_native_session(source)
    _require(ok, reason)
    output = _output_path(output_dir, source, root)
    _require(not os.path.lexists(output), 'output already exists; no overwrite')
    _require(output.parent.is_dir(), 'analysis_outputs parent missing')
    _progress('preflight', 'Validating native source identities and snapshotting all source files')
    before = _source_hashes(source)
    import numpy as np
    output.mkdir()
    _write_json(output / 'plan.json', FIXED_PLAN)
    _progress('v9', f'Running frozen conservative v9 on {device}; personal_Cyton transfer unvalidated')
    model = _diagnose(source, output / 'model', device)
    _require(model.get('completion_status') == 'complete' and model.get('source_domain') == 'personal_Cyton'
             and model.get('source_domain_validated') is False and model.get('production_validated') is False
             and model.get('selected_checkpoint_branch') == 'Wearable'
             and model.get('gpu', {}).get('device') == device, 'invalid model completion/device receipt')
    raw = np.load(output / 'model' / 'original_signal.npy', allow_pickle=False)
    denoised = np.load(output / 'model' / 'denoised.npy', allow_pickle=False)
    mask = np.load(output / 'model' / 'processed_mask.npy', allow_pickle=False)
    _require(raw.ndim == 2 and raw.shape[1] == 8 and len(raw) >= 256 and raw.dtype == np.float64
             and denoised.shape == raw.shape and denoised.dtype == np.float64
             and mask.shape == (len(raw),) and mask.dtype == np.bool_, 'invalid model array layout')
    _require(np.array_equal(raw[~mask], denoised[~mask], equal_nan=True), 'preserved samples changed')
    _require(np.isfinite(denoised[np.isfinite(raw).all(axis=1)]).all(), 'model produced nonfinite output')
    invalid = _invalid_signal(raw)
    _progress('traditional', 'Computing fixed 50 Hz Q30 notch and order-4 1–45 Hz offline bandpass')
    traditional = None if invalid else _traditional(raw)
    if traditional is not None:
        _require(traditional.shape == raw.shape and np.isfinite(traditional).all(), 'invalid traditional output')
        with (output / 'traditional_signal.npy').open('xb') as handle:
            np.save(handle, traditional, allow_pickle=False)
    tracks = {'raw': raw, 'traditional': traditional, 'v9': denoised}
    _progress('comparison', 'Scoring three tracks on shared recorded intervals and computing band powers')
    comparison, trials = _compare(source, tracks, invalid)
    _write_json(output / 'comparison.json', comparison)
    _write_json(output / 'trials.json', trials)
    _progress('personal_decoder', 'Computing fixed personal CCA candidates and development controls')
    from eeg_tools.workstation.personal_decoder_analysis import analyze as analyze_personal_decoder
    targets, recorded_trials, metadata_reason = _classification_inputs(source, raw)
    personal_decoder = analyze_personal_decoder(tracks, targets, recorded_trials, invalid or metadata_reason)
    (output / 'personal_decoder').mkdir()
    for name, value in (('analysis.json', personal_decoder), ('config.json', personal_decoder['config']),
                        ('plan.json', personal_decoder['plan'])):
        _write_json(output / 'personal_decoder' / name, value)
    preview = _preview(tracks)
    _write_json(output / 'preview.json', preview)
    statuses = dict(Counter(w.get('status', 'unknown') for w in model['processing'].get('windows', [])))
    processing = {'processed_samples': model['processing']['processed_samples'],
                  'preserved_samples': model['processing']['preserved_samples'],
                  'window_status_counts': statuses, 'windows': model['processing'].get('windows', [])}
    _require(processing['processed_samples'] == int(mask.sum())
             and processing['preserved_samples'] == int((~mask).sum()), 'processing receipt mismatch')
    names = ['plan.json', 'comparison.json', 'trials.json', 'preview.json', 'model/report.json',
             'model/original_signal.npy', 'model/denoised.npy', 'model/processed_mask.npy',
             'personal_decoder/analysis.json', 'personal_decoder/config.json', 'personal_decoder/plan.json']
    if traditional is not None:
        names.append('traditional_signal.npy')
    artifacts = {name: {'sha256': _sha(output / name)} for name in names}
    _progress('verification', 'Rechecking frozen preflight and all source hashes before completion')
    _preflight()
    after = _source_hashes(source)
    _require(before == after, 'source changed during analysis; bundle remains incomplete')
    result = {
        'schema_version': 1, 'status': 'formal', 'completion_status': 'complete',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_session': str(source), 'output_directory': str(output),
        'source_domain': 'personal_Cyton', 'source_domain_validated': False,
        'selected_checkpoint_branch': 'Wearable', 'out_of_domain_acknowledged': True,
        'production_validated': False, 'runtime_enabled': False,
        'model_version': model['model_version'], 'checkpoint_sha256': model['checkpoint_sha256'],
        'gpu': model['gpu'], 'requested_device': device, 'sampling_rate_hz': 250,
        'processing': processing, 'model_processing': model['processing'],
        'comparison': comparison, 'classification_available': comparison['classification_available'],
        'classification_unavailable_reason': comparison['classification_unavailable_reason'],
        'preview': preview, 'trials': trials, 'artifacts': artifacts, 'personal_decoder': personal_decoder,
        'source_sha256_before': before, 'source_sha256_after': after, 'source_hashes_match': True,
        'raw_sources_modified': False, 'training_performed': False, 'hardware_opened': False,
        'historical_validation_status_rewritten': False, 'acquisition_quality_accepted': False,
        'limitations': [
            'Personal Cyton is an unvalidated transfer domain; Wearable denotes checkpoint branch only.',
            'Traditional filtering uses full-record zero-phase processing; no realtime causality claim.',
            'FFT presented-target matches are proxy counts; annotated gaze is not independent gaze truth.',
            'Band power changes are descriptive and do not establish noise removal or recovery.',
            'Electrode positions, hardware gain and physical units are not independently verified.',
            'V9 protection is per fixed 256-sample window, not a full-record frequency guarantee.',
            'CCA candidates are exploratory development outputs; both personal sessions were previously explored.',
            'CCA score margin is not a probability; noise also produces a forced reference candidate.',
        ],
    }
    _preview_valid(result['preview'])
    json.dumps(result, allow_nan=False)
    for name, receipt in artifacts.items():
        _require(_sha(output / name) == receipt['sha256'], 'artifact changed before completion')
    _require(_source_hashes(source) == before, 'source changed before completion')
    _validate_result(output, result, source, root)
    _write_completion(output, result)  # Completion marker LAST.
    _progress('complete', 'Offline analysis bundle completed', status='complete')
    print(json.dumps({'type': 'result', 'status': 'complete', 'output': str(output),
                      'result': str(output / 'result.json'), 'source_domain': 'personal_Cyton',
                      'production_validated': False}, ensure_ascii=False), flush=True)
    return result
