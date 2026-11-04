"""Explicit offline v9 diagnosis of one native NeuroStation Cyton session.

No training, acquisition changes, unit conversion, resampling or classification.
Only a new bundle beneath workspace analysis_outputs is written. report.json is
written last and is the completion marker; an interrupted bundle is not complete.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
import numpy as np
from eeg_tools import ssvep_test_runtime as rt

SOURCE_FILES = ('session.json', 'channel_config.json', 'raw_brainflow.tsv')
CHANNEL_LABELS = tuple(f'CH{i}' for i in range(1, 9))
EEG_COLUMNS = tuple(f'eeg_ch{i}' for i in range(1, 9))
BRANCH = 'Wearable'
LIMITATIONS = [
    'personal_Cyton is an unvalidated source domain. The Wearable checkpoint '
    'branch is selected only for its native 250 Hz sample rate; matching rates '
    'do not establish equivalent hardware, participants or signal distributions.',
    'No clean ground truth is available: producing output does not establish '
    'denoising efficacy, recovered physiology or acquisition quality acceptance.',
    'CH1..CH8 are source labels. Physical electrode placement and anatomical '
    'correspondence to the training channels have not been verified.',
    'Gain, calibration and physical units have not been independently verified. '
    'The output retains input numerical units without conversion.',
    'The finite mask indicates numerical processability only, not signal quality '
    'acceptance. Existing invalid, flat and incomplete-window preservation applies.',
    'The existing residual limit and protected-band statement apply per fixed '
    '256-sample window; full-record spectra and block boundaries are not validated.',
    'This denoiser has no classifier and does not output a gaze target or '
    'confidence. It is not connected to the real-time workstation UI.',
    'This is an explicit offline diagnostic, not production or clinical validation.',
]


def require(ok, message):
    if not ok:
        raise rt.TestModelError(message)


def _cyton(value):
    return isinstance(value, str) and value.casefold() in ('cyton', 'openbci cyton')


def validate_metadata(session, channels):
    """Validate recorded source identities, without inferring electrode locations."""
    require(type(session) is dict and type(channels) is dict,
            'session.json and channel_config.json must be JSON objects')
    require(_cyton(session.get('board')), 'session.json: board must be Cyton')
    require(type(session.get('channel_count')) is int and session['channel_count'] == 8,
            'session.json: channel_count must be 8')
    require(type(session.get('sampling_rate_hz')) in (int, float)
            and session['sampling_rate_hz'] == 250,
            'session.json: sampling_rate_hz must be 250; no automatic resampling')
    require(session.get('simulated') is False, 'session.json: simulated must be false')
    require(_cyton(channels.get('board')), 'channel_config.json: board must be Cyton')
    require(type(channels.get('sampling_rate_hz')) in (int, float)
            and channels['sampling_rate_hz'] == 250,
            'channel_config.json: sampling_rate_hz must be 250')
    order = channels.get('channel_order')
    require(type(order) is list and len(order) == 8
            and all(type(label) is str for label in order)
            and set(order) == set(CHANNEL_LABELS),
            'channel_config.json: channel_order must contain unique CH1..CH8 source labels')
    entries = channels.get('channels')
    require(type(entries) is list and len(entries) == 8
            and all(type(entry) is dict and type(entry.get('name')) is str for entry in entries)
            and {entry['name'] for entry in entries} == set(CHANNEL_LABELS),
            'channel_config.json: channels must contain unique CH1..CH8 names')
    for entry in entries:
        if 'gui_index' in entry:
            require(type(entry['gui_index']) is int
                    and entry['gui_index'] == CHANNEL_LABELS.index(entry['name']),
                    f"channel_config.json: inconsistent gui_index for {entry['name']}")


def parse_raw(payload):
    """Extract EEG columns by name; never drop or repair a physical data row."""
    text = payload.decode('utf-8-sig')
    reader = csv.reader(io.StringIO(text, newline=''), delimiter='\t', strict=True)
    header = next(reader, None)
    require(reader.line_num == 1, 'raw_brainflow.tsv: header must occupy one physical line')
    require(header and all(header) and len(set(header)) == len(header),
            'raw_brainflow.tsv: header must have unique nonempty column names')
    require('sample_index' in header and all(name in header for name in EEG_COLUMNS),
            'raw_brainflow.tsv: header requires sample_index and eeg_ch1..8')
    require({name for name in header if name.startswith('eeg_ch')} == set(EEG_COLUMNS),
            'raw_brainflow.tsv: header must describe exactly eight EEG source columns')
    eeg_indices = [header.index(name) for name in EEG_COLUMNS]
    index_column = header.index('sample_index')
    rows = []
    for row in reader:
        line = reader.line_num
        require(line == len(rows) + 2,
                f'raw_brainflow.tsv line {line}: multiline numeric rows are not allowed')
        require(len(row) == len(header),
                f'raw_brainflow.tsv line {line}: expected {len(header)} columns, got {len(row)}; no rows discarded')
        require(all(value.strip() for value in row),
                f'raw_brainflow.tsv line {line}: empty numeric value; no rows discarded')
        sample_index = row[index_column]
        require(re.fullmatch(r'0|[1-9][0-9]*', sample_index) is not None
                and int(sample_index) == len(rows),
                f'raw_brainflow.tsv line {line}: sample_index must be strictly 0..T-1')
        try:
            values = [float(value) for value in row]
        except ValueError as exc:
            raise rt.TestModelError(
                f'raw_brainflow.tsv line {line}: nonnumeric value; no rows discarded'
            ) from exc
        rows.append([values[index] for index in eeg_indices])
        require(len(rows) <= 8_000_000, 'raw_brainflow.tsv: signal sample count limit')
    require(len(rows) >= 256, 'raw_brainflow.tsv: at least 256 samples required')
    return np.asarray(rows, dtype=np.float64), header


def _npy_bytes(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def _source_hashes(paths):
    return {name: rt.sha(rt.read_bytes(path)) for name, path in paths.items()}


def diagnose(session_dir, output_dir, *, device='cpu', acknowledge_out_of_domain=False):
    """Return the completed diagnostic report; propagate explicit preflight errors."""
    start = time.perf_counter()
    require(device in ('cpu', 'cuda:0'), 'device must be cpu or cuda:0')
    require(acknowledge_out_of_domain is True,
            '--acknowledge-out-of-domain is required for unvalidated personal_Cyton transfer')
    session_dir = rt.safe_local_path(session_dir)
    require(session_dir.is_dir(), '--session must be a native NeuroStation directory')
    paths = {name: rt.safe_local_path(session_dir / name) for name in SOURCE_FILES}
    require(all(path.is_file() for path in paths.values()),
            'native session requires session.json, channel_config.json and raw_brainflow.tsv')
    output_dir = rt.checked_output_dir(output_dir, sources=tuple(paths.values()))
    require(not output_dir.is_relative_to(session_dir), 'output must not be inside the source session')
    payloads = {name: rt.read_bytes(path) for name, path in paths.items()}
    before = {name: rt.sha(data) for name, data in payloads.items()}
    session = rt.json_read(payloads['session.json'])
    channels = rt.json_read(payloads['channel_config.json'])
    validate_metadata(session, channels)
    signal, header = parse_raw(payloads['raw_brainflow.tsv'])
    count = session.get('recorded_samples_per_channel')
    if count is not None:
        require(type(count) is int and count == len(signal),
                'session.json: recorded_samples_per_channel disagrees with raw rows')
    # This mask is numerical only. Do not treat a finite signal as accepted quality.
    valid = np.isfinite(signal).all(axis=1)
    # Loading occurs only after all data and output checks; frozen bindings stay intact.
    cfg, lock, manifest = rt.preflight()
    checkpoint_sha = manifest['files'][rt.MODEL]
    models, gpu = rt.load_models(device=device)
    require(gpu.get('device') == device, 'runtime device differs from requested device')
    denoised, processed, stats = rt.process_signal(
        signal.copy(), valid, models[BRANCH], sample_rate_hz=250, batch_size=64
    )
    require(isinstance(denoised, np.ndarray) and denoised.shape == signal.shape
            and denoised.dtype == np.float64, 'runtime output must be float64 [time,8]')
    require(isinstance(processed, np.ndarray) and processed.dtype == np.bool_
            and processed.shape == (len(signal),), 'runtime processed mask must be bool [time]')
    require(np.array_equal(denoised[~processed], signal[~processed], equal_nan=True),
            'runtime changed samples marked as preserved')
    after = _source_hashes(paths)
    require(after == before, 'source SHA256 changed during inference; no complete output written')
    arrays = {'original_signal.npy': signal, 'denoised.npy': denoised,
              'processed_mask.npy': processed}
    files = {name: _npy_bytes(array) for name, array in arrays.items()}
    report = {
        'schema_version': 1, 'status': 'formal', 'completion_status': 'complete',
        'created_at_utc': rt.now(), 'model_version': cfg['model_version'],
        'source_session': str(session_dir), 'output_directory': str(output_dir),
        'source_domain': 'personal_Cyton', 'source_domain_validated': False,
        'selected_checkpoint_branch': BRANCH,
        'branch_selection_reason': 'native 250 Hz compatibility only; unvalidated source-domain transfer',
        'out_of_domain_acknowledged': True, 'sampling_rate_hz': 250,
        'source_channel_labels': list(CHANNEL_LABELS),
        'configured_channel_order': channels['channel_order'],
        'extracted_eeg_columns': list(EEG_COLUMNS), 'source_raw_header': header,
        'anatomical_channel_equivalence_verified': False,
        'physical_electrode_positions_verified': False,
        'hardware_gain_and_unit_calibration_verified': False,
        'input_shape': list(signal.shape), 'input_dtype': str(signal.dtype),
        'output_shape': list(denoised.shape), 'output_dtype': str(denoised.dtype),
        'artifacts': {name: {'shape': list(array.shape), 'dtype': str(array.dtype),
                             'sha256': rt.sha(files[name])} for name, array in arrays.items()},
        'finite_mask_meaning': 'all eight EEG values finite; numerical processability only, not quality acceptance',
        'finite_samples': int(valid.sum()), 'nonfinite_samples': int((~valid).sum()),
        'processing': stats, 'checkpoint_sha256': checkpoint_sha,
        'base_checkpoint_sha256': cfg['base_checkpoint_sha256'],
        'frozen_lock_file_count': len(lock.get('files', {})),
        'requested_device': device, 'gpu': gpu,
        'source_sha256_before': before, 'source_sha256_after': after,
        'source_hashes_match': True,
        'inference_executed': True, 'training_performed': False,
        'hardware_opened': False, 'raw_sources_modified': False,
        'runtime_enabled': False, 'production_validated': False,
        'acquisition_quality_accepted': False, 'clean_ground_truth_available': False,
        'classifier_executed': False, 'historical_validation_status_rewritten': False,
        'limitations': LIMITATIONS, 'elapsed_seconds': time.perf_counter() - start,
    }
    # Serialize before creating a bundle, so invalid JSON never leaves a completion marker.
    rt.json_bytes(report)
    output_dir.mkdir()  # Exclusive: competing runs cannot replace an existing bundle.
    for name, data in files.items():
        rt.exclusive_write(output_dir / name, data)
    final_hashes = _source_hashes(paths)
    require(final_hashes == before, 'source SHA256 changed while writing; bundle remains incomplete')
    rt.preflight()  # Recheck frozen model/code bindings before marking completion.
    report['source_sha256_after'] = final_hashes
    report['elapsed_seconds'] = time.perf_counter() - start
    rt.exclusive_write(output_dir / 'report.json', rt.json_bytes(report))  # Completion marker LAST.
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path,
                        help='NEW directory below workspace analysis_outputs; parent must exist')
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--acknowledge-out-of-domain', action='store_true')
    args = parser.parse_args(argv)
    try:
        report = diagnose(args.session, args.output, device=args.device,
                          acknowledge_out_of_domain=args.acknowledge_out_of_domain)
        print(json.dumps({
            'status': 'complete', 'output': report['output_directory'],
            'report': str(Path(report['output_directory']) / 'report.json'),
            'source_domain': report['source_domain'], 'source_domain_validated': False,
            'selected_checkpoint_branch': report['selected_checkpoint_branch'],
            'actual_device': report['gpu']['device'],
            'processed_samples': report['processing']['processed_samples'],
            'preserved_samples': report['processing']['preserved_samples'],
            'source_hashes_match': report['source_hashes_match'],
            'production_validated': False,
        }, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, TypeError, KeyError, ImportError, csv.Error) as exc:
        print(json.dumps({'status': 'error', 'error': str(exc), 'complete': False},
                         ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
