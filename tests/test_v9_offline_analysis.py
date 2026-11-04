"""Synthetic native sessions and fake model only; no real weights/GPU/datasets."""
from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

from eeg_tools.workstation import v9_offline as backend
from scripts import diagnose_neurostation_v9_output as adapter
from scripts import run_v9_offline_analysis as cli

ROOT = Path(__file__).absolute().parents[1]


class OfflineBackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='v9_synthetic_', dir=ROOT / 'analysis_outputs')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'analysis_outputs').mkdir()
        self.source = self.root / 'synthetic_native_session'
        self.source.mkdir()
        self.signal = np.column_stack([
            (1 + ch / 10) * np.sin(2 * np.pi * 10 * np.arange(1600) / 250)
            + 0.4 * np.sin(2 * np.pi * 50 * np.arange(1600) / 250) + ch
            for ch in range(8)])
        self.session = {'board': 'cyton', 'channel_count': 8, 'sampling_rate_hz': 250,
                        'simulated': False, 'recorded_samples_per_channel': len(self.signal)}
        self.channels = {'board': 'cyton', 'sampling_rate_hz': 250,
                         'channel_order': backend.CHANNEL_LABELS,
                         'channels': [{'name': name} for name in backend.CHANNEL_LABELS]}
        self.save_metadata()
        self.save_signal()
        self.save_events()
        self.json_file('protocol.json', {'status': 'formal', 'targets': [
            {'id': 'A', 'frequency_hz': 10}, {'id': 'B', 'frequency_hz': 12}]})
        self.output = backend.new_output_dir(self.source, root=self.root)
        self.preflight = self.add_patch(patch.object(adapter.rt, 'preflight', return_value=(
            adapter.rt.EXPECTED_CONFIG.copy(), {'files': {'fake': '0' * 64}},
            {'files': {adapter.rt.MODEL: 'a' * 64}})))
        self.loader = self.add_patch(patch.object(adapter.rt, 'load_models', return_value=(
            {'Wearable': object()}, {'device': 'cpu', 'device_name': 'FAKE CPU', 'torch_version': 'FAKE'})))
        self.add_patch(patch.object(adapter.rt, 'process_signal', side_effect=self.fake_process))
        self.add_patch(patch.object(backend, '_preflight', return_value=None))

    def add_patch(self, patcher):
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def json_file(self, name, value):
        (self.source / name).write_text(json.dumps(value), encoding='utf-8')

    def save_metadata(self):
        self.json_file('session.json', self.session)
        self.json_file('channel_config.json', self.channels)

    def save_signal(self):
        # Deliberately shuffle EEG columns to verify the frozen named adapter extraction.
        fields = ['sample_index', 'marker', 'eeg_ch8', 'eeg_ch2', 'eeg_ch6', 'eeg_ch4',
                  'eeg_ch1', 'eeg_ch7', 'eeg_ch3', 'eeg_ch5']
        with (self.source / 'raw_brainflow.tsv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fields, delimiter='\t')
            writer.writeheader()
            for index, row in enumerate(self.signal):
                values = {'sample_index': index, 'marker': 0}
                values.update({f'eeg_ch{ch + 1}': value for ch, value in enumerate(row)})
                writer.writerow(values)

    def save_events(self, *, missing_offset=False, gaze=False):
        fields = ['event_name', 'trial_index', 'sample_index', 'presented_target_id', 'gaze_target_id']
        with (self.source / 'events.tsv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fields, delimiter='\t')
            writer.writeheader()
            writer.writerow({'event_name': 'stimulus_onset', 'trial_index': '1',
                             'sample_index': 100, 'presented_target_id': 'A'})
            if gaze:
                writer.writerow({'event_name': 'gaze_target_mark', 'trial_index': '1',
                                 'sample_index': 300, 'gaze_target_id': 'A'})
            if not missing_offset:
                writer.writerow({'event_name': 'stimulus_offset', 'trial_index': '1', 'sample_index': 1350})

    def fake_process(self, signal, valid, model, *, sample_rate_hz, batch_size):
        output = signal.copy()
        mask = np.zeros(len(signal), dtype=bool)
        windows = []
        for start in range(0, len(signal), 256):
            end = min(start + 256, len(signal))
            row = {'start_sample': start, 'end_sample_exclusive': end}
            window = signal[start:end]
            if end - start < 256:
                row['status'] = 'incomplete_tail_preserved'
            elif not valid[start:end].all():
                row['status'] = 'invalid_mask_window_preserved'
            elif backend._invalid_signal(window):
                row['status'] = 'flat_channel_window_preserved'
            else:
                row['status'] = 'processed'
                mask[start:end] = True
                output[start:end] *= 0.999
            windows.append(row)
        return output, mask, {'windows': windows, 'processed_samples': int(mask.sum()),
                              'preserved_samples': int((~mask).sum())}

    def run_job(self, output=None):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = backend.run_job(self.source, output or self.output, root=self.root,
                                     device='cpu', acknowledge_out_of_domain=True)
        self.messages = [json.loads(line) for line in stdout.getvalue().splitlines()]
        return result

    def edit_result(self, function):
        path = self.output / 'result.json'
        result = json.loads(path.read_text(encoding='utf-8'))
        function(result)
        path.write_text(json.dumps(result), encoding='utf-8')

    def test_metadata_eligibility(self):
        self.assertEqual(backend.eligible_native_session(self.source), (True, ''))
        for changes in ({'board': 'ganglion'}, {'sampling_rate_hz': 256},
                        {'simulated': True}, {'simulated': 'false'}, {'channel_count': 4}):
            with self.subTest(changes=changes):
                original = dict(self.session)
                self.session.update(changes)
                self.save_metadata()
                self.assertFalse(backend.eligible_native_session(self.source)[0])
                self.session = original
        self.save_metadata()

    def test_missing_files_and_duplicate_channel_rejected(self):
        self.channels['channel_order'] = ['CH1'] * 8
        self.save_metadata()
        self.assertFalse(backend.eligible_native_session(self.source)[0])
        (self.source / 'raw_brainflow.tsv').unlink()
        self.assertFalse(backend.eligible_native_session(self.source)[0])

    def test_duplicate_metadata_json_rejected(self):
        (self.source / 'session.json').write_text('{"board":"cyton","board":"cyton"}', encoding='utf-8')
        self.assertIn('duplicate', backend.eligible_native_session(self.source)[1])

    def test_output_name_no_creation(self):
        self.assertFalse(self.output.exists())
        self.assertTrue(self.output.name.startswith(backend._prefix(self.source)))
        self.assertEqual(self.output.parent, self.root / 'analysis_outputs')
        other = self.root / 'another_parent' / self.source.name
        self.assertNotEqual(backend._prefix(other), backend._prefix(self.source))

    def test_build_command_explicit_executable_no_shell(self):
        self.assertIsNone(backend.default_ml_python(self.root))
        executable = self.root / '.venv-ml-gpu' / 'Scripts' / 'python.exe'
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b'fake executable; never run')
        script = self.root / 'scripts' / 'run_v9_offline_analysis.py'
        script.parent.mkdir()
        script.write_text('# test placeholder', encoding='utf-8')
        command, cwd = backend.build_command(self.source, self.output, root=self.root)
        self.assertEqual(command[:2], [str(executable), str(script)])
        self.assertEqual(command[-3:], ['--device', 'cuda:0', '--acknowledge-out-of-domain'])
        self.assertEqual(cwd, str(self.root))
        self.assertFalse(self.output.exists())
        self.assertEqual(backend.default_ml_python(self.root), executable)

    def test_bad_device_and_missing_python(self):
        with self.assertRaisesRegex(ValueError, 'ML Python'):
            backend.build_command(self.source, self.output, root=self.root)
        with self.assertRaisesRegex(ValueError, 'device'):
            backend.build_command(self.source, self.output, device='cuda:1', root=self.root)

    def test_completion_shared_preview_hashes_and_progress(self):
        before = backend._source_hashes(self.source)
        result = self.run_job()
        self.assertEqual(backend.read_result(self.output, self.source, self.root), result)
        self.assertEqual(result['source_sha256_before'], before)
        self.assertEqual(backend._source_hashes(self.source), before)
        self.assertEqual(result['source_domain'], 'personal_Cyton')
        self.assertFalse(result['source_domain_validated'])
        self.assertFalse(result['production_validated'])
        self.assertEqual(result['processing']['processed_samples'], 1536)
        self.assertEqual(result['processing']['preserved_samples'], 64)
        np.testing.assert_array_equal(np.load(self.output / 'model' / 'original_signal.npy'), self.signal)
        preview = result['preview']
        self.assertEqual(len(preview['time_s']), 750)
        for track in ('raw', 'traditional', 'v9'):
            self.assertEqual(np.asarray(preview[track]).shape, (8, 750))
        self.assertEqual(self.messages[-1]['type'], 'result')
        self.assertEqual(self.messages[-1]['output'], str(self.output))
        self.assertTrue(all(set(('type', 'status', 'stage', 'message')).issubset(x)
                            for x in self.messages[:-1]))
        self.assertFalse((self.output / '.result.pending.json').exists())

    def test_traditional_exact_fixed_sos_and_power_comparison(self):
        result = self.run_job()
        b, a = iirnotch(50.0, 30.0, fs=250)
        sos = np.concatenate((tf2sos(b, a), butter(4, [1, 45], btype='bandpass', fs=250, output='sos')))
        expected = sosfiltfilt(sos, self.signal, axis=0, padtype='odd', padlen=None)
        np.testing.assert_array_equal(np.load(self.output / 'traditional_signal.npy'), expected)
        self.assertEqual(json.loads((self.output / 'plan.json').read_text()), backend.FIXED_PLAN)
        power = result['comparison']['tracks']['traditional']['frequency_band_power']
        self.assertTrue(power['available'])
        self.assertTrue(all(value < -0.99 for value in power['relative_change_from_raw']['line_49_51_hz']))

    def test_fft_uses_exact_same_recorded_trimmed_windows(self):
        from eeg_tools import offline_analysis
        score = offline_analysis._ssvep_score
        calls = []
        def spy(eeg, rate, targets):
            calls.append(eeg.copy())
            self.assertEqual(rate, 250)
            return score(eeg, rate, targets)
        with patch.object(offline_analysis, '_ssvep_score', side_effect=spy):
            result = self.run_job()
        self.assertEqual(len(calls), 3)
        arrays = [self.signal, np.load(self.output / 'traditional_signal.npy'),
                  np.load(self.output / 'model' / 'denoised.npy')]
        for window, signal in zip(calls, arrays):
            np.testing.assert_array_equal(window, signal[225:1350].T)
        for prediction in result['trials'][0]['predictions'].values():
            self.assertEqual(set(prediction), {'target_id', 'confidence'})

    def test_gaze_annotation_is_not_independent_truth_or_accuracy(self):
        self.save_events(gaze=True)
        result = self.run_job()
        self.assertEqual(result['trials'][0]['gaze_target_id'], 'A')
        self.assertFalse(result['comparison']['accuracy_acceptance_performed'])
        for track in result['comparison']['tracks'].values():
            self.assertEqual(track['gaze_truth_count'], 0)
            self.assertEqual(track['gaze_annotation_count'], 1)
            self.assertNotIn('accuracy', track)
            self.assertNotIn('gaze_target_accuracy', track)

    def test_no_events_still_outputs_waveforms(self):
        (self.source / 'events.tsv').unlink()
        result = self.run_job()
        self.assertFalse(result['classification_available'])
        self.assertIn('missing', result['classification_unavailable_reason'])
        self.assertEqual(result['trials'], [])
        self.assertTrue((self.output / 'traditional_signal.npy').exists())
        backend.read_result(self.output, self.source, self.root)

    def test_no_protocol_still_outputs_waveforms(self):
        (self.source / 'protocol.json').unlink()
        result = self.run_job()
        self.assertFalse(result['classification_available'])
        self.assertTrue(result['preview']['v9'])

    def test_missing_offset_not_invented(self):
        self.save_events(missing_offset=True)
        result = self.run_job()
        self.assertFalse(result['classification_available'])
        self.assertIsNone(result['trials'][0]['offset_sample_index'])
        self.assertEqual(result['trials'][0]['predictions']['raw'], {'target_id': None, 'confidence': None})

    def test_invalid_numeric_values_no_interpolation_or_fft(self):
        for bad in (float('nan'), float('inf')):
            with self.subTest(bad=bad):
                self.signal[12, 3] = bad
                self.save_signal()
                output = backend.new_output_dir(self.source, self.root)
                result = self.run_job(output)
                self.assertFalse((output / 'traditional_signal.npy').exists())
                self.assertIsNone(result['preview']['traditional'])
                self.assertIsNone(result['preview']['raw'][3][12])
                self.assertFalse(result['classification_available'])
                self.assertTrue(all(t['trial_count'] == 0 for t in result['comparison']['tracks'].values()))
                self.assertFalse(result['comparison']['tracks']['raw']['frequency_band_power']['available'])
                backend.read_result(output, self.source, self.root)

    def test_flat_channel_no_traditional_or_fft(self):
        self.signal[:, 4] = 3
        self.save_signal()
        result = self.run_job()
        self.assertIsNone(result['preview']['traditional'])
        self.assertFalse((self.output / 'traditional_signal.npy').exists())
        self.assertFalse(result['classification_available'])
        self.assertEqual(result['processing']['processed_samples'], 0)
        self.assertEqual(result['processing']['window_status_counts']['flat_channel_window_preserved'], 6)

    def test_malformed_raw_rows_rejected_before_model(self):
        path = self.source / 'raw_brainflow.tsv'
        lines = path.read_text().splitlines()
        lines[5] = '\t'.join(lines[5].split('\t')[:-1])
        path.write_text('\n'.join(lines), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'columns'):
            self.run_job()
        self.loader.assert_not_called()
        self.assertFalse((self.output / 'result.json').exists())

    def test_no_acknowledgement_no_bundle_or_model(self):
        with self.assertRaisesRegex(ValueError, 'acknowledgement'):
            backend.run_job(self.source, self.output, root=self.root, device='cpu')
        self.assertFalse(self.output.exists())
        self.loader.assert_not_called()

    def test_existing_output_refused(self):
        self.run_job()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.run_job()

    def test_missing_completion_marker_refused(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'missing'):
            backend.read_result(self.output, self.source, self.root)
        self.assertIsNone(backend.find_latest_result(self.source, self.root))

    def test_artifact_tampering_refused(self):
        self.run_job()
        (self.output / 'traditional_signal.npy').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            backend.read_result(self.output, self.source, self.root)
        self.assertIsNone(backend.find_latest_result(self.source, self.root))

    def test_required_receipt_missing_refused(self):
        self.run_job()
        self.edit_result(lambda r: r['artifacts'].pop('model/denoised.npy'))
        with self.assertRaisesRegex(ValueError, 'receipts missing'):
            backend.read_result(self.output, self.source, self.root)

    def test_source_mismatch_refused(self):
        self.run_job()
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            backend.read_result(self.output, self.root / 'other', self.root)
        self.edit_result(lambda r: r.update(source_session=str(self.root / 'other')))
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            backend.read_result(self.output, root=self.root)

    def test_domain_false_flags_required(self):
        self.run_job()
        self.edit_result(lambda r: r.update(source_domain_validated=True))
        with self.assertRaisesRegex(ValueError, 'source-domain'):
            backend.read_result(self.output, self.source, self.root)

    def test_artifact_path_traversal_refused(self):
        self.run_job()
        self.edit_result(lambda r: r['artifacts'].update({'../outside.npy': {'sha256': 'a' * 64}}))
        with self.assertRaisesRegex(ValueError, 'unsafe artifact'):
            backend.read_result(self.output, self.source, self.root)

    def test_preview_time_grid_tampering_refused(self):
        self.run_job()
        self.edit_result(lambda r: r['preview']['time_s'].__setitem__(1, 1.0))
        with self.assertRaisesRegex(ValueError, 'preview receipt'):
            backend.read_result(self.output, self.source, self.root)

    def test_latest_ignores_incomplete_and_wrong_prefix(self):
        self.run_job()
        unfinished = backend.new_output_dir(self.source, self.root)
        unfinished.mkdir()
        other = backend.new_output_dir(self.root / 'other', self.root)
        other.mkdir()
        self.assertEqual(backend.find_latest_result(self.source, self.root), self.output)

    def test_final_preflight_failure_no_completion(self):
        with patch.object(backend, '_preflight', side_effect=ValueError('frozen preflight failed')):
            with self.assertRaisesRegex(ValueError, 'preflight failed'):
                self.run_job()
        self.assertFalse((self.output / 'result.json').exists())
        self.assertIsNone(backend.find_latest_result(self.source, self.root))

    def test_current_raw_replacement_rejects_stale_result(self):
        self.run_job()
        path = self.source / 'raw_brainflow.tsv'
        content = path.read_text(encoding='utf-8')
        path.write_text(content.replace('\t0\t', '\t1\t', 1), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            backend.read_result(self.output, self.source, self.root)
        self.assertIsNone(backend.find_latest_result(self.source, self.root))

    def test_current_metadata_replacement_rejects_stale_result(self):
        self.run_job()
        self.session['operator_note'] = 'changed since output'
        self.save_metadata()
        with self.assertRaisesRegex(ValueError, 'source changed'):
            backend.read_result(self.output, self.source, self.root)
        self.assertIsNone(backend.find_latest_result(self.source, self.root))

    def test_missing_source_directory_rejects_stale_result(self):
        self.run_job()
        self.source.rename(self.root / 'moved_synthetic_source')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            backend.read_result(self.output, self.source, self.root)
        self.assertIsNone(backend.find_latest_result(self.source, self.root))

    def test_missing_source_file_rejects_stale_result(self):
        self.run_job()
        (self.source / 'events.tsv').unlink()
        with self.assertRaisesRegex(ValueError, 'source changed'):
            backend.read_result(self.output, self.source, self.root)
        self.assertIsNone(backend.find_latest_result(self.source, self.root))

    def test_source_change_detected_before_completion(self):
        def changed():
            (self.source / 'quality.json').write_text('{}', encoding='utf-8')
        with patch.object(backend, '_preflight', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                self.run_job()
        self.assertFalse((self.output / 'result.json').exists())

    def test_interrupted_pending_marker_never_published(self):
        original = backend._write_json
        def interrupted(path, value):
            if path.name == '.result.pending.json':
                path.write_text('{', encoding='utf-8')
                raise KeyboardInterrupt()
            return original(path, value)
        with patch.object(backend, '_write_json', side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.run_job()
        self.assertFalse((self.output / 'result.json').exists())
        self.assertIsNone(backend.find_latest_result(self.source, self.root))

    def test_cli_error_json_line_nonzero(self):
        stdout = io.StringIO()
        with patch.object(cli, 'run_job', side_effect=ValueError('expected failure')):
            with contextlib.redirect_stdout(stdout):
                status = cli.main(['--session', str(self.source), '--output', str(self.output)])
        self.assertEqual(status, 1)
        message = json.loads(stdout.getvalue())
        self.assertEqual(message['type'], 'error')
        self.assertFalse(message['complete'])

    def test_cli_bad_arguments_are_json_errors(self):
        for args in ([], ['--session', str(self.source), '--output', str(self.output), '--device', 'cuda:9']):
            with self.subTest(args=args):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    status = cli.main(args)
                self.assertEqual(status, 1)
                self.assertEqual(json.loads(output.getvalue())['type'], 'error')
                self.assertFalse(self.output.exists())

    def test_malformed_complete_json_structure_is_value_error(self):
        self.run_job()
        self.edit_result(lambda r: r.update(comparison=None))
        with self.assertRaisesRegex(ValueError, 'invalid complete result structure'):
            backend.read_result(self.output, self.source, self.root)

    def test_cli_success_passes_explicit_ack_and_device(self):
        with patch.object(cli, 'run_job', return_value={}) as job:
            code = cli.main(['--session', str(self.source), '--output', str(self.output),
                             '--device', 'cpu', '--acknowledge-out-of-domain'])
        self.assertEqual(code, 0)
        self.assertTrue(job.call_args.kwargs['acknowledge_out_of_domain'])
        self.assertEqual(job.call_args.kwargs['device'], 'cpu')

    def test_gui_import_and_read_do_not_import_scientific_modules(self):
        self.run_job()
        code = '''
import sys
class RejectScientific:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy', 'scipy', 'torch'):
            raise RuntimeError('scientific import refused: ' + fullname)
sys.meta_path.insert(0, RejectScientific())
from eeg_tools.workstation.v9_offline import read_result
result = read_result(sys.argv[1], sys.argv[2], sys.argv[3])
assert result['source_domain'] == 'personal_Cyton'
assert not any(x in sys.modules for x in ('numpy', 'scipy', 'torch'))
'''
        process = subprocess.run([sys.executable, '-c', code, str(self.output), str(self.source), str(self.root)],
                                 cwd=ROOT, text=True, capture_output=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr)


if __name__ == '__main__':
    unittest.main()
