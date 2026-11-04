"""Synthetic-session tests; frozen model and CUDA inference are never loaded."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).absolute().parents[1]
spec = importlib.util.spec_from_file_location(
    'neurostation_v9_output_adapter', ROOT / 'scripts/diagnose_neurostation_v9_output.py'
)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
rt = adapter.rt


class NativeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / 'analysis_outputs')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'synthetic_native_session'
        self.source.mkdir()
        self.output = self.root / 'diagnostic'
        self.session = {'board': 'cyton', 'channel_count': 8, 'sampling_rate_hz': 250,
                        'simulated': False, 'recorded_samples_per_channel': 300}
        self.channels = {
            'board': 'OpenBCI Cyton', 'sampling_rate_hz': 250,
            'channel_order': list(reversed(adapter.CHANNEL_LABELS)),
            'channels': [{'name': name, 'gui_index': int(name[2:]) - 1}
                         for name in reversed(adapter.CHANNEL_LABELS)],
        }
        self.write_metadata()
        # Distinct columns with varying signals, deliberately shuffled in the TSV.
        self.signal = np.arange(300, dtype=np.float64)[:, None] / 100 + np.arange(8)[None, :] * 10
        self.header = ['marker', 'eeg_ch8', 'timestamp_s', 'sample_index',
                       'eeg_ch2', 'eeg_ch6', 'eeg_ch4', 'eeg_ch1', 'eeg_ch7', 'eeg_ch3', 'eeg_ch5']
        self.write_raw()
        self.model = object()
        self.preflight = self.enter_patch(patch.object(rt, 'preflight', return_value=(
            rt.EXPECTED_CONFIG.copy(), {'files': {'frozen': '0' * 64}},
            {'files': {rt.MODEL: 'a' * 64}},
        )))
        self.loader = self.enter_patch(patch.object(rt, 'load_models', side_effect=self.fake_load))
        self.processor = self.enter_patch(patch.object(rt, 'process_signal', side_effect=self.fake_process))

    def enter_patch(self, patcher):
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    def write_metadata(self):
        for name, value in [('session.json', self.session), ('channel_config.json', self.channels)]:
            (self.source / name).write_text(json.dumps(value), encoding='utf-8')

    def write_raw(self):
        lines = ['\t'.join(self.header)]
        for index, signal in enumerate(self.signal):
            values = {'sample_index': str(index), 'marker': '0', 'timestamp_s': str(index / 250)}
            values.update({f'eeg_ch{i + 1}': str(value) for i, value in enumerate(signal)})
            lines.append('\t'.join(values[name] for name in self.header))
        (self.source / 'raw_brainflow.tsv').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def fake_load(self, *, device):
        return {'Wearable': self.model}, {'device': device, 'torch_version': 'FAKE',
                'cuda_version': None, 'device_name': 'FAKE GPU' if device == 'cuda:0' else 'CPU'}

    def fake_process(self, signal, valid, model, *, sample_rate_hz, batch_size):
        # A fixed synthetic response verifies adapter delegation, not model performance.
        self.assertIs(model, self.model)
        self.assertEqual(sample_rate_hz, 250)
        self.assertEqual(batch_size, 64)
        self.assertEqual(signal.dtype, np.float64)
        self.assertEqual(valid.dtype, np.bool_)
        out = signal.copy()
        mask = np.zeros(len(signal), dtype=bool)
        if valid[:256].all():
            out[:256] += .125
            mask[:256] = True
        windows = [{'start_sample': 0, 'end_sample_exclusive': 256,
                    'status': 'processed' if mask.any() else 'invalid_mask_window_preserved'},
                   {'start_sample': 256, 'end_sample_exclusive': 300,
                    'status': 'incomplete_tail_preserved'}]
        return out, mask, {'processed_samples': int(mask.sum()),
                           'preserved_samples': int((~mask).sum()), 'windows': windows}

    def diagnose(self, **kwargs):
        return adapter.diagnose(self.source, self.output,
                                acknowledge_out_of_domain=True, **kwargs)

    def assert_preload_error(self, text):
        with self.assertRaisesRegex(ValueError, text):
            self.diagnose()
        self.loader.assert_not_called()
        self.processor.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertEqual(set(path.name for path in self.source.iterdir()), set(adapter.SOURCE_FILES))

    def change_raw_line(self, line_index, transform):
        path = self.source / 'raw_brainflow.tsv'
        lines = path.read_text(encoding='utf-8').splitlines()
        lines[line_index] = transform(lines[line_index])
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def test_named_extraction_receipts_and_foreign_domain_report(self):
        before = {name: (self.source / name).read_bytes() for name in adapter.SOURCE_FILES}
        report = self.diagnose()
        np.testing.assert_array_equal(np.load(self.output / 'original_signal.npy'), self.signal)
        expected = self.signal.copy()
        expected[:256] += .125
        np.testing.assert_array_equal(np.load(self.output / 'denoised.npy'), expected)
        mask = np.load(self.output / 'processed_mask.npy')
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.shape, (300,))
        self.assertEqual(mask.sum(), 256)
        self.assertEqual(report['processing']['preserved_samples'], 44)
        self.assertEqual(report['source_domain'], 'personal_Cyton')
        self.assertFalse(report['source_domain_validated'])
        self.assertEqual(report['selected_checkpoint_branch'], 'Wearable')
        self.assertFalse(report['classifier_executed'])
        self.assertFalse(report['acquisition_quality_accepted'])
        self.assertFalse(report['runtime_enabled'])
        self.assertEqual(report['checkpoint_sha256'], 'a' * 64)
        self.assertEqual(report['input_shape'], [300, 8])
        self.assertEqual(report['output_dtype'], 'float64')
        self.assertEqual(json.loads((self.output / 'report.json').read_text(encoding='utf-8')), report)
        for name, data in before.items():
            self.assertEqual((self.source / name).read_bytes(), data)
            self.assertEqual(report['source_sha256_before'][name], hashlib.sha256(data).hexdigest())
        self.assertEqual(report['source_sha256_before'], report['source_sha256_after'])
        for name, record in report['artifacts'].items():
            self.assertEqual(record['sha256'], hashlib.sha256((self.output / name).read_bytes()).hexdigest())

    def test_bad_header_refused_before_loading(self):
        self.change_raw_line(0, lambda row: row.replace('eeg_ch4', 'unknown_ch4'))
        self.assert_preload_error('header requires')

    def test_duplicate_header_refused(self):
        self.change_raw_line(0, lambda row: row.replace('eeg_ch4', 'eeg_ch5'))
        self.assert_preload_error('unique nonempty')

    def test_short_row_refused_without_skipping(self):
        self.change_raw_line(12, lambda row: '\t'.join(row.split('\t')[:-1]))
        self.assert_preload_error('expected 11 columns, got 10')

    def test_blank_row_refused_without_skipping(self):
        self.change_raw_line(12, lambda row: '')
        self.assert_preload_error('got 0')

    def test_nonnumeric_auxiliary_column_refused(self):
        self.change_raw_line(12, lambda row: 'bad\t' + row.split('\t', 1)[1])
        self.assert_preload_error('nonnumeric value')

    def test_multiline_numeric_row_refused(self):
        self.change_raw_line(12, lambda row: '"0\n"\t' + row.split('\t', 1)[1])
        self.assert_preload_error('multiline numeric rows')

    def test_sample_sequence_gap_refused(self):
        def gap(row):
            columns = row.split('\t')
            columns[self.header.index('sample_index')] = '999'
            return '\t'.join(columns)
        self.change_raw_line(12, gap)
        self.assert_preload_error('strictly 0..T-1')

    def test_wrong_rate_refused(self):
        self.session['sampling_rate_hz'] = 256
        self.write_metadata()
        self.assert_preload_error('sampling_rate_hz must be 250')

    def test_invalid_hardware_and_source_labels_refused(self):
        for field, value, message in [('board', 'synthetic', 'board must be Cyton'),
                                      ('simulated', True, 'simulated must be false'),
                                      ('channel_count', 4, 'channel_count must be 8')]:
            with self.subTest(field=field):
                old = self.session[field]
                self.session[field] = value
                self.write_metadata()
                self.assert_preload_error(message)
                self.session[field] = old
        self.channels['channel_order'][0] = self.channels['channel_order'][1]
        self.write_metadata()
        self.assert_preload_error('unique CH1..CH8')

    def test_metadata_sample_count_mismatch_refused(self):
        self.session['recorded_samples_per_channel'] = 299
        self.write_metadata()
        self.assert_preload_error('disagrees with raw rows')

    def test_existing_output_refused_without_overwrite(self):
        self.output.mkdir()
        sentinel = self.output / 'keep.txt'
        sentinel.write_text('keep')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.diagnose()
        self.loader.assert_not_called()
        self.assertEqual(sentinel.read_text(), 'keep')
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_output_outside_analysis_outputs_refused(self):
        with self.assertRaisesRegex(ValueError, 'analysis_outputs'):
            adapter.diagnose(self.source, ROOT / 'forbidden_adapter_result',
                             acknowledge_out_of_domain=True)
        self.loader.assert_not_called()

    def test_output_inside_source_refused(self):
        self.output = self.source / 'result'
        self.assert_preload_error('inside the source session')

    def test_acknowledgement_required(self):
        with self.assertRaisesRegex(ValueError, 'acknowledge-out-of-domain'):
            adapter.diagnose(self.source, self.output)
        self.loader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_nonfinite_eeg_forwarded_with_numerical_mask(self):
        self.signal[20, 2] = np.nan
        self.write_raw()
        report = self.diagnose()
        received_signal, received_valid, _ = self.processor.call_args.args
        self.assertTrue(np.isnan(received_signal[20, 2]))
        self.assertFalse(received_valid[20])
        self.assertEqual(received_valid.sum(), 299)
        self.assertEqual(report['nonfinite_samples'], 1)
        self.assertIn('not quality acceptance', report['finite_mask_meaning'])
        np.testing.assert_array_equal(np.load(self.output / 'denoised.npy'), self.signal)
        self.assertEqual(report['processing']['windows'][0]['status'], 'invalid_mask_window_preserved')

    def test_flat_window_preservation_delegated_without_adapter_repair(self):
        self.signal[:, 4] = 123.
        self.write_raw()
        self.processor.side_effect = None
        self.processor.return_value = (self.signal.copy(), np.zeros(300, dtype=bool), {
            'processed_samples': 0, 'preserved_samples': 300,
            'windows': [{'status': 'flat_channel_window_preserved'},
                        {'status': 'incomplete_tail_preserved'}],
        })
        report = self.diagnose()
        np.testing.assert_array_equal(self.processor.call_args.args[0], self.signal)
        self.assertTrue(self.processor.call_args.args[1].all())
        self.assertEqual(report['processing']['windows'][0]['status'], 'flat_channel_window_preserved')
        np.testing.assert_array_equal(np.load(self.output / 'denoised.npy'), self.signal)

    def test_source_hash_change_refuses_completion(self):
        with patch.object(adapter, '_source_hashes', return_value={'raw_brainflow.tsv': 'wrong'}):
            with self.assertRaisesRegex(ValueError, 'source SHA256 changed'):
                self.diagnose()
        self.assertFalse(self.output.exists())

    def test_source_change_during_output_leaves_incomplete_bundle(self):
        before = {name: hashlib.sha256((self.source / name).read_bytes()).hexdigest()
                  for name in adapter.SOURCE_FILES}
        with patch.object(adapter, '_source_hashes', side_effect=[before, {'changed': 'hash'}]):
            with self.assertRaisesRegex(ValueError, 'bundle remains incomplete'):
                self.diagnose()
        self.assertTrue((self.output / 'denoised.npy').exists())
        self.assertFalse((self.output / 'report.json').exists())

    def test_final_preflight_mismatch_leaves_results_without_completion_marker(self):
        source_before = {name: (self.source / name).read_bytes() for name in adapter.SOURCE_FILES}
        first_preflight = self.preflight.return_value
        partial_results = {}

        def final_preflight():
            # The final check runs after all three arrays have been written.
            self.assertFalse((self.output / 'report.json').exists())
            for name in ('original_signal.npy', 'denoised.npy', 'processed_mask.npy'):
                partial_results[name] = (self.output / name).read_bytes()
            raise rt.TestModelError('SHA256 mismatch: frozen checkpoint')

        def preflight_sequence():
            if self.preflight.call_count == 1:
                self.assertFalse(self.output.exists())
                return first_preflight
            return final_preflight()

        self.preflight.side_effect = preflight_sequence
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.diagnose()
        self.assertEqual(self.preflight.call_count, 2)
        self.assertEqual(set(path.name for path in self.output.iterdir()), set(partial_results))
        self.assertFalse((self.output / 'report.json').exists())
        # A retry cannot overwrite the incomplete bundle or load another model.
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.diagnose()
        self.assertEqual(self.preflight.call_count, 2)
        self.loader.assert_called_once_with(device='cpu')
        for name, data in partial_results.items():
            self.assertEqual((self.output / name).read_bytes(), data)
        for name, data in source_before.items():
            self.assertEqual((self.source / name).read_bytes(), data)

    def test_unavailable_cuda_error_has_no_cpu_fallback_or_output(self):
        self.loader.side_effect = rt.TestModelError('CUDA unavailable')
        with self.assertRaisesRegex(ValueError, 'CUDA unavailable'):
            self.diagnose(device='cuda:0')
        self.loader.assert_called_once_with(device='cuda:0')
        self.processor.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_completion_marker_written_last_and_write_failure_leaves_no_report(self):
        written = []
        real_write = rt.exclusive_write
        def trace(path, data):
            written.append(path.name)
            return real_write(path, data)
        with patch.object(rt, 'exclusive_write', side_effect=trace):
            self.diagnose()
        self.assertEqual(written, ['original_signal.npy', 'denoised.npy', 'processed_mask.npy', 'report.json'])
        self.output = self.root / 'interrupted'
        def fail(path, data):
            if path.name == 'denoised.npy':
                raise OSError('synthetic write failure')
            return real_write(path, data)
        with patch.object(rt, 'exclusive_write', side_effect=fail):
            with self.assertRaisesRegex(OSError, 'synthetic write failure'):
                self.diagnose()
        self.assertFalse((self.output / 'report.json').exists())
        self.assertEqual(set(path.name for path in self.source.iterdir()), set(adapter.SOURCE_FILES))

    def test_cli_json_success_and_error_using_fake_cuda(self):
        args = ['--session', str(self.source), '--output', str(self.output),
                '--device', 'cuda:0', '--acknowledge-out-of-domain']
        success, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(success), contextlib.redirect_stderr(error):
            self.assertEqual(adapter.main(args), 0)
        result = json.loads(success.getvalue())
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['actual_device'], 'cuda:0')
        self.loader.assert_called_once_with(device='cuda:0')
        self.assertEqual(error.getvalue(), '')
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
            self.assertEqual(adapter.main(args), 1)
        self.assertEqual(json.loads(error.getvalue())['status'], 'error')
        self.assertEqual(self.loader.call_count, 1)


if __name__ == '__main__':
    unittest.main()
