"""Audit regressions: hostile/stale evidence is not a successful delivery.

All inputs are software-only synthetic sessions in TemporaryDirectory.
"""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from eeg_tools.quality import preview, reporting, replay
from eeg_tools.quality.commands import QualityCommandGate
from neurostation_quality_report import render_report_html


class QualityAuditEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cancel = threading.Event()
        self.session = Path(preview.run_demo(self.root / 'demo', lambda *_: None,
                                            self.cancel, seconds=1, realtime=False)['session_dir'])
        self.raw = self.session / 'raw_blocks'

    def test_saved_replay_binds_verdict_to_the_exact_source_files(self):
        import hashlib
        result = preview.run_replay(self.session, self.root / 'replays', lambda *_: None,
                                    self.cancel, realtime_preview=False)
        saved = json.loads((Path(result['output_dir']) / 'replay.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['source_directory'], str(self.raw.resolve()))
        self.assertEqual(saved['source_hashes']['seal.json'],
                         hashlib.sha256((self.raw / 'seal.json').read_bytes()).hexdigest())
        self.assertEqual(saved['source_hashes']['00000000.npy'],
                         hashlib.sha256((self.raw / '00000000.npy').read_bytes()).hexdigest())
        self.assertEqual(len(saved['source_identity']), 64)
        self.assertTrue(saved['quality_evidence_verified'])

    def test_uncompared_quality_field_tampering_must_not_pass_replay(self):
        path = self.raw / 'quality-v2.jsonl'
        rows = path.read_text(encoding='utf-8').splitlines()
        row = json.loads(rows[0]); row['updated_monotonic_s'] = -123
        rows[0] = json.dumps(row)
        path.write_text('\n'.join(rows) + '\n', encoding='utf-8')
        self.assertFalse(replay.replay_quality(self.raw)['online_replay_match'])

    def test_processing_manifest_tampering_must_not_pass_replay(self):
        path = next(self.raw.glob('processing-*.json'))
        manifest = json.loads(path.read_text(encoding='utf-8'))
        manifest['raw']['unit'] = 'volts'
        path.write_text(json.dumps(manifest), encoding='utf-8')
        self.assertFalse(replay.replay_quality(self.raw)['online_replay_match'])

    def test_missing_seal_is_not_verified_online_evidence(self):
        (self.raw / 'seal.json').unlink()
        result = replay.replay_quality(self.raw)
        self.assertFalse(result['online_replay_match'])
        self.assertTrue(result['raw_hashes_unchanged'])

    def test_bad_quality_rows_are_structured_mismatches_not_crashes(self):
        with (self.raw / 'quality-v2.jsonl').open('a', encoding='utf-8') as handle:
            handle.write('not json\nnull\n')
        result = replay.replay_quality(self.raw)
        self.assertFalse(result['online_replay_match'])
        self.assertTrue(result['mismatches'])

    def test_block_changed_after_recovery_is_rejected_before_consumption(self):
        recover = replay.recover_blocks
        def changed(*args, **kwargs):
            result = recover(*args, **kwargs)
            path = self.raw / result['records'][0]['file']
            data = np.load(path); data[1, 0] += 1
            np.save(path, data)
            return result
        shown = []
        with patch.object(replay, 'recover_blocks', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'integrity'):
                replay.replay_quality(self.raw, on_result=lambda *_: shown.append(True))
        self.assertEqual(shown, [])

    def test_reference_changed_during_replay_invalidates_verdict(self):
        def changed(*_):
            (self.raw / 'quality-v2.jsonl').write_text('null\n', encoding='utf-8')
        self.assertFalse(replay.replay_quality(self.raw, on_result=changed)['online_replay_match'])

    def test_new_command_file_during_report_is_source_change(self):
        recover = reporting.recover_blocks
        def changed(*args, **kwargs):
            result = recover(*args, **kwargs)
            (self.session / 'new-commands.jsonl').write_text(
                json.dumps({'type': 'command', 'accepted': True}) + '\n', encoding='utf-8')
            return result
        output = self.root / 'report'
        with patch.object(reporting, 'recover_blocks', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'source_changed'):
                reporting.build_session_report(self.session, output)
        self.assertFalse(output.exists())

    def test_cancellation_after_aggregation_leaves_no_report(self):
        signal = reporting._signal_updates
        def cancelled(*args, **kwargs):
            result = signal(*args, **kwargs)
            self.cancel.set()
            return result
        output = self.root / 'report'
        with patch.object(reporting, '_signal_updates', side_effect=cancelled):
            with self.assertRaises(InterruptedError):
                reporting.build_session_report(self.session, output, cancelled=self.cancel.is_set)
        self.assertFalse(output.exists())

    @staticmethod
    def fail_html_write(original):
        def open_file(path, mode='r', *args, **kwargs):
            if path.name == 'report.html' and any(flag in mode for flag in ('w', 'x')):
                raise OSError('injected disk failure')
            return original(path, mode, *args, **kwargs)
        return open_file

    def test_report_second_file_failure_never_publishes_half_bundle(self):
        output = self.root / 'report'
        with patch.object(Path, 'open', self.fail_html_write(Path.open)):
            with self.assertRaisesRegex(OSError, 'injected disk failure'):
                reporting.build_session_report(self.session, output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob('.report.*')), [])

    def test_export_second_file_failure_never_publishes_half_bundle(self):
        generated = preview.run_report(self.session, self.root / 'generated', self.cancel)
        output = self.root / 'export'
        with patch.object(preview, 'new_output', return_value=output), \
             patch.object(Path, 'open', self.fail_html_write(Path.open)):
            with self.assertRaisesRegex(OSError, 'injected disk failure'):
                preview.export_report(self.session, generated, self.root, self.cancel)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob('.export.*')), [])

    def test_export_does_not_copy_bytes_changed_after_validation(self):
        generated = preview.run_report(self.session, self.root / 'generated', self.cancel)
        path = Path(generated['output_dir']) / 'report.html'
        def changed(report):
            text = render_report_html(report)
            path.write_text('changed after validation', encoding='utf-8')
            return text
        with patch('neurostation_quality_report.render_report_html', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'report_changed'):
                preview.export_report(self.session, generated, self.root / 'export', self.cancel)

    def test_export_checks_cancellation_before_publication(self):
        generated = preview.run_report(self.session, self.root / 'generated', self.cancel)
        def cancelled(report):
            self.cancel.set()
            return render_report_html(report)
        with patch('neurostation_quality_report.render_report_html', side_effect=cancelled):
            with self.assertRaises((InterruptedError, preview.PreviewCancelled)):
                preview.export_report(self.session, generated, self.root / 'export', self.cancel)


class QualityAuditGateTests(unittest.TestCase):
    def record(self, **changes):
        row = {'schema_version': 2, 'processing_id': 'p', 'stream_id': 's',
               'start': 0, 'stop': 1250, 'updated_monotonic_s': 10.,
               'dimensions': {key: 'pass' for key in
                              ('integrity', 'signal_quality', 'coverage', 'stimulus_timing')}}
        return {**row, **changes}

    def test_malformed_quality_fails_closed_without_raising(self):
        for changes in ({'dimensions': None}, {'dimensions': []}, {'stop': None},
                        {'start': '0'}, {'start': -1}, {'start': False}, {'stop': 1250.5},
                        {'processing_id': []}, {'processing_id': ''}, {'stream_id': {}},
                        {'updated_monotonic_s': True}, {'updated_monotonic_s': 10 ** 999}, {'generation': -1}):
            with self.subTest(changes=changes):
                gate = QualityCommandGate(clock=lambda: 10.)
                gate.observe(self.record(**changes))
                self.assertFalse(gate.assess({'sample_start': 0, 'sample_end': 1249})['accepted'])

    def test_window_fractional_boolean_or_string_indices_are_not_coerced(self):
        for window in ({'sample_start': .25, 'sample_end': 1249.25},
                       {'sample_start': False, 'sample_end': 1249},
                       {'sample_start': '0', 'sample_end': '1249'}, None):
            with self.subTest(window=window):
                gate = QualityCommandGate(clock=lambda: 10.)
                gate.observe(self.record())
                self.assertFalse(gate.assess(window)['accepted'])

    def test_invalid_update_invalidates_pending_and_does_not_rearm_select(self):
        gate = QualityCommandGate(clock=lambda: 10.)
        gate.observe(self.record())
        self.assertEqual(gate.latch('SELECT')[0], 'SELECT')
        self.assertTrue(gate.observe({'schema_version': 2, 'dimensions': None}))
        self.assertFalse(gate.assess({'sample_start': 0, 'sample_end': 1249})['accepted'])
        gate.observe(self.record())
        self.assertEqual(gate.latch('SELECT')[0], 'UNKNOWN')

    def test_later_overlapping_fail_must_not_be_hidden_by_earlier_pass(self):
        gate = QualityCommandGate(clock=lambda: 10.)
        gate.observe(self.record())
        failed = self.record(start=500, stop=750)
        failed['dimensions']['signal_quality'] = 'fail'
        gate.observe(failed)
        self.assertFalse(gate.assess({'sample_start': 0, 'sample_end': 1249})['accepted'])

    def test_bad_clock_input_never_raises(self):
        gate = QualityCommandGate(clock=lambda: 10.)
        gate.observe(self.record())
        for value in (None, 'invalid', True, float('nan'), float('inf'), 10 ** 999):
            # None explicitly uses the injected clock, per public API.
            if value is None:continue
            with self.subTest(now=value):
                self.assertFalse(gate.assess({'sample_start': 0, 'sample_end': 1249}, now=value)['accepted'])

class QualityPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_concurrent_publish_has_one_complete_winner(self):
        from concurrent.futures import ThreadPoolExecutor
        from eeg_tools.quality.evidence import publish_bundle
        barrier = threading.Barrier(2)
        output = self.root / 'published'
        def publish(value):
            try:
                publish_bundle(output, {'one': value, 'two': value}, validate=lambda: barrier.wait(5))
                return 'ok'
            except FileExistsError:
                return 'exists'
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(publish, (b'a', b'b')))
        self.assertEqual(sorted(results), ['exists', 'ok'])
        self.assertEqual((output / 'one').read_bytes(), (output / 'two').read_bytes())
        self.assertEqual(list(self.root.glob('.published.*')), [])

    def test_existing_destination_is_not_overwritten(self):
        from eeg_tools.quality.evidence import publish_bundle
        output = self.root / 'published'
        output.mkdir(); (output / 'old').write_bytes(b'original')
        with self.assertRaises(FileExistsError):
            publish_bundle(output, {'old': b'changed'})
        self.assertEqual((output / 'old').read_bytes(), b'original')

    def test_cancel_at_final_validation_does_not_publish(self):
        from eeg_tools.quality.evidence import publish_bundle
        cancel = threading.Event()
        output = self.root / 'published'
        with self.assertRaises(InterruptedError):
            publish_bundle(output, {'one': b'complete'}, cancelled=cancel.is_set, validate=cancel.set)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_abrupt_process_exit_leaves_only_an_explicit_temporary_directory(self):
        import subprocess
        import sys
        output = self.root / 'published'
        code = ('import os,sys; from eeg_tools.quality.evidence import publish_bundle; '
                "publish_bundle(sys.argv[1], {'one':b'finished'}, validate=lambda:os._exit(73))")
        run = subprocess.run([sys.executable, '-B', '-c', code, str(output)],
                             cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=30)
        self.assertEqual(run.returncode, 73, run.stderr)
        self.assertFalse(output.exists())
        remaining = list(self.root.iterdir())
        self.assertEqual(len(remaining), 1)
        self.assertTrue(remaining[0].name.startswith('.published.') and remaining[0].name.endswith('.tmp'))
