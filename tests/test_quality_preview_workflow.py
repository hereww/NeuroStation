from pathlib import Path
import hashlib
import json
import tempfile
import threading
import unittest

from eeg_tools.quality.preview import (run_demo, run_replay, run_report, export_report,
                                       session_path)
from eeg_tools.quality.replay import replay_quality
from eeg_tools.streaming import recover_blocks
from neurostation_quality_report import render_report_html


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


class QualityPreviewWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cancel = threading.Event()

    def tearDown(self):
        self.temp.cleanup()

    def demo(self, seconds=2, publish=lambda *_: None):
        return Path(run_demo(self.root / 'output', publish, self.cancel,
                             seconds=seconds, realtime=False)['session_dir'])

    def test_same_engine_demo_replay_report_export_preserves_originals(self):
        states = []
        source = self.demo(12, lambda result, *_: states.append(result['dimensions']['signal_quality']))
        before = hashes(source)
        meta = json.loads((source / 'session.json').read_text())
        self.assertEqual(meta['data_origin'], 'synthetic_demo')
        self.assertFalse(meta['hardware_opened'] or meta['stimulus_started'] or meta['model_used'])
        self.assertIn('fail', states)  # Known flat channel, not an invented amplitude threshold.
        self.assertNotIn('pass', states)
        result = run_replay(source, self.root / 'output', lambda *_: None,
                            self.cancel, realtime_preview=False)
        self.assertTrue(result['verification']['online_replay_match'])
        self.assertTrue(result['verification']['raw_hashes_unchanged'])
        self.assertTrue(result['verification']['recording_complete'])
        generated = run_report(source, self.root / 'output', self.cancel)
        self.assertTrue(generated['report']['source_unchanged'])
        signal = generated['report']['signal_quality_updates']
        self.assertEqual(signal['all'], sum(signal['counts'].values()))
        self.assertGreater(signal['counts']['fail'], 0)
        self.assertTrue(signal['seal_evidence_match'])
        self.assertIn('CH2', signal['abnormal_channel_updates'])
        self.assertIsNone(generated['report']['trials']['rates']['acceptance']['value'])
        exported = export_report(source, generated, self.root / 'export', self.cancel)
        self.assertTrue((Path(exported['export_dir']) / 'report.html').is_file())
        self.assertEqual(before, hashes(source))

    def test_cancel_seals_only_known_prefix_as_aborted(self):
        def stop(_result, done, total):
            if done >= 100:
                self.cancel.set()
        source = self.demo(30, stop)
        meta = json.loads((source / 'session.json').read_text())
        self.assertEqual(meta['recording_status'], 'aborted')
        self.assertEqual(meta['sample_count'], 100)
        recovered = recover_blocks(source / 'raw_blocks')
        self.assertFalse(recovered['complete'])
        self.assertFalse(recovered['issues'])
        self.assertTrue(replay_quality(source / 'raw_blocks')['online_replay_match'])

    def test_corrupted_block_is_rejected_before_preview(self):
        source = self.demo()
        block = source / 'raw_blocks/00000000.npy'
        block.write_bytes(block.read_bytes()[:-1] + b'x')
        shown = []
        with self.assertRaisesRegex(ValueError, 'integrity'):
            run_replay(source, self.root / 'output', lambda *args: shown.append(args), self.cancel)
        self.assertEqual(shown, [])

    def test_legacy_report_does_not_invent_replay_or_accuracy(self):
        source = self.root / 'legacy'
        source.mkdir()
        (source / 'session.json').write_text(json.dumps({'session_id':'<script>bad</script>', 'status':'aborted', 'trial_count':3}))
        before = hashes(source)
        with self.assertRaisesRegex(ValueError, 'legacy_no_blocks'):
            run_replay(source, self.root / 'output', lambda *_: None, self.cancel)
        generated = run_report(source, self.root / 'output', self.cancel)
        report = generated['report']
        self.assertEqual(report['raw_blocks']['status'], 'legacy_no_block_journal')
        self.assertEqual(report['trials']['counts']['planned_without_events'], 3)
        self.assertEqual(report['trials']['counts']['all'], 0)
        self.assertIsNone(report['trials']['rates']['acceptance']['value'])
        text = render_report_html(report)
        self.assertNotIn('<script>bad</script>', text)
        self.assertIn('&lt;script&gt;bad', text)
        self.assertIn('无法计算', text)
        self.assertEqual(before, hashes(source))

    def test_outputs_inside_source_are_refused(self):
        source = self.demo()
        before = hashes(source)
        with self.assertRaisesRegex(ValueError, 'output_in_source'):
            run_report(source, source / 'extra', self.cancel)
        self.assertEqual(before, hashes(source))

    def test_exports_are_unique_and_tampered_report_is_refused(self):
        source = self.demo()
        generated = run_report(source, self.root / 'output', self.cancel)
        one = export_report(source, generated, self.root / 'export', self.cancel)
        two = export_report(source, generated, self.root / 'export', self.cancel)
        self.assertNotEqual(one['export_dir'], two['export_dir'])
        (Path(generated['output_dir']) / 'report.html').write_text('modified')
        with self.assertRaisesRegex(ValueError, 'report_changed'):
            export_report(source, generated, self.root / 'export', self.cancel)

    def test_raw_block_selection_resolves_session_and_random_directory_refused(self):
        source = self.demo()
        self.assertEqual(session_path(source / 'raw_blocks'), source)
        with self.assertRaisesRegex(ValueError, 'not_session'):
            session_path(self.root)

    def test_cancelled_replay_has_no_success_output(self):
        source = self.demo()
        def cancel_after_one(*_):
            self.cancel.set()
        with self.assertRaises(InterruptedError):
            run_replay(source, self.root / 'new', cancel_after_one, self.cancel, realtime_preview=False)
        self.assertFalse((self.root / 'new').exists())

    def test_mismatch_is_not_converted_to_pass(self):
        source = self.demo()
        log = source / 'raw_blocks/quality-v2.jsonl'
        rows = log.read_text().splitlines()
        row = json.loads(rows[0]); row['display_sha256'] = 'wrong'
        rows[0] = json.dumps(row)
        log.write_text('\n'.join(rows) + '\n')
        result = run_replay(source, self.root / 'output', lambda *_: None,
                            self.cancel, realtime_preview=False)
        self.assertFalse(result['verification']['online_replay_match'])
        self.assertTrue(result['verification']['raw_hashes_unchanged'])

    def test_quality_log_tamper_is_visible_and_bad_lines_not_dropped(self):
        source = self.demo()
        journal = source / 'raw_blocks/quality-v2.jsonl'
        with journal.open('a', encoding='utf-8') as f:
            f.write('not json\n')
        result = run_report(source, self.root / 'output', self.cancel)['report']
        signal = result['signal_quality_updates']
        self.assertEqual(signal['all'], 11)
        self.assertEqual(signal['counts']['unassessable'], 1)
        self.assertFalse(signal['seal_evidence_match'])
        self.assertEqual(sum(signal['counts'].values()), signal['all'])
        self.assertIn('质量日志封存未核验或不匹配', render_report_html(result))
