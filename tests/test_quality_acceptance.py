"""Fault and denominator cases use temporary synthetic files, never EEG devices."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
from eeg_tools.streaming import BlockRecorder, recover_blocks, AsyncConsumer
from eeg_tools.quality.cache import writer_lock
from eeg_tools.quality.engine import QualityEngine
from eeg_tools.quality.reporting import build_session_report
from eeg_tools.quality.statistics import build_trial_ledger, summarize_trials

ROOT = Path(__file__).resolve().parents[1]

class QualityAcceptanceTests(unittest.TestCase):
    def test_planned_trial_with_no_events_and_session_event_not_disappearing(self):
        events = [{'trial_index': -1, 'event_name': 'session_start'},
                  {'trial_index': 0, 'event_name': 'trial_aborted'},
                  {'trial_index': 1, 'event_name': 'stimulus_onset', 'sample_index': 0},
                  {'trial_index': 1, 'event_name': 'stimulus_offset', 'sample_index': 20}]
        ledger = build_trial_ledger(events, [], 20, planned_trial_count=3)
        report = summarize_trials(ledger)
        self.assertEqual(report['counts']['all'], 2)
        self.assertEqual(report['counts']['ledger_entries'], 3)
        self.assertEqual(report['counts']['observed'], 2)
        self.assertEqual(report['counts']['planned_without_events'], 1)
        self.assertEqual(report['counts']['unassessable'], 2)
        self.assertEqual({r['reason'] for r in ledger}, {'aborted', 'scoring_failed', 'planned_trial_without_events'})
        self.assertEqual(report['rates']['acceptance']['denominator'], 2)

    def test_unlabeled_trial_not_in_labeled_accuracy_denominator(self):
        rows = [{'quality': 'pass', 'decision': 'accepted', 'scoring_status': 'scored',
                 'presented_target_id': None, 'gaze_target_id': None, 'predicted_target_id': 'a'}]
        report = summarize_trials(rows)
        self.assertEqual(report['counts']['model_accepted'], 1)
        self.assertIsNone(report['rates']['presented_target_id_quality_scored_accuracy']['value'])
        self.assertIsNone(report['rates']['gaze_target_id_quality_scored_accuracy']['value'])
        self.assertEqual(summarize_trials([{}])['counts']['unassessable'], 1)

    def test_ndjson_commands_and_unreadable_record_are_counted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'session.json').write_text('{"trial_count":2}', encoding='utf-8')
            (root/'commands.ndjson').write_text('\n'.join([
                json.dumps({'type':'command', 'accepted':True, 'command':'SELECT'}),
                json.dumps({'type':'command', 'accepted':False, 'command':'UNKNOWN'}),
                '{truncated'])+'\n', encoding='utf-8')
            report = build_session_report(root)['report']
            self.assertEqual(report['commands']['all'], 3)
            self.assertEqual(report['commands']['accepted'], 1)
            self.assertEqual(report['commands']['acceptance']['denominator'], 3)
            self.assertIn('commands.ndjson', report['source_hashes'])
            self.assertEqual(report['trials']['counts']['planned_without_events'], 2)

    def test_abrupt_child_exit_recovers_only_durable_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            code = """
import os,sys
from pathlib import Path
import numpy as np
from eeg_tools.streaming import BlockRecorder
r=BlockRecorder(Path(sys.argv[1])/'raw_blocks',session_id='fixture',channel_names=['O1'],eeg_rows=[1],package_row=0,timestamp_row=2)
r.append(np.vstack((np.arange(100)%256,np.zeros(100),100+np.arange(100)/250)))
with (r.directory/'blocks.jsonl').open('a',encoding='utf-8') as f:
    f.write('{interrupted');f.flush();os.fsync(f.fileno())
os._exit(17)
"""
            result = subprocess.run([sys.executable, '-B', '-c', code, temp], cwd=ROOT,
                                    capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 17, result.stderr.decode(errors='replace'))
            report = recover_blocks(Path(temp)/'raw_blocks')
            self.assertFalse(report['complete'])
            self.assertEqual(report['verified_samples'], 100)
            self.assertTrue(report['issues'])
            self.assertFalse((Path(temp)/'raw_blocks/seal.json').exists())

    def test_os_releases_writer_lock_on_abrupt_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp)/'cache.lock'
            code = "from eeg_tools.quality.cache import writer_lock; import os,sys\nwith writer_lock(sys.argv[1]): os._exit(19)"
            result = subprocess.run([sys.executable,'-B','-c',code,str(lock)], cwd=ROOT,
                                    capture_output=True, timeout=30)
            self.assertEqual(result.returncode,19,result.stderr.decode(errors='replace'))
            with writer_lock(lock): pass

    def test_orphan_block_is_not_a_complete_recording(self):
        with tempfile.TemporaryDirectory() as temp:
            r=BlockRecorder(Path(temp)/'raw_blocks',session_id='fixture',channel_names=['O1'],eeg_rows=[1],package_row=0,timestamp_row=2)
            r.append(np.array([[0.],[1.],[100.]]));r.close()
            np.save(r.directory/'00000001.npy',np.ones((3,1)))
            report=recover_blocks(r.directory)
            self.assertFalse(report['complete'])
            self.assertEqual(report['uncommitted_files'],['00000001.npy'])

    def test_assessed_watermark_does_not_jump_over_quality_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            r=BlockRecorder(Path(temp)/'raw_blocks',session_id='fixture',channel_names=['O1'],eeg_rows=[1],package_row=0,timestamp_row=2)
            r.append(np.array([[0.],[1.],[100.]]),lambda *_: (_ for _ in ()).throw(ValueError('quality failed')))
            r.append(np.array([[1.],[1.],[100.004]]),lambda *_:None)
            r.close()
            self.assertEqual(r.durable,2);self.assertEqual(r.assessed,0)

    def test_configuration_change_resets_filter_and_metrics(self):
        engine=QualityEngine(['O1'])
        raw=np.sin(np.arange(750)/10)[None,:]
        record={'stream_id':'s','generation':0,'start':0,'stop':750,'configuration_id':'first'}
        before=engine.process(raw,record)
        record.update(start=750,stop=1500,configuration_id='second')
        after=engine.process(raw,record)
        fresh=QualityEngine(['O1']).process(raw,record)
        self.assertNotEqual(before['processing_id'],after['processing_id'])
        self.assertEqual(after['display_sha256'],fresh['display_sha256'])
        self.assertFalse(any(after['display_valid'][0][:250]))

    def test_blocked_quality_queue_cannot_block_raw_and_drops_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            r=BlockRecorder(Path(temp)/'raw_blocks',session_id='fixture',channel_names=['O1'],eeg_rows=[1],package_row=0,timestamp_row=2)
            entered=threading.Event();release=threading.Event()
            def slow(*_):entered.set();release.wait(timeout=5)
            consumer=AsyncConsumer(r,slow,max_pending=1)
            try:
                r.append(np.array([[0.],[1.],[100.]]),consumer)
                self.assertTrue(entered.wait(timeout=1))
                began=time.monotonic()
                for i in range(1,6):r.append(np.array([[float(i)],[1.],[100+i/250]]),consumer)
                self.assertLess(time.monotonic()-began,1.0)
                self.assertEqual(r.durable,6);self.assertEqual(r.assessed,0)
                self.assertGreater(consumer.dropped,0)
                self.assertFalse(consumer.close(timeout=.01))
                self.assertTrue(any('queue_full' in e['error'] for e in r.consumer_errors))
            finally:
                release.set();self.assertTrue(consumer.close(timeout=2));r.close()
            self.assertTrue(recover_blocks(r.directory)['complete'])
            self.assertLess(r.assessed,r.durable)
