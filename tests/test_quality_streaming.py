import csv
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from eeg_tools.streaming import BlockRecorder, recover_blocks, AcquisitionOwner
from eeg_tools.quality.events import associate_events

class StreamingTests(unittest.TestCase):
    def data(self, n=250, start=0):
        return np.vstack(((np.arange(n)+start)%256, np.sin(np.arange(n)/7), (np.arange(n)+start)/250+100))
    def recorder(self, root):
        return BlockRecorder(Path(root)/'raw_blocks',session_id='s',channel_names=['O1'],eeg_rows=[1],package_row=0,timestamp_row=2)
    def test_wrap_and_original_content(self):
        with tempfile.TemporaryDirectory() as root:
            r=self.recorder(root); raw=self.data(700)
            for i in range(0,700,93): r.append(raw[:,i:i+93])
            r.close(); report=recover_blocks(r.directory)
            self.assertTrue(report['complete']); self.assertEqual(r.generation,0)
            saved=np.concatenate([np.load(r.directory/b['file']) for b in report['records']],axis=1)
            np.testing.assert_array_equal(saved,raw)
    def test_gaps_duplicates_restart_timestamp_burst_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            r=self.recorder(root); a=self.data(12); a[0,3]=a[0,2]; a[0,6]=0; a[2,9]=a[2,8]
            r.append(a); r.close(); report=recover_blocks(r.directory)
            self.assertEqual(report['verified_samples'],12); self.assertGreater(r.generation,3)
            self.assertTrue(any('duplicate_packet' in x['gaps'] for x in report['records']))
    def test_consumer_failure_does_not_prevent_raw_save(self):
        with tempfile.TemporaryDirectory() as root:
            r=self.recorder(root)
            r.append(self.data(),lambda *_: (_ for _ in ()).throw(ValueError('quality failed')))
            r.close(); self.assertEqual(r.durable,250); self.assertEqual(r.assessed,0)
            self.assertEqual(len(r.consumer_errors),1)
    def test_disk_failure_stops_and_partial_prefix_recovers(self):
        with tempfile.TemporaryDirectory() as root:
            r=self.recorder(root); r.append(self.data())
            with patch('eeg_tools.streaming.np.save',side_effect=OSError('disk full')):
                with self.assertRaises(OSError): r.append(self.data(start=250))
            with self.assertRaises(RuntimeError): r.append(self.data())
            r.close(); report=recover_blocks(r.directory)
            self.assertFalse(report['complete']); self.assertEqual(report['verified_samples'],250)
    def test_truncated_journal_and_tamper(self):
        with tempfile.TemporaryDirectory() as root:
            r=self.recorder(root); r.append(self.data()); r.close()
            with (r.directory/'blocks.jsonl').open('a',encoding='utf-8') as f:f.write('{')
            self.assertFalse(recover_blocks(r.directory)['complete'])
            (r.directory/'00000000.npy').write_bytes(b'changed')
            self.assertEqual(recover_blocks(r.directory)['verified_samples'],0)
    def test_backpressure_is_reported(self):
        with tempfile.TemporaryDirectory() as root:
            r=self.recorder(root); ticks=iter([1,1,2]); r.clock=lambda:next(ticks)
            r.append(self.data(1)); r.close()
            self.assertTrue(json.loads((r.directory/'watermarks.json').read_text())['backpressure'])
    def test_unknown_event_time_is_not_zero(self):
        result=associate_events([{'marker_code':10,'event_id':1}],np.array([0,10]),np.array([1,np.nan]))
        self.assertEqual(result['events'][0]['sample_index'],1)
        self.assertIsNone(result['events'][0]['device_timestamp_s'])
        self.assertIsNone(result['events'][0]['physical_stimulus_onset_s'])
    def test_ambiguous_events_remain_unknown(self):
        r=associate_events([{'marker_code':10}],np.array([10,10]),None)
        self.assertEqual(r['events'][0]['association'],'unknown')
    def test_single_owner_no_competing_reads(self):
        with tempfile.TemporaryDirectory() as root:
            r=self.recorder(root); calls=[]; data=self.data()
            class Board:
                def get_board_data(self):
                    calls.append(threading.get_ident())
                    return data if len(calls)==1 else np.empty((3,0))
            owner=AcquisitionOwner(Board(),r); owner.start(); time.sleep(.08); owner.stop(); r.close()
            self.assertEqual(len(set(calls)),1); self.assertEqual(r.durable,250)
