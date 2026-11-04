import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from eeg_tools.streaming import BlockRecorder
from eeg_tools.quality.recording import QualitySink
from eeg_tools.quality.replay import replay_quality
from eeg_tools.quality.reporting import build_session_report

class ReplayTests(unittest.TestCase):
    def test_replay_matches_and_raw_stays_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);r=BlockRecorder(root/'raw_blocks',session_id='fixture',channel_names=['O1'],eeg_rows=[1],package_row=0,timestamp_row=2)
            sink=QualitySink(r.directory,[1],['O1'])
            n=3000;t=np.arange(n)/250
            data=np.vstack((np.arange(n)%256,30*np.sin(2*np.pi*10*t),100+t))
            data[1,1444]=np.nan;data[0,1899]=0
            for start in range(0,n,137):r.append(data[:,start:start+137],sink)
            sink.close();r.close()
            report=replay_quality(r.directory)
            self.assertTrue(report['online_replay_match'],report['mismatches'])
            self.assertTrue(report['raw_hashes_unchanged']);self.assertEqual(report['verified_samples'],n)
    def test_legacy_report_no_trial_denominator_is_not_accuracy(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'session.json').write_text('{"session_id":"old","sample_count":200}',encoding='utf-8')
            result=build_session_report(root)
            self.assertEqual(result['report']['trials']['counts']['all'],0)
            self.assertIsNone(result['report']['commands']['acceptance']['value'])
            self.assertTrue((Path(result['output_dir'])/'report.html').is_file())
