import time
import unittest
from eeg_tools.quality.commands import QualityCommandGate

class QualityCommandTests(unittest.TestCase):
    def filled(self):
        gate=QualityCommandGate(clock=lambda:10.)
        for start in range(0,1250,250):
            gate.observe({'schema_version':2,'processing_id':'p','stream_id':'s','start':start,'stop':start+250,
                          'updated_monotonic_s':10.,'dimensions':{k:'pass' for k in ('integrity','signal_quality','coverage','stimulus_timing')}})
        return gate
    def test_complete_window_and_processing_id(self):
        g=self.filled(); w={'sample_start':0,'sample_end':1249}
        self.assertTrue(g.assess(w,processing_id='p')['accepted'])
        self.assertEqual(g.assess(w,processing_id='wrong')['reason'],'processing_identity_mismatch')
        self.assertEqual(g.assess(w,now=12.01)['reason'],'quality_stale')
    def test_unknown_missing_and_overlap_fail_closed(self):
        g=self.filled();g.records[1]['dimensions']['signal_quality']='unknown'
        self.assertFalse(g.assess({'sample_start':0,'sample_end':1249})['accepted'])
        g=self.filled();g.records.popleft()
        self.assertFalse(g.assess({'sample_start':0,'sample_end':1249})['accepted'])
        g.observe(None);self.assertFalse(g.assess({})['accepted'])
    def test_select_latch_reject_timeout_do_not_rearm(self):
        g=self.filled()
        self.assertEqual(g.latch('SELECT')[0],'SELECT')
        self.assertEqual(g.latch('SELECT'),('UNKNOWN','select_latched'))
        g.latch('UNKNOWN')
        self.assertEqual(g.latch('SELECT')[0],'UNKNOWN')
        g.latch('MOVE_LEFT');self.assertEqual(g.latch('SELECT')[0],'SELECT')
        g.reset_session();self.assertEqual(g.latch('SELECT')[0],'SELECT')
