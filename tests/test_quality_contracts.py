import tempfile
import unittest
from pathlib import Path
from eeg_tools.runtime_bindings import RootResolver, sha256
from eeg_tools.quality.statistics import build_trial_ledger, summarize_trials

class BaselineContracts(unittest.TestCase):
    def test_explicit_root_relocation_and_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); new=root/'new'; new.mkdir(); old=root/'old'; f=new/'weights'; f.write_bytes(b'ok')
            r=RootResolver([{'from':str(old),'to':str(new)}])
            self.assertEqual(r.verify({str(old/'weights'):sha256(f)}),{f.resolve():sha256(f)})
            with self.assertRaises(ValueError): r.verify({str(old/'weights'):'changed'})
            with self.assertRaises(ValueError): r.resolve(old/'absent')
            with self.assertRaises(ValueError): RootResolver([{'from':str(old),'to':str(new)},{'from':str(old/'nested'),'to':str(new)}])
    def test_trials_never_disappear(self):
        events=[{'trial_index':0,'event_name':'stimulus_onset'},
                {'trial_index':1,'event_name':'trial_aborted'},
                {'trial_index':2,'event_name':'stimulus_onset','sample_index':'0'},
                {'trial_index':2,'event_name':'stimulus_offset','sample_index':'100'}]
        ledger=build_trial_ledger(events,[],100)
        s=summarize_trials(ledger)
        self.assertEqual(s['counts']['all'],3)
        self.assertEqual(s['counts']['unassessable'],3)
        self.assertIsNone(s['rates']['presented_target_id_quality_scored_accuracy']['value'])
    def test_all_rejected_and_no_labels(self):
        s=summarize_trials([{'decision':'rejected','quality':'fail'}]*5)
        self.assertEqual(s['counts']['rejected'],5)
        self.assertEqual(s['rates']['acceptance']['denominator'],5)
        self.assertEqual(s['rates']['acceptance']['value'],0)
        self.assertIsNone(summarize_trials([])['rates']['acceptance']['value'])
