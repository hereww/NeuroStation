import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).absolute().parents[1]
spec=importlib.util.spec_from_file_location('personal_audit',ROOT/'scripts/audit_personal_eeg_dataset.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class PersonalAuditTests(unittest.TestCase):
    def test_packet_wrap_continuous(self):
        r=a.packet_summary(np.array([254.,255.,0.,1.]));self.assertEqual(r['expected_step_one'],3);self.assertEqual(r['missing_samples_modulo256_estimate'],0)
    def test_duplicate_and_gap_reported(self):
        r=a.packet_summary(np.array([1.,1.,4.]));self.assertEqual(r['duplicate_step_zero'],1);self.assertEqual(r['missing_samples_modulo256_estimate'],2)
    def test_invalid_packet_rows_not_bridged(self):
        r=a.packet_summary(np.array([1.,np.nan,4.,5.]));self.assertEqual(r['transitions'],1);self.assertEqual(r['invalid_packet_rows'],1)
    def test_no_input_mutation(self):
        rng=np.random.default_rng(1);d=np.zeros((512,11));d[:,0]=np.arange(512)%256;d[:,1:9]=rng.normal(size=(512,8));d[:,9]=np.arange(512)/250
        before=d.copy();r=a.signal_summary(d);np.testing.assert_array_equal(d,before);self.assertEqual(r['actual_samples'],512)
    def test_asymmetric_signed24bit_endpoints(self):
        y=np.array([-a.RAIL,a.POSITIVE_RAIL,0.,180000.])
        np.testing.assert_array_equal(a.rail_match(y),[True,True,False,False])
    def test_negative_rail_and_tail(self):
        x=np.random.default_rng(2).normal(size=(600,8));x[:,4]=-a.RAIL
        r=a.window_summary(x);self.assertEqual(r['full_windows'],2);self.assertEqual(r['flat_channel_windows'],2);self.assertEqual(r['incomplete_tail_samples'],88)
    def test_nonfinite_not_silently_flat(self):
        x=np.zeros((256,8));x[2,1]=np.nan;r=a.window_summary(x)
        self.assertEqual(r['nonfinite_windows'],1);self.assertEqual(r['flat_channel_windows'],0)
    def test_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):a.signal_summary(np.zeros((512,10)))
    def test_24_column_cyton_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.csv';raw=np.zeros((2,24));raw[:,0]=[1,2];raw[:,22]=[1234.,1234.004];raw[:,23]=[0.,3.];np.savetxt(p,raw,delimiter='\t')
            d,r=a.parse_with_review(p,'imported_openbci');self.assertEqual(r['malformed_row_count'],0);np.testing.assert_array_equal(d[:,9],raw[:,22]);np.testing.assert_array_equal(d[:,10],raw[:,23])
    def test_malformed_import_keeps_placeholder_and_raw_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.csv';row='\t'.join(['0']*24);text=row+'\n0\t1\n'+row+'\n';p.write_text(text);before=p.read_bytes()
            d,r=a.parse_with_review(p,'imported_openbci');self.assertEqual(len(d),3);self.assertEqual(r['malformed_rows_one_based'],[2]);self.assertTrue(np.isnan(d[1]).all());self.assertEqual(p.read_bytes(),before)
    def test_external_path_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'dataset';root.mkdir();p=Path(tmp)/'outside';p.write_text('x')
            with self.assertRaises(ValueError):a.checked(root,p)
    def test_classification_never_claims_training_ground_truth(self):
        r=a.classify({'kind':'imported_openbci','actual_count_matches_metadata':True,'signal':{'channels':[{'channel_index_one_based':5,'default_gain24_rail_fraction':.99}],
             'fixed_window_screen':{'flat_channel_windows':2},'all_eeg_zero_rows':0,'packet_sequence':{'nonunit_nonzero_transitions':0,'duplicate_step_zero':0}}})
        self.assertFalse(r['supervised_ssvep_training_ready']);self.assertFalse(r['clean_noisy_pair_available']);self.assertEqual(r['suspect_rail_channels'],[5])
    def test_nonfinite_json_not_written(self):
        with self.assertRaises(ValueError):a.safety.json_bytes({'invalid':float('nan')})

if __name__=='__main__':unittest.main()
