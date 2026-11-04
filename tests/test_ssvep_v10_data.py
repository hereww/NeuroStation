import unittest
import numpy as np
from eeg_tools import ssvep_v10_data as data

class V10DataTests(unittest.TestCase):
    def setUp(self):
        self.cfg=data.load_config()
        self.reference=data.synthetic_references(12,42)
    def test_synthetic_known_reference_normalization_and_seed(self):
        np.testing.assert_array_equal(self.reference,data.synthetic_references(12,42))
        np.testing.assert_allclose(self.reference.mean(axis=-1),0,atol=1e-6)
        np.testing.assert_allclose(np.sqrt((self.reference**2).mean(axis=-1)),1,atol=1e-6)
    def test_corruptions_deterministic_and_labels_absent(self):
        for condition in self.cfg['augmentation']['conditions']:
            first=data.corrupt(self.reference,condition,13,self.cfg)
            np.testing.assert_array_equal(first,data.corrupt(self.reference,condition,13,self.cfg))
            self.assertEqual(first.shape,self.reference.shape)
            self.assertTrue(np.isfinite(first).all())
            if condition=='identity':np.testing.assert_array_equal(first,self.reference)
            else:self.assertGreater(float(np.mean((first-self.reference)**2)),0)
    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):data.synthetic_references(0,1)
        with self.assertRaises(ValueError):data.corrupt(self.reference,'fake',1,self.cfg)
        with self.assertRaises(ValueError):data.corrupt(np.zeros((2,7,512)),'drift',1,self.cfg)
    def test_notch_conditioner_keeps_offset_and_reduces_artificial_line(self):
        t=np.arange(2000)/250
        signal=np.tile((np.sin(2*np.pi*12*t)+2*np.sin(2*np.pi*50*t))[:,None],(1,8))+100
        result=data.condition_line(signal)
        ref=np.tile(np.sin(2*np.pi*12*t)[:,None],(1,8))+100
        self.assertLess(float(np.mean((result[250:-250]-ref[250:-250])**2)),0.01)
        self.assertEqual(signal.shape,result.shape)
    def test_import_remains_lightweight(self):
        import subprocess,sys
        result=subprocess.run([sys.executable,'-c',
            'import sys;from eeg_tools import ssvep_v10_data;assert "numpy" not in sys.modules;assert "torch" not in sys.modules'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
