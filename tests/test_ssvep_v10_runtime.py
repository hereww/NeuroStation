import unittest
import numpy as np
import torch
from eeg_tools.ssvep_denoise_v10 import build_model
from eeg_tools.ssvep_v10_runtime import process_recording

class V10RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model=build_model().eval()
        self.signal=np.random.default_rng(123).standard_normal((1523,8))+np.arange(8)*1000
    def test_full_coverage_reconstruction_shape_and_finite(self):
        result,notch,receipt=process_recording(self.signal,self.model)
        self.assertEqual(result.shape,self.signal.shape)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(len(self.signal),receipt['processed_samples'])
        self.assertEqual(0,receipt['preserved_samples'])
        self.assertFalse(receipt['whole_record_protected_spectrum_guaranteed'])
        self.assertLess(receipt['max_window_protected_complex_energy_ratio'],1e-6)
    def test_zero_attenuation_is_exact_conditioned_overlap_add_with_tail(self):
        with torch.no_grad():
            self.model.encoder[-1].bias.fill_(-100)
        result,notch,receipt=process_recording(self.signal,self.model,batch_size=3)
        np.testing.assert_allclose(result,notch,rtol=1e-7,atol=2e-6)
        self.assertEqual(9,receipt['window_count'])
    def test_invalid_nonfinite_flat_shape_rejected(self):
        for signal in (self.signal[:511],np.zeros((600,8)),self.signal[:,:7],self.signal.astype(np.int32)):
            with self.assertRaises(ValueError):process_recording(signal,self.model)
        changed=self.signal.copy();changed[0,0]=np.nan
        with self.assertRaises(ValueError):process_recording(changed,self.model)

if __name__=='__main__':unittest.main()
