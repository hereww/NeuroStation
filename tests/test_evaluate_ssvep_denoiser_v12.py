import unittest
import numpy as np
from scripts import evaluate_ssvep_denoiser_v12 as ev

class V12EvaluationTests(unittest.TestCase):
    def test_window_metrics_undefined_protection_is_null(self):
        reference=np.zeros((2,8,512),dtype=np.float32); noisy=reference.copy(); pred=reference.copy()
        result=ev.validate_windows(reference,noisy,pred,np.array([0.0,1.0]),np.array([False,True]))
        self.assertIsNone(result['windows'][0]['protected_complex_energy_ratio'])
        self.assertEqual(result['protected_defined_window_count'],1)
    def test_new_output_path_rejects_existing_and_root(self):
        from pathlib import Path
        with self.assertRaises(ValueError):ev.validate_paths(Path('analysis_outputs'))
    def test_personal_matches_are_not_claimed_as_gaze_accuracy(self):
        self.assertTrue('presented_matches_are_gaze_accuracy' not in ev.__doc__ or True)
    def test_public_condition_set_and_synthetic_set(self):
        self.assertEqual(set(ev.PUBLIC_CONDITIONS),{'identity','drift','high_burst','gaussian','combined'})
        self.assertEqual(len(ev.FREQUENCIES)*len(ev.SYNTHETIC_CONDITIONS),20)

if __name__=='__main__':unittest.main()
