import unittest
import numpy as np
import torch
from scripts import evaluate_ssvep_denoiser_v11 as evaluation
from eeg_tools.ssvep_v11_runtime import build_model

class V11EvaluationTests(unittest.TestCase):
    def test_renames_only_track_keys_and_preserves_numerical_evidence(self):
        result=evaluation.renamed({'model_version':'learned_denoise_v11_personal','nested':{'v10':{'mse':0.4}},'array':[{'v10':1}]})
        self.assertEqual(result['nested']['v11']['mse'],0.4)
        self.assertEqual(result['array'][0]['v11'],1)
        self.assertEqual(result['model_version'],'learned_denoise_v11_personal')
    def test_true_frequency_amplitude_phase_evaluation_separates_harmonics(self):
        torch.set_num_threads(2)
        model=build_model().eval()
        with torch.no_grad():model.encoder[-1].bias.fill_(-100)
        rows=evaluation.waveform_preservation({},model)
        self.assertEqual([row['frequency_hz'] for row in rows],[10,12,15,20])
        self.assertTrue(all(row['ground_truth_used_for_evaluation_only'] for row in rows))
        self.assertTrue(all(row['maximum_absolute_amplitude_ratio_change']<.02 for row in rows))
        self.assertTrue(all(row['maximum_absolute_phase_difference_radians']<.02 for row in rows))

if __name__=='__main__':unittest.main()
