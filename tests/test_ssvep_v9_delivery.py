import copy
import unittest
from unittest.mock import patch
from eeg_tools import ssvep_test_runtime as rt
from eeg_tools import ssvep_v9_delivery as d

class DeliveryAuditTests(unittest.TestCase):
    def setUp(self):
        self.r={'model_version':'learned_denoise_v9_ssvep','weights_retrained_this_version':False,'base_checkpoint_sha256':rt.BASE_SHA,
                'inference_lock_sha256':'a'*64,'config_sha256':'a'*64,'model_state_sha256_before':'x','model_state_sha256_after':'x',
                'scope':{'subjects':[5,6],'outer_validation_waveforms_loaded':False,'holdout_waveforms_loaded':False,
                        'D_drive_accessed_this_run':False,'recorded_reference_is_clean_ground_truth':False,
                        'same_seeds_as_v8_paired_comparison':True,'independent_validation':False},
                'deployment':{'runtime_enabled':False,'production_validated':False,'scope':'explicit_offline_test_only'},
                'cases':[{}], 'case_count':4200,'all_internal_gates_passed':False}
    def audited(self,r):
        with patch.object(rt,'read_bytes',return_value=b'{}'),patch.object(rt,'sha',return_value='a'*64),\
             patch('eeg_tools.ssvep_v8_internal_eval.summarize_diagnostic',return_value={'case_count':4200,'all_internal_gates_passed':False}):
            return d.audit_report(r)
    def test_failure_preserved_not_relabelled(self):
        r=self.audited(self.r);self.assertTrue(r['audit_passed']);self.assertFalse(r['production_acceptance'])
    def test_mutated_aggregation_refused(self):
        self.r['all_internal_gates_passed']=True
        with self.assertRaises(ValueError):self.audited(self.r)
    def test_mutated_count_refused(self):
        self.r['case_count']=4199
        with self.assertRaises(ValueError):self.audited(self.r)
    def test_false_training_claim_refused(self):
        self.r['weights_retrained_this_version']=True
        with self.assertRaises(ValueError):self.audited(self.r)
    def test_runtime_holdout_and_clean_truth_flags_refused(self):
        for container,key in [('deployment','runtime_enabled'),('deployment','production_validated'),
                              ('scope','holdout_waveforms_loaded'),('scope','independent_validation'),('scope','recorded_reference_is_clean_ground_truth')]:
            r=copy.deepcopy(self.r);r[container][key]=True
            with self.subTest(key=key),self.assertRaises(ValueError):self.audited(r)
    def test_base_hash_or_model_state_mutation_refused(self):
        for key in ('base_checkpoint_sha256','model_state_sha256_after','model_version','config_sha256','inference_lock_sha256'):
            r=copy.deepcopy(self.r);r[key]='wrong'
            with self.subTest(key=key),self.assertRaises(ValueError):self.audited(r)
    def test_build_action_required(self):self.assertEqual(d.main([]),1)
    def test_multiple_build_actions_refused(self):self.assertEqual(d.main(['--run','--check-only']),1)

if __name__=='__main__':unittest.main()
