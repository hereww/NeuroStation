"""Safety/API contracts for independent single-channel v2, never public-test metrics."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from eeg_tools import learned_denoise_v2 as v2
from scripts import train_denoiser_v2 as trainer


class ConfigAndNormalizationTests(unittest.TestCase):
    def test_formal_offline_only_config_is_independent_and_mutable_copy(self):
        config = v2.load_config()
        self.assertEqual('learned_denoise_v2', config['model_version'])
        self.assertEqual('formal', config['status'])
        self.assertEqual('shadow', config['inference']['mode'])
        self.assertIs(False, config['runtime']['enabled'])
        self.assertIs(False, config['runtime']['production_validated'])
        self.assertEqual('not_assessed', config['runtime']['validation_status'])
        config['model']['kernel_size'] = 7
        self.assertEqual(5, v2.load_config()['model']['kernel_size'])
        actual = v2.load_config(Path(__file__).resolve().parents[1] / 'configs' / 'learned_denoiser_v2.json')
        self.assertEqual(500, actual['window_samples'])
        self.assertEqual(500, actual['hop_samples'])

    def test_strict_config_rejects_wrong_version_shape_nan_and_enablement(self):
        for section, key, value in ((None,'model_version','learned_denoise_v1'), (None,'window_samples',1000),
                                    (None,'hop_samples',250), (None,'sample_rate_hz',float('nan')),
                                    ('model','hidden_channels',48), ('model','kernel_size',3),
                                    ('model','residual_bound',float('inf')), ('runtime','enabled',True),
                                    ('runtime','production_validated',True), ('inference','max_invalid_fraction',.2)):
            config = v2.load_config()
            (config if section is None else config[section])[key] = value
            with self.subTest(section=section,key=key), self.assertRaises(v2.LearnedDenoiseV2Error):
                v2.validate_config(config)
        config = v2.load_config()
        config['unknown'] = 1
        with self.assertRaises(v2.LearnedDenoiseV2Error):
            v2.validate_config(config)

    def test_known_session_audit_keeps_public_pass_and_transfer_unassessed(self):
        actual = v2.load_config(Path(__file__).resolve().parents[1] / 'configs' / 'learned_denoiser_v2.json')
        runtime = actual['runtime']
        self.assertEqual('failed_known_session_regression', runtime['validation_status'])
        self.assertEqual('2026-10-05', runtime['validation_date'])
        self.assertEqual('passed_epoch_holdout', runtime['research_validation_status'])
        self.assertEqual('not_assessed_target_mapping', runtime['transfer_validation_status'])
        self.assertIn('non-independent', runtime['validation_reason'])
        self.assertIn('989.338', runtime['validation_reason'])
        self.assertEqual({'public_epoch_holdout', 'transfer_and_known_v1_regression'}, set(runtime['validation_reports']))
        self.assertIs(False, runtime['enabled'])
        self.assertIs(False, runtime['production_validated'])
        self.assertEqual('formal', actual['status'])
        self.assertEqual('shadow', actual['inference']['mode'])
        # The audit delta must not alter any training/inference configuration.
        self.assertEqual({k:v for k,v in v2.DEFAULT_CONFIG.items() if k != 'runtime'},
                         {k:v for k,v in actual.items() if k != 'runtime'})

    def test_all_validation_states_forbid_automatic_enablement(self):
        for status in ('not_assessed', 'failed_holdout', 'failed_known_session_regression'):
            config = v2.load_config()
            config['runtime']['validation_status'] = status
            for enabled in (False, True):
                config['runtime']['enabled'] = enabled
                with self.subTest(status=status, enabled=enabled):
                    if enabled:
                        with self.assertRaisesRegex(v2.LearnedDenoiseV2Error, 'enablement is forbidden'):
                            v2.validate_config(config)
                    else:
                        v2.validate_config(config)  # explicit diagnostics only

    def test_optional_audit_fields_are_strict_and_legacy_runtime_is_accepted(self):
        v2.validate_config(v2.load_config())  # original four-field runtime
        actual = v2.load_config(Path(__file__).resolve().parents[1] / 'configs' / 'learned_denoiser_v2.json')
        invalid = (
            ('validation_status', 'failed_unknown'),
            ('validation_date', '2026-10-32'),
            ('validation_date', '20261005'),
            ('validation_date', 20261005),
            ('validation_reason', ''),
            ('validation_reason', {'unsafe': 'nested value'}),
            ('research_validation_status', 'production_passed'),
            ('transfer_validation_status', 'accuracy_passed'),
            ('validation_reports', []),
            ('validation_reports', {'public_epoch_holdout': 'only-one-report'}),
            ('validation_reports', {'public_epoch_holdout': '', 'transfer_and_known_v1_regression': 'report'}),
            ('unknown_audit_key', 'not allowed'),
        )
        for key, value in invalid:
            config = deepcopy(actual)
            config['runtime'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(v2.LearnedDenoiseV2Error):
                v2.validate_config(config)

    def test_float64_noisy_mean_ac_rms_and_offset_normalization(self):
        signal = np.arange(500, dtype=np.float64) * .001 + 1e6
        normalized, center, scale = v2.normalize_window(signal)
        self.assertEqual(float(np.mean(signal)), center)
        self.assertAlmostEqual(float(np.sqrt(np.mean((signal-center)**2))), scale, places=13)
        self.assertEqual(np.dtype('float32'), normalized.dtype)
        np.testing.assert_allclose(normalized.astype(np.float64)*scale+center, signal, atol=2e-8, rtol=0)
        shifted, _, scale2 = v2.normalize_window(signal + 1e6)
        np.testing.assert_allclose(normalized, shifted, atol=1e-6, rtol=0)
        self.assertAlmostEqual(scale, scale2, places=8)

    def test_constant_and_invalid_normalization_rejected(self):
        for value in (np.ones(500)*1e6, np.full(500,np.nan), np.zeros((1,500))):
            with self.assertRaises(v2.LearnedDenoiseV2Error):
                v2.normalize_window(value)

    def test_empirical_threshold_strict_boundary_and_ties(self):
        for values in (np.linspace(0,1,100), np.ones(100), np.array([1.,.2,.3]), np.linspace(.1,.9,173)):
            threshold, metadata = trainer.calibrate_validation_threshold(values)
            self.assertLessEqual(float(np.mean(values > threshold)), .01)
            self.assertEqual('>', metadata['comparison'])
            self.assertEqual('validation_only', metadata['scope'])
            self.assertIs(False, metadata['production_calibrated'])
        _, typed_calibration = trainer.calibrate_validation_threshold(np.array([.2,.4]), np.float64(.01))
        self.assertTrue(v2.primitive_metadata(typed_calibration))
        self.assertIs(type(typed_calibration['max_clean_false_replacement_fraction']),float)
        threshold, _ = trainer.calibrate_validation_threshold(np.ones(500))
        self.assertEqual(1.0, threshold)
        for values in (np.array([]), np.array([np.nan]), np.array([1.1])):
            with self.assertRaises(v2.LearnedDenoiseV2Error):
                trainer.calibrate_validation_threshold(values)


if v2.torch is not None:
    class StubModel(v2.torch.nn.Module):
        def __init__(self, probability=.8, mode='good'):
            super().__init__()
            self.anchor = v2.torch.nn.Parameter(v2.torch.zeros(1))
            self.probability = probability
            self.mode = mode
            self.calls = 0

        def forward(self, signal, valid=None):
            self.calls += 1
            residual = .2 * v2.torch.sin(v2.torch.arange(500,device=signal.device)*.1)
            residual = residual - residual.mean()
            residual = residual[None,None].expand_as(signal).clone()
            p = v2.torch.full_like(signal, self.probability)
            if self.mode == 'nonfinite':
                residual[:] = float('nan')
            if self.mode == 'bound':
                residual *= 100
            if self.mode == 'nonzero_mean':
                residual += 1
            if self.mode == 'probability':
                p[:] = 1.5
            if self.mode == 'one_bad_channel':
                residual[0] = float('nan')
            if self.mode == 'runtime_error':
                raise RuntimeError('unsafe test prediction')
            result = {'denoised':signal-residual, 'noise':residual, 'artifact_probability':p}
            if self.mode == 'shape':
                result['artifact_probability'] = p[:,0]
            return result


@unittest.skipIf(v2.torch is None, 'optional PyTorch unavailable')
class ModelAndInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        v2.torch.set_num_threads(2)

    def setUp(self):
        self.config = v2.load_config()
        self.signal = np.stack([np.sin(np.arange(1000)*.07), np.cos(np.arange(1000)*.03)*4+10000])

    def bundle(self, threshold=.5, probability=.8, mode='good'):
        return v2.ModelBundle(StubModel(probability,mode), self.config, Path('unused_v2.pt'), threshold)

    def test_zero_init_model_is_exact_identity_and_bound_zero_mean(self):
        model = v2.build_model(self.config).eval()
        signal = v2.torch.randn(3,1,500)
        output = model(signal)
        v2.torch.testing.assert_close(signal, output['denoised'], rtol=0, atol=0)
        self.assertEqual(0., float(output['noise'].detach().abs().max()))
        with v2.torch.no_grad():
            model.noise_head.weight.normal_(0,10)
            model.noise_head.bias.fill_(10)
        output = model(signal)
        self.assertLessEqual(float(output['noise'].detach().abs().max()), 2.+1e-6)
        self.assertLess(float(output['noise'].detach().mean(dim=-1).abs().max()), 1e-6)
        with self.assertRaises(v2.LearnedDenoiseV2Error):
            model(v2.torch.randn(3,8,500))
        config = v2.load_config()
        config['model']['kernel_size'] = 7
        self.assertEqual((3,1,500), tuple(v2.build_model(config)(signal)['noise'].shape))

    def test_network_channel_independence_and_permutation_equivariance(self):
        v2.torch.manual_seed(4)
        model = v2.build_model(self.config).eval()
        with v2.torch.no_grad():
            model.noise_head.weight.normal_(0,.03)
        bundle = v2.ModelBundle(model,self.config,Path('v2.pt'),.0)
        x = np.stack([np.sin(np.arange(500)*.05),np.cos(np.arange(500)*.07)*8+1e6,np.sin(np.arange(500)*.02)*.01])
        baseline = v2.apply_model_to_array(bundle,x)
        perm = np.array([2,0,1])
        shuffled = v2.apply_model_to_array(bundle,x[perm])
        np.testing.assert_allclose(baseline.model_denoised[perm],shuffled.model_denoised,atol=2e-6,rtol=0)
        np.testing.assert_allclose(baseline.artifact_probability[perm],shuffled.artifact_probability,atol=1e-6,rtol=0)
        other = x.copy()
        other[1] = np.sin(np.arange(500)*2)*1e10
        changed = v2.apply_model_to_array(bundle,other)
        np.testing.assert_allclose(baseline.model_denoised[0],changed.model_denoised[0],atol=1e-6,rtol=0)
        np.testing.assert_allclose(baseline.model_denoised[2],changed.model_denoised[2],atol=1e-6,rtol=0)
        shifted = v2.apply_model_to_array(bundle,x[:1]+1e6)
        np.testing.assert_allclose(shifted.model_denoised,baseline.model_denoised[:1]+1e6,atol=2e-6,rtol=0)
        single = v2.apply_model_to_array(bundle,x[:1])
        np.testing.assert_allclose(baseline.model_denoised[:1],single.model_denoised,atol=1e-6,rtol=0)

    def test_dc_offset_and_positive_scale_equivariance(self):
        bundle = self.bundle()
        result = v2.apply_model_to_array(bundle,self.signal)
        shifted = v2.apply_model_to_array(bundle,self.signal+1e6)
        scaled = v2.apply_model_to_array(bundle,self.signal*100)
        np.testing.assert_allclose(shifted.model_denoised,result.model_denoised+1e6,atol=3e-9,rtol=0)
        np.testing.assert_allclose(scaled.model_denoised,result.model_denoised*100,atol=3e-9,rtol=0)
        np.testing.assert_allclose(result.model_denoised.mean(axis=1),self.signal.mean(axis=1),atol=1e-8,rtol=0)
        self.assertEqual(np.dtype('float64'),result.denoised.dtype)

    def test_high_artifact_low_uncalibrated_confidence_still_replaces(self):
        result = v2.apply_model_to_array(self.bundle(probability=.95),self.signal)
        self.assertTrue(result.replacement_mask.all())
        self.assertTrue((result.confidence < .1).all())
        self.assertIn('NOT_calibrated',result.diagnostics['confidence_interpretation'])
        self.assertFalse(result.diagnostics['release_gate_passed'])

    def test_strict_threshold_equal_one_missing_and_candidate_vs_gated(self):
        for threshold, probability in ((1.,1.),(.5,.5),(None,.9),(.9,.8)):
            with self.subTest(threshold=threshold,probability=probability):
                result = v2.apply_model_to_array(self.bundle(threshold,probability),self.signal)
                self.assertFalse(result.replacement_mask.any())
                np.testing.assert_array_equal(result.denoised,self.signal)
                self.assertFalse(np.array_equal(result.model_denoised,self.signal))
                self.assertFalse(result.rejected_mask.any())

    def test_tail_short_constant_and_invalid_windows_are_raw_rejected(self):
        x = self.signal[:,:503].copy()
        x[0,10] = np.nan
        x[1,:500] = 1e6
        result = v2.apply_model_to_array(self.bundle(),x)
        np.testing.assert_array_equal(result.denoised,x)
        np.testing.assert_array_equal(result.model_denoised,x)
        self.assertTrue(result.rejected_mask.all())
        self.assertFalse(result.replacement_mask.any())
        self.assertTrue((result.confidence == 0).all())
        self.assertTrue(np.isnan(result.artifact_probability).all())
        short = v2.apply_model_to_array(self.bundle(),self.signal[:,:499])
        np.testing.assert_array_equal(short.denoised,self.signal[:,:499])
        self.assertTrue(short.rejected_mask.all())
        good = v2.apply_model_to_array(self.bundle(),self.signal[:,:503])
        self.assertTrue(good.replacement_mask[:,:500].all())
        self.assertTrue(good.rejected_mask[:,500:].all())
        np.testing.assert_array_equal(good.denoised[:,500:],self.signal[:,500:503])

    def test_mask_and_shape_are_strict_and_invalid_original_value_unchanged(self):
        for mask in (np.ones(1000),np.ones((1,1000)),np.full((2,1000),np.nan),np.full((2,1000),.5)):
            with self.assertRaises(v2.LearnedDenoiseV2Error):
                v2.apply_model_to_array(self.bundle(),self.signal,mask)
        for signal in (np.ones(500),np.ones((0,500)),np.ones((2,0)),np.ones((2,500),dtype=complex)):
            with self.assertRaises(v2.LearnedDenoiseV2Error):
                v2.apply_model_to_array(self.bundle(),signal)
        x=self.signal.copy()
        valid=np.ones_like(x,dtype=bool)
        valid[0,20]=False
        result=v2.apply_model_to_array(self.bundle(),x,valid)
        np.testing.assert_array_equal(result.denoised[0,:500],x[0,:500])
        self.assertTrue(result.rejected_mask[0,:500].all())
        for original in (np.inf,-np.inf,np.nan):
            x[0,20]=original
            result=v2.apply_model_to_array(self.bundle(),x)
            np.testing.assert_array_equal(result.denoised[0,:500],x[0,:500])

    def test_badprediction_fallback_and_per_channel_isolation(self):
        for mode in ('nonfinite','bound','nonzero_mean','probability','shape','runtime_error'):
            with self.subTest(mode=mode):
                result=v2.apply_model_to_array(self.bundle(mode=mode),self.signal)
                np.testing.assert_array_equal(result.denoised,self.signal)
                np.testing.assert_array_equal(result.model_denoised,self.signal)
                self.assertTrue(result.rejected_mask.all())
                self.assertFalse(result.replacement_mask.any())
        result=v2.apply_model_to_array(self.bundle(mode='one_bad_channel'),self.signal[:,:500])
        self.assertTrue(result.rejected_mask[0].all())
        self.assertFalse(result.rejected_mask[1].any())
        self.assertTrue(result.replacement_mask[1].all())

    def test_raw_input_readonly_not_mutated_and_fields_same_shape(self):
        x=self.signal.copy()
        original=x.copy()
        x.setflags(write=False)
        result=v2.apply_model_to_array(self.bundle(),x)
        np.testing.assert_array_equal(x,original)
        for field in ('denoised','model_denoised','artifact_probability','confidence','replacement_mask','valid_mask','rejected_mask'):
            self.assertEqual(x.shape,getattr(result,field).shape)
        self.assertFalse(np.shares_memory(result.denoised,x))


@unittest.skipIf(v2.torch is None, 'optional PyTorch unavailable')
class CheckpointAndTrainerTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        self.config=v2.load_config()
        self.path=self.root/'denoiser_v2.pt'
        threshold, calibration=trainer.calibrate_validation_threshold(np.array([.3,.5,.8]))
        self.checkpoint={
            'checkpoint_schema_version':1,'model_version':v2.MODEL_VERSION,'config':self.config,
            'architecture':self.config['model'],'model_state':v2.build_model(self.config).state_dict(),
            'calibrated_threshold':threshold,'validation_threshold':threshold,'threshold_calibration':calibration,
        }

    def test_checkpoint_safe_loading_alias_metadata_and_inference(self):
        v2.torch.save(self.checkpoint,self.path)
        with patch.object(v2.torch,'load',wraps=v2.torch.load) as loader:
            bundle=v2.load_model(self.path,self.config)
        self.assertIs(True,loader.call_args.kwargs['weights_only'])
        self.assertEqual(.8,bundle.calibrated_threshold)
        self.assertEqual(bundle.validation_threshold,bundle.calibrated_threshold)
        self.assertNotIn('model_state',bundle.metadata)
        self.assertIs(bundle.checkpoint_metadata,bundle.metadata)

    def test_failed_known_session_checkpoint_remains_explicit_diagnostic_only(self):
        audited = v2.load_config(Path(__file__).resolve().parents[1] / 'configs' / 'learned_denoiser_v2.json')
        signal = (1e6 + np.sin(np.arange(500, dtype=np.float64) * .05))[None, :]
        original = signal.copy()
        # Frozen checkpoint runtime has only the legacy fields; it is not
        # rewritten when the external deployment config gains audit metadata.
        v2.torch.save(self.checkpoint, self.path)
        legacy_bytes = self.path.read_bytes()
        baseline = v2.apply_model_to_array(v2.load_model(self.path, self.config), signal)
        for checkpoint_config in (self.config, audited):
            if checkpoint_config is not self.config:
                checkpoint = deepcopy(self.checkpoint)
                checkpoint['config'] = deepcopy(checkpoint_config)
                v2.torch.save(checkpoint, self.path)
            checkpoint_bytes = self.path.read_bytes()
            bundle = v2.load_model(self.path, audited)
            result = v2.apply_model_to_array(bundle, signal)
            self.assertEqual('failed_known_session_regression', bundle.config['runtime']['validation_status'])
            self.assertEqual(self.checkpoint['calibrated_threshold'], bundle.calibrated_threshold)
            self.assertIs(True, result.diagnostics['diagnostic_only'])
            self.assertIs(False, result.diagnostics['automatic_processing_enabled'])
            self.assertIs(False, result.diagnostics['release_gate_passed'])
            for field in ('denoised', 'model_denoised', 'artifact_probability', 'confidence',
                          'replacement_mask', 'valid_mask', 'rejected_mask'):
                np.testing.assert_array_equal(getattr(baseline, field), getattr(result, field))
            self.assertEqual(checkpoint_bytes, self.path.read_bytes())
            if checkpoint_config is self.config:
                self.assertEqual(legacy_bytes, checkpoint_bytes)
            np.testing.assert_array_equal(original, signal)
            enabled = deepcopy(audited)
            enabled['runtime']['enabled'] = True
            with patch.object(v2.torch, 'load') as loader:
                with self.assertRaisesRegex(v2.LearnedDenoiseV2Error, 'enablement is forbidden'):
                    v2.load_model(self.path, enabled)
                loader.assert_not_called()
            # Also reject bypass by mutating a bundle after safe loading.
            bundle.config['runtime']['enabled'] = True
            with self.assertRaisesRegex(v2.LearnedDenoiseV2Error, 'enablement is forbidden'):
                v2.apply_model_to_array(bundle, signal)

    def test_version_architecture_threshold_metadata_nonfinite_weights_fail_closed(self):
        variants=[]
        wrong=deepcopy(self.checkpoint); wrong['model_version']='learned_denoise_v1'; variants.append(wrong)
        wrong=deepcopy(self.checkpoint); wrong['checkpoint_schema_version']=9; variants.append(wrong)
        wrong=deepcopy(self.checkpoint); wrong['validation_threshold']=.1; variants.append(wrong)
        wrong=deepcopy(self.checkpoint); wrong['threshold_calibration']['scope']='test'; variants.append(wrong)
        wrong=deepcopy(self.checkpoint); wrong['architecture']['kernel_size']=7; variants.append(wrong)
        wrong=deepcopy(self.checkpoint); wrong['config']['sample_rate_hz']=256; variants.append(wrong)
        wrong=deepcopy(self.checkpoint); wrong['metadata_tuple']=(1,2); variants.append(wrong)
        wrong=deepcopy(self.checkpoint); wrong['model_state']['noise_head.bias'][:]=float('nan'); variants.append(wrong)
        for number, checkpoint in enumerate(variants):
            v2.torch.save(checkpoint,self.path)
            with self.subTest(number=number),self.assertRaises(v2.LearnedDenoiseV2Error):
                v2.load_model(self.path,self.config)
        with self.assertRaises(v2.LearnedDenoiseV2Error):
            v2.load_model(self.root/'missing.pt',self.config)

    def dataset(self, bad_test_signal=False):
        dataset=self.root/'pairs_v2.npz'
        x=np.sin(np.arange(500)*.05).astype(np.float32)
        clean=np.tile(x,(9,1,1))
        noisy=clean.copy()
        types=np.tile(np.array(['clean','eog','emg']),3)
        for i in range(9):
            if types[i]!='clean':
                noisy[i,0]+=.2*np.cos(np.arange(500)*.15).astype(np.float32)
        if bad_test_signal:
            noisy[6:]=np.nan
            clean[6:]=np.nan
        split=np.repeat(np.array(['train','validation','test']),3)
        clean_ids=np.repeat(np.arange(3),3)
        eog_ids=np.where(types=='eog',clean_ids,-1)
        emg_ids=np.where(types=='emg',clean_ids,-1)
        mask=np.broadcast_to((types!='clean')[:,None,None],(9,1,500)).copy()
        np.savez(dataset,clean=clean,noisy=noisy,valid_mask=np.ones((9,1,500),dtype=bool),artifact_mask=mask,
                 split=split,artifact_type=types,clean_epoch_id=clean_ids,eog_epoch_id=eog_ids,emg_epoch_id=emg_ids)
        groups=[]
        for modality in ('eeg','eog','emg'):
            for i,part in enumerate(trainer.SPLITS):
                groups.append({'fingerprint':hashlib.sha256(f'{modality}{i}'.encode()).hexdigest(),
                               'split':part,'epoch_ids_by_source':{modality:[i]}})
        stats={kind:{'raw_epochs':3,'raw_epoch_ids_by_split':{part:[i] for i,part in enumerate(trainer.SPLITS)}} for kind in ('eeg','eog','emg')}
        manifest={'schema_version':2,'status':'formal','dataset_version':'learned_denoiser_v2','requested_samples':9,
                  'output':{'sha256':trainer.sha256(dataset)},
                  'sampling_semantics':{'subject_disjoint_claim':False,'subject_identifiers_available':False},
                  'epoch_split_manifest':{'split_before_resampling_or_mixing':True,'fingerprint_namespace':'global_across_EEG_EOG_EMG',
                                          'fingerprint_groups':groups,'source_statistics':stats}}
        Path(str(dataset)+'.manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        return dataset

    def test_training_loader_ignores_test_signal_values_and_uses_noisy_normalization(self):
        path=self.dataset(bad_test_signal=True)
        data=trainer._load_training_data(path,self.config,20261005,None)
        self.assertEqual([0,1,2],data['train_indices'])
        self.assertEqual([3,4,5],data['validation_indices'])
        self.assertEqual({'train','validation'},set(data['partitions']))
        self.assertNotIn('test_indices',data)
        validation=data['partitions']['validation']
        np.testing.assert_allclose(validation['clean_normalized'][:,0],
            (validation['clean'][:,0].astype(np.float64)-validation['centers'][:,None])/validation['scales'][:,None],atol=1e-6)

    def test_manifest_sha_and_epoch_fingerprint_overlap_are_rejected(self):
        path=self.dataset()
        manifest_path=Path(str(path)+'.manifest.json')
        baseline=json.loads(manifest_path.read_text())
        altered=deepcopy(baseline); altered['output']['sha256']='f'*64
        manifest_path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(v2.LearnedDenoiseV2Error,'SHA256'):
            trainer._load_training_data(path,self.config,1,None)
        altered=deepcopy(baseline)
        altered['epoch_split_manifest']['fingerprint_groups'][1]['fingerprint']=altered['epoch_split_manifest']['fingerprint_groups'][0]['fingerprint']
        manifest_path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(v2.LearnedDenoiseV2Error,'fingerprint'):
            trainer._load_training_data(path,self.config,1,None)

    def test_training_cli_defaults_and_v1_output_refusal(self):
        args=trainer.build_parser().parse_args(['--dataset','data.npz'])
        self.assertEqual((12,64,20261005,2),(args.epochs,args.batch_size,args.seed,args.threads))
        with patch('sys.stdout',io.StringIO()):
            self.assertEqual(2,trainer.main(['--dataset','missing.npz','--output',str(self.root/'denoiser_v1.pt')]))
        self.assertFalse((self.root/'denoiser_v1.pt').exists())

    def test_checkpoint_atomic_failure_preserves_existing_bytes(self):
        self.path.write_bytes(b'existing checkpoint')
        with patch.object(trainer.torch,'save',side_effect=RuntimeError('save fail')):
            with self.assertRaises(RuntimeError):
                trainer._atomic_checkpoint(self.path,self.checkpoint)
        self.assertEqual(b'existing checkpoint',self.path.read_bytes())
        self.assertFalse(list(self.root.glob('*.pending')))

    def test_real_fixture_one_epoch_saves_plain_metadata_and_loads_weights_only(self):
        # Synthetic 9-window fixture only: three train, three validation, and
        # unusable test signals. No real/public dataset or test metric is used.
        path=self.dataset(bad_test_signal=True)
        original_dataset_sha=trainer.sha256(path)
        config_path=self.root/'config_v2.json'
        config_path.write_text(json.dumps(self.config))
        stdout=io.StringIO()
        # Do NOT mock the model, optimizer, epoch runner, metrics or checkpoint.
        with patch('sys.stdout',stdout):
            code=trainer.main(['--dataset',str(path),'--config',str(config_path),
                               '--output',str(self.path),'--epochs','1'])
        self.assertEqual(0,code,stdout.getvalue())
        self.assertEqual(original_dataset_sha,trainer.sha256(path))
        checkpoint=v2.torch.load(self.path,weights_only=True)
        metadata={key:value for key,value in checkpoint.items() if key!='model_state'}
        self.assertTrue(v2.primitive_metadata(metadata))

        def assert_plain(value, location='checkpoint'):
            self.assertNotIsInstance(value,np.generic,location)
            if type(value) is dict:
                for key,child in value.items():
                    self.assertIs(type(key),str,location)
                    assert_plain(child,location+'.'+key)
            elif type(value) is list:
                for index,child in enumerate(value):
                    assert_plain(child,location+'['+str(index)+']')
            else:
                self.assertIn(type(value),(type(None),bool,int,float,str),location)
        assert_plain(metadata)
        for track in ('candidate','gated'):
            self.assertIs(type(checkpoint['validation_report'][track]['corrupted_rmse_improvement_fraction']),float)
        self.assertEqual([0,1,2],checkpoint['train_indices'])
        self.assertEqual([3,4,5],checkpoint['validation_indices'])
        self.assertEqual('validation_only',checkpoint['validation_report']['scope'])
        self.assertEqual('train_validation_only_no_test_metrics',checkpoint['evaluation_scope'])
        self.assertNotIn('test_metrics',checkpoint)
        self.assertGreater(float(checkpoint['model_state']['noise_head.weight'].abs().sum()),0.)
        bundle=v2.load_model(self.path,self.config)
        self.assertEqual(checkpoint['calibrated_threshold'],bundle.calibrated_threshold)
        training_report=json.loads(self.path.with_suffix('.training.json').read_text(encoding='utf-8'))
        assert_plain(training_report,'training_report')
        self.assertEqual(1,training_report['epochs'])
        self.assertEqual(1,len(training_report['history']))
        self.assertEqual('completed',training_report['status'])

    def test_primitive_metadata_still_refuses_numpy_scalars_and_custom_types(self):
        for value in (np.float64(.1),np.float32(.1),np.int64(1),np.bool_(True),object()):
            with self.subTest(scalar_type=type(value).__name__):
                self.assertFalse(v2.primitive_metadata({'nested':[{'value':value}]}))

    def test_mock_training_writes_complete_provenance_without_training_or_test_metrics(self):
        path=self.dataset(bad_test_signal=True)
        config_path=self.root/'config_v2.json'
        config_path.write_text(json.dumps(self.config))
        threshold,calibration=trainer.calibrate_validation_threshold(np.array([.6]))
        report={'scope':'validation_only','clean_false_replacement_fraction':0.,'corrupted_replacement_fraction':.5,
                'candidate':{'corrupted_snr_gain_db':0.},'gated':{'corrupted_snr_gain_db':0.},'gated_is_all_noop':False}
        with patch.object(trainer,'_run_epoch',return_value=1.) as epoch, \
             patch.object(trainer,'_validation_report',return_value=(threshold,calibration,report)),patch('sys.stdout',io.StringIO()):
            code=trainer.main(['--dataset',str(path),'--config',str(config_path),'--output',str(self.path),'--epochs','1'])
        self.assertEqual(0,code)
        self.assertEqual(2,epoch.call_count)
        checkpoint=v2.torch.load(self.path,weights_only=True)
        self.assertEqual([0,1,2],checkpoint['train_indices'])
        self.assertEqual([3,4,5],checkpoint['validation_indices'])
        self.assertEqual(trainer.sha256(path),checkpoint['dataset_sha256'])
        self.assertEqual(trainer.sha256(Path(str(path)+'.manifest.json')),checkpoint['dataset_manifest_sha256'])
        self.assertEqual('train_validation_only_no_test_metrics',checkpoint['evaluation_scope'])
        self.assertEqual(.6,checkpoint['calibrated_threshold'])
        v2.load_model(self.path,self.config)
        self.assertNotIn('test_metrics',checkpoint)


if __name__=='__main__':
    unittest.main()
