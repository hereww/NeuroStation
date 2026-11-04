"""Synthetic/mocked v3 training serialization; no real training or test metrics."""
from __future__ import annotations

import ast
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from eeg_tools import learned_denoise_v2 as v2
from eeg_tools import learned_denoise_v3 as core
from scripts import train_domain_gate_v3 as trainer

ROOT = Path(__file__).resolve().parents[1]


class ThresholdAndContractTests(unittest.TestCase):
    def test_strict_unsupported_order_statistic_floor_ties_and_one(self):
        for values in (np.linspace(0, 1, 100), np.ones(20), np.full(101, .4), np.array([.1, .2, .8]), np.linspace(.1, .9, 273)):
            threshold, calibration = trainer.calibrate_support_threshold(values)
            self.assertGreaterEqual(threshold, .5)
            self.assertLessEqual(float(np.mean(values > threshold)), .01)
            self.assertEqual('validation_only', calibration['scope'])
            self.assertEqual('>', calibration['comparison'])
            self.assertEqual('unsupported_score_order_statistic', calibration['method'])
            self.assertIs(True, calibration['empirical_only'])
            self.assertIs(False, calibration['production_calibrated'])
            self.assertTrue(core.primitive_metadata(calibration))
            self.assertIs(type(calibration['observed_unsupported_false_acceptance_fraction']), float)
        self.assertEqual(1., trainer.calibrate_support_threshold(np.ones(20))[0])
        for values in ([], [np.nan], [np.inf], [-.1], [1.1], [[.1]]):
            with self.assertRaises(trainer.DomainGateTrainingError):
                trainer.calibrate_support_threshold(np.asarray(values))

    def test_supported_scores_cannot_lower_threshold(self):
        labels = np.array([True, True, False, False])
        threshold, _, report = trainer._validation_report(np.array([.1, .3, .6, .9]), labels)
        self.assertEqual(.9, threshold)
        self.assertEqual(0., report['supported_acceptance_fraction'])
        self.assertIs(True, report['all_supported_rejected'])
        self.assertIs(False, report['threshold_lowered_to_raise_positive_acceptance'])
        equal, _, report = trainer._validation_report(np.array([.5, .501, .1, .2]), labels)
        self.assertEqual(.5, equal)
        self.assertEqual(.5, report['supported_acceptance_fraction'])

    def test_class_weights_use_full_epoch_counts_not_minibatch_counts(self):
        labels = np.array([True, True, False, False, False, False, False, False])
        zero, one = trainer.class_balanced_weights(labels)
        self.assertAlmostEqual(8 / 12, zero)
        self.assertEqual(2., one)
        self.assertAlmostEqual(1., np.where(labels, one, zero).mean())
        for values in (np.ones(4, dtype=bool), np.zeros(4, dtype=bool), np.ones(4), np.ones((2, 2), dtype=bool)):
            with self.assertRaises(trainer.DomainGateTrainingError):
                trainer.class_balanced_weights(values)

    def test_cli_contract_has_no_epoch_seed_batch_or_test_override(self):
        args = trainer.make_parser().parse_args([])
        self.assertEqual('cuda', args.device)
        self.assertEqual(2, args.threads)
        self.assertEqual((20, 20261005, 128, .001), (trainer.EPOCHS, trainer.SEED, trainer.BATCH_SIZE, trainer.LEARNING_RATE))
        for flag in ('--epochs', '--seed', '--batch-size', '--split'):
            with patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
                trainer.make_parser().parse_args([flag, '1'])
        tree = ast.parse(Path(trainer.__file__).read_text(encoding='utf-8'))
        calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertNotIn('load_unsupported_holdout', calls)
        self.assertNotIn('loadmat', calls)
        self.assertNotIn('np.load', calls)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'load_domain_split':
                self.assertIn(node.args[1].value, ('train', 'validation'))


@unittest.skipIf(trainer.torch is None, 'optional PyTorch unavailable')
class MockTrainingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'analysis_outputs').mkdir()
        (self.root / 'configs').mkdir()
        (self.root / 'models').mkdir()
        self.output = self.root / 'models/domain_gate_v3.pt'
        self.report_path = self.output.with_suffix('.training.json')
        self.denoiser = self.root / 'models/denoiser_v2.pt'
        v2config = v2.load_config()
        denoiser_checkpoint = {
            'checkpoint_schema_version': 1, 'model_version': v2.MODEL_VERSION,
            'config': v2config, 'architecture': v2config['model'],
            'model_state': v2.build_model(v2config).state_dict(),
            'calibrated_threshold': .8, 'validation_threshold': .8,
            'threshold_calibration': {'scope': 'validation_only', 'comparison': '>',
                'method': 'clean_score_order_statistic', 'score_reduction': 'per_channel_window_mean',
                'production_calibrated': False, 'empirical_only': True,
                'clean_window_count': 10, 'observed_clean_false_replacement_fraction': 0.,
                'max_clean_false_replacement_fraction': .01},
        }
        trainer.torch.save(denoiser_checkpoint, self.denoiser)
        self.contract = json.loads((ROOT / 'analysis_outputs/learned_denoiser_v3_preregistration.json').read_text(encoding='utf-8'))
        self.contract['protected_sha256']['models/denoiser_v2.pt'] = trainer.sha256(self.denoiser)
        self.contract_path = self.root / 'analysis_outputs/learned_denoiser_v3_preregistration.json'
        self.contract_path.write_text(json.dumps(self.contract), encoding='utf-8')
        self.config = core.load_config()
        self.config_path = self.root / 'configs/learned_denoiser_v3.json'
        self.config_path.write_text(json.dumps(self.config), encoding='utf-8')
        self.data = {}
        for split, subject in (('train', 'S6.mat'), ('validation', 'S8.mat')):
            labels = np.array([True, True, True, False, False])
            features = np.tile(np.sin(np.arange(500) * .06).astype(np.float32), (5, 1, 1))
            provenance = [{'split': split, 'subject': None if value else subject,
                           'source_sha256': 'a' * 64, 'source_window_index': index}
                          for index, value in enumerate(labels)]
            self.data[split] = {'normalized': features, 'supported': labels, 'provenance': provenance,
                'sources': {'supported': {'sha256': 'b' * 64, 'manifest_sha256': 'c' * 64},
                            'unsupported': {'subjects': [{'subject': subject, 'sha256': 'd' * 64}]}},
                'dataset_sha256': 'b' * 64, 'split': split, 'test_values_accessed': False,
                'new_unsupported_holdout_accessed': False}
        self.epoch = 0

    def run_mock(self, validation=None, extra_args=None):
        def run_epoch(model, loader, optimizer, device, weights):
            if optimizer is not None:
                self.epoch += 1
                with trainer.torch.no_grad():
                    for parameter in model.parameters(): parameter.fill_(self.epoch * .01)
                return 1., np.empty(0)
            loss = .1 if self.epoch in (3, 4) else .2 + self.epoch * .001
            return loss, np.array([.9, .8, .7, .1, .2])
        selected = deepcopy(self.data)
        if validation is not None: selected['validation'] = validation
        with patch.object(trainer, 'load_domain_split', side_effect=lambda root,split: selected[split]) as loader, \
             patch.object(trainer, '_run_epoch', side_effect=run_epoch) as epoch_runner, \
             patch('sys.stdout', io.StringIO()) as stdout:
            code = trainer.main(['--root', str(self.root), '--device', 'cpu', *(extra_args or [])])
        return code, stdout.getvalue(), loader, epoch_runner

    def test_mock_20_epochs_checkpoint_serialization_core_load_and_no_real_training(self):
        before = trainer.sha256(self.denoiser)
        code, stdout, loader, runner = self.run_mock()
        self.assertEqual(0, code, stdout)
        self.assertEqual(['train', 'validation'], [call.args[1] for call in loader.call_args_list])
        self.assertEqual(40, runner.call_count)
        checkpoint = trainer.torch.load(self.output, weights_only=True, map_location='cpu')
        report = json.loads(self.report_path.read_text(encoding='utf-8'))
        core.validate_checkpoint_metadata(checkpoint, self.config)
        self.assertTrue(core.primitive_metadata({k:v for k,v in checkpoint.items() if k != 'gate_state'}))
        self.assertTrue(core.primitive_metadata(report))
        self.assertEqual(20, len(checkpoint['metadata']['history']))
        self.assertEqual(20, report['epochs'])
        self.assertEqual(3, report['best_epoch'])  # exact epoch4 tie keeps epoch3
        self.assertEqual(.5, checkpoint['support_threshold'])
        self.assertEqual('Adam', report['optimizer']['name'])
        self.assertEqual(.001, report['optimizer']['learning_rate'])
        self.assertIs(False, report['environment']['gpu_training_performed'])
        self.assertEqual('cpu', report['environment']['training_device'])
        self.assertEqual(0, report['environment']['gpu_peak_memory_allocated_bytes'])
        self.assertIn('NOT GPU', report['environment']['cpu_execution_warning'])
        self.assertIn('physically decompress test bytes', report['npz_io_semantics'])
        for key in ('known_benchmark_test_accessed', 'new_unsupported_holdout_accessed',
                    'test_arrays_inspected', 'denoiser_training_or_inference_performed',
                    'automatic_processing_enabled', 'production_validated'):
            self.assertIs(False, report[key])
        for value in checkpoint['gate_state'].values():
            self.assertEqual('cpu', value.device.type)
            trainer.torch.testing.assert_close(value, trainer.torch.full_like(value, .03))
        bundle = core.load_model(self.output, self.denoiser, self.config, device='cpu')
        self.assertEqual(.5, bundle.support_threshold)
        self.assertEqual(before, trainer.sha256(self.denoiser))
        self.assertFalse(list(self.output.parent.glob('*.pending')))

    def test_default_cuda_unavailable_rejects_before_data_reads_no_fallback(self):
        with patch.object(trainer.torch.cuda, 'is_available', return_value=False), \
             patch.object(trainer, 'load_domain_split') as loader, patch('sys.stdout', io.StringIO()) as stdout:
            code = trainer.main(['--root', str(self.root)])
        self.assertEqual(2, code)
        self.assertIn('no silent CPU fallback', stdout.getvalue())
        loader.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_existing_checkpoint_or_report_is_never_overwritten(self):
        for path in (self.output, self.report_path):
            path.write_bytes(b'protected existing')
            code, _, loader, _ = self.run_mock()
            self.assertEqual(2, code)
            self.assertEqual(b'protected existing', path.read_bytes())
            loader.assert_not_called()
            path.unlink()

    def test_preregistration_hyperparameter_and_denoser_identity_changes_refused(self):
        for name, value in (('epochs', 21), ('seed', 123), ('learning_rate', .002)):
            altered = deepcopy(self.contract); altered[name] = value
            self.contract_path.write_text(json.dumps(altered), encoding='utf-8')
            code, _, loader, _ = self.run_mock()
            self.assertEqual(2, code)
            loader.assert_not_called()
        self.contract_path.write_text(json.dumps(self.contract), encoding='utf-8')
        self.denoiser.write_bytes(b'not frozen checkpoint')
        code, _, loader, _ = self.run_mock()
        self.assertEqual(2, code); loader.assert_not_called()

    def test_unauthorized_holdout_provenance_or_nonfinite_input_is_rejected(self):
        values = deepcopy(self.data['validation']); values['provenance'][-1]['subject'] = 'S4.mat'
        code, _, _, runner = self.run_mock(validation=values)
        self.assertEqual(2, code); runner.assert_not_called()
        values = deepcopy(self.data['validation']); values['normalized'][0, 0, 0] = np.nan
        code, _, _, runner = self.run_mock(validation=values)
        self.assertEqual(2, code); runner.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_nonprimitive_metadata_not_silently_coerced_or_pickled(self):
        values = deepcopy(self.data['validation']); values['provenance'][0]['bad'] = np.float64(.1)
        code, _, _, runner = self.run_mock(validation=values)
        self.assertEqual(2, code); runner.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_publication_failure_rolls_back_only_new_files_and_cleans_staging(self):
        original_link = trainer.os.link
        count = 0
        def fail_second(source, target):
            nonlocal count
            count += 1
            if count == 2: raise OSError('synthetic report link failure')
            return original_link(source, target)
        with patch.object(trainer.os, 'link', side_effect=fail_second):
            code, _, _, _ = self.run_mock()
        self.assertEqual(2, code)
        self.assertFalse(self.output.exists()); self.assertFalse(self.report_path.exists())
        self.assertFalse(list(self.output.parent.glob('*.pending')))

    def test_validation_balanced_bce_is_full_split_mean_independent_of_batch_size(self):
        torch = trainer.torch
        class FixedLogits(torch.nn.Module):
            def forward(self, features): return features[:, 0, 0]
        labels = np.array([True, True, False, False, False, False, False, False])
        logits = np.linspace(-1., 1., len(labels)).astype(np.float32)
        features = np.zeros((len(labels), 1, 500), dtype=np.float32); features[:, 0, 0] = logits
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(features), torch.from_numpy(labels.astype(np.float32)))
        per_sample = torch.nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(logits), torch.from_numpy(labels.astype(np.float32)), reduction='none').numpy()
        expected = .5 * (float(per_sample[labels].mean()) + float(per_sample[~labels].mean()))
        for batch_size in (2, 5, 8):
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            loss, scores = trainer._run_epoch(FixedLogits(), loader, None, torch.device('cpu'), trainer.class_balanced_weights(labels))
            self.assertAlmostEqual(expected, loss, places=6)
            self.assertEqual((len(labels),), scores.shape)


if __name__ == '__main__':
    unittest.main()
