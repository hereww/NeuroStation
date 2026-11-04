"""Numerical diagnostics on artificial toys only; never run real v13 diagnostics."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import diagnose_ssvep_v13_loss_conflict as diag

ROOT = Path(__file__).absolute().parents[1]


def scientific():
    import numpy as np
    import torch
    return np, torch


def frozen_cfg():
    # Metadata only; no training/checkpoint/real waveform access.
    from eeg_tools import ssvep_v13_runtime as api
    return api.load_config(ROOT)


class MetadataAndPathsTests(unittest.TestCase):
    def test_import_help_and_missing_consent_without_scientific_stack(self):
        code = r'''
import importlib.abc, runpy, sys
class BlockScientific(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'numpy', 'scipy', 'pandas', 'sklearn'):
            raise AssertionError('scientific import: ' + fullname)
sys.meta_path.insert(0, BlockScientific())
sys.dont_write_bytecode = True
from scripts import diagnose_ssvep_v13_loss_conflict as diag
sys.argv = [str(diag.ROOT / diag.SCRIPT)] + sys.argv[1:]
runpy.run_path(str(diag.ROOT / diag.SCRIPT), run_name='__main__')
'''
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
        for args, status, message in ((['--help'], 0, '--allow-failed-selection-diagnostic'),
                                      (['--output-directory', 'analysis_outputs/never-created'], 2, 'is required'),
                                      (['--output-directory', 'analysis_outputs/never-created',
                                        '--allow-failed-selection-diagnostic', '--batch-size', '0'], 2, 'positive')):
            with self.subTest(args=args):
                result = subprocess.run([sys.executable, '-B', '-c', code, *args], cwd=ROOT,
                                        env=env, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, status, result.stdout + result.stderr)
                self.assertIn(message, result.stdout + result.stderr)
                self.assertNotIn('scientific import', result.stderr)

    def test_consent_checked_before_any_output_creation_or_loader(self):
        with patch.object(diag, 'create_output_directory') as create:
            with self.assertRaisesRegex(ValueError, 'explicit'):
                diag.diagnose('unused')
            create.assert_not_called()

    def test_fixed_indices_and_frozen_role_seeds(self):
        cfg = frozen_cfg()
        recipe = diag.sampling_recipe(cfg)
        expected = [0, 7, 15, 23, 30, 38, 46, 53, 61, 69, 77, 84, 92, 100, 107, 115,
                    123, 131, 138, 146, 154, 161, 169, 177, 185, 192, 200, 208, 215, 223, 231, 239]
        self.assertEqual(recipe['fit']['public']['indices_per_subject'], expected)
        self.assertEqual(recipe['fit']['public']['selected_count'], 128)
        self.assertEqual(recipe['select']['public']['selected_count'], 64)
        self.assertEqual(recipe['select']['long']['indices'], list(range(32)))
        self.assertEqual(recipe['fit']['long']['indices'][::31], [0, 95])
        self.assertEqual(len(set(recipe['fit']['long']['indices'])), 32)
        for role in ('fit', 'select'):
            for index, domain in enumerate(('public', 'long')):
                expected_seed = [20261016, 1, index, 16] if role == 'fit' else [20261016, 2, index]
                self.assertEqual(recipe[role][domain]['case_seed'], expected_seed)
        bad = copy.deepcopy(cfg)
        bad['data']['selection_subjects'] = [7, 8]
        with self.assertRaisesRegex(ValueError, 'cohorts'):
            diag.sampling_recipe(bad)

    def test_new_directory_is_exclusive_and_outside_paths_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'analysis_outputs').mkdir()
            created = diag.create_output_directory('analysis_outputs/diagnostic', root)
            sentinel = created / 'existing.txt'
            sentinel.write_bytes(b'keep')
            with self.assertRaisesRegex(ValueError, 'existing'):
                diag.create_output_directory(created, root)
            self.assertEqual(sentinel.read_bytes(), b'keep')
            for path in ('models/new', 'analysis_outputs', 'analysis_outputs/../escape',
                         'analysis_outputs/missing/new'):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    diag.create_output_directory(path, root)
            self.assertFalse((root / 'models').exists())

    def test_atomic_report_is_final_marker_and_reuse_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'report.json'
            real_replace = os.replace
            def commit(source, destination):
                self.assertFalse(path.exists())
                self.assertEqual(json.loads(Path(source).read_text(encoding='utf-8')), {'ok': True})
                return real_replace(source, destination)
            with patch.object(diag.os, 'replace', side_effect=commit) as replace:
                diag.atomic_report(path, {'ok': True})
                replace.assert_called_once()
            self.assertFalse(path.with_name('report.json.pending').exists())
            with self.assertRaisesRegex(ValueError, 'exists'):
                diag.atomic_report(path, {'ok': False})
            self.assertEqual(json.loads(path.read_text()), {'ok': True})
            nonfinite = Path(temporary) / 'bad.json'
            with self.assertRaises(ValueError):
                diag.atomic_report(nonfinite, {'bad': float('nan')})
            self.assertFalse(nonfinite.exists())


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'), 'torch/numpy needed for toy numerical tests')
class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.np, cls.torch = scientific()
        cls.torch.set_num_threads(2)

    def test_known_conflicting_aligned_orthogonal_zero_and_nonfinite_vectors(self):
        cases = (([1, -2, 3], [-1, 2, -3], -1, -14),
                 ([1, -2, 3], [3, -6, 9], 1, 42),
                 ([1, 0], [0, 2], 0, 0))
        for a, b, cosine, dot in cases:
            with self.subTest(a=a, b=b):
                result = diag.gradient_relation(a, b)
                self.assertTrue(result['finite'])
                self.assertAlmostEqual(result['cosine'], cosine)
                self.assertAlmostEqual(result['dot_product'], dot)
        for a, b, status in (([0, 0], [1, 2], 'lhs_zero'), ([1, 2], [0, 0], 'rhs_zero'),
                              ([0, 0], [0, 0], 'both_zero')):
            result = diag.gradient_relation(a, b)
            self.assertEqual(result['status'], status)
            self.assertIsNone(result['cosine'])
            self.assertEqual(result['dot_product'], 0)
        for bad in (float('nan'), float('inf')):
            result = diag.gradient_relation([1, bad], [2, 3])
            self.assertFalse(result['finite'])
            self.assertIsNone(result['dot_product'])
            self.assertIsNone(result['cosine'])
        overflow = diag.gradient_relation([1e308, 1e308], [1e308, 1e308])
        self.assertFalse(overflow['finite'])
        self.assertEqual(overflow['status'], 'numeric_overflow')
        # Very small nonzero gradients still have a defined cosine.
        self.assertAlmostEqual(diag.gradient_relation([1e-200, 0], [-1e-200, 0])['cosine'], -1)

    def test_sample_count_mean_matches_analytic_gradient_and_full_batch(self):
        torch = self.torch
        model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64).eval()
        with torch.no_grad():
            model.weight.fill_(2)
        x = torch.tensor([[1.], [2.], [3.], [4.], [5.]], dtype=torch.float64)
        y = torch.tensor([[1.], [0.], [4.], [0.], [3.]], dtype=torch.float64)
        def loss(batch):
            a, b = batch
            return (model(a) - b).square().reshape(-1)
        before = diag.snapshot_state(model)
        split, metrics = diag.mean_loss_gradients(model, [(x[:2], y[:2]), (x[2:4], y[2:4]), (x[4:], y[4:])], loss)
        full, full_metrics = diag.mean_loss_gradients(model, [(x, y)], loss)
        # Direct calculus, including deliberately unequal final batch.
        expected = float((2 * x * (2 * x - y)).mean())
        self.assertAlmostEqual(float(split['weight']), expected)
        torch.testing.assert_close(split['weight'], full['weight'], rtol=0, atol=1e-14)
        self.assertEqual(metrics['sample_count'], 5)
        self.assertEqual([row['sample_count'] for row in metrics['batches']], [2, 2, 1])
        self.assertAlmostEqual(metrics['original_unit_mse'], full_metrics['original_unit_mse'])
        diag.assert_read_only_state(model, before)
        self.assertIsNone(model.weight.grad)

    def test_identity_weight_changes_norm_dot_ratio_without_changing_cosine(self):
        torch = self.torch
        gradients = {'public_identity': {'layer.weight': torch.tensor([1., 2.], dtype=torch.float64)},
                     'public_drift': {'layer.weight': torch.tensor([0., 1.], dtype=torch.float64)},
                     'long_identity': {'layer.weight': torch.zeros(2, dtype=torch.float64)},
                     'long_drift': {'layer.weight': torch.tensor([-3., -6.], dtype=torch.float64)}}
        summaries = {name: diag.gradient_summary(value, 100 if name.endswith('identity') else 1)
                     for name, value in gradients.items()}
        result = diag.role_relations(gradients, summaries)
        cross = result['public_identity_vs_long_drift']
        self.assertAlmostEqual(cross['cosine'], -1)
        self.assertAlmostEqual(cross['dot_product'], -15)
        self.assertAlmostEqual(cross['weighted_dot_product'], -1500)
        self.assertAlmostEqual(result['public_identity_to_long_drift_plain_norm_ratio'], 1 / 3)
        self.assertAlmostEqual(result['public_identity_to_long_drift_weighted_norm_ratio'], 100 / 3)
        self.assertEqual(len(result['pairs']), 6)
        self.assertAlmostEqual(summaries['public_identity']['layers']['layer']['weighted_norm'], 100 * 5**0.5)
        self.assertIsNone(result['pairs']['long_identity__long_drift']['cosine'])

    def test_runtime_coordinates_and_gradients_on_toy_are_analytic(self):
        np, torch = self.np, self.torch
        from eeg_tools import ssvep_v13_runtime as api
        class Scale(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.75, dtype=torch.float32))
            def forward(self, signal):
                return signal * self.scale
        model = Scale().eval()
        t = np.arange(512) / 250
        noisy = np.stack([np.broadcast_to(offset + amplitude * np.sin(2 * np.pi * 12 * t), (8, 512))
                          for offset, amplitude in ((3, 2), (-7, 5), (12, 1))]).copy()
        reference = noisy.copy()
        cfg = {'sample_rate_hz': 250}
        before = diag.snapshot_state(model)
        grads, measured = diag.measure_target(model, noisy, reference, 2, cfg)
        normalized, _, rms = api.normalize_noisy(noisy)
        pred = (torch.from_numpy(normalized) * model.scale).detach().numpy()
        expected_output = api.restore_residual(noisy, normalized, pred, rms)
        expected_mse = float(np.mean((expected_output - reference)**2))
        # d/dscale of residual restoration: rms * normalized, using noisy float32 coordinates.
        expected_grad = float(np.mean(2 * (expected_output - reference) * normalized.astype(np.float64) * rms))
        self.assertAlmostEqual(measured['original_unit_mse'], expected_mse, places=12)
        self.assertAlmostEqual(float(grads['scale']), expected_grad, places=5)
        self.assertEqual(measured['batch_count'], 2)
        self.assertAlmostEqual(measured['spectral_input_output_squared_difference']['total'], expected_mse, places=12)
        diag.assert_read_only_state(model, before)

    def test_parseval_nonoverlap_dc_boundaries_nyquist_and_protected_energy(self):
        np = self.np
        n = 1000  # 0.25Hz bins hit every exact boundary.
        t = np.arange(n) / 250
        frequencies = (0.5, 1, 7, 8, 40, 40.25, 125)
        delta = 3 + sum(2 * np.cos(2 * np.pi * frequency * t) for frequency in frequencies)
        x = np.zeros((3, 8, n))
        y = np.broadcast_to(delta, x.shape)
        result = diag.band_energy(x, y)
        for name, expected in {'dc': 9, '0_1': 2, '1_8': 4, '8_40': 4,
                               '40_125': 6, 'outside_protected': 12, 'total': 25}.items():
            self.assertAlmostEqual(result[name], expected, places=10)
        self.assertAlmostEqual(result['total'], float(np.mean(delta**2)), places=10)
        # Odd N has no Nyquist singleton: all positive bins need factor 2.
        odd = np.arange(513)
        signal = np.cos(2 * np.pi * 256 * odd / 513)[None, None, :]
        self.assertAlmostEqual(diag.band_energy(np.zeros_like(signal), signal)['total'], 0.5)
        self.assertEqual(diag.band_energy(y, y)['total'], 0)

    def test_model_tensor_mutation_and_mode_mutation_are_rejected(self):
        torch = self.torch
        model = torch.nn.Linear(2, 1).eval()
        model.register_buffer('counter', torch.tensor(0))
        before = diag.snapshot_state(model)
        model.counter.add_(1)
        with self.assertRaisesRegex(ValueError, 'tensor changed: counter'):
            diag.assert_read_only_state(model, before)
        model.counter.sub_(1)
        model.train()
        with self.assertRaisesRegex(ValueError, 'eval'):
            diag.assert_read_only_state(model, before)
        with self.assertRaisesRegex(ValueError, 'eval'):
            diag.mean_loss_gradients(model, [], lambda batch: batch)

    def test_full_cases_generated_before_selection_and_only_declared_cohorts_read(self):
        np = self.np
        from scripts import train_ssvep_denoiser_v13 as trainer
        cfg = frozen_cfg()
        recipe = diag.sampling_recipe(cfg)
        for role in ('fit', 'select'):
            for domain in ('public', 'long'):
                spec = recipe[role][domain]
                # Artificial windows; mock the expensive generator, retaining full counts.
                reference = np.broadcast_to(np.arange(spec['full_count'], dtype=np.float64)[:, None, None],
                                            (spec['full_count'], 8, spec['window_samples'])).copy()
                def complete(full, seed, passed_cfg):
                    self.assertEqual(len(full), spec['full_count'])
                    self.assertEqual(seed, spec['case_seed'])
                    self.assertIs(passed_cfg, cfg)
                    # Position-dependent artificial noise detects early subsampling.
                    noisy = full + np.arange(len(full))[:, None, None] * 10
                    return {'identity': (full, full), 'drift': (noisy, noisy)}
                with patch.object(trainer, 'load_public_references', return_value=(reference, [])) as public, \
                     patch.object(trainer, 'synthetic_references', return_value=reference) as synthetic, \
                     patch.object(trainer, 'cases_for_domain', side_effect=complete):
                    selected, cases, hashes = diag.domain_cases(ROOT, cfg, role, domain, recipe)
                np.testing.assert_array_equal(selected[:, 0, 0], spec['indices'])
                np.testing.assert_array_equal(cases['drift'][:, 0, 0], np.asarray(spec['indices']) * 11)
                self.assertEqual(hashes['full_reference_sha256'], diag.array_sha(reference))
                if domain == 'public':
                    public.assert_called_once_with(spec['subjects'], cfg, ROOT)
                    synthetic.assert_not_called()
                else:
                    synthetic.assert_called_once_with(spec['full_count'], spec['reference_seed'], cfg)
                    public.assert_not_called()

    def test_archive_roundtrip_complete_parameter_layout_and_hash(self):
        np, torch = self.np, self.torch
        gradients = {role: {target: {'encoder.0.weight': torch.arange(6, dtype=torch.float64).reshape(2, 3)}
                            for target in diag.TARGETS} for role in ('fit', 'select')}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'gradients.npz'
            evidence = diag.save_gradient_archive(path, gradients)
            self.assertEqual(evidence['sha256'], diag.sha(path))
            self.assertEqual(len(evidence['layout']), 8)
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), set(evidence['layout']))
                for key in saved.files:
                    np.testing.assert_array_equal(saved[key], np.arange(6).reshape(2, 3))
                    self.assertEqual(evidence['layout'][key]['array_sha256'], diag.array_sha(saved[key]))
            with self.assertRaises(FileExistsError):
                diag.save_gradient_archive(path, gradients)


if __name__ == '__main__':
    unittest.main()
