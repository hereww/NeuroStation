"""Numerical/contract tests for independent v13; never start CUDA training.

Lock/artifact tests use disposable directories. Real FFT tests run on CPU with
installed torch, while metadata subprocesses deliberately forbid torch/numpy/scipy.
"""
from __future__ import annotations

import copy
import importlib.abc
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from eeg_tools import ssvep_v13_runtime as api
from scripts import train_ssvep_denoiser_v13 as runner


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='neuro-v13-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = api.load_config()
        for name in api.bindings(self.cfg):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if name == api.CONFIG:
                target.write_text(json.dumps(self.cfg), encoding='utf-8')
            elif name == api.PLAN:
                continue
            else:
                target.write_bytes(b'fixture source or waveform archive; metadata never decodes it')
        plan = json.loads((api.ROOT/api.PLAN).read_text(encoding='utf-8'))
        for name in plan['prior_evidence']:
            target = self.root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'frozen prior evidence fixture')
            plan['prior_evidence'][name] = api.sha(target)
        (self.root/api.PLAN).write_text(json.dumps(plan), encoding='utf-8')

    def test_metadata_check_and_freeze_without_scientific_imports(self):
        code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'numpy', 'scipy'):
            raise AssertionError('scientific import during metadata: '+fullname)
sys.meta_path.insert(0, Block())
from scripts import train_ssvep_denoiser_v13 as runner
root = sys.argv[1]
assert runner.check_only(root)['status'] == 'ready_to_freeze'
assert runner.freeze_only(root)['status'] == 'ready'
assert runner.check_only(root)['status'] == 'ready'
assert not any('ssvep_v10' in k or 'ssvep_v12' in k for k in sys.modules)
assert not any(k.split('.')[0] in ('torch','numpy','scipy') for k in sys.modules)
'''
        proc = subprocess.run([sys.executable, '-B', '-c', code, str(self.root)], cwd=api.ROOT,
                              capture_output=True, text=True, timeout=45)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_freeze_binds_plan_sources_and_only_fit_select(self):
        state = runner.freeze_only(self.root)
        self.assertEqual(state['status'], 'ready')
        lock = json.loads((self.root/api.LOCK).read_text())
        self.assertEqual(set(lock['files']), set(api.bindings(self.cfg)))
        self.assertEqual(lock['research_plan_sha256'], api.sha(self.root/api.PLAN))
        self.assertFalse(any('subject_007' in p or 'subject_008' in p or 'session_' in p for p in lock['files']))
        self.assertEqual(lock['roles']['personal'], [])
        self.assertEqual(lock['roles']['evaluation'], [])

    def test_freeze_refuses_existing_lock_without_changing_bytes(self):
        runner.freeze_only(self.root)
        before = (self.root/api.LOCK).read_bytes()
        with self.assertRaisesRegex(ValueError, 'existing output'):
            runner.freeze_only(self.root)
        self.assertEqual((self.root/api.LOCK).read_bytes(), before)

    def test_preflight_refuses_config_mutation(self):
        runner.freeze_only(self.root)
        cfg = copy.deepcopy(self.cfg)
        cfg['design'].append('unreviewed post-freeze edit')
        (self.root/api.CONFIG).write_text(json.dumps(cfg), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'lock mismatch'):
            api.preflight(self.root)

    def test_preflight_refuses_source_and_data_mutations(self):
        runner.freeze_only(self.root)
        for name in (api.SOURCES[0], api.bindings(self.cfg)[-2]):
            with self.subTest(name=name):
                file = self.root/name
                original = file.read_bytes()
                file.write_bytes(original+b'changed')
                with self.assertRaisesRegex(ValueError, 'locked file changed'):
                    api.preflight(self.root)
                file.write_bytes(original)

    def test_preflight_refuses_plan_mutation_and_evidence_mutation(self):
        runner.freeze_only(self.root)
        filename = self.root/api.PLAN
        original = filename.read_bytes()
        plan = json.loads(original)
        plan['objective'] += '; modified'
        filename.write_text(json.dumps(plan), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'plan lock mismatch'):
            api.preflight(self.root)
        filename.write_bytes(original)
        evidence = self.root/next(iter(plan['prior_evidence']))
        evidence.write_bytes(b'changed historical evidence')
        with self.assertRaisesRegex(ValueError, 'prior evidence changed'):
            api.preflight(self.root)

    def test_preflight_refuses_lock_digest_roles_extra_bindings(self):
        runner.freeze_only(self.root)
        file = self.root/api.LOCK
        lock = json.loads(file.read_text())
        changes = [lambda x: x['files'].__setitem__(api.SOURCES[0], '0'*64),
                   lambda x: x['roles'].__setitem__('fit', [1, 2, 7]),
                   lambda x: x['files'].__setitem__('unexpected', '0'*64),
                   lambda x: x.__setitem__('research_plan_sha256', '0'*64)]
        for change in changes:
            value = copy.deepcopy(lock)
            change(value)
            file.write_text(json.dumps(value), encoding='utf-8')
            with self.assertRaises(ValueError):
                api.preflight(self.root)

    def test_outputs_exclusive_and_run_rejects_before_torch_or_cuda(self):
        runner.freeze_only(self.root)
        for name in (api.MODEL, api.REPORT, api.MANIFEST):
            with self.subTest(name=name):
                filename = runner.exclusive_json(self.root, name, {'existing': True})
                before = filename.read_bytes()
                with self.assertRaisesRegex(ValueError, 'existing output'):
                    runner.exclusive_json(self.root, name, {'overwrite': True})
                with self.assertRaisesRegex(ValueError, 'existing output'):
                    runner.run_training(self.root)
                self.assertEqual(before, filename.read_bytes())
                filename.unlink()

    def test_paths_reject_traversal_absolute_streams_links(self):
        for name in ('../old.pt', '/old.pt', 'C:/old.pt', 'models/../old.pt',
                     'models//old.pt', 'models\\old.pt', 'models/old.pt:stream'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                api.safe_path(self.root, name)
        source = self.root/'hardlink-source'
        source.write_bytes(b'preserve')
        linked = self.root/'hardlink'
        linked.hardlink_to(source)
        with self.assertRaisesRegex(ValueError, 'hard-linked'):
            api.safe_path(self.root, 'hardlink')

    def test_legacy_model_config_and_data_roles_rejected(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['data']['fit_subjects'] = [1, 2, 7, 8]
        (self.root/api.CONFIG).write_text(json.dumps(cfg), encoding='utf-8')
        with self.assertRaises(ValueError):
            api.load_config(self.root)
        with self.assertRaisesRegex(ValueError, 'fit/select'):
            runner.load_public_references([7, 8], self.cfg, self.root)


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        cls.np, cls.torch = np, torch
        torch.set_num_threads(2)
        cls.cfg = api.load_config()

    def signal(self, n=4096, count=2, seed=55):
        return runner.synthetic_references(count, seed, self.cfg)

    def test_real_model_dynamic_fft_protection_and_gain_bounds(self):
        np, torch = self.np, self.torch
        model = api.build_model(self.cfg).eval()
        with torch.no_grad():
            model.encoder[-1].bias.fill_(1.5)  # force visible unprotected attenuation
        for n in (512, 4096):
            with self.subTest(n=n):
                x = torch.from_numpy(np.random.default_rng(n).normal(size=(2, 8, n)).astype(np.float32))
                with torch.no_grad():
                    y, aux = model.forward_with_aux(x)
                f = np.fft.rfftfreq(n, 1/250)
                mask = (f == 0) | ((f >= 8) & (f <= 40))
                np.testing.assert_allclose(aux['frequencies_hz'].numpy(), f, atol=1e-5)
                np.testing.assert_array_equal(aux['protected_mask'].numpy(), mask)
                gain = aux['gain'].numpy()
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(np.all((gain >= 0) & (gain <= 1)))
                np.testing.assert_array_equal(gain[..., mask], 1)
                self.assertTrue(np.all(gain[..., ~mask] < 0.5))
                before, after = torch.fft.rfft(x).numpy(), torch.fft.rfft(y).numpy()
                ratio = np.sum(abs(after[..., mask]-before[..., mask])**2)/np.sum(abs(before[..., mask])**2)
                self.assertLess(ratio, 1e-12)
                self.assertLess(float(np.sum(abs(after[..., ~mask])**2)), float(np.sum(abs(before[..., ~mask])**2)))
        self.assertEqual(str(inspect.signature(model.forward)), '(signal)')
        with self.assertRaises(TypeError):
            model(torch.zeros(1,8,512), target=1)

    def test_normalize_restore_identity_dc_and_same_reference_units(self):
        np = self.np
        reference = self.signal() * np.linspace(0.1, 20, 8)[None, :, None] + 1000
        noisy = runner.corrupt(reference, 'combined', 10, self.cfg)
        norm, means, rms = api.normalize_noisy(noisy)
        self.assertEqual(norm.dtype, np.float32)
        self.assertEqual(means.dtype, np.float64)
        self.assertEqual(rms.dtype, np.float64)
        np.testing.assert_array_equal(api.restore_residual(noisy, norm, norm.copy(), rms), noisy)
        target = api.target_in_noisy_coordinates(reference, means, rms)
        out = api.restore_residual(noisy, norm, target, rms)
        # Both coordinates are rounded to float32. Bound the two rounding errors
        # in original units instead of claiming exact inversion of rounded targets.
        bound = (np.abs(np.spacing(norm)).astype(np.float64)
                 + np.abs(np.spacing(target)).astype(np.float64)) * rms / 2
        bound += np.abs(np.spacing(noisy)) * 4
        self.assertTrue(np.all(np.abs(out-reference) <= bound))
        model = api.build_model(self.cfg).eval()
        result = api.process_windows(noisy, model)
        np.testing.assert_allclose(result.mean(-1), noisy.mean(-1), atol=1e-6, rtol=0)

    def test_restored_loss_matches_runtime_exactly_and_gradients(self):
        np, torch = self.np, self.torch
        ref = self.signal(count=1)
        noisy = runner.corrupt(ref, 'drift', 11, self.cfg)
        norm, _, rms = api.normalize_noisy(noisy)
        model = api.build_model(self.cfg)
        x = torch.from_numpy(norm)
        prediction = model(x)
        loss = api.restored_mse_torch(prediction, x, torch.from_numpy(noisy), torch.from_numpy(rms), torch.from_numpy(ref))
        actual = api.restore_residual(noisy, norm, prediction.detach().numpy(), rms)
        np.testing.assert_allclose(loss.detach().numpy(), ((actual-ref)**2).mean((1,2)), rtol=0, atol=1e-12)
        loss.mean().backward()
        self.assertGreater(float(model.encoder[-1].bias.grad.abs().sum()), 0)
        scaled = api.restored_mse_torch(prediction.detach(), x, torch.from_numpy(noisy*3),
                                        torch.from_numpy(rms*3), torch.from_numpy(ref*3))
        np.testing.assert_allclose(scaled.numpy(), loss.detach().numpy()*9, atol=1e-12, rtol=1e-12)

    def test_invalid_window_stats_and_prediction_rejected(self):
        np = self.np
        ref = self.signal(count=1)
        invalid = [np.zeros((1,8,4096)), np.zeros((0,8,4096)), np.zeros((1,7,4096)),
                   np.zeros((1,8,1024)), np.ones((1,8,512),dtype=np.int32), np.full((1,8,512),np.nan)]
        for value in invalid:
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                api.normalize_noisy(value)
        norm, means, rms = api.normalize_noisy(ref)
        with self.assertRaises(ValueError):
            api.target_in_noisy_coordinates(ref, means, rms.astype(np.float32))
        with self.assertRaises(ValueError):
            api.restore_residual(ref, norm, norm.astype(np.float64), rms)
        with self.assertRaises(ValueError):
            api.restore_residual(ref, norm, norm, np.zeros_like(rms))
        with self.assertRaises(ValueError):
            api.process_windows(ref, api.build_model(self.cfg).eval(), 0)

    def test_real_model_invalid_shape_dtype_nan(self):
        torch = self.torch
        model = api.build_model(self.cfg).eval()
        for value in (torch.zeros(1,8,512,dtype=torch.float64), torch.zeros(1,7,4096),
                      torch.zeros(1,8,513), torch.full((1,8,512),float('inf'))):
            with self.assertRaises(ValueError):
                model(value)

    def test_ola_identity_tail_and_independent_notch(self):
        np, torch = self.np, self.torch
        class Identity(torch.nn.Module):
            def forward(self, signal):
                return signal
        for length in (4096, 4097, 7001, 8192):
            with self.subTest(length=length):
                t = np.arange(length)/250
                raw = (np.sin(2*np.pi*12*t) + 0.8*np.sin(2*np.pi*50*t))[:,None] * np.arange(1,9)[None,:] + 100
                output, notch, meta = api.process_recording(raw, Identity().eval(), batch_size=2)
                np.testing.assert_array_equal(output, notch)
                np.testing.assert_array_equal(notch, api.condition_line(raw))
                self.assertEqual(meta['final_window_start'], length-4096)
                self.assertEqual(meta['processed_samples'], length)
                self.assertGreater(float(np.mean((raw-notch)**2)), 0.1)
                np.testing.assert_array_equal(raw, (np.sin(2*np.pi*12*t)+0.8*np.sin(2*np.pi*50*t))[:,None]*np.arange(1,9)[None,:]+100)

    def test_ola_tail_receives_residual_and_no_uncovered_samples(self):
        np, torch = self.np, self.torch
        class Shift(torch.nn.Module):
            def forward(self, signal):
                return signal + 0.25
        raw = np.random.default_rng(1).normal(size=(7001,8))
        out, notch, _ = api.process_recording(raw, Shift().eval(), batch_size=1)
        self.assertTrue(np.all(out-notch > 0.2))
        # Last sample must be processed through the appended final window.
        self.assertGreater(float(np.min(out[-1]-notch[-1])), 0.2)
        with self.assertRaises(ValueError):
            api.process_recording(raw[:512], Shift().eval())

    def test_continuous_synthetic_reproducible_independent_and_not_tiled(self):
        np = self.np
        fit = self.signal(seed=self.cfg['synthetic']['fit_seed'])
        again = self.signal(seed=self.cfg['synthetic']['fit_seed'])
        selection = self.signal(seed=self.cfg['synthetic']['selection_seed'])
        np.testing.assert_array_equal(fit, again)
        self.assertFalse(np.array_equal(fit, selection))
        self.assertEqual(fit.shape, (2,8,4096))
        self.assertGreater(float(np.mean((fit[...,512:]-fit[...,:-512])**2)), 0.1)
        self.assertGreater(float(abs(fit.mean(-1)).max()), 0.01)
        f = np.fft.rfftfreq(4096, 1/250)
        peaks = f[abs(np.fft.rfft(fit-fit.mean(-1,keepdims=True))).argmax(-1)]
        self.assertTrue(np.all((peaks > 7.9)&(peaks < 20.1)))
        for condition in api.CONDITIONS:
            a = runner.corrupt(fit, condition, 91, self.cfg)
            b = runner.corrupt(fit, condition, 91, self.cfg)
            np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            runner.corrupt(fit, 'label_target', 1, self.cfg)

    def test_corruptions_physical_ranges_trend_and_intermittent_burst(self):
        np = self.np
        ref = self.signal(count=2)
        rms = np.std(ref, axis=-1)
        gaussian = runner.corrupt(ref, 'gaussian', 5, self.cfg)-ref
        ratio = np.sqrt(np.mean(gaussian**2,axis=-1))/rms
        self.assertTrue(np.all((ratio >= 0.2-1e-12)&(ratio <= 0.8+1e-12)))
        burst = runner.corrupt(ref, 'high_burst', 5, self.cfg)-ref
        active = np.any(abs(burst)>1e-12, axis=1).sum(axis=-1)
        self.assertTrue(np.all((active > 0)&(active <= 0.3*4096)))
        drift = runner.corrupt(ref, 'drift', 5, self.cfg)-ref
        power = abs(np.fft.rfft(drift))**2
        f = np.fft.rfftfreq(4096, 1/250)
        self.assertGreater(float(power[...,f<1].sum()/power.sum()),0.8)
        np.testing.assert_allclose(drift.mean(-1), 0, atol=1e-14)

    def test_selection_equal_domains_conditions_and_two_identity_gates(self):
        def domain(value, identity):
            return dict(identity_normalized_change=identity,
                        conditions={c:dict(model_mse=value) for c in api.CONDITIONS})
        domains = {'public':domain(2, 0.002), 'long':domain(6, 0.001)}
        summary = runner.selection_summary(domains, self.cfg)
        self.assertEqual(summary['balanced_corrupted_mse'],4)
        self.assertTrue(summary['selection_gate_passed'])
        for name in domains:
            value = copy.deepcopy(domains)
            value[name]['identity_normalized_change']=0.00201
            self.assertFalse(runner.selection_summary(value,self.cfg)['selection_gate_passed'])
        with self.assertRaises(ValueError):
            runner.selection_summary({'public':domains['public']}, self.cfg)

    def test_checkpoint_selection_rejects_failed_gate_and_keeps_earliest_tie(self):
        passed = dict(selection_gate_passed=True, balanced_corrupted_mse=0.5)
        failed = dict(selection_gate_passed=False, balanced_corrupted_mse=0.01)
        self.assertTrue(runner.preferred_candidate(passed, None))
        self.assertTrue(runner.preferred_candidate(passed, 0.6))
        self.assertFalse(runner.preferred_candidate(passed, 0.5))
        self.assertFalse(runner.preferred_candidate(failed, None))
        self.assertFalse(runner.preferred_candidate(failed, 0.6))

    def test_selection_only_passes_signal_and_loss_separates_notch(self):
        np, torch = self.np, self.torch
        class Spy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.received = []
            def forward(self, signal):
                self.received.append((tuple(signal.shape),signal.dtype))
                return signal
        model = Spy()
        refs = {'public': self.signal(count=1)[...,:512].copy(), 'long':self.signal(count=1)}
        cases = {name:runner.cases_for_domain(ref, [20261016,2,i],self.cfg) for i,(name,ref) in enumerate(refs.items())}
        summary = runner.evaluate_selection(model,refs,cases,self.cfg)
        self.assertEqual(len(model.received),10)
        self.assertEqual({v[0][-1] for v in model.received},{512,4096})
        self.assertTrue(all(v[1]==torch.float32 for v in model.received))
        for domain in summary['domains'].values():
            for metrics in domain['conditions'].values():
                self.assertEqual(metrics['model_mse'],metrics['notch_mse'])
                self.assertEqual(metrics['learned_mse_reduction'],0)


if __name__ == '__main__':
    unittest.main()
