"""Behavior-contract tests for the independent v14 candidate.

These tests exercise scope separation, full-record restoration, OLA coverage,
protected spectral behavior, fail-closed metadata, and lazy scientific imports.
They do not train, create checkpoints, or alter the repository.
"""
from __future__ import annotations

import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from eeg_tools import ssvep_v14_runtime as api
from scripts import train_ssvep_denoiser_v14 as runner


class NormalizationContractTests(unittest.TestCase):
    def test_512_window_stats_are_per_window_float64_and_exactly_expected(self):
        noisy = np.stack([
            np.vstack([np.linspace(-2, 3, 512), np.sin(np.linspace(0, 9, 512)) + 2,
                       np.linspace(3.5, 4.5, 512), np.arange(512) * 0.01,
                       np.linspace(-0.5, 0.5, 512) ** 3, np.cos(np.linspace(0, 7, 512)),
                       np.linspace(3, -1, 512), np.sin(np.linspace(0, 4, 512))])
        ]).astype(np.float64)
        normalized, means, rms = api.normalize_noisy_window(noisy)
        expected_mean = noisy.mean(axis=-1, keepdims=True, dtype=np.float64)
        expected_rms = np.sqrt(np.mean((noisy - expected_mean) ** 2, axis=-1,
                                       keepdims=True, dtype=np.float64))
        np.testing.assert_array_equal(means, expected_mean)
        np.testing.assert_array_equal(rms, expected_rms)
        np.testing.assert_array_equal(normalized,
                                      ((noisy - expected_mean) / expected_rms).astype(np.float32))
        self.assertEqual(means.dtype, np.float64)
        self.assertEqual(rms.dtype, np.float64)

    def test_full_record_stats_reused_and_identity_restore_is_exact(self):
        record = (np.arange(5000 * 8, dtype=np.float64).reshape(5000, 8) / 1000.0
                  + np.sin(np.arange(5000)[:, None] / 31.0))
        starts = api.window_starts(len(record))
        normalized, mean, rms, returned = api.normalize_noisy_record(record, starts)
        self.assertEqual(returned, starts)
        self.assertEqual(mean.shape, (8, 1))
        self.assertEqual(rms.shape, (8, 1))
        for index, start in enumerate(starts):
            local = record[start:start + 4096].T
            np.testing.assert_array_equal(normalized[index],
                                          ((local - mean) / rms).astype(np.float32))
        prediction = normalized.copy()
        restored = api.restore_record(record, normalized, prediction, starts, (mean, rms))
        np.testing.assert_array_equal(restored, record)

    def test_record_restore_uses_global_base_not_local_window_mean(self):
        n = 5000
        record = np.linspace(-10.0, 10.0, n, dtype=np.float64)[:, None] * np.ones((1, 8))
        starts = api.window_starts(n)
        normalized, mean, rms, _ = api.normalize_noisy_record(record, starts)
        prediction = np.zeros_like(normalized, dtype=np.float32)
        restored = api.restore_record(record, normalized, prediction, starts, (mean, rms))
        expected = np.broadcast_to(mean.T, record.shape)
        np.testing.assert_allclose(restored, expected, rtol=0, atol=1e-12)
        local_mean = record[starts[0]:starts[0] + 4096].mean(axis=0)
        self.assertFalse(np.allclose(local_mean, mean[:, 0]))
        self.assertFalse(np.allclose(restored[0], local_mean))


class RuntimeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_explicit_scope_rejects_implicit_process_windows(self):
        with self.assertRaises(TypeError):
            api.process_windows(np.zeros((1, 8, 512), dtype=np.float64), object())

    def test_window_starts_tail_receipt_and_input_unchanged(self):
        torch = self.torch

        class Identity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, signal):
                return signal

        original = np.arange(5000 * 8, dtype=np.float64).reshape(5000, 8) / 1000.0
        before = original.copy()
        output, notch, receipt = api.process_recording(original, Identity().eval(), batch_size=2)
        self.assertEqual(output.shape, original.shape)
        self.assertEqual(notch.shape, original.shape)
        self.assertEqual(receipt["window_starts"], [0, 904])
        self.assertEqual(receipt["final_window_start"], 904)
        self.assertTrue(receipt["record_statistics_reused_for_all_subwindows"])
        self.assertFalse(receipt["window_local_mean_reinserted"])
        np.testing.assert_array_equal(original, before)
        np.testing.assert_allclose(output, notch, rtol=0, atol=2e-12)

    def test_protected_mask_and_initial_gain_bounds(self):
        torch = self.torch
        model = api.build_model(api.load_config()).eval()
        signal = torch.randn(2, 8, 512, dtype=torch.float32)
        output, aux = model.forward_with_aux(signal)
        self.assertEqual(tuple(output.shape), tuple(signal.shape))
        frequencies = aux["frequencies_hz"].detach().cpu().numpy()
        mask = aux["protected_mask"].detach().cpu().numpy()
        gain = aux["gain"].detach().cpu().numpy()
        np.testing.assert_array_equal(mask, (frequencies == 0) | ((frequencies >= 8) & (frequencies <= 40)))
        self.assertTrue(np.all((gain >= 0) & (gain <= 1)))
        np.testing.assert_array_equal(gain[:, :, mask], np.ones_like(gain[:, :, mask]))


class MetadataContractTests(unittest.TestCase):
    def test_check_only_is_formal_and_does_not_write_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="neuro-v14-check-") as directory:
            root = Path(directory)
            cfg = api.load_config()
            config_path = root / api.CONFIG
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps(cfg), encoding="utf-8")
            for name in api.bindings(cfg):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if name != api.CONFIG:
                    target.write_bytes(b"metadata fixture")
            result = runner.check_only(root)
            self.assertEqual(result["status"], "ready_to_freeze")
            self.assertFalse(result["runtime_enabled"])
            self.assertFalse(result["production_validated"])
            for output in (api.LOCK, api.MODEL, api.MANIFEST, api.REPORT):
                self.assertFalse((root / output).exists())

    def test_default_loader_rejects_missing_and_failed_selection(self):
        with tempfile.TemporaryDirectory(prefix="neuro-v14-loader-") as directory:
            root = Path(directory)
            cfg = api.load_config()
            (root / api.CONFIG).parent.mkdir(parents=True)
            (root / api.CONFIG).write_text(json.dumps(cfg), encoding="utf-8")
            for name in api.bindings(cfg):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if name == api.CONFIG:
                    continue
                target.write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError, "training lock"):
                api.load_model(root)
            runner.freeze_only(root, write_lock=True)
            state = api.preflight(root)
            failed = {
                "model_version": cfg["model_version"],
                "config_sha256": state["config_sha256"],
                "training_lock_sha256": state["training_lock_sha256"],
                "research_plan_sha256": api.plan_sha(cfg),
                "runtime_enabled": False,
                "production_validated": False,
                "selection_gate_passed": False,
            }
            for name in (api.MANIFEST, api.REPORT):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(failed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ineligible"):
                api.load_model(root)

    def test_import_preflight_is_lazy(self):
        code = r'''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'numpy', 'scipy'):
            raise AssertionError('scientific import during v14 metadata preflight: '+fullname)
sys.meta_path.insert(0, Block())
from eeg_tools import ssvep_v14_runtime as api
from scripts import train_ssvep_denoiser_v14 as runner
from pathlib import Path
import json, tempfile
cfg = api.load_config()
root = Path(tempfile.mkdtemp(prefix='neuro-v14-lazy-'))
config_path = root / api.CONFIG
config_path.parent.mkdir(parents=True)
config_path.write_text(json.dumps(cfg), encoding='utf-8')
for name in api.bindings(cfg):
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if name != api.CONFIG:
        target.write_bytes(b'metadata fixture')
result = runner.check_only(root)
assert result['status'] == 'ready_to_freeze'
assert not any(k.split('.')[0] in ('torch','numpy','scipy') for k in sys.modules)
'''
        proc = subprocess.run([sys.executable, "-B", "-c", code], cwd=api.ROOT,
                              capture_output=True, text=True, timeout=45)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
