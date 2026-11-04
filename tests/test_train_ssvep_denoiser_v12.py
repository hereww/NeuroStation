"""Contract tests for the v12 training runner.

These tests intentionally exercise the normalization boundary independently of
v10's fit/evaluation helpers. They never start CUDA training and never read
personal recordings.
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import torch

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_ssvep_denoiser_v12 as runner


class FakeRuntime:
    def __init__(self):
        self.target_calls = []
        self.restore_calls = []

    def normalize_noisy(self, noisy):
        value = noisy.astype(np.float64)
        means = value.mean(axis=-1, keepdims=True)
        rms = np.sqrt(np.mean(np.square(value - means), axis=-1, keepdims=True))
        rms = np.maximum(rms, 1e-8)
        normalized = ((value - means) / rms).astype(np.float32)
        return normalized, means.astype(np.float64), rms.astype(np.float64)

    def target_in_noisy_coordinates(self, reference, means, rms):
        self.target_calls.append((reference, means, rms))
        return ((reference.astype(np.float64) - means) / rms).astype(np.float32)

    def restore_residual(self, noisy, normalized, prediction, rms):
        self.restore_calls.append((noisy, normalized, prediction, rms))
        return (noisy.astype(np.float64) +
                (prediction.astype(np.float64) - normalized.astype(np.float64)) * rms)


def config():
    return {
        "training": {"seed": 20261015, "batch_size": 64, "identity_loss_weight": 100.0,
                     "gradient_clip": 1.0},
        "engineering_gates": {"identity_mean_normalized_change_max": 0.002},
    }


class V12RunnerTests(unittest.TestCase):
    def test_config_has_fixed_v12_training_contract(self):
        raw = json.loads((ROOT / "configs" / "learned_denoiser_v12_personal.json").read_text(encoding="utf-8"))
        runner.validate_training_config(raw)
        self.assertEqual(raw["training"]["seed"], 20261015)
        self.assertEqual(raw["training"]["epochs"], 24)
        self.assertEqual(raw["training"]["identity_loss_weight"], 100.0)

    def test_no_top_level_numpy_torch_or_cuda_import(self):
        tree = ast.parse((ROOT / "scripts" / "train_ssvep_denoiser_v12.py").read_text(encoding="utf-8"))
        imported = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertNotIn("numpy", imported)
        self.assertNotIn("torch", imported)
        self.assertNotIn("cuda", imported)

    def test_normalization_target_uses_exact_same_noisy_stats(self):
        api = FakeRuntime()
        noisy = np.arange(2 * 8 * 512, dtype=np.float32).reshape(2, 8, 512) / 1000.0
        reference = noisy * 0.25
        result = runner.normalize_batch(api, np, noisy, reference)
        self.assertEqual(len(api.target_calls), 1)
        used_reference, used_means, used_rms = api.target_calls[0]
        self.assertIs(used_reference, reference)
        self.assertIs(used_means, result["means"])
        self.assertIs(used_rms, result["rms"])
        expected = ((reference.astype(np.float64) - result["means"]) / result["rms"]).astype(np.float32)
        np.testing.assert_array_equal(result["target"], expected)

    def test_changing_reference_does_not_change_input_normalization(self):
        api = FakeRuntime()
        noisy = np.random.default_rng(3).normal(size=(2, 8, 512)).astype(np.float32)
        reference_a = np.zeros_like(noisy)
        reference_b = np.full_like(noisy, 7.0)
        a = runner.normalize_batch(api, np, noisy, reference_a)
        b = runner.normalize_batch(api, np, noisy, reference_b)
        np.testing.assert_array_equal(a["normalized"], b["normalized"])
        np.testing.assert_array_equal(a["means"], b["means"])
        np.testing.assert_array_equal(a["rms"], b["rms"])
        self.assertFalse(np.array_equal(a["target"], b["target"]))

    def test_loss_is_equivalent_to_restored_reference_unit_mse(self):
        rng = np.random.default_rng(4)
        prediction = torch.from_numpy(rng.normal(size=(3, 8, 512)).astype(np.float32))
        target = torch.from_numpy(rng.normal(size=(3, 8, 512)).astype(np.float32))
        rms_np = np.array([[0.5, 1.0, 2.0, 0.75, 1.25, 0.4, 3.0, 0.9]], dtype=np.float64).repeat(3, axis=0)
        actual = runner.restored_mse_torch(torch, prediction, target, rms_np)
        expected = ((prediction.double() - target.double()) * torch.from_numpy(rms_np)[:, :, None]).square().mean((1, 2))
        torch.testing.assert_close(actual.double(), expected)

    def test_selection_restoration_is_runtime_forward_and_sameunit(self):
        api = FakeRuntime()
        refs = np.random.default_rng(5).normal(size=(2, 8, 512)).astype(np.float32)
        inputs = [refs.copy() for _ in runner.CONDITIONS]
        # Add deterministic corruption to nonidentity conditions.
        for index in range(1, len(inputs)):
            inputs[index] = inputs[index] + np.float32(index)

        class IdentityModel(torch.nn.Module):
            def forward(self, x):
                return x

        summary = runner.evaluate_selection(torch, np, api, IdentityModel(), refs, inputs,
                                            config(), torch.device("cpu"))
        self.assertEqual(len(api.restore_calls), len(runner.CONDITIONS))
        self.assertIn("condition_sameunit_mse", summary)
        self.assertNotIn("condition_nmse", summary)
        self.assertAlmostEqual(summary["condition_sameunit_mse"]["identity"], 0.0, places=12)
        self.assertAlmostEqual(summary["identity_mean_normalized_change"], 0.0, places=12)
        self.assertTrue(summary["identity_gate_passed"])
        # Runtime restoration receives the normalized noisy input and prediction,
        # not a manually reconstructed alternative.
        noisy, normalized, prediction, rms = api.restore_calls[0]
        np.testing.assert_array_equal(prediction, normalized)
        self.assertEqual(noisy.dtype, np.float32)
        self.assertEqual(rms.dtype, np.float64)

    def test_identity_gate_is_separate_from_sameunit_metric(self):
        values = {name: 1.0 for name in runner.CONDITIONS}
        summary = runner.selection_summary(values, 0.002, config())
        self.assertTrue(summary["identity_gate_passed"])
        summary = runner.selection_summary(values, 0.002000001, config())
        self.assertFalse(summary["identity_gate_passed"])
        self.assertIn("selection_loss_sameunit_mse", summary)
        self.assertNotIn("selection_loss_nmse", summary)

    def test_invalid_noisy_shape_and_nonfinite_are_rejected(self):
        api = FakeRuntime()
        with self.assertRaises(runner.TrainingV12Error):
            runner.normalize_batch(api, np, np.zeros((1, 7, 512), dtype=np.float32))
        invalid = np.zeros((1, 8, 512), dtype=np.float32)
        invalid[0, 0, 0] = np.nan
        with self.assertRaises(runner.TrainingV12Error):
            runner.normalize_batch(api, np, invalid)

    def test_bad_runtime_stats_are_rejected(self):
        class BadRuntime(FakeRuntime):
            def normalize_noisy(self, noisy):
                normalized, means, rms = super().normalize_noisy(noisy)
                return normalized, means.astype(np.float32), rms
        with self.assertRaises(runner.TrainingV12Error):
            runner.normalize_batch(BadRuntime(), np, np.ones((1, 8, 512), dtype=np.float32))

    def test_fit_and_selection_seed_ranges_are_disjoint(self):
        fit = {runner.augmentation_seed(20261015, i, epoch=e)
               for e in range(1, 25) for i in range(5)}
        selection = {runner.augmentation_seed(20261015, i) for i in range(5)}
        self.assertEqual(len(fit), 120)
        self.assertEqual(len(selection), 5)
        self.assertTrue(fit.isdisjoint(selection))
        self.assertEqual(selection, set(range(20361015, 20361020)))

    def test_preferred_candidate_uses_sameunit_metric(self):
        candidate = {"identity_gate_passed": True, "corrupted_mean_sameunit_mse": 2.0,
                     "selection_loss_sameunit_mse": 3.0}
        previous = {"corrupted_mean_sameunit_mse": 2.5, "selection_loss_sameunit_mse": 3.5}
        self.assertTrue(runner.preferred_candidate(candidate, previous, eligible=True))
        self.assertTrue(runner.preferred_candidate(candidate, previous, eligible=False))
        candidate["identity_gate_passed"] = False
        self.assertFalse(runner.preferred_candidate(candidate, previous, eligible=True))

    def test_safe_checkpoint_state_validation_rejects_non_cpu_or_nonfinite(self):
        runner._validate_state(torch, {"weight": torch.ones(2)})
        with self.assertRaises(runner.TrainingV12Error):
            runner._validate_state(torch, {})
        bad = torch.tensor([float("nan")])
        with self.assertRaises(runner.TrainingV12Error):
            runner._validate_state(torch, {"weight": bad})

    def test_output_collision_is_checked_without_runtime_import(self):
        with self.subTest("collision"):
            with self.assertRaises(runner.TrainingV12Error):
                with __import__("tempfile").TemporaryDirectory() as folder:
                    path = Path(folder) / runner.CHECKPOINT
                    path.parent.mkdir(parents=True)
                    path.write_bytes(b"existing")
                    runner.assert_new_outputs(Path(folder))


if __name__ == "__main__":
    unittest.main()
