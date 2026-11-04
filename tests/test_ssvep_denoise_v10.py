"""Synthetic v10 contract tests; never read personal data or model checkpoints.

Run: python -B -m unittest tests.test_ssvep_denoise_v10 -v
"""
from __future__ import annotations

import copy
import inspect
import io
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from eeg_tools.ssvep_denoise_v10 import (
    SpectralDenoiserV10, build_model, validate_config,
)


def random_window(batch=2):
    return torch.randn(batch, 8, 512, generator=torch.Generator().manual_seed(1309))


def exact_bin_window(bin_index, batch=2):
    spectrum = torch.zeros(batch, 8, 257, dtype=torch.complex64)
    spectrum[..., bin_index] = 256 if bin_index in (0, 256) else complex(96, -192)
    return torch.fft.irfft(spectrum, n=512)


class SpectralDenoiserV10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(4)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(2031)
        self.model = build_model()

    def test_default_configuration_and_builder(self):
        self.assertIsInstance(self.model, nn.Module)
        self.assertEqual(self.model.sample_rate_hz, 250)
        self.assertEqual(validate_config()["status"], "formal")
        self.assertIsInstance(build_model(validate_config()), SpectralDenoiserV10)

    def test_real_experiment_config_accepted_and_copied(self):
        path = Path(__file__).resolve().parents[1] / "configs/learned_denoiser_v10_personal.json"
        cfg = json.loads(path.read_text(encoding="utf-8"))
        validated = validate_config(cfg)
        model = build_model(cfg)
        self.assertEqual(validated, cfg)
        cfg["model"]["hidden_channels"] = 99
        self.assertEqual(model.config["model"]["hidden_channels"], 24)
        self.assertEqual(validated["model"]["hidden_channels"], 24)

    def test_configs_are_independent(self):
        cfg = validate_config()
        cfg["model"]["kernels"][0] = 99
        self.assertEqual(validate_config()["model"]["kernels"], [7, 5, 1])

    def test_invalid_config_rejected(self):
        for invalid in ([], False, "config.json", {}, {"model": {}}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_config(invalid)
        mutations = [("sample_rate_hz", 256), ("sample_rate_hz", True),
                     ("sample_rate_hz", float("nan")), ("window_samples", 256),
                     ("channel_count", 7), ("status", "draft")]
        for key, value in mutations:
            cfg = validate_config()
            cfg[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                build_model(cfg)
        for key, value in (("kernels", [5, 5, 1]), ("hidden_channels", 32),
                           ("protected_band_hz", [9., 40.]), ("dc_preserved", 1),
                           ("allow_amplification", True), ("initial_gate_bias", -3.)):
            cfg = validate_config()
            cfg["model"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(cfg)

    def test_architecture_and_initialization(self):
        convs = [layer for layer in self.model.encoder if isinstance(layer, nn.Conv1d)]
        self.assertEqual([(c.in_channels, c.out_channels, c.kernel_size[0]) for c in convs],
                         [(4, 24, 7), (24, 24, 5), (24, 1, 1)])
        self.assertEqual(sum(isinstance(m, nn.GELU) for m in self.model.encoder), 2)
        self.assertEqual(torch.count_nonzero(convs[-1].weight).item(), 0)
        torch.testing.assert_close(convs[-1].bias, torch.tensor([-4.]), rtol=0, atol=0)

    def test_output_aux_shape_dtype_and_forward_agreement(self):
        x = random_window()
        pred, aux = self.model.forward_with_aux(x)
        self.assertEqual(pred.shape, x.shape)
        self.assertEqual(pred.dtype, torch.float32)
        self.assertEqual(aux["attenuation"].shape, (2, 8, 257))
        self.assertEqual(aux["protected_complex_energy_ratio"].shape, (2,))
        self.assertTrue(torch.isfinite(pred).all())
        torch.testing.assert_close(self.model(x), pred, rtol=0, atol=0)
        torch.testing.assert_close(x, random_window(), rtol=0, atol=0)

    def test_bad_shapes_rejected(self):
        for shape in ((8, 512), (0, 8, 512), (1, 7, 512), (1, 8, 256),
                      (1, 8, 513), (1, 8, 512, 1)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "shape"):
                self.model(torch.zeros(shape))
        with self.assertRaises(ValueError):
            self.model([1, 2, 3])

    def test_bad_dtype_rejected(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float64, torch.int32,
                      torch.bool, torch.complex64):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "float32"):
                self.model(torch.zeros(1, 8, 512, dtype=dtype))

    def test_nonfinite_input_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            x = random_window(1)
            x[0, 4, 79] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                self.model(x)

    def test_dense_input_required(self):
        with self.assertRaisesRegex(ValueError, "dense"):
            self.model(random_window(1).to_sparse())

    def test_fft_overflow_rejected(self):
        with self.assertRaisesRegex(ValueError, "FFT overflowed"):
            self.model(torch.full((1, 8, 512), 1e38))

    def test_nonfinite_weights_rejected(self):
        with torch.no_grad():
            self.model.encoder[-1].bias.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "encoder"):
            self.model(random_window(1))

    def test_model_dtype_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "model"):
            self.model.double()(random_window(1))

    def test_noncontiguous_input(self):
        x = random_window().transpose(1, 2).contiguous().transpose(1, 2)
        self.assertFalse(x.is_contiguous())
        torch.testing.assert_close(self.model(x), self.model(x.contiguous()), rtol=0, atol=0)

    def test_four_features_exact_definitions(self):
        x = random_window()
        spec = torch.fft.rfft(x)
        features = self.model._features(spec).reshape(2, 8, 4, 257)
        log = torch.log1p(spec.abs() / math.sqrt(512))
        smooth = F.avg_pool1d(F.pad(log, (2, 2), mode="replicate"), 5, stride=1)
        expected = (log, log.mean(1, keepdim=True).expand_as(log), log - smooth,
                    (torch.arange(257) / 256).view(1, 1, 257).expand_as(log))
        for i, feature in enumerate(expected):
            torch.testing.assert_close(features[:, :, i], feature, rtol=1e-6, atol=1e-6)

    def test_shared_encoder_channel_permutation_equivariance(self):
        with torch.no_grad():
            self.model.encoder[-1].weight.normal_(std=.03)
        x = random_window()
        permutation = torch.tensor([5, 2, 4, 7, 1, 3, 0, 6])
        pred, aux = self.model.forward_with_aux(x)
        perm_pred, perm_aux = self.model.forward_with_aux(x[:, permutation])
        torch.testing.assert_close(perm_pred, pred[:, permutation], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(perm_aux["attenuation"], aux["attenuation"][:, permutation],
                                  rtol=1e-6, atol=1e-6)

    def test_initial_attenuation_and_protection_boundaries(self):
        _, aux = self.model.forward_with_aux(random_window())
        mask = self.model.protected_mask
        expected = torch.zeros(257, dtype=torch.bool)
        expected[0] = True
        expected[17:82] = True  # 17*250/512 >=8; 81*250/512 <=40.
        torch.testing.assert_close(mask, expected)
        self.assertEqual(torch.count_nonzero(aux["attenuation"][..., mask]).item(), 0)
        torch.testing.assert_close(aux["gain"][..., mask], torch.ones(2, 8, 66), rtol=0, atol=0)
        torch.testing.assert_close(aux["attenuation"][..., ~mask],
                                  torch.full((2, 8, 191), torch.sigmoid(torch.tensor(-4.)).item()),
                                  rtol=1e-6, atol=1e-6)

    def test_protected_complex_bins_preserved_with_strong_gate(self):
        with torch.no_grad():
            self.model.encoder[-1].weight.normal_(std=.2)
            self.model.encoder[-1].bias.fill_(2.)
        x = random_window()
        pred, aux = self.model.forward_with_aux(x)
        before, after = (torch.fft.rfft(t, norm="ortho") for t in (x, pred))
        mask = self.model.protected_mask
        torch.testing.assert_close(after[..., mask], before[..., mask], rtol=1e-6, atol=1e-6)
        error = (after[..., mask] - before[..., mask]).abs().square().sum((1, 2))
        expected = error / before[..., mask].abs().square().sum((1, 2))
        torch.testing.assert_close(aux["protected_complex_energy_ratio"], expected,
                                  rtol=1e-6, atol=1e-12)
        self.assertTrue((aux["protected_complex_energy_ratio"] < 1e-6).all())

    def test_no_spectral_or_total_energy_amplification(self):
        with torch.no_grad():
            self.model.encoder[-1].weight.normal_(std=.5)
            self.model.encoder[-1].bias.fill_(1.)
        x = random_window()
        pred, aux = self.model.forward_with_aux(x)
        before, after = (torch.fft.rfft(t, norm="ortho").abs() for t in (x, pred))
        self.assertTrue((after <= before + 1e-6).all())
        self.assertTrue((pred.square().sum((1, 2)) <= x.square().sum((1, 2)) + 1e-6).all())
        self.assertTrue(((aux["attenuation"] >= 0) & (aux["attenuation"] <= 1)).all())
        self.assertTrue(((aux["gain"] >= 0) & (aux["gain"] <= 1)).all())

    def test_dc_and_zero_windows(self):
        for x in (torch.zeros(2, 8, 512), torch.full((2, 8, 512), 2.5)):
            pred, aux = self.model.forward_with_aux(x)
            torch.testing.assert_close(pred, x, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(aux["protected_complex_energy_ratio"],
                                      torch.zeros(2), rtol=0, atol=0)
        _, aux = self.model.forward_with_aux(torch.zeros(2, 8, 512))
        self.assertFalse(aux["protected_energy_defined"].any())

    def test_single_bins_and_nyquist(self):
        for bin_index in (0, 1, 16, 17, 35, 81, 82, 128, 256):
            with self.subTest(bin=bin_index):
                x = exact_bin_window(bin_index)
                pred, aux = self.model.forward_with_aux(x)
                gain = aux["gain"][0, 0, bin_index]
                torch.testing.assert_close(pred, x * gain, rtol=1e-6, atol=1e-6)

    def test_first_step_and_full_network_gradients(self):
        x = random_window().requires_grad_()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=.5)
        pred = self.model(x)
        loss = pred.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.encoder[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.encoder[-1].bias.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        self.model(x).square().mean().backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_protected_bin_loss_has_no_gate_gradient(self):
        x = random_window()
        pred = self.model(x)
        loss = torch.fft.rfft(pred, norm="ortho")[..., self.model.protected_mask].abs().square().mean()
        loss.backward()
        for parameter in self.model.parameters():
            self.assertLessEqual(parameter.grad.abs().max().item(), 1e-6)

    def test_state_save_load_and_repeat_deterministic(self):
        with torch.no_grad():
            self.model.encoder[-1].weight.normal_(std=.1)
        x = random_window()
        expected, aux = self.model.forward_with_aux(x)
        payload = io.BytesIO()
        torch.save(self.model.state_dict(), payload)
        payload.seek(0)
        restored = build_model()
        restored.load_state_dict(torch.load(payload, weights_only=True))
        for run in range(2):
            actual, actual_aux = restored.forward_with_aux(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for key in aux:
                torch.testing.assert_close(actual_aux[key], aux[key], rtol=0, atol=0)

    def test_forward_accepts_signal_only(self):
        for method in (SpectralDenoiserV10.forward, SpectralDenoiserV10.forward_with_aux):
            self.assertEqual(list(inspect.signature(method).parameters), ["self", "x"])
        with self.assertRaises(TypeError):
            self.model(random_window(), target=torch.zeros(1))

    def test_module_build_and_inference_do_not_read_disk(self):
        with patch("builtins.open", side_effect=AssertionError("unexpected file I/O")), \
             patch.object(Path, "open", side_effect=AssertionError("unexpected file I/O")):
            model = build_model()
            model(random_window(1))
            validate_config()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_forward_backward_and_cpu_agreement(self):
        old_tf32 = torch.backends.cudnn.allow_tf32
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_deterministic = torch.backends.cudnn.deterministic
        try:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            with torch.no_grad():
                self.model.encoder[-1].weight.normal_(std=.01)
            cuda_model = copy.deepcopy(self.model).cuda()
            x = random_window()
            cuda_x = x.cuda().requires_grad_()
            pred, aux = cuda_model.forward_with_aux(cuda_x)
            torch.testing.assert_close(pred.cpu(), self.model(x), rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(cuda_model(cuda_x), pred, rtol=0, atol=0)
            self.assertTrue((aux["protected_complex_energy_ratio"] <= 1e-6).all())
            pred.square().mean().backward()
            for parameter in cuda_model.parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        finally:
            torch.backends.cudnn.allow_tf32 = old_tf32
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.deterministic = old_deterministic


if __name__ == "__main__":
    unittest.main()
