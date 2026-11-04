"""Synthetic CPU tests only; no waveform/checkpoint reads or output artifacts.

Run only this file with python -B -m unittest tests.test_ssvep_denoise_v8 -v.
No trainers or earlier model modules are imported.
"""
from __future__ import annotations

import ast
import builtins
import copy
import inspect
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from eeg_tools import ssvep_denoise_v8 as core

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/learned_denoiser_v8_ssvep.json").read_text(encoding="utf-8"))
RATES = (256, 250)
torch.set_num_threads(4)


def wave(bin_index=60, batch=2, amplitude=1.0):
    # Build exact-bin synthetic sine in Fourier coordinates, avoiding float32
    # phase/trigonometric leakage into high bins at the protected boundaries.
    spectrum = torch.zeros(129, dtype=torch.complex64)
    spectrum[bin_index] = complex(0, -128*amplitude)
    signal = torch.fft.irfft(spectrum, n=256)
    return signal.view(1, 1, 256).expand(batch, 8, 256).clone()


def random_window(batch=2):
    return torch.randn(batch, 8, 256, dtype=torch.float32, generator=torch.Generator().manual_seed(772))


def aux_for(x, low=0.0):
    return {"low_delta": torch.full_like(x, low),
            "high_attenuation": torch.zeros(x.shape[0], 8, 88, dtype=torch.float32)}


def loss_case(conditions, out_errors=None, input_errors=None, target_scale=1.0, low=0.0):
    batch = len(conditions)
    target = (torch.arange(256, dtype=torch.float32)%2*2-1).view(1, 1, 256)
    target = target.expand(batch, 8, 256).clone()*target_scale
    if out_errors is None:
        out_errors = [1.0]*batch
    if input_errors is None:
        input_errors = [0.0 if c == "identity" else 2.0 for c in conditions]
    pred = target+torch.tensor(out_errors, dtype=torch.float32).view(batch, 1, 1)
    noisy = target+torch.tensor(input_errors, dtype=torch.float32).view(batch, 1, 1)
    identity = torch.tensor([c == "identity" for c in conditions], dtype=torch.bool)
    return pred, target, noisy, identity, aux_for(pred, low)


def loss_for(conditions, **kwargs):
    return core.compute_loss(*loss_case(conditions, **kwargs), CFG, conditions=conditions)


def summary(ratio=0.8, identity=0.01):
    return {name: {"mean_input_normalized_mse": 0.0 if name == "identity" else 2.0,
                   "mean_output_normalized_mse": identity if name == "identity" else 2.0*ratio}
            for name in core.CONDITIONS}


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7077)

    def test_constants_and_parameter_count(self):
        self.assertEqual(core.DOMAINS, ("Nakanishi2015", "Wearable"))
        self.assertEqual(core.RATES, CFG["sample_rates_hz"])
        self.assertEqual(core.CONDITIONS, tuple(CFG["conditions"]))
        for rate in RATES:
            self.assertEqual(core.model_parameter_count(core.build_model(CFG, rate)), 90666)
            self.assertEqual(core.PARAMETER_COUNT, 90666)

    def test_build_cpu_float32_both_rates(self):
        for rate in RATES:
            model = core.build_model(CFG, rate)
            self.assertEqual(model.sample_rate_hz, rate)
            self.assertTrue(all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()))
            self.assertTrue(all(b.device.type == "cpu" for b in model.buffers()))

    def test_build_overrides_default_dtype(self):
        old = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            model = core.build_model(CFG, 250)
            self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
            self.assertTrue(torch.equal(model(random_window()), random_window()))
        finally:
            torch.set_default_dtype(old)

    def test_build_overrides_default_device(self):
        old = torch.get_default_device()
        try:
            torch.set_default_device("meta")
            model = core.build_model(CFG, 256)
            self.assertTrue(all(p.device.type == "cpu" for p in model.parameters()))
            self.assertTrue(all(b.device.type == "cpu" for b in model.buffers()))
        finally:
            torch.set_default_device(old)

    def test_zero_initialized_heads(self):
        for rate in RATES:
            model = core.build_model(CFG, rate)
            for head in (model.low_head, model.high_gate_head):
                self.assertEqual(torch.count_nonzero(head.weight).item(), 0)
                self.assertEqual(torch.count_nonzero(head.bias).item(), 0)

    def test_exact_identity_train_both_rates(self):
        for rate in RATES:
            model = core.build_model(CFG, rate).train()
            x = random_window()
            self.assertTrue(torch.equal(model(x), x))

    def test_exact_identity_eval_both_rates(self):
        for rate in RATES:
            model = core.build_model(CFG, rate).eval()
            x = random_window()
            pred, aux = model.forward_with_aux(x)
            self.assertTrue(torch.equal(pred, x))
            self.assertEqual(torch.count_nonzero(aux["low_delta"]).item(), 0)
            self.assertEqual(torch.count_nonzero(aux["high_attenuation"]).item(), 0)

    def test_exact_identity_zero_constant_and_large_inputs(self):
        for rate in RATES:
            model = core.build_model(CFG, rate)
            for x in (torch.zeros(1, 8, 256), torch.full((1, 8, 256), 4.0), random_window()*1e4):
                self.assertTrue(torch.equal(model(x), x))

    def test_forward_returns_tensor_and_aux_exact_schema(self):
        for rate in RATES:
            model = core.build_model(CFG, rate)
            x = random_window(3)
            self.assertIsInstance(model(x), torch.Tensor)
            pred, aux = model.forward_with_aux(x)
            self.assertEqual(pred.shape, (3, 8, 256))
            self.assertEqual(pred.dtype, torch.float32)
            self.assertEqual(set(aux), {"low_delta", "high_attenuation"})
            self.assertEqual(aux["low_delta"].shape, pred.shape)
            self.assertEqual(aux["high_attenuation"].shape, (3, 8, 88))
            self.assertTrue(all(t.dtype == torch.float32 for t in aux.values()))

    def test_forward_and_aux_predictions_match(self):
        model = core.build_model(CFG, 256)
        with torch.no_grad():
            model.low_head.bias.fill_(0.1)
            model.high_gate_head.bias.fill_(0.3)
        x = random_window()
        self.assertTrue(torch.equal(model(x), model.forward_with_aux(x)[0]))

    def test_forward_has_no_labels_or_return_aux_argument(self):
        self.assertEqual(list(inspect.signature(core.SSVEPDenoiserV8.forward).parameters), ["self", "x"])
        self.assertEqual(list(inspect.signature(core.SSVEPDenoiserV8.forward_with_aux).parameters), ["self", "x"])
        model = core.build_model(CFG, 250)
        for kw in ("condition", "conditions", "identity", "target", "subject", "return_aux"):
            with self.assertRaises(TypeError):
                model(random_window(), **{kw: True})

    def test_masks_follow_native_rates_and_strict_boundaries(self):
        for rate in RATES:
            model = core.build_model(CFG, rate)
            freq = torch.fft.rfftfreq(256, d=1/rate)
            self.assertTrue(torch.equal(model.low_mask, freq < 8))
            self.assertTrue(torch.equal(model.high_mask, freq > 40))
            self.assertFalse(bool((model.low_mask & model.high_mask).any()))
            if rate == 256:
                self.assertFalse(model.low_mask[8].item())
                self.assertFalse(model.high_mask[40].item())
                self.assertTrue(model.high_mask[41].item())

    def test_high_mapping_is_exact_frozen_raw_clamp(self):
        raw = torch.linspace(-0.2, 1.0, 88).view(1, 1, 88).expand(16, 1, 88).clone()
        for rate in RATES:
            model = core.build_model(CFG, rate)
            with patch.object(model.high_gate_head, "forward", return_value=raw):
                _, aux = model.forward_with_aux(random_window())
            expected = raw.clamp(0, 0.8).reshape(2, 8, 88)
            self.assertTrue(torch.equal(aux["high_attenuation"], expected))

    def test_high_gate_nonnegative_bounded_channel_and_bin_specific(self):
        model = core.build_model(CFG, 250)
        with torch.no_grad():
            model.high_gate_head.weight.zero_()
            model.high_gate_head.weight[0, -1, 0] = 0.2
        x = wave(60)
        x[:, 0] *= 0.1
        x[:, -1] *= 3.0
        gate = model.forward_with_aux(x)[1]["high_attenuation"]
        self.assertTrue(torch.isfinite(gate).all().item())
        self.assertTrue(((gate >= 0) & (gate <= 0.8)).all().item())
        self.assertNotEqual(gate[0, 0, 19].item(), gate[0, -1, 19].item())
        self.assertGreater(gate[0, 0, 19].item(), gate[0, 0, 20].item())

    def test_high_branch_per_bin_scale_not_a_shared_window_scalar(self):
        for rate in RATES:
            model = core.build_model(CFG, rate)
            with torch.no_grad():
                model.high_gate_head.weight[0, -1, 0] = 0.2
            x = random_window()
            pred, aux = model.forward_with_aux(x)
            gate = aux["high_attenuation"]
            self.assertGreater((gate.max(-1).values - gate.min(-1).values).min().item(), 0)
            expected = torch.fft.rfft(x)[..., model.high_mask]*(1-gate)
            torch.testing.assert_close(torch.fft.rfft(pred)[..., model.high_mask],
                                       expected, atol=2e-5, rtol=2e-5)

    def test_high_frequency_no_amplification_even_with_active_low_head(self):
        for rate in RATES:
            model = core.build_model(CFG, rate)
            with torch.no_grad():
                model.high_gate_head.bias.fill_(0.3)
                model.low_head.weight.normal_(0, 0.03)
            x = random_window()
            before = torch.fft.rfft(x).abs()[..., model.high_mask]
            after = torch.fft.rfft(model(x)).abs()[..., model.high_mask]
            self.assertTrue((after <= before+3e-5).all().item())

    def test_high_max_attenuation_of_single_bin(self):
        for rate in RATES:
            model = core.build_model(CFG, rate)
            with torch.no_grad():
                model.high_gate_head.bias.fill_(10)
            x = wave()
            torch.testing.assert_close(model(x), 0.2*x, atol=5e-6, rtol=5e-6)

    def test_high_branch_does_not_change_low_or_protected_bins(self):
        for rate in RATES:
            model = core.build_model(CFG, rate)
            with torch.no_grad():
                model.high_gate_head.bias.fill_(1)
            x = random_window()
            residual = torch.fft.rfft(model(x)-x)
            self.assertLess(residual[..., ~model.high_mask].abs().max().item(), 2e-5)

    def test_protected_8_to_40_residual_zero_both_heads_active(self):
        for rate in RATES:
            model = core.build_model(CFG, rate)
            with torch.no_grad():
                model.low_head.weight.normal_(0, 0.2)
                model.low_head.bias.fill_(0.2)
                model.high_gate_head.bias.fill_(0.4)
            x = random_window()
            protected = ~(model.low_mask | model.high_mask)
            residual = torch.fft.rfft(model(x)-x)
            self.assertLess(residual[..., protected].abs().max().item(), 2e-5)

    def test_task_boundary_bins_8_and_40_not_attenuated(self):
        model = core.build_model(CFG, 256)
        with torch.no_grad():
            model.high_gate_head.bias.fill_(10)
        for bin_index in (8, 40):
            x = wave(bin_index)
            torch.testing.assert_close(model(x), x, atol=1e-6, rtol=1e-6)

    def test_low_branch_exact_preprojection_formula(self):
        for rate in RATES:
            model = core.build_model(CFG, rate)
            with torch.no_grad():
                model.low_head.weight.normal_(0, 0.2)
            x = random_window()
            features = model.activation(model.blocks(model.input(x)))
            delta_pre = 0.5*torch.tanh(model.low_head(features))
            self.assertTrue((delta_pre.abs() <= 0.5).all().item())
            expected = torch.fft.irfft(torch.fft.rfft(delta_pre)*model.low_mask, n=256)
            aux = model.forward_with_aux(x)[1]
            self.assertTrue(torch.equal(aux["low_delta"], expected))
            self.assertLess(torch.fft.rfft(aux["low_delta"])[..., ~model.low_mask].abs().max().item(), 2e-5)

    def test_zero_high_gate_head_finite_nonzero_gradient_at_clamp_boundary(self):
        for rate in RATES:
            model = core.build_model(CFG, rate)
            x = wave()
            (model(x)-0.8*x).square().mean().backward()
            for parameter in (model.high_gate_head.weight, model.high_gate_head.bias):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
            self.assertTrue((model.high_gate_head.bias.grad < 0).all().item())

    def test_inclusive_clamp_boundary_gradient_is_one(self):
        x = torch.tensor(0.0, requires_grad=True)
        core._inclusive_attenuation_clamp(x).backward()
        self.assertEqual(x.grad.item(), 1.0)

    def test_low_head_finite_nonzero_gradient_at_zero(self):
        for rate in RATES:
            model = core.build_model(CFG, rate)
            x = random_window()
            (model(x)-(x+0.1*wave(3)+0.02)).square().mean().backward()
            for p in (model.low_head.weight, model.low_head.bias):
                self.assertTrue(torch.isfinite(p.grad).all().item())
                self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_loss_backward_both_branches_at_initial_identity(self):
        for rate in RATES:
            model = core.build_model(CFG, rate)
            noisy = wave()+0.5*wave(3)
            target = 0.8*wave()+0.7*wave(3)
            pred, aux = model.forward_with_aux(noisy)
            losses = core.compute_loss(pred, target, noisy, torch.zeros(2, dtype=torch.bool), aux, CFG,
                                       conditions=["high_frequency_burst"]*2)
            losses["total"].backward()
            for p in (model.low_head.weight, model.high_gate_head.weight):
                self.assertTrue(torch.isfinite(p.grad).all().item())
                self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_shared_tcn_gradients_after_heads_move(self):
        model = core.build_model(CFG, 256)
        with torch.no_grad():
            model.low_head.weight.normal_(0, 0.05)
            model.high_gate_head.weight.normal_(0, 0.005)
            model.high_gate_head.bias.fill_(0.2)
        model(random_window()).square().mean().backward()
        for p in (model.input.weight, model.blocks[0].net[0].weight):
            self.assertTrue(torch.isfinite(p.grad).all().item())
            self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_noncontiguous_input_supported(self):
        x = random_window().transpose(1, 2).contiguous().transpose(1, 2)
        self.assertFalse(x.is_contiguous())
        self.assertTrue(torch.equal(core.build_model(CFG, 256)(x), x))

    def test_state_roundtrip_without_file_io(self):
        a, b = core.build_model(CFG, 250), core.build_model(CFG, 250)
        with torch.no_grad():
            a.high_gate_head.bias.fill_(0.3)
        b.load_state_dict(a.state_dict(), strict=True)
        self.assertTrue(torch.equal(a(random_window()), b(random_window())))
        self.assertNotIn("low_mask", a.state_dict())
        self.assertNotIn("high_mask", a.state_dict())

    def test_build_does_not_mutate_config(self):
        original = copy.deepcopy(CFG)
        core.build_model(CFG, 256)
        self.assertEqual(CFG, original)

    def test_fresh_models_independent_state(self):
        a, b = core.build_model(CFG, 256), core.build_model(CFG, 256)
        self.assertNotEqual(a.low_head.weight.data_ptr(), b.low_head.weight.data_ptr())
        with torch.no_grad():
            a.high_gate_head.bias.fill_(1)
        self.assertEqual(b.high_gate_head.bias.abs().sum().item(), 0)

    def test_invalid_rate_rejection(self):
        for rate in (None, 0, 128, 251, 500, 256.0, True, "256"):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                core.build_model(CFG, rate)

    def test_invalid_frozen_model_settings_rejected(self):
        for name, value in (("hidden_channels", 16), ("kernel_size", 5), ("dilations", [1]),
                            ("group_norm_groups", 4), ("dropout", 0.1),
                            ("low_residual_limit_normalized", 0.25),
                            ("high_attenuation_mapping", "sigmoid"), ("high_attenuation_max", 1.0),
                            ("protected_band_hz", [7., 40.]), ("post_projection_clip", True),
                            ("zero_initialize_high_gate", False)):
            cfg = copy.deepcopy(CFG)
            cfg["model"][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                core.build_model(cfg, 256)

    def test_formal_version_shape_and_condition_contract(self):
        for name, value in (("status", "draft"), ("model_version", "v6"), ("model_type", "wrong"),
                            ("input_channels", 4), ("window_samples", 128), ("schema_version", True),
                            ("sample_rates_hz", {"Wearable": 256}), ("conditions", ["identity"])):
            cfg = copy.deepcopy(CFG)
            cfg[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                core.build_model(cfg, 250)

    def test_missing_config_or_wrong_type_rejected(self):
        for cfg in (None, [], {}, {"model": {}}):
            with self.assertRaises(ValueError):
                core.build_model(cfg, 256)

    def test_forward_bad_shapes_and_non_tensor_rejected(self):
        model = core.build_model(CFG, 256)
        for x in (None, [], torch.zeros(8, 256), torch.zeros(0, 8, 256),
                  torch.zeros(1, 7, 256), torch.zeros(1, 8, 255), torch.zeros(1, 8, 256, 1)):
            with self.assertRaises(ValueError):
                model(x)

    def test_forward_bad_dtypes_and_layout_rejected(self):
        model = core.build_model(CFG, 256)
        for x in (random_window().double(), random_window().half(), random_window().to(torch.int64),
                  random_window().to(torch.complex64), random_window().to_sparse()):
            with self.assertRaises(ValueError):
                model(x)

    def test_forward_nonfinite_rejected(self):
        model = core.build_model(CFG, 250)
        for value in (float("nan"), float("inf"), -float("inf")):
            x = random_window()
            x[0, 0, 0] = value
            with self.assertRaises(ValueError):
                model(x)

    def test_nonfinite_branch_parameters_rejected(self):
        for name in ("low_head", "high_gate_head"):
            model = core.build_model(CFG, 250)
            with torch.no_grad():
                getattr(model, name).weight[0, 0] = float("nan")
            with self.assertRaises(ValueError):
                model(random_window())

    def test_high_mapping_matches_log_magnitude_spectral_encoder(self):
        for rate in RATES:
            model = core.build_model(CFG, rate)
            with torch.no_grad():
                model.high_gate_head.weight.normal_(0, 0.03)
                model.high_gate_head.bias.fill_(0.2)
            x = random_window()
            magnitude = torch.log1p(torch.fft.rfft(x)[..., model.high_mask].abs()/math.sqrt(256.0))
            encoded = model.high_spectral_encoder(magnitude.reshape(-1, 1, 88))
            inputs = torch.cat([encoded, magnitude.reshape(-1, 1, 88)], dim=1)
            expected = model.high_gate_head(inputs).clamp(0, 0.8).reshape(2, 8, 88)
            actual = model.forward_with_aux(x)[1]["high_attenuation"]
            self.assertTrue(torch.equal(actual, expected))

    def test_low_projection_has_no_postprojection_clip(self):
        model = core.build_model(CFG, 256)
        logits = torch.full((2, 8, 256), -20.0)
        logits[..., :128] = 20.0
        with patch.object(model.low_head, "forward", return_value=logits):
            delta = model.forward_with_aux(random_window())[1]["low_delta"]
        expected = torch.fft.irfft(torch.fft.rfft(0.5*logits.tanh())*model.low_mask, n=256)
        self.assertTrue(torch.equal(delta, expected))
        self.assertGreater(delta.abs().max().item(), 0.5)

    def test_model_rejects_mismatched_device_or_converted_dtype(self):
        model = core.build_model(CFG, 256)
        with self.assertRaises(ValueError):
            model(torch.empty(1, 8, 256, device="meta", dtype=torch.float32))
        model.double()
        with self.assertRaises(ValueError):
            model(random_window())
    def test_input_not_modified_in_place(self):
        model = core.build_model(CFG, 250)
        x = random_window()
        before = x.clone()
        with torch.no_grad():
            model.high_gate_head.bias.fill_(0.3)
        model(x)
        self.assertTrue(torch.equal(before, x))



class LossTests(unittest.TestCase):
    def test_loss_keyword_only_conditions(self):
        args = loss_case(["transient"])
        with self.assertRaises(TypeError):
            core.compute_loss(*args, CFG)
        with self.assertRaises(TypeError):
            core.compute_loss(*args, CFG, ["transient"])

    def test_hf4_other1_weighted_relative_mse(self):
        losses = loss_for(["high_frequency_burst", "transient"], out_errors=[2., 1.])
        self.assertAlmostEqual(losses["corrupted"].item(), (4*1+0.25)/5, places=6)
        self.assertEqual(losses["corrupted_sum"].item(), 4.25)
        self.assertEqual(losses["corrupted_weight_sum"].item(), 5.0)
        self.assertEqual(losses["corrupted_count"], 2)

    def test_other_corrupted_weight_one_all_three_families(self):
        losses = loss_for(["transient", "low_frequency_drift", "common_mode"], out_errors=[1., 2., 3.])
        self.assertAlmostEqual(losses["corrupted"].item(), (0.25+1+2.25)/3, places=6)
        self.assertEqual(losses["corrupted_weight_sum"].item(), 3)

    def test_identity60_no_input_ratio_division(self):
        losses = loss_for(["identity"], out_errors=[2.])
        self.assertEqual(losses["identity"].item(), 4)
        self.assertAlmostEqual(losses["total"].item(), 240.004, places=4)
        self.assertEqual(losses["corrupted"].item(), 0)

    def test_hf_low_branch_penalty20_only_hf(self):
        args = list(loss_case(["high_frequency_burst", "transient"], out_errors=[0., 0.]))
        args[4]["low_delta"][0].fill_(2)
        args[4]["low_delta"][1].fill_(100)
        losses = core.compute_loss(*args, CFG, conditions=["high_frequency_burst", "transient"])
        self.assertEqual(losses["hf_low_branch_change"].item(), 4)
        self.assertAlmostEqual(losses["total"].item(), 80.004, places=4)

    def test_hf_low_penalty_is_not_target_variance_scaled(self):
        losses = loss_for(["high_frequency_burst"], out_errors=[0.], target_scale=4., low=2.)
        self.assertEqual(losses["hf_low_branch_change"].item(), 4)

    def test_all_change001_in_already_normalized_coordinates(self):
        losses = loss_for(["transient", "identity"], out_errors=[0., 0.], target_scale=4.)
        self.assertEqual(losses["change"].item(), 2)
        self.assertAlmostEqual(losses["total"].item(), 0.002, places=7)

    def test_full_five_condition_formula(self):
        labels = list(core.CONDITIONS)
        losses = loss_for(labels, low=0.2)
        expected = 0.25+60*1+20*0.04+0.001*1
        self.assertAlmostEqual(losses["total"].item(), expected, places=5)
        self.assertEqual(losses["case_count"], 5)
        self.assertEqual(losses["corrupted_count"], 4)
        self.assertEqual(losses["identity_count"], 1)
        self.assertEqual(losses["hf_count"], 1)
        self.assertEqual(losses["corrupted_weight_sum"].item(), 7)

    def test_relative_input_floor001(self):
        losses = loss_for(["transient"], out_errors=[0.1], input_errors=[0.])
        self.assertAlmostEqual(losses["corrupted"].item(), 1., places=5)

    def test_constant_target_variance_epsilon(self):
        args = list(loss_case(["identity"], out_errors=[0.]))
        args[1].zero_()
        args[2].zero_()
        args[0].fill_(0.01)
        losses = core.compute_loss(*args, CFG, conditions=["identity"])
        self.assertAlmostEqual(losses["identity"].item(), 10000, delta=0.02)
        self.assertTrue(torch.isfinite(losses["total"]).item())

    def test_target_demeaned_variance_not_mean_square(self):
        args = list(loss_case(["identity"], out_errors=[1.]))
        for tensor in args[:3]:
            tensor.add_(10)
        losses = core.compute_loss(*args, CFG, conditions=["identity"])
        self.assertEqual(losses["identity"].item(), 1)

    def test_error_variance_is_per_channel(self):
        args = list(loss_case(["identity"], out_errors=[1.]))
        scales = torch.arange(1, 9, dtype=torch.float32).view(1, 8, 1)
        args[1] *= scales
        args[0] = args[1]+1
        args[2] = args[1].clone()
        losses = core.compute_loss(*args, CFG, conditions=["identity"])
        self.assertAlmostEqual(losses["identity"].item(), (1/scales.square()).mean().item(), places=6)

    def test_no_identity_and_no_hf_groups_zero(self):
        losses = loss_for(["common_mode"])
        self.assertEqual(losses["identity"].item(), 0)
        self.assertEqual(losses["hf_low_branch_change"].item(), 0)
        self.assertEqual(losses["identity_count"], 0)
        self.assertEqual(losses["hf_count"], 0)

    def test_all_identity_exact_zero_loss(self):
        losses = loss_for(["identity", "identity"], out_errors=[0., 0.])
        for name in ("total", "corrupted", "identity", "hf_low_branch_change", "change"):
            self.assertEqual(losses[name].item(), 0)
        self.assertEqual(losses["corrupted_weight_sum"].item(), 0)

    def test_loss_condition_code_tensor_and_list_equivalent(self):
        labels = list(core.CONDITIONS)
        args = loss_case(labels)
        reference = core.compute_loss(*args, CFG, conditions=labels)
        for codes in (list(range(5)), tuple(range(5)), torch.arange(5, dtype=torch.int64)):
            actual = core.compute_loss(*args, CFG, conditions=codes)
            self.assertTrue(torch.equal(reference["total"], actual["total"]))

    def test_hf_mask_changes_loss_not_predictions(self):
        model = core.build_model(CFG, 250)
        noisy = random_window()
        pred, aux = model.forward_with_aux(noisy)
        target = noisy+0.1
        identity = torch.zeros(2, dtype=torch.bool)
        a = core.compute_loss(pred, target, noisy, identity, aux, CFG,
                              conditions=["high_frequency_burst", "transient"])
        b = core.compute_loss(pred, target, noisy, identity, aux, CFG,
                              conditions=["transient", "common_mode"])
        self.assertEqual(a["corrupted_weight_sum"].item(), 5)
        self.assertEqual(b["corrupted_weight_sum"].item(), 2)
        self.assertTrue(torch.equal(pred, model(noisy)))

    def test_loss_pred_and_low_delta_gradients(self):
        args = list(loss_case(["high_frequency_burst"]))
        args[0].requires_grad_()
        args[4]["low_delta"].fill_(0.2).requires_grad_()
        loss = core.compute_loss(*args, CFG, conditions=["high_frequency_burst"])
        loss["total"].backward()
        for value in (args[0], args[4]["low_delta"]):
            self.assertTrue(torch.isfinite(value.grad).all().item())
            self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_loss_does_not_mutate_input_aux_or_config(self):
        args = loss_case(["transient"])
        tensors = [t.clone() for t in args[:4]]
        aux = {k: v.clone() for k, v in args[4].items()}
        cfg = copy.deepcopy(CFG)
        core.compute_loss(*args, CFG, conditions=["transient"])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(args[:4], tensors)))
        self.assertTrue(all(torch.equal(args[4][k], v) for k, v in aux.items()))
        self.assertEqual(CFG, cfg)

    def test_loss_wrong_shape_dtype_type_rejected(self):
        originals = loss_case(["transient"])
        for index in range(3):
            for invalid in (None, originals[index].double(), originals[index][..., :255],
                            originals[index].to_sparse()):
                args = list(originals)
                args[index] = invalid
                with self.subTest(index=index), self.assertRaises(ValueError):
                    core.compute_loss(*args, CFG, conditions=["transient"])

    def test_loss_rejects_mismatched_devices(self):
        for index in (1, 2, 3):
            args = list(loss_case(["transient"]))
            args[index] = torch.empty_like(args[index], device="meta")
            with self.assertRaises(ValueError):
                core.compute_loss(*args, CFG, conditions=["transient"])
        for name in ("low_delta", "high_attenuation"):
            args = list(loss_case(["transient"]))
            args[4][name] = torch.empty_like(args[4][name], device="meta")
            with self.assertRaises(ValueError):
                core.compute_loss(*args, CFG, conditions=["transient"])
        args = loss_case(["transient"])
        with self.assertRaises(ValueError):
            core.compute_loss(*args, CFG, conditions=torch.empty(1, device="meta", dtype=torch.int64))
    def test_loss_empty_batch_rejected(self):
        x = torch.zeros(0, 8, 256)
        with self.assertRaises(ValueError):
            core.compute_loss(x, x, x, torch.zeros(0, dtype=torch.bool), aux_for(x), CFG, conditions=[])

    def test_loss_identity_must_be_bool_vector(self):
        originals = loss_case(["transient"])
        for invalid in (False, torch.zeros(1), torch.zeros(1, 1, dtype=torch.bool)):
            args = list(originals)
            args[3] = invalid
            with self.assertRaises(ValueError):
                core.compute_loss(*args, CFG, conditions=["transient"])

    def test_loss_identity_condition_consistency(self):
        args = loss_case(["identity"])
        with self.assertRaises(ValueError):
            core.compute_loss(*args, CFG, conditions=["high_frequency_burst"])

    def test_loss_bad_condition_strings_codes_rejected(self):
        args = loss_case(["transient"])
        for labels in (None, "transient", [], ["unknown"], [False], [1.9], [-1], [5],
                       torch.tensor([1.]), torch.tensor([[1]]), ["transient", "common_mode"]):
            with self.subTest(labels=str(labels)), self.assertRaises(ValueError):
                core.compute_loss(*args, CFG, conditions=labels)

    def test_loss_missing_aux_rejected(self):
        original = loss_case(["transient"])
        for aux in (None, {}, {"low_delta": original[4]["low_delta"]},
                    {"high_attenuation": original[4]["high_attenuation"]}):
            args = list(original)
            args[4] = aux
            with self.assertRaises(ValueError):
                core.compute_loss(*args, CFG, conditions=["transient"])

    def test_loss_invalid_low_delta_rejected(self):
        originals = loss_case(["transient"])
        for value in (None, torch.zeros(1, 8, 255), torch.zeros(1, 8, 256, dtype=torch.float64)):
            args = list(originals)
            args[4] = dict(originals[4], low_delta=value)
            with self.assertRaises(ValueError):
                core.compute_loss(*args, CFG, conditions=["transient"])

    def test_loss_invalid_gate_shape_range_dtype_rejected(self):
        originals = loss_case(["transient"])
        for value in (None, torch.zeros(1, 8), torch.zeros(1, 1, 1), torch.zeros(1, 8, 1), torch.zeros(1, 8, 88).double(),
                      torch.full((1, 8, 88), -0.001), torch.full((1, 8, 88), 0.8001)):
            args = list(originals)
            args[4] = dict(originals[4], high_attenuation=value)
            with self.assertRaises(ValueError):
                core.compute_loss(*args, CFG, conditions=["transient"])

    def test_loss_nonfinite_each_tensor_and_aux_rejected(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            for index in range(3):
                args = list(loss_case(["transient"]))
                args[index][0, 0, 0] = invalid
                with self.assertRaises(ValueError):
                    core.compute_loss(*args, CFG, conditions=["transient"])
            for name in ("low_delta", "high_attenuation"):
                args = list(loss_case(["transient"]))
                args[4][name][0, 0, 0] = invalid
                with self.assertRaises(ValueError):
                    core.compute_loss(*args, CFG, conditions=["transient"])

    def test_loss_overflow_from_finite_inputs_rejected(self):
        args = list(loss_case(["identity"]))
        args[0].fill_(1e30)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            core.compute_loss(*args, CFG, conditions=["identity"])

    def test_frozen_loss_weight_and_nonfinite_config_rejected(self):
        for name in core._LOSS:
            for value in (float("nan"), float("inf"), -1.0, True):
                cfg = copy.deepcopy(CFG)
                cfg["loss"][name] = value
                with self.assertRaises(ValueError):
                    core.compute_loss(*loss_case(["transient"]), cfg, conditions=["transient"])

    def test_accumulator_weighted_unequal_batches_matches_full(self):
        labels = ["identity", "high_frequency_burst", "transient", "common_mode", "identity"]
        args = loss_case(labels, out_errors=[0.1, 1., 2., 3., 0.2], low=0.1)
        full = core.compute_loss(*args, CFG, conditions=labels)
        accumulator = core.LossAccumulator()
        for begin, end in ((0, 1), (1, 3), (3, 5)):
            batch = [v[begin:end] for v in args[:4]]
            batch.append({k: v[begin:end] for k, v in args[4].items()})
            accumulator.update(core.compute_loss(*batch, CFG, conditions=labels[begin:end]))
        result = accumulator.result()
        for name in ("total", "corrupted", "identity", "hf_low_branch_change", "change"):
            self.assertAlmostEqual(result[name], full[name].item(), places=5)
        self.assertEqual(result["case_count"], 5)
        self.assertEqual(result["corrupted_weight_sum"], 6)

    def test_accumulator_empty_and_invalid_rejected_without_mutation(self):
        acc = core.LossAccumulator()
        with self.assertRaises(ValueError):
            acc.result()
        valid = loss_for(["transient"])
        for invalid in (dict(valid, change_sum=torch.tensor(float("nan"))),
                        dict(valid, case_count=True), dict(valid, corrupted_weight_sum=torch.tensor(4.))):
            with self.assertRaises(ValueError):
                acc.update(invalid)
        self.assertEqual(acc.counts["case"], 0)


class SelectionAndBoundaryTests(unittest.TestCase):
    def test_inclusive_clamp_forward_equals_stock_mapping(self):
        logits = torch.cat((torch.linspace(-10, 10, 1001), torch.tensor([0., 0.1, -0.1])))
        raw = logits.tanh()
        self.assertTrue(torch.equal(core._inclusive_attenuation_clamp(raw), raw.clamp(0, 0.8)))

    def test_inclusive_clamp_both_boundaries_inside_gradient(self):
        raw = torch.tensor([0., 0.8], requires_grad=True)
        core._inclusive_attenuation_clamp(raw).sum().backward()
        self.assertTrue(torch.equal(raw.grad, torch.ones(2)))

    def test_negative_and_over_bound_attenuation_gradient_zero(self):
        raw = torch.tensor([-1., -0.01, 0.81, 1.], requires_grad=True)
        core._inclusive_attenuation_clamp(raw).sum().backward()
        self.assertTrue(torch.equal(raw.grad, torch.zeros(4)))
        for rate in RATES:
            for bias in (-0.1, 1.0):
                model = core.build_model(CFG, rate)
                with torch.no_grad():
                    model.high_gate_head.bias.fill_(bias)
                model.forward_with_aux(random_window())[1]["high_attenuation"].sum().backward()
                for p in (model.high_gate_head.weight, model.high_gate_head.bias):
                    self.assertTrue(torch.isfinite(p.grad).all().item())
                    self.assertEqual(p.grad.abs().sum().item(), 0)

    def test_selection_key_matches_v6_order(self):
        self.assertEqual(core.selection_key(summary(), {"all_passed": True}, 0.3, 18),
                         (0, 0.8, 0.01, 0.3, 18))
        self.assertEqual(core.selection_key(summary(), {"all_passed": False}, 0.3, 1)[0], 1)

    def test_selection_eligible_first_and_earliest_epoch(self):
        eligible = core.selection_key(summary(1.5), {"all_passed": True}, 10., 18)
        failed = core.selection_key(summary(0.1), {"all_passed": False}, 0., 1)
        self.assertLess(eligible, failed)
        self.assertLess(core.selection_key(summary(), {"all_passed": True}, 1., 1),
                        core.selection_key(summary(), {"all_passed": True}, 1., 18))

    def test_selection_worst_ratio_of_mean_errors(self):
        data = summary()
        data["common_mode"]["mean_input_normalized_mse"] = 4.
        data["common_mode"]["mean_output_normalized_mse"] = 4.8
        self.assertEqual(core.selection_key(data, {"all_passed": True}, 0., 2)[1], 1.2)

    def test_selection_identity_and_total_tie_breakers(self):
        a = core.selection_key(summary(identity=0.001), {"all_passed": True}, 10., 18)
        b = core.selection_key(summary(identity=0.01), {"all_passed": True}, 0., 1)
        self.assertLess(a, b)
        self.assertLess(core.selection_key(summary(), {"all_passed": True}, 0.1, 18),
                        core.selection_key(summary(), {"all_passed": True}, 0.2, 1))

    def test_selection_epoch1_through18_only(self):
        for epoch in range(1, 19):
            self.assertEqual(core.selection_key(summary(), {"all_passed": True}, 0., epoch)[-1], epoch)
        for epoch in (0, 19, -1, True, 1.0, "1"):
            with self.assertRaises(ValueError):
                core.selection_key(summary(), {"all_passed": True}, 0., epoch)

    def test_selection_invalid_errors_gate_and_missing_condition(self):
        for value in (float("nan"), float("inf"), -1., True, "1"):
            data = summary()
            data["transient"]["mean_output_normalized_mse"] = value
            with self.assertRaises(ValueError):
                core.selection_key(data, {"all_passed": True}, 0., 1)
            with self.assertRaises(ValueError):
                core.selection_key(summary(), {"all_passed": True}, value, 1)
        for gate in ({}, {"all_passed": 1}, None):
            with self.assertRaises(ValueError):
                core.selection_key(summary(), gate, 0., 1)
        data = summary()
        data.pop("common_mode")
        with self.assertRaises(ValueError):
            core.selection_key(data, {"all_passed": True}, 0., 1)

    def test_selection_invalid_baseline_identity_and_overflow(self):
        for value in (0., -1., float("nan"), float("inf")):
            data = summary()
            data["transient"]["mean_input_normalized_mse"] = value
            with self.assertRaises(ValueError):
                core.selection_key(data, {"all_passed": True}, 0., 1)
        for value in (-1., float("nan"), float("inf"), None):
            with self.assertRaises(ValueError):
                core.selection_key(summary(identity=value), {"all_passed": True}, 0., 1)
        data = summary()
        data["transient"].update(mean_input_normalized_mse=1e-300, mean_output_normalized_mse=1e300)
        with self.assertRaises(ValueError):
            core.selection_key(data, {"all_passed": True}, 0., 1)

    def test_compute_paths_never_open_save_load_or_publish(self):
        model = core.build_model(CFG, 256)
        with patch.object(builtins, "open", side_effect=AssertionError("no file I/O")), \
             patch.object(torch, "load", side_effect=AssertionError("no checkpoint reads")), \
             patch.object(torch, "save", side_effect=AssertionError("no output writes")):
            x = random_window()
            pred, aux = model.forward_with_aux(x)
            core.compute_loss(pred, x, x, torch.ones(2, dtype=torch.bool), aux, CFG,
                              conditions=["identity"]*2)
            core.selection_key(summary(), {"all_passed": True}, 0., 1)

    def test_module_has_no_io_or_legacy_data_imports(self):
        source = inspect.getsource(core)
        tree = ast.parse(source)
        forbidden = {"open", "load", "save", "read_bytes", "read_text", "write_text", "write_bytes",
                     "load_training_records", "preflight", "publish"}
        calls = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                 for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, (ast.Name, ast.Attribute))}
        self.assertFalse(calls & forbidden)
        self.assertNotIn("sigmoid", source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse(node.module and any(v in node.module for v in
                                                      ("v5", "v6", "dataset", "trainer", "pathlib")))

    def test_test_file_contains_no_file_writes_or_other_suites(self):
        source = Path(__file__).read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, ("write_text", "write_bytes", "mkdir", "save", "load"))
            if isinstance(node, ast.ImportFrom):
                self.assertFalse(node.module and node.module.startswith("scripts"))


if __name__ == "__main__":
    unittest.main()


