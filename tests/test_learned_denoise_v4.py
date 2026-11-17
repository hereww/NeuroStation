"""CPU-only v4 fixtures; no dataset, held-out access, optimizer or training run."""
from __future__ import annotations
from tests.synthetic_research_fixtures import research_contract

from copy import deepcopy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from eeg_tools import learned_denoise_v4 as v4

ROOT = Path(__file__).resolve().parents[1]
ARRAY_FIELDS = ("denoised", "model_denoised", "artifact_probability", "confidence",
                "replacement_mask", "valid_mask", "rejected_mask")


def calibration(count=100, observed=0.0):
    return {"scope": "validation_only", "comparison": ">", "method": "clean_identity_order_statistic",
            "clean_window_count": count, "observed_clean_false_replacement_fraction": observed,
            "max_clean_false_replacement_fraction": .01, "empirical_only": True,
            "production_calibrated": False}


def synthetic(samples=1007):
    t = np.arange(samples, dtype=np.float64) / 250
    neural = .8 * np.sin(2 * np.pi * 12 * t) + .3 * np.cos(2 * np.pi * 31 * t)
    eog = 1.8 * np.sin(2 * np.pi * 1.5 * t + .2)
    return np.stack([3 + neural + eog, -4 + 1.7 * (neural + eog)], axis=1)


def assert_bits(test, actual, expected):
    test.assertEqual(actual.dtype, np.dtype(np.float64))
    np.testing.assert_array_equal(np.ascontiguousarray(actual).view(np.uint64),
                                  np.ascontiguousarray(expected).view(np.uint64))


class ConfigAndCalibrationTests(unittest.TestCase):
    def test_defaults_match_frozen_contract_and_config_and_dataclass_field_order(self):
        contract = research_contract('v4-interface.json')
        config = v4.load_config(ROOT / "configs" / "learned_denoiser_v4.json")
        self.assertEqual(contract["config"], config)
        self.assertEqual(v4.DEFAULT_CONFIG, config)
        self.assertEqual("learned_denoise_v4", v4.MODEL_VERSION)
        self.assertEqual(set(contract["core"]["checkpoint_keys"]), v4.CHECKPOINT_KEYS)
        self.assertEqual(["model", "config", "artifact_threshold", "device", "metadata"],
                         [item.name for item in fields(v4.ModelBundle)])
        self.assertEqual([*ARRAY_FIELDS, "diagnostics"], [item.name for item in fields(v4.DenoiseResult)])
        config["model"]["dilations"][0] = 99
        self.assertEqual(1, v4.load_config()["model"]["dilations"][0])
        a = v4.ModelBundle(None, v4.load_config(), .5, "cpu")
        b = v4.ModelBundle(None, v4.load_config(), .5, "cpu")
        a.metadata["fixture"] = 1
        self.assertEqual({}, b.metadata)

    def test_frozen_architecture_and_all_safety_guards_cannot_be_loosened(self):
        edits = [(None, "status", "draft"), (None, "schema_version", True),
                 (None, "sample_rate_hz", 256), (None, "input_channels", 8),
                 (None, "window_samples", 512), (None, "hop_samples", 250),
                 (None, "normalization", "raw_dc"), ("model", "hidden_channels", 48),
                 ("model", "dilations", [1, 2, 4]), ("model", "kernel_size", 7),
                 ("model", "dropout", .1), ("model", "residual_peak_bound", 3.),
                 ("model", "residual_band_hz", [.5, 40.]),
                 ("inference", "min_ac_rms", 0.), ("inference", "max_invalid_fraction", .01),
                 ("inference", "threshold_comparison", ">="), ("inference", "artifact_threshold_floor", .4),
                 ("inference", "clean_false_replacement_max", .02),
                 ("inference", "max_high_frequency_fraction_above_45hz", 1.),
                 ("inference", "max_abs_dc_to_ac_rms", 1e12),
                 ("inference", "max_change_ac_rms_ratio", 2.),
                 ("inference", "protected_band_hz", [8., 30.]),
                 ("inference", "protected_band_relative_tolerance", .1),
                 ("loss", "time_weight", float("nan")), ("loss", "artifact_weight", 0.),
                 ("runtime", "enabled", True), ("runtime", "enabled", 0),
                 ("runtime", "production_validated", True), ("runtime", "validation_status", "passed"),
                 ("runtime", "deployment_scope", "automatic")]
        for section, key, value in edits:
            config = v4.load_config()
            (config if section is None else config[section])[key] = value
            with self.subTest(section=section, key=key), self.assertRaises(v4.LearnedDenoiseV4Error):
                v4.validate_config(config)
        for config in (None, {}, {**v4.load_config(), "extra": False}):
            with self.assertRaises(v4.LearnedDenoiseV4Error):
                v4.validate_config(config)

    def test_config_path_requires_complete_finite_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            with self.assertRaises(v4.LearnedDenoiseV4Error):
                v4.load_config(path)
            for text in ("{", "[]", "{}", '{"status":"formal"}'):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(v4.LearnedDenoiseV4Error):
                    v4.load_config(path)
            path.write_text(json.dumps(v4.DEFAULT_CONFIG), encoding="utf-8")
            self.assertEqual(v4.DEFAULT_CONFIG, v4.load_config(path))

    def test_primitive_metadata_rejects_numpy_nonfinite_objects_tuples_and_cycles(self):
        self.assertTrue(v4.primitive_metadata({"plain": [None, True, 1, .1, "label"]}))
        cycle = []; cycle.append(cycle)
        for value in (np.float64(.5), np.int64(1), np.zeros(2), float("nan"), float("inf"),
                      {1: "not-a-string-key"}, (1, 2), Path("custom"), cycle):
            with self.subTest(kind=type(value)):
                self.assertFalse(v4.primitive_metadata(value))

    def test_strict_order_statistic_floor_ties_and_empirical_one_percent(self):
        cases = [np.full(10, .5), np.full(100, .9), np.linspace(.1, .99, 99),
                 np.r_[np.full(99, .1), .9], np.r_[np.full(98, .1), .8, .9],
                 np.array([.8, .8, .9]), np.array([1.]), np.random.default_rng(61).uniform(0, 1, 5001)]
        for scores in cases:
            for maximum in (0., .005, .01):
                with self.subTest(size=len(scores), maximum=maximum):
                    threshold = v4.select_artifact_threshold(scores, maximum)
                    self.assertGreaterEqual(threshold, .5)
                    self.assertLessEqual(threshold, 1.)
                    self.assertLessEqual(np.mean(scores > threshold), maximum)
                    if np.mean(scores > .5) <= maximum:
                        self.assertEqual(.5, threshold)
        self.assertEqual(.8, v4.select_artifact_threshold(np.r_[np.zeros(98), .8, .9]))
        self.assertEqual(.9, v4.select_artifact_threshold(np.full(100, .9)))

    def test_protected_band_metric_is_exact_preregistered_l2_with_ac_floor(self):
        t = np.arange(500) / 250
        cases = [5 + np.sin(2 * np.pi * 8 * t) + .4 * np.sin(2 * np.pi * 12 * t)
                 + .6 * np.cos(2 * np.pi * 40 * t) + .9 * np.cos(2 * np.pi * 1.5 * t),
                 3 + np.sin(2 * np.pi * 1.5 * t)]
        frequency = np.fft.rfftfreq(500, 1 / 250)
        selected = (frequency >= 8) & (frequency <= 40)
        for raw in cases:
            candidate = raw + 1e-8 * np.sin(2 * np.pi * 12 * t)
            raw_ac = raw - raw.mean()
            rms = float(np.sqrt(np.mean(raw_ac ** 2)))
            expected = np.linalg.norm(np.fft.rfft(candidate - raw)[selected]) / max(
                np.linalg.norm(np.fft.rfft(raw_ac)[selected]), np.sqrt(500) * rms * 1e-12)
            actual = v4._protected_difference(raw, candidate, 250., [8., 40.])
            np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0)

    def test_invalid_threshold_calibration_inputs_are_rejected(self):
        for scores in ([], [[.5]], [np.nan], [np.inf], [1.1], [-.1], [True], [".5"]):
            with self.subTest(scores=scores), self.assertRaises(v4.LearnedDenoiseV4Error):
                v4.select_artifact_threshold(scores)
        for maximum, floor in ((.02, .5), (-.1, .5), (True, .5), (.01, .49), (.01, 1.1), (.01, np.nan)):
            with self.assertRaises(v4.LearnedDenoiseV4Error):
                v4.select_artifact_threshold([.5], maximum, floor)


if v4.torch is not None:
    class FixtureModel(v4.torch.nn.Module):
        def __init__(self, logit=2.197224577, mode="eog", factor=.3):
            super().__init__()
            self.logit, self.mode, self.factor = logit, mode, factor
            self.seen = []

        def forward(self, signal):
            self.seen.append(signal.detach().cpu().numpy().copy())
            if self.mode == "throw" or (self.mode == "selective_throw" and bool((signal[:, 0, 0] > 0).any())):
                raise RuntimeError("fixture prediction error")
            frequencies = v4.torch.fft.rfftfreq(500, d=1 / 250, device=signal.device)
            band = (frequencies >= .5) & (frequencies <= 6.)
            residual = v4.torch.fft.irfft(v4.torch.fft.rfft(signal) * band, n=500) * self.factor
            if self.mode == "zero":
                residual = v4.torch.zeros_like(signal)
            elif self.mode == "out_of_band":
                t = v4.torch.arange(500, device=signal.device, dtype=v4.torch.float64) / 250
                high = (.8 * v4.torch.sin(2 * v4.torch.pi * 17 * t)).to(v4.torch.float32)
                residual = residual + high[None, None]
            elif self.mode == "excessive":
                t = v4.torch.arange(500, device=signal.device) / 250
                residual = 1.8 * v4.torch.sin(2 * v4.torch.pi * 2 * t)[None, None]
            elif self.mode == "nan_clean":
                residual = v4.torch.full_like(signal, float("nan"))
            clean = signal - residual
            logits = v4.torch.full((signal.shape[0],), self.logit, dtype=v4.torch.float32, device=signal.device)
            if self.mode == "nan_row":
                clean = clean.clone(); clean[0, 0, 1] = float("nan")
            elif self.mode == "nan_logit_row":
                logits[0] = float("nan")
            elif self.mode == "bad_clean_shape":
                clean = clean[:, 0]
            elif self.mode == "bad_clean_dtype":
                clean = clean.double()
            elif self.mode == "bad_logit_shape":
                logits = logits[:, None]
            elif self.mode == "bad_logit_dtype":
                logits = v4.torch.ones(signal.shape[0], dtype=v4.torch.bool, device=signal.device)
            elif self.mode == "dict":
                return {"clean": clean, "logits": logits}
            return clean, logits


@unittest.skipIf(v4.torch is None, "optional PyTorch unavailable")
class ModelAndInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = v4.torch.get_num_threads()
        v4.torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        v4.torch.set_num_threads(cls.previous_threads)

    def bundle(self, model=None, threshold=.5):
        return v4.ModelBundle(FixtureModel().eval() if model is None else model.eval(),
                              v4.load_config(), threshold, "cpu")

    def assert_safe(self, result, shape):
        for key in ARRAY_FIELDS:
            self.assertEqual(shape, getattr(result, key).shape)
        for key in ("replacement_mask", "valid_mask", "rejected_mask"):
            self.assertEqual(np.dtype(bool), getattr(result, key).dtype)
        self.assertTrue(result.diagnostics["diagnostic_only"])
        for key in ("automatic_processing_enabled", "release_gate_passed", "production_validated", "missing_data_repaired"):
            self.assertIs(False, result.diagnostics[key])
        self.assertFalse(result.diagnostics["runtime"]["enabled"])
        self.assertTrue(v4.primitive_metadata(result.diagnostics))

    def test_real_model_forward_shape_identity_batch_independence_and_differentiability(self):
        with v4.torch.random.fork_rng(devices=[]):
            v4.torch.manual_seed(4)
            model = v4.build_model(v4.load_config()).eval()
            signal = v4.torch.randn(3, 1, 500)
            clean, logits = model(signal)
            v4.torch.testing.assert_close(clean, signal, rtol=0, atol=0)
            self.assertEqual((3,), tuple(logits.shape))
            alone_clean, alone_logits = model(signal[1:2])
            v4.torch.testing.assert_close(clean[1:2], alone_clean)
            v4.torch.testing.assert_close(logits[1:2], alone_logits)
            with v4.torch.no_grad():
                model.residual_head.weight.normal_(0, .03)
            differentiable = signal.clone().requires_grad_()
            prediction, scores = model(differentiable)
            (prediction.square().mean() + scores.square().mean()).backward()
            self.assertTrue(v4.torch.isfinite(differentiable.grad).all())
            self.assertTrue(all(v4.torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None))
        for bad in (v4.torch.zeros(1, 8, 500), v4.torch.zeros(1, 1, 512), signal.double(),
                    v4.torch.full((1, 1, 500), float("nan"))):
            with self.assertRaises(v4.LearnedDenoiseV4Error):
                model(bad)

    def test_torch_projection_inclusive_edges_and_uniform_peak_limit_not_clipping(self):
        model = v4.build_model(v4.load_config())
        t = np.arange(500, dtype=np.float64) / 250
        raw = 10 + 8 * np.sin(2 * np.pi * .5 * t) + 4 * np.cos(2 * np.pi * 6 * t)
        raw += 3 * np.sin(2 * np.pi * 6.5 * t) + 2 * np.sin(2 * np.pi * 40 * t)
        projected = model._project_residual(v4.torch.tensor(raw[None, None], dtype=v4.torch.float32)).detach().numpy()[0, 0]
        intended = 8 * np.sin(2 * np.pi * .5 * t) + 4 * np.cos(2 * np.pi * 6 * t)
        intended *= 2 / np.max(np.abs(intended))
        np.testing.assert_allclose(projected, intended, rtol=0, atol=2e-6)
        frequency = np.fft.rfftfreq(500, d=1 / 250)
        outside = (frequency < .5) | (frequency > 6)
        self.assertLess(np.max(np.abs(np.fft.rfft(projected)[outside])), 2e-5)
        self.assertLessEqual(np.max(np.abs(projected)), 2.000001)
        self.assertGreater(abs(np.fft.rfft(projected)[1]), 10)
        self.assertGreater(abs(np.fft.rfft(projected)[12]), 10)

    def test_numpy_float64_projection_band_dc_and_uniform_peak_bound(self):
        t = np.arange(500) / 250
        raw = 5 + 7 * np.sin(2 * np.pi * .5 * t) + 3 * np.sin(2 * np.pi * 6 * t) + np.cos(2 * np.pi * 31 * t)
        projected = v4._project_numpy(raw, 250., (.5, 6.), 2.)
        low = 7 * np.sin(2 * np.pi * .5 * t) + 3 * np.sin(2 * np.pi * 6 * t)
        low *= 2 / np.max(np.abs(low))
        np.testing.assert_allclose(projected, low, atol=3e-15)
        frequency = np.fft.rfftfreq(500, 1 / 250)
        self.assertLess(np.max(np.abs(np.fft.rfft(projected)[(frequency < .5) | (frequency > 6)])), 5e-13)
        self.assertLess(abs(projected.mean()), 1e-15)

    def test_nonnoop_eog_changes_but_dc_and_entire_protected_fft_are_unchanged(self):
        signal = synthetic()
        before = signal.tobytes()
        result = v4.apply_model_to_array(self.bundle(), signal)
        self.assert_safe(result, signal.shape)
        self.assertTrue(result.replacement_mask[:1000].all())
        self.assertGreater(result.diagnostics["changed_samples"], 0)
        self.assertEqual(before, signal.tobytes())
        frequency = np.fft.rfftfreq(500, 1 / 250)
        protected = (frequency >= 8) & (frequency <= 40)
        for start in (0, 500):
            expected_dc = signal[start:start + 500].mean(0)
            # FFT/reduction implementations differ by a few float64 ULPs across
            # NumPy versions/CPU platforms. Keep a scale-aware roundoff bound,
            # not a signal-quality tolerance; the protected FFT gate stays fixed.
            dc_roundoff = 8 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(expected_dc))))
            np.testing.assert_allclose(result.denoised[start:start + 500].mean(0), expected_dc,
                                       rtol=0, atol=dc_roundoff)
            np.testing.assert_allclose(np.fft.rfft(result.denoised[start:start + 500], axis=0)[protected],
                                       np.fft.rfft(signal[start:start + 500], axis=0)[protected], rtol=0, atol=5e-12)
        assert_bits(self, result.denoised[1000:], signal[1000:])

    def test_numpy_reprojection_removes_finite_out_of_band_prediction_changes(self):
        signal = synthetic(500)
        low = v4.apply_model_to_array(self.bundle(), signal)
        polluted = v4.apply_model_to_array(self.bundle(FixtureModel(mode="out_of_band")), signal)
        self.assertTrue(polluted.replacement_mask.all())
        # FFT projection is an orthogonal projector: ||P e||_inf <= ||e||_2.
        # The 17 Hz component is exactly outside the passband. Bound only the
        # float32 subtraction/quantization error, not the protected-band guard.
        high = .8 * np.sin(2 * np.pi * 17 * np.arange(500, dtype=np.float64) / 250)
        for channel in range(signal.shape[1]):
            normalized, _, scale = v4._normalize(signal[:, channel], 1e-8)
            tensor = v4.torch.from_numpy(normalized[None, None])
            a, _ = FixtureModel()(tensor)
            b, _ = FixtureModel(mode="out_of_band")(tensor)
            rounding = a.detach().numpy()[0, 0].astype(np.float64) - b.detach().numpy()[0, 0].astype(np.float64) - high
            bound = scale * np.linalg.norm(rounding) + 16 * np.finfo(np.float64).eps * np.max(np.abs(signal[:, channel]))
            self.assertLess(bound, 1e-5)  # Tight fixture-only float32 numerical bound.
            self.assertLessEqual(np.max(np.abs(polluted.denoised[:, channel] - low.denoised[:, channel])), bound)
        frequency = np.fft.rfftfreq(500, 1 / 250)
        protected = (frequency >= 8) & (frequency <= 40)
        np.testing.assert_allclose(np.fft.rfft(polluted.denoised, axis=0)[protected],
                                   np.fft.rfft(signal, axis=0)[protected], rtol=0, atol=5e-12)

    def test_channels_alone_and_permuted_are_identical_with_failure_isolation(self):
        signal = synthetic()
        result = v4.apply_model_to_array(self.bundle(), signal)
        permuted = v4.apply_model_to_array(self.bundle(), signal[:, [1, 0]])
        for key in ARRAY_FIELDS:
            np.testing.assert_array_equal(getattr(result, key)[:, [1, 0]], getattr(permuted, key))
        for channel in range(2):
            alone = v4.apply_model_to_array(self.bundle(), signal[:, channel])
            self.assertEqual((len(signal), 1), alone.denoised.shape)
            for key in ARRAY_FIELDS:
                np.testing.assert_array_equal(getattr(result, key)[:, channel:channel + 1], getattr(alone, key))
        mixed = signal.copy(); mixed[:, 1] *= -1
        broken = v4.apply_model_to_array(self.bundle(FixtureModel(mode="selective_throw")), mixed)
        self.assertTrue(broken.rejected_mask[:, 0].all())
        self.assertTrue(broken.replacement_mask[:1000, 1].all())
        self.assertEqual(1, broken.diagnostics["batch_forward_failures"])
        self.assertEqual(4, broken.diagnostics["single_window_retry_calls"])

    def test_huge_dc_with_sufficient_ac_uses_original_float64_scale_and_guards_small_ac(self):
        waveform = synthetic(500)[:, 0] - 3
        signal = (1e12 + waveform * 1e11)[:, None]
        model = FixtureModel()
        result = v4.apply_model_to_array(self.bundle(model), signal)
        self.assertTrue(result.replacement_mask.all())
        self.assertEqual(np.dtype(np.float64), result.denoised.dtype)
        self.assertLessEqual(abs(float(result.denoised.mean()) - float(signal.mean())), 3e-4)
        center = float(signal.mean())
        rms = float(np.sqrt(np.mean((signal - center) ** 2)))
        normalized = model.seen[0][0, 0]
        output, _ = model(v4.torch.tensor(normalized[None, None]))
        residual = v4._project_numpy(normalized.astype(np.float64) - output.detach().numpy()[0, 0].astype(np.float64), 250., (.5, 6.), 2.)
        expected = signal[:, 0] - residual * rms
        np.testing.assert_allclose(result.denoised[:, 0], expected, rtol=0, atol=3e-4)
        unsafe_model = FixtureModel()
        unsafe = (1e12 + waveform)[:, None]
        rejected = v4.apply_model_to_array(self.bundle(unsafe_model), unsafe)
        assert_bits(self, rejected.denoised, unsafe)
        self.assertEqual([], unsafe_model.seen)
        self.assertEqual(1, rejected.diagnostics["dc_guard_windows"])
        extreme = (1e308 + waveform * 1e298)[:, None]
        assert_bits(self, v4.apply_model_to_array(self.bundle(), extreme).denoised, extreme)

    def test_high_frequency_energy_guard_rejects_before_network_but_boundary_45_is_allowed(self):
        t = np.arange(500) / 250
        bad = (np.sin(2 * np.pi * 12 * t) + 2 * np.sin(2 * np.pi * 60 * t))[:, None]
        model = FixtureModel()
        result = v4.apply_model_to_array(self.bundle(model), bad)
        assert_bits(self, result.denoised, bad)
        self.assertEqual([], model.seen)
        self.assertTrue(result.rejected_mask.all())
        self.assertEqual(1, result.diagnostics["high_frequency_guard_windows"])
        edge = (np.sin(2 * np.pi * 12 * t) + 2 * np.sin(2 * np.pi * 45 * t))[:, None]
        accepted = v4.apply_model_to_array(self.bundle(), edge)
        self.assertFalse(accepted.rejected_mask.any())

    def test_constant_nonfinite_invalid_and_tail_windows_preserve_exact_raw(self):
        signal = synthetic()
        signal[0:500, 0] = 7.
        signal.view(np.uint64)[750, 0] = 0x7FF8000000000142
        signal[1000, 0] = -0.
        signal[1005, 1] = np.inf
        mask = np.ones(signal.shape, dtype=bool)
        mask[50, 1] = False
        before = signal.tobytes()
        result = v4.apply_model_to_array(self.bundle(), signal, mask)
        for position in ((slice(0, 1000), 0), (slice(0, 500), 1), (slice(1000, None), slice(None))):
            assert_bits(self, result.denoised[position], signal[position])
            assert_bits(self, result.model_denoised[position], signal[position])
            self.assertTrue(result.rejected_mask[position].all())
            self.assertTrue(np.isnan(result.artifact_probability[position]).all())
            self.assertTrue((result.confidence[position] == 0).all())
        self.assertEqual(before, signal.tobytes())
        short = v4.apply_model_to_array(self.bundle(), signal[:19])
        assert_bits(self, short.denoised, signal[:19])
        self.assertTrue(short.rejected_mask.all())
        mask1d = np.ones(500, dtype=bool); mask1d[1] = False
        rejected = v4.apply_model_to_array(self.bundle(), synthetic(500)[:, 0], mask1d)
        self.assertTrue(rejected.rejected_mask.all())

    def test_network_nonfinite_shapes_dtypes_and_probability_are_fail_closed(self):
        signal = synthetic(500)
        modes = ["nan_clean", "bad_clean_shape", "bad_clean_dtype", "bad_logit_shape", "bad_logit_dtype", "dict", "throw"]
        fixtures = [FixtureModel(mode=mode) for mode in modes] + [FixtureModel(logit=value) for value in (np.nan, np.inf, -np.inf)]
        for model in fixtures:
            with self.subTest(mode=model.mode, logit=model.logit):
                result = v4.apply_model_to_array(self.bundle(model), signal)
                assert_bits(self, result.denoised, signal)
                assert_bits(self, result.model_denoised, signal)
                self.assertTrue(result.rejected_mask.all())
                self.assertFalse(result.replacement_mask.any())
                self.assertTrue(np.isnan(result.artifact_probability).all())
                self.assertTrue((result.confidence == 0).all())
                self.assertEqual(2, result.diagnostics["prediction_rejected_windows"])

    def test_change_rms_and_protected_band_postrestore_guards_reject_candidates(self):
        signal = synthetic(500)
        excessive = v4.apply_model_to_array(self.bundle(FixtureModel(mode="excessive")), signal)
        assert_bits(self, excessive.denoised, signal)
        self.assertTrue(excessive.rejected_mask.all())
        self.assertEqual(2, excessive.diagnostics["change_guard_windows"])
        with patch.object(v4, "_protected_difference", return_value=.01):
            protected = v4.apply_model_to_array(self.bundle(), signal)
        assert_bits(self, protected.denoised, signal)
        self.assertEqual(2, protected.diagnostics["protected_band_rejected_windows"])

    def test_strict_greater_threshold_ties_and_candidate_are_distinct_from_gated_raw(self):
        signal = synthetic(500)
        for logit, threshold in ((0., .5), (100., 1.)):
            result = v4.apply_model_to_array(self.bundle(FixtureModel(logit), threshold), signal)
            self.assertFalse(result.replacement_mask.any())
            self.assertFalse(result.rejected_mask.any())
            assert_bits(self, result.denoised, signal)
            self.assertFalse(np.array_equal(result.model_denoised, signal))
            np.testing.assert_array_equal(result.confidence, 1 - result.artifact_probability)

    def test_noop_acceptance_mask_is_not_actual_change_and_signed_zero_survives(self):
        signal = synthetic(500)
        signal[19, 0] = -0.
        result = v4.apply_model_to_array(self.bundle(FixtureModel(mode="zero")), signal)
        self.assertTrue(result.replacement_mask.all())
        assert_bits(self, result.denoised, signal)
        self.assertEqual(0, result.diagnostics["changed_samples"])
        self.assertEqual(signal.size, result.diagnostics["replaced_samples"])

    def test_batches_at_most_128_safe_windows_without_changing_per_window_result(self):
        window = synthetic(500)[:, 0]
        raw = np.tile(window, 130)[:, None]
        model = FixtureModel()
        batched = v4.apply_model_to_array(self.bundle(model), raw)
        self.assertEqual([128, 2], [value.shape[0] for value in model.seen])
        self.assertEqual(2, batched.diagnostics["model_forward_calls"])
        self.assertEqual(0, batched.diagnostics["single_window_retry_calls"])
        self.assertTrue(batched.replacement_mask.all())
        reference = v4.apply_model_to_array(self.bundle(), window)
        for name in ARRAY_FIELDS:
            expected = np.tile(getattr(reference, name), (130, 1))
            np.testing.assert_allclose(getattr(batched, name), expected, rtol=0, atol=1e-12, equal_nan=True)

    def test_one_nonfinite_row_is_rejected_without_contaminating_other_channel(self):
        signal = synthetic(500)
        for mode in ("nan_row", "nan_logit_row"):
            result = v4.apply_model_to_array(self.bundle(FixtureModel(mode=mode)), signal)
            self.assertTrue(result.rejected_mask[:, 0].all())
            self.assertFalse(result.replacement_mask[:, 0].any())
            assert_bits(self, result.denoised[:, 0], signal[:, 0])
            self.assertTrue(result.replacement_mask[:, 1].all())
            self.assertTrue(np.isnan(result.artifact_probability[:, 0]).all())
            self.assertEqual(1, result.diagnostics["model_forward_calls"])
            self.assertEqual(0, result.diagnostics["single_window_retry_calls"])
            self.assertEqual(1, result.diagnostics["prediction_rejected_windows"])

    def test_float32_fft_even_under_outer_autocast_and_half_fft_is_never_called(self):
        model = v4.build_model(v4.load_config()).eval()
        signal = v4.torch.randn(2, 1, 500)
        with patch.object(v4.torch.fft, "rfft", wraps=v4.torch.fft.rfft) as fft:
            with v4.torch.autocast(device_type="cpu", dtype=v4.torch.bfloat16):
                clean, logits = model(signal)
            self.assertEqual(v4.torch.float32, clean.dtype)
            self.assertEqual(v4.torch.float32, logits.dtype)
            self.assertTrue(all(call.args[0].dtype == v4.torch.float32 for call in fft.call_args_list))
        with patch.object(v4.torch.fft, "rfft", wraps=v4.torch.fft.rfft) as fft:
            with self.assertRaises(v4.LearnedDenoiseV4Error):
                model._project_residual(signal.half())
            self.assertEqual(0, fft.call_count)

    def test_invalid_signal_mask_threshold_and_mutated_runtime_refuse_explicitly(self):
        for raw in (np.zeros((1, 1, 500)), np.zeros((0, 2)), np.zeros((500, 0)),
                    np.zeros((500, 1), dtype=bool), np.full((500, 1), 2 ** 63, dtype=np.uint64),
                    np.zeros((500, 1), dtype=np.complex128)):
            with self.assertRaises(v4.LearnedDenoiseV4Error):
                v4.apply_model_to_array(self.bundle(), raw)
        signal = synthetic(500)
        for mask in (np.ones(500, dtype=bool), np.ones((500, 2), dtype=float), np.zeros((2, 500), dtype=bool)):
            with self.assertRaises(v4.LearnedDenoiseV4Error):
                v4.apply_model_to_array(self.bundle(), signal, mask)
        for threshold in (True, .49, 1.1, float("nan"), float("inf")):
            with self.assertRaises(v4.LearnedDenoiseV4Error):
                v4.apply_model_to_array(self.bundle(threshold=threshold), signal)
        bundle = self.bundle(); bundle.config["runtime"]["enabled"] = True
        with self.assertRaises(v4.LearnedDenoiseV4Error):
            v4.apply_model_to_array(bundle, signal)
        with patch.object(v4.torch.cuda, "is_available", return_value=False), self.assertRaises(v4.LearnedDenoiseV4Error):
            v4.load_model("not-opened.pt", device="cuda")


@unittest.skipIf(v4.torch is None, "optional PyTorch unavailable")
class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = v4.torch.get_num_threads()
        v4.torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        v4.torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="v4 CPU checkpoint fixture ")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "fixture.pt"
        self.config = v4.load_config()
        with v4.torch.random.fork_rng(devices=[]):
            v4.torch.manual_seed(4)
            self.model = v4.build_model(self.config)
            with v4.torch.no_grad():
                self.model.residual_head.weight.normal_(0, .02)
                self.model.artifact_head.weight.zero_()
                self.model.artifact_head.bias.fill_(2.197224577)
        self.checkpoint = {"checkpoint_schema_version": 1, "model_version": v4.MODEL_VERSION,
            "config": deepcopy(self.config), "model_state": self.model.state_dict(), "artifact_threshold": .5,
            "threshold_calibration": calibration(), "dataset_sha256": "a" * 64,
            "metadata": {"fixture_only": True}}
        self.save(self.checkpoint)

    def save(self, checkpoint):
        v4.torch.save(checkpoint, self.path)

    def test_real_cpu_roundtrip_strict_weights_only_no_source_mutation(self):
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with patch.object(v4.torch, "load", wraps=v4.torch.load) as loader:
            bundle = v4.load_model(self.path, device="cpu")
        self.assertEqual(1, loader.call_count)
        self.assertIs(True, loader.call_args.kwargs["weights_only"])
        self.assertEqual("cpu", loader.call_args.kwargs["map_location"])
        self.assertEqual(self.config, bundle.config)
        self.assertEqual("cpu", bundle.device)
        self.assertEqual(.5, bundle.artifact_threshold)
        self.assertTrue(v4.primitive_metadata(bundle.metadata))
        self.assertNotIn("model_state", bundle.metadata)
        self.assertTrue(all(not parameter.requires_grad for parameter in bundle.model.parameters()))
        signal = synthetic(500)
        result = v4.apply_model_to_array(bundle, signal)
        self.assertFalse(result.rejected_mask.any())
        self.assertTrue(result.replacement_mask.all())
        self.assertGreater(result.diagnostics["changed_samples"], 0)
        self.assertEqual(before, hashlib.sha256(self.path.read_bytes()).hexdigest())
        path = self.root / "formal.json"
        path.write_text(json.dumps(self.config), encoding="utf-8")
        self.assertEqual(self.config, v4.load_model(self.path, path).config)

    def test_schema_extra_keys_bad_model_and_unsafe_metadata_are_refused(self):
        variants = []
        for key, value in (("model_version", "learned_denoise_v2"), ("checkpoint_schema_version", True),
                           ("artifact_threshold", .49), ("artifact_threshold", True), ("artifact_threshold", float("nan")),
                           ("artifact_threshold", 1.1), ("dataset_sha256", "A" * 64),
                           ("metadata", {"invalid": float("inf")}), ("metadata", {"tuple": (1, 2)}),
                           ("metadata", {"numpy": np.float64(.5)}), ("metadata", ["not-dict"])):
            bad = deepcopy(self.checkpoint); bad[key] = value; variants.append(bad)
        bad = deepcopy(self.checkpoint); bad["extra"] = "forbidden"; variants.append(bad)
        bad = deepcopy(self.checkpoint); del bad["metadata"]; variants.append(bad)
        bad = deepcopy(self.checkpoint); bad["config"]["model"]["kernel_size"] = 7; variants.append(bad)
        bad = deepcopy(self.checkpoint); bad["config"]["inference"]["protected_band_relative_tolerance"] = .1; variants.append(bad)
        bad = deepcopy(self.checkpoint); bad["config"]["runtime"]["production_validated"] = True; variants.append(bad)
        variants.extend([[], {"not-a-checkpoint": True}])
        for index, bad in enumerate(variants):
            self.save(bad)
            with self.subTest(index=index), self.assertRaises(v4.LearnedDenoiseV4Error):
                v4.load_model(self.path)

    def test_validation_clean_only_provenance_and_empirical_one_percent_required(self):
        for key, value in (("scope", "test"), ("comparison", ">="), ("method", "unsupported_order_statistic"),
                           ("clean_window_count", 0), ("clean_window_count", True), ("empirical_only", False),
                           ("production_calibrated", True), ("observed_clean_false_replacement_fraction", .011),
                           ("observed_clean_false_replacement_fraction", -.01),
                           ("max_clean_false_replacement_fraction", .02)):
            bad = deepcopy(self.checkpoint); bad["threshold_calibration"][key] = value
            self.save(bad)
            with self.subTest(key=key), self.assertRaises(v4.LearnedDenoiseV4Error):
                v4.load_model(self.path)
        valid = deepcopy(self.checkpoint); valid["threshold_calibration"] = calibration(100, .01)
        self.save(valid)
        self.assertEqual(.5, v4.load_model(self.path).artifact_threshold)

    def test_state_strict_keys_shape_real_float_and_finite_before_and_after_cast(self):
        key = "artifact_head.bias"
        variants = []
        for value in (v4.torch.zeros(2), v4.torch.zeros(1, dtype=v4.torch.int64),
                      v4.torch.full((1,), float("nan")), v4.torch.full((1,), float("inf")),
                      v4.torch.full((1,), 1e300, dtype=v4.torch.float64)):
            bad = deepcopy(self.checkpoint); bad["model_state"][key] = value; variants.append(bad)
        bad = deepcopy(self.checkpoint); del bad["model_state"][key]; variants.append(bad)
        bad = deepcopy(self.checkpoint); bad["model_state"]["extra.weight"] = v4.torch.ones(1); variants.append(bad)
        for index, bad in enumerate(variants):
            self.save(bad)
            with self.subTest(index=index), self.assertRaises(v4.LearnedDenoiseV4Error):
                v4.load_model(self.path)

    def test_custom_pickle_missing_file_and_external_unsafe_config_never_load_unsafely(self):
        bad = deepcopy(self.checkpoint); bad["metadata"]["custom"] = Path("unsafe")
        self.save(bad)
        with self.assertRaises(v4.LearnedDenoiseV4Error):
            v4.load_model(self.path)
        for path in ("", self.root / "missing.pt"):
            with self.assertRaises(v4.LearnedDenoiseV4Error):
                v4.load_model(path)
        self.save(self.checkpoint)
        config = deepcopy(self.config); config["inference"]["max_abs_dc_to_ac_rms"] = 200
        with self.assertRaises(v4.LearnedDenoiseV4Error):
            v4.load_model(self.path, config)


if __name__ == "__main__":
    unittest.main()
