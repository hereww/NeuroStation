"""Fixture-only safety tests: no training, public data or held-out evaluation."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from eeg_tools import learned_denoise_v2 as v2
from eeg_tools import learned_denoise_v3 as v3

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("denoised", "model_denoised", "artifact_probability", "confidence", "replacement_mask",
          "valid_mask", "rejected_mask", "support_probability", "domain_rejected_mask")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration():
    return {"scope": "validation_only", "comparison": ">", "empirical_only": True,
            "production_calibrated": False, "method": "unsupported_score_order_statistic",
            "unsupported_window_count": 100, "observed_unsupported_false_acceptance_fraction": 0.0,
            "max_unsupported_false_acceptance_fraction": 0.01}


class ConfigurationTests(unittest.TestCase):
    def test_formal_config_matches_frozen_cnn_and_is_independent_of_v2(self):
        actual = v3.load_config(ROOT / "configs" / "learned_denoiser_v3.json")
        self.assertEqual(v3.DEFAULT_CONFIG, actual)
        self.assertEqual("learned_denoise_v3", v3.MODEL_VERSION)
        self.assertEqual("formal", actual["status"])
        self.assertEqual(250, actual["sample_rate_hz"])
        self.assertEqual(1, actual["input_channels"])
        self.assertEqual((500, 500), (actual["window_samples"], actual["hop_samples"]))
        frozen = json.loads((ROOT / "analysis_outputs" / "learned_denoiser_v3_preregistration.json").read_text(encoding="utf-8"))
        for key in ("channels", "kernel_sizes", "strides", "hidden_features", "dropout"):
            self.assertEqual(frozen["model"][key], actual["model"][key])
        self.assertEqual(set(frozen["interfaces"]["checkpoint_keys"]), v3.CHECKPOINT_KEYS)
        config = v3.load_config()
        config["model"]["channels"][0] = 99
        self.assertEqual(16, v3.load_config()["model"]["channels"][0])
        self.assertEqual("learned_denoise_v2", v2.load_config()["model_version"])

    def test_unsafe_configuration_and_nonfinite_values_are_rejected(self):
        cases = [(None, "status", "draft"), (None, "schema_version", True),
                 (None, "model_version", v2.MODEL_VERSION), (None, "sample_rate_hz", 256),
                 (None, "input_channels", 8), (None, "window_samples", 512), (None, "hop_samples", 250),
                 (None, "normalization", "raw_dc"), ("model", "channels", [32, 32, 32]),
                 ("model", "channels", [True, 32, 32]), ("model", "kernel_sizes", [7, 7, 5]),
                 ("model", "strides", [1, 2, 2]), ("model", "hidden_features", 64),
                 ("model", "dropout", .2), ("model", "dropout", float("nan")),
                 ("denoiser", "frozen", False), ("denoiser", "device", "cuda"),
                 ("inference", "threshold_comparison", ">="), ("inference", "max_invalid_fraction", .1),
                 ("inference", "min_ac_rms", 0), ("inference", "min_ac_rms", float("inf")),
                 ("inference", "support_threshold_floor", .4),
                 ("inference", "max_unsupported_false_acceptance_fraction", .02),
                 ("runtime", "enabled", True), ("runtime", "enabled", 0),
                 ("runtime", "production_validated", True), ("runtime", "validation_status", "passed"),
                 ("runtime", "deployment_scope", "automatic")]
        for section, key, value in cases:
            config = v3.load_config()
            (config if section is None else config[section])[key] = value
            with self.subTest(section=section, key=key, value=value), self.assertRaises(v3.LearnedDenoiseV3Error):
                v3.validate_config(config)
        for malformed in (None, {}, {**v3.load_config(), "extra": "forbidden"}):
            with self.assertRaises(v3.LearnedDenoiseV3Error):
                v3.validate_config(malformed)

    def test_loader_refuses_missing_malformed_unknown_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "formal.json"
            with self.assertRaises(v3.LearnedDenoiseV3Error):
                v3.load_config(path)
            for text in ("{", "[]", '{"extra": true}', '{"model":{"dropout": NaN}}'):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(v3.LearnedDenoiseV3Error):
                    v3.load_config(path)
            path.write_text('{"status":"formal"}', encoding="utf-8")
            self.assertEqual(v3.DEFAULT_CONFIG, v3.load_config(path))

    def test_metadata_is_plain_finite_and_rejects_numpy_tensors_tuples_cycles(self):
        self.assertTrue(v3.primitive_metadata({"value": [None, True, 1, .5, "text"]}))
        cycle = []; cycle.append(cycle)
        for value in (float("nan"), float("inf"), np.float64(.3), np.int64(1), np.ones(2),
                      (1, 2), {1: "non-string-key"}, Path("custom"), cycle):
            with self.subTest(type=type(value)):
                self.assertFalse(v3.primitive_metadata(value))
        if v3.torch is not None:
            self.assertFalse(v3.primitive_metadata(v3.torch.ones(1)))


if v3.torch is not None:
    class StubDenoiser(v3.torch.nn.Module):
        def __init__(self, probability=.9, unsafe=False):
            super().__init__()
            self.anchor = v3.torch.nn.Parameter(v3.torch.tensor(0.0))
            self.probability, self.unsafe = probability, unsafe

        def forward(self, signal, valid_mask=None):
            noise = .2 * signal
            probability = v3.torch.full_like(signal, self.probability)
            if self.unsafe:
                probability[signal[:, 0, 0] > 0] = float("nan")
            return {"noise": noise, "denoised": signal - noise, "artifact_probability": probability}

    class StubGate(v3.torch.nn.Module):
        def __init__(self, logit=10., mode="safe"):
            super().__init__()
            self.logit_value, self.mode = logit, mode
            self.seen = []

        def forward(self, signal):
            self.seen.append(signal.detach().cpu().numpy().copy())
            if self.mode == "throw" or (self.mode == "selective_throw" and signal[0, 0, 0] > 0):
                raise RuntimeError("fixture domain predictor rejected")
            if self.mode == "shape":
                return v3.torch.zeros((signal.shape[0], 1), device=signal.device)
            if self.mode == "bool":
                return v3.torch.ones(signal.shape[0], dtype=v3.torch.bool, device=signal.device)
            return v3.torch.full((signal.shape[0],), self.logit_value, device=signal.device, dtype=v3.torch.float32)


@unittest.skipIf(v3.torch is None, "optional PyTorch unavailable")
class ArchitectureAndInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = v3.torch.get_num_threads()
        v3.torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        v3.torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.config = v3.load_config()
        t = np.arange(1007) / 250
        waveform = np.sin(2 * np.pi * 10 * t + .4) + .3 * np.cos(2 * np.pi * 19 * t)
        self.signal = np.stack([12 + waveform, -35 - 2 * waveform]).astype(np.float64)

    def bundle(self, gate=None, probability=.9, threshold=.5, unsafe=False):
        denoiser = v2.ModelBundle(StubDenoiser(probability, unsafe).eval(), v2.load_config(), Path("fixture-v2.pt"), .5)
        return v3.ModelBundle(StubGate().eval() if gate is None else gate.eval(), denoiser,
                              deepcopy(self.config), threshold, {}, Path("fixture-v3.pt"), "cpu")

    def assert_safety(self, result):
        for field in FIELDS:
            self.assertEqual(self.signal.shape, getattr(result, field).shape)
        for field in ("replacement_mask", "valid_mask", "rejected_mask", "domain_rejected_mask"):
            self.assertEqual(np.dtype(bool), getattr(result, field).dtype)
        diagnostics = result.diagnostics
        self.assertTrue(diagnostics["diagnostic_only"])
        for key in ("automatic_processing_enabled", "release_gate_passed", "production_validated", "missing_data_repaired"):
            self.assertIs(False, diagnostics[key])
        self.assertFalse(diagnostics["runtime"]["enabled"])
        self.assertIn("NOT_calibrated", diagnostics["confidence_interpretation"])
        self.assertEqual("cpu", diagnostics["denoiser_device"])
        self.assertTrue(v3.primitive_metadata(diagnostics))

    def test_gate_exact_architecture_logits_batch_and_strict_inputs(self):
        gate = v3.build_gate(self.config).eval()
        conv = [layer for layer in gate.modules() if isinstance(layer, v3.torch.nn.Conv1d)]
        self.assertEqual([1, 16, 32], [layer.in_channels for layer in conv])
        self.assertEqual([16, 32, 32], [layer.out_channels for layer in conv])
        self.assertEqual([(9,), (7,), (5,)], [layer.kernel_size for layer in conv])
        self.assertEqual([(2,), (2,), (2,)], [layer.stride for layer in conv])
        self.assertEqual(3, sum(isinstance(layer, v3.torch.nn.ReLU) for layer in gate.modules()))
        self.assertEqual(1, sum(isinstance(layer, v3.torch.nn.AdaptiveAvgPool1d) for layer in gate.modules()))
        linear = [layer for layer in gate.modules() if isinstance(layer, v3.torch.nn.Linear)]
        self.assertEqual([(32, 32), (32, 1)], [(layer.in_features, layer.out_features) for layer in linear])
        normalized = v3.torch.randn(3, 1, 500)
        with v3.torch.no_grad():
            self.assertEqual((3,), tuple(gate(normalized).shape))
            v3.torch.testing.assert_close(gate(normalized)[[2, 0, 1]], gate(normalized[[2, 0, 1]]))
        for bad in (v3.torch.zeros(2, 8, 500), v3.torch.zeros(2, 1, 512),
                    v3.torch.zeros(0, 1, 500), normalized.double(), v3.torch.full((1, 1, 500), float("nan"))):
            with self.assertRaises(v3.LearnedDenoiseV3Error):
                gate(bad)

    def test_negative_domain_preserves_raw_bits_and_v2_candidate_and_confidence(self):
        x = self.signal.copy()
        x[0, 1000] = -0.0
        x.view(np.uint64)[0, -1] = 0x7FF8000000000123
        before = x.view(np.uint64).copy()
        bundle = self.bundle(StubGate(-10.))
        result = v3.apply_model_to_array(bundle, x)
        base = v2.apply_model_to_array(bundle.denoiser, x)
        np.testing.assert_array_equal(result.denoised.view(np.uint64), before)
        np.testing.assert_array_equal(x.view(np.uint64), before)
        np.testing.assert_array_equal(result.model_denoised, base.model_denoised)
        np.testing.assert_array_equal(result.artifact_probability, base.artifact_probability)
        np.testing.assert_array_equal(result.confidence, base.confidence)
        self.assertTrue(result.domain_rejected_mask.all())
        self.assertTrue(result.rejected_mask.all())
        self.assertFalse(result.replacement_mask.any())
        self.assertEqual(0, result.diagnostics["changed_samples"])
        self.assertEqual(4, result.diagnostics["unsupported_windows"])
        self.assert_safety(result)

    def test_positive_domain_exactly_preserves_frozen_v2_behavior_and_tail(self):
        bundle = self.bundle()
        base = v2.apply_model_to_array(bundle.denoiser, self.signal)
        result = v3.apply_model_to_array(bundle, self.signal)
        for name in FIELDS[:7]:
            np.testing.assert_array_equal(getattr(result, name), getattr(base, name))
        self.assertTrue(result.replacement_mask[:, :1000].all())
        self.assertFalse(result.domain_rejected_mask[:, :1000].any())
        self.assertTrue(result.domain_rejected_mask[:, 1000:].all())
        np.testing.assert_array_equal(result.denoised[:, 1000:].view(np.uint64), self.signal[:, 1000:].copy().view(np.uint64))
        np.testing.assert_allclose(result.denoised[:, :1000].reshape(2, 2, 500).mean(-1),
                                   self.signal[:, :1000].reshape(2, 2, 500).mean(-1), atol=1e-7)
        self.assertGreater(result.diagnostics["changed_samples"], 0)
        self.assert_safety(result)

    def test_strict_support_threshold_equality_one_and_v2_no_replacement(self):
        for logit, threshold, probability in ((0., .5, .9), (100., 1., .9), (10., .5, .2)):
            with self.subTest(logit=logit, threshold=threshold, probability=probability):
                result = v3.apply_model_to_array(self.bundle(StubGate(logit), probability, threshold), self.signal)
                self.assertFalse(result.replacement_mask.any())
                np.testing.assert_array_equal(result.denoised.view(np.uint64), self.signal.view(np.uint64))
                if probability == .2:
                    self.assertFalse(result.domain_rejected_mask[:, :1000].any())
                    self.assertFalse(result.rejected_mask[:, :1000].any())

    def test_invalid_constant_short_and_masked_windows_reject_without_gate_acceptance(self):
        x = self.signal.copy()
        x[0, 2] = np.nan
        x[0, 550] = np.inf
        x[1, :500] = 17.
        valid = np.ones(x.shape, dtype=bool)
        valid[1, 650] = False
        gate = StubGate()
        result = v3.apply_model_to_array(self.bundle(gate), x, valid_mask=valid)
        np.testing.assert_array_equal(result.denoised.view(np.uint64), x.view(np.uint64))
        np.testing.assert_array_equal(result.model_denoised, x)
        self.assertTrue(result.rejected_mask.all())
        self.assertTrue(result.domain_rejected_mask.all())
        self.assertTrue(np.isnan(result.support_probability).all())
        self.assertTrue(np.isnan(result.artifact_probability).all())
        self.assertTrue((result.confidence == 0).all())
        self.assertEqual([], gate.seen)
        self.assertEqual(3, result.diagnostics["invalid_windows"])
        self.assertEqual(1, result.diagnostics["constant_or_normalization_rejected_windows"])
        short = v3.apply_model_to_array(self.bundle(), self.signal[:, :17])
        self.assertTrue(short.domain_rejected_mask.all())
        np.testing.assert_array_equal(short.denoised, self.signal[:, :17])

    def test_bad_gate_logits_shapes_types_and_exception_do_not_silently_accept(self):
        for logit, mode in ((float("nan"), "safe"), (float("inf"), "safe"), (float("-inf"), "safe"),
                            (10., "shape"), (10., "bool"), (10., "throw")):
            bundle = self.bundle(StubGate(logit, mode))
            with self.subTest(logit=logit, mode=mode):
                result = v3.apply_model_to_array(bundle, self.signal)
                self.assertFalse(result.replacement_mask.any())
                self.assertTrue(result.rejected_mask.all())
                self.assertTrue(np.isnan(result.support_probability).all())
                self.assertEqual(4, result.diagnostics["domain_prediction_rejected_windows"])
                np.testing.assert_array_equal(result.denoised, self.signal)
                np.testing.assert_array_equal(result.model_denoised,
                                              v2.apply_model_to_array(bundle.denoiser, self.signal).model_denoised)

    def test_channel_permutation_independence_and_one_channel_exception_isolation(self):
        one = v3.apply_model_to_array(self.bundle(), self.signal)
        permuted = v3.apply_model_to_array(self.bundle(), self.signal[[1, 0]])
        for name in FIELDS:
            np.testing.assert_array_equal(getattr(one, name)[[1, 0]], getattr(permuted, name))
        for channel in range(2):
            alone = v3.apply_model_to_array(self.bundle(), self.signal[channel:channel + 1])
            for name in FIELDS:
                np.testing.assert_array_equal(getattr(one, name)[channel:channel + 1], getattr(alone, name))
        broken_gate = v3.apply_model_to_array(self.bundle(StubGate(mode="selective_throw")), self.signal)
        self.assertTrue(broken_gate.domain_rejected_mask[0].all())
        self.assertTrue(broken_gate.replacement_mask[1, :1000].all())
        broken_v2 = v3.apply_model_to_array(self.bundle(unsafe=True), self.signal)
        self.assertTrue(broken_v2.rejected_mask[0].all())
        self.assertTrue(broken_v2.replacement_mask[1, :1000].all())
        self.assertTrue(np.isnan(broken_v2.artifact_probability[0]).all())
        self.assertTrue((broken_v2.confidence[0] == 0).all())

    def test_normalized_inputs_exclude_units_dc_and_preserve_scale_shift_equivariance(self):
        with v3.torch.random.fork_rng(devices=[]):
            v3.torch.manual_seed(31)
            real_gate = v3.build_gate(self.config).eval()
        raw = self.signal[:, :1000]
        gate1, gate2 = StubGate(), StubGate()
        result1 = v3.apply_model_to_array(self.bundle(gate1), raw)
        result2 = v3.apply_model_to_array(self.bundle(gate2), raw * 64 + 1024)
        for a, b in zip(gate1.seen, gate2.seen):
            self.assertEqual((1, 1, 500), a.shape)
            self.assertEqual(np.dtype(np.float32), a.dtype)
            np.testing.assert_allclose(a, b, atol=1e-7)
            self.assertAlmostEqual(0., float(a.mean()), places=6)
            self.assertAlmostEqual(1., float(np.sqrt(np.mean(a ** 2))), places=6)
        np.testing.assert_allclose(result2.denoised, result1.denoised * 64 + 1024, rtol=0, atol=1e-10)
        learned1 = v3.apply_model_to_array(self.bundle(real_gate), raw)
        learned2 = v3.apply_model_to_array(self.bundle(real_gate), raw * 64 + 1024)
        np.testing.assert_allclose(learned1.support_probability, learned2.support_probability, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(learned1.replacement_mask, learned2.replacement_mask)

    def test_raw_readonly_bitwise_immutable_and_malformed_v2_result_falls_back(self):
        raw = self.signal.copy()
        before = raw.tobytes()
        raw.setflags(write=False)
        result = v3.apply_model_to_array(self.bundle(), raw)
        self.assertEqual(before, raw.tobytes())
        self.assertFalse(np.shares_memory(result.denoised, raw))
        with patch.object(v2, "apply_model_to_array", return_value={"unsafe": "not a result"}):
            fallback = v3.apply_model_to_array(self.bundle(), raw)
        np.testing.assert_array_equal(fallback.denoised, raw)
        self.assertTrue(fallback.rejected_mask.all())
        self.assertTrue(np.isnan(fallback.support_probability).all())
        self.assertIn("invalid frozen v2 result shapes", fallback.diagnostics["v2_prediction_error"])

    def test_bad_masks_input_threshold_or_cuda_request_fail_explicitly(self):
        for raw in (np.zeros(500), np.zeros((1, 0)), np.zeros((1, 500), dtype=bool),
                    np.full((1, 500), 2 ** 63, dtype=np.uint64)):
            with self.assertRaises(v3.LearnedDenoiseV3Error):
                v3.apply_model_to_array(self.bundle(), raw)
        for mask in (np.ones(500, dtype=bool), np.full(self.signal.shape, .5)):
            with self.assertRaises(v3.LearnedDenoiseV3Error):
                v3.apply_model_to_array(self.bundle(), self.signal, valid_mask=mask)
        for threshold in (.49, float("nan"), True, 1.1):
            with self.assertRaises(v3.LearnedDenoiseV3Error):
                v3.apply_model_to_array(self.bundle(threshold=threshold), self.signal)
        with patch.object(v3.torch.cuda, "is_available", return_value=False), self.assertRaisesRegex(v3.LearnedDenoiseV3Error, "no CPU fallback"):
            v3.load_model("no-access-gate.pt", "no-access-v2.pt", device="cuda")


@unittest.skipIf(v3.torch is None, "optional PyTorch unavailable")
class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = v3.torch.get_num_threads()
        v3.torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        v3.torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="v3 CPU fixture ")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path, self.v2_path = self.directory / "gate.pt", self.directory / "frozen-v2.pt"
        self.config = v3.load_config()
        v2_config = v2.load_config()
        with v3.torch.random.fork_rng(devices=[]):
            v3.torch.manual_seed(20261005)
            model = v2.build_model(v2_config)
            with v3.torch.no_grad():
                model.noise_head.weight.normal_(0., .02)
                model.probability_head.weight.zero_()
                model.probability_head.bias.fill_(10.)
            gate = v3.build_gate(self.config)
            with v3.torch.no_grad():
                for parameter in gate.parameters():
                    parameter.zero_()
                gate.logit.bias.fill_(10.)
        self.v2_checkpoint = {"checkpoint_schema_version": 1, "model_version": v2.MODEL_VERSION,
            "config": v2_config, "architecture": deepcopy(v2_config["model"]), "model_state": model.state_dict(),
            "calibrated_threshold": .5,
            "threshold_calibration": {"scope": "validation_only", "comparison": ">", "production_calibrated": False,
                "method": "clean_score_order_statistic", "score_reduction": "per_channel_window_mean", "empirical_only": True,
                "clean_window_count": 100, "observed_clean_false_replacement_fraction": 0.,
                "max_clean_false_replacement_fraction": .01}}
        v3.torch.save(self.v2_checkpoint, self.v2_path)
        self.checkpoint = {"checkpoint_schema_version": 1, "model_version": v3.MODEL_VERSION,
            "config": deepcopy(self.config), "gate_state": gate.state_dict(), "support_threshold": .5,
            "threshold_calibration": calibration(), "denoiser_sha256": sha256(self.v2_path),
            "dataset_sha256": "1" * 64, "metadata": {}}
        self.save()
        t = np.arange(507) / 250
        self.signal = np.stack([10 + np.sin(2 * np.pi * 11 * t), 5 + 2 * np.cos(2 * np.pi * 13 * t)])

    def save(self, checkpoint=None):
        v3.torch.save(self.checkpoint if checkpoint is None else checkpoint, self.path)

    def test_real_small_cpu_checkpoint_roundtrip_only_nine_keys_and_weights_only(self):
        before = (sha256(self.path), sha256(self.v2_path))
        self.assertEqual(v3.CHECKPOINT_KEYS, set(self.checkpoint))
        with patch.object(v3.torch, "load", wraps=v3.torch.load) as loader:
            bundle = v3.load_model(self.path, self.v2_path, device="cpu")
        self.assertEqual(3, loader.call_count)
        for call in loader.call_args_list:
            self.assertIs(True, call.kwargs["weights_only"])
            self.assertEqual("cpu", call.kwargs["map_location"])
        self.assertEqual("cpu", bundle.device)
        self.assertEqual(self.path.resolve(), bundle.checkpoint_path)
        self.assertEqual(self.config, bundle.config)
        self.assertTrue(v3.primitive_metadata(bundle.metadata))
        self.assertNotIn("gate_state", bundle.metadata)
        self.assertTrue(all(not parameter.requires_grad for parameter in bundle.denoiser.model.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in bundle.gate.parameters()))
        result = v3.apply_model_to_array(bundle, self.signal)
        base = v2.apply_model_to_array(bundle.denoiser, self.signal)
        for field in FIELDS[:7]:
            np.testing.assert_array_equal(getattr(result, field), getattr(base, field))
        self.assertTrue(result.replacement_mask[:, :500].all())
        self.assertGreater(result.diagnostics["changed_samples"], 0)
        self.assertEqual(before, (sha256(self.path), sha256(self.v2_path)))

    def test_negative_real_checkpoint_is_raw_while_candidate_is_still_v2(self):
        self.checkpoint["gate_state"]["logit.bias"].fill_(-10.)
        self.save()
        bundle = v3.load_model(self.path, self.v2_path, self.config)
        result = v3.apply_model_to_array(bundle, self.signal)
        np.testing.assert_array_equal(result.denoised.view(np.uint64), self.signal.view(np.uint64))
        base = v2.apply_model_to_array(bundle.denoiser, self.signal)
        np.testing.assert_array_equal(result.model_denoised, base.model_denoised)
        self.assertFalse(np.array_equal(result.model_denoised[:, :500], self.signal[:, :500]))
        self.assertTrue(result.rejected_mask.all())

    def test_checkpoint_metadata_provenance_and_weights_fail_closed(self):
        edits = [("model_version", v2.MODEL_VERSION), ("checkpoint_schema_version", True),
                 ("support_threshold", .49), ("support_threshold", 1.01), ("support_threshold", True),
                 ("support_threshold", float("nan")), ("denoiser_sha256", "0" * 64),
                 ("dataset_sha256", "not-a-hash"), ("metadata", {"nan": float("nan")}),
                 ("metadata", {"tuple": (1, 2)}), ("metadata", {"numpy": np.float64(.5)})]
        for key, value in edits:
            bad = deepcopy(self.checkpoint); bad[key] = value
            self.save(bad)
            with self.subTest(key=key, value=value), self.assertRaises(v3.LearnedDenoiseV3Error):
                v3.load_model(self.path, self.v2_path)
        for key, value in (("scope", "test"), ("comparison", ">="), ("empirical_only", False),
                           ("production_calibrated", True), ("method", "clean_score_order_statistic"),
                           ("unsupported_window_count", 0), ("unsupported_window_count", True),
                           ("observed_unsupported_false_acceptance_fraction", .011),
                           ("observed_unsupported_false_acceptance_fraction", -.1),
                           ("max_unsupported_false_acceptance_fraction", .02)):
            bad = deepcopy(self.checkpoint); bad["threshold_calibration"][key] = value
            self.save(bad)
            with self.subTest(calibration_key=key), self.assertRaises(v3.LearnedDenoiseV3Error):
                v3.load_model(self.path, self.v2_path)
        variants = []
        bad = deepcopy(self.checkpoint); del bad["dataset_sha256"]; variants.append(bad)
        bad = deepcopy(self.checkpoint); bad["config"]["model"]["kernel_sizes"][0] = 11; variants.append(bad)
        bad = deepcopy(self.checkpoint); bad["config"]["runtime"]["enabled"] = True; variants.append(bad)
        bad = deepcopy(self.checkpoint); bad["architecture"] = {"wrong": "model"}; variants.append(bad)
        bad = deepcopy(self.checkpoint); bad["gate_state"]["logit.bias"].fill_(float("nan")); variants.append(bad)
        bad = deepcopy(self.checkpoint); bad["gate_state"]["logit.bias"] = v3.torch.zeros(2); variants.append(bad)
        bad = deepcopy(self.checkpoint); del bad["gate_state"]["logit.bias"]; variants.append(bad)
        for index, bad in enumerate(variants):
            self.save(bad)
            with self.subTest(variant=index), self.assertRaises(v3.LearnedDenoiseV3Error):
                v3.load_model(self.path, self.v2_path)

    def test_external_config_architecture_inference_binding_and_missing_files(self):
        modified = deepcopy(self.config)
        modified["inference"]["min_ac_rms"] *= 2
        v3.validate_config(modified)
        with self.assertRaisesRegex(v3.LearnedDenoiseV3Error, "does not match"):
            v3.load_model(self.path, self.v2_path, modified)
        config_path = self.directory / "config.json"
        config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.assertEqual(self.config, v3.load_model(self.path, self.v2_path, config_path).config)
        for gate, denoiser in (("", self.v2_path), (self.path, ""), (self.directory / "missing.pt", self.v2_path),
                               (self.path, self.directory / "missing.pt")):
            with self.assertRaises(v3.LearnedDenoiseV3Error):
                v3.load_model(gate, denoiser)

    def test_bound_v2_nonfinite_weights_or_wrong_rate_are_not_accepted(self):
        for bad_rate in (False, True):
            frozen = deepcopy(self.v2_checkpoint)
            if bad_rate:
                frozen["config"]["sample_rate_hz"] = 256
            else:
                frozen["model_state"]["noise_head.bias"].fill_(float("inf"))
            v3.torch.save(frozen, self.v2_path)
            self.checkpoint["denoiser_sha256"] = sha256(self.v2_path)
            self.save()
            with self.subTest(bad_rate=bad_rate), self.assertRaises(v3.LearnedDenoiseV3Error):
                v3.load_model(self.path, self.v2_path)

    def test_pickle_object_and_mutating_checkpoint_are_refused_without_fallback(self):
        bad = deepcopy(self.checkpoint); bad["metadata"] = {"custom": Path("unsafe")}
        self.save(bad)
        with self.assertRaises(v3.LearnedDenoiseV3Error):
            v3.load_model(self.path, self.v2_path)
        self.save()
        original_load = v3.torch.load
        def altered_load(*args, **kwargs):
            value = original_load(*args, **kwargs)
            if Path(args[0]) == self.path:
                with self.path.open("ab") as handle:
                    handle.write(b"fixture concurrent mutation")
            return value
        with patch.object(v3.torch, "load", side_effect=altered_load), self.assertRaisesRegex(v3.LearnedDenoiseV3Error, "changed during loading"):
            v3.load_model(self.path, self.v2_path)

    @unittest.skipUnless(v3.torch is not None and v3.torch.cuda.is_available(), "CUDA unavailable; no silent CPU fallback")
    def test_explicit_cuda_gate_and_cpu_frozen_denoiser(self):
        bundle = v3.load_model(self.path, self.v2_path, device="cuda")
        self.assertTrue(bundle.device.startswith("cuda:"))
        self.assertTrue(all(parameter.device.type == "cuda" for parameter in bundle.gate.parameters()))
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in bundle.denoiser.model.parameters()))
        result = v3.apply_model_to_array(bundle, self.signal)
        self.assertTrue(result.replacement_mask[:, :500].all())
        self.assertEqual("cpu", result.diagnostics["denoiser_device"])


if __name__ == "__main__":
    unittest.main()
