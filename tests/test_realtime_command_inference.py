"""Synthetic/mocked inference tests; no personal EEG or locked model is read."""
from __future__ import annotations

from copy import deepcopy
import io
import json
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from eeg_tools import realtime_command_inference as api
from eeg_tools import matched_ssvep_decoder as decoder
from scripts import run_realtime_command_inference as cli


def window(n=1250):
    time = np.arange(n) / 250
    samples = np.stack([np.sin(2 * np.pi * 10 * time + ch / 7)
                        + 0.2 * np.sin(2 * np.pi * 20 * time + ch / 9) for ch in range(8)])
    return {"type": "window", "request_id": 7, "generation": 3, "sample_rate_hz": 250,
            "channel_names": list(api.CHANNEL_NAMES), "samples": samples.tolist()}


class SyntheticLogits(torch.nn.Module):
    """Known logits only; never an EEG-trained model or accuracy evidence."""

    def __init__(self, mean=None, logits=(8., 0., 0.)):
        super().__init__()
        self.register_buffer("feature_mean", torch.tensor(np.zeros(72) if mean is None else mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.ones(72))
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))
        self.inputs = []

    def forward(self, features):
        self.inputs.append(features.detach().cpu().clone())
        return self.logits.unsqueeze(0).expand(features.shape[0], 3)


def engine_for(request=None, logits=(8., 0., 0.), *, mean=None):
    if mean is None:
        mean = decoder.spectral_features(np.asarray((request or window())["samples"]))
    model = SyntheticLogits(mean, logits)
    frozen_flags = deepcopy(decoder.FLAGS)
    with patch.object(api, "load_selected_model", return_value=(model, frozen_flags, {"synthetic": True})) as loader:
        engine = api.RealtimeCommandInference(Path("synthetic-mock-run"))
        loader.assert_called_once_with(Path("synthetic-mock-run"), torch.device("cpu"))
    return engine, model, frozen_flags


class InferenceTests(unittest.TestCase):
    def assert_unknown(self, response):
        self.assertEqual(response["prediction"], {"command": "UNKNOWN", "accepted": False, "executed": False})
        self.assertFalse(response["personal_validated"])
        self.assertFalse(response["runtime_enabled"])
        self.assertFalse(response["production_validated"])

    def test_actual_features_and_forward_request_association(self):
        request = window(1250)
        engine, model, original_flags = engine_for(request)
        expected_flags = deepcopy(original_flags)
        response = engine.infer(request)
        self.assertEqual(response["prediction"], {"command": "MOVE_LEFT", "accepted": True, "executed": False})
        self.assertEqual((response["request_id"], response["generation"]), (7, 3))
        self.assertEqual(response["source"], "realtime_eeg_model")
        self.assertTrue(response["experiment_only"])
        self.assertNotIn("live_game_only", response)
        self.assertNotIn("game", response)
        self.assertTrue(response["window_matches_training"])
        self.assertEqual(response["sample_count"], 1250)
        self.assertEqual(len(response["probabilities"]), 3)
        np.testing.assert_array_equal(model.inputs[0].numpy()[0], decoder.spectral_features(np.asarray(request["samples"])))
        self.assertEqual(original_flags, expected_flags)
        self.assertNotIn("samples", response)

    def test_exactly_five_seconds_without_padding_cropping_or_resampling(self):
        engine, model, _ = engine_for()
        for n in (0, 250, 999, 1000, 1139, 1249, 1251, 1500):
            with self.subTest(n=n), patch.object(decoder, "spectral_features") as extract:
                response = engine.infer(window(n))
                self.assert_unknown(response)
                self.assertEqual(response["reason"], "invalid_sample_shape")
                extract.assert_not_called()
        self.assertEqual(model.inputs, [])
        request = window()
        with patch.object(decoder, "spectral_features", wraps=decoder.spectral_features) as extract:
            response = engine.infer(request)
        self.assertTrue(response["window_matches_training"])
        self.assertEqual(response["sample_count"], 1250)
        np.testing.assert_array_equal(extract.call_args.args[0], request["samples"])
        self.assertEqual(extract.call_args.args[0].shape, (8, 1250))

    def test_exact_schema_shape_rate_channel_association_and_no_labels(self):
        engine, model, _ = engine_for()
        cases = []
        for n in (999, 1000, 1249, 1251):
            cases.append(window(n))
        cases.extend([[], None, "window"])
        bad = window(); bad["samples"][1].pop(); cases.append(bad)
        bad = window(); bad["samples"] = bad["samples"][:7]; cases.append(bad)
        bad = window(); bad["samples"][0][0] = "0.2"; cases.append(bad)
        bad = window(); bad["samples"][0][0] = True; cases.append(bad)
        bad = window(); bad["samples"] = np.zeros((1000, 8)).tolist(); cases.append(bad)
        bad = window(); bad["sample_rate_hz"] = 256; cases.append(bad)
        bad = window(); bad["channel_names"][0] = "FP1"; cases.append(bad)
        bad = window(); bad["channel_names"] = list(reversed(api.CHANNEL_NAMES)); cases.append(bad)
        bad = window(); bad["request_id"] = True; cases.append(bad)
        bad = window(); bad["generation"] = -1; cases.append(bad)
        for forbidden in ("target", "labels", "label", "groundtruth", "expectedcommand", "frequency_hz", "prediction"):
            bad = window(); bad[forbidden] = "MOVE_LEFT"; cases.append(bad)
        for bad in cases:
            with self.subTest(type=type(bad)):
                self.assert_unknown(engine.infer(bad))
        self.assertEqual(model.inputs, [])

    def test_flat_nonfinite_rejected_before_model(self):
        engine, model, _ = engine_for()
        for value in (0., float("nan"), float("inf")):
            bad = window()
            bad["samples"][3] = [value] * 1250
            response = engine.infer(bad)
            self.assert_unknown(response)
            self.assertIn(response["reason"], ("flat_channel", "samples_must_be_finite"))
        self.assertEqual(model.inputs, [])

    def test_all_three_commands_including_select_are_directly_preserved(self):
        for logits, expected, reason in (((8, 0, 0), "MOVE_LEFT", "accepted"),
                                         ((0, 8, 0), "MOVE_RIGHT", "accepted"),
                                         ((0, 0, 8), "SELECT", "accepted"),
                                         ((0, 0, 0), "UNKNOWN", "model_confidence_gate")):
            engine, _, _ = engine_for(logits=logits)
            response = engine.infer(window())
            self.assertEqual(response["prediction"]["command"], expected)
            self.assertEqual(response["reason"], reason)
            self.assertFalse(response["prediction"]["executed"])
            if reason == "accepted":
                self.assertEqual(response["raw_prediction"]["command"], expected)
                self.assertEqual(response["prediction"]["command"], expected)

    def test_fixed_ood_limits_inclusive_with_raw_probability_retention(self):
        engine, model, _ = engine_for(mean=np.zeros(72))
        cases = [(np.array([8.] + [4.] * 71, dtype=np.float32), True),
                 (np.array([8.01] + [0.] * 71, dtype=np.float32), False),
                 (np.full(72, 4.1, dtype=np.float32), False)]
        for features, passed in cases:
            with patch.object(decoder, "spectral_features", return_value=features):
                response = engine.infer(window())
            self.assertEqual(response["ood"]["passed"], passed)
            self.assertTrue(response["ood"]["heuristic"])
            self.assertFalse(response["ood"]["validated"])
            self.assertEqual(response["raw_prediction"]["command"], "MOVE_LEFT")
            self.assertEqual(len(response["probabilities"]), 3)
            if not passed:
                self.assert_unknown(response)
                self.assertEqual(response["reason"], "ood_feature_gate")
        self.assertEqual(len(model.inputs), 3)

    def test_bad_logits_or_features_and_forward_exception_fail_closed(self):
        engine, model, _ = engine_for(logits=(float("nan"), 0, 0))
        self.assert_unknown(engine.infer(window()))
        with patch.object(decoder, "spectral_features", return_value=np.full(72, float("nan"))):
            self.assert_unknown(engine.infer(window()))
        with patch.object(model, "forward", side_effect=RuntimeError("synthetic failure")):
            response = engine.infer(window())
        self.assert_unknown(response)
        self.assertEqual(response["reason"], "inference_failed")

    def test_ready_no_global_flags_changed_and_cuda_no_fallback(self):
        before = deepcopy(decoder.FLAGS)
        engine, _, _ = engine_for()
        ready = engine.ready()
        self.assertEqual(ready["type"], "ready")
        self.assertTrue(ready["model_loaded"] and ready["experiment_only"])
        self.assertFalse(ready["personal_validated"])
        self.assertEqual(ready["device"], "cpu")
        self.assertEqual(decoder.FLAGS, before)
        with patch("torch.cuda.is_available", return_value=False), patch.object(api, "load_selected_model") as loader:
            with self.assertRaisesRegex(ValueError, "CUDA unavailable"):
                api.RealtimeCommandInference("mock", device="cuda")
            loader.assert_not_called()


class NDJSONTests(unittest.TestCase):
    def test_protocol_continues_invalid_json_and_stop_with_association(self):
        engine, _, _ = engine_for()
        inputs = b"not-json\n" + json.dumps(window()).encode() + b'\n{"type":"stop","request_id":9,"generation":3}\n'
        out, err = io.StringIO(), io.StringIO()
        code = cli.serve(engine, io.BytesIO(inputs), out, err)
        messages = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(code, 0)
        self.assertEqual([m["type"] for m in messages], ["ready", "prediction", "prediction", "stopped"])
        self.assertEqual(messages[1]["reason"], "invalid_json")
        self.assertEqual(messages[2]["request_id"], 7)
        self.assertEqual(messages[3]["request_id"], 9)

    def test_oversized_line_drained_without_losing_next_request(self):
        engine, _, _ = engine_for()
        huge = b"x" * (cli.MAX_LINE_BYTES * 3 + 2) + b'\n{"type":"stop"}\n'
        out = io.StringIO()
        cli.serve(engine, io.BytesIO(huge), out, io.StringIO())
        messages = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual([m["type"] for m in messages], ["ready", "prediction", "stopped"])
        self.assertEqual(messages[1]["reason"], "line_too_large")

    def test_duplicate_keys_nonfinite_unicode_and_bad_stop_rejected(self):
        engine, _, _ = engine_for()
        inputs = b'{"type":"window","type":"stop"}\n{"type":"window","samples":NaN}\n\xff\n{"type":"stop","target":10}\n'
        out = io.StringIO()
        self.assertEqual(cli.serve(engine, io.BytesIO(inputs), out, io.StringIO()), 0)
        messages = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual([m["reason"] for m in messages[1:]], ["invalid_json"] * 3 + ["invalid_stop_request"])
        self.assertTrue(all(m["prediction"]["command"] == "UNKNOWN" for m in messages[1:]))

    def test_main_load_failure_exit2_no_ready_and_default_cpu(self):
        out, err = io.StringIO(), io.StringIO()
        with patch.object(cli, "RealtimeCommandInference", side_effect=ValueError("mock source hash mismatch")) as loader:
            code = cli.main([], input_stream=io.BytesIO(), output=out, errors=err)
        self.assertEqual(code, 2)
        loader.assert_called_once_with(api.DEFAULT_RUN_DIR, device="cpu")
        message = json.loads(out.getvalue())
        self.assertEqual(message["type"], "error")
        self.assertEqual(message["error"], "model_load_failed")
        self.assertFalse(message["model_loaded"])
        self.assertIn("hash mismatch", err.getvalue())

    def test_stdout_contains_only_json_even_if_loader_prints(self):
        engine, _, _ = engine_for()
        def noisy(*args, **kwargs):
            print("synthetic loader log")
            return engine
        out, err = io.StringIO(), io.StringIO()
        with patch.object(cli, "RealtimeCommandInference", side_effect=noisy):
            code = cli.main([], input_stream=io.BytesIO(b'{"type":"stop"}\n'), output=out, errors=err)
        self.assertEqual(code, 0)
        self.assertIn("synthetic loader log", err.getvalue())
        self.assertEqual([json.loads(line)["type"] for line in out.getvalue().splitlines()], ["ready", "stopped"])


class StrictContractTests(unittest.TestCase):
    def test_ready_and_all_envelopes_have_formal_nonexecution_flags(self):
        engine, _, _ = engine_for()
        output = io.StringIO()
        cli.serve(engine, io.BytesIO(b'bad json\n{"type":"stop","request_id":0,"generation":0}\n'), output, io.StringIO())
        for envelope in map(json.loads, output.getvalue().splitlines()):
            with self.subTest(kind=envelope["type"]):
                for key, value in api.FLAGS.items():
                    self.assertEqual(envelope[key], value)
                self.assertEqual(envelope["source"], "realtime_eeg_model")
                self.assertEqual(envelope["status"], "formal")
                self.assertTrue(envelope["experiment_only"])
                for key in ("personal_validated", "runtime_enabled", "production_validated",
                            "default_deployment_authorized", "executed"):
                    self.assertIs(envelope[key], False)
                self.assertNotIn("game", envelope)
                self.assertNotIn("live_game_only", envelope)
        ready = engine.ready()
        self.assertEqual(ready["sample_count"], 1250)
        self.assertEqual(ready["training_sample_count"], 1250)
        self.assertEqual(ready["window_seconds"], 5)
        self.assertEqual(ready["sample_rate_hz"], 250)
        self.assertEqual(ready["channel_names"], list(api.CHANNEL_NAMES))
        self.assertEqual(ready["command_gate"], decoder.GATE)
        self.assertEqual(ready["command_gate"]["min_probability"], .65)
        self.assertEqual(ready["command_gate"]["min_probability_margin"], .15)
        self.assertEqual(ready["commands"], {"10": "MOVE_LEFT", "12": "MOVE_RIGHT", "15": "SELECT"})

    def test_real_72_feature_mlp_and_actual_softmax(self):
        request = window()
        features = decoder.spectral_features(np.asarray(request["samples"]))
        model = decoder.HarmonicSpectrumMLP()
        with torch.no_grad():
            model.feature_mean.copy_(torch.tensor(features))
            model.feature_std.fill_(2.)
            for parameter in model.parameters():
                parameter.zero_()
            model.layers[-1].bias.copy_(torch.tensor([0., 0., 8.]))
        with patch.object(api, "load_selected_model", return_value=(model, {}, {})):
            engine = api.RealtimeCommandInference("mock")
        self.assertFalse(model.training)
        response = engine.infer(request)
        self.assertEqual(response["prediction"]["command"], "SELECT")
        self.assertEqual(response["raw_prediction"]["predicted_frequency_hz"], 15)
        np.testing.assert_allclose(response["probabilities"], torch.softmax(torch.tensor([0., 0., 8.]), -1).numpy())
        self.assertEqual(response["ood"]["maxabs"], 0.)

    def test_inclusive_probability_threshold_without_tuning(self):
        engine, _, _ = engine_for()
        # Exact double values isolate the locked boundary from float32 rounding.
        for probability, accepted in ((.649999, False), (.65, True), (.650001, True)):
            probabilities = torch.tensor([[probability, 1 - probability, 0.]], dtype=torch.float64)
            with self.subTest(probability=probability), patch.object(torch, "softmax", return_value=probabilities):
                response = engine.infer(window())
            self.assertEqual(response["prediction"]["accepted"], accepted)
            self.assertEqual(response["raw_prediction"]["accepted"], accepted)
            self.assertEqual(response["reason"], "accepted" if accepted else "model_confidence_gate")
            self.assertTrue(response["ood"]["passed"])
            self.assertEqual(response["probabilities"], probabilities[0].tolist())

    def test_ood_standardizes_once_but_forward_receives_raw_features(self):
        engine, model, _ = engine_for(mean=np.full(72, 10.))
        engine.feature_std[:] = 2.
        features = np.full(72, 18., dtype=np.float32)
        with patch.object(decoder, "spectral_features", return_value=features):
            response = engine.infer(window())
        self.assertEqual(response["ood"]["maxabs"], 4.)
        self.assertEqual(response["ood"]["p95abs"], 4.)
        self.assertTrue(response["prediction"]["accepted"])
        np.testing.assert_array_equal(model.inputs[0].numpy(), features[None, :])

    def test_bad_model_standardizers_fail_loading(self):
        for kind in ("shape", "nan", "std_floor", "negative_std"):
            model = SyntheticLogits()
            if kind == "shape":
                model.feature_mean = torch.zeros(71)
            elif kind == "nan":
                model.feature_mean[0] = float("nan")
            else:
                model.feature_std[0] = 0. if kind == "std_floor" else -1.
            with self.subTest(kind=kind), patch.object(api, "load_selected_model", return_value=(model, {}, {})):
                with self.assertRaisesRegex(ValueError, "invalid_model_standardizer"):
                    api.RealtimeCommandInference("mock")

    def test_bad_feature_shape_and_logits_shape_fail_closed(self):
        engine, model, _ = engine_for()
        for features in (np.zeros(71), np.zeros((1, 72)), np.full(72, np.inf)):
            with patch.object(decoder, "spectral_features", return_value=features):
                response = engine.infer(window())
            self.assertEqual(response["reason"], "nonfinite_features")
            self.assertEqual(response["prediction"]["command"], "UNKNOWN")
        self.assertEqual(model.inputs, [])
        for logits in (torch.zeros(3), torch.zeros(1, 4), torch.full((1, 3), float("inf")), "bad"):
            with patch.object(model, "forward", return_value=logits):
                response = engine.infer(window())
            self.assertEqual(response["reason"], "nonfinite_or_invalid_logits")
            self.assertEqual(response["prediction"]["command"], "UNKNOWN")

    def test_rejected_window_keeps_each_valid_association_id(self):
        engine, _, _ = engine_for()
        request = window()
        request["target"] = 10
        response = engine.infer(request)
        self.assertEqual((response["request_id"], response["generation"]), (7, 3))
        request["request_id"] = True
        response = engine.infer(request)
        self.assertIsNone(response["request_id"])
        self.assertEqual(response["generation"], 3)
        request["request_id"] = 0
        request["generation"] = 0
        response = engine.infer(request)
        self.assertEqual((response["request_id"], response["generation"]), (0, 0))

    def test_cuda_is_explicit_and_forwarded_to_verified_loader(self):
        model = SyntheticLogits()
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
            api, "load_selected_model", return_value=(model, {}, {})
        ) as loader:
            engine = api.RealtimeCommandInference("mock", device="cuda")
        loader.assert_called_once_with(Path("mock"), torch.device("cuda"))
        self.assertEqual(engine.ready()["device"], "cuda")


class ProtocolBoundaryTests(unittest.TestCase):
    def run_protocol(self, wire):
        engine, _, _ = engine_for()
        output, errors = io.StringIO(), io.StringIO()
        code = cli.serve(engine, io.BytesIO(wire), output, errors)
        self.assertEqual(code, 0)
        return [json.loads(line) for line in output.getvalue().splitlines()]

    def test_exact_line_limit_and_one_byte_over_then_resynchronization(self):
        stop = b'{"type":"stop","request_id":5,"generation":3}'
        exact = stop + b' ' * (cli.MAX_LINE_BYTES - len(stop) - 1) + b'\n'
        self.assertEqual(len(exact), cli.MAX_LINE_BYTES)
        self.assertEqual([m["type"] for m in self.run_protocol(exact)], ["ready", "stopped"])
        oversized = exact[:-1] + b' \n'
        messages = self.run_protocol(oversized + stop + b'\n')
        self.assertEqual([m["type"] for m in messages], ["ready", "prediction", "stopped"])
        self.assertEqual(messages[1]["reason"], "line_too_large")
        self.assertEqual(messages[2]["request_id"], 5)

    def test_eof_and_unterminated_last_line(self):
        self.assertEqual([m["type"] for m in self.run_protocol(b'')], ["ready"])
        wire = json.dumps(window(), separators=(",", ":")).encode()
        self.assertLessEqual(len(wire), cli.MAX_LINE_BYTES)
        messages = self.run_protocol(wire)
        self.assertEqual([m["type"] for m in messages], ["ready", "prediction"])
        self.assertEqual(messages[1]["request_id"], 7)
        messages = self.run_protocol(b'x' * (cli.MAX_LINE_BYTES + 1))
        self.assertEqual(messages[1]["reason"], "line_too_large")

    def test_stop_does_not_read_or_infer_following_requests(self):
        engine, model, _ = engine_for()
        wire = b'{"type":"stop","request_id":9,"generation":2}\n' + json.dumps(window()).encode() + b'\n'
        source = io.BytesIO(wire)
        output = io.StringIO()
        cli.serve(engine, source, output, io.StringIO())
        self.assertEqual([json.loads(line)["type"] for line in output.getvalue().splitlines()], ["ready", "stopped"])
        self.assertEqual(model.inputs, [])
        self.assertNotEqual(source.read(), b'')

    def test_bad_stop_ids_fields_and_valid_partial_ids(self):
        for request in ({"type": "stop", "labels": []}, {"type": "stop", "generation": -1},
                        {"type": "stop", "request_id": False}, {"type": "stop", "request_id": "7"}):
            messages = self.run_protocol(json.dumps(request).encode() + b'\n{"type":"stop"}\n')
            self.assertEqual(messages[1]["reason"], "invalid_stop_request")
            self.assertEqual(messages[1]["prediction"]["command"], "UNKNOWN")
            self.assertEqual(messages[-1]["type"], "stopped")
        messages = self.run_protocol(b'{"type":"stop","generation":0}\n')
        self.assertEqual(messages[-1]["generation"], 0)
        self.assertIsNone(messages[-1]["request_id"])

    def test_deep_json_and_overflow_json_number_fail_closed(self):
        nested = b'[' * 2000 + b'0' + b']' * 2000 + b'\n'
        request = window()
        request["samples"][0][0] = float("inf")
        huge_number = json.dumps(request).encode().replace(b'Infinity', b'1e999')
        messages = self.run_protocol(nested + huge_number + b'\n{"type":"stop"}\n')
        self.assertIn(messages[1]["reason"], ("invalid_json", "request_must_be_object"))
        self.assertEqual(messages[1]["prediction"]["command"], "UNKNOWN")
        self.assertEqual(messages[2]["reason"], "samples_must_be_finite")
        self.assertEqual(messages[2]["request_id"], 7)

    def test_all_diagnostic_sources_share_a_bounded_stderr_budget(self):
        engine, _, _ = engine_for()
        original_infer, original_ready = engine.infer, engine.ready
        def noisy_load(*args, **kwargs):
            print("synthetic loader log")
            return engine
        def noisy_ready():
            print("synthetic ready log", file=sys.stderr)
            return original_ready()
        def noisy_infer(request):
            print("x" * (cli.MAX_LOG_BYTES * 2))
            print("more logs", file=sys.stderr)
            return original_infer(request)
        wire = json.dumps(window()).encode() + b'\n' + json.dumps(window()).encode() + b'\n{"type":"stop"}\n'
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(cli, "RealtimeCommandInference", side_effect=noisy_load), patch.object(
            engine, "infer", side_effect=noisy_infer
        ), patch.object(engine, "ready", side_effect=noisy_ready):
            self.assertEqual(cli.main([], input_stream=io.BytesIO(wire), output=output, errors=errors), 0)
        self.assertLessEqual(len(errors.getvalue().encode("utf-8")), cli.MAX_LOG_BYTES)
        self.assertIn("synthetic loader log", errors.getvalue())
        self.assertIn("synthetic ready log", errors.getvalue())
        self.assertEqual(errors.getvalue().count("[stderr log limit reached]"), 1)
        self.assertEqual([json.loads(line)["type"] for line in output.getvalue().splitlines()],
                         ["ready", "prediction", "prediction", "stopped"])

    def test_multibyte_diagnostics_and_oversized_exception_are_bounded(self):
        for exc in (RuntimeError("x" * (cli.MAX_LOG_BYTES * 2)), ValueError("\u6d4b" * cli.MAX_LOG_BYTES)):
            output, errors = io.StringIO(), io.StringIO()
            with patch.object(cli, "RealtimeCommandInference", side_effect=exc):
                code = cli.main([], input_stream=io.BytesIO(), output=output, errors=errors)
            self.assertEqual(code, 2)
            self.assertLessEqual(len(errors.getvalue().encode("utf-8")), cli.MAX_LOG_BYTES)
            message = json.loads(output.getvalue())
            self.assertEqual(message["type"], "error")
            for key, value in api.FLAGS.items():
                self.assertEqual(message[key], value)

    def test_help_and_invalid_arguments_keep_stdout_json_only(self):
        for args, expected in ((["--help"], 0), (["--device", "automatic"], 2)):
            output, errors = io.StringIO(), io.StringIO()
            with patch.object(cli, "RealtimeCommandInference") as loader:
                code = cli.main(args, input_stream=io.BytesIO(), output=output, errors=errors)
            loader.assert_not_called()
            self.assertEqual(code, expected)
            self.assertTrue(errors.getvalue())
            if code:
                self.assertEqual(json.loads(output.getvalue())["error"], "invalid_arguments")
            else:
                self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
