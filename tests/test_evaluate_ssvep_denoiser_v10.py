"""Lightweight v10 evaluator contracts; synthetic arrays and temporary files only.

No real checkpoint loading, CUDA, public data, personal EEG or evaluation run.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scripts import evaluate_ssvep_denoiser_v10 as evaluation


CONFIG = {
    "data": {"evaluation_subjects": [7, 8]},
    "runtime": {"enabled": False, "production_validated": False},
    "engineering_gates": {
        "identity_mean_normalized_change_max": 0.002,
        "corrupted_mean_error_reduction_min": 0.05,
        "protected_window_complex_energy_ratio_max": 1e-6,
        "all_conditions_nonworse": True,
        "independent_personal_validation": False,
    },
}


class EvaluationPathsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def source(self, path):
        path.mkdir(parents=True)
        for name in ("session.json", "channel_config.json", "raw_brainflow.tsv"):
            (path / name).write_text("placeholder", encoding="utf-8")
        return path

    def test_new_output_is_checked_without_creating(self):
        output, sources = evaluation.validate_paths("analysis_outputs/new", root=self.root)
        self.assertEqual(output, self.root / "analysis_outputs/new")
        self.assertEqual(sources, [])
        self.assertFalse(output.exists())

    def test_reject_existing_empty_or_populated_output(self):
        output = self.root / "analysis_outputs/old"
        output.mkdir(parents=True)
        for populated in (False, True):
            if populated:
                (output / "keep.txt").write_text("preserved", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already exists"):
                evaluation.validate_paths(output, root=self.root)
        self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "preserved")

    def test_reject_outside_analysis_root_and_traversal(self):
        for value in ("analysis_outputs", "other/new", "analysis_outputs/../new",
                      "analysis_outputs_sibling/new", "analysis_outputs/new:stream"):
            with self.subTest(path=value), self.assertRaises(ValueError):
                evaluation.validate_paths(value, root=self.root)

    def test_reject_output_inside_source(self):
        source = self.source(self.root / "analysis_outputs/source")
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluation.validate_paths(source / "new", [source], root=self.root)

    def test_reject_source_inside_output_and_duplicate_sources(self):
        source = self.source(self.root / "analysis_outputs/bundle/source")
        with self.assertRaises(ValueError):
            evaluation.validate_paths(source.parent, [source], root=self.root)
        with self.assertRaisesRegex(ValueError, "duplicate/overlapping"):
            evaluation.validate_paths("analysis_outputs/new", [source, source], root=self.root)

    def test_missing_or_incomplete_source_rejected(self):
        for source in (self.root / "missing", self.root):
            with self.subTest(source=source), self.assertRaises(ValueError):
                evaluation.validate_paths("analysis_outputs/new", [source], root=self.root)

    def test_valid_source_does_not_get_written(self):
        source = self.source(self.root / "source")
        before = {p.name: p.read_bytes() for p in source.iterdir()}
        _, sources = evaluation.validate_paths("analysis_outputs/new", [source], root=self.root)
        self.assertEqual(sources, [source])
        self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})

    def test_cli_invalid_path_fails_before_evaluation(self):
        with patch.object(evaluation, "ROOT", self.root), patch.object(evaluation, "evaluate") as run:
            with contextlib.redirect_stderr(io.StringIO()):
                status = evaluation.main(["--output-directory", str(self.root / "outside")])
        self.assertEqual(status, 2)
        run.assert_not_called()

    def test_cli_requires_output_and_valid_device(self):
        for arguments in ([], ["--output-directory", "analysis_outputs/new", "--device", "cuda:7"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                evaluation.main(arguments)
            self.assertEqual(error.exception.code, 2)

    def test_cli_defaults_gpu_and_preserves_repeated_sources(self):
        stub = {"output_directory": str(self.root / "analysis_outputs/new"),
                "research_gate": {"passed": False}}
        with patch.object(evaluation, "validate_paths"), patch.object(evaluation, "evaluate", return_value=stub) as run:
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                status = evaluation.main(["--output-directory", "analysis_outputs/new",
                                          "--personal-session", "first", "--personal-session", "second"])
        self.assertEqual(status, 0)
        run.assert_called_once_with("analysis_outputs/new", ["first", "second"], device="cuda:0")
        result = json.loads(stream.getvalue())
        self.assertIs(result["runtime_enabled"], False)
        self.assertIs(result["research_gate_passed"], False)

    def test_exclusive_strict_json_write(self):
        path = self.root / "report.json"
        evaluation._write_json(path, {"count": 1})
        with self.assertRaises(FileExistsError):
            evaluation._write_json(path, {"count": 2})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"count": 1})
        with self.assertRaises(ValueError):
            evaluation._write_json(self.root / "bad.json", {"number": float("nan")})
        self.assertFalse((self.root / "bad.json").exists())

    def test_import_and_help_are_lightweight(self):
        code = ("import sys; from scripts import evaluate_ssvep_denoiser_v10; "
                "assert not {'torch','numpy','scipy'} & set(sys.modules)")
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class EvaluationNumericsTests(unittest.TestCase):
    def test_synthetic_twenty_cases_reproducible_with_dc_background(self):
        count = 0
        for frequency in evaluation.FREQUENCIES:
            for condition in evaluation.SYNTHETIC_CONDITIONS:
                clean, noisy = evaluation.synthetic_case(frequency, condition, 31)
                same_clean, same_noisy = evaluation.synthetic_case(frequency, condition, 31)
                self.assertEqual(clean.shape, (4096, 8))
                self.assertEqual(noisy.dtype, np.float64)
                np.testing.assert_array_equal(clean, same_clean)
                np.testing.assert_array_equal(noisy, same_noisy)
                self.assertTrue((clean.mean(axis=0) > 99).all())
                if condition == "identity":
                    np.testing.assert_array_equal(clean, noisy)
                else:
                    self.assertGreater(float(np.mean((clean - noisy) ** 2)), 0)
                count += 1
        self.assertEqual(count, 20)
        with self.assertRaises(ValueError):
            evaluation.synthetic_case(11, "identity", 31)

    def test_input_validation_rejects_invalid_channels_short_flat_nonfinite(self):
        good, _ = evaluation.synthetic_case(12, "identity", 1)
        bad_nan = good.copy()
        bad_nan[10, 2] = np.nan
        flat = good.copy()
        flat[:, 3] = 5
        for value in (good[:, :7], good[:511], good.astype(np.int64), good.T, bad_nan, flat, []):
            with self.subTest(shape=getattr(value, "shape", None)), self.assertRaises(ValueError):
                evaluation.validate_signal(value)
        self.assertIs(evaluation.validate_signal(good), good)

    def test_receipt_preserves_contract_and_checks_appended_tail_window(self):
        for samples in (512, 640, 4096, 24371, 24372):
            raw = np.random.default_rng(1).normal(size=(samples, 8))
            receipt = {"window_samples": 512, "hop_samples": 128,
                       "processed_samples": samples,
                       "window_count": (samples - 512) // 128 + 1 + int((samples - 512) % 128 != 0),
                       "max_window_protected_complex_energy_ratio": 0.0}
            self.assertIs(evaluation.validate_processed(raw, raw.copy(), raw.copy(), receipt), receipt)
            wrong = dict(receipt, processed_samples=samples - 1)
            with self.assertRaises(ValueError):
                evaluation.validate_processed(raw, raw, raw, wrong)
            with self.assertRaises(ValueError):
                evaluation.validate_processed(raw, raw.astype(np.float32), raw, receipt)

    def test_centered_errors_separate_dc_and_edge_effects(self):
        clean, _ = evaluation.synthetic_case(15, "identity", 1)
        shifted = clean + 7
        boundary = clean.copy()
        boundary[:512] += 1
        errors = evaluation.centered_errors(clean, {"identity": clean, "dc": shifted, "boundary": boundary})
        self.assertEqual(errors["identity"]["full"]["clean_centered_mse"], 0)
        self.assertLess(errors["dc"]["full"]["clean_centered_mse"], 1e-20)
        self.assertAlmostEqual(errors["dc"]["full"]["mean_offset_mse"], 49)
        self.assertGreater(errors["boundary"]["full"]["clean_centered_mse"], 0)
        self.assertEqual(errors["boundary"]["interior"]["clean_centered_mse"], 0)

    def test_window_metrics_exclude_undefined_protection_and_identity_ratio(self):
        reference = np.ones((2, 8, 512), dtype=np.float32)
        result = evaluation.window_metrics(reference, reference, reference,
                                           np.array([0.0, np.nan]), np.array([True, False]))
        self.assertIsNone(result["output_to_input_mse_ratio"])
        self.assertEqual(result["max_defined_protected_complex_energy_ratio"], 0)
        self.assertEqual(result["protected_undefined_window_count"], 1)
        self.assertIsNone(result["windows"][1]["protected_complex_energy_ratio"])
        result = evaluation.window_metrics(reference, reference, reference,
                                           np.zeros(2), np.zeros(2, dtype=bool))
        self.assertIsNone(result["max_defined_protected_complex_energy_ratio"])

    def test_window_metrics_reject_invalid_shape_or_defined_aux(self):
        reference = np.ones((2, 8, 512), dtype=np.float32)
        for prediction, ratios, mask in ((reference[:, :7], [0, 0], [True, True]),
                                         (reference, [np.nan, 0], [True, True]),
                                         (reference, [0, 0], [1, 1])):
            with self.assertRaises(ValueError):
                evaluation.window_metrics(reference, reference, prediction, ratios, mask)

    def gate_reports(self):
        ref = np.ones((2, 8, 512), dtype=np.float32)
        conditions = {}
        for condition in evaluation.PUBLIC_CONDITIONS:
            noisy = ref if condition == "identity" else ref + 1
            conditions[condition] = evaluation.window_metrics(ref, noisy, ref,
                                                              np.zeros(2), np.ones(2, dtype=bool))
        cases = [{"frequency_hz": hz, "condition": condition,
                  "cca": {"v10": {"matches_known_frequency": True}}}
                 for hz in evaluation.FREQUENCIES for condition in evaluation.SYNTHETIC_CONDITIONS]
        return {"conditions": conditions}, {"cases": cases}

    def test_passing_research_gate_never_enables_runtime(self):
        public, synthetic = self.gate_reports()
        gate = evaluation.research_gate(public, synthetic, CONFIG)
        self.assertIs(gate["passed"], True)
        self.assertIs(gate["runtime_enabled"], False)
        self.assertIs(gate["production_validated"], False)
        self.assertIs(gate["independent_personal_validation"], False)
        self.assertIs(gate["clean_ground_truth_available"], False)

    def test_gate_rejects_all_undefined_or_error_regression_or_wrong_frequency(self):
        public, synthetic = self.gate_reports()
        for mutation in ("undefined", "regression", "frequency", "identity", "duplicate"):

            current, cases = copy.deepcopy(public), copy.deepcopy(synthetic)
            if mutation == "undefined":
                for row in current["conditions"].values():
                    row["protected_defined_window_count"] = 0
                    row["max_defined_protected_complex_energy_ratio"] = None
            elif mutation == "regression":
                current["conditions"]["gaussian"]["output_mse"] = 2
            elif mutation == "frequency":
                cases["cases"][0]["cca"]["v10"]["matches_known_frequency"] = False
            elif mutation == "duplicate":
                cases["cases"][1] = copy.deepcopy(cases["cases"][0])
            else:
                current["conditions"]["identity"]["mean_normalized_change_from_input"] = 0.003
            with self.subTest(mutation=mutation):
                self.assertIs(evaluation.research_gate(current, cases, CONFIG)["passed"], False)

    def test_public_uses_direct_forward_mock_data_no_recording_runtime(self):
        import torch
        from eeg_tools import ssvep_v10_data as data
        from eeg_tools import ssvep_v10_runtime as runtime
        reference = np.ones((2, 8, 512), dtype=np.float32)
        identities = [{"subject_id": 7}, {"subject_id": 8}]

        class DummyModel:
            calls = 0
            def eval(self):
                return self
            def forward_with_aux(self, tensor):
                self.calls += 1
                return tensor.clone(), {
                    "protected_complex_energy_ratio": torch.zeros(len(tensor)),
                    "protected_energy_defined": torch.ones(len(tensor), dtype=torch.bool),
                }

        model = DummyModel()
        with patch.object(data, "load_reference_windows", return_value=(reference, identities)) as load:
            with patch.object(data, "corrupt", side_effect=lambda ref, condition, seed, cfg: ref.copy()):
                with patch.object(runtime, "process_recording", side_effect=AssertionError("not neural stage")):
                    result = evaluation.evaluate_public(model, CONFIG, "cpu", root=Path("unread"))
        load.assert_called_once_with([7, 8], root=Path("unread"))
        self.assertEqual(model.calls, 5)
        self.assertFalse(result["checkpoint_selection_performed"])
        self.assertEqual(result["conditions"]["identity"]["output_mse"], 0)
        self.assertEqual(result["conditions"]["identity"]["windows"][1]["identity"]["subject_id"], 8)


if __name__ == "__main__":
    unittest.main()
