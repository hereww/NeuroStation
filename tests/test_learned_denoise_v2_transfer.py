from __future__ import annotations

import ast
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat
from eeg_tools.learned_denoise_v2 import DenoiseResult
from scripts import evaluate_denoiser_v2_transfer as evaluation


def metadata(mapped: bool = False) -> dict:
    return {
        "schema_version": "1.0", "dataset": "Dong2023",
        "metadata_status": "onset_confirmed_target_axis_mapping_confirmed" if mapped else "onset_confirmed_target_axis_mapping_pending",
        "sources": [{"url": "fixture://independent-author-audit", "role": "primary fixture evidence"}],
        "sample_rate_hz": 250, "array_key": "eegdata", "expected_shape": [2, 1250, 2, 1],
        "axis_order": ["channel", "time", "target", "block"], "target_count": 2,
        "target_frequencies_hz": [12.3, 8.7] if mapped else None,
        "target_axis_mapping_confirmed": mapped, "confirmed_frequency_set_hz": [8.7, 12.3],
        # Deliberately WRONG candidate order; evaluation must never use it as labels.
        "candidate_target_axis_frequencies_hz": [8.7, 12.3],
        "trial_structure": {"stimulus_onset_in_stored_epoch_s": .5, "stimulus_offset_in_stored_epoch_s": 4.5},
        "analysis_start_s": .5, "analysis_duration_s": 4.,
        "analysis_window_samples": {"start": 125, "stop_exclusive": 1125, "count": 1000, "indexing": "zero_based_half_open"},
    }


def bundle() -> SimpleNamespace:
    return SimpleNamespace(config={"sample_rate_hz": 250, "window_samples": 500, "hop_samples": 500}, validation_threshold=.6)


def fake_infer(_bundle, raw, valid_mask=None) -> DenoiseResult:
    # Sample-independent fake to check candidate/gated/common support without training.
    candidate = raw * .98
    replaced = np.zeros_like(raw, dtype=bool)
    replaced[0, :500] = True
    gated = np.where(replaced, candidate, raw)
    return DenoiseResult(gated, candidate, np.full_like(raw, .7), np.full_like(raw, .3),
                         replaced, np.ones_like(raw, dtype=bool), np.zeros_like(raw, dtype=bool),
                         {"window_samples": 500, "hop_samples": 500, "fixture": True})


def dong_fixture(root: Path, meta: dict) -> Path:
    time = np.arange(1250) / 250
    data = np.empty((2, 1250, 2, 1))
    for target, hz in enumerate([12.3, 8.7]):
        data[:, :, target, 0] = np.stack([2 * np.sin(2 * np.pi * hz * time), np.cos(2 * np.pi * hz * time)])
    for name in evaluation.SUBJECTS:
        savemat(root / name, {"eegdata": data})
    path = root / "metadata.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    return path


def regression_fixture(path: Path, count: int = 26, rate: float = 250) -> None:
    # Insert other sessions around the 26 fixed known windows, no score selection.
    time = np.arange(1000) / 250
    clean = np.tile(np.stack([np.sin(2 * np.pi * 9 * time), 2 * np.cos(2 * np.pi * 13 * time)]), (count + 2, 1, 1))
    artifact = np.zeros_like(clean, dtype=np.uint8)
    artifact[:, 0, :500] = 1
    noisy = clean + .5 * artifact
    sessions = np.array(["other"] + [evaluation.REGRESSION_SESSION] * count + ["other"])
    np.savez(path, noisy=noisy, clean=clean, artifact_mask=artifact, session_id=sessions,
             sample_rate_hz=np.full(count + 2, rate))


class MetadataTests(unittest.TestCase):
    def test_current_pending_mapping_uses_set_not_candidate(self):
        m = metadata()
        spec = evaluation.validate_metadata(m, 250)
        self.assertIsNone(spec["targets"])
        np.testing.assert_array_equal(spec["frequency_set"], [8.7, 12.3])
        self.assertEqual(spec["spectral_slice"], slice(125, 1125))
        self.assertEqual(spec["cca_slice"], slice(125, 1125))
        self.assertTrue(spec["timing_verified"])

    def test_local_source_references_and_supplemented_onset_status_are_supported(self):
        m = metadata()
        m["sources"].append({"role": "local tensor metadata/hash verification", "local_path": "fixture/LOCAL_MAT_VERIFICATION.json"})
        m["metadata_status"] = "updated_author_supplement_audit"
        m["stimulus_onset_status"] = "confirmed_primary_author_paper"
        m["author_latency_adjusted_start_sample"] = 165
        spec = evaluation.validate_metadata(m, 250)
        self.assertEqual(spec["spectral_slice"], slice(125, 1125))
        self.assertTrue(spec["timing_verified"])
        m["sources"][-1].pop("local_path")
        with self.assertRaisesRegex(evaluation.TransferEvaluationError, "audit reference"):
            evaluation.validate_metadata(m, 250)

    def test_confirmed_mapping_preserves_author_order_not_sorted_order(self):
        spec = evaluation.validate_metadata(metadata(True), 250)
        np.testing.assert_array_equal(spec["targets"], [12.3, 8.7])

    def test_all_required_metadata_fields_are_mandatory(self):
        m = metadata()
        for key in m:
            if key in ("candidate_target_axis_frequencies_hz", "trial_structure"):
                continue
            broken = deepcopy(m)
            del broken[key]
            with self.subTest(key=key), self.assertRaises(evaluation.TransferEvaluationError):
                evaluation.validate_metadata(broken, 250)

    def test_rejects_sample_rate_shape_axes_and_frequency_ambiguity(self):
        edits = [("sample_rate_hz", 1000), ("sample_rate_hz", True), ("sample_rate_hz", None),
                 ("expected_shape", [2, 1250, 2]), ("expected_shape", [2, 1250, 2, 0]),
                 ("axis_order", ["channel", "time", "target", "target"]), ("target_count", 40),
                 ("target_axis_mapping_confirmed", "true"), ("target_frequencies_hz", [8.7, 12.3]),
                 ("confirmed_frequency_set_hz", [8.7, 8.7]), ("sources", [])]
        for key, value in edits:
            m = metadata()
            m[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(evaluation.TransferEvaluationError):
                evaluation.validate_metadata(m, 250)

    def test_mapping_confirmation_without_frequency_order_is_rejected(self):
        m = metadata(True)
        for frequencies in (None, [8.7], [8.7, 10.1]):
            m["target_frequencies_hz"] = frequencies
            with self.subTest(frequencies=frequencies), self.assertRaises(evaluation.TransferEvaluationError):
                evaluation.validate_metadata(m, 250)

    def test_seconds_samples_and_onset_consistency_fail_closed(self):
        edits = [("analysis_start_s", 0), ("analysis_duration_s", 3.9),
                 ("analysis_start_s", .501),
                 ("analysis_window_samples", {"start": 125, "stop_exclusive": 1125, "count": 999, "indexing": "zero_based_half_open"}),
                 ("trial_structure", {"stimulus_onset_in_stored_epoch_s": None, "stimulus_offset_in_stored_epoch_s": 4.5})]
        for key, value in edits:
            m = metadata()
            m[key] = value
            with self.subTest(key=key), self.assertRaises(evaluation.TransferEvaluationError):
                evaluation.validate_metadata(m, 250)

    def test_unverified_onset_no_guessed_stimulus_crop(self):
        m = metadata()
        m["metadata_status"] = "timing_not_verified"
        m["trial_structure"] = {}
        with self.assertRaisesRegex(evaluation.TransferEvaluationError, "whole-stored-epoch"):
            evaluation.validate_metadata(m, 250)
        m["analysis_start_s"], m["analysis_duration_s"] = 0, 5
        m["analysis_window_samples"] = {"start": 0, "stop_exclusive": 1250, "count": 1250, "indexing": "zero_based_half_open"}
        spec = evaluation.validate_metadata(m, 250)
        self.assertFalse(spec["timing_verified"])
        self.assertIsNone(spec["cca_slice"])  # No implicit first-four-seconds selection.
        m["cca_window_samples"] = {"start": 0, "stop_exclusive": 1000, "count": 1000, "indexing": "zero_based_half_open"}
        self.assertEqual(evaluation.validate_metadata(m, 250)["cca_slice"], slice(0, 1000))
        m["cca_window_samples"]["count"] = 999
        with self.assertRaises(evaluation.TransferEvaluationError):
            evaluation.validate_metadata(m, 250)

    def test_4d_signal_shape_must_match_no_squeezing(self):
        m = metadata()
        for shape in ((2, 1250, 2), (2, 1000, 2, 1), (1250, 2, 2, 1)):
            with self.subTest(shape=shape), self.assertRaisesRegex(evaluation.TransferEvaluationError, "4D shape"):
                evaluation.canonical_epochs(np.zeros(shape), m)
        data = np.arange(5000).reshape(2, 1250, 2, 1)
        np.testing.assert_array_equal(evaluation.canonical_epochs(data, m), data)
        m["axis_order"] = ["target", "block", "channel", "time"]
        m["expected_shape"] = [2, 1, 2, 1250]
        np.testing.assert_array_equal(evaluation.canonical_epochs(data.transpose(2, 3, 0, 1), m), data)

    def test_mat_loader_preserves_singleton_block_axis_and_struct_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = metadata()
            dong_fixture(root, m)
            data = evaluation.load_dong_epochs(root / "S2.mat", m)
            self.assertEqual(data.shape, (2, 1250, 2, 1))
            savemat(root / "nested.mat", {"payload": {"eegdata": data}})
            m["array_key"] = "payload.eegdata"
            np.testing.assert_array_equal(evaluation.load_dong_epochs(root / "nested.mat", m), data)


class ProjectionCCATests(unittest.TestCase):
    def test_exact_projection_recovers_off_bin_harmonics_and_dc(self):
        f, rate, n = 10.37, 250, 999
        t = np.arange(n) / rate
        amplitudes = [3., 2., .7]
        x = 200 + sum(a * np.sin(2 * np.pi * (i + 1) * f * t + .3 * i) for i, a in enumerate(amplitudes))
        actual = evaluation.exact_frequency_amplitudes(np.stack([x, 2 * x]), f, rate)
        np.testing.assert_allclose(actual, [amplitudes, np.array(amplitudes) * 2], atol=1e-11)
        nearest = np.argmin(np.abs(np.fft.rfftfreq(n, 1 / rate) - f))
        fft_amplitude = 2 * np.abs(np.fft.rfft(x - x.mean())[nearest]) / n
        self.assertGreater(abs(fft_amplitude - 3), .1)

    def test_projection_rejects_wrong_shape_and_nyquist(self):
        for signal, hz in ((np.zeros(1000), 8.7), (np.zeros((1, 4)), 8.7), (np.zeros((1, 1000)), 50)):
            with self.assertRaises(evaluation.TransferEvaluationError):
                evaluation.exact_frequency_amplitudes(signal, hz, 250)

    def test_cca_predictions_do_not_use_labels(self):
        f = np.array([12.3, 8.7])
        t = np.arange(1000) / 250
        epochs = np.stack([np.stack([np.sin(2 * np.pi * hz * t), np.cos(2 * np.pi * hz * t)]) for hz in f])
        good = evaluation.cca_top1(epochs, np.array([0, 1]), f, 250)
        wrong = evaluation.cca_top1(epochs, np.array([1, 0]), f, 250)
        self.assertEqual(good["accuracy"], 1)
        self.assertEqual(wrong["accuracy"], 0)
        np.testing.assert_array_equal(good["scores"], wrong["scores"])
        np.testing.assert_array_equal(good["predictions"], wrong["predictions"])

    def test_cca_rejects_labels_and_shape_errors(self):
        epochs, f = np.zeros((2, 2, 1000)), np.array([8.7, 12.3])
        for labels in (np.array([[0, 1]]), np.array([0., 1.]), np.array([True, False]), np.array([-1, 1]), np.array([0, 2])):
            with self.subTest(labels=labels), self.assertRaises(evaluation.TransferEvaluationError):
                evaluation.cca_top1(epochs, labels, f, 250)
        with self.assertRaises(evaluation.TransferEvaluationError):
            evaluation.cca_top1(np.zeros((2, 1000)), np.array([0, 1]), f, 250)
        with self.assertRaises(evaluation.TransferEvaluationError):
            evaluation.cca_scores(epochs[0], np.array([[8.7, 12.3]]), 250)

    def test_rank_deficiency_does_not_add_arbitrary_qr_directions(self):
        scores = evaluation.cca_scores(np.zeros((2, 1000)), np.array([8.7, 12.3]), 250)
        np.testing.assert_array_equal(scores, [0., 0.])
        t = np.arange(1000) / 250
        x = np.sin(2 * np.pi * 8.7 * t)
        one = evaluation.cca_scores(x[None, :], np.array([8.7, 12.3]), 250)
        repeated = evaluation.cca_scores(np.stack([x, x, 2 * x]), np.array([8.7, 12.3]), 250)
        np.testing.assert_allclose(one, repeated, atol=1e-12)

    def test_undefined_amplitude_ratios_are_visible_not_selectively_hidden(self):
        summary = evaluation.amplitude_summary(np.zeros((2, 3)), np.ones((2, 3)), np.ones((2, 3)))
        ratio = summary["harmonics"]["fundamental"]["gated_to_raw_ratio"]
        self.assertEqual(ratio["undefined_count"], 2)
        self.assertIsNone(ratio["mean"])
        checks = evaluation.preservation_checks(summary, {"raw": 1., "gated": 1.}, .6)
        self.assertFalse(checks["passed"])


class ReportAndRegressionTests(unittest.TestCase):
    def test_pending_mapping_reports_agreement_not_fake_accuracy_or_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, b = Path(tmp), bundle()
            path = dong_fixture(root, metadata())
            before = {p.name: evaluation.sha256(p) for p in root.iterdir()}
            with patch.object(evaluation.v2, "apply_model_to_array", side_effect=fake_infer) as apply:
                report = evaluation.evaluate_dong(b, root, path, list(evaluation.SUBJECTS))
            self.assertEqual(apply.call_count, 4)
            self.assertTrue(all(call.args[1].shape == (2, 1000) for call in apply.call_args_list))
            self.assertFalse(report["candidate_mapping_used_as_labels"])
            self.assertFalse(report["target_axis_mapping_confirmed"])
            self.assertEqual(report["cca_accuracy_status"], "not_assessed")
            self.assertIsNone(report["cca_top1_accuracy"])
            self.assertIsNone(report["target_labelled_amplitude"])
            self.assertEqual(report["preservation_checks"]["status"], "not_assessed")
            self.assertIsNone(report["preservation_checks"]["passed"])
            self.assertEqual(report["cca_raw_prediction_agreement"], {"candidate": 1., "gated": 1.})
            self.assertEqual(len(report["epoch_reports"]), 4)
            self.assertEqual(report["frequency_set_spectral_preservation"]["harmonics"]["h1"]["raw_amplitude"]["count"], 16)
            self.assertEqual(report["epoch_reports"][0]["frequency_set_amplitudes_by_channel_frequency_h1_h2_h3"]["raw"].shape, (2, 2, 3))
            self.assertEqual(before, {p.name: evaluation.sha256(p) for p in root.iterdir()})

    def test_confirmed_mapping_enables_author_order_accuracy_not_production(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = dong_fixture(root, metadata(True))
            with patch.object(evaluation.v2, "apply_model_to_array", side_effect=fake_infer):
                report = evaluation.evaluate_dong(bundle(), root, path, list(evaluation.SUBJECTS))
            self.assertEqual(report["cca_top1_accuracy"], {"raw": 1., "candidate": 1., "gated": 1.})
            self.assertEqual(report["epoch_reports"][0]["target_hz"], 12.3)
            self.assertTrue(report["preservation_checks"]["passed"])
            self.assertFalse(report["production_validated"])
            self.assertFalse(report["paired_denoising_efficacy_demonstrated"])
            self.assertFalse(report["downstream_stimulus_recognition_acceptance"])
            self.assertGreater(report["raw_change_cleanish_relative_ac_rms"]["candidate"]["mean"], 0)
            self.assertEqual(report["raw_change_cleanish_relative_ac_rms"]["gated"]["mean"], 0)

    def test_whole_epoch_fallback_timing_not_verified_no_implicit_cca(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, m = Path(tmp), metadata()
            m["metadata_status"], m["trial_structure"] = "timing_not_verified", {}
            m["analysis_start_s"], m["analysis_duration_s"] = 0, 5
            m["analysis_window_samples"] = {"start": 0, "stop_exclusive": 1250, "count": 1250, "indexing": "zero_based_half_open"}
            path = dong_fixture(root, m)
            with patch.object(evaluation.v2, "apply_model_to_array", side_effect=fake_infer):
                report = evaluation.evaluate_dong(bundle(), root, path, list(evaluation.SUBJECTS))
            self.assertEqual(report["timing_status"], "timing_not_verified")
            self.assertIn("whole_stored_epoch", report["analysis_scope"])
            self.assertIsNone(report["cca_raw_prediction_agreement"])
            self.assertEqual(report["preservation_checks"]["status"], "not_assessed")

    def test_fixed_subjects_cannot_be_score_selected(self):
        for subjects in (["S2.mat"], ["S1.mat", "S3.mat"], ["S3.mat", "S2.mat"]):
            with self.subTest(subjects=subjects), self.assertRaisesRegex(evaluation.TransferEvaluationError, "fixed transfer"):
                evaluation.evaluate_dong(bundle(), Path("unused"), Path("unused"), subjects)

    def test_regression_exact_26_windows_not_heldout_no_tuning(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, b = Path(tmp) / "v1.npz", bundle()
            regression_fixture(path)
            before, frozen = evaluation.sha256(path), deepcopy(vars(b))
            with patch.object(evaluation.v2, "apply_model_to_array", side_effect=fake_infer) as apply:
                report = evaluation.evaluate_regression(b, path, evaluation.REGRESSION_SESSION)
            self.assertEqual(apply.call_count, 26)
            self.assertEqual(report["diagnostic_name"], "known_v1_regression_stress_test")
            self.assertFalse(report["independent"])
            self.assertFalse(report["heldout"])
            self.assertFalse(report["test_tuned"])
            self.assertFalse(report["acceptance_passed"])
            self.assertTrue(report["seen_by_v1_experiments"])
            np.testing.assert_array_equal(report["window_indices"], np.arange(1, 27))
            self.assertEqual(set(report["summary"]), {"noisy", "candidate", "gated"})
            self.assertEqual(report["replacement_fraction_distribution"]["mean"], .25)
            self.assertEqual(report["cleanish_replacement_fraction_distribution"]["mean"], 0)
            for name in ("candidate", "gated"):
                self.assertGreater(report["summary"][name]["maximum_absolute_change_original_units"], 0)
            self.assertEqual(vars(b), frozen)
            self.assertEqual(evaluation.sha256(path), before)

    def test_regression_rejects_wrong_session_count_sample_rate_and_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v1.npz"
            regression_fixture(path)
            with self.assertRaises(evaluation.TransferEvaluationError):
                evaluation.load_regression(path, "independent", 250)
            regression_fixture(path, count=25)
            with self.assertRaisesRegex(evaluation.TransferEvaluationError, "26"):
                evaluation.load_regression(path, evaluation.REGRESSION_SESSION, 250)
            regression_fixture(path, rate=1000)
            with self.assertRaisesRegex(evaluation.TransferEvaluationError, "sample_rate"):
                evaluation.load_regression(path, evaluation.REGRESSION_SESSION, 250)
            regression_fixture(path)
            with np.load(path) as archive:
                arrays = {k: archive[k] for k in archive.files}
            arrays["clean"] = arrays["clean"][:, :, :-1]
            np.savez(path, **arrays)
            with self.assertRaisesRegex(evaluation.TransferEvaluationError, "shapes"):
                evaluation.load_regression(path, evaluation.REGRESSION_SESSION, 250)

    def test_inference_api_shapes_support_and_gating_are_checked(self):
        raw, b = np.ones((2, 1000)), bundle()
        for field, replacement in (("denoised", np.ones((1, 1000))),
                                   ("artifact_probability", np.ones(1000)),
                                   ("valid_mask", np.zeros_like(raw, dtype=bool)),
                                   ("valid_mask", np.ones_like(raw, dtype=np.uint8)),
                                   ("rejected_mask", np.ones_like(raw, dtype=bool)),
                                   ("denoised", np.zeros_like(raw))):
            good = fake_infer(b, raw)
            bad = DenoiseResult(**{**vars(good), field: replacement})
            with self.subTest(field=field), patch.object(evaluation.v2, "apply_model_to_array", return_value=bad), self.assertRaises(evaluation.TransferEvaluationError):
                evaluation.infer(b, raw)
        b.config["hop_samples"] = 250
        with self.assertRaisesRegex(evaluation.TransferEvaluationError, "windows500 hop500"):
            evaluation.infer(b, raw)

    def test_clean_change_uses_raw_ac_rms_and_original_units(self):
        raw = np.array([[1., 3., 1., 3.]])
        metrics = evaluation.change_metrics(raw, raw + 2, np.ones_like(raw, dtype=bool))
        self.assertEqual(metrics["change_rmse_original_units"], 2)
        self.assertEqual(metrics["maximum_absolute_change_original_units"], 2)
        self.assertEqual(metrics["change_rmse_relative_raw_ac_rms"], 2)

    def test_thresholds_predeclared_and_no_threshold_or_training_cli(self):
        self.assertEqual(evaluation.PRESERVATION_THRESHOLDS, {
            "gated_mean_target_amplitude_ratio_min": .95, "gated_mean_target_amplitude_ratio_max": 1.05,
            "gated_cca_accuracy_drop_max": .025})
        options = evaluation.build_parser()._option_string_actions
        for forbidden in ("--threshold", "--train", "--calibrate", "--overwrite", "--select-best-subjects"):
            self.assertNotIn(forbidden, options)
        tree = ast.parse(Path(evaluation.__file__).read_text(encoding="utf-8-sig"))
        attributes = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
        for forbidden in ("backward", "fit", "train", "save", "savez", "savez_compressed"):
            self.assertNotIn(forbidden, attributes)
        with patch.object(evaluation.v2, "apply_model_to_array", side_effect=fake_infer):
            b = bundle()
            b.validation_threshold = None
            raw = np.ones((2, 1000, 3))
            checks = evaluation.preservation_checks(evaluation.amplitude_summary(raw, raw, raw), {"raw": 1., "gated": 1.}, None)
            self.assertFalse(checks["passed"])

    def test_predeclared_gate_inclusive_boundaries_without_score_tuning(self):
        raw = np.ones((2, 3))
        for scale, expected in ((.95, True), (1.05, True), (.94999, False), (1.05001, False)):
            amplitude = evaluation.amplitude_summary(raw, raw, raw * scale)
            checks = evaluation.preservation_checks(amplitude, {"raw": .975, "gated": .95}, .6)
            self.assertEqual(checks["passed"], expected)
        amplitude = evaluation.amplitude_summary(raw, raw, raw)
        self.assertFalse(evaluation.preservation_checks(amplitude, {"raw": .975, "gated": .9499}, .6)["passed"])

    def test_coverage_counts_channels_windows_and_tail_without_filtering(self):
        raw = np.ones((2, 1250))
        result = fake_infer(bundle(), raw)
        coverage = evaluation.coverage_metrics(result)
        self.assertEqual(coverage["channel_full_window_count"], 4)
        self.assertEqual(coverage["replacement_channel_window_count"], 1)
        self.assertEqual(coverage["replacement_channel_window_fraction"], .25)
        self.assertEqual(coverage["replacement_fraction"], .2)
        self.assertEqual(coverage["tail_samples_per_channel"], 250)

    def test_report_exclusive_creation_json_no_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            evaluation.write_report(output, {"undefined": np.array([np.nan]), "pass": False})
            before = output.read_bytes()
            self.assertEqual(json.loads(before)["undefined"], [None])
            with self.assertRaises(FileExistsError):
                evaluation.write_report(output, {"pass": True})
            self.assertEqual(output.read_bytes(), before)

    def test_threads_default_and_invalid_values_rejected_without_model_loading(self):
        parser = evaluation.build_parser()
        args = parser.parse_args(["--model", "model", "--config", "config", "--output", "output"])
        self.assertEqual(args.threads, 2)
        for value in ("0", "-2", "1.5", "NaN", "invalid"):
            with self.subTest(value=value), patch.object(evaluation.v2, "load_model") as load, patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                evaluation.main(["--model", "model", "--config", "config", "--output", "output", "--threads", value])
            load.assert_not_called()
        for value in ("1", "2", "8"):
            self.assertEqual(evaluation.positive_threads(value), int(value))

    def test_cli_both_sections_and_each_mode_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = dong_fixture(root, metadata())
            model, config, dataset = root / "model.pt", root / "config.json", root / "v1.npz"
            model.write_bytes(b"fixture model not loaded")
            config.write_text("{}", encoding="utf-8")
            regression_fixture(dataset)
            base = ["--model", str(model), "--config", str(config)]
            dong = ["--dong-dir", str(root), "--metadata-json", str(meta), "--subjects", "S2.mat", "S3.mat"]
            regression = ["--regression-dataset", str(dataset), "--regression-session-id", evaluation.REGRESSION_SESSION]
            snapshots = {p.name: evaluation.sha256(p) for p in root.iterdir()}
            for name, flags, expected in (("both", dong + regression, {"dong2023_transfer_preservation", "known_v1_regression_stress_test"}),
                                          ("dong", dong, {"dong2023_transfer_preservation"}),
                                          ("regression", regression, {"known_v1_regression_stress_test"})):
                output = root / f"{name}.json"
                with patch.object(evaluation.v2, "load_config", return_value=bundle().config), patch.object(evaluation.v2, "load_model", return_value=bundle()), patch.object(evaluation.v2, "apply_model_to_array", side_effect=fake_infer), patch.object(evaluation.v2.torch, "set_num_threads") as set_threads, patch("sys.stdout", new=io.StringIO()):
                    thread_flags = ["--threads", "3"] if name == "regression" else []
                    self.assertEqual(evaluation.main(base + flags + thread_flags + ["--output", str(output)]), 0)
                    set_threads.assert_called_once_with(3 if name == "regression" else 2)
                report = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(set(report["sections"]), expected)
                self.assertFalse(report["test_tuned"])
                self.assertFalse(report["release_gate_passed"])
                self.assertEqual(report["frozen_validation_threshold"], .6)
                self.assertEqual(report["execution"]["torch_cpu_threads"], 3 if name == "regression" else 2)
            for name, digest in snapshots.items():
                self.assertEqual(evaluation.sha256(root / name), digest)

    def test_cli_failures_before_model_inference_or_any_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output.json"
            base = ["--model", str(root / "model"), "--config", str(root / "config"), "--output", str(output)]
            for flags in ([], ["--dong-dir", str(root)], ["--metadata-json", str(root / "meta")]):
                with patch.object(evaluation.v2, "load_model") as load, patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                    evaluation.main(base + flags)
                load.assert_not_called()
                self.assertFalse(output.exists())
            output.write_text("old report", encoding="utf-8")
            with patch.object(evaluation.v2, "load_model") as load, patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                evaluation.main(base + ["--regression-dataset", str(root / "v1.npz")])
            load.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "old report")


if __name__ == "__main__":
    unittest.main()
