"""Synthetic/mock v3 evaluation/CLI fixtures; never reads real S4/S5 or old test metrics."""
from __future__ import annotations
from tests.synthetic_research_fixtures import research_contract

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from scripts import evaluate_domain_gate_v3 as evaluation
from scripts import run_learned_denoise_v3 as offline

ROOT = Path(__file__).resolve().parents[1]


def fixture_result(raw, valid=None, accepted=True, artifact=True, candidate=None):
    valid = np.isfinite(raw) if valid is None else valid.copy()
    accepted = np.broadcast_to(np.asarray(accepted, dtype=bool), raw.shape) & valid
    replacement = accepted & np.broadcast_to(np.asarray(artifact, dtype=bool), raw.shape)
    candidate = raw.copy() if candidate is None else candidate.copy()
    return types.SimpleNamespace(
        denoised=np.where(replacement, candidate, raw), model_denoised=candidate,
        artifact_probability=np.full(raw.shape, 0.9 if artifact is True else 0.1),
        confidence=np.full(raw.shape, 0.1 if artifact is True else 0.9),
        replacement_mask=replacement, valid_mask=valid, rejected_mask=~accepted,
        support_probability=np.where(accepted, 0.8, 0.2), domain_rejected_mask=~accepted,
        diagnostics={"synthetic_fixture": True})


def arrays(result, shape):
    return {field: getattr(result, field).reshape(shape) for field in offline.RESULT_FIELDS}


def passing_sections():
    benchmark = {"summaries": {
        "corrupted": {"domain_accepted_fraction": 0.8, "gated_rmse_improvement_fraction": 0.4,
                      "actual_replacement_windows": 10, "actually_changed_windows": 10},
        "clean": {"replacement_window_fraction": 0.005}}}
    subjects = {subject: {"false_acceptance_fraction": 0.005, "single_channel_500_window_count": 2560,
                          "domain_scored_window_count": 2560, "mother_trial_count": 160,
                          "domain_refused_raw_fallback_exact_original_units": True}
                for subject in evaluation.SUBJECTS}
    negative = {"false_acceptance_fraction": 0.005, "single_channel_500_window_count": 5120,
                "domain_scored_window_count": 5120, "subjects": subjects}
    known = {"gated_to_raw_RMSE_ratio": 1.0}
    return benchmark, negative, known


class SummaryTests(unittest.TestCase):
    def test_all_predeclared_combined_gates_research_only(self):
        gate = evaluation.research_gate(*passing_sections())
        self.assertTrue(gate["domain_gate_research_acceptance_passed"])
        self.assertFalse(gate["production_validated"])
        self.assertFalse(gate["release_gate_passed"])
        self.assertFalse(gate["automatic_enabled"])
        self.assertEqual(evaluation.THRESHOLDS, gate["thresholds_predeclared"])

    def test_all_noop_and_cosmetic_zero_residual_cannot_pass(self):
        for field in ("actual_replacement_windows", "actually_changed_windows"):
            benchmark, negative, known = passing_sections()
            benchmark["summaries"]["corrupted"][field] = 0
            with self.subTest(field=field):
                self.assertFalse(evaluation.research_gate(benchmark, negative, known)["domain_gate_research_acceptance_passed"])
        benchmark, negative, known = passing_sections()
        benchmark["summaries"]["corrupted"]["gated_rmse_improvement_fraction"] = 0
        self.assertFalse(evaluation.research_gate(benchmark, negative, known)["domain_gate_research_acceptance_passed"])

    def test_individual_subject_failure_hidden_by_pooled_mean_is_rejected(self):
        benchmark, negative, known = passing_sections()
        negative["subjects"]["S4.mat"]["false_acceptance_fraction"] = 0.02
        self.assertFalse(evaluation.research_gate(benchmark, negative, known)["domain_gate_research_acceptance_passed"])

    def test_excluded_or_unscored_windows_are_not_successful_rejection(self):
        for count in (0, 5119):
            benchmark, negative, known = passing_sections()
            negative["domain_scored_window_count"] = count
            self.assertFalse(evaluation.research_gate(benchmark, negative, known)["domain_gate_research_acceptance_passed"])

    def test_acceptance_not_artifact_replacement_and_mother_counter_is_explicit(self):
        raw = np.ones((4, 1, 500), dtype=np.float64)
        accepted = np.asarray([True, False, False, False])[:, None]
        result = fixture_result(raw[:, 0], accepted=accepted, artifact=False)
        provenance = [{"subject": "S4.mat" if i < 2 else "S5.mat", "stored_target_axis_index": 0,
                       "block_axis_index": 0, "channel_index_zero_based": i % 2, "window_index_in_trial": 0}
                      for i in range(4)]
        report = evaluation.summarize_unsupported(raw, arrays(result, raw.shape), provenance)
        self.assertEqual(1, report["false_accepted_windows"])
        self.assertEqual(0, report["actual_replacement_windows"])
        self.assertEqual(0.5, report["subjects"]["S4.mat"]["false_acceptance_fraction"])
        self.assertEqual(1, report["subjects"]["S4.mat"]["mother_trials_with_any_domain_acceptance"])
        self.assertEqual(0, report["subjects"]["S4.mat"]["gated_change_RMSE_original_units"])
        self.assertFalse(report["clean_ground_truth_available"])
        self.assertNotIn("SNR_gain_db", report)
        self.assertNotIn("accuracy", report)
        self.assertEqual("not_assessed_target_mapping", report["accuracy_status"])

    def test_original_units_domain_fallback_and_strict_threshold_are_enforced(self):
        raw = (1e6 + np.sin(np.arange(500) / 10))[None, :]
        result = fixture_result(raw, accepted=False, candidate=raw + 100)
        offline.validate_result(result, raw, np.ones_like(raw, dtype=bool), 0.5)
        result.denoised[0, 0] += 0.01
        with self.assertRaises(evaluation.EvaluationError):
            offline.validate_result(result, raw, np.ones_like(raw, dtype=bool), 0.5)
        result = fixture_result(raw, accepted=True, artifact=False)
        result.support_probability.fill(0.5)
        with self.assertRaisesRegex(evaluation.EvaluationError, "strict"):
            offline.validate_result(result, raw, np.ones_like(raw, dtype=bool), 0.5)

    def test_paired_metrics_on_actual_gated_output_and_window_vs_mother_units(self):
        t = np.arange(1000) / 250
        clean = np.tile(np.sqrt(2) * np.sin(2 * np.pi * 10 * t), (2, 2, 1))
        raw = clean + np.sin(2 * np.pi * 2 * t)[None, None, :]
        candidate = clean + (raw - clean) * 0.5
        result = fixture_result(raw.reshape(4, 1000), candidate=candidate.reshape(4, 1000))
        summary = evaluation.paired_summary(clean, raw, arrays(result, raw.shape))
        self.assertEqual(2, summary["mother_epoch_count"])
        self.assertEqual(8, summary["single_channel_500_window_count"])
        self.assertEqual(8, summary["domain_accepted_windows"])
        self.assertAlmostEqual(0.5, summary["gated_rmse_improvement_fraction"])
        self.assertLess(summary["metrics"]["gated"]["RMSE"], summary["metrics"]["raw"]["RMSE"])


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v3 freeze fixture ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prereg = research_contract('v3-preregistration.json')
        self.prereg_path = self.root / "analysis_outputs" / "learned_denoiser_v3_preregistration.json"
        self.prereg_path.parent.mkdir()
        self.protected = self.root / "protected_fixture.txt"
        self.protected.write_text("synthetic fixture only")
        self.prereg["protected_sha256"] = {"protected_fixture.txt": evaluation.sha256(self.protected)}
        self.prereg_path.write_text(json.dumps(self.prereg))
        names = ["models/domain_gate_v3.pt", "models/denoiser_v2.pt", "configs/learned_denoiser_v3.json",
                 "scripts/evaluate_domain_gate_v3.py", "eeg_tools/learned_denoise_v3.py",
                 "eeg_tools/learned_domain_dataset_v3.py", "scripts/train_domain_gate_v3.py"]
        self.paths = []
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture source; not real weights")
            self.paths.append(path)
        self.checkpoint, self.denoiser, self.config = self.paths[:3]
        self.lock_path = self.root / "analysis_outputs" / "learned_denoiser_v3_experiment_lock.json"
        self.lock = {"status": "frozen_before_evaluation", "test_access_before_freeze": False,
                     "items": [{"path": str(p), "sha256": evaluation.sha256(p)} for p in [self.prereg_path, *self.paths]]}
        self.write_lock()

    def write_lock(self):
        self.lock_path.write_text(json.dumps(self.lock))

    def check(self):
        return evaluation.validate_freeze(self.root, self.lock_path, self.prereg_path, self.prereg,
                                          self.checkpoint, self.denoiser, self.config)

    def test_preregistration_and_verified_source_proof_required(self):
        self.assertEqual(evaluation.THRESHOLDS, {k:v for k,v in evaluation.load_preregistration(self.prereg_path)["predeclared_acceptance"].items() if k in evaluation.THRESHOLDS})
        frozen = self.check()
        self.assertIn(str(self.checkpoint), frozen)
        self.paths[-1].write_text("changed training source")
        with self.assertRaisesRegex(evaluation.EvaluationError, "changed"):
            self.check()

    def test_missing_lock_denies_before_loading_model_or_any_new_holdout(self):
        self.lock_path.unlink()
        with patch.object(evaluation, "load_bundle") as model, patch.object(evaluation, "evaluate_unsupported") as negative, redirect_stdout(io.StringIO()):
            code = evaluation.main(["--root", str(self.root)])
        self.assertEqual(2, code)
        model.assert_not_called()
        negative.assert_not_called()

    def test_no_pre_freeze_claim_or_missing_items_cannot_be_guessed(self):
        self.lock.pop("test_access_before_freeze")
        self.write_lock()
        with self.assertRaises(evaluation.EvaluationError):
            self.check()
        self.lock["test_access_before_freeze"] = False
        self.lock["items"] = self.lock["items"][:-1]
        self.write_lock()
        with self.assertRaises(evaluation.EvaluationError):
            self.check()

    def test_check_only_no_benchmark_regression_or_negative_read(self):
        bundle = types.SimpleNamespace(support_threshold=0.75)
        with patch.object(evaluation, "load_bundle", return_value=(types.SimpleNamespace(MODEL_VERSION="learned_denoise_v3"), bundle)), patch.object(evaluation, "evaluate_known_benchmark") as old, patch.object(evaluation, "evaluate_unsupported") as negative, patch.object(evaluation, "evaluate_known_session") as known, redirect_stdout(io.StringIO()) as stdout:
            code = evaluation.main(["--root", str(self.root), "--check-only"])
        self.assertEqual(0, code, stdout.getvalue())
        for mock in (old, negative, known): mock.assert_not_called()
        self.assertFalse(json.loads(stdout.getvalue())["holdout_loaded"])
        self.assertEqual(0.75, bundle.support_threshold)

    def test_existing_output_guard_precedes_loading_or_reading_any_data(self):
        output = self.root / "keep.json"
        output.write_text("historical report")
        with patch.object(evaluation, "load_preregistration") as contract, redirect_stdout(io.StringIO()):
            self.assertEqual(2, evaluation.main(["--root", str(self.root), "--output", str(output)]))
        contract.assert_not_called()
        self.assertEqual("historical report", output.read_text())
        with self.assertRaises(evaluation.EvaluationError): evaluation.reserve_output(output, {})

    def test_main_mock_evaluation_never_changes_threshold_or_source_hashes(self):
        benchmark, negative, known = passing_sections()
        bundle = types.SimpleNamespace(support_threshold=0.75)
        before = {str(p):evaluation.sha256(p) for p in [self.prereg_path, *self.paths]}
        output = self.root / "v3-fixture.json"
        with patch.object(evaluation, "load_bundle", return_value=(types.SimpleNamespace(MODEL_VERSION="learned_denoise_v3"), bundle)), patch.object(evaluation, "evaluate_known_benchmark", return_value=benchmark), patch.object(evaluation, "evaluate_unsupported", return_value=negative), patch.object(evaluation, "evaluate_known_session", return_value=known), redirect_stdout(io.StringIO()) as stdout:
            code = evaluation.main(["--root", str(self.root), "--output", str(output)])
        self.assertEqual(0, code, stdout.getvalue())
        report = json.loads(output.read_text())
        self.assertTrue(report["known_v2_benchmark_not_new_holdout"])
        self.assertFalse(report["known_v1_stress_independent"])
        self.assertFalse(report["threshold_changed"])
        self.assertFalse(report["production_validated"])
        self.assertEqual(0.75, bundle.support_threshold)
        self.assertEqual(before, {p:evaluation.sha256(Path(p)) for p in before})


class FreshInferenceTests(unittest.TestCase):
    def test_known_v1_stress_fresh_synthetic_inference_not_old_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.npz"
            t = np.arange(1000) / 250
            clean = np.tile(np.sin(2 * np.pi * 10 * t), (26, 2, 1))
            raw = clean + 0.1 * np.sin(2 * np.pi * 2 * t)[None, None]
            np.savez(path, noisy=raw, clean=clean, artifact_mask=np.zeros_like(raw,dtype=bool),
                     session_id=np.repeat(evaluation.REGRESSION_SESSION,26), sample_rate_hz=np.repeat(250.,26))
            bundle = types.SimpleNamespace(support_threshold=0.75)
            calls = []
            def infer(bundle, raw, valid_mask):
                calls.append(raw.shape)
                return fixture_result(raw, valid_mask, accepted=False, candidate=raw + 100)
            core = types.SimpleNamespace(apply_model_to_array=infer)
            before = evaluation.sha256(path)
            report = evaluation.evaluate_known_session(core, bundle, path, batch_size=16)
            self.assertTrue(calls)
            self.assertFalse(report["independent"])
            self.assertFalse(report["heldout"])
            self.assertTrue(report["v3_fresh_inference"])
            self.assertFalse(report["old_v2_report_used_as_v3_measurement"])
            self.assertEqual(1., report["gated_to_raw_RMSE_ratio"])
            self.assertGreater(report["summary"]["metrics"]["candidate"]["RMSE"], report["summary"]["metrics"]["raw"]["RMSE"])
            self.assertEqual(before,evaluation.sha256(path))

    def test_known_v2_benchmark_explicitly_not_new_holdout(self):
        clean = np.tile(np.sin(np.arange(500)/10),(3,1,1))
        noisy = clean.copy(); noisy[1:] += .1
        data = {"clean":clean,"noisy":noisy,"valid_mask":np.ones_like(clean,dtype=bool),"artifact_type":np.array(["clean","eog","emg"])}
        bundle = types.SimpleNamespace(support_threshold=.75,denoiser=types.SimpleNamespace(metadata={}))
        core = types.SimpleNamespace(apply_model_to_array=lambda b,x,valid_mask:fixture_result(x,valid_mask,accepted=False))
        with patch.object(evaluation.benchmark_v2,"_load_dataset",return_value=(data,{},"fixture-sha","fixture-manifest-sha")), patch.object(evaluation.benchmark_v2,"_select_evaluation_indices",return_value=(np.arange(3),{"synthetic":True})):
            report = evaluation.evaluate_known_benchmark(core,bundle,Path("fixture-only"),2)
        self.assertEqual("test",report["split"])
        self.assertTrue(report["known_v2_benchmark_not_new_holdout"])
        self.assertFalse(report["new_independent_holdout"])
        self.assertEqual(0.,report["summaries"]["corrupted"]["gated_rmse_improvement_fraction"])


class OfflineFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="v3 offline fixture ")
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.source=self.root/"signal.npz"
        self.raw=(1e6+np.sin(np.arange(500)/10))[None,:]
        np.savez(self.source,signal=self.raw,sample_rate_hz=250.,timestamp_s=np.arange(500)/250.)
        self.paths=[]
        for name in ("gate.pt","denoiser.pt","config.json"):
            path=self.root/name;path.write_text("fixture");self.paths.append(path)
        self.bundle=types.SimpleNamespace(support_threshold=.75,config={"runtime":{"enabled":False,"production_validated":False}})
        self.core=types.SimpleNamespace(MODEL_VERSION="learned_denoise_v3",apply_model_to_array=lambda b,x,valid_mask:fixture_result(x,valid_mask,accepted=False,candidate=x+100))

    def invoke(self,output):
        with patch.object(offline,"load_bundle",return_value=(self.core,self.bundle)),redirect_stdout(io.StringIO()) as stdout:
            code=offline.main(["--root",str(self.root),"--input",str(self.source),"--checkpoint",str(self.paths[0]),"--denoiser-checkpoint",str(self.paths[1]),"--config",str(self.paths[2]),"--output",str(output)])
        return code,stdout.getvalue()

    def test_original_precision_raw_fallback_source_and_threshold_immutable(self):
        before=offline.readers._snapshot(self.source)
        code,stdout=self.invoke(self.root/"diagnostic")
        self.assertEqual(0,code,stdout)
        with np.load(self.root/"diagnostic"/"denoised_signal.npz",allow_pickle=False) as result:
            np.testing.assert_array_equal(result["raw"],self.raw)
            np.testing.assert_array_equal(result["denoised"],self.raw)
            for field in offline.RESULT_FIELDS:self.assertEqual(self.raw.shape,result[field].shape)
        report=json.loads((self.root/"diagnostic"/"quality_metrics.json").read_text())
        self.assertEqual(before,offline.readers._snapshot(self.source))
        self.assertFalse(report["threshold_changed"])
        self.assertFalse(report["automatic"])
        self.assertFalse(report["release"])
        self.assertTrue(report["source_hashes_unchanged"])
        self.assertEqual(.75,self.bundle.support_threshold)

    def test_existing_output_is_refused_without_model_load(self):
        output=self.root/"diagnostic";output.mkdir()
        with patch.object(offline,"load_bundle") as model,redirect_stdout(io.StringIO()):
            code=offline.main(["--root",str(self.root),"--input",str(self.source),"--output",str(output)])
        self.assertEqual(2,code);model.assert_not_called()

    def test_continuity_and_rate_fail_before_model_inference(self):
        np.savez(self.source,signal=self.raw,sample_rate_hz=256.)
        with patch.object(offline,"load_bundle") as model,redirect_stdout(io.StringIO()):
            code=offline.main(["--root",str(self.root),"--input",str(self.source)])
        self.assertEqual(2,code);model.assert_not_called()


if __name__=="__main__":
    unittest.main()
