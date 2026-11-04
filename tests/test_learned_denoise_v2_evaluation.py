"""Synthetic fixtures only: no public benchmark training or independent test run."""
from __future__ import annotations

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

from eeg_tools.learned_dataset_v2 import DEFAULT_OUTPUT, build_dataset_v2
from eeg_tools import learned_denoise_v2 as core
from scripts import evaluate_denoiser_v2 as evaluation
from scripts import run_learned_denoise_v2 as offline


class ProvenanceFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / "sources"
        cls.source.mkdir()
        rng = np.random.default_rng(4321)
        for kind in ("EEG", "EOG", "EMG"):
            np.save(cls.source / f"{kind}_all_epochs.npy", rng.normal(size=(12, 512)))
        cls.dataset = cls.root / "fixture.npz"
        build_dataset_v2(cls.source, cls.dataset, samples=60, seed=43)
        cls.data, cls.manifest, cls.digest, cls.manifest_digest = evaluation._load_dataset(cls.dataset)
        cls.config = core.load_config()
        cls.checkpoint = {
            "checkpoint_schema_version": 1, "model_version": core.MODEL_VERSION,
            "config": cls.config, "architecture": cls.config["model"],
            "dataset_sha256": cls.digest, "dataset_manifest_sha256": cls.manifest_digest,
            "dataset_manifest": cls.manifest,
            "train_indices": np.flatnonzero(cls.data["split"] == "train").tolist(),
            "validation_indices": np.flatnonzero(cls.data["split"] == "validation").tolist(),
            "calibrated_threshold": 0.8, "validation_threshold": 0.8,
            "threshold_calibration": {
                "scope": "validation_only", "comparison": ">", "method": "clean_score_order_statistic",
                "score_reduction": "per_channel_window_mean",
                "max_clean_false_replacement_fraction": 0.01, "empirical_only": True,
                "production_calibrated": False,
                "clean_window_count": int(np.count_nonzero((cls.data["split"] == "validation") & (cls.data["artifact_type"] == "clean"))),
                "observed_clean_false_replacement_fraction": 0.0,
            },
        }
        cls.model = cls.root / "fixture.pt"
        cls.model.write_bytes(b"synthetic model placeholder; load_model always mocked")
        cls.config_path = cls.root / "fixture-config.json"
        cls.config_path.write_text(json.dumps(cls.config), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def select(self, checkpoint=None, data=None, manifest=None, digest=None, manifest_digest=None):
        return evaluation._select_evaluation_indices(
            checkpoint if checkpoint is not None else self.checkpoint,
            data if data is not None else self.data,
            manifest if manifest is not None else self.manifest,
            dataset_sha256=digest or self.digest, manifest_sha256=manifest_digest or self.manifest_digest)

    def test_defaults_match_builder_and_predeclared_thresholds(self):
        args = evaluation.build_parser().parse_args([])
        self.assertEqual(DEFAULT_OUTPUT, evaluation.ROOT / args.dataset)
        self.assertEqual(0.1, evaluation.RESEARCH_THRESHOLDS["all_test_gated_rmse_improvement_min"])
        self.assertEqual(1.0, evaluation.RESEARCH_THRESHOLDS["corrupted_test_gated_snr_gain_db_min"])
        self.assertEqual(0.02, evaluation.RESEARCH_THRESHOLDS["clean_gated_change_rmse_ac_rms_units_max"])
        self.assertFalse(evaluation.BASELINE_PARAMETERS["test_tuned"])
        self.assertEqual(2, args.threads)
        self.assertEqual(2, offline.build_parser().parse_args([]).threads)
        self.assertEqual(0.0, self.config["inference"]["max_invalid_fraction"])

    def test_fixed_test_excludes_all_train_validation_indices(self):
        indices, audit = self.select()
        np.testing.assert_array_equal(indices, np.flatnonzero(self.data["split"] == "test"))
        self.assertFalse(set(indices) & set(self.checkpoint["validation_indices"]))
        self.assertFalse(set(indices) & set(self.checkpoint["train_indices"]))
        self.assertTrue(audit["test_pre_fixed"])
        self.assertFalse(audit["subject_disjoint_claim"])
        self.assertFalse(audit["validation_used_as_test"])

    def test_missing_or_wrong_npz_manifest_hash_fails_closed(self):
        for field in ("dataset_sha256", "dataset_manifest_sha256", "dataset_manifest"):
            checkpoint = deepcopy(self.checkpoint)
            checkpoint.pop(field)
            with self.subTest(field=field), self.assertRaises(evaluation.EvaluationError):
                self.select(checkpoint=checkpoint)
        with self.assertRaises(evaluation.EvaluationError):
            self.select(digest="0" * 64)
        with self.assertRaises(evaluation.EvaluationError):
            self.select(manifest_digest="0" * 64)

    def test_checkpoint_full_indices_cannot_use_validation_as_test(self):
        for field in ("train_indices", "validation_indices"):
            checkpoint = deepcopy(self.checkpoint)
            checkpoint[field][0] = int(np.flatnonzero(self.data["split"] == "test")[0])
            with self.subTest(field=field), self.assertRaises(evaluation.EvaluationError):
                self.select(checkpoint=checkpoint)
        checkpoint = deepcopy(self.checkpoint)
        checkpoint["selected_train_indices"] = checkpoint["validation_indices"][:1]
        with self.assertRaises(evaluation.EvaluationError):
            self.select(checkpoint=checkpoint)

    def test_source_epoch_id_cannot_cross_split(self):
        data = {key: value.copy() for key, value in self.data.items()}
        test = int(np.flatnonzero(data["split"] == "test")[0])
        train = int(np.flatnonzero(data["split"] == "train")[0])
        data["clean_epoch_id"][test] = data["clean_epoch_id"][train]
        with self.assertRaisesRegex(evaluation.EvaluationError, "isolated"):
            self.select(data=data)

    def test_global_fingerprint_and_source_pool_duplicates_rejected(self):
        for mutation in ("fingerprint", "epoch", "pool"):
            manifest = deepcopy(self.manifest)
            audit = manifest["epoch_split_manifest"]
            if mutation == "fingerprint":
                audit["fingerprint_groups"].append(deepcopy(audit["fingerprint_groups"][0]))
                audit["fingerprint_groups"][-1]["split"] = "test"
            elif mutation == "epoch":
                group = next(g for g in audit["fingerprint_groups"] if "eeg" in g["epoch_ids_by_source"])
                group["epoch_ids_by_source"]["eeg"].append(group["epoch_ids_by_source"]["eeg"][0])
            else:
                audit["source_statistics"]["eeg"]["raw_epoch_ids_by_split"]["test"].append(10000)
            checkpoint = deepcopy(self.checkpoint)
            checkpoint["dataset_manifest"] = manifest
            with self.subTest(mutation=mutation), self.assertRaises(evaluation.EvaluationError):
                self.select(checkpoint=checkpoint, manifest=manifest)

    def test_calibration_scope_alias_and_counts_cannot_be_guessed(self):
        for key, value in (("scope", "test"), ("comparison", ">="), ("method", "test_grid_search"),
                           ("production_calibrated", True), ("clean_window_count", 0),
                           ("observed_clean_false_replacement_fraction", 0.1)):
            checkpoint = deepcopy(self.checkpoint)
            checkpoint["threshold_calibration"][key] = value
            with self.subTest(key=key), self.assertRaises(evaluation.EvaluationError):
                self.select(checkpoint=checkpoint)
        checkpoint = deepcopy(self.checkpoint)
        checkpoint["validation_threshold"] = 0.1
        with self.assertRaises(evaluation.EvaluationError):
            self.select(checkpoint=checkpoint)

    def test_manifest_and_unused_noise_ids_are_enforced(self):
        data = {key: value.copy() for key, value in self.data.items()}
        data["eog_epoch_id"][data["artifact_type"] != "eog"] = 0
        with self.assertRaises(evaluation.EvaluationError):
            self.select(data=data)
        data = {key: value.copy() for key, value in self.data.items()}
        data["clean"] *= 10
        data["noisy"] *= 10
        with self.assertRaisesRegex(evaluation.EvaluationError, "RMS"):
            self.select(data=data)

    def test_existing_output_guard_is_before_model_load_and_test(self):
        output = self.root / "existing.json"
        output.write_text("historical result", encoding="utf-8")
        with patch.object(core, "load_model") as load, redirect_stdout(io.StringIO()):
            code = evaluation.main(["--output", str(output)])
        self.assertEqual(2, code)
        load.assert_not_called()
        self.assertEqual("historical result", output.read_text(encoding="utf-8"))
        with self.assertRaises(evaluation.EvaluationError):
            evaluation._reserve_report(output)
        evaluation._reserve_report(output, overwrite=True)
        self.assertEqual("reserved_no_test_result_yet", json.loads(output.read_text())["status"])

    def test_check_only_does_not_infer_or_replace_existing_report(self):
        output = self.root / "checked-report.json"
        output.write_text("keep", encoding="utf-8")
        bundle = types.SimpleNamespace(metadata=self.checkpoint, calibrated_threshold=0.8)
        with patch.object(core, "load_model", return_value=bundle), patch.object(core, "apply_model_to_array") as infer, redirect_stdout(io.StringIO()) as stdout:
            code = evaluation.main(["--dataset", str(self.dataset), "--model", str(self.model),
                                    "--config", str(self.config_path), "--output", str(output), "--check-only"])
        self.assertEqual(0, code, stdout.getvalue())
        infer.assert_not_called()
        self.assertEqual("keep", output.read_text())
        self.assertFalse(json.loads(stdout.getvalue())["test_inference_executed"])

    def test_fixture_evaluation_batches_only_test_and_keeps_all_methods(self):
        output = self.root / "fixture-report.json"
        bundle = types.SimpleNamespace(metadata=self.checkpoint, calibrated_threshold=0.8)
        calls = []
        def infer(bundle, x, valid_mask):
            calls.append(x.copy())
            return noop_result(x, valid_mask)
        before = evaluation.sha256(self.dataset)
        with patch.object(core, "load_model", return_value=bundle), patch.object(core, "apply_model_to_array", side_effect=infer), redirect_stdout(io.StringIO()) as stdout:
            code = evaluation.main(["--dataset", str(self.dataset), "--model", str(self.model),
                                    "--config", str(self.config_path), "--output", str(output), "--batch-size", "4"])
        self.assertEqual(0, code, stdout.getvalue())
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual({"candidate", "gated", "noop", "bandpass"}, set(report["metrics"]))
        np.testing.assert_array_equal(np.concatenate(calls), self.data["noisy"][self.data["split"] == "test", 0])
        self.assertGreater(len(calls), 1)
        self.assertEqual(before, evaluation.sha256(self.dataset))
        self.assertFalse(report["semi_synthetic_research_gate_passed"])
        self.assertFalse(report["release_gate_passed"])
        self.assertFalse(report["gate"]["checks"]["non_noop_corrupted"]["passed"])


def noop_result(raw, valid):
    return core.DenoiseResult(raw.copy(), raw.copy(), np.zeros_like(raw), np.ones_like(raw),
                             np.zeros_like(valid), valid.copy(), ~valid.copy(), {"fixture": True})


class MetricTests(unittest.TestCase):
    def fixture(self):
        t = np.arange(500) / 250
        clean = np.tile(np.sqrt(2) * np.sin(2 * np.pi * 10 * t), (3, 1, 1))
        noise = np.tile(0.8 * np.sin(2 * np.pi * 2 * t), (3, 1, 1))
        noise[0] = 0
        noisy = clean + noise
        valid = np.ones_like(clean, dtype=bool)
        artifact = valid.copy()
        artifact[0] = False
        labels = np.array(["clean", "eog", "emg"])
        return clean, noisy, valid, artifact, labels

    def test_report_subgroups_segments_candidate_and_gated_on_same_scale(self):
        clean, noisy, valid, artifact, labels = self.fixture()
        gated = clean + 0.5 * (noisy - clean)
        candidate = gated.copy()
        candidate[0] = clean[0] * 2  # deliberate scale bug must remain visible
        metrics = evaluation.summarize_methods(clean, noisy, valid, artifact, labels, candidate, gated, artifact)
        self.assertEqual({"all", "corrupted", "clean", "eog", "emg"}, set(metrics["gated"]))
        self.assertAlmostEqual(6.0205999, metrics["gated"]["corrupted"]["SNR_gain_db"], places=5)
        self.assertAlmostEqual(1.0, metrics["candidate"]["clean"]["relative_RMSE"], places=5)
        self.assertEqual(0, metrics["gated"]["clean"]["clean_false_replacement_fraction"])
        self.assertEqual(1000, metrics["gated"]["corrupted"]["segments"]["artifact"]["sample_count"])
        self.assertEqual(0.0, metrics["gated"]["clean"]["clean_signal_distortion_RMSE"])
        gate = evaluation.research_gate(metrics)
        self.assertTrue(gate["semi_synthetic_research_gate_passed"])
        self.assertFalse(gate["release_gate_passed"])
        self.assertFalse(gate["production_validated"])

    def test_all_noop_and_empty_required_group_cannot_pass(self):
        clean, noisy, valid, artifact, labels = self.fixture()
        metrics = evaluation.summarize_methods(clean, noisy, valid, artifact, labels, noisy, noisy, np.zeros_like(valid))
        self.assertFalse(evaluation.research_gate(metrics)["semi_synthetic_research_gate_passed"])
        altered = deepcopy(metrics)
        altered["gated"]["clean"]["valid_sample_count"] = 0
        self.assertFalse(evaluation.research_gate(altered)["checks"]["clean_distortion"]["passed"])

    def test_nonfinite_prediction_cannot_hide_difficult_valid_samples(self):
        clean, noisy, valid, artifact, labels = self.fixture()
        prediction = noisy.copy()
        prediction[1, 0, 0] = np.nan
        with self.assertRaises(evaluation.EvaluationError):
            evaluation.summarize_methods(clean, noisy, valid, artifact, labels, prediction, noisy, artifact)

    def test_segment_metrics_and_clean_window_rate_not_sample_rate(self):
        clean, noisy, valid, artifact, labels = self.fixture()
        replacement = np.zeros_like(valid)
        replacement[0, 0, 0] = True
        metrics = evaluation.summarize_methods(clean, noisy, valid, artifact, labels, noisy, noisy, replacement)
        self.assertEqual(1.0, metrics["gated"]["clean"]["replacement_window_fraction"])
        self.assertEqual(0.002, metrics["gated"]["clean"]["replacement_fraction"])
        self.assertFalse(evaluation.research_gate(metrics)["checks"]["clean_replacement_windows"]["passed"])


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.signal = np.tile(np.sin(np.arange(500) / 10), (2, 1)).astype(np.float32)
        self.npz = self.root / "input.npz"
        self.model = self.root / "model.pt"
        self.model.write_bytes(b"fixture")
        self.config = self.root / "config.json"
        self.config.write_text("{}")
        self.write_npz()

    def write_npz(self, rate=250, **extras):
        np.savez(self.npz, signal=self.signal, sample_rate_hz=rate,
                 timestamp_s=np.arange(500) / 250, **extras)

    def session(self):
        source = self.root / "session"
        source.mkdir()
        (source / "session.json").write_text(json.dumps({"sampling_rate_hz": 250}))
        (source / "quality_metrics.json").write_text('{"status":"historical_not_passed"}')
        lines = ["sample_index\tpackage_num\ttimestamp_s\teeg_ch1\teeg_ch2"]
        lines += [f"{i}\t{i % 256}\t{i / 250}\t{self.signal[0,i]}\t{self.signal[1,i]}" for i in range(500)]
        (source / "raw_brainflow.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return source

    def invoke(self, source_args, output):
        with patch.object(core, "load_config", return_value=core.DEFAULT_CONFIG), patch.object(core, "load_model", return_value=types.SimpleNamespace(calibrated_threshold=0.8, metadata={"threshold_calibration": {"scope": "validation_only"}})), patch.object(core, "apply_model_to_array", side_effect=lambda bundle, x, valid_mask: noop_result(x, valid_mask)), redirect_stdout(io.StringIO()) as stdout:
            code = offline.main([*source_args, "--model", str(self.model), "--config", str(self.config), "--output-dir", str(output)])
        return code, stdout.getvalue()

    def test_npz_independent_outputs_and_input_hash_immutable(self):
        before = offline._snapshot(self.npz)
        output = self.root / "diagnostic"
        code, stdout = self.invoke(["--input", str(self.npz)], output)
        self.assertEqual(0, code, stdout)
        self.assertEqual(before, offline._snapshot(self.npz))
        report = json.loads((output / "quality_metrics.json").read_text())
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["automatic"])
        self.assertFalse(report["release"])
        self.assertFalse(report["missing_data_repaired"])
        self.assertTrue((output / "denoised_signal.csv").exists())
        with np.load(output / "denoised_signal.npz", allow_pickle=False) as arrays:
            for field in offline.RESULT_FIELDS:
                self.assertEqual(self.signal.shape, arrays[field].shape)

    def test_session_raw_quality_and_metadata_immutable(self):
        source = self.session()
        before = offline._snapshot(source)
        code, stdout = self.invoke(["--session", str(source)], self.root / "session-diagnostic")
        self.assertEqual(0, code, stdout)
        self.assertEqual(before, offline._snapshot(source))
        self.assertEqual("historical_not_passed", json.loads((source / "quality_metrics.json").read_text())["status"])

    def test_bad_rate_and_gaps_refused_before_loading_model(self):
        self.write_npz(rate=256)
        with self.assertRaises(evaluation.EvaluationError):
            offline._read_npz(self.npz)
        self.write_npz(package_num=np.r_[np.arange(499) % 256, 0])
        with self.assertRaises(evaluation.EvaluationError):
            offline._read_npz(self.npz)
        source = self.session()
        raw = source / "raw_brainflow.tsv"
        raw.write_text(raw.read_text().replace("3\t3\t", "3\t9\t"))
        with self.assertRaises(evaluation.EvaluationError):
            offline._read_session(source)

    def test_nonpositive_thread_counts_rejected_before_optional_model_load(self):
        with patch.object(core, "load_model") as load, redirect_stdout(io.StringIO()):
            self.assertEqual(2, offline.main(["--input", str(self.npz), "--threads", "0"]))
            self.assertEqual(2, evaluation.main(["--threads", "-1"]))
        load.assert_not_called()

    def test_unknown_rate_and_continuity_not_guessed(self):
        np.savez(self.npz, signal=self.signal, sample_rate_hz=250)
        with self.assertRaisesRegex(evaluation.EvaluationError, "continuity"):
            offline._read_npz(self.npz)
        source = self.session()
        (source / "session.json").write_text("{}")
        with self.assertRaisesRegex(evaluation.EvaluationError, "explicitly"):
            offline._read_session(source)

    def test_output_must_be_new_and_not_inside_source(self):
        source = self.session()
        with self.assertRaises(evaluation.EvaluationError):
            offline._ensure_output_separate(source / "derived" / "v2", source)
        with self.assertRaises(evaluation.EvaluationError):
            offline._ensure_output_separate(self.root, self.npz)
        with self.assertRaises(evaluation.EvaluationError):
            offline._ensure_output_separate(self.npz, self.npz)

    def test_nonreplacement_and_invalid_samples_cannot_be_fabricated(self):
        valid = np.ones_like(self.signal, dtype=bool)
        result = noop_result(self.signal, valid)
        result.denoised[0, 0] += 1
        with self.assertRaises(evaluation.EvaluationError):
            offline._validate_result(result, self.signal, valid)
        valid[0, 0] = False
        result = noop_result(self.signal, np.ones_like(valid))
        with self.assertRaises(evaluation.EvaluationError):
            offline._validate_result(result, self.signal, valid)


if __name__ == "__main__":
    unittest.main()
