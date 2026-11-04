"""Synthetic-only v4 evaluator tests; no real holdout/model/training access."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from scripts import evaluate_denoiser_v4 as ev


@dataclass
class FakeResult:
    denoised: np.ndarray
    model_denoised: np.ndarray
    artifact_probability: np.ndarray
    confidence: np.ndarray
    replacement_mask: np.ndarray
    valid_mask: np.ndarray
    rejected_mask: np.ndarray
    diagnostics: dict


def fake_apply(bundle, signal, valid_mask=None):
    valid = np.isfinite(signal) if valid_mask is None else valid_mask & np.isfinite(signal)
    rejected = np.broadcast_to(~np.all(valid, axis=0), signal.shape).copy()
    mask = valid & ~rejected
    return FakeResult(signal.copy(), signal.copy(), np.full(signal.shape, .7), np.full(signal.shape, .3),
                      mask, valid, rejected, {"synthetic": True})


def config_contract():
    # Design JSON only: no raw data/holdout/weights.
    config = ev.read_json(ev.ROOT / "configs/learned_denoiser_v4.json")
    contract = ev.read_json(ev.ROOT / "analysis_outputs/learned_denoiser_v4_interface_contract.json")
    prereg = ev.read_json(ev.ROOT / "analysis_outputs/learned_denoiser_v4_preregistration.json")
    return config, contract, prereg


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def frozen_fixture(root):
    config, interface, prereg = config_contract()
    files = list(ev.BASE_FILES) + [f"{ev.DONG_DIR}/{s}" for s in ev.DONG_SUBJECTS] + ["datasets/synthetic_source.mat", "tests/extra_v4.py"]
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"synthetic placeholder, NOT real data or source")
    prereg["data"]["sources"] = {"synthetic": {"relative_path": "datasets/synthetic_source.mat", "sha256": ev.sha256(root / "datasets/synthetic_source.mat")}}
    config_path = root / "configs/learned_denoiser_v4.json"
    interface_path = root / "analysis_outputs/learned_denoiser_v4_interface_contract.json"
    prereg_path = root / "analysis_outputs/learned_denoiser_v4_preregistration.json"
    write_json(config_path, config)
    write_json(interface_path, interface)
    write_json(prereg_path, prereg)
    dataset = root / "datasets/processed/learned_denoiser_v4.npz"
    training_path = root / "analysis_outputs/learned_denoiser_v4_training_source_lock.json"
    common = {"schema_version": 1, "model_version": ev.MODEL_VERSION,
              "config_sha256": ev.sha256(config_path), "preregistration_sha256": ev.sha256(prereg_path),
              "dataset_sha256": ev.sha256(dataset), "automatic_enabled": False, "production_validated": False}
    training = {**common, "stage": "training", "files": {rel: ev.sha256(root / rel) for rel in files}}
    write_json(training_path, training)
    model = root / "models/denoiser_v4.pt"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"synthetic checkpoint placeholder, never torch loaded")
    record = model.with_suffix(".training.json")
    write_json(record, {"synthetic": True})
    evaluation = {**common, "stage": "evaluation", "checkpoint_sha256": ev.sha256(model),
                  "training_source_lock_sha256": ev.sha256(training_path), "files": dict(training["files"])}
    # Evaluation-only source/inputs belong to the evaluation lock, not the
    # already frozen training-source set.
    for rel in ev.EVALUATION_FILES + ["datasets/processed/learned_denoiser_v1.npz",
                                      ev.DONG_METADATA, f"{ev.DONG_DIR}/DATASET_MANIFEST.json"]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic evaluation-only placeholder, never real holdout")
        evaluation["files"][rel] = ev.sha256(path)
    for path in (model, record, training_path):
        evaluation["files"][path.relative_to(root).as_posix()] = ev.sha256(path)
    lock_path = root / "analysis_outputs/learned_denoiser_v4_experiment_lock.json"
    write_json(lock_path, evaluation)
    return {"root": root, "lock_path": lock_path, "interface_path": interface_path, "config_path": config_path,
            "checkpoint": model, "dataset": dataset, "prereg_path": prereg_path, "training_lock_path": training_path}


class InterfaceAndLockTests(unittest.TestCase):
    def test_actual_contract_signatures_and_dataclass(self):
        config, interface, _ = config_contract()
        ev.validate_interface(interface, config)
        self.assertIn("(", interface["core"]["functions"][0])
        interface["core"]["result"] = "arrays [C,T]"
        with self.assertRaises(ev.V4EvaluationError):
            ev.validate_interface(interface, config)

    def test_missing_function_config_and_runtime_rejected(self):
        config, interface, _ = config_contract()
        bad = deepcopy(interface)
        bad["core"]["functions"] = ["load_config(path=None)"]
        with self.assertRaises(ev.V4EvaluationError): ev.validate_interface(bad, config)
        config["runtime"]["enabled"] = True
        interface["config"] = deepcopy(config)
        with self.assertRaises(ev.V4EvaluationError): ev.validate_interface(interface, config)

    def test_real_flat_schema_and_every_mapping_entry_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = frozen_fixture(Path(directory))
            result = ev.inspect_lock(**fixture)
            self.assertTrue(result["all_payloads_present"])
            # This file is NOT in required subset: whole map still must verify it.
            (fixture["root"] / "tests/extra_v4.py").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                ev.inspect_lock(**fixture)

    def test_missing_required_hash_and_top_level_binding_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            for rel in ev.EVALUATION_FILES + ["datasets/processed/learned_denoiser_v1.npz",
                                               ev.DONG_METADATA, f"{ev.DONG_DIR}/DATASET_MANIFEST.json"]:
                with self.subTest(missing=rel):
                    fixture = frozen_fixture(Path(directory))
                    lock = ev.read_json(fixture["lock_path"])
                    del lock["files"][rel]
                    write_json(fixture["lock_path"], lock)
                    with self.assertRaisesRegex(ValueError, "missing required"):
                        ev.inspect_lock(**fixture)
            fixture = frozen_fixture(Path(directory))
            lock = ev.read_json(fixture["lock_path"])
            lock["checkpoint_sha256"] = "0" * 64
            write_json(fixture["lock_path"], lock)
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                ev.inspect_lock(**fixture)

    def test_root_escape_absolute_path_and_invalid_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for files in ({"../escape": "0" * 64}, {str(root / "a"): "0" * 64}, {"a": "A" * 64}):
                with self.subTest(files=files), self.assertRaises(ValueError):
                    ev.lock_files(root, {"files": files}, "fixture")

    def test_check_only_never_loads_data_or_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen_fixture(root)
            output = root / "check.json"
            with patch.object(ev, "load_bundle", side_effect=AssertionError("must not load model")), \
                 patch.object(ev.importlib, "import_module", side_effect=AssertionError("must not import data")), \
                 patch.object(ev, "evaluate_paired", side_effect=AssertionError("must not score")):
                self.assertEqual(ev.main(["--root", str(root), "--check-only", "--output", str(output)]), 0)
            report = ev.read_json(output)
            self.assertEqual(report["frozen_check_status"], "passed")
            self.assertFalse(report["holdout_accessed"])
            self.assertFalse(report["training_executed"])
            before = output.read_bytes()
            with patch.object(ev, "inspect_lock", side_effect=AssertionError("existing report must stop first")):
                self.assertEqual(ev.main(["--root", str(root), "--check-only", "--output", str(output)]), 1)
            self.assertEqual(output.read_bytes(), before)

    def test_missing_lock_reports_not_ready_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(ev, "load_bundle", side_effect=AssertionError("must not infer")):
                self.assertEqual(ev.main(["--root", str(root), "--check-only", "--output", "check.json"]), 1)
            report = ev.read_json(root / "check.json")
            self.assertEqual(report["frozen_check_status"], "failed")
            self.assertFalse(report["holdout_accessed"])

    def test_full_failure_report_reserved_before_test_loader_and_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = frozen_fixture(root)
            output = root / "evaluation.json"
            def loader(path):
                self.assertEqual(ev.read_json(output)["evaluation_status"], "started")
                raise ValueError("synthetic loader failure")
            module = SimpleNamespace(load_evaluation_split=Mock(side_effect=loader))
            bundle = SimpleNamespace(metadata={"dataset_sha256": ev.sha256(fixture["dataset"])},
                                     device="cpu", artifact_threshold=.5)
            core = SimpleNamespace(torch=SimpleNamespace(__version__="mock"))
            with patch.object(ev, "load_bundle", return_value=(core, bundle)), \
                 patch.object(ev.importlib, "import_module", return_value=module):
                self.assertEqual(ev.main(["--root", str(root), "--output", str(output)]), 1)
            report = ev.read_json(output)
            self.assertTrue(report["holdout_accessed"])
            self.assertFalse(report["evaluation_executed"])
            self.assertEqual(report["evaluation_status"], "failed")
            self.assertFalse(report["threshold_tuning_performed"])


class SyntheticMetricTests(unittest.TestCase):
    def setUp(self):
        self.limits = config_contract()[2]["predeclared_acceptance"]
        self.t = np.arange(500) / 250

    def test_batch_independence_shape_masks_and_order(self):
        windows = np.arange(7 * 500, dtype=float).reshape(7, 500)
        mask = np.ones(windows.shape, dtype=bool)
        mask[2, 10] = False
        calls = []
        def apply(bundle, signal, valid_mask=None):
            calls.append(signal.shape)
            return fake_apply(bundle, signal, valid_mask)
        batch = ev.infer_windows(None, windows, valid_mask=mask, chunk_size=3, apply=apply)
        single = ev.infer_windows(None, windows, valid_mask=mask, chunk_size=1, apply=fake_apply)
        for key in ev.ARRAY_FIELDS: np.testing.assert_array_equal(batch[key], single[key])
        self.assertEqual(calls, [(500, 3), (500, 3), (500, 1)])
        np.testing.assert_array_equal(batch["denoised"], windows)
        self.assertTrue(batch["rejected_mask"][2].all())
        bad = fake_apply(None, windows.T)
        bad.denoised = bad.denoised.T
        with self.assertRaises(ValueError): ev.validate_result(bad, windows.T)

    def test_original_DC_global_rmse_not_selective_or_mean_window_rmse(self):
        raw = np.stack([2 + np.sin(2*np.pi*12*self.t), 6 + 2*np.sin(2*np.pi*12*self.t)])
        clean = raw - np.array([1, 3])[:, None]
        result = ev.infer_windows(None, raw, apply=fake_apply)
        summary = ev.metric_summary(raw, clean, result)
        self.assertAlmostEqual(summary["metrics"]["gated"]["global_rmse_original_units"], np.sqrt(5))
        self.assertNotAlmostEqual(summary["metrics"]["gated"]["global_rmse_original_units"], 2)
        self.assertEqual(summary["metrics"]["gated"]["snr_gain_db"], 0)

    def test_normalization_all_samples_not_mean_of_ratios_or_accepted_only(self):
        error = np.stack([np.ones(500), np.zeros(500)])
        self.assertAlmostEqual(ev.normalized_rmse(error, np.array([1., 2.])), np.sqrt(.5))
        self.assertIsNone(ev.normalized_rmse(error, np.array([0., 2.])))
        self.assertEqual(ev.normalized_rmse(np.zeros_like(error), np.zeros(2)), 0)

    def test_zero_error_is_finite_failclosed(self):
        raw = np.sin(2*np.pi*12*self.t)[None]
        result = ev.infer_windows(None, raw, apply=fake_apply)
        summary = ev.metric_summary(raw, raw, result)
        self.assertIsNone(summary["metrics"]["gated"]["snr_gain_db"])
        self.assertFalse(ev.gate(None, 1., ">=")["passed"])
        json.dumps(ev.jsonable(summary), allow_nan=False)

    def test_spectral_projection_floor_and_zero_reference(self):
        raw = (np.sin(2*np.pi*12*self.t) + .2*np.cos(2*np.pi*1*self.t))[None]
        low = raw + .1*np.sin(2*np.pi*2*self.t)
        high = raw + .1*np.sin(2*np.pi*20*self.t)
        self.assertLess(ev.protected_discrepancy(raw, low)[0], 1e-6)
        self.assertGreater(ev.protected_discrepancy(raw, high)[0], 1e-6)
        np.testing.assert_array_equal(ev.protected_discrepancy(np.zeros_like(raw), np.zeros_like(raw)), [0])
        self.assertTrue(np.isnan(ev.protected_discrepancy(np.zeros_like(raw), high)[0]))
        low_raw = np.sin(2*np.pi*1*self.t)[None]
        discrepancy = ev.protected_discrepancy(low_raw, low_raw + .1*np.cos(2*np.pi*2*self.t))[0]
        reference = np.linalg.norm(np.fft.rfft(low_raw-low_raw.mean())[0, 16:81])
        scale = ev.rms(low_raw-low_raw.mean())
        numerator = np.linalg.norm(np.fft.rfft(.1*np.cos(2*np.pi*2*self.t))[16:81])
        # Cancellation rounding depends on actual stored subtraction, not analytic delta.
        actual_num = np.linalg.norm(np.fft.rfft((low_raw + .1*np.cos(2*np.pi*2*self.t) - low_raw)/scale)[0,16:81])
        self.assertAlmostEqual(discrepancy, actual_num/max(reference/scale, np.sqrt(500)*1e-12))

    def test_noop_cannot_pass_coverage_even_accepted_mask(self):
        raw = np.tile(np.sin(2*np.pi*12*self.t), (4,1))
        clean = raw.copy()
        clean[2:] -= 1
        data = {"noisy": raw, "clean": clean, "is_identity": np.array([True, True, False, False]),
                "group_id": np.array([1,2,1,2]), "channel_index": np.array([0,1,0,1]),
                "segment_id": np.arange(4), "window_start": np.zeros(4, dtype=int)}
        report = ev.evaluate_paired(data, None, self.limits, apply=fake_apply)
        self.assertFalse(report["gates"]["corrupted_actual_change_coverage"]["passed"])
        self.assertEqual(report["summary"]["corrupted"]["coverage"]["replacement_fraction"], 1)
        self.assertEqual(report["summary"]["corrupted"]["coverage"]["actual_change_fraction"], 0)
        self.assertEqual(len(report["per_window"]), 4)
        self.assertEqual(set(report["per_group_and_channel"]["group_id"]), {"1", "2"})
        self.assertEqual(report["gates"]["corrupted_improvement"]["limit"], self.limits["corrupted_rmse_improvement_min"])

    def test_known_v1_label_not_independent_and_all26(self):
        raw = np.tile(np.sin(2*np.pi*12*np.arange(1000)/250), (26,8,1))
        data = {"noisy": raw, "clean": raw-.1, "artifact_mask": np.zeros(raw.shape, dtype=bool)}
        report = ev.evaluate_known(data, None, self.limits, apply=fake_apply)
        self.assertFalse(report["independent"])
        self.assertEqual(report["name"], "known_v1_regression_stress_test")
        self.assertEqual(report["single_channel_500_window_count"], 416)
        self.assertFalse(report["threshold_tuning_performed"])
        self.assertTrue(report["gates"]["gated_to_raw_rmse_ratio"]["passed"])
        data["noisy"] = raw[:25]
        with self.assertRaises(ValueError): ev.evaluate_known(data, None, self.limits, apply=fake_apply)

    def test_dong_strict_shape_rate_interval_gain_and_order(self):
        meta = ev.read_json(ev.ROOT / ev.DONG_METADATA)
        ev.validate_dong_metadata(meta)
        for key, value in (("sample_rate_hz", 1000), ("axis_order", ["time", "channel", "target", "block"]),
                           ("target_axis_mapping_confirmed", True)):
            bad = deepcopy(meta); bad[key] = value
            with self.assertRaises(ValueError): ev.validate_dong_metadata(bad)
        raw = np.arange(8*1250*40*4, dtype=np.float64).reshape(8,1250,40,4)
        windows = ev.dong_windows(raw)
        self.assertEqual(windows.shape, (2560,500))
        np.testing.assert_array_equal(windows[0], raw[0,125:625,0,0])
        np.testing.assert_array_equal(windows[1], raw[0,625:1125,0,0])
        np.testing.assert_array_equal(windows[2], raw[1,125:625,0,0])
        with self.assertRaises(ValueError): ev.dong_windows(raw.transpose(1,0,2,3))
        with self.assertRaises(ValueError): ev.evaluate_dong(windows, [{}]*2560, meta, None, self.limits, apply=fake_apply)

    def test_invalid_result_cannot_change_raw_without_replacement(self):
        raw = np.ones((500,2))
        result = fake_apply(None, raw)
        result.replacement_mask[:] = False
        result.denoised[0,0] += 1
        with self.assertRaisesRegex(ValueError, "outside replacement"): ev.validate_result(result, raw)


if __name__ == "__main__":
    unittest.main()
