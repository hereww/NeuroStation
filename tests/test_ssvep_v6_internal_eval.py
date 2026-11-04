"""Synthetic-only v6 diagnostic tests. No real NPZ/PT, CUDA, or external drive I/O."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import importlib
import io
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from eeg_tools import ssvep_v6_contract as c
from eeg_tools import ssvep_v6_internal_eval as e


def config():
    # The sole real artifact read here is the explicitly permitted formal JSON.
    return json.loads((e.ROOT / "configs/learned_denoiser_v6_ssvep.json").read_text(encoding="utf-8"))


def case_rows():
    rows = []
    for dataset in e.DATASETS:
        for sid in e.SUBJECTS:
            for trial in range(e.TRIALS[dataset]):
                for condition in e.CONDITIONS:
                    identity = condition == "identity"
                    a, b = (0.0, 0.0) if identity else (1.0, 0.9)
                    rows.append({"dataset": dataset, "subject_id": sid, "trial_index": trial,
                                 "condition": condition, "seed": e.diagnostic_seed(dataset, sid, trial, condition),
                                 "window_samples": 256, "window_start_sample": e.STARTS[dataset],
                                 "sample_rate_hz": e.RATES[dataset], "input_normalized_mse": a,
                                 "output_normalized_mse": b, "relative_reduction": None if identity else (a-b)/a,
                                 "identity_normalized_change": b if identity else None,
                                 "outcome": "tie" if identity else "win",
                                 "protected_complex_energy_ratio": 0.0 if identity else None,
                                 "protected_weighted_abs_phase_radians": 0.0 if identity else None,
                                 "undefined_phase_bin_count": 0})
    return rows


class FakeTensor:
    def __init__(self, array, torch, device="cpu"):
        self.array = np.asarray(array)
        self.torch = torch
        self.dtype = self.array.dtype
        self.shape = self.array.shape
        self.device = SimpleNamespace(type=device.split(":")[0])
        self.requires_grad = False

    def to(self, *, device, dtype):
        self.array = self.array.astype(dtype, copy=False)
        self.dtype = self.array.dtype
        self.device = SimpleNamespace(type=device.split(":")[0])
        self.torch.events.append("tensor_to_" + device)
        return self

    def detach(self):
        return self

    def cpu(self):
        self.device = SimpleNamespace(type="cpu")
        return self

    def contiguous(self):
        self.array = np.ascontiguousarray(self.array)
        return self

    def numpy(self):
        return self.array


class FakeModel:
    def __init__(self, torch, dataset):
        self.torch, self.dataset = torch, dataset
        self.training = True
        self.weight = FakeTensor(np.zeros(1, dtype=np.float32), torch)
        self.weight.requires_grad = True
        self.batches = []
        self.mutate = False
        self.bad_prediction = None

    def parameters(self):
        return iter([self.weight])

    def state_dict(self):
        return {"synthetic_weight": self.weight}

    def eval(self):
        self.training = False
        self.torch.events.append("model_eval")
        return self

    def requires_grad_(self, value):
        self.weight.requires_grad = value
        self.torch.events.append("model_no_grad")
        return self

    def to(self, **kwargs):
        self.weight.to(**kwargs)
        return self

    def __call__(self, tensor):
        assert self.torch.depth == 1, "forward outside inference_mode"
        assert not self.training and not self.weight.requires_grad
        assert tensor.device.type == "cuda" and tensor.dtype == np.float32
        self.batches.append(tensor.shape[0])
        self.torch.events.append("forward_" + self.dataset)
        if self.mutate:
            self.weight.array += 1
        out = tensor.array.copy()
        if self.bad_prediction == "nan":
            out[0, 0, 0] = np.nan
        if self.bad_prediction == "shape":
            out = out[:, :, :-1]
        if self.bad_prediction == "dtype":
            out = out.astype(np.float64)
        result = FakeTensor(out, self.torch, "cuda:0")
        result.requires_grad = self.bad_prediction == "grad"
        return result


class FakeTorch:
    float32 = np.dtype("float32")
    __version__ = "synthetic-no-real-torch"
    version = SimpleNamespace(cuda="synthetic")

    def __init__(self):
        self.depth = 0
        self.events = []
        self.available = True
        self.cuda = SimpleNamespace(is_available=lambda: self.available, device_count=lambda: 1,
                                    set_device=lambda i: self.events.append("cuda_set"),
                                    synchronize=lambda i: self.events.append("cuda_sync"),
                                    get_device_name=lambda i: "synthetic GPU; never hardware")
        self.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                        cudnn=SimpleNamespace(allow_tf32=True, benchmark=True, deterministic=False))
        self.load = Mock(side_effect=self._load)

    def _load(self, path, *, map_location, weights_only):
        assert map_location == "cpu" and weights_only is True
        self.events.append("torch_load_cpu_weights_only")
        return {"synthetic": "not a real checkpoint"}

    def set_num_threads(self, value):
        assert value == 4

    def use_deterministic_algorithms(self, value):
        assert value is True

    def zeros(self, shape, *, device, dtype):
        self.events.append("fake_cuda_probe")
        return FakeTensor(np.zeros(shape, dtype=dtype), self, device)

    def from_numpy(self, array):
        return FakeTensor(array, self)

    def isfinite(self, tensor):
        return np.isfinite(tensor.array)

    @contextmanager
    def inference_mode(self):
        self.events.append("enter_inference_mode")
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1
            self.events.append("exit_inference_mode")


def records():
    # Broadcast in-memory synthetic full-size records; no file-based waveforms.
    result = []
    for dataset in e.DATASETS:
        shape = e.SHAPES[dataset]
        rng = np.random.default_rng(1234 + e.RATES[dataset])
        signal = rng.normal(size=(1, shape[1], 8)).astype(np.float64)
        for sid in e.SUBJECTS:
            result.append(SimpleNamespace(dataset=dataset, subject_id=sid,
                                          sample_rate_hz=e.RATES[dataset],
                                          signal=np.broadcast_to(signal, shape)))
    return result


class Harness:
    """All GPU/model/data access mocked; safety/path/publication use real shared code."""
    def __init__(self, root):
        self.root, self.cfg, self.torch = root, config(), FakeTorch()
        self.models = {d: FakeModel(self.torch, d) for d in e.DATASETS}
        self.summaries = []
        self.fs_seen = []
        self.record_list = records()
        self.state = SimpleNamespace(root=root, config=self.cfg, lock_sha256="e"*64,
                                     stage="internal_evaluation", check_only=False)
        self.state.report = lambda: {"schema_version": 1, "status": "formal", "stage": "internal_evaluation",
                                     "deferred": [], "verified_hashes": {}, "check_only": self.state.check_only}
        for rel, obj in ((c.CONFIG_RELATIVE, self.cfg), (c.PREREG_RELATIVE, {"synthetic": True}),
                         (c.TRAIN_LOCK_RELATIVE, {"training_performed": False}),
                         (c.TRAIN_REPORT_RELATIVE, {"training_performed": True})):
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(obj), encoding="utf-8")
        self.train_sha = c.file_sha256(root, c.TRAIN_LOCK_RELATIVE)
        (root / c.EVAL_LOCK_RELATIVE).write_text(json.dumps({"training_performed": True,
            "training_lock_sha256": self.train_sha, "checkpoint_sha256": "c"*64}), encoding="utf-8")
        self.api = SimpleNamespace(**{k: getattr(c, k) for k in (
            "V6Error", "MODEL_VERSION", "CONFIG_RELATIVE", "PREREG_RELATIVE", "TRAIN_LOCK_RELATIVE",
            "EVAL_LOCK_RELATIVE", "CHECKPOINT_RELATIVE", "TRAIN_REPORT_RELATIVE", "safe_path",
            "assert_new_outputs", "publish_bundle")})
        self.api.file_sha256 = Mock(side_effect=lambda r, p: "c"*64 if p == c.CHECKPOINT_RELATIVE
                                    else c.file_sha256(r, p))
        self.api.preflight = Mock(side_effect=self.preflight)
        self.api.verify_frozen = Mock(side_effect=lambda state: self.torch.events.append("verify_frozen"))
        self.api.load_training_records = Mock(side_effect=self.load_records)
        self.api.corrupt_case = Mock(side_effect=self.corrupt)
        self.api.normalize_window = self.normalize
        self.api.restore_window = self.restore
        self.api.case_metrics = Mock(side_effect=self.metrics)
        self.api.summarize_cases = Mock(side_effect=self.summarize)
        self.api.gate_summary = Mock(return_value={"all_passed": True, "gates": {"synthetic": {"passed": True}}})
        self.validator = Mock(side_effect=self.validate)
        self.model_module = SimpleNamespace(validate_checkpoint=self.validator)

    def preflight(self, *, root, stage, check_only):
        assert root == self.root and stage == "internal_evaluation"
        self.state.check_only = check_only
        self.torch.events.append("preflight")
        return self.state

    def validate(self, ck, cfg, training_sha):
        assert ck == {"synthetic": "not a real checkpoint"}
        assert cfg == self.cfg and training_sha == self.train_sha
        self.torch.events.append("validate_checkpoint_cpu")
        assert all(m.weight.device.type == "cpu" for m in self.models.values())
        return self.models

    def load_records(self, state):
        assert not state.check_only
        self.torch.events.append("load_only_synthetic_5_6")
        return self.record_list

    def corrupt(self, ref, dataset, sid, trial, condition, cfg, *, purpose, epoch):
        assert purpose == "diagnostic" and epoch == 0 and cfg == self.cfg
        return ref.copy() if condition == "identity" else ref + 0.1, e.diagnostic_seed(dataset, sid, trial, condition)

    @staticmethod
    def normalize(noisy):
        mean = noisy.mean(axis=1, keepdims=True)
        rms = np.sqrt(np.mean((noisy - mean)**2, axis=1, keepdims=True) + 1e-8)
        return ((noisy - mean)/rms).astype(np.float32), mean, rms

    @staticmethod
    def restore(noisy, norm, pred, rms):
        return noisy + (pred.astype(np.float64) - norm.astype(np.float64))*rms

    def metrics(self, ref, noisy, out, *, identity, sample_rate_hz):
        self.fs_seen.append(sample_rate_hz)
        return c.case_metrics(ref, noisy, out, identity=identity, sample_rate_hz=sample_rate_hz)

    def summarize(self, rows, *, allow_mixed_datasets=False):
        # Structural fake: numerical aggregate and gate boundaries tested below
        # with real shared helper. No duplicated aggregate mathematics here.
        if not allow_mixed_datasets:
            assert len({r["dataset"] for r in rows}) == 1
        self.summaries.append(rows)
        return {k: {"case_count": sum(r["condition"] == k for r in rows)}
                for k in e.CONDITIONS if any(r["condition"] == k for r in rows)}

    def import_module(self, name):
        if name == "torch":
            return self.torch
        if name == "eeg_tools.ssvep_denoise_v6":
            return self.model_module
        raise AssertionError("unexpected import: " + name)

    @contextmanager
    def installed(self):
        with patch.object(e, "_contract", return_value=self.api), \
             patch.object(e.importlib, "import_module", side_effect=self.import_module), \
             patch.dict(e.os.environ, {}, clear=False):
            yield self


class CaseValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = case_rows()

    def test_complete_cohort_counts(self):
        result = e.validate_cases(self.rows)
        self.assertEqual(result["case_count"], 4200)
        self.assertEqual(result["reference_trial_count"], 840)
        self.assertEqual(result["phase_case_count"], 840)
        self.assertTrue(result["complete_cohort"] and result["phase_verifiable"])

    def test_seed_is_diagnostic_not_selection(self):
        cfg = config()
        for dataset in e.DATASETS:
            for sid in e.SUBJECTS:
                for condition in e.CONDITIONS:
                    seed = e.diagnostic_seed(dataset, sid, 0, condition)
                    self.assertEqual(seed, c._case_seed(dataset, sid, 0, condition, "diagnostic", 0, cfg))
                    self.assertNotEqual(seed, c._case_seed(dataset, sid, 0, condition, "internal_selection", 0, cfg))

    def test_missing_and_duplicate_refused(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[0]], self.rows[:-1] + [self.rows[0]]):
            with self.subTest(length=len(rows)), self.assertRaises(c.V6Error):
                e.validate_cases(rows)

    def test_illegal_provenance_and_nonfinite_refused(self):
        for key, value in (("subject_id", 4), ("subject_id", 7), ("subject_id", 8), ("subject_id", 9),
                           ("subject_id", 10), ("subject_id", True), ("trial_index", -1),
                           ("seed", self.rows[0]["seed"]-1), ("window_start_sample", 0),
                           ("sample_rate_hz", 250), ("condition", "combined"),
                           ("output_normalized_mse", float("nan")), ("input_normalized_mse", float("inf")),
                           ("protected_complex_energy_ratio", float("inf")), ("undefined_phase_bin_count", True)):
            rows = list(self.rows)
            rows[0] = {**rows[0], key: value}
            with self.subTest(key=key, value=value), self.assertRaises(c.V6Error):
                e.validate_cases(rows)

    def test_labels_waveforms_forbidden(self):
        for key in ("label", "waveform", "reference"):
            rows = list(self.rows)
            rows[0] = {**rows[0], key: []}
            with self.subTest(key=key), self.assertRaises(c.V6Error):
                e.validate_cases(rows)

    def test_phase_undefined_is_not_faked_zero(self):
        rows = list(self.rows)
        rows[0] = {**rows[0], "protected_weighted_abs_phase_radians": None, "undefined_phase_bin_count": 1}
        result = e.validate_cases(rows)
        self.assertFalse(result["phase_verifiable"])
        self.assertEqual(result["undefined_phase_case_count"], 1)
        rows[0] = {**rows[0], "protected_complex_energy_ratio": None, "undefined_phase_bin_count": 0}
        self.assertFalse(e.validate_cases(rows)["phase_verifiable"])

    def test_actual_metrics_native_fs_masks(self):
        rng = np.random.default_rng(29)
        ref = rng.normal(size=(8, 256)).astype(np.float64)
        change = np.sin(2*np.pi*8*np.arange(256)/256)[None, :] * 0.1
        out = ref + change
        a = c.case_metrics(ref, ref, out, identity=True, sample_rate_hz=256)
        b = c.case_metrics(ref, ref, out, identity=True, sample_rate_hz=250)
        self.assertGreater(a["protected_complex_energy_ratio"], 1e-6)
        self.assertLess(b["protected_complex_energy_ratio"], 1e-20)
        zero = np.zeros((8, 256), dtype=np.float64)
        undefined = c.case_metrics(zero, zero, zero, identity=True, sample_rate_hz=250)
        self.assertIsNone(undefined["protected_weighted_abs_phase_radians"])
        self.assertIsNone(undefined["protected_complex_energy_ratio"])

    def test_real_per_dataset_summarizer_ratio_of_means(self):
        rows = [r for r in self.rows if r["dataset"] == "Wearable"]
        summary = c.summarize_cases(rows)
        self.assertEqual(summary["identity"]["case_count"], 480)
        self.assertAlmostEqual(summary["high_frequency_burst"]["ratio_of_mean_reduction"], 0.1)
        self.assertTrue(c.gate_summary(summary, config())["all_passed"])
        # Unequal baselines distinguish ratio-of-means from mean-of-ratios.
        selected = deepcopy([r for r in rows if r["condition"] == "transient"][:2])
        selected[0].update(input_normalized_mse=1.0, output_normalized_mse=0.0, relative_reduction=1.0)
        selected[1].update(input_normalized_mse=9.0, output_normalized_mse=9.0, relative_reduction=0.0, outcome="tie")
        self.assertAlmostEqual(c.summarize_cases(selected)["transient"]["ratio_of_mean_reduction"], 0.1)

    def test_gate_exact_boundaries_and_undefined(self):
        cfg = config()
        summary = c.summarize_cases([r for r in self.rows if r["dataset"] == "Nakanishi2015"])
        summary["high_frequency_burst"]["ratio_of_mean_reduction"] = 0.05
        for condition in ("transient", "low_frequency_drift", "common_mode"):
            summary[condition]["ratio_of_mean_reduction"] = 0.0
        summary["identity"].update(mean_identity_normalized_change=0.0005, mean_output_normalized_mse=0.0005,
                                  mean_protected_complex_energy_ratio=1e-6,
                                  mean_protected_weighted_abs_phase_radians=0.02)
        self.assertTrue(c.gate_summary(summary, cfg)["all_passed"])
        for condition, field, value in (("high_frequency_burst", "ratio_of_mean_reduction", np.nextafter(.05, 0.0)),
                                       ("transient", "ratio_of_mean_reduction", -1e-15),
                                       ("identity", "mean_identity_normalized_change", .0005000001),
                                       ("identity", "mean_protected_complex_energy_ratio", 1.000001e-6),
                                       ("identity", "mean_protected_weighted_abs_phase_radians", .02000001),
                                       ("identity", "mean_protected_weighted_abs_phase_radians", None),
                                       ("identity", "undefined_phase_case_count", 1)):
            changed = deepcopy(summary)
            changed[condition][field] = float(value) if isinstance(value, np.floating) else value
            if field == "mean_identity_normalized_change":
                changed[condition]["mean_output_normalized_mse"] = value
            with self.subTest(field=field):
                self.assertFalse(c.gate_summary(changed, cfg)["all_passed"])


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ssvep_v6_synthetic_eval_", dir=e.ROOT.parent)
        self.root = Path(self.temp.name)
        self.h = Harness(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_4200_fake_gpu_run_and_exclusive_publication(self):
        with self.h.installed():
            report = e.evaluate(self.root)
        self.assertEqual(report["case_count"], 4200)
        self.assertEqual(report["phase_case_count"], 840)
        self.assertTrue(report["all_internal_gates_passed"])
        self.assertEqual(report["deployment"]["status"], "blocked")
        self.assertFalse(report["scope"]["runtime_enabled"])
        self.assertFalse(report["scope"]["production_validated"])
        self.assertTrue(report["scope"]["checkpoint_prior_training"])
        self.assertFalse(report["scope"]["training_performed_this_evaluation"])
        self.assertFalse(report["gpu"]["amp"])
        self.assertEqual(report["model_state_sha256_before"], report["model_state_sha256_after"])
        self.assertEqual(self.h.torch.depth, 0)
        self.assertEqual(self.h.torch.load.call_count, 1)
        self.h.api.load_training_records.assert_called_once_with(self.h.state)
        self.assertEqual(self.h.fs_seen.count(256), 1800)
        self.assertEqual(self.h.fs_seen.count(250), 2400)
        self.assertEqual(self.h.models["Nakanishi2015"].batches, ([64]*14+[4])*2)
        self.assertEqual(self.h.models["Wearable"].batches, ([64]*18+[48])*2)
        self.assertEqual(self.h.api.corrupt_case.call_count, 4200)
        events = self.h.torch.events
        self.assertLess(events.index("torch_load_cpu_weights_only"), events.index("validate_checkpoint_cpu"))
        self.assertLess(events.index("validate_checkpoint_cpu"), events.index("load_only_synthetic_5_6"))
        self.assertEqual(events[events.index("torch_load_cpu_weights_only")-1], "verify_frozen")
        self.assertEqual(events[events.index("load_only_synthetic_5_6")-1], "verify_frozen")
        self.assertEqual(set(report["aggregates"]), {"global", "dataset", "condition", "dataset_condition", "subject", "subject_condition"})
        self.assertEqual(self.h.api.summarize_cases.call_count, 1+2+5+10+4+20)
        self.assertEqual(self.h.api.gate_summary.call_count, 2)
        paths = (self.root / (e.DEFAULT_OUTPUT_PREFIX+".json"), self.root / (e.DEFAULT_OUTPUT_PREFIX+".md"))
        saved = json.loads(paths[0].read_text(encoding="utf-8"))
        self.assertEqual(saved, report)
        self.assertIn("Hann-OLA", paths[1].read_text(encoding="utf-8"))
        with self.h.installed(), self.assertRaises(c.V6Error):
            e.evaluate(self.root)
        self.assertEqual(self.h.api.preflight.call_count, 1)

    def test_check_only_json_no_binary_cuda_or_model(self):
        original_read = Path.read_text
        reads = []
        def read(path, *args, **kwargs):
            reads.append(path)
            self.assertNotIn(path.suffix, (".pt", ".npz"))
            return original_read(path, *args, **kwargs)
        with self.h.installed(), patch.object(Path, "read_text", read):
            report = e.evaluate(self.root, check_only=True)
        self.assertTrue(reads)
        self.h.api.load_training_records.assert_not_called()
        self.h.torch.load.assert_not_called()
        self.h.validator.assert_not_called()
        self.assertNotIn("cuda_set", self.h.torch.events)
        self.assertFalse(report["checkpoint_loaded"] or report["cuda_initialized"] or report["waveforms_loaded"])
        self.assertEqual(report["case_count"], 0)
        self.assertEqual(report["phase_case_count"], 0)
        self.assertEqual(report["checks"]["checkpoint_schema"], "not_executed")
        self.assertEqual(report["checks"]["npz_schema"], "not_executed")
        self.assertFalse(report["all_internal_gates_passed"])
        self.assertTrue((self.root/(e.DEFAULT_OUTPUT_PREFIX+"_check_only.json")).exists())
        self.assertFalse((self.root/(e.DEFAULT_OUTPUT_PREFIX+".json")).exists())
        self.assertTrue(report["scope"]["checkpoint_prior_training"])

    def test_collision_either_json_or_md_before_preflight_and_load(self):
        for suffix in (".json", ".md", "_check_only.json", "_check_only.md"):
            check = suffix.startswith("_check_only")
            p = self.root/(e.DEFAULT_OUTPUT_PREFIX+suffix)
            p.write_text("existing synthetic output", encoding="utf-8")
            with self.h.installed(), self.assertRaises(c.V6Error):
                e.evaluate(self.root, check_only=check)
            p.unlink()
        self.h.api.preflight.assert_not_called()
        self.h.torch.load.assert_not_called()
        self.h.api.load_training_records.assert_not_called()

    def test_cuda_mandatory_no_cpu_fallback(self):
        self.h.torch.available = False
        with self.h.installed(), self.assertRaisesRegex(c.V6Error, "mandatory"):
            e.evaluate(self.root)
        self.h.torch.load.assert_not_called()
        self.h.api.load_training_records.assert_not_called()

    def test_wrong_checkpoint_hash_before_deserialization(self):
        path = self.root / c.EVAL_LOCK_RELATIVE
        lock = json.loads(path.read_text())
        lock["checkpoint_sha256"] = "a"*64
        path.write_text(json.dumps(lock))
        with self.h.installed(), self.assertRaisesRegex(c.V6Error, "checkpoint byte SHA"):
            e.evaluate(self.root)
        self.h.torch.load.assert_not_called()
        self.h.api.load_training_records.assert_not_called()

    def test_training_lock_actual_hash_and_prior_training_required(self):
        for field, value in (("training_lock_sha256", "a"*64), ("training_performed", False)):
            p = self.root / c.EVAL_LOCK_RELATIVE
            p.write_text(json.dumps({"training_performed": True, "training_lock_sha256": self.h.train_sha,
                                     "checkpoint_sha256": "c"*64, field: value}))
            with self.h.installed(), self.assertRaises(c.V6Error):
                e.evaluate(self.root, check_only=True)
        self.h.torch.load.assert_not_called()

    def test_record_duplicate_missing_and_prohibited_subject(self):
        for sid in (1, 4, 7, 8, 9, 10):
            records_ = records()
            records_[0].subject_id = sid
            with self.subTest(sid=sid), self.assertRaises(c.V6Error):
                e._validate_records(records_)
        data = records()
        for value in (data[:-1], data[:-1]+[data[0]]):
            with self.assertRaises(c.V6Error):
                e._validate_records(value)
        data = records()
        data[0].signal = np.full(e.SHAPES[data[0].dataset], np.nan, dtype=np.float64)
        with self.assertRaises(c.V6Error):
            e._validate_records(data)

    def test_no_grad_and_prediction_safety(self):
        for failure in ("nan", "shape", "dtype", "grad"):
            self.h.models["Nakanishi2015"].bad_prediction = failure
            with self.h.installed(), self.subTest(failure=failure), self.assertRaisesRegex(c.V6Error, "prediction"):
                e.evaluate(self.root)
            self.assertEqual(self.h.torch.depth, 0)
        self.assertFalse((self.root/(e.DEFAULT_OUTPUT_PREFIX+".json")).exists())

    def test_model_state_rechecked_after_forward(self):
        self.h.models["Wearable"].mutate = True
        with self.h.installed(), self.assertRaisesRegex(c.V6Error, "state mutated"):
            e.evaluate(self.root)
        self.assertFalse((self.root/(e.DEFAULT_OUTPUT_PREFIX+".json")).exists())

    def test_phase_failure_blocks_even_fake_passing_domain_gates(self):
        rows = case_rows()
        rows[0].update(protected_weighted_abs_phase_radians=None, undefined_phase_bin_count=1)
        with self.h.installed():
            result = e.summarize_diagnostic(rows, self.h.cfg)
        self.assertTrue(all(g["all_passed"] for g in result["dataset_gates"].values()))
        self.assertFalse(result["all_internal_gates_passed"])
        self.assertFalse(result["phase_verifiable"])

    def test_conflicting_cuda_environment_refused_before_model_import(self):
        with self.h.installed(), patch.dict(e.os.environ, {"CUBLAS_WORKSPACE_CONFIG": "wrong"}), \
             self.assertRaisesRegex(c.V6Error, "conflicting CUBLAS"):
            e.evaluate(self.root)
        self.h.torch.load.assert_not_called()
        self.h.api.load_training_records.assert_not_called()

    def test_validator_rejection_before_cuda_model_move_or_data_load(self):
        self.h.validator.side_effect = c.V6Error("synthetic strict checkpoint rejection")
        with self.h.installed(), self.assertRaisesRegex(c.V6Error, "strict checkpoint"):
            e.evaluate(self.root)
        self.assertTrue(all(m.weight.device.type == "cpu" for m in self.h.models.values()))
        self.h.api.load_training_records.assert_not_called()

    def test_training_lock_rechecked_after_synthetic_inference(self):
        def synchronize(index):
            self.h.torch.events.append("cuda_sync")
            if self.h.torch.events.count("cuda_sync") == 2:
                (self.root / c.TRAIN_LOCK_RELATIVE).write_text('{"synthetic_changed": true}')
        self.h.torch.cuda.synchronize = synchronize
        with self.h.installed(), self.assertRaisesRegex(c.V6Error, "training lock SHA"):
            e.evaluate(self.root)
        self.assertFalse((self.root/(e.DEFAULT_OUTPUT_PREFIX+".json")).exists())

    def test_flat_condition_groups_all_delegate_without_reusing_summary(self):
        with self.h.installed():
            result = e.summarize_diagnostic(case_rows(), self.h.cfg)
        for key, value in result["aggregates"]["dataset_condition"].items():
            dataset, condition = key.split("/")
            self.assertEqual(set(value), {condition})
            self.assertEqual(value[condition]["case_count"], e.TRIALS[dataset]*2)
        for rows in self.h.summaries:
            self.assertEqual(len({(r["dataset"],r["subject_id"],r["trial_index"],r["condition"]) for r in rows}), len(rows))

    def test_full_preflight_state_cannot_be_check_only_or_other_stage(self):
        for stage, check in (("training", False), ("internal_evaluation", True)):
            self.h.state.stage, self.h.state.check_only = stage, check
            with self.h.installed(), self.assertRaises(c.V6Error):
                e.run_diagnostic(self.h.state)
        self.h.torch.load.assert_not_called()


class CliAndPathTests(unittest.TestCase):
    def test_cli_no_input_fs_selection_device_overrides(self):
        parser = e.build_parser()
        args = parser.parse_args(["--check-only"])
        self.assertTrue(args.check_only)
        self.assertEqual(args.output_prefix, e.DEFAULT_OUTPUT_PREFIX)
        for flag in ("--root", "--config", "--checkpoint", "--sample-rate", "--seed", "--subjects", "--device", "--batch-size"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit), patch("sys.stderr"):
                parser.parse_args([flag, "ignored"])

    def test_output_paths_reject_foreign_holdout_traversal(self):
        with tempfile.TemporaryDirectory(dir=e.ROOT.parent) as temp:
            root = Path(temp)
            for value in ("D:/SSVEP/result", "../analysis_outputs/ssvep_v6_bad", "/analysis_outputs/ssvep_v6_bad",
                          "analysis_outputs/ssvep_v5_report", "analysis_outputs/ssvep_v6/holdout9_10",
                          "datasets/holdout/ssvep_v6", "analysis_outputs/ssvep_v6:stream",
                          "analysis_outputs\\ssvep_v6_bad", "analysis_outputs/ssvep_v6_bad.json"):
                with self.subTest(value=value), self.assertRaises(c.V6Error):
                    e.output_paths(root, value)
            self.assertNotEqual(e.output_paths(root, e.DEFAULT_OUTPUT_PREFIX),
                                e.output_paths(root, e.DEFAULT_OUTPUT_PREFIX, check_only=True))

    def test_parent_reparse_refused_before_preflight(self):
        with tempfile.TemporaryDirectory(dir=e.ROOT.parent) as temp:
            root = Path(temp)
            parent = root / "analysis_outputs"
            parent.mkdir()
            original = Path.lstat
            def lstat(path, *args, **kwargs):
                info = original(path, *args, **kwargs)
                if path == parent:
                    return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
                return info
            with patch.object(Path, "lstat", lstat), patch.object(c, "preflight") as pre, self.assertRaises(c.V6Error):
                e.evaluate(root, check_only=True)
            pre.assert_not_called()

    def test_strict_json_rejects_duplicate_and_nonfinite(self):
        with tempfile.TemporaryDirectory(dir=e.ROOT.parent) as temp:
            p = Path(temp) / "synthetic.json"
            for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
                p.write_text(text)
                with self.subTest(text=text), self.assertRaises(c.V6Error):
                    e._read_json(p)

    def test_cli_returns_refusal_and_success_status(self):
        with patch.object(e, "evaluate", side_effect=c.V6Error("synthetic refusal")), patch("builtins.print"):
            self.assertEqual(e.main(["--check-only"]), 2)
        report = {"check_only": True, "case_count": 0, "phase_case_count": 0, "all_internal_gates_passed": False}
        with patch.object(e, "evaluate", return_value=report) as run, patch("builtins.print"):
            self.assertEqual(e.main(["--check-only"]), 0)
        run.assert_called_once_with(check_only=True, output_prefix=e.DEFAULT_OUTPUT_PREFIX)

    def test_bundle_no_clobber_is_shared_exclusive_publication(self):
        with tempfile.TemporaryDirectory(dir=e.ROOT.parent) as temp:
            root = Path(temp)
            paths = e.output_paths(root, e.DEFAULT_OUTPUT_PREFIX)
            c.publish_bundle(root, {paths[0]: b"{}\n", paths[1]: b"synthetic\n"})
            before = [(root/p).read_bytes() for p in paths]
            with self.assertRaises(c.V6Error):
                c.publish_bundle(root, {paths[0]: b"changed", paths[1]: b"changed"})
            self.assertEqual(before, [(root/p).read_bytes() for p in paths])


class CpuCudaTransport:
    """Hardware-only substitution: real torch tensors, real inference_mode, real models.

    CUDA transfers/probes are routed to CPU. No A/B numerical, schema, checkpoint,
    summarizer, gate, preflight, publication, or verifier API is substituted.
    """
    def __init__(self, torch, root):
        self.real, self.root = torch, root
        self.float32 = torch.float32
        self.__version__ = torch.__version__
        self.version = SimpleNamespace(cuda="synthetic_CPU_transport_NOT_actual_CUDA")
        self.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                        cudnn=SimpleNamespace(allow_tf32=True, benchmark=True, deterministic=False))
        self.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                    set_device=lambda index: None, synchronize=lambda index: None,
                                    get_device_name=lambda index: "SYNTHETIC CPU transport; no real GPU")
        self.load_paths = []

    def load(self, path, **kwargs):
        path = Path(path)
        assert path == self.root / c.CHECKPOINT_RELATIVE
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        self.load_paths.append(path)
        return self.real.load(path, **kwargs)  # ONLY the freshly generated synthetic checkpoint.

    def set_num_threads(self, value):
        self.real.set_num_threads(value)

    def use_deterministic_algorithms(self, value):
        self.real.use_deterministic_algorithms(value)

    def zeros(self, shape, *, device, dtype):
        assert device == "cuda:0"
        return self.real.zeros(shape, device="cpu", dtype=dtype)

    def from_numpy(self, value):
        torch = self.real
        class Transfer:
            def to(self, *, device, dtype):
                assert device == "cuda:0" and dtype == torch.float32
                return torch.from_numpy(value).to(device="cpu", dtype=dtype)
        return Transfer()

    def isfinite(self, value):
        return self.real.isfinite(value)

    def inference_mode(self):
        return self.real.inference_mode()


class RealApiIntegrationTests(unittest.TestCase):
    """Real A/B integration with synthetic, internally consistent 40/42-file fixtures.

    CPU-only; checkpoint serialization is newly generated, never a real weight
    artifact. The legacy waveform I/O boundary supplies in-memory synthetic
    records. A itself still verifies cohort, metadata, hashes and frozen state.
    Only fixture hash constants and physical CUDA transport are substituted;
    formal model parameters, sampling, corruption, metrics and gates are intact.
    """
    @classmethod
    def setUpClass(cls):
        # These fixture builders invoke genuine A/B APIs. No training is run.
        cls.a_fixtures = importlib.import_module("test_ssvep_v6_contract")
        cls.b_fixtures = importlib.import_module("test_ssvep_denoise_v6")
        cls.b = importlib.import_module("eeg_tools.ssvep_denoise_v6")
        cls.torch = cls.b.torch

    def setUp(self):
        # A combined suite may already have initialized CUDA in earlier B tests.
        # C must preserve that incoming state, not assume a fresh process.
        self.cuda_initialized_before = self.torch.cuda.is_initialized()
        self.temp = tempfile.TemporaryDirectory(prefix="ssvep_v6_real_api_synthetic_", dir=e.ROOT.parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(e.os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}))
        # Never permit actual GPU initialization, even if an accidental transfer slips through.
        self.stack.enter_context(patch.object(self.torch.cuda, "_lazy_init",
                                             side_effect=AssertionError("actual CUDA forbidden in synthetic tests")))
        train = self.a_fixtures.Fixture(self.root, stage="training")
        self.train_sha = train.digest(c.TRAIN_LOCK_RELATIVE)
        self.fixture = self.a_fixtures.Fixture(self.root, stage="internal_evaluation")
        f = self.fixture
        self.stack.enter_context(f.hashes())
        cfg = c.load_config(self.root)
        self.stack.enter_context(patch.multiple(self.b, ROOT=self.root,
            CONFIG_SHA256=f.config_sha, PREREG_SHA256=f.prereg_sha,
            REFERENCE_MANIFEST_SHA256=f.digest(c.MANIFEST_RELATIVE),
            PARTITION_LOCK_SHA256=f.digest(c.PARTITION_RELATIVE)))
        self.stack.enter_context(patch.multiple(self.b_fixtures, ROOT=self.root, CFG=cfg, LOCK_SHA=self.train_sha))
        # Real metadata validation, real build_model and genuine CPU state_dicts.
        report = self.b_fixtures.synthetic_report()
        self.ck = self.b.checkpoint_metadata(cfg, report, self.train_sha)
        self.ck["models"] = {d: self.b.build_model(cfg, e.RATES[d]).state_dict() for d in e.DATASETS}
        self.assertEqual(set(self.ck), self.b.CHECKPOINT_KEYS)
        buffer = io.BytesIO()
        self.torch.save(self.ck, buffer)
        f.put(c.CHECKPOINT_RELATIVE, buffer.getvalue())
        f.put_json(c.TRAIN_REPORT_RELATIVE, report)
        for relative in (c.CHECKPOINT_RELATIVE, c.TRAIN_REPORT_RELATIVE):
            f.lock["files"][relative] = f.digest(relative)
        f.lock.update(training_performed=True, training_lock_sha256=self.train_sha,
                      checkpoint_sha256=f.digest(c.CHECKPOINT_RELATIVE))
        f.put_json(c.EVAL_LOCK_RELATIVE, f.lock)
        self.cfg = cfg
        self.synthetic_records = records()
        for record in self.synthetic_records:
            relative = f"{c.REFERENCE_DIR}/{record.dataset}/subject_{record.subject_id:03d}.npz"
            record.npz_sha256 = f.lock["files"][relative]
            record.sidecar_sha256 = f.lock["files"][relative+".manifest.json"]

    @contextmanager
    def physical_boundaries(self):
        transport = CpuCudaTransport(self.torch, self.root)
        original_import = importlib.import_module
        def imports(name, *args, **kwargs):
            return transport if name == "torch" else original_import(name, *args, **kwargs)
        original_to = self.torch.nn.Module.to
        moved = []
        def module_to(model, *args, **kwargs):
            if kwargs.get("device") == "cuda:0":
                assert not model.training and all(not p.requires_grad for p in model.parameters())
                kwargs = {**kwargs, "device": "cpu"}
                moved.append(model)
            return original_to(model, *args, **kwargs)
        legacy = original_import("eeg_tools.ssvep_denoise_v5")
        def memory_records(root, subjects):
            assert root == self.root and subjects == (5, 6)
            return self.synthetic_records
        with patch.object(e.importlib, "import_module", side_effect=imports), \
             patch.object(self.torch.nn.Module, "to", module_to), \
             patch.object(legacy, "load_reference_records", side_effect=memory_records) as load:
            yield transport, moved, load

    def test_real_ab_complete_4200_orchestration_math_and_publication(self):
        # Spies execute actual shared functions; they do not return fabricated results.
        real_bytes = c._read_bytes
        def guarded_bytes(root, relative):
            path = c.safe_path(root, relative)
            self.assertTrue(path.is_relative_to(self.root))
            if path.suffix == ".npz":
                self.assertIn(path.name, ("subject_005.npz", "subject_006.npz"))
            return real_bytes(root, relative)
        with self.physical_boundaries() as (transport, moved, load), \
             patch.object(c, "_read_bytes", side_effect=guarded_bytes), \
             patch.object(c, "summarize_cases", wraps=c.summarize_cases) as summarize, \
             patch.object(c, "gate_summary", wraps=c.gate_summary) as gate, \
             patch.object(c, "case_metrics", wraps=c.case_metrics) as metrics, \
             patch.object(self.b, "validate_checkpoint", wraps=self.b.validate_checkpoint) as validator:
            report = e.evaluate(self.root)
        self.assertEqual(report["case_count"], 4200)
        self.assertEqual(report["phase_case_count"], 840)
        self.assertTrue(report["complete_cohort"] and report["phase_verifiable"])
        # Fresh zero-init models are intentionally not claimed to pass the HF reduction gate.
        self.assertFalse(report["all_internal_gates_passed"])
        self.assertFalse(report["scope"]["runtime_enabled"] or report["scope"]["production_validated"])
        self.assertEqual(report["deployment"]["status"], "blocked")
        self.assertEqual(report["model_state_sha256_before"], report["model_state_sha256_after"])
        self.assertTrue(report["aggregate_policy"]["mixed_aggregates_descriptive_normalized_only"])
        self.assertFalse(report["aggregate_policy"]["native_raw_power_pooled"])
        self.assertEqual(len(report["preflight"]["verified_hashes"])+len(report["preflight"]["deferred"]), 42)
        self.assertEqual(transport.load_paths, [self.root/c.CHECKPOINT_RELATIVE])
        self.assertEqual(len(moved), 2)
        for model in moved:
            self.assertTrue(all(p.device.type == "cpu" and not p.requires_grad for p in model.parameters()))
        load.assert_called_once_with(self.root, subjects=(5, 6))
        validator.assert_called_once()
        self.assertEqual(metrics.call_count, 4200)
        self.assertEqual({call.kwargs["sample_rate_hz"] for call in metrics.call_args_list}, {250, 256})
        self.assertEqual(summarize.call_count, 42)
        mixed_calls = [call for call in summarize.call_args_list if call.kwargs.get("allow_mixed_datasets", False)]
        self.assertEqual(len(mixed_calls), 6)  # global plus five cross-domain conditions, no others.
        for call in summarize.call_args_list:
            if not call.kwargs.get("allow_mixed_datasets", False):
                self.assertEqual(len({r["dataset"] for r in call.args[0]}), 1)
        # B's real validator also checks 18 historical per-domain epoch gates.
        # The final two calls are C's diagnostic gates, never mixed/global gates.
        self.assertEqual(gate.call_count, len(e.DATASETS)*self.cfg["training"]["epochs"]+2)
        for call in gate.call_args_list[-2:]:
            self.assertIn(call.args[0], report["aggregates"]["dataset"].values())
        saved = json.loads((self.root/(e.DEFAULT_OUTPUT_PREFIX+".json")).read_text(encoding="utf-8"))
        self.assertEqual(saved, report)
        self.assertEqual(self.torch.cuda.is_initialized(), self.cuda_initialized_before)

    def test_real_a_check_only_42_bindings_no_binary_reads_or_model_import(self):
        real_bytes = c._read_bytes
        def no_binary(root, relative):
            self.assertNotIn(Path(relative).suffix, (".npz", ".pt"))
            return real_bytes(root, relative)
        with patch.object(c, "_read_bytes", side_effect=no_binary), \
             patch.object(e.importlib, "import_module", wraps=importlib.import_module) as imports:
            report = e.evaluate(self.root, check_only=True)
        self.assertFalse(any(call.args[0] in ("torch", "eeg_tools.ssvep_denoise_v6") for call in imports.call_args_list))
        self.assertEqual(report["case_count"], 0)
        self.assertEqual(report["phase_case_count"], 0)
        self.assertEqual(report["checks"]["checkpoint_schema"], "not_executed")
        self.assertEqual(len(report["preflight"]["verified_hashes"])+len(report["preflight"]["deferred"]), 42)
        self.assertEqual(len(report["preflight"]["deferred"]), 13)  # Twelve synthetic NPZs plus PT.
        self.assertTrue(report["scope"]["checkpoint_prior_training"])
        self.assertFalse(report["scope"]["training_performed_this_evaluation"])
        self.assertEqual(self.torch.cuda.is_initialized(), self.cuda_initialized_before)

    def test_fresh_process_check_only_never_imports_torch_or_reads_binary(self):
        # Use this test's synthetic 42-file tree, never the real workspace data.
        # Independent proof remains valid even if the parent suite initialized CUDA.
        code = r"""
import builtins
import json
from pathlib import Path
import sys
from eeg_tools import ssvep_v6_contract as c
from eeg_tools import ssvep_v6_internal_eval as e
assert 'torch' not in sys.modules
root = Path(sys.argv[1])
c.CONFIG_SHA256, c.PREREG_SHA256 = sys.argv[2:4]
original_import = builtins.__import__
def no_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('check-only must not import torch or initialize CUDA')
    return original_import(name, *args, **kwargs)
builtins.__import__ = no_torch
original_bytes = c._read_bytes
def no_binary(base, relative):
    path = c.safe_path(base, relative)
    assert path.is_relative_to(root)
    assert path.suffix not in ('.npz', '.pt'), 'check-only binary read forbidden'
    return original_bytes(base, relative)
c._read_bytes = no_binary
report = e.evaluate(root, check_only=True,
                    output_prefix='analysis_outputs/ssvep_v6_fresh_process_synthetic')
assert 'torch' not in sys.modules
assert 'eeg_tools.ssvep_denoise_v6' not in sys.modules
assert report['case_count'] == report['phase_case_count'] == 0
assert report['cuda_initialized'] is False
assert report['checkpoint_loaded'] is False
assert report['waveforms_loaded'] is False
assert report['checks']['checkpoint_schema'] == 'not_executed'
assert len(report['preflight']['verified_hashes']) + len(report['preflight']['deferred']) == 42
print(json.dumps({'torch_imported': False, 'cuda_initialized': False,
                  'binary_bytes_read': False, 'case_count': 0}))
"""
        process = subprocess.run(
            [sys.executable, "-B", "-c", code, str(self.root),
             self.fixture.config_sha, self.fixture.prereg_sha],
            cwd=e.ROOT, env={**e.os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        evidence = json.loads(process.stdout)
        self.assertEqual(evidence, {"torch_imported": False, "cuda_initialized": False,
                                    "binary_bytes_read": False, "case_count": 0})
        self.assertEqual(self.torch.cuda.is_initialized(), self.cuda_initialized_before)

    def test_real_b_cpu_models_metadata_keys_and_strict_validator(self):
        models = self.b.validate_checkpoint(self.ck, self.cfg, self.train_sha)
        self.assertEqual(set(models), set(e.DATASETS))
        for dataset, model in models.items():
            self.assertFalse(model.training)
            self.assertEqual(model.sample_rate_hz, e.RATES[dataset])
            self.assertTrue(all(p.device.type == "cpu" and p.dtype == self.torch.float32 for p in model.parameters()))
        invalid = dict(self.ck)
        invalid["production_validated"] = True
        with self.assertRaises(c.V6Error):
            self.b.validate_checkpoint(invalid, self.cfg, self.train_sha)
        invalid = dict(self.ck)
        invalid["unrecognized"] = True
        with self.assertRaises(c.V6Error):
            self.b.validate_checkpoint(invalid, self.cfg, self.train_sha)
        self.assertEqual(self.torch.cuda.is_initialized(), self.cuda_initialized_before)

    def test_real_a_mixed_opt_in_descriptive_and_default_refusal(self):
        rows = case_rows()
        with self.assertRaises(c.V6Error):
            c.summarize_cases(rows)
        mixed = c.summarize_cases(rows, allow_mixed_datasets=True)
        self.assertEqual(mixed["identity"]["case_count"], 840)
        result = e.summarize_diagnostic(rows, self.cfg)
        self.assertEqual(result["aggregates"]["global"], mixed)
        self.assertEqual(result["aggregates"]["condition"]["identity"], {"identity": mixed["identity"]})
        self.assertTrue(result["all_internal_gates_passed"])
        self.assertEqual(set(result["dataset_gates"]), set(e.DATASETS))

    def test_real_a_phase_undefined_stays_null_and_blocks(self):
        rows = case_rows()
        rows[0].update(protected_complex_energy_ratio=None, protected_weighted_abs_phase_radians=None)
        result = e.summarize_diagnostic(rows, self.cfg)
        self.assertFalse(result["phase_verifiable"] or result["all_internal_gates_passed"])
        self.assertEqual(result["undefined_phase_case_count"], 1)
        self.assertIsNone(result["aggregates"]["global"]["identity"]["mean_protected_weighted_abs_phase_radians"])
        self.assertFalse(result["dataset_gates"]["Nakanishi2015"]["all_passed"])
        self.assertTrue(result["dataset_gates"]["Wearable"]["all_passed"])

    def test_real_a_frozen_mutation_refused_before_checkpoint_load(self):
        relative = c.NEW_SOURCE_PATHS[0]
        state = c.preflight(self.root, stage="internal_evaluation", check_only=False)
        (self.root/relative).write_text("synthetic freeze tamper")
        with self.physical_boundaries() as (transport, _, load), self.assertRaises(c.V6Error):
            e.run_diagnostic(state)
        self.assertFalse(transport.load_paths)
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
