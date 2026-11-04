"""Synthetic-only tests: no real NPZ/checkpoint bytes, no CUDA initialization.

Fixture binaries are inert placeholder bytes in temporary directories, never
waveforms or checkpoint archives. Real project reads are config JSON only.
CPU tensor tests use tiny synthetic state dictionaries, not frozen weights.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from eeg_tools import ssvep_internal_benchmark_v5 as b
from scripts import benchmark_ssvep_internal_v5 as cli


def protocol():
    return {"schema_version": 1, "status": "formal", "experiment": "ssvep_v5_internal_synthetic_diagnostic",
            "subjects": list(b.SUBJECTS), "datasets": list(b.DATASETS), "conditions": list(b.CONDITIONS),
            "window_samples": b.WINDOW, "start_samples": dict(b.STARTS), "sample_rates_hz": dict(b.RATES),
            "seed": b.SEED, "seed_formula": b.SEED_FORMULA, "batch_size": 64, "device": "cuda:0",
            "normalization_epsilon": 1e-8, "inference": b.INFERENCE, "scope": dict(b.SCOPE),
            "metrics_definition": {"purpose": "synthetic test only"}, "expected_case_count": 4200,
            "expected_reference_trials": 840, "bound_files": {p: "a" * 64 for p in b.INPUT_PATHS}}


def case(dataset="Nakanishi2015", sid=5, trial=0, condition="transient", a=1.0, z=0.5):
    return {"dataset": dataset, "subject_id": sid, "trial_index": trial, "condition": condition,
            "seed": b.case_seed(dataset, sid, trial, condition), "window_start_sample": b.STARTS[dataset],
            "window_samples": 256, "sample_rate_hz": b.RATES[dataset], "input_normalized_mse": float(a),
            "output_normalized_mse": float(z), "relative_reduction": (a - z) / a if a > 0 and condition != "identity" else None,
            "identity_normalized_change": float(z) if condition == "identity" else None,
            "outcome": "win" if z < a else "loss" if z > a else "tie"}


class SyntheticFixture:
    """All writes isolated to TemporaryDirectory, never the project artifacts."""
    def __init__(self, root, config):
        self.root = root
        self.config = config
        for p in b.INPUT_PATHS + b.CODE_PATHS:
            self.put(p, b"inert synthetic placeholder, not waveform or weights")
        self.put_json(b.CONFIG_PATH, config)
        partition = {"schema_version": 1, "status": "formal", "stage": "reference_identity_freeze",
                     "partitions": {"train": [1, 2, 3, 4, 5, 6], "validation": [7, 8], "holdout": [9, 10]},
                     "raw_reference_is_clean_ground_truth": False, "runtime": {"enabled": False, "production_validated": False}}
        self.put_json(b.PARTITION_PATH, partition)
        rows = []
        for domain in b.DATASETS:
            for sid in b.SUBJECTS:
                p = f"{b.REFERENCE_DIR}/{domain}/subject_{sid:03d}.npz"
                digest = self.digest(p)
                side = {"schema_version": 1, "status": "formal", "reference_version": "ssvep_reference_v5",
                        "dataset": domain, "subject_id": sid, "split": "train", "reference_shape": list(b.SHAPES[domain]),
                        "reference_dtype": "float64", "native_sample_rate_hz": b.RATES[domain], "onset_sample_zero_based": b.STARTS[domain],
                        "source_lock_file_sha256": self.digest(b.PARTITION_PATH), "raw_reference_is_clean_ground_truth": False,
                        "runtime_enabled": False, "npz_sha256": digest, "materialized_trials": b.SHAPES[domain][0],
                        "source": {"path": "D:\\SSVEP\\PROHIBITED.mat"}}
                self.put_json(p + ".manifest.json", side)
                rows.append({"dataset": domain, "subject_id": sid, "split": "train", "relative_path": f"{domain}/subject_{sid:03d}.npz",
                             "trials": b.SHAPES[domain][0], "npz_sha256": digest, "manifest_sha256": self.digest(p + ".manifest.json")})
        # Existing outer metadata must NEVER cause path/stat/sidecar/byte access.
        rows.append({"dataset": "Wearable", "subject_id": 7, "split": "validation", "relative_path": "D:/DO_NOT_ACCESS.npz"})
        manifest = {"schema_version": 1, "status": "formal", "reference_version": "ssvep_reference_v5", "build_completed": True,
                    "source_lock_file_sha256": self.digest(b.PARTITION_PATH), "raw_reference_is_clean_ground_truth": False,
                    "runtime": {"enabled": False, "production_validated": False}, "materialized_subjects": rows,
                    "source_root": "D:\\SSVEP", "holdout_sources": [{"subject_id": 9, "path": "D:\\NEVER.mat"}]}
        self.put_json(b.MANIFEST_PATH, manifest)
        self.proto = protocol()
        self.proto["bound_files"] = {p: self.digest(p) for p in b.INPUT_PATHS}
        self.put_json(b.PROTOCOL_PATH, self.proto)
        self.lock = {"schema_version": 1, "status": "formal", "stage": "frozen_before_internal_waveform_read",
                     "expected_case_count": 4200, "scope": dict(b.SCOPE),
                     "files": {p: self.digest(p) for p in b.FROZEN_PATHS}}
        self.put_json(b.LOCK_PATH, self.lock)

    def put(self, p, contents):
        target = self.root / p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)

    def put_json(self, p, contents):
        self.put(p, json.dumps(contents, allow_nan=False).encode())

    def digest(self, p):
        return hashlib.sha256((self.root / p).read_bytes()).hexdigest()

    def patches(self):
        return (patch.object(b, "PROTOCOL_SHA256", self.digest(b.PROTOCOL_PATH)),
                patch.object(b.core, "REFERENCE_MANIFEST_SHA256", self.digest(b.MANIFEST_PATH)),
                patch.object(b.core, "PARTITION_LOCK_SHA256", self.digest(b.PARTITION_PATH)))


class MathAndSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = b.core.load_config(root=b.ROOT)  # JSON only.

    def setUp(self):
        self.reference = np.random.default_rng(71).normal(size=(8, 256)).astype(np.float64)

    def test_seed_all_4200_unique_and_exact(self):
        seeds = {b.case_seed(d, s, t, c) for d in b.DATASETS for s in b.SUBJECTS for t in range(b.SHAPES[d][0]) for c in b.CONDITIONS}
        self.assertEqual(len(seeds), 4200)
        self.assertEqual(b.case_seed("Nakanishi2015", 5, 0, "identity"), 2026600701)
        self.assertEqual(b.case_seed("Wearable", 6, 239, "common_mode"), b.SEED + 10000000 + 600000 + 2390 + 4)
        for args in (("Wearable", 7, 0, "identity"), ("Wearable", 9, 0, "identity"), ("other", 5, 0, "identity"),
                     ("Wearable", True, 0, "identity"), ("Nakanishi2015", 5, 180, "identity"),
                     ("Wearable", 5, -1, "identity"), ("Wearable", 5, 0.0, "identity"), ("Wearable", 5, 0, "unknown")):
            with self.subTest(args=args), self.assertRaises(b.InternalBenchmarkError):
                b.case_seed(*args)

    def test_identity_is_independent_copy_and_does_not_call_corrupt(self):
        with patch.object(b.core, "corrupt_window", side_effect=AssertionError("identity generator must not run")):
            result, _ = b.synthetic_corruption(self.reference, "Wearable", 5, 0, "identity", self.cfg)
        np.testing.assert_array_equal(result, self.reference)
        self.assertFalse(np.shares_memory(result, self.reference))

    def test_each_family_legacy_deterministic_override_no_config_mutation(self):
        original = deepcopy(self.cfg)
        for condition in b.CONDITIONS[1:]:
            with self.subTest(condition=condition):
                result, seed = b.synthetic_corruption(self.reference, "Wearable", 6, 4, condition, self.cfg)
                again, again_seed = b.synthetic_corruption(self.reference, "Wearable", 6, 4, condition, self.cfg)
                override = deepcopy(original)
                override["corruption"].update(families=[condition], identity_probability=0, combine_probability=0)
                expected, identity, families = b.core.corrupt_window(self.reference, 250, override, seed)
                np.testing.assert_array_equal(result, again)
                np.testing.assert_array_equal(result, expected)
                self.assertEqual(seed, again_seed)
                self.assertFalse(identity)
                self.assertEqual(families, [condition])
        self.assertEqual(self.cfg, original)

    def test_metric_channel_denominator_and_unitless_aggregation(self):
        ref = self.reference * np.arange(1, 9)[:, None]
        noisy, out = ref + 2.0, ref + 1.0
        metric = b.case_metrics(ref, noisy, out)
        denominator = ((ref - ref.mean(axis=1, keepdims=True)) ** 2).mean(axis=1, keepdims=True) + 1e-8
        self.assertAlmostEqual(metric["input_normalized_mse"], float(np.mean(4.0 / denominator)))
        self.assertAlmostEqual(metric["output_normalized_mse"], float(np.mean(1.0 / denominator)))
        self.assertAlmostEqual(metric["relative_reduction"], 0.75)

    def test_identity_zero_input_null_reduction_normalized_change(self):
        metric = b.case_metrics(self.reference, self.reference.copy(), self.reference + 1.0, identity=True)
        self.assertEqual(metric["input_normalized_mse"], 0)
        self.assertIsNone(metric["relative_reduction"])
        self.assertEqual(metric["identity_normalized_change"], metric["output_normalized_mse"])
        self.assertEqual(metric["outcome"], "loss")
        with self.assertRaises(b.InternalBenchmarkError):
            b.case_metrics(self.reference, self.reference + 1, self.reference, identity=True)
        zero = b.case_metrics(np.zeros((8, 256)), np.zeros((8, 256)), np.zeros((8, 256)), identity=True)
        self.assertEqual(zero["output_normalized_mse"], 0.0)
        self.assertEqual(zero["outcome"], "tie")

    def test_restore_preserves_noisy_float64_baseline_without_roundtrip(self):
        noisy = self.reference + 1e7
        normalized, mean, rms = b.core.normalize_window(noisy)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertEqual(mean.dtype, np.float64)
        self.assertEqual(rms.dtype, np.float64)
        np.testing.assert_array_equal(b.restore_window(noisy, normalized, normalized.copy(), rms), noisy)
        prediction = normalized + np.float32(0.125)
        restored = b.restore_window(noisy, normalized, prediction, rms)
        expected = noisy + (prediction.astype(np.float64) - normalized.astype(np.float64)) * rms
        np.testing.assert_array_equal(restored, expected)
        self.assertEqual(restored.dtype, np.float64)

    def test_nonfinite_shape_dtype_and_overflow_fail_closed(self):
        for bad in (np.zeros((7, 256)), np.zeros((8, 255)), np.full((8, 256), np.nan), np.full((8, 256), np.inf), np.zeros((8, 256), dtype=int)):
            with self.subTest(shape=bad.shape), self.assertRaises(b.InternalBenchmarkError):
                b.case_metrics(bad, self.reference, self.reference)
        with self.assertRaises(b.InternalBenchmarkError):
            b.case_metrics(np.full((8, 256), 1e308), np.full((8, 256), -1e308), self.reference)
        n, _, rms = b.core.normalize_window(self.reference)
        for bad in (np.zeros((8,)), np.full((8, 1), -1.0), np.full((8, 1), np.nan), rms.astype(np.float32)):
            with self.assertRaises(b.InternalBenchmarkError):
                b.restore_window(self.reference, n, n, bad)
        with self.assertRaises(b.InternalBenchmarkError):
            b.restore_window(self.reference, n, n.astype(np.float64), rms)

    def test_ratio_of_means_not_mean_reductions_identity_null(self):
        rows = [case(a=1.0, z=0.0), case(trial=1, a=9.0, z=9.0), case(condition="identity", a=0.0, z=0.2)]
        result = b.aggregate_cases(rows, require_complete=False)
        self.assertAlmostEqual(result["condition"]["transient"]["ratio_of_mean_reduction"], 0.1)
        self.assertIsNone(result["condition"]["identity"]["ratio_of_mean_reduction"])
        self.assertEqual(result["condition"]["identity"]["mean_identity_normalized_change"], 0.2)
        self.assertEqual(result["global"]["case_count"], 3)
        self.assertEqual(result["global"]["win_count"], 1)
        self.assertIn("Nakanishi2015:subject_005", result["subject"])

    def test_complete_cohort_840_trials_4200_cases_and_duplicate_rejection(self):
        rows = [case(d, s, t, c, a=0.0 if c == "identity" else 1.0, z=0.0)
                for d in b.DATASETS for s in b.SUBJECTS for t in range(b.SHAPES[d][0]) for c in b.CONDITIONS]
        aggregates = b.aggregate_cases(rows)
        self.assertEqual(aggregates["global"]["case_count"], 4200)
        self.assertEqual(aggregates["condition"]["identity"]["case_count"], 840)
        for bad in (rows[:-1], rows + [rows[0]], [rows[0], rows[0]]):
            with self.assertRaises(b.InternalBenchmarkError):
                b.aggregate_cases(bad)
        for mutation in ({"target_id": 0}, {"predictions": []}, {"subject_id": 7}, {"seed": 9}, {"output_normalized_mse": float("inf")}):
            with self.assertRaises(b.InternalBenchmarkError):
                b.aggregate_cases([{**rows[0], **mutation}], require_complete=False)

    def test_case_reduction_and_identity_change_types_are_strict(self):
        for row in ({**case(a=1.0, z=0.0), "relative_reduction": True},
                    {**case(condition="identity", a=0.0, z=1.0), "identity_normalized_change": True}):
            with self.assertRaises(b.InternalBenchmarkError):
                b.aggregate_cases([row], require_complete=False)

    def test_protocol_strict_values_types_all_required_fields(self):
        p = protocol()
        p["bound_files"][b.MANIFEST_PATH] = b.core.REFERENCE_MANIFEST_SHA256
        p["bound_files"][b.PARTITION_PATH] = b.core.PARTITION_LOCK_SHA256
        p["unknown_descriptive_metadata"] = "explicitly permitted"
        b.validate_preregistration(p)
        for key in ("subjects", "datasets", "conditions", "window_samples", "start_samples", "sample_rates_hz", "seed", "seed_formula", "device", "batch_size", "scope", "metrics_definition", "bound_files"):
            bad = deepcopy(p)
            del bad[key]
            with self.subTest(key=key), self.assertRaises(b.InternalBenchmarkError):
                b.validate_preregistration(bad)
        for mutation in ({"subjects": [5, 7]}, {"conditions": list(reversed(b.CONDITIONS))}, {"schema_version": True}, {"window_samples": 256.0},
                         {"seed": 20261007}, {"device": "cpu"}, {"normalization_epsilon": 1e-12}, {"expected_case_count": 4199}):
            with self.subTest(mutation=mutation), self.assertRaises(b.InternalBenchmarkError):
                b.validate_preregistration({**p, **mutation})
        for key in b.SCOPE:
            bad = deepcopy(p)
            bad["scope"][key] = True
            with self.assertRaises(b.InternalBenchmarkError):
                b.validate_preregistration(bad)
        bad = deepcopy(p)
        bad["bound_files"]["D:/other.npz"] = "f" * 64
        with self.assertRaises(b.InternalBenchmarkError):
            b.validate_preregistration(bad)

    def test_lock_full_binding_schema_and_flags(self):
        p = protocol()
        lock = {"schema_version": 1, "status": "formal", "stage": "frozen_before_internal_waveform_read", "expected_case_count": 4200,
                "files": {**p["bound_files"], b.PROTOCOL_PATH: b.PROTOCOL_SHA256, **{x: "b" * 64 for x in b.CODE_PATHS}}}
        b.validate_lock(lock, p, b.PROTOCOL_SHA256)
        for mutation in ({"stage": "after_waveforms"}, {"status": "draft"}, {"training_performed": True}, {"runtime_enabled": 0}, {"expected_case_count": 0}, {"unknown_nested_flags": {"allowed": True}}):
            with self.assertRaises(b.InternalBenchmarkError):
                b.validate_lock({**lock, **mutation}, p, b.PROTOCOL_SHA256)
        for target in b.FROZEN_PATHS:
            bad = deepcopy(lock)
            del bad["files"][target]
            with self.assertRaises(b.InternalBenchmarkError):
                b.validate_lock(bad, p, b.PROTOCOL_SHA256)
        bad = deepcopy(lock)
        bad["files"][b.CHECKPOINT_PATH] = "c" * 64
        with self.assertRaises(b.InternalBenchmarkError):
            b.validate_lock(bad, p, b.PROTOCOL_SHA256)

class PathAndPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = b.core.load_config(root=b.ROOT)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-internal-v5-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = SyntheticFixture(self.root, self.cfg)
        for context in self.fixture.patches():
            self.enterContext(context)

    def test_check_only_never_reads_binary_or_initializes_cuda(self):
        original_open = Path.open
        opened = []
        def guard(path, *args, **kwargs):
            relative = path.relative_to(self.root).as_posix()
            self.assertNotIn(relative, b.BINARY_PATHS)
            self.assertNotIn("subject_007", relative)
            self.assertNotIn("subject_008", relative)
            opened.append(relative)
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", guard), patch.object(b.core, "load_reference_records", side_effect=AssertionError("waveform loader called")), \
             patch.object(b, "cuda_preflight", side_effect=AssertionError("CUDA called")), patch.object(b, "validate_checkpoint", side_effect=AssertionError("weights called")):
            state = b.preflight(self.root, check_only=True)
        self.assertEqual(set(state.deferred_hashes), set(b.BINARY_PATHS))
        self.assertEqual(len(state.selected), 4)
        self.assertEqual(len(state.verified_hashes), 13)
        self.assertTrue(opened)
        report = state.report()
        self.assertFalse(report["inference_performed"])
        self.assertFalse(report["waveforms_loaded"])
        self.assertFalse(report["checkpoint_loaded"])
        self.assertIn("NOT EXECUTED", b.render_markdown(report))

    def test_full_hashes_only_selected_binaries_without_loader_or_cuda(self):
        original_open = Path.open
        binaries = []
        def guard(path, *args, **kwargs):
            rel = path.relative_to(self.root).as_posix()
            self.assertIn(rel, set(b.FROZEN_PATHS) | {b.LOCK_PATH})
            if path.suffix in (".npz", ".pt"):
                self.assertIn(rel, b.BINARY_PATHS)
                binaries.append(rel)
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", guard), patch.object(b.core, "load_reference_records", side_effect=AssertionError("loader called")), \
             patch.object(b, "cuda_preflight", side_effect=AssertionError("CUDA called")):
            state = b.preflight(self.root, check_only=False)
        self.assertEqual(set(binaries), set(b.BINARY_PATHS))
        self.assertFalse(state.deferred_hashes)
        self.assertEqual(len(state.verified_hashes), 18)

    def test_check_only_defers_binary_tampering_full_rejects_it(self):
        self.fixture.put(b.NPZ_PATHS[0], b"different inert synthetic bytes")
        b.preflight(self.root, check_only=True)
        with self.assertRaisesRegex(b.InternalBenchmarkError, "SHA mismatch"):
            b.preflight(self.root, check_only=False)
        self.fixture.put(b.CHECKPOINT_PATH, b"also changed")
        with self.assertRaises(b.InternalBenchmarkError):
            b.preflight(self.root, check_only=False)

    def test_freeze_hash_code_tampering_and_lock_changes_rejected(self):
        state = b.preflight(self.root, check_only=True)
        self.fixture.put_json(b.LOCK_PATH, {**self.fixture.lock, "description": "changed freeze"})
        with self.assertRaisesRegex(b.InternalBenchmarkError, "lock changed"):
            b._verify_frozen(state)
        self.fixture.put(b.CODE_PATHS[0], b"modified reviewed code")
        with self.assertRaisesRegex(b.InternalBenchmarkError, "SHA mismatch"):
            b.preflight(self.root, check_only=True)

    def test_missing_lock_and_protocol_digest_fail_before_cuda_or_loader(self):
        (self.root / b.LOCK_PATH).unlink()
        with patch.object(b, "cuda_preflight") as gpu, patch.object(b.core, "load_reference_records") as loader:
            with self.assertRaises(OSError):
                b.run_benchmark(self.root)
            gpu.assert_not_called()
            loader.assert_not_called()
        self.fixture.put_json(b.LOCK_PATH, self.fixture.lock)
        with patch.object(b, "PROTOCOL_SHA256", "0" * 64), self.assertRaisesRegex(b.InternalBenchmarkError, "preregistration SHA"):
            b.preflight(self.root)

    def test_sidecar_shape_dtype_subject_runtime_flags_rejected(self):
        p = b.SIDECAR_PATHS[0]
        side = json.loads((self.root / p).read_text())
        for mutation in ({"reference_shape": [180, 256, 8]}, {"subject_id": 7}, {"runtime_enabled": True},
                         {"native_sample_rate_hz": 250}, {"raw_reference_is_clean_ground_truth": True}, {"reference_dtype": "float32"}):
            self.fixture.put_json(p, {**side, **mutation})
            # Direct catalog checks exercise schema independently of hash failure.
            with self.subTest(mutation=mutation), self.assertRaises(b.InternalBenchmarkError):
                b._catalog(self.root, self.fixture.proto)
        self.fixture.put_json(p, side)
        b._catalog(self.root, self.fixture.proto)

    def test_duplicate_catalog_identity_and_illegal_subject_fail(self):
        manifest = json.loads((self.root / b.MANIFEST_PATH).read_text())
        for row in (manifest["materialized_subjects"][0], {"dataset": "Wearable", "subject_id": 9}):
            modified = deepcopy(manifest)
            modified["materialized_subjects"].append(row)
            self.fixture.put_json(b.MANIFEST_PATH, modified)
            with self.assertRaises(b.InternalBenchmarkError):
                b._catalog(self.root, self.fixture.proto)

    def test_json_duplicate_and_nonfinite_rejected(self):
        for contents in (b'{"a": 1, "a": 2}', b'{"a": NaN}', b'{"a": Infinity}'):
            self.fixture.put(b.LOCK_PATH, contents)
            with self.assertRaises(b.InternalBenchmarkError):
                b.read_json(self.root, b.LOCK_PATH)

    def test_paths_forbid_traversal_external_foreign_drive_and_ads_before_stat(self):
        bad = ("../outside", "analysis_outputs/../models", "D:/SSVEP/test.npz", "D:\\SSVEP\\test.npz", "E:\\other", "C:relative", "//server/share/evil", "analysis_outputs/report:ads", str(self.root.parent / "external"))
        for value in bad:
            with self.subTest(value=value), patch.object(Path, "lstat", side_effect=AssertionError("unsafe path stat")), self.assertRaises(b.InternalBenchmarkError):
                b.safe_path(self.root, value)
        for value in ("models", "configs"):
            with self.assertRaises(b.InternalBenchmarkError):
                b.output_directory(self.root, value)
        self.assertEqual(b.output_directory(self.root, "analysis_outputs/new"), self.root / "analysis_outputs/new")
        self.assertEqual(b.safe_path(self.root, self.root / "analysis_outputs"), self.root / "analysis_outputs")

    def test_symlink_and_reparse_ancestors_refused(self):
        link = self.root / "analysis_outputs" / "alias"
        try:
            link.symlink_to(self.root / "models", target_is_directory=True)
        except OSError:
            pass  # Windows may lack symlink privilege; reparse branch tested below.
        else:
            with self.assertRaisesRegex(b.InternalBenchmarkError, "symlink/reparse"):
                b.output_directory(self.root, "analysis_outputs/alias/new")
            link.unlink()
        original = Path.lstat
        def reparse(path):
            info = original(path)
            if path == self.root / "analysis_outputs":
                return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
            return info
        with patch.object(Path, "lstat", reparse), self.assertRaisesRegex(b.InternalBenchmarkError, "symlink/reparse"):
            b.output_directory(self.root, "analysis_outputs/new")

    def test_atomic_no_clobber_both_reports_and_nonfinite_report(self):
        report = b.preflight(self.root).report()
        paths = b.publish_reports(self.root, "analysis_outputs/reports", report, check_only=True)
        before = [p.read_bytes() for p in paths]
        self.assertEqual(json.loads(before[0])["case_count"], 0)
        self.assertIn(b"NO Hann OLA", before[1])
        with self.assertRaises(b.InternalBenchmarkError):
            b.publish_reports(self.root, "analysis_outputs/reports", report, check_only=True)
        self.assertEqual(before, [p.read_bytes() for p in paths])
        with self.assertRaises(b.InternalBenchmarkError):
            b.publish_reports(self.root, "analysis_outputs/other", {**report, "bad": float("nan")}, check_only=True)
        self.assertFalse((self.root / "analysis_outputs/other").exists())
        self.assertFalse(list((self.root / "analysis_outputs/reports").glob("*.pending")))

    def test_atomic_competing_second_writer_preserved_first_rolled_back(self):
        report = b.preflight(self.root).report()
        paths = b.report_paths(self.root, "analysis_outputs/race", check_only=True)
        real_link = b.os.link
        def competing_link(source, destination):
            if Path(destination).suffix == ".md":
                Path(destination).write_text("competing artifact", encoding="utf-8")
            return real_link(source, destination)
        with patch.object(b.os, "link", side_effect=competing_link), self.assertRaises(FileExistsError):
            b.publish_reports(self.root, "analysis_outputs/race", report, check_only=True)
        self.assertFalse(paths[0].exists())
        self.assertEqual(paths[1].read_text(), "competing artifact")
        self.assertFalse(list(paths[0].parent.glob("*.pending")))

    def test_cli_check_only_and_no_redirect_flags_or_output_access(self):
        with patch.object(cli, "ROOT", self.root), patch.object(b, "run_benchmark", side_effect=AssertionError("inference called")), patch("builtins.print"):
            self.assertEqual(cli.main(["--check-only", "--output-dir", "analysis_outputs/cli"]), 0)
        for option in ("--reference-dir", "--config", "--checkpoint", "--device", "--subject", "--alpha", "--check"):
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                cli.main([option, "arbitrary"])
        with patch.object(cli, "ROOT", self.root), patch.object(b, "preflight") as preflight, patch("sys.stderr"), self.assertRaises(SystemExit):
            cli.main(["--check-only", "--output-dir", "../escape"])
        preflight.assert_not_called()


@unittest.skipIf(b.core.torch is None, "PyTorch unavailable; synthetic CPU tensor validation requires it")
class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = b.core.load_config(root=b.ROOT)

    def setUp(self):
        torch = b.core.torch
        # Deliberately tiny CPU-only model to test schema/key/dtype/load contract.
        self.enterContext(patch.object(b.core, "build_model", side_effect=lambda cfg: torch.nn.Conv1d(8, 8, 1)))
        report = {"schema_version": 1, "status": "formal", "model_version": b.core.MODEL_VERSION,
                  "reference_manifest_sha256": b.core.REFERENCE_MANIFEST_SHA256, "partition_lock_sha256": b.core.PARTITION_LOCK_SHA256,
                  "holdout_waveforms_loaded": False, "outer_validation_used_for_selection": False,
                  "runtime_enabled": False, "production_validated": False, "amp": False}
        self.ck = b.core.checkpoint_metadata(self.cfg, report)
        self.ck["models"] = {d: torch.nn.Conv1d(8, 8, 1).state_dict() for d in b.DATASETS}

    def test_complete_checkpoint_cpu_eval_no_grad(self):
        models = b.validate_checkpoint(self.ck, self.cfg)
        self.assertEqual(set(models), set(b.DATASETS))
        for model in models.values():
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad and p.device.type == "cpu" for p in model.parameters()))

    def test_checkpoint_keys_schema_domain_runtime_config_flags_rejected(self):
        for key in self.ck:
            bad = deepcopy(self.ck)
            del bad[key]
            with self.subTest(key=key), self.assertRaises(b.InternalBenchmarkError):
                b.validate_checkpoint(bad, self.cfg)
        for key, value in (("checkpoint_schema_version", 2), ("runtime", {"enabled": True}), ("holdout_waveforms_loaded", True),
                           ("clean_ground_truth_used", True), ("outer_validation_used_for_selection", True), ("models", {}), ("unexpected", False)):
            with self.subTest(key=key), self.assertRaises(b.InternalBenchmarkError):
                b.validate_checkpoint({**self.ck, key: value}, self.cfg)
        for mutation in ({"runtime_enabled": True}, {"production_validated": True}, {"amp": True}):
            bad = deepcopy(self.ck)
            bad["training_report"].update(mutation)
            with self.assertRaises(b.InternalBenchmarkError):
                b.validate_checkpoint(bad, self.cfg)
        bad = deepcopy(self.ck)
        bad["config"]["corruption"]["identity_probability"] = 0
        with self.assertRaises(b.InternalBenchmarkError):
            b.validate_checkpoint(bad, self.cfg)

    def test_state_keys_shapes_dtypes_nonfinite_fail_closed(self):
        torch = b.core.torch
        original = self.ck["models"]["Wearable"]
        key = next(iter(original))
        for replacement in (torch.zeros((1,)), original[key].double(), torch.full_like(original[key], float("nan")),
                            torch.full_like(original[key], float("inf")), original[key].clone().requires_grad_(True), "not a tensor"):
            bad = deepcopy(self.ck)
            bad["models"]["Wearable"][key] = replacement
            with self.assertRaises(b.InternalBenchmarkError):
                b.validate_checkpoint(bad, self.cfg)
        for mutation in ("missing", "extra"):
            bad = deepcopy(self.ck)
            if mutation == "missing":
                del bad["models"]["Wearable"][key]
            else:
                bad["models"]["Wearable"]["unregistered"] = torch.zeros(1)
            with self.assertRaises(b.InternalBenchmarkError):
                b.validate_checkpoint(bad, self.cfg)

    def test_no_cuda_fallback_and_check_only_never_touches_torch_load_cuda(self):
        with patch.object(b.core.torch.cuda, "is_available", return_value=False), patch.object(b.core.torch.cuda, "set_device") as set_device:
            with self.assertRaisesRegex(b.InternalBenchmarkError, "CPU fallback refused"):
                b.cuda_preflight()
            set_device.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            fixture = SyntheticFixture(Path(tmp), self.cfg)
            from contextlib import ExitStack
            with ExitStack() as stack:
                for context in fixture.patches():
                    stack.enter_context(context)
                stack.enter_context(patch.object(b.core.torch, "load", side_effect=AssertionError("checkpoint bytes read")))
                stack.enter_context(patch.object(b.core.torch.cuda, "is_available", side_effect=AssertionError("CUDA queried")))
                b.preflight(Path(tmp), check_only=True)

class SyntheticOrchestrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = b.core.load_config(root=b.ROOT)

    def test_full_fixed_cohort_inference_mode_batches_and_cpu_weights_load(self):
        calls, events = [], []
        in_inference = False
        class InferenceContext:
            def __enter__(self):
                nonlocal in_inference
                in_inference = True
            def __exit__(self, *args):
                nonlocal in_inference
                in_inference = False
        class Tensor:
            def __init__(self, array):
                self.array = array
                self.dtype = array.dtype
                self.shape = array.shape
            def to(self, **kwargs):
                self_dtype = np.dtype(kwargs["dtype"])
                if kwargs["device"] != "cuda:0" or self_dtype != np.float32:
                    raise AssertionError("wrong device/dtype")
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.array
        class Model:
            training = False
            def __init__(self, domain):
                self.domain = domain
            def to(self, **kwargs):
                if kwargs != {"device": "cuda:0", "dtype": np.float32}:
                    raise AssertionError("wrong model device")
                return self
            def parameters(self):
                return [SimpleNamespace(requires_grad=False)]
            def __call__(self, inputs):
                if not in_inference:
                    raise AssertionError("forward outside inference_mode")
                if not 0 < inputs.shape[0] <= 64 or inputs.shape[1:] != (8, 256):
                    raise AssertionError("wrong batch")
                calls.append((self.domain, inputs.shape[0]))
                return Tensor(inputs.array.copy())  # Exact zero residual diagnostic.
        def fake_load(path, **kwargs):
            self.assertEqual(Path(path), b.ROOT / b.CHECKPOINT_PATH)
            self.assertEqual(kwargs, {"map_location": "cpu", "weights_only": True})
            events.append("CPU_weights")
            return {"synthetic": True}
        def fake_loader(root, *, subjects):
            self.assertEqual(root, b.ROOT)
            self.assertEqual(subjects, (5, 6))
            self.assertEqual(events, ["CUDA_preflight", "verify", "CPU_weights", "verify"])
            events.append("internal_loader")
            return records
        def fake_cuda():
            events.append("CUDA_preflight")
            return {"device": "cuda:0", "mocked": True}
        def verify(state):
            events.append("verify")
        selected, records = [], []
        for d in b.DATASETS:
            for sid in b.SUBJECTS:
                row = {"dataset": d, "subject_id": sid, "npz_sha256": "a" * 64, "sidecar_sha256": "b" * 64}
                selected.append(row)
                signal = np.random.default_rng(sid).normal(size=b.SHAPES[d]).astype(np.float64)
                records.append(b.core.ReferenceRecord(d, sid, b.RATES[d], signal, "a" * 64, "b" * 64))
        state = SimpleNamespace(root=b.ROOT, deferred_hashes=(), config=self.cfg, selected=selected,
                                protocol={"experiment": "ssvep_v5_internal_synthetic_diagnostic", "metrics_definition": "synthetic"},
                                lock={"files": {}}, protocol_sha256="a" * 64, lock_sha256="b" * 64)
        fake_torch = SimpleNamespace(load=fake_load, float32=np.float32, from_numpy=Tensor,
                                     isfinite=lambda tensor: np.isfinite(tensor.array), inference_mode=InferenceContext)
        with patch.object(b, "preflight", return_value=state) as preflight, patch.object(b, "cuda_preflight", side_effect=fake_cuda), \
             patch.object(b, "_verify_frozen", side_effect=verify), patch.object(b.core, "torch", fake_torch), \
             patch.object(b, "validate_checkpoint", return_value={d: Model(d) for d in b.DATASETS}), \
             patch.object(b.core, "load_reference_records", side_effect=fake_loader):
            report = b.run_benchmark(b.ROOT)
        preflight.assert_called_once_with(b.ROOT, check_only=False)
        self.assertEqual(report["case_count"], 4200)
        self.assertEqual(report["reference_trial_count"], 840)
        self.assertEqual(report["aggregates"]["global"]["mean_output_normalized_mse"], report["aggregates"]["global"]["mean_input_normalized_mse"])
        self.assertEqual(report["aggregates"]["global"]["tie_count"], 4200)
        self.assertEqual(sum(n for _, n in calls), 4200)
        for domain in b.DATASETS:
            trials = b.SHAPES[domain][0]
            batch_sizes = [size for d, size in calls if d == domain]
            self.assertEqual(batch_sizes, ([64] * ((trials * 5) // 64) + [(trials * 5) % 64]) * 2)
        self.assertEqual(events[-1], "verify")
        self.assertFalse(in_inference)
        self.assertTrue(report["inference_performed"])
        self.assertEqual(report["scope"], b.SCOPE)
        self.assertIn("NO Hann OLA", b.render_markdown(report))
        self.assertTrue(all(set(row) == set(case()) for row in report["cases"]))

    def test_loader_not_reached_after_preflight_or_gpu_failure(self):
        with patch.object(b, "preflight", side_effect=b.InternalBenchmarkError("not frozen")), \
             patch.object(b, "cuda_preflight") as cuda, patch.object(b.core, "load_reference_records") as loader:
            with self.assertRaises(b.InternalBenchmarkError):
                b.run_benchmark()
            cuda.assert_not_called()
            loader.assert_not_called()
        with patch.object(b, "preflight", return_value=SimpleNamespace(deferred_hashes=())), \
             patch.object(b, "cuda_preflight", side_effect=b.InternalBenchmarkError("no CUDA")), \
             patch.object(b.core, "load_reference_records") as loader:
            with self.assertRaises(b.InternalBenchmarkError):
                b.run_benchmark()
            loader.assert_not_called()

    def test_illegal_duplicate_nonfinite_records_rejected(self):
        selected, records = [], []
        for d in b.DATASETS:
            for sid in b.SUBJECTS:
                selected.append({"dataset": d, "subject_id": sid, "npz_sha256": "a", "sidecar_sha256": "b"})
                records.append(b.core.ReferenceRecord(d, sid, b.RATES[d], np.zeros(b.SHAPES[d], dtype=np.float64), "a", "b"))
        state = SimpleNamespace(selected=selected)
        self.assertEqual(len(b._validate_records(records, state)), 4)
        for bad in (records[:-1], [records[0]] * 4):
            with self.assertRaises(b.InternalBenchmarkError):
                b._validate_records(bad, state)
        original = records[0]
        for mutation in ({"subject_id": 7}, {"signal": np.zeros((180, 256, 8))}, {"signal": np.full(b.SHAPES[original.dataset], np.inf)},
                         {"signal": original.signal.astype(np.float32)}, {"sample_rate_hz": 250}, {"npz_sha256": "bad"}):
            values = dict(vars(original))
            values.update(mutation)
            with self.assertRaises(b.InternalBenchmarkError):
                b._validate_records([b.core.ReferenceRecord(**values)] + records[1:], state)


if __name__ == "__main__":
    unittest.main()
