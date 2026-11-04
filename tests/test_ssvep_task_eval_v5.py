"""Offline synthetic-only tests for the v5 task-preservation contract."""
from __future__ import annotations

from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from eeg_tools import ssvep_task_eval_v5 as task
from scripts import evaluate_ssvep_task_v5 as cli


class TaskEvaluationV5Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ssvep_task_v5_")
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name).resolve()
        self.reference = self.workspace / task.REFERENCE_RELATIVE
        self.reference.mkdir(parents=True)
        self.subjects = {}
        self.rows = []
        for dataset, sid in [("Nakanishi2015", 1), ("Nakanishi2015", 7), ("Nakanishi2015", 8), ("Wearable", 7)]:
            spec = task.reference_safety.SPECS[dataset]
            target = np.repeat(np.arange(3, dtype=np.int64), 2)
            timepoints = 1114 if dataset == "Nakanishi2015" else 710
            samples = np.arange(timepoints) / spec["sample_rate_hz"]
            signal = np.empty((len(target), timepoints, 2), dtype=np.float64)
            for index, label in enumerate(target):
                frequency = spec["frequency_order_hz"][label]
                wave = np.sin(2 * np.pi * frequency * samples) + .2 * np.sin(4 * np.pi * frequency * samples)
                signal[index] = 100. + wave[:, None] * np.array([1., .6])[None, :]
            arrays = {"reference": signal, "trial_index": np.arange(len(target), dtype=np.int64),
                "target_id": target, "block_id": np.tile(np.arange(2, dtype=np.int64), 3),
                "electrode_id": np.full(len(target), -1, dtype=np.int64) if dataset == "Nakanishi2015" else np.tile(np.arange(2, dtype=np.int64), 3),
                "frequency_hz": np.array(spec["frequency_order_hz"], dtype=np.float64)[target]}
            if dataset == "Wearable":
                arrays["phase_pi_multiple"] = np.array(spec["phase_pi_multiples"], dtype=np.float64)[target]
            side = {"schema_version": 1, "reference_version": task.REFERENCE_VERSION, "status": "formal",
                "dataset": dataset, "subject_id": sid, "split": "train" if sid <= 6 else "validation",
                "group_key": f"{dataset}:subject_{sid:03d}", "reference_shape": list(signal.shape),
                "reference_dtype": "float64", "reference_sha256": task.reference_safety._array_sha(signal),
                "native_sample_rate_hz": spec["sample_rate_hz"], "onset_sample_zero_based": spec["onset_sample_zero_based"],
                "frequency_order_hz": deepcopy(spec["frequency_order_hz"]), "phase_pi_multiples": deepcopy(spec["phase_pi_multiples"]),
                "native_units": None, "channel_order": None, "output_axes": ["trial", "time", "channel"],
                "trial_order": spec["trial_order"], "electrode_order": deepcopy(spec["electrode_order"]),
                "materialized_trials": len(target), "raw_reference_is_clean_ground_truth": False}
            row = self.write_subject(self.reference, arrays, side, include_embedded=True)
            self.rows.append(row)
            self.subjects[(dataset, sid)] = (arrays, side)
        self.manifest = {"schema_version": 1, "reference_version": task.REFERENCE_VERSION, "status": "formal",
            "build_completed": True, "materialized_subjects": self.rows, "holdout_waveform_loaded": False,
            "holdout_materialized": False, "raw_reference_is_clean_ground_truth": False,
            "model_metrics_computed": False, "runtime": {"enabled": False, "production_validated": False},
            # Metadata identities are allowed; no corresponding NPZ exists.
            "holdout_identities": [{"dataset": "Nakanishi2015", "subject_id": 9, "split": "holdout"}]}
        self.write_json(self.reference / "manifest.json", self.manifest)
        self.pin = mock.patch.object(task, "FROZEN_MANIFEST_SHA256", task._hash(self.reference / "manifest.json"))
        self.pin.start()
        self.addCleanup(self.pin.stop)
        self.network = mock.patch("socket.socket", side_effect=AssertionError("network is forbidden in tests"))
        self.network.start()
        self.addCleanup(self.network.stop)

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")

    def write_subject(self, directory, arrays, side, include_embedded=False):
        dataset, sid = side["dataset"], side["subject_id"]
        npz = directory / dataset / f"subject_{sid:03d}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        side = deepcopy(side)
        side.pop("npz_sha256", None)
        side.pop("npz_relative_path", None)
        members = deepcopy(arrays)
        side["npz_members"] = sorted(list(members) + (["manifest_json"] if include_embedded else []))
        if include_embedded:
            members["manifest_json"] = np.asarray(json.dumps(side, sort_keys=True, allow_nan=False))
        with npz.open("wb") as handle:
            np.savez(handle, **members)
        digest = task._hash(npz)
        side["npz_sha256"] = digest
        side["npz_relative_path"] = npz.name
        sidepath = Path(str(npz) + ".manifest.json")
        self.write_json(sidepath, side)
        return {"dataset": dataset, "subject_id": sid, "split": side["split"],
            "relative_path": npz.relative_to(directory).as_posix(), "npz_sha256": digest,
            "manifest_sha256": task._hash(sidepath), "trials": side["materialized_trials"]}

    def repin(self):
        self.write_json(self.reference / "manifest.json", self.manifest)
        self.pin.stop()
        self.pin = mock.patch.object(task, "FROZEN_MANIFEST_SHA256", task._hash(self.reference / "manifest.json"))
        self.pin.start()
        self.addCleanup(self.pin.stop)

    def candidate(self, name="candidate_a", transform=None, side_transform=None, signal_key="processed", embedded=False):
        directory = self.workspace / name
        directory.mkdir()
        rows = []
        for key, (arrays, original_side) in self.subjects.items():
            source_row = next(row for row in self.rows if (row["dataset"], row["subject_id"]) == key)
            side = {field: deepcopy(original_side[field]) for field in task.BINDING_FIELDS}
            side.update({"schema_version": 1, "reference_version": task.REFERENCE_VERSION, "status": "formal",
                "signal_shape": list(arrays["reference"].shape), "signal_key": signal_key,
                "materialized_trials": len(arrays["reference"]),
                "reference_npz_sha256": source_row["npz_sha256"], "reference_manifest_sha256": source_row["manifest_sha256"]})
            members = {k: v.copy() for k, v in arrays.items() if k != "reference"}
            members[signal_key] = arrays["reference"].copy()
            if transform is not None:
                transform(key, members)
            if side_transform is not None:
                side_transform(key, side)
            rows.append(self.write_subject(directory, members, side, include_embedded=embedded))
        manifest = {"schema_version": 1, "reference_version": task.REFERENCE_VERSION, "status": "formal",
            "build_completed": True, "reference_manifest_sha256": task.FROZEN_MANIFEST_SHA256,
            "materialized_subjects": rows}
        self.write_json(directory / "manifest.json", manifest)
        return directory

    def evaluate(self, **kwargs):
        return task.evaluate_task(self.reference, workspace=self.workspace, **kwargs)

    def rejected(self, **kwargs):
        with self.assertRaises(task.SSVEPTaskEvalV5Error):
            self.evaluate(**kwargs)

    def test_fft_predictions_and_metrics(self):
        report = self.evaluate()
        self.assertEqual(len(report["subjects"]), 3)
        self.assertEqual(report["macro_accuracy"], 1.)
        for subject in report["subjects"]:
            self.assertEqual(subject["baseline"]["accuracy"], 1.)
            self.assertEqual(subject["trials"]["baseline_prediction"], subject["trials"]["target_id"])
            self.assertGreaterEqual(subject["baseline"]["mean_confidence"], 0.)
            self.assertLessEqual(subject["baseline"]["mean_confidence"], 1.)
            self.assertGreater(subject["baseline"]["mean_target_frequency_power"], 0.)
            self.assertTrue(0 < subject["baseline"]["mean_relative_score"] <= 1)

    def test_candidate_identity_agreement_and_ratio(self):
        candidate = self.candidate(embedded=True)
        report = self.evaluate(candidate_dirs=[candidate])
        for subject in report["subjects"]:
            result = subject["candidates"][0]
            self.assertEqual(result["metrics"], subject["baseline"])
            self.assertEqual(result["prediction_agreement"], 1.)
            np.testing.assert_allclose(result["spectral_score_ratio"]["per_trial"], 1.)

    def test_amplitude_scaling_has_fourfold_power(self):
        candidate = self.candidate(transform=lambda _, a: a["processed"].__imul__(2.))
        report = self.evaluate(candidate_dirs=[candidate])
        for subject in report["subjects"]:
            np.testing.assert_allclose(subject["candidates"][0]["spectral_score_ratio"]["per_trial"], 4.)

    def test_predictions_are_not_target_labels(self):
        def swap_wave(_, a):
            a["processed"] = np.roll(a["processed"], 2, axis=0)
        candidate = self.candidate(transform=swap_wave)
        report = self.evaluate(candidate_dirs=[candidate])
        for subject in report["subjects"]:
            self.assertEqual(subject["candidates"][0]["prediction_agreement"], 0.)
            self.assertEqual(subject["candidates"][0]["metrics"]["accuracy"], 0.)

    def test_multiple_candidates_summarized_independently(self):
        a = self.candidate()
        b = self.candidate("candidate_b", transform=lambda _, a: a["processed"].__imul__(2.))
        report = self.evaluate(candidate_dirs=[a, b])
        for summary in report["dataset_summary"].values():
            self.assertEqual(len(summary["candidates"]), 2)
            self.assertAlmostEqual(summary["candidates"][0]["mean_spectral_score_ratio"], 1.)
            self.assertAlmostEqual(summary["candidates"][1]["mean_spectral_score_ratio"], 4.)

    def test_custom_safe_signal_key(self):
        candidate = self.candidate(signal_key="filtered_signal")
        report = self.evaluate(candidate_dirs=[candidate], candidate_signal_key="filtered_signal")
        self.assertEqual(report["candidate_signal_key"], "filtered_signal")

    def test_train_only_reads_selected_subject(self):
        real = task._read_subject
        with mock.patch.object(task, "_read_subject", wraps=real) as reader:
            report = self.evaluate(split="train")
        self.assertEqual(reader.call_count, 1)
        self.assertEqual([r["subject_id"] for r in report["subjects"]], [1])

    def test_validation_never_loads_train_npz(self):
        paths = []
        real_load = np.load
        def spy(path, *args, **kwargs):
            paths.append(Path(path).name)
            return real_load(path, *args, **kwargs)
        with mock.patch.object(task.np, "load", side_effect=spy):
            self.evaluate()
        self.assertNotIn("subject_001.npz", paths)

    def test_check_only_never_opens_npz_bytes(self):
        candidate = self.candidate()
        original_open = Path.open
        def safe_open(path, *args, **kwargs):
            if path.suffix == ".npz":
                raise AssertionError("check-only must not read NPZ bytes")
            return original_open(path, *args, **kwargs)
        with mock.patch.object(Path, "open", safe_open), mock.patch.object(task.np, "load", side_effect=AssertionError("no arrays")):
            report = self.evaluate(candidate_dirs=[candidate], check_only=True)
        self.assertFalse(report["waveforms_loaded"])
        self.assertFalse(report["npz_content_hashes_verified"])
        self.assertFalse(report["trial_alignment_values_verified"])

    def test_wrong_candidate_shape_binding(self):
        candidate = self.candidate(side_transform=lambda _, s: s["signal_shape"].__setitem__(0, 7))
        self.rejected(candidate_dirs=[candidate])

    def test_wrong_candidate_array_shape_even_with_valid_npz_hash(self):
        candidate = self.candidate(transform=lambda _, a: a.__setitem__("processed", a["processed"][:, :, :1]))
        self.rejected(candidate_dirs=[candidate])

    def test_candidate_target_relabel_rejected(self):
        candidate = self.candidate(transform=lambda _, a: a["target_id"].__setitem__(0, 1))
        self.rejected(candidate_dirs=[candidate])

    def test_candidate_trial_reorder_rejected(self):
        candidate = self.candidate(transform=lambda _, a: a.__setitem__("trial_index", a["trial_index"][::-1]))
        self.rejected(candidate_dirs=[candidate])

    def test_candidate_block_relabel_rejected(self):
        candidate = self.candidate(transform=lambda _, a: a["block_id"].__setitem__(0, 1))
        self.rejected(candidate_dirs=[candidate])

    def test_candidate_electrode_relabel_rejected(self):
        candidate = self.candidate(transform=lambda _, a: a["electrode_id"].__setitem__(0, 0))
        self.rejected(candidate_dirs=[candidate])

    def test_candidate_frequency_relabel_rejected(self):
        candidate = self.candidate(transform=lambda _, a: a["frequency_hz"].__setitem__(0, 10.))
        self.rejected(candidate_dirs=[candidate])

    def test_candidate_phase_relabel_rejected(self):
        def change(key, arrays):
            if key[0] == "Wearable":
                arrays["phase_pi_multiple"][0] = .5
        candidate = self.candidate(transform=change)
        self.rejected(candidate_dirs=[candidate])

    def test_candidate_label_dtype_change_rejected(self):
        candidate = self.candidate(transform=lambda _, a: a.__setitem__("target_id", a["target_id"].astype(np.float64)))
        self.rejected(candidate_dirs=[candidate])

    def test_candidate_nonfinite_rejects_entire_report(self):
        candidate = self.candidate(transform=lambda _, a: a["processed"].__setitem__((0, 0, 0), np.nan))
        output = self.workspace / "result.json"
        self.rejected(candidate_dirs=[candidate], output=output, markdown=True)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".md").exists())

    def test_object_pickle_arrays_rejected_before_numpy_load(self):
        candidate = self.candidate(transform=lambda _, a: a.__setitem__("processed", a["processed"].astype(object)))
        with mock.patch.object(task.np, "load", wraps=np.load) as loader:
            self.rejected(candidate_dirs=[candidate])
        self.assertEqual(loader.call_count, 1)  # reference only, not candidate

    def test_clean_signal_key_rejected(self):
        for key in ("clean", "clean_signal", "groundTruth", "target_signal", "target_id", "reference"):
            with self.subTest(key=key):
                self.rejected(candidate_signal_key=key)

    def test_unused_clean_npz_member_rejected(self):
        candidate = self.candidate(transform=lambda _, a: a.__setitem__("clean", np.zeros(1)))
        self.rejected(candidate_dirs=[candidate])

    def test_forbidden_candidate_directory_name(self):
        candidate = self.candidate("ground_truth_candidate")
        self.rejected(candidate_dirs=[candidate])

    def test_nested_candidate_ground_truth_metadata_rejected(self):
        candidate = self.candidate(side_transform=lambda _, s: s.update(provenance={"ground_truth": "no"}))
        self.rejected(candidate_dirs=[candidate], check_only=True)

    def test_candidate_keyword_value_rejected(self):
        candidate = self.candidate(side_transform=lambda _, s: s.update(description="trained on clean signals"))
        self.rejected(candidate_dirs=[candidate], check_only=True)

    def test_holdout_split_rejected(self):
        self.rejected(split="holdout")

    def test_unlisted_holdout_npz_rejected_without_loading(self):
        holdout = self.reference / "Nakanishi2015/subject_009.npz"
        holdout.write_bytes(b"never read")
        with mock.patch.object(task.np, "load", side_effect=AssertionError("no waveform reads")):
            self.rejected(check_only=True)

    def test_unknown_units_preserved_not_guessed(self):
        report = self.evaluate()
        self.assertFalse(report["native_units_assumed"])
        self.assertTrue(all(row["native_units"] is None and not row["native_units_assumed"] for row in report["subjects"]))

    def test_window_bounds_rejected(self):
        arrays, side = deepcopy(self.subjects[("Nakanishi2015", 7)])
        arrays["reference"] = np.ascontiguousarray(arrays["reference"][:, :1000, :])
        side["reference_shape"] = list(arrays["reference"].shape)
        side["reference_sha256"] = task.reference_safety._array_sha(arrays["reference"])
        row = self.write_subject(self.reference, arrays, side, include_embedded=True)
        self.rows[1] = row
        self.repin()
        self.rejected(check_only=True)

    def test_candidate_timing_cannot_adapt(self):
        candidate = self.candidate(side_transform=lambda _, s: s.update(onset_sample_zero_based=s["onset_sample_zero_based"] + 1))
        self.rejected(candidate_dirs=[candidate], check_only=True)

    def test_candidate_root_sha_binding_required(self):
        candidate = self.candidate()
        path = candidate / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["reference_manifest_sha256"] = "0" * 64
        self.write_json(path, manifest)
        self.rejected(candidate_dirs=[candidate], check_only=True)

    def test_frozen_root_manifest_tamper(self):
        path = self.reference / "manifest.json"
        path.write_text(path.read_text() + " ")
        self.rejected(check_only=True)

    def test_reference_sidecar_tamper(self):
        path = self.reference / "Nakanishi2015/subject_007.npz.manifest.json"
        path.write_text(path.read_text() + " ")
        self.rejected(check_only=True)

    def test_reference_npz_tamper(self):
        path = self.reference / "Nakanishi2015/subject_007.npz"
        with path.open("ab") as handle:
            handle.write(b"tampered")
        self.rejected()

    def test_unknown_reference_version(self):
        self.manifest["reference_version"] = "ssvep_reference_future"
        self.repin()
        self.rejected(check_only=True)

    def test_reference_nonfinite_rejected(self):
        arrays, side = deepcopy(self.subjects[("Nakanishi2015", 7)])
        arrays["reference"][0, 0, 0] = np.inf
        side["reference_sha256"] = task.reference_safety._array_sha(arrays["reference"])
        self.rows[1] = self.write_subject(self.reference, arrays, side, include_embedded=True)
        self.repin()
        self.rejected()

    def test_reference_frequency_label_inconsistent(self):
        arrays, side = deepcopy(self.subjects[("Nakanishi2015", 7)])
        arrays["frequency_hz"][0] = 10.
        self.rows[1] = self.write_subject(self.reference, arrays, side, include_embedded=True)
        self.repin()
        self.rejected()

    def test_zero_energy_ratios_explicitly_undefined(self):
        ratio = task._ratio(np.zeros(3), np.zeros(3))
        self.assertEqual(ratio["per_trial"], [None, None, None])
        self.assertEqual(ratio["undefined_trials"], 3)
        self.assertIsNone(ratio["mean"])

    def test_spectral_overflow_rejected(self):
        candidate = self.candidate(transform=lambda _, a: a["processed"].__imul__(1e200))
        self.rejected(candidate_dirs=[candidate])

    def test_dataset_and_electrode_domains_are_separate(self):
        report = self.evaluate(candidate_dirs=[self.candidate()])
        self.assertEqual(set(report["dataset_summary"]), {"Nakanishi2015", "Wearable"})
        self.assertEqual(set(report["domain_summary"]), {"Nakanishi2015/electrode_unknown", "Wearable/dry", "Wearable/wet"})
        self.assertEqual(report["domain_summary"]["Wearable/dry"]["baseline"]["trials"], 3)

    def test_flags_remain_formal_but_not_accepted(self):
        report = self.evaluate()
        self.assertEqual(report["status"], "formal")
        for key in ("holdout_loaded", "clean_ground_truth", "runtime_enabled", "production_validated", "model_metrics_computed", "training_performed", "threshold_selection_performed"):
            self.assertIs(report[key], False)
        self.assertIn("reporting_only", report["validation_use"])
        self.assertNotIn("snr", report)
        self.assertNotIn("rmse", report)

    def test_reports_json_markdown_and_overwrite_rejection(self):
        output = self.workspace / "reports/task.json"
        report = self.evaluate(output=output, markdown=True)
        self.assertEqual(json.loads(output.read_text())["subjects"], report["subjects"])
        self.assertIn("not model", output.with_suffix(".md").read_text())
        with mock.patch.object(task.np, "load", side_effect=AssertionError("must refuse output first")):
            self.rejected(output=output)

    def test_existing_markdown_blocks_even_json_only_output(self):
        output = self.workspace / "result.json"
        output.with_suffix(".md").write_text("existing report")
        self.rejected(output=output)
        self.assertFalse(output.exists())

    def test_output_outside_workspace_rejected(self):
        self.rejected(output=self.workspace.parent / "out.json")

    def test_d_drive_source_and_output_rejected(self):
        self.rejected(output="D:/SSVEP/result.json")
        with self.assertRaises(task.SSVEPTaskEvalV5Error):
            task.evaluate_task("D:/SSVEP", workspace=self.workspace)
        self.rejected(candidate_dirs=["D:/SSVEP"])

    def test_lexical_traversal_rejected(self):
        self.rejected(output="reports/../result.json")

    def test_foreign_reference_directory_rejected(self):
        other = self.workspace / "foreign_reference"
        other.mkdir()
        with self.assertRaises(task.SSVEPTaskEvalV5Error):
            task.evaluate_task(other, workspace=self.workspace)

    def test_output_inside_input_rejected(self):
        self.rejected(output=self.reference / "result.json")

    def test_reparse_junction_rejected(self):
        original = task.reference_safety.safety._is_link
        def junction(path):
            return Path(path) == self.reference or original(path)
        with mock.patch.object(task.reference_safety.safety, "_is_link", side_effect=junction):
            self.rejected(check_only=True)

    def test_input_change_during_evaluation_rejected_no_report(self):
        real_read = task._read_subject
        output = self.workspace / "out.json"
        changed = False
        def change(catalog, item, key):
            nonlocal changed
            result = real_read(catalog, item, key)
            if not changed:
                changed = True
                path = self.reference / "manifest.json"
                path.write_text(path.read_text() + " ")
            return result
        with mock.patch.object(task, "_read_subject", side_effect=change):
            self.rejected(output=output)
        self.assertFalse(output.exists())

    def test_check_only_report_writes_without_waveforms(self):
        output = self.workspace / "metadata.json"
        with mock.patch.object(task.np, "load", side_effect=AssertionError("no numeric loading")):
            report = self.evaluate(check_only=True, output=output, markdown=True)
        self.assertTrue(output.exists())
        self.assertFalse(report["waveforms_loaded"])
        self.assertIn("NOT verified", output.with_suffix(".md").read_text())

    def test_cli_check_only_offline(self):
        output = self.workspace / "cli_result.json"
        with mock.patch.object(task, "WORKSPACE", self.workspace), contextlib.redirect_stdout(io.StringIO()):
            code = cli.main(["--reference-dir", str(self.reference), "--output", str(output), "--check-only", "--markdown"])
        self.assertEqual(code, 0)
        self.assertFalse(json.loads(output.read_text())["holdout_loaded"])
        self.assertTrue(output.with_suffix(".md").exists())

    def test_cli_does_not_offer_holdout_or_trust_bypass(self):
        parser = cli.make_parser()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--output", "x.json", "--split", "holdout"])
        self.assertFalse(any(action.dest in ("expected_manifest_sha", "allow_holdout", "allow_foreign_reference") for action in parser._actions))


    def test_exact_fft_score_bins_and_confidence_formula(self):
        arrays, side = self.subjects[("Wearable", 7)]
        decoded = task.spectral_decode(arrays["reference"], side)
        length = 500
        start = side["onset_sample_zero_based"]
        x = arrays["reference"][0, start:start + length].copy()
        x -= x.mean(axis=0)
        w = np.hanning(length)
        power = np.square(np.abs(np.fft.rfft(x * w[:, None], axis=0))) / np.sum(w * w)
        grid = np.fft.rfftfreq(length, 1 / 250.)
        manual = []
        for f in side["frequency_order_hz"]:
            i = min(range(len(grid)), key=lambda k: abs(grid[k] - f))
            j = min(range(len(grid)), key=lambda k: abs(grid[k] - 2 * f))
            manual.append(power[i].mean() + .5 * power[j].mean())
        np.testing.assert_allclose(decoded["scores"][0], manual, rtol=1e-13)
        best, second = sorted(manual)[-1], sorted(manual)[-2]
        self.assertAlmostEqual(decoded["confidence"][0], (best - second) / (best + task.EPSILON))
        self.assertEqual(decoded["fundamental_bins"][0], 18)  # 9.25 Hz: tie -> 9 Hz

    def test_outside_stimulation_window_cannot_change_predictions(self):
        def outside(key, arrays):
            spec = task.reference_safety.SPECS[key[0]]
            start = spec["onset_sample_zero_based"]
            end = start + task.WINDOW_LENGTHS[key[0]]
            arrays["processed"][:, :start] += 1e6
            arrays["processed"][:, end:] -= 1e6
        candidate = self.candidate(transform=outside)
        report = self.evaluate(candidate_dirs=[candidate])
        for row in report["subjects"]:
            self.assertEqual(row["candidates"][0]["prediction_agreement"], 1.)
            np.testing.assert_allclose(row["candidates"][0]["spectral_score_ratio"]["per_trial"], 1.)

    def test_candidate_sidecar_tamper_rejected_metadata_only(self):
        candidate = self.candidate()
        path = candidate / "Nakanishi2015/subject_007.npz.manifest.json"
        path.write_text(path.read_text() + " ")
        self.rejected(candidate_dirs=[candidate], check_only=True)

    def test_candidate_subject_binding_cannot_change(self):
        candidate = self.candidate(side_transform=lambda _, s: s.update(reference_npz_sha256="0" * 64))
        self.rejected(candidate_dirs=[candidate], check_only=True)

    def test_candidate_units_cannot_be_invented(self):
        candidate = self.candidate(side_transform=lambda _, s: s.update(native_units="microvolt"))
        self.rejected(candidate_dirs=[candidate], check_only=True)

    def test_macro_subject_accuracy_is_not_micro_accuracy(self):
        a = {"trials": 2, "accuracy": 1., "mean_confidence": .5,
             "mean_relative_score": .8, "mean_target_frequency_power": 2.}
        b = {**a, "trials": 6, "accuracy": 0.}
        summary = task._aggregate([a, b])
        self.assertEqual(summary["macro_accuracy"], .5)
        self.assertEqual(summary["accuracy"], .25)

    def test_linked_subject_file_rejected_without_waveforms(self):
        target = self.reference / "Nakanishi2015/subject_007.npz"
        original = task.reference_safety.safety._is_link
        with mock.patch.object(task.reference_safety.safety, "_is_link", side_effect=lambda p: Path(p) == target or original(p)):
            self.rejected(check_only=True)

    def test_incomplete_candidate_rejected(self):
        candidate = self.candidate()
        manifest_path = candidate / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["build_completed"] = False
        self.write_json(manifest_path, manifest)
        self.rejected(candidate_dirs=[candidate], check_only=True)

    def test_unreadable_inventory_is_not_silently_accepted(self):
        def unreadable(top, **kwargs):
            kwargs["onerror"](PermissionError("synthetic unreadable directory"))
            return iter(())
        with mock.patch.object(task.os, "walk", side_effect=unreadable):
            self.rejected(check_only=True)



    def short_subject_and_mixed_candidate(self):
        key = ("Nakanishi2015", 8)
        arrays, side = deepcopy(self.subjects[key])
        arrays = {name: np.ascontiguousarray(value[:3]) for name, value in arrays.items()}
        side["reference_shape"] = list(arrays["reference"].shape)
        side["materialized_trials"] = 3
        side["reference_sha256"] = task.reference_safety._array_sha(arrays["reference"])
        self.subjects[key] = arrays, side
        index = next(i for i, row in enumerate(self.rows) if (row["dataset"], row["subject_id"]) == key)
        self.rows[index] = self.write_subject(self.reference, arrays, side, include_embedded=True)
        self.repin()
        def mixed(subject, members):
            if subject == ("Nakanishi2015", 7):
                members["processed"] = np.roll(members["processed"], 2, axis=0)
            elif subject[0] == "Wearable":
                members["processed"][0] = members["processed"][2]
        return self.candidate(transform=mixed)

    def test_global_micro_uses_correct_trial_counts_not_subject_mean(self):
        candidate = self.short_subject_and_mixed_candidate()
        report = self.evaluate(candidate_dirs=[candidate])
        global_summary = report["global_accuracy_summary"]
        self.assertEqual(global_summary["baseline"]["trials"], 15)
        self.assertEqual(report["micro_accuracy"], 1.)
        metrics = global_summary["candidates"][0]["metrics"]
        self.assertAlmostEqual(metrics["micro_accuracy"], 8 / 15)
        self.assertAlmostEqual(metrics["macro_accuracy"], (0 + 1 + 5 / 6) / 3)
        self.assertEqual(metrics["accuracy"], metrics["micro_accuracy"])
        self.assertNotIn("mean_target_frequency_power", global_summary["baseline"])
        self.assertNotIn("mean_target_frequency_power", metrics)

    def test_subject_accuracy_delta_exact_direction(self):
        report = self.evaluate(candidate_dirs=[self.short_subject_and_mixed_candidate()])
        for row in report["subjects"]:
            candidate = row["candidates"][0]
            expected = candidate["metrics"]["accuracy"] - row["baseline"]["accuracy"]
            self.assertEqual(candidate["accuracy_delta"], expected)
        row = next(row for row in report["subjects"] if row["dataset"] == "Nakanishi2015" and row["subject_id"] == 7)
        self.assertEqual(row["candidates"][0]["accuracy_delta"], -1.)

    def test_dataset_domain_and_global_macro_micro_deltas(self):
        report = self.evaluate(candidate_dirs=[self.short_subject_and_mixed_candidate()])
        dataset = report["dataset_summary"]["Nakanishi2015"]["candidates"][0]
        self.assertAlmostEqual(dataset["macro_accuracy_delta"], -.5)
        self.assertAlmostEqual(dataset["micro_accuracy_delta"], 1 / 3 - 1)
        dry = report["domain_summary"]["Wearable/dry"]["candidates"][0]
        wet = report["domain_summary"]["Wearable/wet"]["candidates"][0]
        self.assertAlmostEqual(dry["micro_accuracy_delta"], 2 / 3 - 1)
        self.assertEqual(wet["micro_accuracy_delta"], 0.)
        glob = report["global_accuracy_summary"]["candidates"][0]
        self.assertAlmostEqual(glob["micro_accuracy_delta"], 8 / 15 - 1)
        self.assertAlmostEqual(glob["macro_accuracy_delta"], (1 + 5 / 6) / 3 - 1)

    def test_explicit_task_model_flags_with_and_without_candidate(self):
        baseline = self.evaluate()
        self.assertTrue(baseline["task_metrics_computed"])
        self.assertFalse(baseline["candidate_task_metrics_computed"])
        self.assertFalse(baseline["learned_model_loaded"])
        self.assertFalse(baseline["model_metrics_computed"])
        candidate = self.candidate()
        full = self.evaluate(candidate_dirs=[candidate])
        self.assertTrue(full["task_metrics_computed"])
        self.assertTrue(full["candidate_task_metrics_computed"])
        self.assertFalse(full["learned_model_loaded"])
        self.assertFalse(full["model_metrics_computed"])
        self.assertIn("candidate-derived task metrics", full["model_metrics_scope"])
        metadata = self.evaluate(candidate_dirs=[candidate], check_only=True)
        self.assertFalse(metadata["task_metrics_computed"])
        self.assertFalse(metadata["candidate_task_metrics_computed"])
        self.assertFalse(metadata["learned_model_loaded"])
        self.assertIsNone(metadata["micro_accuracy"])
        self.assertIsNone(metadata["global_accuracy_summary"])

    def test_micro_alias_on_aggregate(self):
        summary = task._aggregate([
            {"trials": 3, "correct_trials": 2, "accuracy": 2 / 3, "mean_confidence": .4,
             "mean_relative_score": .7, "mean_target_frequency_power": 1.},
            {"trials": 7, "correct_trials": 1, "accuracy": 1 / 7, "mean_confidence": .4,
             "mean_relative_score": .7, "mean_target_frequency_power": 1.}])
        self.assertEqual(summary["micro_accuracy"], .3)
        self.assertEqual(summary["accuracy"], .3)
        self.assertNotEqual(summary["macro_accuracy"], .3)

    def test_limitations_describe_ratio_guard_and_fft_scope(self):
        report = self.evaluate(check_only=True)
        limitations = " ".join(report["limitations"])
        self.assertIn("not evidence of absent EEG response", limitations)
        self.assertIn("phase preservation or artifact removal", limitations)
        self.assertIn("not a replication of published CCA/FBCCA", limitations)
        markdown = task.render_markdown(report)
        self.assertIn("model_metrics_computed=false means no learned weights", markdown)
        self.assertIn("ratio denominator<=EPS", markdown)


if __name__ == "__main__":
    unittest.main()
