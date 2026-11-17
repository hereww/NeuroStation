"""Meaningful block/label isolation and signal tests; no public waveforms read."""
from dataclasses import replace
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

from scripts import benchmark_matched_public_ssvep as b

TARGETS = (10, 12, 15)


def epoch(frequency, seed=0, n=500):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 250
    # Stable trial phase and channel-specific gains; random noise varies by block.
    return np.stack([(1 + c / 8) * (np.sin(2*np.pi*frequency*t + c*0.12) +
                     0.35*np.sin(4*np.pi*frequency*t + c*0.08)) + 0.12*rng.normal(size=n)
                     for c in range(8)])


def dataset(blocks=6, subjects=(1,), depths=("fixed",), sessions=(1,), targets=TARGETS, kind="wang2016"):
    trials = []
    for subject in subjects:
        for depth in depths:
            for session in sessions:
                for block in range(blocks):
                    for condition, frequency in enumerate(targets):
                        group = b.canonical([subject, block, depth, session])
                        trials.append(b.Trial(subject, block, depth, session, condition, float(frequency), group,
                                              epoch(frequency, subject*100+block*10+condition), "synthetic", len(trials)))
    return b.Dataset(trials, {"metadata_authority": "synthetic unit test; not public evidence",
                             "sample_window": {"start": 0, "stop": 500}}, {}, kind)


def wang_spec():
    names = ["CH"+str(i) for i in range(64)]
    channels = [0, 2, 25, 29, 43, 51, 60, 62]
    for index, name in zip(channels, ("FP1", "FP2", "C3", "C4", "P7", "P8", "O1", "O2")):
        names[index] = name
    frequencies = [30.1 + i*0.1 for i in range(40)]
    for index, frequency in zip((2, 4, 7), TARGETS):
        frequencies[index] = frequency
    return {"schema_version": 1, "status": "formal", "metadata_authority": "synthetic Wang-shaped fixture",
            "sample_rate_hz": 250, "mat_variable": "data", "expected_shape": [64, 1500, 40, 6],
            "axes": {"channel": 0, "sample": 1, "condition": 2, "block": 3},
            "source_channel_names": names, "target_channel_map": dict(zip(b.CHANNELS, [names[i] for i in channels])),
            "frequency_order_hz": frequencies, "sample_window": {"start": 125, "stop": 1375},
            "depth": "unspecified_by_source", "session": "single_recording", "amplitude_unit": "unknown"}


class SignalTests(unittest.TestCase):
    def test_cca_and_fbcca_identify_sinusoids(self):
        for cls in (b.CCA, b.FBCCA):
            model = cls(TARGETS)
            for f in TARGETS:
                with self.subTest(algorithm=cls.__name__, frequency=f):
                    result = model.forward(epoch(f, 876+f))
                    self.assertEqual(result["predicted_frequency_hz"], f)
                    self.assertGreater(result["margin"], 0.05)

    def test_trca_actually_fits_generalized_eigenproblem_and_templates(self):
        source = dataset()
        train = [t for t in source.trials if t.block < 4]
        model = b.TRCA(TARGETS)
        with self.assertRaisesRegex(ValueError, "must fit"):
            model.forward(epoch(10))
        model.fit(np.stack([t.eeg for t in train]), [t.frequency_hz for t in train])
        self.assertEqual(model.fit_counts, {10: 4, 12: 4, 15: 4})
        for f in TARGETS:
            filtered = np.stack([b.preprocess(t.eeg) for t in train if t.frequency_hz == f])
            filtered -= filtered.mean(axis=-1, keepdims=True)
            np.testing.assert_allclose(model.templates[f], filtered.mean(axis=0))
            q = np.einsum("tcs,tds->cd", filtered, filtered)
            total = filtered.sum(axis=0)
            s = total @ total.T - q
            ridge = 1e-6 * max(np.trace(q)/8, 1e-12)
            w = model.filters[f]
            np.testing.assert_allclose(s @ w, model.eigenvalues[f]*(q+ridge*np.eye(8)) @ w, rtol=1e-5, atol=1e-5)
            self.assertEqual(model.forward(epoch(f, 989))["predicted_frequency_hz"], f)
        with self.assertRaisesRegex(ValueError, "already fitted"):
            model.fit(np.stack([t.eeg for t in train]), [t.frequency_hz for t in train])

    def test_forward_signature_never_accepts_truth_or_trial_object(self):
        for cls in (b.CCA, b.FBCCA, b.TRCA):
            self.assertEqual(list(inspect.signature(cls.forward).parameters), ["self", "eeg"])
            model = cls(TARGETS)
            with self.assertRaises(TypeError):
                model.forward(epoch(10), true_label=10)

    def test_nan_wrong_channels_and_short_window_rejected(self):
        for x in (np.zeros((7, 500)), np.zeros((8, 249)), np.full((8, 500), np.nan),
                  np.full((8, 500), np.inf), np.ones((8, 500), dtype=complex)):
            with self.subTest(shape=x.shape, dtype=x.dtype), self.assertRaises(ValueError):
                b.CCA(TARGETS).forward(x)

    def test_silent_eeg_unknown_for_all_algorithms(self):
        source = dataset()
        source.trials = [replace(t, eeg=np.zeros_like(t.eeg)) if t.block >= 4 else t for t in source.trials]
        report = b.evaluate(source, frequencies=TARGETS)
        self.assertEqual(len(report["rows"]), 18)
        for row in report["rows"]:
            self.assertFalse(row["accepted"])
            self.assertEqual(row["command"], "UNKNOWN")
            self.assertIsNone(row["candidate_frequency_hz"])
        for stats in report["algorithms"].values():
            self.assertEqual(stats["acceptance_rate"], 0)
            self.assertIsNone(stats["accepted_label_match_rate"])

    def test_wrong_recorded_truth_does_not_change_forward(self):
        source = dataset()
        # Swap evaluation labels/condition identities while leaving EEG intact;
        # complete condition/block membership stays valid, but all scores must
        # follow EEG rather than the recorded truth.
        swapped = {10: 12, 12: 15, 15: 10}
        bad = dataset()
        bad.trials = [replace(t, frequency_hz=float(swapped[int(t.frequency_hz)]),
                              condition_index=TARGETS.index(swapped[int(t.frequency_hz)])) if t.block >= 4 else t
                      for t in bad.trials]
        a = b.evaluate(source, frequencies=TARGETS)
        c = b.evaluate(bad, frequencies=TARGETS)
        for original, relabeled in zip(source.trials, bad.trials):
            np.testing.assert_array_equal(original.eeg, relabeled.eeg)
        self.assertEqual(len(a["rows"]), len(c["rows"]))
        for original, relabeled in zip(a["rows"], c["rows"]):
            left, right = original["prediction"], relabeled["prediction"]
            self.assertEqual(left.keys(), right.keys())
            self.assertEqual(left["predicted_frequency_hz"], right["predicted_frequency_hz"])
            self.assertEqual(left["scores"].keys(), right["scores"].keys())
            # LAPACK reductions may vary by a few ULPs across repeated Mac calls.
            # Labels/decisions remain exact; only float64 score roundoff is allowed.
            for key in left:
                if key == "scores":
                    np.testing.assert_allclose(list(left[key].values()), list(right[key].values()),
                                               rtol=0, atol=32 * np.finfo(np.float64).eps)
                elif key != "predicted_frequency_hz":
                    np.testing.assert_allclose(left[key], right[key],
                                               rtol=0, atol=32 * np.finfo(np.float64).eps)
            for key in ("accepted", "command", "candidate_frequency_hz"):
                self.assertEqual(original[key], relabeled[key])
        self.assertTrue(all(not r["accepted_label_match"] for r in c["rows"]))


class SplitAndContractTests(unittest.TestCase):
    def test_full_group_disjoint_for_subject_depth_and_session(self):
        data = dataset(subjects=(1, 2), depths=(0.5, 1), sessions=("A", "B"))
        selected, assignments, records = b.select_and_split(data, TARGETS)
        self.assertEqual(len(records), 8)
        groups = {s: {t.group_id for t in selected if assignments[t.group_id] == s}
                  for s in ("train", "validation", "test")}
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])
        for record in records:
            self.assertEqual(record["train"]["blocks"], [0, 1, 2, 3])
            self.assertEqual(record["validation"]["blocks"], [4])
            self.assertEqual(record["test"]["blocks"], [5])

    def test_test_and_validation_eeg_cannot_change_trca_fit(self):
        source = dataset()
        a = b.evaluate(source, frequencies=TARGETS)
        source.trials = [replace(t, eeg=epoch(12, 404)) if t.block >= 4 else t for t in source.trials]
        c = b.evaluate(source, frequencies=TARGETS)
        self.assertEqual(a["trca_calibration"], c["trca_calibration"])
        heldout_ids = {r["trial_id"] for r in c["rows"]}
        self.assertFalse(heldout_ids & set(c["trca_calibration"][0]["training_trial_ids"]))

    def test_missing_20_reported_and_four_targets_refused(self):
        report = b.evaluate(dataset(), frequencies=TARGETS)
        self.assertEqual(report["missing_personal_targets"], [20])
        self.assertEqual(report["chance_baseline"], 1/3)
        self.assertFalse(report["four_frequency_evaluation"])
        self.assertFalse(any(r["prediction"]["predicted_frequency_hz"] == 20 for r in report["rows"]))
        with self.assertRaisesRegex(ValueError, "absent.*20"):
            b.evaluate(dataset(), frequencies=b.PERSONAL_TARGETS)

    def test_four_target_dataset_runs_without_fabrication(self):
        report = b.evaluate(dataset(targets=b.PERSONAL_TARGETS, kind="prepared"), frequencies=b.PERSONAL_TARGETS)
        self.assertEqual(report["missing_personal_targets"], [])
        self.assertEqual(report["chance_baseline"], 0.25)
        self.assertTrue(report["four_frequency_evaluation"])
        for algorithm in report["algorithms"].values():
            self.assertEqual(algorithm["by_split"]["test"]["evaluation_trial_count"], 4)

    def test_duplicate_trial_malformed_group_and_missing_block_trial_rejected(self):
        source = dataset()
        source.trials.append(source.trials[0])
        with self.assertRaisesRegex(ValueError, "duplicate trial"):
            b.select_and_split(source, TARGETS)
        source = dataset()
        source.trials[0] = replace(source.trials[0], group_id='["incomplete"]')
        with self.assertRaisesRegex(ValueError, "group_id"):
            b.select_and_split(source, TARGETS)
        source = dataset()
        source.trials.pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            b.select_and_split(source, TARGETS)

    def test_adaptive_prepared_three_and_four_blocks(self):
        for count in (3, 4):
            report = b.evaluate(dataset(blocks=count, kind="prepared"), frequencies=TARGETS)
            group = report["split_groups"][0]
            self.assertEqual(group["train"]["blocks"], list(range(count-2)))
            self.assertEqual(group["validation"]["blocks"], [count-2])
            self.assertEqual(group["test"]["blocks"], [count-1])
            if count == 3:
                self.assertEqual(report["trca_calibration"][0]["status"], "unavailable")
                self.assertEqual(report["algorithms"]["TRCA"]["status"], "unavailable")
            else:
                self.assertEqual(report["trca_calibration"][0]["status"], "fitted")
        with self.assertRaisesRegex(ValueError, ">=3"):
            b.evaluate(dataset(blocks=2, kind="prepared"), frequencies=TARGETS)

    def test_report_and_commands_new_files_and_unknown_contract(self):
        report = b.evaluate(dataset(), frequencies=TARGETS, margin_threshold=1, score_threshold=1)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output, commands = root/"report.json", root/"commands.jsonl"
            b.save_new_outputs(report, output, commands)
            loaded = json.loads(output.read_text(encoding="utf-8"))
            for key, value in b.FLAGS.items():
                self.assertEqual(loaded[key], value)
            self.assertEqual(loaded["evaluation_scope"], "same_subject_heldout_blocks")
            rows = [json.loads(line) for line in commands.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 18)
            self.assertTrue(all(r["command"] == "UNKNOWN" and r["executed"] is False for r in rows))
            for algorithm in loaded["algorithms"].values():
                self.assertEqual(sum(sum(v.values()) for v in algorithm["all_trial_confusion_matrix"].values()), 6)
            before = output.read_bytes(), commands.read_bytes()
            with self.assertRaisesRegex(ValueError, "overwrite"):
                b.save_new_outputs(report, output, commands)
            self.assertEqual(before, (output.read_bytes(), commands.read_bytes()))

    def test_output_conflicts_and_existing_commands_preserved(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report, commands = root/"new.json", root/"old.jsonl"
            commands.write_text("existing content", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                b.save_new_outputs({"rows": []}, report, commands)
            self.assertFalse(report.exists())
            self.assertEqual(commands.read_text(encoding="utf-8"), "existing content")
            with self.assertRaisesRegex(ValueError, "must differ"):
                b.save_new_outputs({"rows": []}, report, report)

    def test_thresholds_invalid_and_mapping_override(self):
        for kwargs in ({"score_threshold": np.nan}, {"margin_threshold": np.inf}, {"margin_threshold": -1}):
            with self.assertRaises(ValueError):
                b.evaluate(dataset(), frequencies=TARGETS, **kwargs)
        self.assertEqual(b.command_map("10=PREVIOUS", TARGETS)[10], "PREVIOUS")
        for raw in ("10=UNKNOWN", "10=GO,10=BACK", "10=bad command", "11=GO"):
            with self.assertRaises(ValueError):
                b.command_map(raw, TARGETS)


class WangAdapterTests(unittest.TestCase):
    def test_real_mat_fixture_exact_order_crop_map_and_excluded_waveforms(self):
        spec = wang_spec()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/"spec.json"
            path.write_text(json.dumps(spec), encoding="utf-8")
            data = np.zeros((64, 1500, 40, 6), dtype=np.float64)
            for condition, frequency in zip((2, 4, 7), TARGETS):
                for block in range(6):
                    data[[0, 2, 25, 29, 43, 51, 60, 62], 125:1375, condition, block] = epoch(frequency, block+100, 1250)
            savemat(root/"S1.mat", {"data": data}, do_compression=True)
            del data
            # Invalid file contents ensure the excluded waveform is never opened.
            (root/"S30.mat").write_bytes(b"reserved; must not be read")
            loaded = b.load_wang(root, path, subjects=[1])
            self.assertEqual(loaded.provenance["excluded_subjects"], [30])
            self.assertFalse(loaded.provenance["excluded_subject_waveforms_read"])
            self.assertEqual(loaded.provenance["source_channel_indices"], [0, 2, 25, 29, 43, 51, 60, 62])
            self.assertEqual(loaded.provenance["source_channel_names"][0], "FP1")
            self.assertEqual(loaded.provenance["condition_indices_by_frequency"], {"10": [2], "12": [4], "15": [7], "20": []})
            self.assertEqual(len(loaded.trials), 18)
            np.testing.assert_allclose(loaded.trials[0].eeg, epoch(10, 100, 1250))
            self.assertEqual(loaded.trials[0].eeg.shape, (8, 1250))
            output, commands = root/"result.json", root/"commands.jsonl"
            with patch.object(b, "load_wang", return_value=loaded):
                report = b.benchmark(dataset="wang2016", mat_root=root, source_spec=path, subjects=[1],
                                     frequencies=TARGETS, output=output, commands_output=commands)
            self.assertEqual(report["source_provenance"]["sources"][0]["verified_variables"],
                             [{"name": "data", "shape": [64, 1500, 40, 6], "matlab_class": "double"}])
            self.assertEqual(report["missing_personal_targets"], [20])
            for algorithm in report["algorithms"].values():
                self.assertEqual(algorithm["by_split"]["test"]["all_trial_label_match_rate"], 1)

    def test_wang_missing_metadata_and_invalid_sampling_map_shape_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/"spec.json"
            for modify in (lambda s: s.pop("sample_window"), lambda s: s.update(sample_rate_hz=1000),
                           lambda s: s["target_channel_map"].update(Fp1="Fp1"),
                           lambda s: s["expected_shape"].__setitem__(0, 63),
                           lambda s: s.pop("metadata_authority")):
                spec = wang_spec()
                modify(spec)
                path.write_text(json.dumps(spec), encoding="utf-8")
                with self.assertRaises(ValueError):
                    b.load_wang(root, path, subjects=[1])
            with self.assertRaisesRegex(ValueError, "exactly one"):
                b.load_wang(root, path)

    def test_cli_accepts_explicit_subjects_and_reports_errors_without_outputs(self):
        args = b.make_parser().parse_args(["--dataset", "wang2016", "--subjects", "1,2,29",
                                          "--frequencies", "10,12,15", "--output", "new.json"])
        self.assertEqual(args.subjects, "1,2,29")
        with TemporaryDirectory() as directory:
            output = Path(directory)/"report.json"
            with patch("sys.stderr"):
                self.assertEqual(b.main(["--dataset", "wang2016", "--frequencies", "10,12,15", "--output", str(output)]), 2)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
