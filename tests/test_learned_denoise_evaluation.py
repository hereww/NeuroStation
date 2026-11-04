from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from eeg_tools import learned_denoise as ld
from scripts import evaluate_denoiser as evaluation
from scripts import run_learned_denoise as offline
from scripts.train_denoiser import _dataset_sha256, _save_split_metadata, _seeded_session_groups, _split_by_session, _split_metadata, _split_summary


class SessionSplitEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sessions = np.repeat(["s0", "s1", "s2", "s3", "s4"], 3)
        self.indices = _split_by_session(self.sessions, 42)
        self.checkpoint = {
            f"{name}_sessions": sorted(set(self.sessions[index].tolist()))
            for name, index in zip(("train", "validation", "test"), self.indices)
        }

    def test_five_groups_split_is_complete_and_strictly_disjoint(self) -> None:
        self.assertEqual([3, 1, 1], [len(self.checkpoint[f"{name}_sessions"]) for name in ("train", "validation", "test")])
        for left, right in ((0, 1), (0, 2), (1, 2)):
            self.assertFalse(set(self.sessions[self.indices[left]]) & set(self.sessions[self.indices[right]]))
            self.assertFalse(set(self.indices[left]) & set(self.indices[right]))
        np.testing.assert_array_equal(np.sort(np.concatenate(self.indices)), np.arange(len(self.sessions)))
        for expected, actual in zip(self.indices, _split_by_session(self.sessions, 42)):
            np.testing.assert_array_equal(expected, actual)

    def test_large_public_group_enters_training_without_changing_seeded_test(self) -> None:
        labels = np.repeat(["EEGdenoiseNet", "a", "b", "c", "d"], [2400, 28, 27, 26, 26])
        seed = 20261005
        initial = _seeded_session_groups(labels, seed)
        self.assertIn("EEGdenoiseNet", initial[1])
        train, validation, test = _split_by_session(labels, seed)
        groups = [set(labels[index].tolist()) for index in (train, validation, test)]
        self.assertIn("EEGdenoiseNet", groups[0])
        self.assertEqual(set(initial[2]), groups[2])
        np.testing.assert_array_equal(test, np.flatnonzero(np.isin(labels, initial[2])))
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual([3, 1, 1], [len(group) for group in groups])
        np.testing.assert_array_equal(np.sort(np.concatenate((train, validation, test))), np.arange(2507))
        metadata = _split_metadata(labels, (train, validation, test), seed)
        self.assertEqual("session_group_window_aware_fixed_test", metadata["strategy"])
        self.assertEqual(initial[2], metadata["initial_session_groups"]["test"])
        self.assertAlmostEqual(1.0, sum(metadata["actual_window_fractions"].values()))
        self.assertEqual(2400, metadata["group_window_counts"]["EEGdenoiseNet"])
        for name, index in zip(("train", "validation", "test"), (train, validation, test)):
            self.assertAlmostEqual(len(index) / 2507, metadata["actual_window_fractions"][name])
        checkpoint = {f"{name}_sessions": sorted(group) for name, group in zip(("train", "validation", "test"), groups)}
        checkpoint["split_metadata"] = metadata
        np.testing.assert_array_equal(evaluation._select_evaluation_indices(checkpoint, labels), test)

    def test_equal_size_groups_keep_original_seeded_assignment(self) -> None:
        for seed in range(10):
            seeded = _seeded_session_groups(self.sessions, seed)
            for expected, actual in zip(seeded, _split_by_session(self.sessions, seed)):
                self.assertEqual(set(expected), set(self.sessions[actual].tolist()))

    def test_largest_fixed_test_group_is_not_silently_moved_into_training(self) -> None:
        labels = np.repeat(["EEGdenoiseNet", "a", "b", "c", "d"], [2400, 28, 27, 26, 26])
        seed = next(seed for seed in range(100) if "EEGdenoiseNet" in _seeded_session_groups(labels, seed)[2])
        with self.assertRaisesRegex(ld.LearnedDenoiseError, "fixed test split"):
            _split_by_session(labels, seed)

    def test_split_sidecar_is_complete_but_console_summary_is_compact(self) -> None:
        metadata = _split_metadata(self.sessions, self.indices, 42)
        with tempfile.TemporaryDirectory() as directory:
            path = _save_split_metadata(Path(directory) / "model.pt", metadata)
            self.assertEqual("model.split.json", path.name)
            self.assertEqual(metadata, json.loads(path.read_text(encoding="utf-8")))
        summary = _split_summary(metadata)
        self.assertNotIn("session_id_by_window", summary)
        self.assertNotIn("window_indices", json.dumps(summary))
        self.assertIn("actual_window_fractions", summary)
        self.assertEqual(metadata["partitions"]["test"]["window_count"], summary["partitions"]["test"]["window_count"])

    def test_three_to_twenty_groups_always_keep_three_nonempty_partitions(self) -> None:
        for count in range(3, 21):
            labels = np.repeat(np.arange(count).astype(str), 2)
            split = _split_by_session(labels, 5)
            self.assertTrue(all(len(part) for part in split))
            self.assertEqual(len(labels), len(set(np.concatenate(split).tolist())))
            groups = [set(labels[part]) for part in split]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_fewer_than_three_real_groups_are_rejected(self) -> None:
        for groups in (["one"], ["one", "two"]):
            with self.subTest(groups=groups), self.assertRaisesRegex(ld.LearnedDenoiseError, "at least 3 real"):
                _split_by_session(np.repeat(groups, 100), 7)

    def test_missing_or_multidimensional_session_groups_are_rejected(self) -> None:
        for labels in (np.array([]), np.array([["a", "b", "c"]]), np.array(["a", "b", ""]), np.array(["a", "b", "nan"])):
            with self.subTest(labels=labels), self.assertRaises(ld.LearnedDenoiseError):
                _split_by_session(labels, 1)

    def test_heldout_selector_excludes_training_and_validation(self) -> None:
        selected = evaluation._select_evaluation_indices(self.checkpoint, self.sessions)
        np.testing.assert_array_equal(selected, self.indices[2])
        selected_groups = set(self.sessions[selected])
        self.assertFalse(selected_groups & set(self.checkpoint["train_sessions"]))
        self.assertFalse(selected_groups & set(self.checkpoint["validation_sessions"]))

    def test_all_pairwise_overlaps_are_rejected_even_in_diagnostics(self) -> None:
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            for diagnostic in (False, True):
                checkpoint = deepcopy(self.checkpoint)
                checkpoint[f"{right}_sessions"].append(checkpoint[f"{left}_sessions"][0])
                with self.subTest(left=left, right=right, diagnostic=diagnostic), self.assertRaisesRegex(ld.LearnedDenoiseError, "overlap"):
                    evaluation._select_evaluation_indices(checkpoint, self.sessions, diagnostic_all=diagnostic)

    def test_missing_split_cannot_claim_heldout(self) -> None:
        with self.assertRaises(ld.LearnedDenoiseError):
            evaluation._select_evaluation_indices({}, self.sessions)
        np.testing.assert_array_equal(evaluation._select_evaluation_indices({}, self.sessions, diagnostic_all=True), np.arange(len(self.sessions)))

    def test_missing_or_extra_dataset_session_is_rejected(self) -> None:
        for labels in (self.sessions[:-3], np.append(self.sessions, "unexpected")):
            with self.subTest(labels=labels), self.assertRaises(ld.LearnedDenoiseError):
                evaluation._select_evaluation_indices(self.checkpoint, labels)

    def test_hash_is_required_to_match_when_recorded(self) -> None:
        checkpoint = {**self.checkpoint, "dataset_sha256": "a" * 64}
        evaluation._select_evaluation_indices(checkpoint, self.sessions, dataset_sha256="a" * 64)
        for supplied in (None, "b" * 64):
            with self.assertRaisesRegex(ld.LearnedDenoiseError, "SHA256"):
                evaluation._select_evaluation_indices(checkpoint, self.sessions, dataset_sha256=supplied)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.npz"
            path.write_bytes(b"dataset bytes")
            self.assertEqual(hashlib.sha256(b"dataset bytes").hexdigest(), _dataset_sha256(path))

    def test_full_split_metadata_is_verified(self) -> None:
        metadata = _split_metadata(self.sessions, self.indices, 42)
        checkpoint = {**self.checkpoint, "split_metadata": metadata}
        evaluation._select_evaluation_indices(checkpoint, self.sessions)
        for field, value in (("window_count", 99), ("window_indices", [0]), ("sessions", ["s0", "s1"])):
            altered = deepcopy(checkpoint)
            altered["split_metadata"]["partitions"]["test"][field] = value
            with self.subTest(field=field), self.assertRaises(ld.LearnedDenoiseError):
                evaluation._select_evaluation_indices(altered, self.sessions)
        with self.assertRaisesRegex(ld.LearnedDenoiseError, "order"):
            evaluation._select_evaluation_indices(checkpoint, self.sessions[::-1])

    def test_diagnostic_switch_is_explicit_and_old_cli_still_parses(self) -> None:
        args = evaluation.build_parser().parse_args(["dataset.npz", "--model", "model.pt", "--device", "cpu"])
        self.assertFalse(args.diagnostic_all)
        args = evaluation.build_parser().parse_args(["dataset.npz", "--model", "model.pt", "--diagnostic-all"])
        self.assertTrue(args.diagnostic_all)


if ld.torch is not None:
    class StubModel(ld.torch.nn.Module):
        def __init__(self, bad: str | None = None) -> None:
            super().__init__()
            self.anchor = ld.torch.nn.Parameter(ld.torch.zeros(1))
            self.calls = 0
            self.inputs = []
            self.bad = bad

        def forward(self, signal, valid):
            self.calls += 1
            self.inputs.append(signal.clone())
            prediction = {
                "denoised": signal + 1,
                "artifact_probability": ld.torch.full_like(signal, 0.8),
                "confidence": ld.torch.ones_like(signal[:, :1]),
            }
            if self.bad in prediction:
                prediction[self.bad][:] = float("nan")
            elif self.bad == "infinity":
                prediction["denoised"][:] = float("inf")
            elif self.bad == "overflow":
                prediction["denoised"][:] = ld.torch.finfo(ld.torch.float32).max
            elif self.bad == "shape":
                prediction["confidence"] = prediction["confidence"][:, 0]
            elif self.bad == "probability":
                prediction["artifact_probability"][:] = 1.5
            elif self.bad == "range":
                prediction["confidence"][:] = 2
            return prediction


@unittest.skipIf(ld.torch is None, "optional PyTorch is not installed")
class InferenceSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = deepcopy(ld.DEFAULT_CONFIG)
        self.config.update(channel_count=2, window_samples=8, hop_samples=4)
        self.signal = np.vstack([np.arange(8), np.arange(8) + 5]).astype(np.float32)

    def bundle(self, bad=None):
        return ld.ModelBundle(StubModel(bad), self.config, Path("unused.pt"))

    def test_valid_prediction_is_used_without_mutating_input(self) -> None:
        signal = self.signal.copy()
        result = ld.apply_model_to_array(self.bundle(), signal)
        self.assertTrue(result["replacement_mask"].all())
        self.assertTrue(np.isfinite(result["denoised"]).all())
        self.assertFalse(np.array_equal(signal, result["denoised"]))
        np.testing.assert_array_equal(signal, self.signal)

    def test_shape_and_mask_broadcasting_and_nonbinary_values_are_rejected(self) -> None:
        for mask in (np.ones(8), np.ones((1, 8)), np.full((2, 8), np.nan), np.full((2, 8), 0.5)):
            with self.subTest(mask=mask), self.assertRaises(ld.LearnedDenoiseError):
                ld.apply_model_to_array(self.bundle(), self.signal, mask)
        for signal in (np.zeros(8), np.zeros((3, 8)), np.zeros((2, 0)), np.zeros((2, 8), dtype=complex), np.full((2, 8), 1e100)):
            with self.subTest(shape=signal.shape), self.assertRaises(ld.LearnedDenoiseError):
                ld.apply_model_to_array(self.bundle(), signal)

    def test_bad_window_does_not_infer_or_replace_despite_high_confidence(self) -> None:
        signal = self.signal.copy()
        signal[:, :3] = np.nan
        bundle = self.bundle()
        result = ld.apply_model_to_array(bundle, signal)
        self.assertEqual(0, bundle.model.calls)
        self.assertFalse(result["replacement_mask"].any())
        self.assertTrue((result["confidence"] == 0).all())
        np.testing.assert_array_equal(result["denoised"], signal)
        np.testing.assert_array_equal(result["model_denoised"], signal)

    def test_one_wholly_missing_channel_cannot_hide_in_global_fraction(self) -> None:
        self.config["channel_count"] = 8
        signal = np.ones((8, 8), dtype=np.float32)
        signal[0] = np.nan  # Global missing fraction is only 12.5%.
        result = ld.apply_model_to_array(self.bundle(), signal)
        self.assertFalse(result["replacement_mask"].any())

    def test_per_real_window_gating_and_overlap_veto(self) -> None:
        signal = np.vstack([np.arange(12), np.arange(12) + 5]).astype(np.float32)
        signal[:, 8:] = np.nan
        bundle = self.bundle()
        result = ld.apply_model_to_array(bundle, signal)
        self.assertEqual(1, bundle.model.calls)
        self.assertTrue(result["replacement_mask"][:, :4].all())
        self.assertFalse(result["replacement_mask"][:, 4:].any())
        np.testing.assert_array_equal(result["denoised"][:, 4:], signal[:, 4:])
        self.assertTrue((result["confidence"][4:] == 0).all())

    def test_padding_does_not_count_as_acquisition_missing_samples(self) -> None:
        result = ld.apply_model_to_array(self.bundle(), self.signal[:, :3])
        self.assertTrue(result["replacement_mask"].all())
        self.assertEqual((2, 3), result["denoised"].shape)

    def test_invalid_values_remain_original_even_if_preserve_flag_is_false(self) -> None:
        self.config["inference"]["preserve_invalid_samples"] = False
        for original in (np.nan, np.inf, -np.inf, 999.0):
            signal = self.signal.copy()
            signal[0, 2] = original
            mask = np.ones_like(signal, dtype=bool)
            mask[0, 2] = False
            bundle = self.bundle()
            result = ld.apply_model_to_array(bundle, signal, mask)
            self.assertFalse(result["replacement_mask"][0, 2])
            np.testing.assert_array_equal(result["denoised"][0, 2:3], signal[0, 2:3])
            np.testing.assert_array_equal(result["model_denoised"][0, 2:3], signal[0, 2:3])
            self.assertEqual(0.0, float(bundle.model.inputs[0][0, 0, 2]))

    def test_bad_predictions_safely_fall_back(self) -> None:
        for bad in ("denoised", "artifact_probability", "confidence", "infinity", "overflow", "shape", "probability", "range"):
            with self.subTest(bad=bad):
                result = ld.apply_model_to_array(self.bundle(bad), self.signal)
                self.assertFalse(result["replacement_mask"].any())
                self.assertTrue((result["confidence"] == 0).all())
                np.testing.assert_array_equal(result["denoised"], self.signal)
                np.testing.assert_array_equal(result["model_denoised"], self.signal)


@unittest.skipIf(ld.torch is None, "optional PyTorch is not installed")
class CheckpointAndCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = deepcopy(ld.DEFAULT_CONFIG)
        self.config.update(channel_count=2, window_samples=8, hop_samples=4)
        self.checkpoint = {
            "model_state": ld.ResidualTCNDenoiser(2, hidden_channels=4).state_dict(),
            "hidden_channels": 4,
            "model_version": ld.MODEL_VERSION,
            "config": deepcopy(self.config),
            "train_sessions": ["train"], "validation_sessions": ["validation"], "test_sessions": ["test"],
        }
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "model.pt"
        ld.torch.save(self.checkpoint, self.path)

    def test_load_uses_weights_only_and_legacy_checkpoint_is_readable(self) -> None:
        with patch.object(ld.torch, "load", wraps=ld.torch.load) as loader:
            bundle = ld.load_model(self.path, self.config)
        self.assertTrue(loader.call_args.kwargs["weights_only"])
        self.assertEqual(["test"], bundle.checkpoint_metadata["test_sessions"])
        self.assertNotIn("model_state", bundle.checkpoint_metadata)

    def test_version_sample_rate_channels_and_channel_order_mismatches_fail(self) -> None:
        for key, value in (("model_version", "unsupported"), ("sample_rate_hz", 256), ("channel_count", 8), ("channel_order", ["B", "A"]), ("window_samples", 16)):
            altered = deepcopy(self.checkpoint)
            altered["config"][key] = value
            ld.torch.save(altered, self.path)
            with self.subTest(key=key), self.assertRaises(ld.LearnedDenoiseError):
                ld.load_model(self.path, self.config)
        altered = {**self.checkpoint, "checkpoint_schema_version": 999}
        ld.torch.save(altered, self.path)
        with self.assertRaisesRegex(ld.LearnedDenoiseError, "checkpoint_schema_version"):
            ld.load_model(self.path, self.config)

    def test_evaluation_cli_only_runs_test_windows_and_labels_diagnostics(self) -> None:
        dataset = Path(self.directory.name) / "data.npz"
        config_path = Path(self.directory.name) / "config.json"
        config_path.write_text(json.dumps(self.config), encoding="utf-8")
        signal = np.tile(np.arange(8, dtype=np.float32), (3, 2, 1))
        np.savez(dataset, noisy=signal, clean=signal, artifact_mask=np.zeros_like(signal), session_id=np.array(["train", "validation", "test"]))
        self.checkpoint["dataset_sha256"] = _dataset_sha256(dataset)
        bundle = ld.ModelBundle(StubModel(), self.config, self.path, self.checkpoint)
        base = [str(dataset), "--model", str(self.path), "--config", str(config_path)]
        for diagnostic, expected_windows in ((False, 1), (True, 3)):
            bundle.model.calls = 0
            output = io.StringIO()
            with patch.object(evaluation, "load_model", return_value=bundle), patch("sys.stdout", output):
                code = evaluation.main(base + (["--diagnostic-all"] if diagnostic else []))
            self.assertEqual(0, code, output.getvalue())
            report = json.loads(output.getvalue())
            self.assertNotIn("window_indices", report)
            self.assertEqual(expected_windows, bundle.model.calls)
            self.assertEqual(expected_windows, report["windows"])
            self.assertEqual(not diagnostic, report["heldout"])
            self.assertEqual("all_diagnostic_not_heldout" if diagnostic else "heldout_test", report["evaluation_scope"])
            self.assertTrue(report["dataset_hash_verified"])
            self.assertIn("clean_ish", report["target_provenance"])
            if not diagnostic:
                self.assertEqual(["test"], report["session_groups"])
        self.checkpoint["dataset_sha256"] = "f" * 64
        with patch.object(evaluation, "load_model", return_value=bundle), patch("sys.stdout", io.StringIO()):
            self.assertEqual(2, evaluation.main(base))


class FailedValidationSafetyTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def setUp(self) -> None:
        self.config = ld.load_config(self.ROOT / "configs" / "learned_denoiser_v1.json")

    def test_default_automatic_processing_is_disabled_but_config_remains_formal(self) -> None:
        self.assertIs(False, self.config["runtime"]["enabled"])
        self.assertEqual("formal", self.config["status"])
        self.assertEqual("shadow", self.config["inference"]["mode"])
        self.assertEqual("failed_holdout", self.config["runtime"]["validation_status"])
        self.assertIs(False, self.config["runtime"]["production_validated"])
        self.assertTrue(self.config["runtime"]["validation_reason"])
        self.assertEqual("analysis_outputs/learned_denoiser_v1_heldout.json", self.config["runtime"]["validation_reports"]["heldout"])
        self.assertEqual("analysis_outputs/ssvep_preservation_v1_session_split.json", self.config["runtime"]["validation_reports"]["ssvep_preservation"])
        ld.validate_config(self.config)

    def launcher_check(self, config: dict, model_present: bool = False) -> dict:
        source = (self.ROOT / "scripts" / "start_learned_workstation.ps1").read_text(encoding="utf-8")
        check = source.split("$DependencyCheck = @'", 1)[1].split("\n'@", 1)[0].strip()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "configs" / "learned_denoiser_v1.json").write_text(json.dumps(config), encoding="utf-8")
            if model_present:
                model = root / config["runtime"]["model_path"]
                model.parent.mkdir(parents=True)
                model.write_bytes(b"must not load a failed model")
            # Model readiness tests must not construct a board or touch hardware.
            board_module = types.ModuleType("brainflow.board_shim")
            board_module.BoardShim = Mock(side_effect=AssertionError("hardware must not be constructed"))
            brainflow_module = types.ModuleType("brainflow")
            brainflow_module.__path__ = []
            pyside = types.ModuleType("PySide6")
            pyside.QtCore = types.ModuleType("PySide6.QtCore")
            output = io.StringIO()
            with patch.dict("sys.modules", {"brainflow": brainflow_module, "brainflow.board_shim": board_module, "PySide6": pyside}), \
                    patch.object(Path, "cwd", return_value=root), \
                    patch.object(ld, "load_model", side_effect=AssertionError("disabled model must not be loaded")), \
                    patch("sys.stdout", output):
                exec(compile(check, "<launcher_dependency_check>", "exec"), {})
            board_module.BoardShim.assert_not_called()
            return json.loads(output.getvalue())

    def test_launcher_reports_pipeline_ready_not_model_pass_when_model_is_present(self) -> None:
        report = self.launcher_check(self.config, model_present=True)
        self.assertEqual("pipeline_ready", report["status"])
        self.assertIs(True, report["pipeline_ready"])
        self.assertIs(True, report["automatic_shadow_disabled"])
        self.assertIs(False, report["shadow_enabled"])
        self.assertEqual("failed", report["model_validation"])
        self.assertEqual("failed_holdout", report["validation_status"])
        self.assertEqual("manual_gui_only", report["execution"])
        self.assertEqual("skipped_automatic_shadow_disabled", report["model_load_status"])
        self.assertIs(False, report["hardware_opened"])
        self.assertIs(False, report["visual_stimulus_started"])

    def test_launcher_missing_model_is_safe_for_disabled_automatic_processing(self) -> None:
        report = self.launcher_check(self.config)
        self.assertIs(False, report["model_available"])
        self.assertIs(True, report["pipeline_ready"])
        self.assertIs(True, report["automatic_shadow_disabled"])
        self.assertEqual("failed", report["model_validation"])

    def test_launcher_refuses_flipping_enabled_for_failed_model(self) -> None:
        altered = deepcopy(self.config)
        altered["runtime"]["enabled"] = True
        with self.assertRaisesRegex(SystemExit, "Automatic shadow refused.*failed"):
            self.launcher_check(altered, model_present=True)

    def test_launcher_enabled_with_missing_model_is_safely_refused(self) -> None:
        altered = deepcopy(self.config)
        altered["runtime"].update(enabled=True, validation_status="passed")
        with self.assertRaisesRegex(SystemExit, "model missing"):
            self.launcher_check(altered)

    def test_offline_audit_metadata_marks_failed_validation_not_release_success(self) -> None:
        audit = offline._diagnostic_audit_metadata(self.config)
        self.assertEqual("failed_holdout", audit["model_validation_status"])
        self.assertIs(False, audit["automatic_processing_enabled"])
        self.assertIs(True, audit["diagnostic_only"])
        self.assertIs(False, audit["release_gate_passed"])
        self.assertIn("failed", audit["warning"].lower())
        self.assertIn("NOT", audit["warning"])

    def test_disabled_runtime_does_not_block_explicit_offline_cli_and_both_reports_are_audited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "session"
            session.mkdir()
            raw = session / "raw_brainflow.tsv"
            raw.write_bytes(b"original acquisition bytes")
            before = offline._sha256(raw)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(self.config), encoding="utf-8")
            model_path = root / "model.pt"
            model_path.write_bytes(b"mock checkpoint")
            output_dir = session / "derived" / "diagnostic"
            signal = np.arange(64, dtype=np.float32).reshape(8, 8)
            valid = np.ones_like(signal, dtype=bool)
            prediction = {"denoised": signal.copy(), "artifact_probability": np.zeros_like(signal),
                          "confidence": np.ones(8, dtype=np.float32), "valid_mask": valid}
            with patch.object(offline, "read_session_eeg", return_value=(signal, 250.0, valid)), \
                    patch.object(offline, "load_model", return_value=object()) as loader, \
                    patch.object(offline, "apply_model_to_array", return_value=prediction), \
                    patch("sys.stdout", io.StringIO()):
                code = offline.main([str(session), "--model", str(model_path), "--config", str(config_path), "--output-dir", str(output_dir)])
            self.assertEqual(0, code)
            loader.assert_called_once()
            self.assertEqual(before, offline._sha256(raw))
            for filename in ("quality_metrics.json", "model_manifest.json"):
                report = json.loads((output_dir / filename).read_text(encoding="utf-8"))
                self.assertEqual("failed_holdout", report["model_validation_status"])
                self.assertIs(False, report["automatic_processing_enabled"])
                self.assertIs(True, report["diagnostic_only"])
                self.assertIs(False, report["release_gate_passed"])
                self.assertIn("failed", report["warning"].lower())
            metrics = json.loads((output_dir / "quality_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual("completed", metrics["status"])
            self.assertEqual(0.0, metrics["replacement_fraction"])
            self.assertEqual(0.0, metrics["artifact_fraction"])
            with np.load(output_dir / "denoised_signal.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(signal, saved["denoised"])


if __name__ == "__main__":
    unittest.main()
