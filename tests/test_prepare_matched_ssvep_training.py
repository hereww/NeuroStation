"""Real MAT-file regression tests for the exploratory data preparer.

Synthetic fixtures describe their own explicit schema; they assert no Gu2024
metadata and require neither a downloaded dataset nor a trained model.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np
from scipy.io import loadmat, savemat
from scipy.io.matlab import MatReadError

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_matched_ssvep_training.py"
MODULE_SPEC = importlib.util.spec_from_file_location("matched_ssvep_preparer", SCRIPT)
assert MODULE_SPEC and MODULE_SPEC.loader
preparer = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(preparer)

# Deliberately scattered and reordered in the 64-channel source.
CHANNEL_INDICES = (53, 6, 41, 13, 60, 19, 32, 2)
FREQUENCIES = (10.0, 12.0, 15.0, 20.0)
CHANNEL_NAMES = ("Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2")


class PrepareMatchedSSVEPTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_source(self, *, permutation=(0, 1, 2, 3), samples=9,
                    rows=None, subject=7, dtype=np.int32, compressed=True):
        if rows is None:
            rows = [{"condition_index": i, "frequency_hz": frequency,
                     "depth": 1, "session": "session-A"}
                    for i, frequency in enumerate(FREQUENCIES)]
        shape = (64, samples, len(rows), 2)
        channel, sample, condition, block = np.indices(shape, dtype=np.int32)
        canonical = (channel * 100000 + sample * 100 + condition * 2 + block).astype(dtype)
        source = np.transpose(canonical, permutation)
        path = self.root / "source.mat"
        savemat(path, {"EEG": source, "unrelated": np.arange(3)}, do_compression=compressed)
        names = [f"synthetic-channel-{i}" for i in range(64)]
        for index, name in zip(CHANNEL_INDICES, CHANNEL_NAMES):
            names[index] = name
        spec = {
            "schema_version": 1, "status": "formal", "sample_rate_hz": 250,
            "mat_variable": "EEG", "expected_shape": list(source.shape),
            "axes": {name: permutation.index(i)
                     for i, name in enumerate(("channel", "sample", "condition", "block"))},
            "channel_names": names, "subject_id": subject,
            # The preparer must follow original condition indices, not JSON row order.
            "condition_table": list(reversed(copy.deepcopy(rows))),
        }
        return path, spec, canonical

    def assert_source_unchanged(self, path, original_bytes, original_mtime) -> None:
        self.assertEqual(path.read_bytes(), original_bytes)
        self.assertEqual(path.stat().st_mtime_ns, original_mtime)

    def assert_flags(self, values) -> None:
        def scalar(key):
            value = values[key]
            return value.item() if isinstance(value, np.ndarray) else value
        self.assertEqual(scalar("status"), "formal")
        self.assertIs(scalar("exploratory_only"), True)
        for key in ("runtime_enabled", "production_validated", "training_executed",
                    "deployment_executed", "default_deployment_authorized"):
            self.assertIs(scalar(key), False)

    def assert_rejected(self, path, spec, *, message=None, output=None, **kwargs):
        output = output or self.root / "rejected"
        original_bytes, original_mtime = path.read_bytes(), path.stat().st_mtime_ns
        with self.assertRaises(preparer.MatchedSSVEPPreparationError) as caught:
            preparer.prepare_matched_ssvep_training(path, spec, output, **kwargs)
        if message is not None:
            self.assertIn(message, str(caught.exception))
        self.assertFalse(output.exists())
        self.assert_source_unchanged(path, original_bytes, original_mtime)
        return caught.exception

    def test_all_axis_permutations_preserve_exact_values_channels_and_conditions(self) -> None:
        for permutation in itertools.permutations(range(4)):
            with self.subTest(permutation=permutation):
                rows = [{"condition_index": i, "frequency_hz": frequency,
                         "depth": "depth-A", "session": 3}
                        for i, frequency in enumerate((15.0, 11.0, 20.0, 10.0, 12.0))]
                path, spec, canonical = self.make_source(permutation=permutation, rows=rows)
                original_bytes, original_mtime = path.read_bytes(), path.stat().st_mtime_ns
                original_spec = copy.deepcopy(spec)
                output = self.root / ("axes-" + "".join(map(str, permutation)))
                manifest = preparer.prepare_matched_ssvep_training(path, spec, output)
                saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(saved, manifest)
                self.assertEqual(spec, original_spec)
                selected = (0, 2, 3, 4)
                with np.load(output / manifest["npz_file"], allow_pickle=False) as data:
                    expected = np.stack([canonical[list(CHANNEL_INDICES), :, condition, block]
                                         for condition in selected for block in range(2)])
                    np.testing.assert_array_equal(data["eeg"], expected)
                    self.assertEqual(data["eeg"].dtype, canonical.dtype)
                    np.testing.assert_array_equal(data["channel_names"], CHANNEL_NAMES)
                    np.testing.assert_array_equal(data["source_channel_indices"], CHANNEL_INDICES)
                    np.testing.assert_array_equal(data["condition_index"], np.repeat(selected, 2))
                    np.testing.assert_array_equal(data["frequency_hz"], [15, 15, 20, 20, 10, 10, 12, 12])
                    np.testing.assert_array_equal(data["block"], [0, 1] * 4)
                    np.testing.assert_array_equal(data["source_sample_indices"], np.arange(9))
                    self.assert_flags(data)
                    self.assertEqual(data["source_sha256"].item(), hashlib.sha256(original_bytes).hexdigest())
                self.assert_flags(manifest)
                self.assertEqual(manifest["true_frequencies_hz"], list(FREQUENCIES))
                self.assertEqual(manifest["source_channel_indices"], list(CHANNEL_INDICES))
                self.assertFalse(manifest["split_assigned"])
                self.assertFalse(manifest["training_authorized"])
                self.assertFalse(manifest["signal_processing_applied"])
                self.assertFalse(manifest["raw_reference_is_clean_ground_truth"])
                self.assertEqual(manifest["npz_sha256"], hashlib.sha256((output / manifest["npz_file"]).read_bytes()).hexdigest())
                self.assert_source_unchanged(path, original_bytes, original_mtime)
                np.testing.assert_array_equal(loadmat(path)["EEG"], np.transpose(canonical, permutation))

    def test_all_depth_session_conditions_survive_with_typed_groups_shared_across_frequencies(self) -> None:
        rows = []
        for depth, session, frequency in itertools.product((1, "1"), (1, "1"), FREQUENCIES):
            rows.append({"condition_index": len(rows), "frequency_hz": frequency,
                         "depth": depth, "session": session})
        rows.append({"condition_index": len(rows), "frequency_hz": 20.00000001,
                     "depth": "other", "session": "other"})
        for subject in (7, "7", 7.0):
            with self.subTest(subject=subject, type=type(subject)):
                path, spec, canonical = self.make_source(rows=rows, subject=subject)
                output = self.root / f"identities-{type(subject).__name__}"
                manifest = preparer.prepare_matched_ssvep_training(path, spec, output)
                self.assertEqual(manifest["selected_conditions"], rows[:-1])
                self.assertEqual(manifest["eeg_shape"], [32, 8, 9])
                with np.load(output / manifest["npz_file"], allow_pickle=False) as data:
                    grouped = {}
                    for trial, (row, block) in enumerate(itertools.product(rows[:-1], range(2))):
                        for key, original in (("subject", subject), ("depth", row["depth"]),
                                              ("session", row["session"])):
                            decoded = json.loads(data[key][trial])
                            self.assertEqual(decoded, original)
                            self.assertIs(type(decoded), type(original))
                        group = json.loads(data["group_id"][trial])
                        self.assertEqual(group, [subject, block, row["depth"], row["session"]])
                        for actual, original in zip(group, [subject, block, row["depth"], row["session"]]):
                            self.assertIs(type(actual), type(original))
                        self.assertEqual(data["condition_index"][trial], row["condition_index"])
                        np.testing.assert_array_equal(data["eeg"][trial], canonical[list(CHANNEL_INDICES), :, row["condition_index"], block])
                        grouped.setdefault(data["group_id"][trial], []).append(data["frequency_hz"][trial])
                    # Splitting by these groups cannot put different frequencies of
                    # the same subject/block/depth/session into different splits.
                    self.assertEqual(len(grouped), 8)
                    for frequencies in grouped.values():
                        self.assertEqual(sorted(frequencies), list(FREQUENCIES))

    def test_explicit_five_second_window_has_exactly_1250_original_samples(self) -> None:
        path, spec, canonical = self.make_source(samples=1285, permutation=(2, 0, 3, 1), dtype=np.float32)
        original_bytes, original_mtime = path.read_bytes(), path.stat().st_mtime_ns
        spec["sample_window"] = {"start": 17, "stop": 1267}
        output = self.root / "five-seconds"
        manifest = preparer.prepare_matched_ssvep_training(path, spec, output)
        self.assertEqual(manifest["sample_count"], 1250)
        self.assertEqual(manifest["source_sample_count"], 1285)
        self.assertEqual(manifest["duration_seconds"], 5.0)
        with np.load(output / manifest["npz_file"], allow_pickle=False) as data:
            self.assertEqual(data["eeg"].shape, (8, 8, 1250))
            self.assertEqual(data["eeg"].dtype, np.dtype("float32"))
            np.testing.assert_array_equal(data["source_sample_indices"], np.arange(17, 1267))
            expected = np.stack([canonical[list(CHANNEL_INDICES), 17:1267, condition, block]
                                 for condition in range(4) for block in range(2)])
            np.testing.assert_array_equal(data["eeg"], expected)
        self.assert_source_unchanged(path, original_bytes, original_mtime)

    def test_no_window_retains_1285_samples_and_does_not_claim_five_seconds(self) -> None:
        path, spec, _ = self.make_source(samples=1285)
        output = self.root / "full-window"
        manifest = preparer.prepare_matched_ssvep_training(path, spec, output)
        self.assertEqual(manifest["sample_count"], 1285)
        self.assertEqual(manifest["duration_seconds"], 5.14)
        self.assertEqual(manifest["sample_window"], {"start": 0, "stop": 1285})
        with np.load(output / manifest["npz_file"], allow_pickle=False) as data:
            self.assertEqual(data["eeg"].shape[-1], 1285)
            np.testing.assert_array_equal(data["source_sample_indices"], np.arange(1285))

    def test_missing_twenty_hz_is_rejected_without_rounding_or_partial_output(self) -> None:
        for substitute in (19.0, 19.999999999, 20.000000001):
            with self.subTest(substitute=substitute):
                path, spec, _ = self.make_source()
                spec["condition_table"][0]["frequency_hz"] = substitute
                error = self.assert_rejected(path, spec, message="missing exact target")
                self.assertEqual(error.audit["missing_target_frequencies_hz"], [20.0])
                self.assertEqual(error.audit["missing_channel_names"], [])
                self.assertIn(substitute, error.audit["available_frequencies_hz"])

    def test_missing_exact_electrode_is_rejected_without_channel_repair(self) -> None:
        path, spec, _ = self.make_source()
        spec["channel_names"][CHANNEL_INDICES[0]] = "fp1"
        error = self.assert_rejected(path, spec, message="missing exact target")
        self.assertEqual(error.audit["missing_channel_names"], ["Fp1"])
        self.assertEqual(error.audit["missing_target_frequencies_hz"], [])

    def test_illegal_metadata_is_rejected_before_output(self) -> None:
        path, valid, _ = self.make_source()
        cases = [
            ("draft", lambda s: s.update(status="draft")),
            ("boolean schema", lambda s: s.update(schema_version=True)),
            ("resampling", lambda s: s.update(sample_rate_hz=256)),
            ("boolean rate", lambda s: s.update(sample_rate_hz=True)),
            ("unknown field", lambda s: s.update(gu_metadata="unverified")),
            ("empty variable", lambda s: s.update(mat_variable=" ")),
            ("duplicate axes", lambda s: s["axes"].update(block=0)),
            ("boolean axis", lambda s: s["axes"].update(channel=False)),
            ("extra axis", lambda s: s["axes"].update(depth=4)),
            ("63 channels", lambda s: s["expected_shape"].__setitem__(0, 63)),
            ("zero samples", lambda s: s["expected_shape"].__setitem__(1, 0)),
            ("boolean shape", lambda s: s["expected_shape"].__setitem__(3, True)),
            ("short channel list", lambda s: s["channel_names"].pop()),
            ("duplicate electrode", lambda s: s["channel_names"].__setitem__(0, "FP1")),
            ("whitespace electrode", lambda s: s["channel_names"].__setitem__(0, " X ")),
            ("empty subject", lambda s: s.update(subject_id=" ")),
            ("boolean subject", lambda s: s.update(subject_id=True)),
            ("NaN subject", lambda s: s.update(subject_id=float("nan"))),
            ("infinite depth", lambda s: s["condition_table"][0].update(depth=float("inf"))),
            ("object session", lambda s: s["condition_table"][0].update(session={})),
            ("empty session", lambda s: s["condition_table"][0].update(session="")),
            ("missing condition", lambda s: s["condition_table"].pop()),
            ("duplicate index", lambda s: s["condition_table"][0].update(condition_index=0)),
            ("negative index", lambda s: s["condition_table"][0].update(condition_index=-1)),
            ("boolean index", lambda s: s["condition_table"][0].update(condition_index=True)),
            ("NaN frequency", lambda s: s["condition_table"][0].update(frequency_hz=float("nan"))),
            ("Nyquist frequency", lambda s: s["condition_table"][0].update(frequency_hz=125)),
            ("extra row field", lambda s: s["condition_table"][0].update(block=1)),
            ("negative window", lambda s: s.update(sample_window={"start": -1, "stop": 5})),
            ("empty window", lambda s: s.update(sample_window={"start": 5, "stop": 5})),
            ("out of range window", lambda s: s.update(sample_window={"start": 0, "stop": 10})),
            ("boolean window", lambda s: s.update(sample_window={"start": False, "stop": 5})),
        ]
        for name, mutate in cases:
            with self.subTest(case=name):
                spec = copy.deepcopy(valid)
                mutate(spec)
                self.assert_rejected(path, spec)

    def test_explicit_api_overrides_cannot_change_formal_spec(self) -> None:
        path, spec, _ = self.make_source()
        self.assert_rejected(path, spec, mat_variable="other", message="mat_variable does not match")
        self.assert_rejected(path, spec, axis_spec={"channel": 1, "sample": 0, "condition": 2, "block": 3}, message="axis_spec does not match")

    def test_nan_or_infinity_anywhere_in_source_is_rejected(self) -> None:
        # The bad value is outside both the selected channels and sample window.
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                path, spec, canonical = self.make_source(dtype=np.float64)
                canonical[0, 0, 0, 0] = value
                savemat(path, {"EEG": canonical}, do_compression=True)
                spec["sample_window"] = {"start": 1, "stop": 5}
                self.assert_rejected(path, spec, message="non-finite")

    def test_missing_variable_wrong_shape_struct_and_complex_data_are_rejected(self) -> None:
        for case in ("missing", "shape", "struct", "complex"):
            with self.subTest(case=case):
                path, spec, canonical = self.make_source()
                if case == "missing":
                    savemat(path, {"other": canonical})
                elif case == "shape":
                    savemat(path, {"EEG": canonical[:, :-1]})
                elif case == "struct":
                    savemat(path, {"EEG": {"nested": canonical}})
                else:
                    savemat(path, {"EEG": canonical.astype(np.complex64) + 1j})
                self.assert_rejected(path, spec)

    def test_empty_and_truncated_real_mat_files_fail_cleanly(self) -> None:
        for compressed in (False, True):
            for remaining in (0, 10, 150, -16):
                with self.subTest(compressed=compressed, remaining=remaining):
                    path, spec, canonical = self.make_source(compressed=compressed)
                    # Cut the selected EEG payload, not the unused variable that
                    # follows it in the general fixture.
                    savemat(path, {"EEG": canonical}, do_compression=compressed)
                    path.write_bytes(path.read_bytes()[:remaining])
                    self.assert_rejected(path, spec, message="unreadable MAT",
                                         output=self.root / f"truncated-{compressed}-{remaining}")

    def test_corrupt_real_compressed_mat_fails_cleanly(self) -> None:
        path, spec, canonical = self.make_source()
        savemat(path, {"EEG": canonical}, do_compression=True)
        raw = bytearray(path.read_bytes())
        # Damage the compressed stream checksum after its valid MAT header.
        raw[-1] ^= 1
        path.write_bytes(raw)
        self.assert_rejected(path, spec, message="unreadable MAT")

    def test_scipy_reader_errors_are_normalized_at_header_and_payload_stages(self) -> None:
        path, spec, _ = self.make_source()
        for reader in ("whosmat", "loadmat"):
            for error in (MatReadError("truncated"), zlib.error("damaged compression")):
                with self.subTest(reader=reader, error=type(error).__name__):
                    with patch.object(preparer, reader, side_effect=error):
                        self.assert_rejected(path, spec, message="unreadable MAT")

    def test_invalid_memory_limit_and_source_over_limit_are_rejected(self) -> None:
        path, spec, canonical = self.make_source()
        for limit in (True, 0, -1, 1.5, canonical.nbytes):
            with self.subTest(limit=limit):
                self.assert_rejected(path, spec, max_array_bytes=limit)

    def test_existing_directory_or_file_is_never_overwritten(self) -> None:
        path, spec, _ = self.make_source()
        original_bytes, original_mtime = path.read_bytes(), path.stat().st_mtime_ns
        for case in ("empty-dir", "populated-dir", "file"):
            with self.subTest(case=case):
                output = self.root / case
                if case == "file":
                    output.write_bytes(b"caller-owned file")
                else:
                    output.mkdir()
                    if case == "populated-dir":
                        (output / "matched_ssvep_training.npz").write_bytes(b"caller-owned NPZ")
                        (output / "manifest.json").write_bytes(b"caller-owned manifest")
                before = {str(p): p.read_bytes() for p in output.iterdir()} if output.is_dir() else output.read_bytes()
                with self.assertRaisesRegex(preparer.MatchedSSVEPPreparationError, "refusing to overwrite"):
                    preparer.prepare_matched_ssvep_training(path, spec, output)
                after = {str(p): p.read_bytes() for p in output.iterdir()} if output.is_dir() else output.read_bytes()
                self.assertEqual(before, after)
        self.assert_source_unchanged(path, original_bytes, original_mtime)

    def test_second_run_cannot_overwrite_successful_output(self) -> None:
        path, spec, _ = self.make_source()
        output = self.root / "prepared"
        preparer.prepare_matched_ssvep_training(path, spec, output)
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        with self.assertRaisesRegex(preparer.MatchedSSVEPPreparationError, "refusing to overwrite"):
            preparer.prepare_matched_ssvep_training(path, spec, output)
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_dangling_output_symlink_is_rejected(self) -> None:
        path, spec, _ = self.make_source()
        output, target = self.root / "existing-link", self.root / "absent-target"
        try:
            output.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")
        with self.assertRaisesRegex(preparer.MatchedSSVEPPreparationError, "refusing to overwrite"):
            preparer.prepare_matched_ssvep_training(path, spec, output)
        self.assertTrue(output.is_symlink())
        self.assertFalse(target.exists())

    def test_existing_symlink_guard_runs_before_path_resolution(self) -> None:
        # Exercise the guard even on Windows without symlink-creation privilege.
        path, spec, _ = self.make_source()
        output = self.root / "mocked-dangling-link"
        original_is_symlink = Path.is_symlink
        def is_symlink(candidate):
            return candidate == output or original_is_symlink(candidate)
        with patch.object(Path, "is_symlink", is_symlink):
            self.assert_rejected(path, spec, output=output, message="refusing to overwrite")

    def test_directory_created_concurrently_is_preserved(self) -> None:
        path, spec, _ = self.make_source()
        output = self.root / "race"
        real_loader = preparer._load_mat_variable
        def concurrent_loader(*args, **kwargs):
            result = real_loader(*args, **kwargs)
            output.mkdir()
            (output / "owned.txt").write_bytes(b"other invocation")
            return result
        with patch.object(preparer, "_load_mat_variable", side_effect=concurrent_loader):
            with self.assertRaises(FileExistsError):
                preparer.prepare_matched_ssvep_training(path, spec, output)
        self.assertEqual((output / "owned.txt").read_bytes(), b"other invocation")
        self.assertEqual([p.name for p in output.iterdir()], ["owned.txt"])

    def test_failed_payload_or_manifest_write_removes_only_new_outputs(self) -> None:
        path, spec, _ = self.make_source()
        original_bytes, original_mtime = path.read_bytes(), path.stat().st_mtime_ns
        for stage in ("npz", "manifest"):
            with self.subTest(stage=stage):
                output = self.root / f"failed-{stage}"
                if stage == "npz":
                    def fail_npz(handle, **kwargs):
                        handle.write(b"partial NPZ")
                        raise OSError("simulated full disk")
                    mocked = patch.object(preparer.np, "savez_compressed", side_effect=fail_npz)
                else:
                    def fail_manifest(value, handle, **kwargs):
                        handle.write("partial manifest")
                        raise OSError("simulated full disk")
                    mocked = patch.object(preparer.json, "dump", side_effect=fail_manifest)
                with mocked, self.assertRaisesRegex(OSError, "full disk"):
                    preparer.prepare_matched_ssvep_training(path, spec, output)
                self.assertFalse(output.exists())
                self.assert_source_unchanged(path, original_bytes, original_mtime)

    def test_cleanup_preserves_unrelated_concurrent_file(self) -> None:
        path, spec, _ = self.make_source()
        output = self.root / "failed-with-other-file"
        def fail_npz(handle, **kwargs):
            (output / "owned.txt").write_bytes(b"caller-owned")
            handle.write(b"partial NPZ")
            raise OSError("simulated failure")
        with patch.object(preparer.np, "savez_compressed", side_effect=fail_npz):
            with self.assertRaises(OSError):
                preparer.prepare_matched_ssvep_training(path, spec, output)
        self.assertEqual([p.name for p in output.iterdir()], ["owned.txt"])
        self.assertEqual((output / "owned.txt").read_bytes(), b"caller-owned")

    def test_source_change_during_read_prevents_publication(self) -> None:
        path, spec, _ = self.make_source()
        real_loader = preparer._load_mat_variable
        def mutate_source(*args, **kwargs):
            result = real_loader(*args, **kwargs)
            with path.open("ab") as handle:
                handle.write(b"external concurrent change")
            return result
        with patch.object(preparer, "_load_mat_variable", side_effect=mutate_source):
            with self.assertRaisesRegex(preparer.MatchedSSVEPPreparationError, "source changed"):
                preparer.prepare_matched_ssvep_training(path, spec, self.root / "changed")
        self.assertFalse((self.root / "changed").exists())

    def test_source_json_rejects_duplicate_keys_and_nonfinite_metadata(self) -> None:
        _, spec, _ = self.make_source()
        spec_path = self.root / "spec.json"
        for text in (json.dumps(spec).replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'),
                     json.dumps({**spec, "subject_id": float("nan")})):
            with self.subTest(text=text[:80]):
                spec_path.write_text(text, encoding="utf-8")
                with self.assertRaises(preparer.MatchedSSVEPPreparationError):
                    preparer.load_source_spec(spec_path)

    def test_cli_success_and_read_failure_keep_formal_exploratory_flags(self) -> None:
        from contextlib import redirect_stdout
        from io import StringIO
        path, spec, _ = self.make_source()
        spec_path = self.root / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        output = self.root / "cli-output"
        args = ["--mat", str(path), "--source-spec", str(spec_path), "--output-dir", str(output)]
        captured = StringIO()
        with redirect_stdout(captured):
            self.assertEqual(preparer.main(args), 0)
        self.assert_flags(json.loads(captured.getvalue()))
        path.write_bytes(b"")
        args[-1] = str(self.root / "cli-failed")
        captured = StringIO()
        with redirect_stdout(captured):
            self.assertEqual(preparer.main(args), 2)
        error = json.loads(captured.getvalue())
        self.assert_flags(error)
        self.assertIn("unreadable MAT", error["error"])
        self.assertFalse(Path(args[-1]).exists())


if __name__ == "__main__":
    unittest.main()
