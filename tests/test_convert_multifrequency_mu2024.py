from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.convert_multifrequency_mu2024 import (  # noqa: E402
    TRIAL_SAMPLES,
    _matches_frequency_filter,
    apply_trigger_correction,
    convert_file,
    convert_files,
    find_trigger_onsets,
    frequency_tuple_for_stimulus,
    main,
    target_combinations,
    test_methods_for_session,
    write_conversion,
)


class MultiFrequencyMu2024ConverterTests(unittest.TestCase):
    def test_lexicographic_combination_mapping_and_stimulation_types(self) -> None:
        self.assertEqual(target_combinations(1), [(7,), (11,), (13,), (17,), (19,), (23,)])
        self.assertEqual(
            target_combinations(2),
            [
                (7, 11),
                (7, 13),
                (7, 17),
                (7, 19),
                (7, 23),
                (11, 13),
                (11, 17),
                (11, 19),
                (11, 23),
                (13, 17),
                (13, 19),
                (13, 23),
                (17, 19),
                (17, 23),
                (19, 23),
            ],
        )
        self.assertEqual(frequency_tuple_for_stimulus(6, 20), (17, 19, 23))

    def test_official_participant_test_sequence(self) -> None:
        self.assertEqual(test_methods_for_session(1, 1), ("T1", "T1", "T1", "T1"))
        self.assertEqual(test_methods_for_session(1, 2), ("T21", "T22", "T23"))
        self.assertEqual(test_methods_for_session(1, 5), ("T21", "T22", "T23"))
        self.assertEqual(test_methods_for_session(2, 2), ("T23", "T21", "T22"))
        self.assertEqual(test_methods_for_session(1, 6), ("T31", "T32"))
        self.assertEqual(test_methods_for_session(1, 9), ("T31", "T32"))

    def test_dead_time_and_all_official_correction_rules(self) -> None:
        trigger = np.zeros(5000, dtype=float)
        trigger[[511, 512, 1023, 1024, 1535]] = 1
        # MATLAB starts at one-based sample 512, therefore zero-based 511 is
        # the first eligible onset and an onset 512 samples later is eligible.
        self.assertEqual(find_trigger_onsets(trigger).tolist(), [511, 1023, 1535])

        raw = np.arange(60, dtype=np.int64)
        corrected, raw_ordinals, rule = apply_trigger_correction(3, 2, raw)
        self.assertEqual(raw_ordinals.tolist(), list(range(2, 17)) + list(range(18, 33)) + list(range(35, 50)))
        self.assertEqual(corrected.tolist(), (raw_ordinals - 1).tolist())
        self.assertIn("P03_Ses2", rule)

        corrected, raw_ordinals, rule = apply_trigger_correction(1, 2, raw)
        self.assertEqual(raw_ordinals.tolist(), list(range(1, 46)))
        self.assertEqual(len(corrected), 45)
        self.assertIn("keep_first_45", rule)

        corrected, raw_ordinals, rule = apply_trigger_correction(9, 9, raw[:2])
        self.assertEqual(corrected.tolist(), [0, 1])
        self.assertEqual(raw_ordinals.tolist(), [1, 2])
        self.assertEqual(rule, "none")

    def _write_synthetic_mat(self, directory: Path, *, participant: int = 1, session: int = 1) -> Path:
        trial_count = 24 if session == 1 else 45 if session <= 5 else 40
        onset_step = 3072
        onsets = np.asarray([512 + onset_step * index for index in range(trial_count)], dtype=np.int64)
        sample_count = int(onsets[-1] + TRIAL_SAMPLES + 16)
        data = np.zeros((10, sample_count), dtype=np.float64)
        data[0] = np.arange(sample_count) / 512.0
        for channel in range(6):
            data[channel + 1] = (channel + 1) * 1000 + np.arange(sample_count)
        combinations = target_combinations(session)
        for ordinal, onset in enumerate(onsets):
            data[7, onset] = (ordinal % len(combinations)) + 1
            data[9, onset] = 999  # Must not become the label.
        path = directory / f"P{participant:02d}_Ses{session}.mat"
        savemat(path, {f"data_ses{session}": data})
        return path

    def test_trial_window_row_8_labels_and_source_read_only(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = self._write_synthetic_mat(root)
            before = hashlib.sha256(path.read_bytes()).digest()
            records = convert_file(path)
            after = hashlib.sha256(path.read_bytes()).digest()

            self.assertEqual(before, after)
            self.assertEqual(len(records), 24)
            self.assertEqual(records[0].eeg.shape, (6, TRIAL_SAMPLES))
            self.assertEqual(records[0].onset, 512)
            self.assertEqual(records[0].stimulus_index, 1)
            self.assertEqual(records[0].frequency_tuple, (7,))
            self.assertEqual(records[0].test_method, "T1")
            self.assertEqual(records[0].correction_rule, "none")
            self.assertEqual(records[-1].stimulus_index, 6)

    def test_near20_filters_include_the_requested_frequency_sets(self) -> None:
        with TemporaryDirectory() as temp:
            path = self._write_synthetic_mat(Path(temp))
            all_records = convert_files([path])
            near20 = [record for record in all_records if _matches_frequency_filter(record, True, False)]
            adjacent = [record for record in all_records if _matches_frequency_filter(record, False, True)]
            self.assertEqual(len(near20), 12)  # 17/19/23, each repeated four times.
            self.assertEqual(len(adjacent), 8)  # 19/23, each repeated four times.
            self.assertTrue(all(set(record.frequency_tuple) & {17, 19, 23} for record in near20))
            self.assertTrue(all(set(record.frequency_tuple) & {19, 23} for record in adjacent))

    def test_npz_manifest_and_refuse_overwrite(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._write_synthetic_mat(root)
            records = convert_files([source], near20_only=True)
            output = root / "output"
            npz_path, manifest_path = write_conversion(
                records,
                output_directory=output,
                dataset_root=root,
                participants=(1,),
                sessions=(1,),
                near20_only=True,
                include_adjacent_20=False,
            )
            self.assertTrue(npz_path.is_file())
            self.assertTrue(manifest_path.is_file())
            with np.load(npz_path) as converted:
                self.assertEqual(converted["eeg"].shape, (12, 6, TRIAL_SAMPLES))
                self.assertEqual(converted["status"].item(), "formal")
                self.assertTrue(bool(converted["exploratory_only"].item()))
                self.assertFalse(bool(converted["runtime_enabled"].item()))
                self.assertEqual(converted["frequency_tuple"][0], "17")
                self.assertEqual(converted["stimulation_type"][0], "single")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["exploratory_only"])
            self.assertFalse(manifest["default_deployment_authorized"])
            self.assertFalse(manifest["personal_gaze_accuracy_claim"])
            self.assertEqual(manifest["record_count"], 12)
            self.assertEqual(manifest["trial_metadata"][0]["stimulus_index"], 4)
            with self.assertRaises(FileExistsError):
                write_conversion(
                    records,
                    output_directory=output,
                    dataset_root=root,
                    participants=(1,),
                    sessions=(1,),
                    near20_only=True,
                    include_adjacent_20=False,
                )

    def test_cli_refuses_unbounded_conversion_but_default_is_safe_preview(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_synthetic_mat(root)
            output = root / "output"
            with self.assertRaises(SystemExit):
                main(["--dataset-root", str(root), "--output-directory", str(output)])
            self.assertEqual(
                main(["--dataset-root", str(root)]),
                0,
            )


if __name__ == "__main__":
    unittest.main()
