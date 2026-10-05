from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from eeg_tools.learned_dataset import build_eegdenoisenet_arrays, build_training_arrays, synthesize_artifacts
from eeg_tools.learned_denoise import (
    LearnedDenoiseError,
    load_config,
    robust_denormalize,
    robust_normalize,
    torch_available,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "learned_denoiser_v1.json"


class LearnedDenoiseTests(unittest.TestCase):
    def test_formal_config_is_valid(self) -> None:
        config = load_config(CONFIG)
        self.assertEqual("formal", config["status"])
        self.assertEqual("residual_tcn", config["model_type"])
        self.assertEqual(8, config["channel_count"])
        self.assertEqual("shadow", config["inference"]["mode"])

    def test_config_rejects_non_formal_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            value = json.loads(CONFIG.read_text(encoding="utf-8"))
            value["status"] = "draft"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(LearnedDenoiseError, "status must be formal"):
                load_config(path)

    def test_normalization_round_trip_with_invalid_samples(self) -> None:
        values = np.asarray([[1.0, 2.0, np.nan, 4.0], [3.0, 3.5, 4.0, 4.5]], dtype=np.float32)
        valid = np.isfinite(values)
        normalized, center, scale = robust_normalize(values, valid)
        restored = robust_denormalize(normalized, center, scale)
        np.testing.assert_allclose(restored[valid], values[valid], rtol=1e-5, atol=1e-5)
        self.assertTrue(np.all(np.isfinite(normalized)))

    def test_artifact_synthesis_is_reproducible_and_marks_samples(self) -> None:
        clean = np.zeros((8, 1000), dtype=np.float32)
        first = synthesize_artifacts(clean, 250.0, np.random.default_rng(7))
        second = synthesize_artifacts(clean, 250.0, np.random.default_rng(7))
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(clean.shape, first[0].shape)
        self.assertEqual(clean.shape, first[1].shape)
        self.assertGreater(int(np.count_nonzero(first[1])), 0)

    def test_build_eegdenoisenet_arrays_if_downloaded(self) -> None:
        dataset_dir = ROOT / "datasets" / "external" / "eegdenoisenet"
        if not (dataset_dir / "EEG_all_epochs.npy").exists():
            self.skipTest("EEGdenoiseNet was not downloaded")
        arrays = build_eegdenoisenet_arrays(dataset_dir, max_windows=2, seed=1)
        self.assertEqual((2, 8, 1000), arrays["clean"].shape)
        self.assertEqual(arrays["clean"].shape, arrays["noisy"].shape)
        self.assertEqual(arrays["clean"].shape, arrays["artifact_mask"].shape)
        self.assertTrue(np.all(np.isfinite(arrays["noisy"])))

    def test_build_training_arrays_from_small_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "session_test"
            session.mkdir()
            rows = ["sample_index\t" + "\t".join(f"eeg_ch{i}" for i in range(1, 9))]
            for index in range(1600):
                rows.append(str(index) + "\t" + "\t".join(str(np.sin(index / 10.0) + channel) for channel in range(8)))
            (session / "raw_brainflow.tsv").write_text("\n".join(rows), encoding="utf-8")
            (session / "session.json").write_text(json.dumps({"sampling_rate_hz": 250}), encoding="utf-8")
            arrays = build_training_arrays(root, max_windows=2, seed=1)
            self.assertEqual((2, 8, 1000), arrays["clean"].shape)
            self.assertEqual(arrays["clean"].shape, arrays["noisy"].shape)
            self.assertEqual(arrays["clean"].shape, arrays["artifact_mask"].shape)


if __name__ == "__main__":
    unittest.main()

