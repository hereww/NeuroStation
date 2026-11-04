import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts import apply_ssvep_denoiser_v5 as exporter


class SyntheticExporterFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ssvep_apply_v5_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "datasets" / "processed" / "reference").mkdir(parents=True)
        (self.root / "analysis_outputs").mkdir()
        (self.root / "configs").mkdir()
        self.config_path = self.root / "configs" / "config.json"
        self.checkpoint_path = self.root / "checkpoint.pt"
        self.config_path.write_text("{}", encoding="utf-8")
        self.checkpoint_path.write_bytes(b"synthetic-checkpoint")
        self.reference = self.root / "datasets" / "processed" / "reference"
        self.reference_manifest = self.reference / "manifest.json"
        self.reference_manifest.write_text(json.dumps({"synthetic": True}), encoding="utf-8")

        signal = np.arange(2 * 12 * 8, dtype=np.float64).reshape(2, 12, 8)
        labels = {
            "trial_index": np.arange(2, dtype=np.int64),
            "target_id": np.array([0, 1], dtype=np.int64),
            "block_id": np.array([0, 1], dtype=np.int64),
            "electrode_id": np.array([-1, -1], dtype=np.int64),
            "frequency_hz": np.array([8.0, 8.6], dtype=np.float64),
        }
        self.arrays = {"reference": signal, **labels}
        self.side = {
            "schema_version": 1,
            "reference_version": "ssvep_reference_v5",
            "status": "formal",
            "dataset": "Nakanishi2015",
            "subject_id": 1,
            "split": "train",
            "group_key": "Nakanishi2015:subject_001",
            "reference_shape": list(signal.shape),
            "reference_dtype": "float64",
            "native_sample_rate_hz": 256,
            "onset_sample_zero_based": 0,
            "frequency_order_hz": [8.0, 8.6],
            "phase_pi_multiples": None,
            "native_units": "uV",
            "channel_order": None,
            "output_axes": ["trial", "time", "channel"],
            "trial_order": "synthetic",
            "electrode_order": None,
            "materialized_trials": 2,
            "reference_sha256": "0" * 64,
        }
        self.source_npz = self.reference / "Nakanishi2015" / "subject_001.npz"
        self.source_npz.parent.mkdir()
        with self.source_npz.open("wb") as handle:
            np.savez_compressed(handle, **self.arrays)
        self.source_sidecar = Path(str(self.source_npz) + ".manifest.json")
        self.source_sidecar.write_text(json.dumps(self.side), encoding="utf-8")

        self.catalog = {
            "manifest_hash": exporter._sha(self.reference_manifest),
            "directory": self.reference,
            "candidate": False,
            "tracked": {},
            "subjects": {
                ("Nakanishi2015", 1): {
                    "side": self.side,
                    "shape": tuple(signal.shape),
                    "path": self.source_npz,
                    "side_hash": "1" * 64,
                    "npz_hash": "2" * 64,
                }
            },
        }
        self.patches = mock.patch.multiple(
            exporter,
            ROOT=self.root,
        )
        self.patches.start()
        self.addCleanup(self.patches.stop)
        self.core_constants = mock.patch.multiple(
            exporter.core,
            REFERENCE_RELATIVE=Path("datasets/processed/reference"),
            CONFIG_RELATIVE=Path("configs/config.json"),
            REFERENCE_MANIFEST_SHA256=self.catalog["manifest_hash"],
            MODEL_VERSION="synthetic-model-v5",
        )
        self.core_constants.start()
        self.addCleanup(self.core_constants.stop)
        self.task_constants = mock.patch.multiple(
            exporter.task_eval,
            REFERENCE_VERSION="ssvep_reference_v5",
            WINDOW_LENGTHS={"Nakanishi2015": 8, "Wearable": 8},
        )
        self.task_constants.start()
        self.addCleanup(self.task_constants.stop)

    def run_export(self, output="candidate_v5", report="report.json", extra=None):
        args = [
            "--reference-dir", str(self.reference),
            "--config", str(self.config_path),
            "--checkpoint", str(self.checkpoint_path),
            "--output-dir", str(self.root / "datasets" / "processed" / output),
            "--output-report", str(self.root / "analysis_outputs" / report),
            "--device", "cpu",
        ]
        if extra:
            args.extend(extra)
        return exporter.main(args)

    def patch_runtime(self, *, catalog=None, read=None, verify=None, denoise=None):
        patches = [
            mock.patch.object(exporter.task_eval, "_inventory", return_value={self.source_npz}),
            mock.patch.object(exporter.task_eval, "_catalog", return_value=catalog or self.catalog),
            mock.patch.object(exporter.task_eval, "_read_subject", side_effect=read or (lambda *_: self.arrays)),
            mock.patch.object(exporter.task_eval, "_verify_unchanged", side_effect=verify or (lambda *_: None)),
            mock.patch.object(exporter.core, "load_config", return_value={"candidate_signal_key": "processed_v5"}),
            mock.patch.object(exporter.core, "load_checkpoint", return_value={"loaded_models": {"Nakanishi2015": object()}}),
            mock.patch.object(exporter.core, "denoise_array", side_effect=denoise or (lambda _model, signal, *_: signal + 1.0)),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def test_rejects_untrusted_reference_paths_without_loading(self):
        with self.assertRaises(exporter.ApplySSVEPDenoiserV5Error):
            exporter._trusted_reference(self.root.parent / "outside")
        with self.assertRaises(exporter.ApplySSVEPDenoiserV5Error):
            exporter._trusted_reference(Path("D:/SSVEP/reference"))
        with self.assertRaises(exporter.ApplySSVEPDenoiserV5Error):
            exporter._trusted_reference(Path("datasets/processed/../processed/reference"))

        other = self.root / "datasets" / "processed" / "other"
        other.mkdir()
        with mock.patch.object(exporter.core, "load_checkpoint") as load_checkpoint:
            with self.assertRaises(exporter.ApplySSVEPDenoiserV5Error):
                self.run_export(extra=["--reference-dir", str(other)])
        load_checkpoint.assert_not_called()

    def test_symlink_reference_is_rejected(self):
        link = self.root / "datasets" / "processed" / "linked-reference"
        try:
            link.symlink_to(self.reference, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        with self.assertRaises(exporter.ApplySSVEPDenoiserV5Error):
            exporter._trusted_reference(link)

    def test_metadata_precheck_precedes_waveform_and_model_loading(self):
        failure = RuntimeError("metadata rejected")
        with mock.patch.object(exporter.task_eval, "_inventory", return_value=set()), \
             mock.patch.object(exporter.task_eval, "_catalog", side_effect=failure), \
             mock.patch.object(exporter.core, "load_config", return_value={"candidate_signal_key": "processed_v5"}), \
             mock.patch.object(exporter.core, "load_checkpoint") as load_checkpoint, \
             mock.patch.object(exporter.task_eval, "_read_subject") as read_subject:
            with self.assertRaisesRegex(exporter.ApplySSVEPDenoiserV5Error, "metadata rejected"):
                self.run_export()
        load_checkpoint.assert_not_called()
        read_subject.assert_not_called()
        self.assertFalse((self.root / "datasets" / "processed" / "candidate_v5").exists())

    def test_checkpoint_loaded_models_binding_is_used(self):
        model = object()
        seen = []

        def denoise(received_model, signal, *_):
            seen.append(received_model)
            return signal

        self.patch_runtime(denoise=denoise)
        with mock.patch.object(exporter.core, "load_checkpoint", return_value={"loaded_models": {"Nakanishi2015": model}}):
            self.assertEqual(self.run_export(), 0)
        self.assertEqual(seen, [model])

    def test_npz_has_real_content_and_labels_align(self):
        self.patch_runtime()
        self.assertEqual(self.run_export(), 0)
        output = self.root / "datasets" / "processed" / "candidate_v5"
        npz = output / "Nakanishi2015" / "subject_001.npz"
        sidecar = Path(str(npz) + ".manifest.json")
        with np.load(npz, allow_pickle=False) as loaded:
            self.assertIn("processed_v5", loaded.files)
            np.testing.assert_array_equal(loaded["processed_v5"], self.arrays["reference"] + 1.0)
            for key in ("trial_index", "target_id", "block_id", "electrode_id", "frequency_hz"):
                np.testing.assert_array_equal(loaded[key], self.arrays[key])
            embedded = json.loads(str(loaded["manifest_json"].item()))
        side = json.loads(sidecar.read_text(encoding="utf-8"))
        self.assertEqual(embedded, {k: v for k, v in side.items() if k not in {"npz_sha256", "npz_relative_path"}})
        self.assertEqual(side["checkpoint_sha256"], exporter._sha(self.checkpoint_path))
        self.assertEqual(side["reference_trial_count"], 2)
        self.assertEqual(side["reference_window_samples"], 8)
        self.assertEqual(side["reference_sample_rate_hz"], 256)
        self.assertEqual(side["reference_native_units"], "uV")
        self.assertNotIn("clean", json.dumps(side).lower())

    def test_hash_tamper_is_detected_and_failure_is_preserved(self):
        def tampering_denoise(_model, signal, *_):
            self.checkpoint_path.write_bytes(b"tampered")
            return signal

        self.patch_runtime(denoise=tampering_denoise)
        with self.assertRaises(exporter.ApplySSVEPDenoiserV5Error):
            self.run_export()
        output = self.root / "datasets" / "processed" / "candidate_v5"
        self.assertTrue(output.exists())
        self.assertTrue((output / "failure_record.json").exists())
        self.assertFalse((self.root / "analysis_outputs" / "report.json").exists())

    def test_publication_is_exclusive(self):
        self.patch_runtime()
        self.assertEqual(self.run_export(), 0)
        with self.assertRaises(exporter.ApplySSVEPDenoiserV5Error):
            self.run_export()


if __name__ == "__main__":
    unittest.main()
