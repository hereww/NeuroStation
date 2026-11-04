"""Fast temporary-fixture tests of the v2 raw-epoch data contract."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import savemat
from scipy.signal import resample_poly

from eeg_tools.learned_dataset_v2 import DatasetV2Error, build_dataset_v2
from scripts.build_denoise_dataset_v2 import build_parser, main


class LearnedDatasetV2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "pairs.npz"
        rng = np.random.default_rng(4402)
        t = np.arange(512, dtype=float) / 256
        self.pools = {}
        for source_index, kind in enumerate(("eeg", "eog", "emg")):
            # Distinct DC/amplitudes check that source units cannot dominate SNR.
            raw = rng.normal(size=(14, 512)) * (source_index + 1) * 11
            raw += np.sin(2 * np.pi * (source_index + 4) * t)[None, :] * 7
            raw += 50 * (source_index + 1)
            raw = np.concatenate([raw, raw[0:1], raw[3:4]], axis=0)
            self.pools[kind] = raw
        # Identical waveforms in different source pools share a global group.
        self.pools['eog'][1] = self.pools['eeg'][2]
        self.pools['emg'][1] = self.pools['eeg'][2]
        self.save_sources()
        self.source_hashes = self.hash_sources()

    def save_sources(self):
        for kind, raw in self.pools.items():
            np.save(self.source / f"{kind.upper()}_all_epochs.npy", raw, allow_pickle=False)

    def hash_sources(self):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.iterdir() if p.is_file()}

    def build(self, **kwargs):
        options = dict(samples=120, seed=20261005, identity_fraction=0.4)
        options.update(kwargs)
        report = build_dataset_v2(self.source, self.output, **options)
        with np.load(self.output, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        manifest = json.loads(Path(str(self.output) + ".manifest.json").read_text(encoding="utf-8"))
        return report, arrays, manifest

    @staticmethod
    def fingerprint(epoch):
        canonical = np.array(epoch, dtype='<f8', copy=True)
        canonical[canonical == 0] = 0.0
        return hashlib.sha256(canonical.tobytes()).hexdigest()

    def test_single_channel_contract_mask_and_target_allocations(self):
        report, arrays, manifest = self.build()
        for key in ('clean', 'noisy', 'valid_mask', 'artifact_mask'):
            self.assertEqual((120, 1, 500), arrays[key].shape)
        self.assertEqual(np.dtype('float32'), arrays['clean'].dtype)
        self.assertEqual(np.dtype('float32'), arrays['noisy'].dtype)
        self.assertEqual(np.dtype('bool'), arrays['valid_mask'].dtype)
        self.assertEqual(np.dtype('bool'), arrays['artifact_mask'].dtype)
        self.assertTrue(arrays['valid_mask'].all())
        for key in ('split', 'artifact_type'):
            self.assertEqual('U', arrays[key].dtype.kind)
        for key in ('clean_epoch_id', 'eog_epoch_id', 'emg_epoch_id'):
            self.assertEqual('i', arrays[key].dtype.kind)
            self.assertEqual((120,), arrays[key].shape)
        self.assertEqual({'train': 84, 'validation': 18, 'test': 18}, report['split_sample_targets'])
        self.assertEqual(report['split_sample_targets'], manifest['split_sample_targets'])
        identity = arrays['artifact_type'] == 'clean'
        self.assertEqual(48, int(identity.sum()))
        self.assertFalse(arrays['artifact_mask'][identity].any())
        self.assertTrue(arrays['artifact_mask'][~identity].all())
        np.testing.assert_array_equal(arrays['clean'][identity], arrays['noisy'][identity])
        self.assertTrue((arrays['eog_epoch_id'][arrays['artifact_type'] != 'eog'] == -1).all())
        self.assertTrue((arrays['emg_epoch_id'][arrays['artifact_type'] != 'emg'] == -1).all())
        self.assertEqual({'clean', 'eog', 'emg'}, set(arrays['artifact_type']))
        self.assertEqual(self.source_hashes, self.hash_sources())

    def test_raw_ids_and_fingerprints_are_split_disjoint_before_mixing(self):
        _, arrays, manifest = self.build()
        groups = manifest['epoch_split_manifest']['fingerprint_groups']
        by_fingerprint = {}
        assignments = {kind: {} for kind in self.pools}
        for group in groups:
            fingerprint = group['fingerprint']
            self.assertNotIn(fingerprint, by_fingerprint)
            by_fingerprint[fingerprint] = group['split']
            for kind, ids in group['epoch_ids_by_source'].items():
                for epoch_id in ids:
                    self.assertEqual(fingerprint, self.fingerprint(self.pools[kind][epoch_id]))
                    self.assertNotIn(epoch_id, assignments[kind])
                    assignments[kind][epoch_id] = group['split']
        for kind, raw in self.pools.items():
            self.assertEqual(set(range(len(raw))), set(assignments[kind]))
            self.assertEqual(assignments[kind][0], assignments[kind][14])
            self.assertEqual(assignments[kind][3], assignments[kind][15])
        self.assertEqual(assignments['eeg'][2], assignments['eog'][1])
        self.assertEqual(assignments['eeg'][2], assignments['emg'][1])
        for i, split in enumerate(arrays['split']):
            self.assertEqual(split, assignments['eeg'][int(arrays['clean_epoch_id'][i])])
            for kind in ('eog', 'emg'):
                epoch_id = int(arrays[f'{kind}_epoch_id'][i])
                if epoch_id != -1:
                    self.assertEqual(split, assignments[kind][epoch_id])
        self.assertTrue(manifest['epoch_split_manifest']['split_before_resampling_or_mixing'])
        self.assertGreater(manifest['epoch_split_manifest']['duplicate_global_epochs'], 0)

    def test_ac_rms_and_snr_preserve_original_epoch_target_and_artifact_identity(self):
        _, arrays, manifest = self.build(snr_min_db=-7.0, snr_max_db=20.0)
        clean = arrays['clean'].astype(np.float64)
        rms = np.sqrt(np.mean((clean - clean.mean(axis=-1, keepdims=True)) ** 2, axis=-1))
        np.testing.assert_allclose(rms, 1, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(clean.mean(axis=-1), 0, atol=1e-7)
        repeated = {}
        for i, kind in enumerate(arrays['artifact_type']):
            epoch_id = int(arrays['clean_epoch_id'][i])
            expected = self.pools['eeg'][epoch_id]
            expected = resample_poly(expected - expected.mean(), 125, 128)
            expected -= expected.mean()
            expected /= np.sqrt(np.mean(expected ** 2))
            np.testing.assert_allclose(clean[i, 0], expected, atol=2e-7, rtol=1e-6)
            if epoch_id in repeated:
                np.testing.assert_array_equal(arrays['clean'][i], repeated[epoch_id])
            repeated[epoch_id] = arrays['clean'][i]
            noise = arrays['noisy'][i, 0].astype(np.float64) - clean[i, 0]
            if kind == 'clean':
                self.assertIsNone(manifest['sample_snr_db'][i])
                continue
            noise_rms = np.sqrt(np.mean((noise - noise.mean()) ** 2))
            achieved = float(20 * np.log10(rms[i, 0] / noise_rms))
            self.assertGreaterEqual(achieved, -7.00001)
            self.assertLessEqual(achieved, 20.00001)
            self.assertAlmostEqual(manifest['sample_snr_db'][i], achieved, places=5)
            self.assertAlmostEqual(float(noise.mean()), 0, places=6)
            artifact_id = int(arrays[f'{kind}_epoch_id'][i])
            artifact = self.pools[str(kind)][artifact_id]
            artifact = resample_poly(artifact - artifact.mean(), 125, 128)
            artifact -= artifact.mean()
            artifact /= np.sqrt(np.mean(artifact ** 2))
            np.testing.assert_allclose(noise / noise_rms, artifact, rtol=3e-5, atol=2e-6)

    def test_deterministic_arrays_and_assignments_and_changed_seed(self):
        _, first, first_manifest = self.build()
        second_output = self.root / 'second.npz'
        build_dataset_v2(self.source, second_output, samples=120, seed=20261005)
        with np.load(second_output, allow_pickle=False) as second:
            for key, value in first.items():
                np.testing.assert_array_equal(value, second[key])
        second_manifest = json.loads(Path(str(second_output) + '.manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(first_manifest['epoch_split_manifest'], second_manifest['epoch_split_manifest'])
        self.assertEqual(first_manifest['sample_snr_db'], second_manifest['sample_snr_db'])
        self.assertEqual(first_manifest['output']['sha256'], second_manifest['output']['sha256'])
        third_output = self.root / 'third.npz'
        build_dataset_v2(self.source, third_output, samples=120, seed=20261006)
        with np.load(third_output, allow_pickle=False) as third:
            self.assertFalse(np.array_equal(first['noisy'], third['noisy']))
        self.assertEqual(self.source_hashes, self.hash_sources())

    def test_manifest_is_json_safe_and_hashes_match_sources_and_output(self):
        report, _, manifest = self.build()
        self.assertEqual('formal', manifest['status'])
        self.assertEqual('epoch-disjoint semi-synthetic', manifest['sampling_semantics']['generalization_claim'])
        self.assertFalse(manifest['sampling_semantics']['subject_disjoint_claim'])
        self.assertFalse(manifest['sampling_semantics']['subject_identifiers_available'])
        self.assertFalse(manifest['sampling_semantics']['epoch_concatenation'])
        self.assertFalse(manifest['sampling_semantics']['channel_repetition'])
        for kind, metadata in manifest['sources'].items():
            path = Path(metadata['path'])
            self.assertTrue(path.is_absolute())
            self.assertEqual(self.source_hashes[path.name], metadata['sha256'])
            self.assertEqual([16, 512], metadata['shape'])
            self.assertEqual(256, metadata['sample_rate_hz'])
            self.assertEqual('npy', metadata['format'])
        digest = hashlib.sha256(self.output.read_bytes()).hexdigest()
        self.assertEqual(digest, report['npz_sha256'])
        self.assertEqual(digest, manifest['output']['sha256'])
        json.dumps(manifest, allow_nan=False)

    def test_signed_zero_duplicate_epochs_use_same_group(self):
        first = self.pools['eeg'][0].copy()
        first[5] = 0.0
        second = first.copy()
        second[5] = -0.0
        self.pools['eeg'][0] = first
        self.pools['eeg'][14] = second
        self.save_sources()
        _, _, manifest = self.build()
        fingerprint = self.fingerprint(first)
        groups = [group for group in manifest['epoch_split_manifest']['fingerprint_groups']
                  if group['fingerprint'] == fingerprint]
        self.assertEqual(1, len(groups))
        self.assertEqual([0, 14], groups[0]['epoch_ids_by_source']['eeg'])

    def test_mat_fallback_and_npy_priority_with_verified_fs(self):
        for kind, raw in self.pools.items():
            path = self.source / f'{kind.upper()}_all_epochs.mat'
            # Deliberately different content: confirms NPY is used, not MAT signal.
            savemat(path, {f'{kind.upper()}_all_epochs': raw + 17, 'fs': np.asarray([[256]])})
        _, npy_arrays, manifest = self.build()
        self.assertTrue(all(item['format'] == 'npy' for item in manifest['sources'].values()))
        for item in manifest['sources'].values():
            self.assertEqual('companion_mat_fs', item['sample_rate_evidence']['kind'])
            metadata_path = Path(item['sample_rate_evidence']['path'])
            self.assertEqual(hashlib.sha256(metadata_path.read_bytes()).hexdigest(), item['sample_rate_evidence']['sha256'])
        # Restore MAT values to original, remove NPY only inside this temp fixture.
        for kind, raw in self.pools.items():
            savemat(self.source / f'{kind.upper()}_all_epochs.mat', {f'{kind.upper()}_all_epochs': raw, 'fs': [[256]]})
            (self.source / f'{kind.upper()}_all_epochs.npy').unlink()
        self.output = self.root / 'mat_only.npz'
        _, mat_arrays, manifest = self.build()
        for key, value in npy_arrays.items():
            np.testing.assert_array_equal(value, mat_arrays[key])
        self.assertTrue(all(item['format'] == 'mat' for item in manifest['sources'].values()))

    def test_invalid_values_shapes_zero_ac_and_rate_are_refused(self):
        path = self.source / 'EEG_all_epochs.npy'
        original = self.pools['eeg'].copy()
        invalid_sources = []
        for value in (np.nan, np.inf):
            changed = original.copy()
            changed[2, 9] = value
            invalid_sources.append(changed)
        changed = original.copy()
        changed[3] = 19
        invalid_sources.extend([changed, original[:, :500], original.reshape(16, 1, 512)])
        for raw in invalid_sources:
            with self.subTest(shape=raw.shape, finite=np.isfinite(raw).all()):
                np.save(path, raw, allow_pickle=False)
                with self.assertRaises(DatasetV2Error):
                    build_dataset_v2(self.source, self.output, samples=30)
                self.assertFalse(self.output.exists())
        np.save(path, original, allow_pickle=False)
        savemat(self.source / 'EEG_all_epochs.mat', {'EEG_all_epochs': original, 'fs': [[512]]})
        with self.assertRaisesRegex(DatasetV2Error, 'fs must be 256'):
            build_dataset_v2(self.source, self.output, samples=30)

    def test_missing_source_invalid_parameters_and_output_source_guard(self):
        for kwargs in [
            {'samples': 0}, {'samples': 2}, {'samples': 5.0}, {'seed': -1},
            {'identity_fraction': -0.1}, {'identity_fraction': 1.1}, {'identity_fraction': np.nan},
            {'identity_fraction': None}, {'identity_fraction': '0.4'},
            {'snr_min_db': np.inf}, {'snr_min_db': 21, 'snr_max_db': 20},
            {'snr_min_db': -1e308}, {'snr_min_db': 10000, 'snr_max_db': 10001},
            {'snr_max_db': '20'},
            {'fractions': (0.7, 0.2, 0.2)}, {'fractions': (0.7, 0.3, 0)},
            {'fractions': (0.7, np.nan, 0.3)}, {'fractions': (1, 0)},
            {'fractions': None}, {'fractions': 'bad'},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(DatasetV2Error):
                build_dataset_v2(self.source, self.output, **kwargs)
        with self.assertRaisesRegex(DatasetV2Error, 'does not exist'):
            build_dataset_v2(self.root / 'missing', self.output)
        (self.source / 'EOG_all_epochs.npy').unlink()
        with self.assertRaisesRegex(DatasetV2Error, 'source missing'):
            build_dataset_v2(self.source, self.output, samples=30)
        self.save_sources()
        with self.assertRaisesRegex(DatasetV2Error, 'sources are immutable'):
            build_dataset_v2(self.source, self.source / 'unsafe.npz', samples=30, overwrite=True)
        self.assertEqual(self.source_hashes, self.hash_sources())

    def test_duplicate_only_source_cannot_spread_one_fingerprint_across_splits(self):
        repeated = np.repeat(self.pools['eeg'][0:1], 7, axis=0)
        np.save(self.source / 'EEG_all_epochs.npy', repeated, allow_pickle=False)
        with self.assertRaisesRegex(DatasetV2Error, 'fingerprint groups'):
            build_dataset_v2(self.source, self.output, samples=30)
        self.assertFalse(self.output.exists())

    def test_identity_extremes_and_fixed_snr(self):
        _, identity, manifest = self.build(identity_fraction=1.0)
        self.assertTrue((identity['artifact_type'] == 'clean').all())
        self.assertFalse(identity['artifact_mask'].any())
        np.testing.assert_array_equal(identity['clean'], identity['noisy'])
        self.assertTrue(all(value is None for value in manifest['sample_snr_db']))
        self.output = self.root / 'artifacts_only.npz'
        _, artifacts, manifest = self.build(identity_fraction=0, snr_min_db=4.0, snr_max_db=4.0)
        self.assertTrue(artifacts['artifact_mask'].all())
        self.assertEqual({'eog', 'emg'}, set(artifacts['artifact_type']))
        self.assertTrue(all(value == 4 for value in manifest['sample_snr_db']))

    def test_cli_help_defaults_and_small_run(self):
        parser = build_parser()
        defaults = parser.parse_args([])
        self.assertEqual(12000, defaults.samples)
        self.assertEqual(20261005, defaults.seed)
        self.assertEqual(0.4, defaults.identity_fraction)
        self.assertEqual(-7.0, defaults.snr_min_db)
        self.assertEqual(20.0, defaults.snr_max_db)
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            result = main(['--source-dir', str(self.source), '--output', str(self.output), '--samples', '30'])
        self.assertEqual(0, result, stream.getvalue())
        self.assertEqual([30, 1, 500], json.loads(stream.getvalue())['shape'])
        help_text = parser.format_help()
        for option in ('--source-dir', '--output', '--samples', '--seed', '--identity-fraction', '--snr-min-db', '--snr-max-db'):
            self.assertIn(option, help_text)

    def test_existing_outputs_require_explicit_overwrite(self):
        self.build()
        original_npz = self.output.read_bytes()
        with self.assertRaisesRegex(DatasetV2Error, 'already exists'):
            build_dataset_v2(self.source, self.output, samples=30)
        self.assertEqual(original_npz, self.output.read_bytes())
        build_dataset_v2(self.source, self.output, samples=30, overwrite=True)
        with np.load(self.output, allow_pickle=False) as archive:
            self.assertEqual((30, 1, 500), archive['clean'].shape)
        self.assertEqual(self.source_hashes, self.hash_sources())


if __name__ == '__main__':
    unittest.main()
