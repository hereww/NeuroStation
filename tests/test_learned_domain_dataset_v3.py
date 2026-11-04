"""Synthetic/mocked source-contract tests; never open real Dong subjects."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat
from eeg_tools import learned_domain_dataset_v3 as domain
from eeg_tools.learned_denoise_v2 import normalize_window

ROOT = Path(__file__).resolve().parents[1]


class DomainDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        time = np.arange(1250, dtype=np.float64)[None, :, None, None]
        phase = (np.arange(8)[:, None, None, None] * .03
                 + np.arange(40)[None, None, :, None] * .02
                 + np.arange(4)[None, None, None, :] * .04)
        cls.raw = 1e6 + 20 * np.sin(time * .09 + phase)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.npz = self.root / domain.SUPPORT_NPZ
        self.npz.parent.mkdir(parents=True)
        self.signal = np.tile(np.sin(np.arange(500) * .07).astype(np.float32), (9, 1, 1))
        self.signal[7:] = np.nan  # poison test values: must not be inspected
        self.mask = np.ones((9, 1, 500), dtype=bool); self.mask[7:] = False
        self.split = np.array(['train'] * 4 + ['validation'] * 3 + ['test'] * 2)
        self.contract = json.loads((ROOT / domain.PREREGISTRATION).read_text(encoding='utf-8'))
        self.meta = json.loads((ROOT / domain.DONG_METADATA).read_text(encoding='utf-8'))
        self.metadata_path = self.root / domain.DONG_METADATA
        self.metadata_path.parent.mkdir(parents=True)
        self.metadata_path.write_text(json.dumps(self.meta), encoding='utf-8')
        for subject in (*domain.TRAIN_SUBJECTS, *domain.VALIDATION_SUBJECTS, *domain.HOLDOUT_SUBJECTS):
            path = self.root / domain.DONG_DIR / subject
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('synthetic mocked MAT identity ' + subject).encode())
        self.write_support()

    def write_support(self):
        np.savez_compressed(self.npz, noisy=self.signal, split=self.split, valid_mask=self.mask,
                            clean=np.array([object()], dtype=object))  # never load clean
        manifest = {'output': {'sha256': domain.sha256(self.npz)},
                    'preprocessing': {'target_rate_hz': 250, 'target_samples': 500},
                    'sources': {name: {'path': 'synthetic_' + name, 'sha256': 'a' * 64}
                                for name in ('eeg', 'eog', 'emg')}}
        manifest_path = Path(str(self.npz) + '.manifest.json')
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        self.contract['protected_sha256'][domain.SUPPORT_NPZ] = domain.sha256(self.npz)
        self.contract['protected_sha256'][domain.SUPPORT_NPZ + '.manifest.json'] = domain.sha256(manifest_path)
        contract_path = self.root / domain.PREREGISTRATION
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(self.contract), encoding='utf-8')

    def load(self, split='train', raw=None):
        with patch.object(domain, 'loadmat', return_value={'eegdata': self.raw if raw is None else raw}):
            return domain.load_domain_split(self.root, split)

    def test_train_validation_all_subject_trials_channels_windows_no_test_use(self):
        for split, positive, subjects in (('train', 4, domain.TRAIN_SUBJECTS), ('validation', 3, domain.VALIDATION_SUBJECTS)):
            with self.subTest(split=split), patch.object(domain, 'loadmat', return_value={'eegdata': self.raw}) as loader, \
                 patch.object(domain, 'sha256', wraps=domain.sha256) as hashing:
                result = domain.load_domain_split(self.root, split)
            self.assertEqual((positive + 5120, 1, 500), result['normalized'].shape)
            self.assertEqual(np.dtype('float32'), result['normalized'].dtype)
            self.assertEqual(np.dtype('bool'), result['supported'].dtype)
            self.assertEqual(positive, int(result['supported'].sum()))
            self.assertTrue(np.isfinite(result['normalized']).all())
            self.assertEqual(list(subjects), [Path(call.args[0]).name for call in loader.call_args_list])
            accessed = [Path(call.args[0]).name for call in hashing.call_args_list]
            self.assertFalse(set(domain.HOLDOUT_SUBJECTS) & set(accessed))
            self.assertIs(False, result['test_values_accessed'])
            self.assertIn('physically decompress test bytes', result['npz_io_semantics'])
            negative = result['provenance'][positive:]
            for subject in subjects:
                rows = [row for row in negative if row['subject'] == subject]
                self.assertEqual(2560, len(rows))
                identities = {(row['target_axis_index'], row['block_index'], row['channel_index'], row['window_index']) for row in rows}
                self.assertEqual(2560, len(identities))
                self.assertEqual({125, 625}, {row['start_sample'] for row in rows})
                self.assertTrue(all(row['source_sha256'] == domain.sha256(self.root / domain.DONG_DIR / subject) for row in rows))
            self.assertEqual(set(np.flatnonzero(self.split == split).tolist()),
                             {row['source_window_index'] for row in result['provenance'][:positive]})

    def test_normalized_features_match_v2_float64_each_window_not_path_dc_or_amplitude(self):
        result = self.load()
        start = 4 + ((17 * 4 + 2) * 8 + 6) * 2 + 1
        expected, _, _ = normalize_window(self.raw[6, 625:1125, 17, 2])
        np.testing.assert_array_equal(expected, result['normalized'][start, 0])
        shifted = self.load(raw=self.raw * 7 + 1e6)
        np.testing.assert_allclose(result['normalized'][4:], shifted['normalized'][4:], rtol=0, atol=2e-6)
        self.assertTrue(all(not {'dc', 'mean', 'scale', 'amplitude_units'} & row.keys() for row in result['provenance']))
        self.assertEqual((1, 500), result['normalized'].shape[1:])

    def test_test_or_holdout_split_rejected_before_source_io(self):
        for split in ('test', 'holdout', 'unsupported_holdout', 'valid', None):
            with self.subTest(split=split), patch.object(domain, '_contract') as contract, \
                 patch.object(domain, 'loadmat') as loader, self.assertRaises(domain.DomainDatasetV3Error):
                domain.load_domain_split(self.root, split)
            contract.assert_not_called(); loader.assert_not_called()

    def test_separate_holdout_loader_synthetic_only_no_support_npz(self):
        with patch.object(domain, 'loadmat', return_value={'eegdata': self.raw}) as loader, \
             patch.object(domain.np, 'load', side_effect=AssertionError('holdout must not load supported NPZ')):
            result = domain.load_unsupported_holdout(self.root)
        self.assertEqual(list(domain.HOLDOUT_SUBJECTS), [Path(call.args[0]).name for call in loader.call_args_list])
        self.assertEqual((5120, 1, 500), result['normalized'].shape)
        self.assertFalse(result['supported'].any())
        self.assertEqual({'S4.mat', 'S5.mat'}, {p['subject'] for p in result['provenance']})

    def test_metadata_wrong_rate_interval_axes_rejected_without_mat_read(self):
        for field, value in (('sample_rate_hz', 1000), ('expected_shape', [8, 1000, 40, 4]),
                             ('axis_order', ['time', 'channel', 'target', 'block']),
                             ('stimulus_onset_status', 'unknown'),
                             ('analysis_window_samples', {'start': 126, 'stop_exclusive': 1126, 'count': 1000, 'indexing': 'zero_based_half_open'})):
            altered = deepcopy(self.meta); altered[field] = value
            self.metadata_path.write_text(json.dumps(altered), encoding='utf-8')
            with self.subTest(field=field), patch.object(domain, 'loadmat') as loader, self.assertRaises(domain.DomainDatasetV3Error):
                domain.load_domain_split(self.root, 'train')
            loader.assert_not_called()

    def test_nonfinite_constant_wrong_shape_or_complex_mat_never_excludes_windows(self):
        invalid = self.raw.copy(); invalid[2, 200, 3, 1] = np.nan
        constant = self.raw.copy(); constant[1, 125:625, 4, 2] = 1e6
        for raw in (invalid, constant, self.raw[:, :-1], self.raw.astype(complex)):
            with self.subTest(shape=raw.shape, dtype=str(raw.dtype)), self.assertRaises(domain.DomainDatasetV3Error):
                self.load(raw=raw)

    def test_supported_nonfinite_constant_mask_or_dtype_rejected_not_filtered(self):
        original = self.signal.copy()
        cases = ('nonfinite', 'constant', 'mask', 'dtype')
        for name in cases:
            self.signal = original.copy(); self.mask[0] = True
            if name == 'nonfinite': self.signal[0, 0, 0] = np.inf
            elif name == 'constant': self.signal[0] = 5
            elif name == 'mask': self.mask[0, 0, 0] = False
            else: self.signal = self.signal.astype(np.float64)
            self.write_support()
            with self.subTest(name=name), self.assertRaises(domain.DomainDatasetV3Error):
                self.load()

    def test_source_identity_mismatch_and_preregistered_subject_changes_refused(self):
        with self.npz.open('ab') as handle: handle.write(b'changed source')
        with self.assertRaisesRegex(domain.DomainDatasetV3Error, 'SHA256'):
            self.load()
        self.write_support()
        self.contract['data']['unsupported_train_subjects'] = ['S4.mat', 'S7.mat']
        (self.root / domain.PREREGISTRATION).write_text(json.dumps(self.contract), encoding='utf-8')
        with patch.object(domain, 'loadmat') as loader, self.assertRaises(domain.DomainDatasetV3Error):
            self.load()
        loader.assert_not_called()

    def test_sources_not_mutated(self):
        paths = [self.npz, self.metadata_path, Path(str(self.npz) + '.manifest.json'),
                 self.root / domain.DONG_DIR / 'S6.mat', self.root / domain.DONG_DIR / 'S7.mat']
        before = {str(p): domain.sha256(p) for p in paths}
        self.load()
        self.assertEqual(before, {str(p): domain.sha256(p) for p in paths})

    def test_real_scipy_round_trip_of_synthetic_mat(self):
        for subject in domain.TRAIN_SUBJECTS:
            savemat(self.root / domain.DONG_DIR / subject, {'eegdata': self.raw}, do_compression=True)
        result = domain.load_domain_split(self.root, 'train')
        self.assertEqual(5124, len(result['normalized']))
        np.testing.assert_array_equal(normalize_window(self.raw[0, 125:625, 0, 0])[0], result['normalized'][4, 0])


if __name__ == '__main__':
    unittest.main()
