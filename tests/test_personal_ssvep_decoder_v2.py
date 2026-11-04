"""Contract tests for the posterior-channel v2 personal decoder."""
from __future__ import annotations
import importlib.abc
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from eeg_tools import personal_ssvep_decoder_v2 as decoder


class PersonalDecoderV2Tests(unittest.TestCase):
    def test_formal_config_and_posterior_subset(self):
        config = decoder.get_config()
        self.assertEqual(config['status'], 'formal')
        self.assertEqual(config['selected_channel_indices'], [4, 5, 6, 7])
        self.assertEqual(config['selected_electrode_positions_from_preset'], ['P7', 'P8', 'O1', 'O2'])
        self.assertFalse(config['source_domain_validated'])
        self.assertFalse(config['gaze_accuracy_claim'])

    def test_forward_does_not_accept_recorded_labels(self):
        signal = np.random.default_rng(20261014).standard_normal((8, 1250)).astype(np.float64)
        result = decoder.decode_window(signal, 250.0, [('a', 10.0), ('b', 12.0), ('c', 15.0), ('d', 20.0)])
        self.assertTrue(result['available'])
        self.assertEqual(result['selected_channel_indices'], [4, 5, 6, 7])
        self.assertFalse(result['gaze_accuracy_claim'])

    def test_bad_shape_and_sample_rate_are_rejected(self):
        with self.assertRaises(ValueError):
            decoder.decode_window(np.zeros((4, 1000), dtype=np.float64), 250.0, [('a', 10.0)])
        with self.assertRaises(ValueError):
            decoder.decode_window(np.zeros((8, 1000), dtype=np.float64), 249.0, [('a', 10.0)])

    def test_help_import_is_lazy(self):
        code = """
import importlib.abc, runpy, sys
blocked = {'numpy', 'scipy'}
assert not blocked.intersection(sys.modules)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('scientific import during evaluator help')
sys.meta_path.insert(0, Block())
sys.argv = ['evaluate_personal_ssvep_decoder_v2.py', '--help']
try:
    runpy.run_module('scripts.evaluate_personal_ssvep_decoder_v2', run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
"""
        result = subprocess.run([sys.executable, '-S', '-B', '-c', code], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
