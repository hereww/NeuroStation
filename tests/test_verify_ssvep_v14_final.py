"""Contract tests for the v14 final diagnostic verification."""
from __future__ import annotations
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from scripts import verify_ssvep_v14_final as verifier


class V14FinalVerificationTests(unittest.TestCase):
    def test_existing_diagnostic_artifacts_verify(self):
        evaluation = ROOT / 'analysis_outputs' / 'v14_evaluation_20261014_diagnostic'
        figures = ROOT / 'analysis_outputs' / 'v14_diagnostic_figures_20261014'
        result = verifier.verify(evaluation, figures)
        self.assertEqual(result['conclusion'], 'diagnostic_artifacts_complete_but_not_deployable')
        self.assertTrue(result['checks']['source_files_unchanged'])
        self.assertTrue(result['checks']['six_figures_present_and_bound'])
        self.assertFalse(result['runtime_enabled'])
        self.assertFalse(result['default_deployment_authorized'])

    def test_output_path_is_new_and_under_analysis_outputs(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'analysis_outputs') as directory:
            with self.assertRaisesRegex(ValueError, 'already exists'):
                verifier.create_output(Path(directory))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'under analysis_outputs'):
                verifier.create_output(Path(directory) / 'out')

    def test_help_does_not_import_numpy(self):
        code = """
import importlib.abc, runpy, sys
assert 'numpy' not in sys.modules
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'numpy':
            raise AssertionError('numpy imported during help')
sys.meta_path.insert(0, Block())
sys.argv = ['verify_ssvep_v14_final.py', '--help']
try:
    runpy.run_module('scripts.verify_ssvep_v14_final', run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
"""
        result = subprocess.run([sys.executable, '-S', '-B', '-c', code], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
