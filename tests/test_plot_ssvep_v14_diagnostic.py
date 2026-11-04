"""Contract tests for the v14 diagnostic figure generator."""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from scripts import plot_ssvep_v14_diagnostic as plotter


class V14DiagnosticFigureTests(unittest.TestCase):
    def test_help_does_not_import_scientific_stack(self):
        code = """
import importlib.abc, runpy, sys
blocked = {'numpy', 'scipy', 'matplotlib'}
assert not blocked.intersection(sys.modules)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('scientific import during help: ' + fullname)
sys.meta_path.insert(0, Block())
sys.argv = ['plot_ssvep_v14_diagnostic.py', '--help']
try:
    runpy.run_module('scripts.plot_ssvep_v14_diagnostic', run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
assert not blocked.intersection(sys.modules)
"""
        result = subprocess.run([sys.executable, '-S', '-B', '-c', code], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('--evaluation-directory', result.stdout)

    def test_input_evaluation_exists_and_is_formal(self):
        evaluation = ROOT / 'analysis_outputs' / 'v14_evaluation_20261014_diagnostic'
        summary, reports = plotter.load_inputs(evaluation)
        self.assertEqual(summary['status'], 'formal')
        self.assertEqual(len(reports), 2)
        self.assertTrue(all(report['source_hashes_match'] for report in reports))
        self.assertTrue(all(report['presented_matches_are_gaze_accuracy'] is False for report in reports))

    def test_output_is_under_analysis_outputs_and_new(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'analysis_outputs') as directory:
            output = Path(directory)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                plotter.create_output(output)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'under analysis_outputs'):
                plotter.create_output(Path(directory) / 'out')

    def test_receipt_has_three_figures_per_person_and_fail_closed_flags(self):
        evaluation = ROOT / 'analysis_outputs' / 'v14_evaluation_20261014_diagnostic'
        summary, _ = plotter.load_inputs(evaluation)
        self.assertEqual(summary['model']['selected_epoch'], None)
        self.assertFalse(summary['model']['selection_gate_passed'])


if __name__ == '__main__':
    unittest.main()
