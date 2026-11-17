"""Contract tests for the fixed v14 event-window sensitivity diagnostic."""
from __future__ import annotations
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from scripts import diagnose_ssvep_v14_event_window_sensitivity as diagnostic


from tests.historical_evidence import historical_evidence

class V14EventWindowSensitivityTests(unittest.TestCase):
    def test_grid_is_fixed_and_not_selection(self):
        self.assertEqual(diagnostic.TRIMS, (0, 50, 100, 125, 150, 200))
        self.assertFalse(diagnostic.FALSE_FLAGS['presented_matches_are_gaze_accuracy'])

    @historical_evidence
    def test_existing_evaluation_is_failed_selection(self):
        summary = diagnostic.read_json(ROOT / 'analysis_outputs' / 'v14_evaluation_20261014_diagnostic' / 'summary.json')
        self.assertIsNone(summary['model']['selected_epoch'])
        self.assertFalse(summary['model']['selection_gate_passed'])

    def test_help_is_lazy(self):
        code = """
import importlib.abc, runpy, sys
blocked = {'numpy', 'scipy'}
assert not blocked.intersection(sys.modules)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('scientific import during help')
sys.meta_path.insert(0, Block())
sys.argv = ['diagnose_ssvep_v14_event_window_sensitivity.py', '--help']
try:
    runpy.run_module('scripts.diagnose_ssvep_v14_event_window_sensitivity', run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
"""
        result = subprocess.run([sys.executable, '-S', '-B', '-c', code], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
