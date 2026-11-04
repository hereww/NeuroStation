"""Contract tests for the v3 personal decoder evaluator."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from scripts import evaluate_personal_ssvep_decoder_v3 as evaluator


class EvaluatorV3Tests(unittest.TestCase):
    def test_existing_report_is_formal_and_fail_closed(self):
        report = evaluator.read_json(ROOT / "analysis_outputs" / "personal_ssvep_decoder_v3_20261014" / "summary.json")
        self.assertEqual(report["status"], "formal")
        self.assertTrue(report["source_files_unchanged"])
        self.assertTrue(report["v3_is_evidence_gated_diagnostic"])
        self.assertTrue(report["accepted_predictions_allow_unknown"])
        self.assertFalse(report["runtime_enabled"])
        self.assertEqual(len(report["personal"]), 2)
        self.assertIn("unknown_trial_count", report["personal"][0]["tracks"]["raw"])

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
sys.argv = ['evaluate_personal_ssvep_decoder_v3.py', '--help']
try:
    runpy.run_module('scripts.evaluate_personal_ssvep_decoder_v3', run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
"""
        result = subprocess.run([sys.executable, "-S", "-B", "-c", code], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
