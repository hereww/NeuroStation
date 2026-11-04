"""Contract tests for the v2 personal decoder evaluator."""
from __future__ import annotations
import subprocess
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from scripts import evaluate_personal_ssvep_decoder_v2 as ev
class EvaluatorV2Tests(unittest.TestCase):
 def test_existing_report_is_formal_and_source_unchanged(self):
  d=ev.read_json(ROOT/'analysis_outputs'/'personal_ssvep_decoder_v2_20261014'/'summary.json')
  self.assertEqual(d['status'],'formal');self.assertTrue(d['source_files_unchanged']);self.assertFalse(d['runtime_enabled']);self.assertTrue(d['v2_is_fixed_posterior_diagnostic'])
  self.assertEqual(len(d['personal']),2)
  self.assertEqual(d['personal'][1]['v2']['raw']['presented_match_count'],9)
 def test_help_is_lazy(self):
  code="""import importlib.abc,runpy,sys
blocked={'numpy','scipy'}
assert not blocked.intersection(sys.modules)
class B(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in blocked: raise AssertionError('scientific import during help')
sys.meta_path.insert(0,B());sys.argv=['evaluate_personal_ssvep_decoder_v2.py','--help']
try:runpy.run_module('scripts.evaluate_personal_ssvep_decoder_v2',run_name='__main__')
except SystemExit as e:assert e.code==0
"""
  r=subprocess.run([sys.executable,'-S','-B','-c',code],cwd=ROOT,text=True,capture_output=True);self.assertEqual(r.returncode,0,r.stdout+r.stderr)
if __name__=='__main__':unittest.main()
