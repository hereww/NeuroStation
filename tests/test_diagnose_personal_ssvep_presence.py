"""Contract tests for personal SSVEP evidence characterization."""
from __future__ import annotations
import subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from scripts import diagnose_personal_ssvep_presence as d
class EvidenceTests(unittest.TestCase):
 def test_help_is_lazy(self):
  code="""import importlib.abc,runpy,sys
blocked={'numpy','scipy'}
assert not blocked.intersection(sys.modules)
class B(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in blocked: raise AssertionError('scientific import during help')
sys.meta_path.insert(0,B());sys.argv=['diagnose_personal_ssvep_presence.py','--help']
try:runpy.run_module('scripts.diagnose_personal_ssvep_presence',run_name='__main__')
except SystemExit as e:assert e.code==0
"""
  r=subprocess.run([sys.executable,'-S','-B','-c',code],cwd=ROOT,text=True,capture_output=True);self.assertEqual(r.returncode,0,r.stdout+r.stderr)
 def test_sources_fixed(self):
  self.assertEqual(len(d.FALSE_FLAGS),11)
  self.assertFalse(d.FALSE_FLAGS['physiological_clean_truth'])
if __name__=='__main__':unittest.main()
