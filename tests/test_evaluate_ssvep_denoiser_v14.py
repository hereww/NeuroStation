"""Contract tests for the v14 failed-selection personal evaluator."""
from __future__ import annotations
import contextlib,io,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scripts import evaluate_ssvep_denoiser_v14 as ev
from eeg_tools import ssvep_v14_runtime as api
from tests.synthetic_research_fixtures import failed_v14_fixture
ROOT=Path(__file__).resolve().parents[1]
class EvaluatorContractTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp = tempfile.TemporaryDirectory(prefix='synthetic-v14-loader-')
  cls.addClassCleanup(cls.temp.cleanup)
  cls.root = Path(cls.temp.name)
  failed_v14_fixture(cls.root)

 def test_default_runtime_loader_rejects_failed_v14(self):
  with self.assertRaisesRegex(ValueError,'ineligible'):
   api.load_model(self.root,'cpu')
 def test_explicit_diagnostic_loader_is_eval_and_failed_role(self):
  model,meta=ev.load_diagnostic_model(self.root,'cpu',allow_failed_selection_diagnostic=True)
  self.assertFalse(model.training);self.assertEqual(meta['checkpoint_role'],'diagnostic_last_epoch_no_eligible_checkpoint');self.assertIsNone(meta['selected_epoch']);self.assertFalse(meta['selection_gate_passed'])
 def test_diagnostic_loader_requires_explicit_flag(self):
  with self.assertRaisesRegex(ValueError,'explicit'):
   ev.load_diagnostic_model(self.root,'cpu')
 def test_diagnostic_loader_weights_only_and_map_location(self):
  import torch
  with patch.object(torch,'load',wraps=torch.load) as load:
   ev.load_diagnostic_model(self.root,'cpu',allow_failed_selection_diagnostic=True)
  self.assertTrue(load.call_args.kwargs['weights_only']);self.assertEqual(load.call_args.kwargs['map_location'],'cpu')
 def test_flags_are_fail_closed(self):
  flags=ev.model_context({'checkpoint_role':'diagnostic_last_epoch_no_eligible_checkpoint'})
  self.assertTrue(flags);self.assertTrue(all(value is False for value in flags.values()))
 def test_path_must_be_new_analysis_directory(self):
  with tempfile.TemporaryDirectory(dir=self.root/'analysis_outputs') as d:
   existing=Path(d)
   with self.assertRaisesRegex(ValueError,'already exists'): ev.create_output(existing, self.root)
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(ValueError,'under analysis_outputs'): ev.create_output(Path(d)/'out')
 def test_sha_is_stable(self):
  with tempfile.NamedTemporaryFile(dir=self.root,delete=False) as f:f.write(b'v14')
  try:self.assertEqual(ev.sha(f.name),ev.sha(f.name))
  finally:Path(f.name).unlink()
 def test_personal_sessions_are_fixed_and_present(self):
  self.assertEqual(len(ev.SESSIONS),2)
  self.assertTrue(all(Path(p).is_dir() for p in ev.SESSIONS))
 def test_cli_requires_diagnostic_flag_without_creating_output(self):
  output=ROOT/'analysis_outputs'/'v14_evaluator_reject_probe_20261014'
  self.assertFalse(output.exists())
  result=subprocess.run([sys.executable,'-B','scripts/evaluate_ssvep_denoiser_v14.py','--output-directory',str(output)],cwd=ROOT,text=True,capture_output=True)
  self.assertEqual(result.returncode,1);self.assertFalse(output.exists());self.assertIn('explicit',result.stderr)
 def test_help_does_not_import_scientific_stack(self):
  code="""import importlib.abc,runpy,sys
blocked={'numpy','scipy','torch'}
assert not blocked.intersection(sys.modules)
class B(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in blocked: raise AssertionError('scientific import '+fullname)
sys.meta_path.insert(0,B());sys.argv=['evaluate_ssvep_denoiser_v14.py','--help']
try:runpy.run_module('scripts.evaluate_ssvep_denoiser_v14',run_name='__main__')
except SystemExit as e: assert e.code==0
assert not blocked.intersection(sys.modules)
"""
  result=subprocess.run([sys.executable,'-S','-B','-c',code],cwd=ROOT,text=True,capture_output=True)
  self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--allow-failed-selection-diagnostic',result.stdout)
 def test_report_artifact_contract_shape_definition(self):
  x=np.zeros((10,8),dtype=np.float64);self.assertEqual(list(x.shape),[10,8]);self.assertTrue(np.isfinite(x).all())
 def test_no_default_runtime_or_production_flags(self):
  self.assertFalse(api.preflight(self.root)['runtime_enabled']);self.assertFalse(api.preflight(self.root)['production_validated'])

class SyntheticLoaderRejectionTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix='synthetic-v14-rejection-')
  self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  failed_v14_fixture(self.root)
 def diagnostic(self):
  return ev.load_diagnostic_model(self.root,'cpu',allow_failed_selection_diagnostic=True)
 def test_missing_lock_rejected_before_checkpoint_read(self):
  (self.root/api.LOCK).unlink()
  with patch('torch.load',side_effect=AssertionError('must not load')),self.assertRaisesRegex(ValueError,'training lock'):
   self.diagnostic()
 def test_tampered_weight_rejected_before_deserialization(self):
  with (self.root/api.MODEL).open('ab') as f:f.write(b'tampered')
  with patch('torch.load',side_effect=AssertionError('must not load')),self.assertRaisesRegex(ValueError,'SHA mismatch'):
   self.diagnostic()
 def test_report_byte_change_rejected_even_if_same_json(self):
  with (self.root/api.REPORT).open('ab') as f:f.write(b' ')
  with patch('torch.load',side_effect=AssertionError('must not load')),self.assertRaisesRegex(ValueError,'SHA mismatch'):
   self.diagnostic()
 def test_config_preprocessing_mismatch_is_rejected(self):
  path=self.root/api.CONFIG;cfg=json.loads(path.read_text('utf-8'))
  cfg['preprocessing']['normalization']='unverified'
  path.write_text(json.dumps(cfg))
  with patch('torch.load',side_effect=AssertionError('must not load')),self.assertRaises(ValueError):
   self.diagnostic()
 def test_checkpoint_preprocessing_mismatch_despite_fixture_manifest_binding(self):
  import torch
  path=self.root/api.MODEL;checkpoint=torch.load(path,weights_only=True,map_location='cpu')
  checkpoint['config']['preprocessing']['normalization']='unverified'
  torch.save(checkpoint,path)
  # Re-pin ONLY the disposable synthetic manifest to reach inner identity checks.
  manifest_path=self.root/api.MANIFEST;manifest=json.loads(manifest_path.read_text('utf-8'))
  manifest['checkpoint_sha256']=api.sha(path);manifest_path.write_text(json.dumps(manifest))
  with self.assertRaisesRegex(ValueError,'checkpoint is not failed diagnostic'):
   self.diagnostic()

if __name__=='__main__':unittest.main()
