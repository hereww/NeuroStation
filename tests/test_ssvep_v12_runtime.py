import subprocess,sys,unittest
import numpy as np
from eeg_tools import ssvep_v12_runtime as runtime

class V12RuntimeTests(unittest.TestCase):
 def test_shared_statistics_identity_and_reference_independence(self):
  rng=np.random.default_rng(5); noisy=rng.normal(size=(3,8,512)).astype(np.float32)+100
  normalized,means,rms=runtime.normalize_noisy(noisy); target=runtime.target_in_noisy_coordinates(noisy,means,rms); restored=runtime.restore_residual(noisy,normalized,normalized,rms)
  self.assertEqual(normalized.dtype,np.float32);self.assertEqual(means.dtype,np.float64);self.assertEqual(rms.dtype,np.float64)
  np.testing.assert_allclose(target,normalized,atol=1e-6);np.testing.assert_allclose(restored,noisy,atol=1e-12)
  changed=noisy.copy();changed[0,0,0]+=99
  normalized2,_,_=runtime.normalize_noisy(noisy)
  np.testing.assert_array_equal(normalized,normalized2)
 def test_rejects_flat_nonfinite_and_wrong_shape(self):
  for value in (np.zeros((2,8,512),dtype=np.float32),np.full((2,8,512),np.nan,dtype=np.float32),np.zeros((2,7,512),dtype=np.float32)):
   with self.assertRaises(ValueError):runtime.normalize_noisy(value)
 def test_import_no_scientific_modules(self):
  result=subprocess.run([sys.executable,'-c','import sys;import eeg_tools.ssvep_v12_runtime as r;r.load_config();assert not {"numpy","scipy","torch"}&sys.modules.keys()'],capture_output=True,text=True)
  self.assertEqual(result.returncode,0,result.stderr)
if __name__=='__main__':unittest.main()
