import unittest
import numpy as np
from eeg_tools import ssvep_denoise_v5 as core

@unittest.skipIf(core.torch is None, "torch unavailable")
class HannOLA(unittest.TestCase):
    def test_identity_model_preserves_native_signal(self):
        torch=core.torch
        class Identity(torch.nn.Module):
            def forward(self,x): return x
        rng=np.random.default_rng(20261007); x=rng.normal(size=(2,710,8)).astype(np.float64)
        y=core.denoise_array(Identity(),x,250,core.load_config(),"cpu")
        np.testing.assert_allclose(y,x,rtol=0,atol=1e-12)
    def test_short_signal_is_refused(self):
        x=np.zeros((1,255,8),dtype=np.float64)
        with self.assertRaises(core.SSVEPDenoiseV5Error): core.denoise_array(type("I",(),{"eval":lambda s:None})(),x,250,core.load_config(),"cpu")

if __name__=="__main__": unittest.main()
