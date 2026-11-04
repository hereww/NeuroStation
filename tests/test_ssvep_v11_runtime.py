import subprocess
import sys
import unittest
from eeg_tools import ssvep_v11_runtime as runtime

class V11RuntimeTests(unittest.TestCase):
    def test_new_config_and_shared_architecture_are_distinct_metadata(self):
        config=runtime.load_config()
        model=runtime.build_model(config)
        self.assertEqual(config['model_version'],'learned_denoise_v11_personal')
        self.assertEqual(model.config['model_version'],'learned_denoise_v10_personal')
        self.assertEqual(config['training']['identity_loss_weight'],100)
        self.assertFalse(config['runtime']['enabled'])
    def test_import_and_metadata_config_do_not_load_scientific_runtime(self):
        result=subprocess.run([sys.executable,'-c',
            'import sys;from eeg_tools import ssvep_v11_runtime as r;r.load_config();assert not {"torch","numpy","scipy"}&sys.modules.keys()'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
    def test_invalid_device_rejected_before_manifest(self):
        with self.assertRaises(ValueError):runtime.load_model(device='cuda:8')

if __name__=='__main__':unittest.main()
