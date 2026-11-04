import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from test_denoise import DenoisePipelineTests, PIPELINE
from eeg_tools.denoise import run_denoise_pipeline
from eeg_tools.quality.cache import implementation_identity
from eeg_tools.runtime_bindings import sha256

class CacheTests(unittest.TestCase):
    def test_tamper_never_hits_or_overwrites_old_revision(self):
        with tempfile.TemporaryDirectory() as root:
            session,_=DenoisePipelineTests()._make_session(Path(root))
            before=sha256(session/'raw_brainflow.tsv')
            first=run_denoise_pipeline(session,PIPELINE); out=Path(first['output_dir'])
            (out/'denoised_50hz.npz').write_bytes(b'corrupt')
            second=run_denoise_pipeline(session,PIPELINE)
            self.assertEqual(second['status'],'completed');self.assertNotEqual(second['output_dir'],first['output_dir'])
            self.assertEqual((out/'denoised_50hz.npz').read_bytes(),b'corrupt')
            self.assertEqual(sha256(session/'raw_brainflow.tsv'),before)
    def test_implementation_change_and_partial_output(self):
        with tempfile.TemporaryDirectory() as root:
            session,_=DenoisePipelineTests()._make_session(Path(root)); first=run_denoise_pipeline(session,PIPELINE)
            original=implementation_identity()
            with patch('eeg_tools.quality.cache.implementation_identity',return_value={**original,'test_revision':2}):
                second=run_denoise_pipeline(session,PIPELINE)
            self.assertNotEqual(first['cache_key'],second['cache_key'])
            self.assertNotEqual(first['output_dir'],second['output_dir'])
            (Path(first['output_dir'])/'complete_v2.json').unlink()
            third=run_denoise_pipeline(session,PIPELINE)
            self.assertEqual(third['status'],'completed')
    def test_input_change_invalidates_without_raw_rewrite(self):
        with tempfile.TemporaryDirectory() as root:
            session,_=DenoisePipelineTests()._make_session(Path(root)); first=run_denoise_pipeline(session,PIPELINE)
            p=session/'events.tsv';p.write_text(p.read_text(encoding='utf-8')+'\n',encoding='utf-8')
            second=run_denoise_pipeline(session,PIPELINE)
            self.assertNotEqual(first['cache_key'],second['cache_key'])
