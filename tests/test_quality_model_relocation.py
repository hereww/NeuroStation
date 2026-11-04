"""Synthetic, locked artifact relocation tests; no personal data or model training."""
import copy,json,tempfile,unittest
from pathlib import Path
import torch
from eeg_tools.model_relocation import load_relocated_model
from eeg_tools.runtime_bindings import sha256
from scripts import train_matched_ssvep_decoder as f

class ModelRelocationTests(unittest.TestCase):
    def fixture(self, root):
        run=root/'run';run.mkdir();data=root/'data';data.mkdir();(data/'fixture.bin').write_bytes(b'synthetic')
        oldcode=root/'old-code';olddata=root/'old-data'
        codes={str(oldcode/Path(k).relative_to(f.ROOT)):v for k,v in f.code_hashes().items()}
        sources={str(olddata/'fixture.bin'):sha256(data/'fixture.bin')}
        spec={'schema_version':1,'status':'formal','axes':{'channel':0,'sample':1,'condition':2,'block':3},
              'expected_shape':[64,1500,40,6],'sample_rate_hz':250,'sample_window':{'start':125,'stop':1375},
              'metadata_authority':'synthetic-fixture-only','source_channel_names':list(f.baseline.CHANNELS)+[f'F{i}' for i in range(56)],
              'target_channel_map':{n:n for n in f.baseline.CHANNELS},'frequency_order_hz':[round(8+.2*i,1) for i in range(40)],
              'mat_variable':'data','depth':'fixed','session':'fixture'}
        p={**f.decoder.FLAGS,'scope':f.SCOPE,'subjects':f.SPLITS,'model':f.decoder.MODEL_SPEC,
           'preprocessing':f.decoder.PREPROCESSING,'gate':f.decoder.GATE,'frequencies_hz':list(f.decoder.FREQUENCIES),
           'commands':{str(k):v for k,v in f.decoder.COMMANDS.items()},'statistical_baselines':f.BASELINE_SPEC,
           'source_spec':spec,'code_hashes':codes,'source_hashes':sources,'training':{'epochs':1}}
        history=[{'epoch':1,'validation_accuracy':.5,'validation_cross_entropy':1.,'train_cross_entropy':1.}]
        (run/'history.json').write_text(json.dumps({'epochs':history}),encoding='utf-8')
        (run/'validation_report.json').write_text('{}',encoding='utf-8')
        torch.save(f.decoder.HarmonicSpectrumMLP().state_dict(),run/'best_weights.pt')
        s={**f.decoder.FLAGS,'scope':f.SCOPE,'subjects':f.SPLITS,'selection_rule':f.SELECTION_RULE,
           'checkpoint_file':'best_weights.pt','checkpoint_format':'state_dict_only','final_test_evaluated':False,
           'artifacts':{n:sha256(run/n) for n in ('best_weights.pt','history.json','validation_report.json')},
           'selected':history[0],'source_hashes':sources,'preregistration_sha256':f.digest(p)}
        self.write(run,p,s)
        return run,p,s,[{'from':str(oldcode),'to':str(f.ROOT)},{'from':str(olddata),'to':str(data)}]
    def write(self,run,p,s):
        s['preregistration_sha256']=f.digest(p)
        for name,value in [('preregistration',p),('selection',s)]:
            (run/(name+'.json')).write_text(json.dumps(value),encoding='utf-8')
            (run/(name+'_lock.json')).write_text(json.dumps({'sha256':f.digest(value)}),encoding='utf-8')
    def test_legal_relocation_preserves_original_manifest_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            run,p,s,m=self.fixture(Path(temp));before={x.name:sha256(x) for x in run.iterdir()}
            model,observed,_=load_relocated_model(run,torch.device('cpu'),m)
            self.assertEqual(observed,p);self.assertEqual(before,{x.name:sha256(x) for x in run.iterdir()})
            self.assertFalse(observed['runtime_enabled'])
    def test_wrong_mapping_and_content_tamper_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            run,p,s,m=self.fixture(Path(temp));wrong=copy.deepcopy(m);wrong[0]['to']=str(Path(temp)/'bad')
            with self.assertRaises(ValueError):load_relocated_model(run,'cpu',wrong)
            (Path(temp)/'data/fixture.bin').write_bytes(b'tampered')
            with self.assertRaises(ValueError):load_relocated_model(run,'cpu',m)
    def test_preprocessing_change_even_with_new_locks_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            run,p,s,m=self.fixture(Path(temp));p=copy.deepcopy(p);p['preprocessing']['unexpected_filter']='1-45'
            self.write(run,p,s)
            with self.assertRaisesRegex(ValueError,'manifest contract'):load_relocated_model(run,'cpu',m)
    def test_weight_and_code_digest_tamper_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            run,p,s,m=self.fixture(Path(temp));(run/'best_weights.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):load_relocated_model(run,'cpu',m)
        with tempfile.TemporaryDirectory() as temp:
            run,p,s,m=self.fixture(Path(temp));p['code_hashes'][next(iter(p['code_hashes']))]='0'*64;self.write(run,p,s)
            with self.assertRaises(ValueError):load_relocated_model(run,'cpu',m)
