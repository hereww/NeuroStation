from __future__ import annotations
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from eeg_tools import ssvep_test_runtime as rt
from eeg_tools.ssvep_denoise_v9 import SSVEPDenoiserV9,CHANGE_BUDGET

spec=importlib.util.spec_from_file_location('v9_cli',rt.ROOT/'scripts/run_ssvep_test_model.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)

class SpectralCore(nn.Module):
    def __init__(self,scale=1.,rate=250):
        super().__init__();self.scale=scale;self.sample_rate_hz=rate
        self.weight=nn.Parameter(torch.tensor(0.,dtype=torch.float32))
    def forward(self,x):
        f=torch.fft.rfftfreq(256,d=1/self.sample_rate_hz)
        hf=torch.fft.irfft(torch.fft.rfft(x)*(f>40),n=256)
        return x-self.scale*hf

def sample(n=512):
    rng=np.random.default_rng(12)
    return rng.standard_normal((n,8)).astype(np.float64)

class AdapterTests(unittest.TestCase):
    def test_large_residual_obeys_per_window_budget(self):
        m=SSVEPDenoiserV9(SpectralCore(0.8));x=torch.randn(17,8,256)
        y,a=m.forward_with_aux(x)
        self.assertTrue(torch.all(a['normalized_change']<=CHANGE_BUDGET))
        self.assertTrue(torch.all(a['gain']<1))
        self.assertEqual(y.dtype,torch.float32)
    def test_small_residual_is_not_increased(self):
        m=SSVEPDenoiserV9(SpectralCore(.0001));x=torch.randn(3,8,256)
        y,a=m.forward_with_aux(x)
        self.assertTrue(torch.equal(y,m.core(x)))
        self.assertTrue(torch.all(a['gain']==1))
    def test_zero_residual_exact_identity(self):
        m=SSVEPDenoiserV9(SpectralCore(0));x=torch.randn(4,8,256)
        self.assertTrue(torch.equal(m(x),x))
    def test_protected_bins_at_both_native_rates(self):
        for rate in (250,256):
            x=torch.randn(4,8,256);m=SSVEPDenoiserV9(SpectralCore(.8,rate))
            f=torch.fft.rfftfreq(256,d=1/rate);delta=torch.fft.rfft(m(x)-x)
            self.assertLess(float(delta[..., (f>=8)&(f<=40)].abs().max()),1e-5)
    def test_high_frequency_not_amplified(self):
        x=torch.randn(2,8,256);m=SSVEPDenoiserV9(SpectralCore(.8));f=torch.fft.rfftfreq(256,d=1/250)
        before=torch.fft.rfft(x)[...,f>40].abs();after=torch.fft.rfft(m(x))[...,f>40].abs()
        self.assertTrue(torch.all(after<=before+1e-5))
    def test_frozen_core_no_parameter_mutation(self):
        m=SSVEPDenoiserV9(SpectralCore(.8));before={k:v.clone() for k,v in m.state_dict().items()}
        m(torch.randn(3,8,256))
        self.assertTrue(all(not p.requires_grad for p in m.parameters()))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in m.state_dict().items()))
    def test_nonfinite_prediction_refused(self):
        m=SSVEPDenoiserV9(SpectralCore(float('nan')))
        with self.assertRaisesRegex(ValueError,'nonfinite'):m(torch.ones(1,8,256))

class ProcessingTests(unittest.TestCase):
    def setUp(self):self.m=SSVEPDenoiserV9(SpectralCore(.8))
    def run_signal(self,x,mask=None,**kwargs):
        return rt.process_signal(x,np.ones(len(x),dtype=bool) if mask is None else mask,self.m,sample_rate_hz=250,**kwargs)
    def test_copy_shape_budget_and_tail(self):
        x=sample(700);before=x.copy();x.setflags(write=False)
        out,mask,r=self.run_signal(x)
        np.testing.assert_array_equal(x,before);self.assertEqual(out.shape,x.shape)
        np.testing.assert_array_equal(out[512:],x[512:]);self.assertEqual(mask.sum(),512)
        self.assertTrue(all(w['normalized_change']<=CHANGE_BUDGET*1.0001 for w in r['windows'] if w['status']=='processed'))
        self.assertFalse(r['boundary_and_full_record_spectral_preservation_validated'])
    def test_invalid_entire_block_preserved(self):
        x=sample();mask=np.ones(512,dtype=bool);mask[10]=False
        out,processed,r=self.run_signal(x,mask)
        np.testing.assert_array_equal(out[:256],x[:256]);self.assertEqual(processed.sum(),256)
    def test_nonfinite_entire_block_preserved(self):
        x=sample();x[1,3]=np.nan;x[2,5]=np.inf
        out,mask,r=self.run_signal(x)
        np.testing.assert_array_equal(out[:256],x[:256]);self.assertFalse(mask[:256].any())
    def test_flat_channel_preserves_window(self):
        x=sample();x[:256,1]=100
        out,mask,r=self.run_signal(x)
        np.testing.assert_array_equal(out[:256],x[:256]);self.assertEqual(r['windows'][0]['status'],'flat_channel_window_preserved')
    def test_explicit_rate_required(self):
        with self.assertRaisesRegex(ValueError,'rate'):rt.process_signal(sample(),np.ones(512,dtype=bool),self.m,sample_rate_hz=256)
    def test_mask_dtype_shape_refused(self):
        for mask in (np.ones(512),np.ones((512,1),dtype=bool),np.ones(500,dtype=bool)):
            with self.subTest(shape=mask.shape),self.assertRaises(ValueError):self.run_signal(sample(),mask)
    def test_wrong_shape_dtype_batch_refused(self):
        for x in (np.ones((512,7)),np.ones((8,256)),np.ones((255,8)),np.ones((256,8),dtype='int64')):
            with self.assertRaises(ValueError):self.run_signal(x)
        for batch in (0,257,True,1.2):
            with self.assertRaises(ValueError):self.run_signal(sample(),batch_size=batch)
    def test_chunk_invariance(self):
        x=sample(1024);a=self.run_signal(x,batch_size=1)[0];b=self.run_signal(x,batch_size=4)[0]
        np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-6)
    def test_float32_supported_output_float64(self):
        x=sample().astype('float32');out,mask,r=self.run_signal(x)
        self.assertEqual(out.dtype,np.float64);self.assertEqual(mask.sum(),512)
    def test_numeric_units_and_dc_retained(self):
        x=sample()+1000;out,mask,r=self.run_signal(x)
        np.testing.assert_allclose(out.reshape(2,256,8).mean(axis=1),x.reshape(2,256,8).mean(axis=1),atol=1e-5)

class IOTests(unittest.TestCase):
    def test_fixed_config_and_bool_substitution(self):
        rt.validate_config(rt.EXPECTED_CONFIG)
        for key,value in [('schema_version',True),('change_budget_normalized_mse',.0005),('status','draft')]:
            cfg={**rt.EXPECTED_CONFIG,key:value}
            with self.assertRaises(ValueError):rt.validate_config(cfg)
    def test_duplicate_nonfinite_json_refused(self):
        for data in (b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":Infinity}'):
            with self.assertRaises(ValueError):rt.json_read(data)
    def test_traversal_unc_ads_and_device_refused(self):
        for path in ('../bad.npy',r'\\server\x',r'file.npy:evil','NUL','COM1.npy',''):
            with self.subTest(path=path),self.assertRaises(ValueError):rt.safe_local_path(path)
    def test_exclusive_write_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x';rt.exclusive_write(p,b'old')
            with self.assertRaises(FileExistsError):rt.exclusive_write(p,b'new')
            self.assertEqual(p.read_bytes(),b'old')
    def test_hash_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x';p.write_bytes(b'old')
            with self.assertRaisesRegex(ValueError,'SHA256'):rt.verify_files({'x':'0'*64},Path(tmp))
    def test_source_output_ancestor_and_existing_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'analysis_outputs';a.mkdir();source=a/'new'/'input.npy'
            with self.assertRaises(ValueError):rt.checked_output_dir(a/'new',sources=(source,),root=root)
            with self.assertRaises(ValueError):rt.checked_output_dir(root/'datasets',root=root)
            (a/'existing').mkdir()
            with self.assertRaises(ValueError):rt.checked_output_dir(a/'existing',root=root)
    def test_formats_roundtrip_and_object_refusal(self):
        x=sample(256);b=io.BytesIO();np.save(b,x);np.testing.assert_array_equal(rt.load_signal(b.getvalue(),'.npy'),x)
        b=io.BytesIO();np.savez(b,signal=x);np.testing.assert_array_equal(rt.load_signal(b.getvalue(),'.npz'),x)
        b=io.BytesIO();np.savetxt(b,x,delimiter=',');np.testing.assert_allclose(rt.load_signal(b.getvalue(),'.csv'),x,atol=1e-12)
        b=io.BytesIO();np.save(b,np.array([{}],dtype=object))
        with self.assertRaises(ValueError):rt.load_signal(b.getvalue(),'.npy')
    def test_missing_npz_key_refused(self):
        b=io.BytesIO();np.savez(b,wrong=sample(256))
        with self.assertRaises(ValueError):rt.load_signal(b.getvalue(),'.npz')

class CLITests(unittest.TestCase):
    def setUp(self):
        self.manifest={'domain_internal_gates':{'Nakanishi2015':False,'Wearable':False},'files':{rt.MODEL:'a'*64}}
        self.patcher=patch.object(rt,'preflight',return_value=(rt.EXPECTED_CONFIG,{},self.manifest));self.patcher.start();self.addCleanup(self.patcher.stop)
    def runcli(self,args):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):return cli.main(args)
    def test_checkonly_no_torch_or_inference_call(self):
        with patch.object(rt,'load_models',side_effect=AssertionError('no load')):
            self.assertEqual(self.runcli(['--check-only']),0)
    def test_missing_native_rate_and_mapping_refused(self):
        self.assertEqual(self.runcli(['--input','no.npy','--domain','Wearable','--sample-rate-hz','256']),1)
        self.assertEqual(self.runcli(['--input','no.npy','--domain','Wearable','--sample-rate-hz','250']),1)
    def test_missing_mask_and_acknowledgements_refused(self):
        self.assertEqual(self.runcli(['--input','no.npy','--domain','Wearable','--sample-rate-hz','250','--channel-order','0,1,2,3,4,5,6,7']),1)
    def test_cli_full_fake_model_and_input_unchanged(self):
        with tempfile.TemporaryDirectory(dir=rt.ROOT/'analysis_outputs') as tmp:
            root=Path(tmp);inp=root/'x.npy';inp.write_bytes(cli.npy_bytes(sample(600)));before=inp.read_bytes()
            args=['--input',str(inp),'--domain','Wearable','--sample-rate-hz','250','--channel-order','0,1,2,3,4,5,6,7',
                  '--all-finite-valid','--acknowledge-offline-test','--acknowledge-unverified-channel-mapping','--output',str(root/'result')]
            with patch.object(rt,'load_models',return_value=({'Wearable':SSVEPDenoiserV9(SpectralCore(.8))},{'device':'cpu'})):
                self.assertEqual(self.runcli(args),0)
                self.assertEqual(self.runcli(args),1)
            self.assertEqual(inp.read_bytes(),before)
            r=json.loads((root/'result'/'report.json').read_text(encoding='utf-8'))
            self.assertTrue(r['inference_executed']);self.assertFalse(r['hardware_opened']);self.assertFalse(r['production_validated'])
            self.assertEqual(r['processing']['processed_samples'],512)

if __name__=='__main__':unittest.main()
