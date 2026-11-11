"""Synthetic-only tests of v8 diagnostics; never real EEG/PT/NPZ or CUDA."""
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from eeg_tools import ssvep_v8_internal_eval as e
from eeg_tools import ssvep_v8_experiment as experiment
from tests.synthetic_research_fixtures import embedded_preregistration_bytes


def config():
    # Metadata only: fixed config already present, never real data.
    return json.loads((e.ROOT/e.CONFIG_RELATIVE).read_text(encoding='utf-8'))


def cases():
    cfg=config(); rows=[]
    for d in e.DATASETS:
        for sid in (5,6):
            for trial in range(e.TRIALS[d]):
                for k in e.CONDITIONS:
                    identity=k=='identity'; a=0. if identity else 1.; b=.0001 if identity else .9
                    rows.append({'dataset':d,'subject_id':sid,'trial_index':trial,'condition':k,'seed':e.case_seed(d,sid,trial,k,cfg),'window_samples':256,'window_start_sample':e.STARTS[d],'sample_rate_hz':e.RATES[d],'input_normalized_mse':a,'output_normalized_mse':b,'relative_reduction':None if identity else (a-b)/a,'identity_normalized_change':b if identity else None,'outcome':'loss' if identity else 'win','protected_complex_energy_ratio':1e-10 if identity else None,'protected_weighted_abs_phase_radians':1e-5 if identity else None,'undefined_phase_bin_count':0})
    return rows


class CohortTests(unittest.TestCase):
    def setUp(self):self.cfg=config(); self.rows=cases()
    def test_complete_exact4200_native_rates(self):
        r=e.validate_cases(self.rows,self.cfg); self.assertEqual(r['case_count'],4200);self.assertEqual(r['phase_case_count'],840);self.assertTrue(r['complete_cohort'] and r['phase_verifiable'])
    def test_missing_case_refused(self):
        with self.assertRaises(e.DiagnosticError):e.validate_cases(self.rows[:-1],self.cfg)
    def test_duplicate_case_refused(self):
        self.rows[-1]=dict(self.rows[0])
        with self.assertRaises(e.DiagnosticError):e.validate_cases(self.rows,self.cfg)
    def test_outer_subject_refused(self):
        self.rows[0]['subject_id']=7
        with self.assertRaises(e.DiagnosticError):e.validate_cases(self.rows,self.cfg)
    def test_holdout_subject_refused(self):
        self.rows[0]['subject_id']=9
        with self.assertRaises(e.DiagnosticError):e.validate_cases(self.rows,self.cfg)
    def test_seed_dtype_and_compatibility(self):
        self.rows[0]['seed']+=1
        with self.assertRaises(e.DiagnosticError):e.validate_cases(self.rows,self.cfg)
    def test_wrong_native_rate_refused(self):
        self.rows[0]['sample_rate_hz']=250
        with self.assertRaises(e.DiagnosticError):e.validate_cases(self.rows,self.cfg)
    def test_nonfinite_refused(self):
        self.rows[0]['output_normalized_mse']=float('nan')
        with self.assertRaises(e.DiagnosticError):e.validate_cases(self.rows,self.cfg)
    def test_bool_mse_refused(self):
        self.rows[0]['output_normalized_mse']=False
        with self.assertRaises(e.DiagnosticError):e.validate_cases(self.rows,self.cfg)
    def test_wrong_reduction_refused(self):
        self.rows[1]['relative_reduction']=.5
        with self.assertRaises(e.DiagnosticError):e.validate_cases(self.rows,self.cfg)
    def test_waveform_and_labels_refused(self):
        self.rows[0]['frequency_label']=8
        with self.assertRaises(e.DiagnosticError):e.validate_cases(self.rows,self.cfg)
    def test_undefined_null_preserved_and_blocked(self):
        self.rows[0]['protected_weighted_abs_phase_radians']=None
        self.rows[0]['undefined_phase_bin_count']=2
        result=e.summarize_diagnostic(self.rows,self.cfg)
        self.assertFalse(result['phase_verifiable'] or result['all_internal_gates_passed'])
        self.assertEqual(result['undefined_phase_case_count'],1)
        self.assertIsNone(result['aggregates']['dataset']['Nakanishi2015']['identity']['mean_protected_weighted_abs_phase_radians'])
    def test_phase_missing_support_null_blocks(self):
        self.rows[0]['protected_complex_energy_ratio']=None
        result=e.summarize_diagnostic(self.rows,self.cfg)
        self.assertFalse(result['all_internal_gates_passed'])
    def test_phase_undefined_cannot_fake_zero(self):
        self.rows[0]['undefined_phase_bin_count']=1
        with self.assertRaises(e.DiagnosticError):e.validate_cases(self.rows,self.cfg)
    def test_phase_above_pi_refused(self):
        self.rows[0]['protected_weighted_abs_phase_radians']=math.pi+.1
        with self.assertRaises(e.DiagnosticError):e.validate_cases(self.rows,self.cfg)
    def test_nonidentity_phase_refused(self):
        self.rows[1]['protected_complex_energy_ratio']=0.
        with self.assertRaises(e.DiagnosticError):e.validate_cases(self.rows,self.cfg)
    def test_model_state_not_part_of_rows(self):
        self.rows[0]['reference']=[]
        with self.assertRaises(e.DiagnosticError):e.validate_cases(self.rows,self.cfg)


class SummaryTests(unittest.TestCase):
    def setUp(self):self.cfg=config();self.rows=cases()
    def test_domains_separate_and_global_descriptive(self):
        with self.assertRaises(e.DiagnosticError):e.summarize(self.rows)
        result=e.summarize_diagnostic(self.rows,self.cfg)
        self.assertTrue(result['all_internal_gates_passed'])
        self.assertEqual(set(result['dataset_gates']),set(e.DATASETS))
        self.assertEqual(result['aggregates']['global']['identity']['case_count'],840)
        self.assertEqual(result['aggregates']['dataset']['Nakanishi2015']['identity']['case_count'],360)
        self.assertEqual(result['aggregates']['dataset']['Wearable']['identity']['case_count'],480)
    def test_ratio_of_means_not_mean_reduction(self):
        selected=[r for r in self.rows if r['dataset']=='Nakanishi2015' and r['condition']=='transient'][:2]
        selected[0].update(input_normalized_mse=1.,output_normalized_mse=.5)
        selected[1].update(input_normalized_mse=9.,output_normalized_mse=9.)
        row=e.summarize(selected)['transient']
        self.assertAlmostEqual(row['ratio_of_mean_reduction'],.05)
        self.assertNotAlmostEqual(row['ratio_of_mean_reduction'],.25)
    def test_hf_failure_blocks_only_one_domain(self):
        for row in self.rows:
            if row['dataset']=='Wearable' and row['condition']=='high_frequency_burst':
                row.update(output_normalized_mse=1.1,relative_reduction=(1.-1.1),outcome='loss')
        result=e.summarize_diagnostic(self.rows,self.cfg)
        self.assertFalse(result['all_internal_gates_passed'])
        self.assertTrue(result['dataset_gates']['Nakanishi2015']['all_passed'])
        self.assertFalse(result['dataset_gates']['Wearable']['all_passed'])
    def test_exact_identity_threshold_inclusive(self):
        summary=e.summarize([r for r in self.rows if r['dataset']=='Wearable'])
        summary['identity']['mean_identity_normalized_change']=.0005
        self.assertTrue(e.gates(summary,self.cfg)['all_passed'])
        summary['identity']['mean_identity_normalized_change']=.0005000001
        self.assertFalse(e.gates(summary,self.cfg)['all_passed'])
    def test_frozen_thresholds_refuse_changes(self):
        self.cfg['internal_research_gates']['high_frequency_burst_mean_error_reduction_min']=0.
        with self.assertRaises(e.DiagnosticError):e.gates({},self.cfg)
    def test_empty_summary_refused(self):
        with self.assertRaises(e.DiagnosticError):e.summarize([])
    def test_nonbool_mixed_refused(self):
        with self.assertRaises(e.DiagnosticError):e.summarize(self.rows,mixed=1)
    def test_all_group_counts(self):
        r=e.summarize_diagnostic(self.rows,self.cfg)
        self.assertEqual(len(r['aggregates']['subject']),4)
        self.assertEqual(len(r['aggregates']['subject_condition']),20)
        self.assertEqual(len(r['aggregates']['dataset_condition']),10)
        self.assertEqual(sum(sum(s['case_count'] for s in v.values()) for v in r['aggregates']['subject'].values()),4200)
    def test_seed_fresh_and_no_collision(self):
        seeds=[r['seed'] for r in self.rows]
        self.assertEqual(len(set(seeds)),4200)
        self.assertGreater(min(seeds),2**32)
        self.assertNotEqual(self.cfg['seeds']['training'],self.cfg['seeds']['internal_selection'])


class SafetyTests(unittest.TestCase):
    def test_cold_import_no_torch_or_model(self):
        p=subprocess.run([sys.executable,'-B','-c',"import sys; import eeg_tools.ssvep_v8_internal_eval; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; assert 'eeg_tools.ssvep_denoise_v8' not in sys.modules"],cwd=e.ROOT,capture_output=True,text=True,timeout=60)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
    def test_cli_overrides_refused_before_evaluate(self):
        with patch.object(e,'evaluate',side_effect=AssertionError('must not execute')):
            with self.assertRaises(SystemExit):e.main(['--device','cpu'])
    def test_checkonly_state_cannot_infer(self):
        s=e.State(Path('.'),config(),{},'0'*64,'0'*64,(),{},{},True)
        with self.assertRaises(e.DiagnosticError):e.infer_cases(s,[],{},None)
        with self.assertRaises(e.DiagnosticError):e.run_diagnostic(s)
    def test_output_path_traversal_refused(self):
        with self.assertRaises(ValueError):e.output_paths(e.ROOT,'analysis_outputs/../foreign')
    def test_foreign_drive_refused(self):
        with self.assertRaises(ValueError):e.output_paths(e.ROOT,'D:/SSVEP/out')
    def test_checkpoint_output_refused(self):
        with self.assertRaises(ValueError):e.output_paths(e.ROOT,'models/denoiser_v8_ssvep')
    def test_collision_either_output_refused(self):
        with tempfile.TemporaryDirectory(dir=e.ROOT.parent) as tmp:
            root=Path(tmp);(root/'analysis_outputs').mkdir()
            (root/(e.PREFIX+'.md')).write_text('synthetic')
            with self.assertRaises(ValueError):e.output_paths(root)
    def test_no_nan_or_tensor_json(self):
        self.assertFalse(e.finite_tree({'x':float('inf')}))
        self.assertFalse(e.finite_tree({'x':object()}))
        self.assertTrue(e.finite_tree({'x':[0,False,None,'s',.1]}))
    def test_markdown_explicit_internal_and_blocked(self):
        text=e.markdown({'all_internal_gates_passed':True,'case_count':4200}).decode()
        self.assertIn('deployment blocked',text)
        self.assertIn('not independent validation',text)
    def test_missing_lock_refused_no_torch(self):
        with tempfile.TemporaryDirectory(dir=e.ROOT.parent) as tmp:
            with self.assertRaises(ValueError):e.preflight(Path(tmp))



class PreflightFixture:
    """56 synthetic bindings: real config metadata, arbitrary binary placeholders."""
    def __init__(self,root):
        self.root=root
        protocol=json.loads(embedded_preregistration_bytes(experiment))
        for rel in protocol['bound_files']:
            self.put(rel,(e.ROOT/e.CONFIG_RELATIVE).read_bytes() if rel==e.CONFIG_RELATIVE else ('SYNTHETIC '+rel).encode())
        protocol['bound_files']={p:self.digest(p) for p in protocol['bound_files']}
        self.put(e.PREREG_RELATIVE,(json.dumps(protocol)+'\n').encode())
        self.config_sha=self.digest(e.CONFIG_RELATIVE);self.prereg_sha=self.digest(e.PREREG_RELATIVE)
        files={**protocol['bound_files'],e.PREREG_RELATIVE:self.prereg_sha}
        for rel in protocol['new_source_files']:
            self.put(rel,('SYNTHETIC source '+rel).encode());files[rel]=self.digest(rel)
        scope=dict.fromkeys(e.FALSE_FLAGS,False)
        train={'schema_version':1,'status':'formal','stage':'frozen_before_v8_training_waveform_read','files':files,'scope':scope,'waveforms_loaded':False,'checkpoint_loaded':False}
        self.put(e.TRAIN_LOCK,(json.dumps(train)+'\n').encode())
        self.put(e.CHECKPOINT,b'SYNTHETIC PT not a serialized weight')
        self.put(e.TRAIN_REPORT,b'{"synthetic_only":true}\n')
        files={**files,e.TRAIN_LOCK:self.digest(e.TRAIN_LOCK),e.CHECKPOINT:self.digest(e.CHECKPOINT),e.TRAIN_REPORT:self.digest(e.TRAIN_REPORT)}
        evaluation={'schema_version':1,'status':'formal','stage':'frozen_before_v8_internal_diagnostic','files':files,'scope':scope,'training_performed':True,'training_lock_sha256':files[e.TRAIN_LOCK],'checkpoint_sha256':files[e.CHECKPOINT]}
        self.put(e.EVAL_LOCK,(json.dumps(evaluation)+'\n').encode())
        for rel in e.CODE_PATHS:self.put(rel,('SYNTHETIC evaluator '+rel).encode())
        code={'schema_version':1,'status':'formal','stage':'frozen_before_v8_internal_diagnostic_code_execution','files':{p:self.digest(p) for p in e.CODE_PATHS},'scope':scope}
        self.put(e.CODE_LOCK,(json.dumps(code)+'\n').encode())
    def put(self,rel,data):
        p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    def digest(self,rel):return hashlib.sha256((self.root/rel).read_bytes()).hexdigest()


class PreflightIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='v8_diag_SYNTHETIC_',dir=e.ROOT.parent)
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.f=PreflightFixture(self.root)
        self.hashes=patch.multiple(e,CONFIG_SHA256=self.f.config_sha,PREREG_SHA256=self.f.prereg_sha)
        self.hashes.start();self.addCleanup(self.hashes.stop)
    def test_checkonly68_binding_no_binary_byte_reads(self):
        c=e._c();original=c._read_bytes;original_hash=c.file_sha256;seen=[]
        def guarded(root,rel):
            seen.append(rel)
            if Path(rel).suffix in ('.npz','.pt'):raise AssertionError('forbidden binary read in checkonly')
            return original(root,rel)
        def guarded_hash(root,rel):
            seen.append(rel)
            if Path(rel).suffix in ('.npz','.pt'):raise AssertionError('forbidden binary hash in checkonly')
            return original_hash(root,rel)
        with patch.object(c,'_read_bytes',side_effect=guarded), patch.object(c,'file_sha256',side_effect=guarded_hash):
            state=e.preflight(self.root,check_only=True);e.verify_frozen(state)
        self.assertEqual(len(state.files),68)
        self.assertEqual(len(state.deferred),14)
        self.assertEqual(len(state.verified),54)
        self.assertFalse(any('subject_007' in p or 'subject_008' in p or 'subject_009' in p or 'subject_010' in p for p in seen))
    def test_full_diagnostic_hashes_internal5_6_and_pt_not_fit(self):
        c=e._c();original=c._read_bytes;original_hash=c.file_sha256;seen=[]
        def guarded(root,rel):
            seen.append(rel)
            if Path(rel).suffix=='.npz' and Path(rel).name not in ('subject_005.npz','subject_006.npz'):
                raise AssertionError('fit data bytes forbidden in diagnostic')
            return original(root,rel)
        def guarded_hash(root,rel):
            seen.append(rel)
            if Path(rel).suffix=='.npz' and Path(rel).name not in ('subject_005.npz','subject_006.npz'):
                raise AssertionError('fit NPZ hashing forbidden in diagnostic')
            return original_hash(root,rel)
        with patch.object(c,'_read_bytes',side_effect=guarded), patch.object(c,'file_sha256',side_effect=guarded_hash):
            state=e.preflight(self.root,check_only=False);e.verify_frozen(state)
        self.assertEqual(len(state.deferred),8)
        self.assertEqual(len(state.verified),60)
        self.assertIn(e.CHECKPOINT,seen)
    def test_code_freeze_mutation_refused(self):
        state=e.preflight(self.root)
        self.f.put(e.CODE_PATHS[0],b'synthetic tamper')
        with self.assertRaises(ValueError):e.verify_frozen(state)
    def test_eval_lock_mutation_refused(self):
        state=e.preflight(self.root)
        self.f.put(e.EVAL_LOCK,b'{}')
        with self.assertRaises(ValueError):e.verify_frozen(state)
    def test_true_unsafe_scope_refused(self):
        r=json.loads((self.root/e.EVAL_LOCK).read_text());r['scope']['holdout_waveforms_loaded']=True
        self.f.put(e.EVAL_LOCK,json.dumps(r).encode())
        with self.assertRaises(ValueError):e.preflight(self.root)
    def test_training_must_have_occurred(self):
        r=json.loads((self.root/e.EVAL_LOCK).read_text());r['training_performed']=False
        self.f.put(e.EVAL_LOCK,json.dumps(r).encode())
        with self.assertRaises(ValueError):e.preflight(self.root)
    def test_extra_lock_source_refused(self):
        r=json.loads((self.root/e.EVAL_LOCK).read_text());r['files']['foreign.npz']='0'*64
        self.f.put(e.EVAL_LOCK,json.dumps(r).encode())
        with self.assertRaises(ValueError):e.preflight(self.root)
    def test_missing_code_lock_field_refused(self):
        r=json.loads((self.root/e.CODE_LOCK).read_text());r['files'].pop(e.CODE_PATHS[0])
        self.f.put(e.CODE_LOCK,json.dumps(r).encode())
        with self.assertRaises(ValueError):e.preflight(self.root)




class RealModelSyntheticTransportTests(unittest.TestCase):
    def test_actual_model4200_diagnostic_no_gpu_no_eeg(self):
        """Real math/forward, in-memory synthetic records; CUDA transfer replaced only."""
        import numpy as np
        import torch
        from eeg_tools import ssvep_denoise_v8 as model_api
        cfg=config();before_cuda=torch.cuda.is_initialized()
        original_threads=torch.get_num_threads();torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads,original_threads)
        models={d:model_api.build_model(cfg,e.RATES[d]) for d in e.DATASETS}
        records=[]
        for d in e.DATASETS:
            length=1114 if d=='Nakanishi2015' else 710
            t=np.arange(length,dtype=np.float64)/e.RATES[d]
            window=np.stack([np.sin(2*np.pi*(11.3+ch*.7)*t)+.2*np.cos(2*np.pi*2.3*t) for ch in range(8)],axis=-1)
            for sid in (5,6):
                signal=np.broadcast_to(window,(e.TRIALS[d],length,8)).copy()
                records.append(SimpleNamespace(dataset=d,subject_id=sid,sample_rate_hz=e.RATES[d],signal=signal))
        class Transport:
            float32=torch.float32
            @staticmethod
            def inference_mode():return torch.inference_mode()
            @staticmethod
            def isfinite(v):return torch.isfinite(v)
            @staticmethod
            def from_numpy(v):
                class Transfer:
                    def to(self,*,device,dtype):
                        assert device=='cuda:0' and dtype==torch.float32
                        return torch.from_numpy(v).to(device='cpu',dtype=dtype)
                return Transfer()
        state=e.State(e.ROOT,cfg,{},'0'*64,'0'*64,(),{},{},False)
        modes=[]
        hooks=[m.register_forward_pre_hook(lambda m,args:modes.append(torch.is_grad_enabled())) for m in models.values()]
        try:
            with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('actual CUDA forbidden')):
                rows=e.infer_cases(state,records,models,Transport)
        finally:
            for hook in hooks:hook.remove()
        result=e.summarize_diagnostic(rows,cfg)
        self.assertEqual(result['case_count'],4200)
        self.assertEqual(result['phase_case_count'],840)
        self.assertFalse(result['all_internal_gates_passed'])  # Untrained identity cannot pass HF improvement gate.
        self.assertTrue(result['phase_verifiable'])
        self.assertTrue(modes and not any(modes))
        self.assertTrue(all(p.grad is None for m in models.values() for p in m.parameters()))
        for row in rows:
            self.assertEqual(row['input_normalized_mse'],row['output_normalized_mse'])
        self.assertEqual(torch.cuda.is_initialized(),before_cuda)



if __name__=='__main__':unittest.main()
