import unittest
import json
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
from eeg_tools import ssvep_denoise_v5 as core

class V5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[1]; cls.cfg=core.load_config(root=cls.root)
    def test_config_formal_and_runtime_disabled(self):
        self.assertEqual(self.cfg["status"],"formal"); self.assertFalse(self.cfg["runtime"]["enabled"]); self.assertEqual(self.cfg["training"]["fit_subjects"],[1,2,3,4])
    def test_corruption_is_deterministic_and_shape_preserving(self):
        x=np.zeros((8,256),dtype=np.float64); x[0]=np.sin(np.arange(256)/10)
        a=core.corrupt_window(x,250,self.cfg,42); b=core.corrupt_window(x,250,self.cfg,42)
        np.testing.assert_array_equal(a[0],b[0]); self.assertEqual(a[1:],b[1:]); self.assertEqual(a[0].shape,x.shape)
    def test_corruption_has_identity_and_nonidentity_seeds(self):
        x=np.ones((8,256),dtype=np.float64)
        identity=[]
        for seed in range(100): identity.append(core.corrupt_window(x,250,self.cfg,seed)[1])
        self.assertIn(True,identity); self.assertIn(False,identity)
    def test_normalization_finite(self):
        x=np.random.default_rng(1).normal(size=(8,256)); n,m,s=core.normalize_window(x)
        self.assertEqual(n.dtype,np.float32); self.assertEqual(n.shape,x.shape); self.assertTrue(np.isfinite(n).all()); self.assertTrue(np.isfinite(m).all()); self.assertTrue(np.isfinite(s).all())
    @unittest.skipIf(core.torch is None,"torch unavailable")
    def test_model_forward_and_loss(self):
        torch=core.torch; model=core.build_model(self.cfg); x=torch.randn(4,8,256); target=x.clone(); identity=torch.tensor([True,False,True,False]); pred=model(x); self.assertEqual(tuple(pred.shape),(4,8,256)); losses=core.compute_loss(pred,target,x,identity,250,self.cfg); self.assertTrue(all(bool(torch.isfinite(v)) for v in losses.values()))
    @unittest.skipIf(core.torch is None,"torch unavailable")
    def test_model_parameter_count_and_cpu_inference_shape(self):
        torch=core.torch; model=core.build_model(self.cfg); self.assertGreater(sum(p.numel() for p in model.parameters()),1000); x=np.random.default_rng(2).normal(size=(2,300,8)); y=core.denoise_array(model,x,250,self.cfg,"cpu"); self.assertEqual(y.shape,x.shape); self.assertEqual(y.dtype,np.float64); self.assertTrue(np.isfinite(y).all())
    def test_window_dataset_is_subject_partitioned(self):
        s=np.random.default_rng(3).normal(size=(2,300,8)).astype(np.float64); records=[core.ReferenceRecord("Wearable",1,250,s,"a"*64,"b"*64),core.ReferenceRecord("Wearable",5,250,s,"c"*64,"d"*64)]
        ds=core.ReferenceWindowDataset(records,[1],self.cfg); self.assertTrue(len(ds)>0); self.assertTrue(all(item[0]==0 for item in ds.entries)); sample=ds[0]; self.assertEqual(sample["noisy"].shape,(8,256)); self.assertEqual(sample["target"].shape,(8,256))
    def test_checkpoint_metadata_has_safety_flags(self):
        report={"reference_manifest_sha256":core.REFERENCE_MANIFEST_SHA256,"partition_lock_sha256":core.PARTITION_LOCK_SHA256}
        meta=core.checkpoint_metadata(self.cfg,report); self.assertFalse(meta["clean_ground_truth_used"]); self.assertFalse(meta["holdout_waveforms_loaded"]); self.assertFalse(meta["outer_validation_used_for_selection"])
    def test_checkpoint_metadata_rejects_wrong_binding(self):
        with self.assertRaises(core.SSVEPDenoiseV5Error): core.checkpoint_metadata(self.cfg,{"reference_manifest_sha256":"0"*64,"partition_lock_sha256":core.PARTITION_LOCK_SHA256})
    def test_loader_rejects_validation_and_holdout_subjects(self):
        # The original check never reached the subject gate when history was
        # absent. Bind an actual disposable manifest, not a historical hash.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest=root/core.REFERENCE_RELATIVE/"manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"fixture_scope":"synthetic_contract_only_not_historical_training_evidence",
                                            "materialized_subjects": []}))
            with patch.object(core,"REFERENCE_MANIFEST_SHA256",core.sha256(manifest)), \
                 patch.object(np,"load",side_effect=AssertionError("must not load waveforms")):
                for subjects in ([7],[9],[],[1,7]):
                    with self.subTest(subjects=subjects), self.assertRaisesRegex(core.SSVEPDenoiseV5Error,"only permits train"):
                        core.load_reference_records(root,subjects)
    def test_config_rejects_runtime_enablement(self):
        bad=dict(self.cfg); bad["runtime"]=dict(self.cfg["runtime"]); bad["runtime"]["enabled"]=True
        with self.assertRaises(core.SSVEPDenoiseV5Error): core.validate_config(bad)

if __name__=="__main__": unittest.main()
