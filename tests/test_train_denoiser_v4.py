"""Synthetic/mocked v4 training checks. No research training or holdout metrics."""
from __future__ import annotations
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import numpy as np
from eeg_tools import learned_dataset_v4 as data
from eeg_tools import learned_denoise_v4 as core
from scripts import train_denoiser_v4 as trainer

ROOT=Path(__file__).resolve().parents[1]
_spec=importlib.util.spec_from_file_location("synthetic_dataset_v4_test_helpers",ROOT/"tests/test_learned_dataset_v4.py")
_helpers=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_helpers)


def locked_fixture(root):
    prereg,_=_helpers.fixture(root)
    data.build_dataset(root)
    config_path=root/"configs/learned_denoiser_v4.json"
    config_path.parent.mkdir(parents=True,exist_ok=True)
    config_path.write_text(json.dumps(core.load_config(ROOT/"configs/learned_denoiser_v4.json")),encoding="utf-8")
    paths=list(trainer.SOURCE_FILES)+["configs/learned_denoiser_v4.json",data.DEFAULT_PREREGISTRATION_RELATIVE,
        data.DEFAULT_INTERFACE_RELATIVE,data.DEFAULT_DATASET_RELATIVE,data.DEFAULT_DATASET_RELATIVE+".manifest.json"]
    paths += [s["relative_path"] for s in prereg["data"]["sources"].values()]
    for relative in trainer.SOURCE_FILES:
        dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,dest)
    # Extra listed files must ALL be verified, not merely the required subset.
    (root/"extra_locked.txt").write_text("audited",encoding="utf-8");paths.append("extra_locked.txt")
    lock={"schema_version":1,"model_version":core.MODEL_VERSION,"stage":"training",
          "files":{p:data.sha256(root/p) for p in paths},
          "preregistration_sha256":data.sha256(root/data.DEFAULT_PREREGISTRATION_RELATIVE),
          "config_sha256":data.sha256(config_path),"dataset_sha256":data.sha256(root/data.DEFAULT_DATASET_RELATIVE)}
    lock_path=root/trainer.DEFAULT_LOCK
    lock_path.write_text(json.dumps(lock),encoding="utf-8")
    return trainer.make_parser().parse_args(["--root",str(root)]),lock


class TrainingV4Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.config=core.load_config(ROOT/"configs/learned_denoiser_v4.json")
        self.torch=core.torch
        if self.torch is None:self.skipTest("optional torch absent")
        self.torch.set_num_threads(2)

    def verify(self,args):
        return trainer.verify_source_lock(self.root,self.root/args.source_lock,self.root/args.config,
                self.root/args.preregistration,self.root/args.dataset,self.root/args.interface_contract)

    def test_cli_fixed_hyperparameters_no_tuning_flags(self):
        args=trainer.make_parser().parse_args([])
        self.assertEqual(args.device,"cuda");self.assertEqual(args.threads,2)
        self.assertEqual((trainer.SEED,trainer.EPOCHS,trainer.BATCH_SIZE),(20261007,30,128))
        with redirect_stdout(io.StringIO()),patch("sys.stderr",new=io.StringIO()):
            with self.assertRaises(SystemExit):trainer.make_parser().parse_args(["--epochs","31"])
        with patch.object(self.torch.cuda,"is_available",return_value=False):
            for device in ("cuda","cpu"):
                with self.assertRaisesRegex(trainer.TrainingV4Error,"CUDA required"):trainer.resolve_training_device(device)

    def test_cuda_unavailable_refuses_before_loading_data_or_lock(self):
        args=trainer.make_parser().parse_args(["--root",str(self.root),"--check-only"])
        with patch.object(self.torch.cuda,"is_available",return_value=False),patch.object(data,"load_split") as loader:
            with self.assertRaisesRegex(trainer.TrainingV4Error,"CUDA required"):trainer.train(args)
            loader.assert_not_called()

    def test_training_contract_changes_rejected(self):
        prereg=json.loads((ROOT/data.DEFAULT_PREREGISTRATION_RELATIVE).read_text(encoding="utf-8"))
        trainer.validate_training_contract(prereg,self.config)
        for key,value in (("epochs",31),("seed",10),("batch_size",64),("optimizer","Adam"),("learning_rate",.001),("weight_decay",.01)):
            changed=deepcopy(prereg);changed["training"][key]=value
            with self.assertRaisesRegex(trainer.TrainingV4Error,key):trainer.validate_training_contract(changed,self.config)

    def test_loss_is_full_window_mean_identity_five_and_half_bce(self):
        torch=self.torch
        prediction=torch.ones(2,1,500,requires_grad=True)
        target=torch.zeros_like(prediction)
        logits=torch.zeros(2,requires_grad=True)
        identity=torch.tensor([True,False])
        loss,mse,bce=trainer.combined_loss(prediction,target,logits,identity,self.config)
        self.assertAlmostEqual(float(mse.detach()),3.)
        self.assertAlmostEqual(float(bce.detach()),float(np.log(2)),places=6)
        self.assertAlmostEqual(float(loss.detach()),3+.5*float(np.log(2)),places=6)
        loss.backward();self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertLess(float(logits.grad[1]),0);self.assertGreater(float(logits.grad[0]),0)

    def test_loss_shape_and_nonfinite_refused(self):
        torch=self.torch;x=torch.zeros(2,1,500);labels=torch.tensor([True,False])
        with self.assertRaisesRegex(trainer.TrainingV4Error,"shape"):trainer.combined_loss(x,x,torch.zeros(2,1),labels,self.config)
        x[0,0,0]=float("nan")
        with self.assertRaisesRegex(trainer.TrainingV4Error,"nonfinite"):trainer.combined_loss(x,x,torch.zeros(2),labels,self.config)

    def test_calibration_matches_core_all_clean_strict_order_statistic_and_floor(self):
        identity=np.array([True]*100+[False]*100)
        for clean in (np.full(100,.2),np.linspace(.1,.99,100),np.ones(100),np.r_[np.zeros(99),1.]):
            scores=np.r_[clean,np.full(100,.95)]
            threshold,cal,rates=trainer.calibrate_validation(scores,identity,self.config)
            self.assertEqual(threshold,core.select_artifact_threshold(clean))
            self.assertLessEqual(np.mean(clean>threshold),.01)
            self.assertEqual(cal["comparison"],">")
            self.assertEqual(cal["clean_window_count"],100)
            self.assertEqual(cal["floor"],.5)
            self.assertIs(type(cal["observed_clean_false_replacement_fraction"]),float)
            self.assertIs(type(cal["order_statistic"]["zero_based_index"]),int)
            self.assertTrue(core.primitive_metadata(cal))
            self.assertEqual(rates["corrupted_replacement_fraction"],float(np.mean(scores[~identity]>threshold)))
        self.assertEqual(trainer.calibrate_validation(np.ones(200),identity,self.config)[1]["observed_clean_false_replacement_fraction"],0.)

    def test_source_lock_required_and_every_entry_verified(self):
        args,lock=locked_fixture(self.root)
        bindings=self.verify(args)
        self.assertEqual(bindings["training_source_lock_sha256"],data.sha256(self.root/trainer.DEFAULT_LOCK))
        (self.root/"extra_locked.txt").write_text("changed",encoding="utf-8")
        with self.assertRaisesRegex(trainer.TrainingV4Error,"extra_locked"):self.verify(args)

    def test_source_lock_binding_stage_missing_path_and_executed_source(self):
        args,original=locked_fixture(self.root)
        for mode in ("stage","missing","binding","escape","executed"):
            lock=deepcopy(original)
            with self.subTest(mode=mode):
                if mode=="stage":lock["stage"]="evaluation"
                elif mode=="missing":del lock["files"]["scripts/train_denoiser_v4.py"]
                elif mode=="binding":lock["dataset_sha256"]="0"*64
                elif mode=="escape":lock["files"]["../outside.txt"]="0"*64
                else:
                    path=self.root/"scripts/train_denoiser_v4.py"
                    path.write_text("# substituted source",encoding="utf-8")
                    lock["files"]["scripts/train_denoiser_v4.py"]=data.sha256(path)
                (self.root/trainer.DEFAULT_LOCK).write_text(json.dumps(lock),encoding="utf-8")
                with self.assertRaises((trainer.TrainingV4Error,OSError)):self.verify(args)

    def test_real_synthetic_cpu_forward_backward_and_safe_checkpoint_loading(self):
        args,_=locked_fixture(self.root)
        values=data.load_split(self.root/args.dataset,"train")
        model=core.build_model(self.config)
        optimizer=self.torch.optim.AdamW(model.parameters(),lr=trainer.LEARNING_RATE,weight_decay=trainer.WEIGHT_DECAY)
        before={k:v.clone() for k,v in model.state_dict().items()}
        train_metrics,_=trainer.run_epoch(model,values,self.config,"cpu",optimizer,self.torch.Generator().manual_seed(trainer.SEED))
        validation,scores=trainer.run_epoch(model,values,self.config,"cpu")
        threshold,cal,_=trainer.calibrate_validation(scores,values["is_identity"],self.config)
        self.assertTrue(any(not self.torch.equal(before[k],v) for k,v in model.state_dict().items()))
        self.assertTrue(core.primitive_metadata(validation));self.assertTrue(core.primitive_metadata(train_metrics))
        checkpoint={"checkpoint_schema_version":1,"model_version":core.MODEL_VERSION,"config":self.config,
            "model_state":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            "artifact_threshold":threshold,"threshold_calibration":cal,"dataset_sha256":data.sha256(self.root/args.dataset),
            "metadata":{"synthetic_fixture_only":True,"validation":validation,"bindings":self.verify(args)}}
        output=self.root/"synthetic.pt";report=self.root/"synthetic.training.json"
        trainer.publish_artifacts(output,report,checkpoint,{"validation":validation,"test_values_accessed":False})
        bundle=core.load_model(output,self.config,"cpu")
        self.assertEqual(bundle.artifact_threshold,threshold)
        saved=self.torch.load(output,weights_only=True,map_location="cpu")
        self.assertEqual(set(saved),set(core.CHECKPOINT_KEYS))
        self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["checkpoint_sha256"],data.sha256(output))
        with self.assertRaisesRegex(trainer.TrainingV4Error,"overwrite"):trainer.publish_artifacts(output,report,checkpoint,{})

    def test_numpy_scalars_not_accepted_serialization(self):
        args,_=locked_fixture(self.root)
        state=core.build_model(self.config).state_dict()
        threshold,cal,_=trainer.calibrate_validation(np.array([.2,.8]),np.array([True,False]),self.config)
        checkpoint={"checkpoint_schema_version":1,"model_version":core.MODEL_VERSION,"config":self.config,"model_state":state,
            "artifact_threshold":threshold,"threshold_calibration":cal,"dataset_sha256":"a"*64,"metadata":{"bad":np.float64(.2)}}
        with self.assertRaisesRegex(core.LearnedDenoiseV4Error,"primitives"):core.validate_checkpoint_metadata(checkpoint,self.config)
        with self.assertRaisesRegex(trainer.TrainingV4Error,"primitive"):
            trainer.publish_artifacts(self.root/"bad.pt",self.root/"bad.json",checkpoint,{"bad":np.int64(1)})

    def test_check_only_real_tiny_probe_no_metrics_no_publication_no_test_member(self):
        args,_=locked_fixture(self.root);args.check_only=True
        with np.load(self.root/args.dataset) as archive:klass=type(archive)
        original=klass.__getitem__;seen=[]
        def guarded(archive,key):
            seen.append(key)
            if key.startswith("test_"):raise AssertionError("test selected")
            return original(archive,key)
        with patch.object(trainer,"resolve_training_device",return_value="cpu"),patch.object(klass,"__getitem__",guarded),patch.object(trainer,"run_epoch",side_effect=AssertionError("dataset model metric in preflight")):
            result=trainer.train(args)
        self.assertEqual(result["status"],"integrity_ready")
        self.assertFalse(result["environment"]["gpu_used"])
        self.assertTrue(result["probe"]["forward_backward_update"])
        self.assertFalse(result["scope"]["test_values_accessed"])
        self.assertFalse((self.root/args.output).exists())
        self.assertTrue(all(not k.startswith("test_") for k in seen))

    def test_mocked_full_30_epochs_best_first_tie_and_all_history_serializable(self):
        args,_=locked_fixture(self.root)
        real_epoch=trainer.run_epoch
        model=core.build_model(self.config)
        vals=data.load_split(self.root/args.dataset,"validation")
        valmetrics,scores=real_epoch(model,vals,self.config,"cpu")
        calls=[]
        def mocked_epoch(model,values,config,device,optimizer=None,generator=None):
            calls.append(values["split"])
            if optimizer is not None:return {"combined_loss":1.,"weighted_time_mse":.1,"artifact_bce":.6,"windows":len(values["normalized"])},None
            count=calls.count("validation")
            metrics=deepcopy(valmetrics);metrics["combined_loss"]=2. if count==1 else 1.
            return metrics,scores.copy()
        with patch.object(trainer,"resolve_training_device",return_value="cpu"),patch.object(trainer,"run_epoch",side_effect=mocked_epoch),redirect_stdout(io.StringIO()):
            result=trainer.train(args)
        self.assertEqual(result["best_epoch"],2)
        self.assertEqual(calls,["train","validation"]*30)
        report=json.loads((self.root/"models/denoiser_v4.training.json").read_text(encoding="utf-8"))
        self.assertEqual(len(report["history"]),30)
        self.assertTrue(core.primitive_metadata(report))
        self.assertFalse(report["production_validated"])
        self.assertFalse(report["test_values_accessed"])
        bundle=core.load_model(self.root/args.output,self.config,"cpu")
        self.assertEqual(bundle.metadata["metadata"]["best_epoch"],2)

    def test_source_mutation_during_preflight_rejected_no_output(self):
        args,_=locked_fixture(self.root);args.check_only=True
        def mutation(config,device):
            (self.root/"extra_locked.txt").write_text("changed",encoding="utf-8")
            return {}
        with patch.object(trainer,"resolve_training_device",return_value="cpu"),patch.object(trainer,"tiny_probe",side_effect=mutation):
            with self.assertRaisesRegex(trainer.TrainingV4Error,"SHA256"):trainer.train(args)
        self.assertFalse((self.root/args.output).exists())

    def test_bad_model_state_is_not_published(self):
        threshold,cal,_=trainer.calibrate_validation(np.array([.2,.8]),np.array([True,False]),self.config)
        checkpoint={"checkpoint_schema_version":1,"model_version":core.MODEL_VERSION,"config":self.config,
            "model_state":{},"artifact_threshold":threshold,"threshold_calibration":cal,
            "dataset_sha256":"a"*64,"metadata":{}}
        with self.assertRaises(core.LearnedDenoiseV4Error):
            trainer.publish_artifacts(self.root/"bad.pt",self.root/"bad.json",checkpoint,{})
        self.assertFalse((self.root/"bad.pt").exists());self.assertFalse((self.root/"bad.json").exists())
        self.assertEqual(list(self.root.glob("*.pending")),[])

    def test_atomic_publication_second_link_failure_cleans_up(self):
        import os
        threshold,cal,_=trainer.calibrate_validation(np.array([.2,.8]),np.array([True,False]),self.config)
        checkpoint={"checkpoint_schema_version":1,"model_version":core.MODEL_VERSION,"config":self.config,
            "model_state":core.build_model(self.config).state_dict(),"artifact_threshold":threshold,"threshold_calibration":cal,
            "dataset_sha256":"a"*64,"metadata":{}}
        original=os.link;calls=[]
        def fail(src,dst):
            calls.append(dst)
            if len(calls)==2:raise OSError("synthetic link failure")
            return original(src,dst)
        with patch("os.link",side_effect=fail):
            with self.assertRaisesRegex(OSError,"synthetic"):trainer.publish_artifacts(self.root/"out.pt",self.root/"out.json",checkpoint,{})
        self.assertFalse((self.root/"out.pt").exists());self.assertFalse((self.root/"out.json").exists())
        self.assertEqual(list(self.root.glob("*.pending")),[])

if __name__ == "__main__":unittest.main()
