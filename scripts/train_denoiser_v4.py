"""Frozen v4 training: CUDA, train/validation only, source-lock bound artifacts.

No test member is selected, inspected, scored or used for calibration. Whole-file
SHA256 identity verification necessarily reads the NPZ bytes, including bytes
belonging to the test members; this is not a claim of physical test-byte isolation.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eeg_tools import learned_dataset_v4 as dataset
from eeg_tools import learned_denoise_v4 as core

SEED = 20261007
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
WEIGHT_DECAY = 0.0001
DEFAULT_LOCK = "analysis_outputs/learned_denoiser_v4_training_source_lock.json"
SOURCE_FILES = ("eeg_tools/learned_denoise_v4.py", "eeg_tools/learned_dataset_v4.py",
                "scripts/train_denoiser_v4.py", "scripts/build_denoise_dataset_v4.py")


class TrainingV4Error(ValueError):
    """Frozen contract, integrity, device or serialization failed."""


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if type(value) is not dict or not core.primitive_metadata(value):
        raise TrainingV4Error("metadata must be finite Python JSON primitives")
    return value


def _root_path(root, path):
    result = (root / path).resolve()
    if not result.is_relative_to(root) or result == root:
        raise TrainingV4Error("artifact path must remain inside root")
    return result


def _relative(root, path):
    return path.resolve().relative_to(root).as_posix()


def verify_source_lock(root, lock_path, config_path, prereg_path, dataset_path, interface_path):
    """Verify EVERY entry, never parse a model metric or select a test NPZ member."""
    root = Path(root).resolve()
    before = dataset.sha256(lock_path)
    lock = _json(lock_path)
    if (type(lock.get("schema_version")) is not int or lock["schema_version"] != 1
            or lock.get("model_version") != core.MODEL_VERSION or lock.get("stage") != "training"):
        raise TrainingV4Error("training source_lock schema/version/stage mismatch")
    files = lock.get("files")
    if type(files) is not dict or not files:
        raise TrainingV4Error("training source_lock needs a nonempty files mapping")
    required = set(SOURCE_FILES) | {_relative(root,p) for p in
        (config_path, prereg_path, dataset_path, Path(str(dataset_path)+".manifest.json"), interface_path)}
    prereg = _json(prereg_path)
    for source in prereg.get("data",{}).get("sources",{}).values():
        required.add(_relative(root, _root_path(root,source["relative_path"])))
    if not required <= set(files):
        raise TrainingV4Error("source_lock missing required source/config/prereg/dataset bindings")
    for relative, expected in files.items():
        if type(relative) is not str or Path(relative).is_absolute() or "\\" in relative:
            raise TrainingV4Error("source_lock files must use repo-relative POSIX paths")
        dataset._hex_digest(expected,"source_lock SHA256")
        path = _root_path(root,relative)
        if not path.is_file() or dataset.sha256(path) != expected:
            raise TrainingV4Error(f"source_lock SHA256 mismatch: {relative}")
    # A different --root cannot substitute unrelated copies for executed code.
    for relative, path in ((SOURCE_FILES[0],Path(core.__file__)),
                           (SOURCE_FILES[1],Path(dataset.__file__)),
                           (SOURCE_FILES[2],Path(__file__))):
        if dataset.sha256(path) != files[relative]:
            raise TrainingV4Error(f"executed source differs from training source_lock: {relative}")
    for key,path in (("preregistration_sha256",prereg_path),("config_sha256",config_path),("dataset_sha256",dataset_path)):
        if lock.get(key) != dataset.sha256(path):
            raise TrainingV4Error(f"training source_lock {key} mismatch")
    if dataset.sha256(lock_path) != before:
        raise TrainingV4Error("source_lock changed during verification")
    return {"training_source_lock_sha256":before,"source_lock_file_count":int(len(files)),
            "files":dict(files),"preregistration_sha256":lock["preregistration_sha256"],
            "config_sha256":lock["config_sha256"],"dataset_sha256":lock["dataset_sha256"]}


def validate_training_contract(prereg,config):
    expected = dict(seed=SEED,epochs=EPOCHS,batch_size=BATCH_SIZE,optimizer="AdamW",
                    learning_rate=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    values=prereg.get("training",{})
    for key,value in expected.items():
        if type(values.get(key)) is not type(value) or values[key] != value:
            raise TrainingV4Error(f"preregistered training {key} differs from fixed v4 contract")
    core.validate_config(config)
    if config["loss"] != dict(time_weight=1.0,clean_identity_multiplier=5.0,artifact_weight=0.5):
        raise TrainingV4Error("frozen v4 loss mismatch")


def _torch():
    if core.torch is None:
        raise TrainingV4Error("PyTorch is required")
    return core.torch


def resolve_training_device(device):
    torch=_torch()
    selected=torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise TrainingV4Error("CUDA required for training/check-only; no silent CPU fallback")
    index=selected.index if selected.index is not None else torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise TrainingV4Error("CUDA device index unavailable")
    return f"cuda:{index}"


def combined_loss(prediction,target,logits,is_identity,config):
    torch=_torch()
    if (prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1:] != (1,500)
            or logits.shape != (len(prediction),) or is_identity.shape != logits.shape):
        raise TrainingV4Error("model/loss shape mismatch")
    if not bool(torch.isfinite(prediction).all() and torch.isfinite(target).all() and torch.isfinite(logits).all()):
        raise TrainingV4Error("nonfinite model/loss input")
    mse=(prediction-target).square().mean(dim=(1,2))
    weights=torch.where(is_identity,config["loss"]["clean_identity_multiplier"],1.0)
    time_loss=(mse*weights).mean()*config["loss"]["time_weight"]
    bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,(~is_identity).to(dtype=logits.dtype))
    return time_loss+config["loss"]["artifact_weight"]*bce,time_loss,bce


def run_epoch(model,values,config,device,optimizer=None,generator=None):
    torch=_torch()
    training=optimizer is not None
    model.train(training)
    n=len(values["normalized"])
    order=torch.randperm(n,generator=generator).numpy() if training else np.arange(n)
    totals=np.zeros(3,dtype=np.float64)
    scores=np.empty(n,dtype=np.float64) if not training else None
    squared={"all":[0.,0.,0],"identity":[0.,0.,0],"corrupted":[0.,0.,0]}
    for start in range(0,n,BATCH_SIZE):
        indices=order[start:start+BATCH_SIZE]
        x=torch.from_numpy(values["normalized"][indices]).to(device=device,dtype=torch.float32)
        y=torch.from_numpy(values["target"][indices]).to(device=device,dtype=torch.float32)
        identity=torch.from_numpy(values["is_identity"][indices]).to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction,logits=model(x)
            combined,timeloss,bce=combined_loss(prediction,y,logits,identity,config)
            if training:
                combined.backward()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise TrainingV4Error("nonfinite gradient; refusing training artifact")
                optimizer.step()
        totals += np.asarray([float(combined.detach().cpu()),float(timeloss.detach().cpu()),float(bce.detach().cpu())])*len(indices)
        if not training:
            pred=prediction.detach().cpu().numpy().astype(np.float64)
            probs=torch.sigmoid(logits.detach()).cpu().numpy().astype(np.float64)
            if not np.all(np.isfinite(probs)):raise TrainingV4Error("nonfinite validation score")
            scores[indices]=probs
            # Original float64 paired pure GT, including original DC, is never redefined.
            raw=values["noisy"][indices]
            candidate=raw+(pred[:,0]-values["normalized"][indices,0].astype(np.float64))*values["noisy_ac_rms"][indices,None]
            truth=values["clean"][indices]
            if not np.all(np.isfinite(candidate)):raise TrainingV4Error("nonfinite validation restoration")
            for key,mask in (("all",np.ones(len(indices),dtype=bool)),("identity",values["is_identity"][indices]),("corrupted",~values["is_identity"][indices])):
                squared[key][0]+=float(np.sum(np.square(raw[mask]-truth[mask]),dtype=np.float64))
                squared[key][1]+=float(np.sum(np.square(candidate[mask]-truth[mask]),dtype=np.float64))
                squared[key][2]+=int(mask.sum())*500
    result={"combined_loss":float(totals[0]/n),"weighted_time_mse":float(totals[1]/n),"artifact_bce":float(totals[2]/n),"windows":int(n)}
    if not training:
        result["candidate_rmse"]={key:{"noisy":float(math.sqrt(sums[0]/sums[2])),"candidate":float(math.sqrt(sums[1]/sums[2])),"windows":int(sums[2]//500)} for key,sums in squared.items()}
        result["candidate_scope"]="validation raw float64 original pure GT; ungated candidate, not guarded/gated inference"
    if not core.primitive_metadata(result):raise TrainingV4Error("nonfinite/nonprimitive epoch metrics")
    return result,scores


def calibrate_validation(scores,is_identity,config):
    scores=np.asarray(scores,dtype=np.float64)
    identity=np.asarray(is_identity)
    if identity.dtype != np.dtype(bool) or identity.shape != scores.shape or not identity.any() or not (~identity).any():
        raise TrainingV4Error("validation requires all matched clean/corrupted rows")
    clean=scores[identity]
    limit=float(config["inference"]["clean_false_replacement_max"])
    floor=float(config["inference"]["artifact_threshold_floor"])
    threshold=float(core.select_artifact_threshold(clean,false_replacement_max=limit,floor=floor))
    allowed=int(math.floor(len(clean)*limit))
    index=max(0,len(clean)-allowed-1)
    calibration={"scope":"validation_only","comparison":">","method":"clean_identity_order_statistic",
                 "clean_window_count":int(len(clean)),"observed_clean_false_replacement_fraction":float(np.mean(clean>threshold)),
                 "max_clean_false_replacement_fraction":limit,"floor":floor,
                 "strict_false_replacement_max":limit,"order_statistic":{"zero_based_index":int(index),"value":float(np.sort(clean)[index]),"allowed_exceedances":allowed},
                 "floor_already_safe":bool(np.mean(clean>floor)<=limit),
                 "empirical_only":True,"production_calibrated":False}
    rates={"clean_false_replacement_fraction":calibration["observed_clean_false_replacement_fraction"],
           "corrupted_replacement_fraction":float(np.mean(scores[~identity]>threshold)),
           "comparison":">","scope":"validation threshold-only decisions; raw safety guards not applied"}
    return threshold,calibration,rates


def tiny_probe(config,device):
    """Only generated data; float32 FFT500, no AMP, real forward/backward/update."""
    torch=_torch()
    model=core.build_model(config).to(device=device,dtype=torch.float32)
    t=torch.arange(500,device=device,dtype=torch.float32)/250
    x=(torch.sin(2*math.pi*12*t)+.6*torch.sin(2*math.pi*2*t))[None,None].repeat(2,1,1)
    target=x.clone();target[1]-=.6*torch.sin(2*math.pi*2*t)
    identity=torch.tensor([True,False],device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    prediction,logits=model(x)
    loss,*_=combined_loss(prediction,target,logits,identity,config)
    loss.backward()
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise TrainingV4Error("tiny probe nonfinite gradient")
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise TrainingV4Error("tiny probe nonfinite state")
    return {"forward_backward_update":True,"generated_windows":2,"float32":True,"amp":False,"device":device,"gpu_used":str(device).startswith("cuda")}


def publish_artifacts(output,report,checkpoint,training_report):
    """Exclusive atomic files; failures roll back only links created by this call."""
    if output == report or output.exists() or report.exists():raise TrainingV4Error("output/report already exists; refuse overwrite")
    if not core.primitive_metadata(training_report):raise TrainingV4Error("training report not finite primitive metadata")
    core.validate_checkpoint_metadata(checkpoint,checkpoint["config"])
    staged=[];published=[]
    try:
        for target in (output,report):
            target.parent.mkdir(parents=True,exist_ok=True)
            fd,name=tempfile.mkstemp(prefix=target.name+".",suffix=".pending",dir=target.parent)
            os.close(fd);staged.append((Path(name),target))
        _torch().save(checkpoint,staged[0][0])
        # Loadability is checked before publication, not assumed from torch.save.
        safe=_torch().load(staged[0][0],map_location="cpu",weights_only=True)
        core.validate_checkpoint_metadata(safe,checkpoint["config"])
        core.load_model(staged[0][0],checkpoint["config"],device="cpu")
        training_report=deepcopy(training_report)
        training_report["checkpoint_sha256"]=dataset.sha256(staged[0][0])
        staged[1][0].write_text(json.dumps(training_report,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
        for temporary,target in staged:
            os.link(temporary,target);published.append((temporary,target))
    except Exception:
        for temporary,target in published:
            if target.exists() and temporary.samefile(target):target.unlink()
        raise
    finally:
        for temporary,_ in staged:temporary.unlink(missing_ok=True)


def make_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--config",type=Path,default=Path("configs/learned_denoiser_v4.json"))
    parser.add_argument("--dataset",type=Path,default=Path(dataset.DEFAULT_DATASET_RELATIVE))
    parser.add_argument("--preregistration",type=Path,default=Path(dataset.DEFAULT_PREREGISTRATION_RELATIVE))
    parser.add_argument("--interface-contract",type=Path,default=Path(dataset.DEFAULT_INTERFACE_RELATIVE))
    parser.add_argument("--source-lock",type=Path,default=Path(DEFAULT_LOCK))
    parser.add_argument("--output",type=Path,default=Path("models/denoiser_v4.pt"))
    parser.add_argument("--report",type=Path)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--check-only",action="store_true")
    return parser


def train(args):
    started=time.perf_counter();torch=_torch()
    root=args.root.resolve()
    cfg_path,prereg_path,data_path,interface_path,lock_path,output=(_root_path(root,p) for p in
        (args.config,args.preregistration,args.dataset,args.interface_contract,args.source_lock,args.output))
    report=_root_path(root,args.report) if args.report is not None else output.with_suffix(".training.json")
    if not args.check_only and (output.exists() or report.exists() or output==report):raise TrainingV4Error("output/report already exists; refuse overwrite")
    if type(args.threads) is not int or args.threads < 1:raise TrainingV4Error("threads must be positive")
    device=resolve_training_device(args.device)
    os.environ["OMP_NUM_THREADS"]=str(args.threads);os.environ["MKL_NUM_THREADS"]=str(args.threads)
    torch.set_num_threads(args.threads)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(SEED);torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.cuda.reset_peak_memory_stats(device)
    config=core.load_config(cfg_path);prereg=_json(prereg_path)
    validate_training_contract(prereg,config)
    bindings=verify_source_lock(root,lock_path,cfg_path,prereg_path,data_path,interface_path)
    train_values=dataset.load_split(data_path,"train")
    val_values=dataset.load_split(data_path,"validation")
    manifest=train_values["manifest"]
    if (val_values["dataset_sha256"] != bindings["dataset_sha256"] or train_values["dataset_sha256"] != bindings["dataset_sha256"]
            or manifest["preregistration_sha256"] != bindings["preregistration_sha256"]
            or manifest["interface_contract_sha256"] != dataset.sha256(interface_path)):
        raise TrainingV4Error("dataset/preregistration/interface binding mismatch")
    for key in ("train","validation","test"):
        if manifest["group_partitions"][key] != prereg["data"][f"{key}_groups"]:raise TrainingV4Error("preregistered groups differ from dataset")
    if manifest["group_assignments"] != prereg["data"]["group_assignments"]:raise TrainingV4Error("preregistered assignments differ")
    for role,source in prereg["data"]["sources"].items():
        if manifest["source_lock"][role]["sha256"] != source["sha256"]:raise TrainingV4Error("raw source binding mismatch")
    if set(train_values["group_id"]) & set(val_values["group_id"]):raise TrainingV4Error("train/validation overlap")
    verify_source_lock(root,lock_path,cfg_path,prereg_path,data_path,interface_path)
    env={"python":sys.version,"torch":str(torch.__version__),"numpy":str(np.__version__),"device":device,
         "cuda_runtime":str(torch.version.cuda) if torch.version.cuda is not None else None,
         "gpu_used":device.startswith("cuda"),"gpu_name":torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
         "threads":int(args.threads),"amp":False,"fft_dtype":"float32"}
    scope={"selected_splits":["train","validation"],"test_values_accessed":False,
           "test_access_definition":"No test member selected/inspected or used for training, metrics, best selection or threshold calibration; whole-file hashing reads all NPZ bytes for identity.",
           "test_model_metrics_computed":False,"subject_disjoint_claim":False,
           "production_validated":False,"automatic_processing_enabled":False}
    if args.check_only:
        probe=tiny_probe(config,device)
        end=verify_source_lock(root,lock_path,cfg_path,prereg_path,data_path,interface_path)
        if end != bindings:raise TrainingV4Error("source_lock binding changed")
        return {"status":"integrity_ready","bindings":bindings,"environment":env,"scope":scope,"probe":probe,
                "train_windows":int(len(train_values["normalized"])),"validation_windows":int(len(val_values["normalized"])),"published":False}
    model=core.build_model(config).to(device=device,dtype=torch.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    generator=torch.Generator(device="cpu").manual_seed(SEED)
    history=[];best_loss=math.inf;best_state=None;best_epoch=0;best_scores=None;best_validation=None
    for epoch in range(1,EPOCHS+1):
        epoch_start=time.perf_counter()
        training,_=run_epoch(model,train_values,config,device,optimizer,generator)
        validation,scores=run_epoch(model,val_values,config,device)
        threshold,calibration,rates=calibrate_validation(scores,val_values["is_identity"],config)
        entry={"epoch":int(epoch),"train":training,"validation":validation,"artifact_threshold":threshold,
               "validation_replacement":rates,"elapsed_seconds":float(time.perf_counter()-epoch_start)}
        history.append(entry)
        if validation["combined_loss"] < best_loss: # strictly less: first exact tie wins.
            best_loss=float(validation["combined_loss"]);best_epoch=int(epoch)
            best_state={k:v.detach().to("cpu").clone() for k,v in model.state_dict().items()}
            best_scores=scores.copy();best_validation=deepcopy(validation)
        print(json.dumps(entry,ensure_ascii=True,allow_nan=False),flush=True)
    if best_state is None or any(not bool(torch.isfinite(v).all()) for v in best_state.values()):raise TrainingV4Error("best state missing/nonfinite")
    threshold,calibration,rates=calibrate_validation(best_scores,val_values["is_identity"],config)
    final_bindings=verify_source_lock(root,lock_path,cfg_path,prereg_path,data_path,interface_path)
    if final_bindings != bindings:raise TrainingV4Error("training source_lock binding changed")
    metadata={"seed":SEED,"epochs":EPOCHS,"batch_size":BATCH_SIZE,"optimizer":"AdamW","learning_rate":LEARNING_RATE,
              "weight_decay":WEIGHT_DECAY,"best_epoch":best_epoch,"best_validation":best_validation,
              "selection":"minimum validation combined loss; first epoch on exact tie", "bindings":bindings,
              "dataset_manifest_sha256":train_values["manifest_sha256"],"source_sha256":{k:v["sha256"] for k,v in prereg["data"]["sources"].items()},
              "groups":{"train":prereg["data"]["train_groups"],"validation":prereg["data"]["validation_groups"]},
              "train_windows":int(len(train_values["normalized"])),"validation_windows":int(len(val_values["normalized"])),
              "environment":env,"scope":scope}
    checkpoint={"checkpoint_schema_version":1,"model_version":core.MODEL_VERSION,"config":deepcopy(config),
                "model_state":best_state,"artifact_threshold":threshold,"threshold_calibration":calibration,
                "dataset_sha256":bindings["dataset_sha256"],"metadata":metadata}
    core.validate_checkpoint_metadata(checkpoint,config)
    training_report={"schema_version":1,"model_version":core.MODEL_VERSION,"status":"completed_offline_training_not_production_validation",
                     "created_at_utc":datetime.now(timezone.utc).isoformat(),"metadata":metadata,"history":history,
                     "best_epoch":best_epoch,"artifact_threshold":threshold,"threshold_calibration":calibration,"best_validation_replacement":rates,
                     "elapsed_seconds":float(time.perf_counter()-started),
                     "gpu_peak_allocated_bytes":int(torch.cuda.max_memory_allocated(device)) if device.startswith("cuda") else 0,
                     "gpu_peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0,
                     "checkpoint":str(output),"report":str(report),**scope}
    publish_artifacts(output,report,checkpoint,training_report)
    return {"status":"completed_offline_training_not_production_validation","output":str(output),"report":str(report),
            "checkpoint_sha256":dataset.sha256(output),"best_epoch":best_epoch,"artifact_threshold":threshold,
            "training_source_lock_sha256":bindings["training_source_lock_sha256"],**scope}


def main(argv=None):
    args=make_parser().parse_args(argv)
    try:
        result=train(args)
        print(json.dumps(result,ensure_ascii=True,allow_nan=False,indent=2))
        return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as error:
        print(json.dumps({"status":"error","error":str(error)},ensure_ascii=True,allow_nan=False),flush=True)
        return 2

if __name__ == "__main__":raise SystemExit(main())
