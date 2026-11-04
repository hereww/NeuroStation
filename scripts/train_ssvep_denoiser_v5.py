"""Train the frozen v5 SSVEP reference-reconstruction candidate on GPU only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json, os, random, sys, tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from eeg_tools import ssvep_denoise_v5 as core

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    core.torch.manual_seed(seed); core.torch.cuda.manual_seed_all(seed)

def _jsonable(v):
    if isinstance(v,dict): return {str(k):_jsonable(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [_jsonable(x) for x in v]
    if isinstance(v,np.generic): return v.item()
    if isinstance(v,float):
        if not np.isfinite(v): raise ValueError("nonfinite metadata")
        return float(v)
    return v

def _batch(batch,device):
    torch=core.torch
    noisy=torch.as_tensor(batch["noisy"],device=device,dtype=torch.float32)
    target=torch.as_tensor(batch["target"],device=device,dtype=torch.float32)
    identity=torch.as_tensor(batch["identity"],device=device,dtype=torch.bool)
    return noisy,target,identity

def run_epoch(model,loader,config,device,optimizer=None,epoch=0):
    training=optimizer is not None; model.train(training); loader.dataset.set_epoch(epoch); sums={k:0.0 for k in ("total","time","band","identity","change")}; count=0
    for batch in loader:
        noisy,target,identity=_batch(batch,device)
        if training: optimizer.zero_grad(set_to_none=True)
        pred=model(noisy); fs=int(batch["sample_rate_hz"][0]); losses=core.compute_loss(pred,target,noisy,identity,fs,config)
        if training:
            losses["total"].backward(); core.torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step()
        n=int(noisy.shape[0]); count+=n
        for key in sums: sums[key]+=float(losses[key].detach().cpu())*n
    if count<=0: raise core.SSVEPDenoiseV5Error("empty epoch")
    return {k:sums[k]/count for k in sums}

def train_domain(records,domain,config,device):
    torch=core.torch; tr=[r for r in records if r.dataset==domain]
    fit=core.ReferenceWindowDataset(tr,config["training"]["fit_subjects"],config); internal=core.ReferenceWindowDataset(tr,config["training"]["internal_validation_subjects"],config)
    train_loader=torch.utils.data.DataLoader(fit,batch_size=config["training"]["batch_size"],shuffle=True,num_workers=0,drop_last=False)
    val_loader=torch.utils.data.DataLoader(internal,batch_size=config["training"]["batch_size"],shuffle=False,num_workers=0,drop_last=False)
    model=core.build_model(config).to(device=device,dtype=torch.float32); opt=torch.optim.AdamW(model.parameters(),lr=config["training"]["learning_rate"],weight_decay=config["training"]["weight_decay"]); history=[]; best_score=None; best_epoch=None; best_state=None
    for epoch in range(config["training"]["epochs"]):
        train=run_epoch(model,train_loader,config,device,opt,epoch); val=run_epoch(model,val_loader,config,device,None,0); row={"epoch":epoch+1,"train":train,"internal_validation":val}; history.append(row)
        if best_score is None or val["total"]<best_score:
            best_score=float(val["total"]); best_epoch=epoch+1; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    return best_state,{"dataset":domain,"fit_subjects":config["training"]["fit_subjects"],"internal_validation_subjects":config["training"]["internal_validation_subjects"],"fit_windows":len(fit),"internal_validation_windows":len(internal),"best_epoch":best_epoch,"best_internal_validation_total_loss":best_score,"history":history}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--reference-dir",type=Path,default=core.REFERENCE_RELATIVE); p.add_argument("--config",type=Path,default=core.CONFIG_RELATIVE); p.add_argument("--output-checkpoint",type=Path,required=True); p.add_argument("--output-report",type=Path,required=True); p.add_argument("--device",choices=("cuda","cuda:0"),default="cuda"); args=p.parse_args(argv)
    if core.torch is None: raise SystemExit("PyTorch unavailable")
    cfg=core.load_config(args.config,root=ROOT)
    if args.output_checkpoint.exists() or args.output_report.exists(): raise SystemExit("refuse overwrite existing training artifact")
    if not core.torch.cuda.is_available(): raise SystemExit("CUDA required; CPU fallback refused")
    set_seed(cfg["training"]["seed"]); core.torch.cuda.set_device(0); device=args.device
    # The loader permits only train subjects 1..6 and therefore cannot open outer validation/holdout.
    subjects=tuple(cfg["training"]["fit_subjects"]+cfg["training"]["internal_validation_subjects"]); records=core.load_reference_records(ROOT,subjects=subjects)
    bindings=[{"dataset":r.dataset,"subject_id":r.subject_id,"npz_sha256":r.npz_sha256,"sidecar_sha256":r.sidecar_sha256} for r in records]; states={}; reports={}
    for domain in ("Nakanishi2015","Wearable"): states[domain],reports[domain]=train_domain(records,domain,cfg,device)
    for r in records:
        path=ROOT/core.REFERENCE_RELATIVE/r.dataset/f"subject_{r.subject_id:03d}.npz"
        if core.sha256(path)!=r.npz_sha256: raise SystemExit(f"source changed during training: {path}")
    report={"schema_version":1,"status":"formal","model_version":core.MODEL_VERSION,"created_at_utc":datetime.now(timezone.utc).isoformat(),"device":device,"cuda_device_name":core.torch.cuda.get_device_name(0),"reference_manifest_sha256":core.REFERENCE_MANIFEST_SHA256,"partition_lock_sha256":core.PARTITION_LOCK_SHA256,"source_binding":bindings,"domain_reports":reports,"target_definition":"recorded_reference_reconstruction; reference is not clean ground truth","holdout_waveforms_loaded":False,"outer_validation_used_for_selection":False,"validation_task_metrics_computed":False,"runtime_enabled":False,"production_validated":False,"amp":False}
    checkpoint=core.checkpoint_metadata(cfg,report); checkpoint["models"]=states; args.output_checkpoint.parent.mkdir(parents=True,exist_ok=True); args.output_report.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=args.output_checkpoint.name+".",suffix=".pending",dir=args.output_checkpoint.parent); os.close(fd); Path(tmp).unlink(missing_ok=True)
    try:
        core.torch.save(checkpoint,tmp); loaded=core.torch.load(tmp,map_location="cpu",weights_only=True)
        if loaded.get("model_version")!=core.MODEL_VERSION or set(loaded.get("models",{}))!={"Nakanishi2015","Wearable"}: raise SystemExit("checkpoint self-check failed")
        Path(tmp).replace(args.output_checkpoint); args.output_report.write_text(json.dumps(_jsonable(report),ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    finally: Path(tmp).unlink(missing_ok=True)
    print(json.dumps({"status":"completed","checkpoint":str(args.output_checkpoint),"report":str(args.output_report),"device":device,"holdout_loaded":False,"outer_validation_used_for_selection":False,"runtime_enabled":False},ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())

