"""Train an offline Wang2016 spectral MLP; final test is a separate one-shot CLI."""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys
from datetime import datetime, timezone
from pathlib import Path
if __package__ in (None,""): sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from scripts import benchmark_matched_public_ssvep as baseline
from eeg_tools import matched_ssvep_decoder as decoder
ROOT=Path(__file__).resolve().parents[1]
SPLITS={"train":list(range(1,25)),"validation":list(range(25,30)),"test":list(range(30,36))}
SEED=20261015
SELECTION_RULE="max validation raw accuracy, min validation cross_entropy, earliest epoch"
SCOPE="Wang2016 offline cross-subject three-frequency task decoding; no personal sessions; no denoising claim"
BASELINE_SPEC={"algorithms":["CCA","FBCCA"],"harmonics":3,"subbands":[list(b) for b in baseline.SUBBANDS],"weights":list(baseline.WEIGHTS),"score_thresholds":dict(baseline.SCORE_THRESHOLDS),"margin_threshold":0.05,"selection":"fixed; no fit and no tuning on test"}
def require(ok,msg): decoder.require(ok,msg)
def digest(v): return hashlib.sha256(baseline.canonical(v).encode()).hexdigest()
def write_json_new(path,value):
    with Path(path).open("x",encoding="utf-8") as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.write("\n")
def code_hashes():
    names=("scripts/train_matched_ssvep_decoder.py","eeg_tools/matched_ssvep_decoder.py","scripts/benchmark_matched_public_ssvep.py","scripts/benchmark_open_ssvep_commands.py")
    return {str(ROOT/n):baseline.sha256_file(ROOT/n) for n in names}
def verify_hashes(hashes):
    for name,expected in hashes.items(): require(Path(name).is_file() and baseline.sha256_file(Path(name))==expected,f"source/hash changed: {name}")
def select_device(name="cuda",require_cuda=False):
    require(name in ("cuda","cpu"),"device must be cuda or cpu")
    require(not require_cuda or name=="cuda","require_cuda forbids CPU execution")
    if name=="cuda": require(torch.cuda.is_available(),"CUDA unavailable; CPU fallback is forbidden")
    return torch.device(name)
def check_spec(spec):
    require(spec.get("schema_version")==1 and spec.get("status")=="formal","formal spec schema 1 required")
    axes,shape=spec.get("axes",{}),spec.get("expected_shape",[])
    require(set(axes)=={"channel","sample","condition","block"} and sorted(axes.values())==[0,1,2,3],"explicit verified axes required")
    require([shape[axes[k]] for k in ("channel","sample","condition","block")] == [64,1500,40,6],"verified Wang shape required")
    require(spec.get("sample_rate_hz")==250 and spec.get("sample_window")=={"start":125,"stop":1375},"requires verified 250Hz [125:1375]")
    require(bool(spec.get("metadata_authority")),"metadata authority missing")
    names,mapping=spec.get("source_channel_names",[]),spec.get("target_channel_map",{})
    require(len(names)==64 and len(set(names))==64 and set(mapping)==set(baseline.CHANNELS) and len(set(mapping.values()))==8 and all(v in names for v in mapping.values()),"verified eight-channel mapping required")
    frequencies=spec.get("frequency_order_hz",[])
    require(len(frequencies)==40 and all(type(f) in (int,float) and np.isfinite(f) for f in frequencies),"complete source frequency order required")
    require(all(frequencies.count(f)==1 for f in decoder.FREQUENCIES) and 20 not in frequencies,"three-frequency-only 10/12/15 spec required")
    require(isinstance(spec.get("mat_variable"),str) and spec["mat_variable"],"MAT variable required")
    baseline.identity(spec.get("depth")); baseline.identity(spec.get("session"))
def make_preregistration(mat_root,source_spec,*,epochs=100,device="cuda",require_cuda=False):
    require(type(epochs) is int and 1<=epochs<=500,"epochs must be 1..500")
    mat_root,source_spec=Path(mat_root).resolve(),Path(source_spec).resolve(); spec=baseline.read_json(source_spec); check_spec(spec)
    files={s:mat_root/f"S{s}.mat" for s in range(1,36)}; require(all(p.is_file() for p in files.values()),"missing required S1..S35 files")
    metadata=spec.get("provenance",{}).get("metadata_files",[]); require(len(metadata)==3 and {r.get("file") for r in metadata}=={"Readme.txt","64-channels.loc","Freq_Phase.mat"},"verified metadata evidence required")
    evidence={}
    for row in metadata:
        p=Path(row["local_path"]).resolve(); require(p.is_file() and baseline.sha256_file(p)==row["sha256"],"metadata evidence SHA mismatch"); evidence[str(p)]=row["sha256"]
    source_hashes={str(source_spec):baseline.sha256_file(source_spec),**evidence,**{str(p.resolve()):baseline.sha256_file(p) for p in files.values()}}
    expected={}
    for split,subjects in SPLITS.items():
        expected[split]=[{"subject":s,"block":b,"frequency_hz":f,"condition_index":spec["frequency_order_hz"].index(f),"source_trial_index":spec["frequency_order_hz"].index(f)*6+b,"source_file":str(files[s].resolve()),"group_id":baseline.canonical([s,b,spec["depth"],spec["session"]])} for s in subjects for f in decoder.FREQUENCIES for b in range(6)]
    return {**decoder.FLAGS,"schema_version":1,"scope":SCOPE,"mat_root":str(mat_root),"source_spec_path":str(source_spec),"source_spec":spec,"source_hashes":source_hashes,"code_hashes":code_hashes(),"subjects":SPLITS,"expected_trials":expected,"expected_counts":{k:len(v) for k,v in expected.items()},"frequencies_hz":list(decoder.FREQUENCIES),"commands":{str(k):v for k,v in decoder.COMMANDS.items()},"seed":SEED,"preprocessing":decoder.PREPROCESSING,"model":decoder.MODEL_SPEC,"gate":decoder.GATE,"statistical_baselines":BASELINE_SPEC,"training":{"epochs":epochs,"batch_size":32,"learning_rate":0.001,"weight_decay":0.001,"optimizer":"AdamW","loss":"CrossEntropyLoss"},"selection_rule":SELECTION_RULE,"device_requested":device,"require_cuda":require_cuda,"personal_sessions_read":False,"final_test_waveforms_read_for_selection":False,"hyperparameter_sweep":False}
def validate_dataset(dataset,prereg,split_names):
    require(type(dataset) is baseline.Dataset and dataset.kind=="wang2016","only load_wang Dataset accepted")
    spec=prereg["source_spec"]; require(dataset.provenance.get("source_spec")==spec and dataset.provenance.get("source_spec_file_sha256")==prereg["source_hashes"][prereg["source_spec_path"]],"loader provenance mismatch")
    expected=[r for name in split_names for r in prereg["expected_trials"][name]]; key=lambda r:(r["subject"],r["block"],r["frequency_hz"]); em={key(r):r for r in expected}; require(len(em)==len(expected),"split overlap in expected manifest")
    hashes={str(Path(p).resolve()):h for p,h in dataset.input_hashes.items()}; required={prereg["source_spec_path"]}|{r["source_file"] for r in expected}; require(set(hashes)==required and all(prereg["source_hashes"].get(p)==h for p,h in hashes.items()),"loader hashes/source mismatch")
    require(len(dataset.trials)==len(expected),"missing subject/label/trial or extra data"); observed=set(); groups={name:set() for name in split_names}; parts={name:[] for name in split_names}
    for t in dataset.trials:
        require(type(t) is baseline.Trial and type(t.subject) is int and type(t.block) is int and t.frequency_hz in decoder.FREQUENCIES,"invalid Trial or exact label")
        identity=(t.subject,t.block,t.frequency_hz); require(identity in em and identity not in observed,"duplicate/missing expected trial"); observed.add(identity); row=em[identity]
        require(t.condition_index==row["condition_index"] and t.source_trial_index==row["source_trial_index"] and str(Path(t.source_file).resolve())==row["source_file"] and t.group_id==row["group_id"] and t.depth==spec["depth"] and t.session==spec["session"],"trial grouping/label/source mismatch")
        require(isinstance(t.eeg,np.ndarray) and t.eeg.shape==(8,1250) and t.eeg.dtype.kind in "fiu" and np.isfinite(t.eeg).all(),"requires finite actual (8,1250); no padding")
        split=next(name for name in split_names if t.subject in prereg["subjects"][name]); parts[split].append(t); groups[split].add(t.group_id)
    for i,a in enumerate(split_names):
        for b in split_names[i+1:]: require(not groups[a]&groups[b],"split group overlap")
    require(observed==set(em),"missing subject or label")
    for values in parts.values(): values.sort(key=lambda t:(t.subject,t.condition_index,t.block))
    return parts

def select_epoch(history):
    require(isinstance(history,list) and history,"empty history")
    for i,r in enumerate(history,1):
        require(r.get("epoch")==i and all(type(r.get(k)) in (int,float) and np.isfinite(r[k]) for k in ("validation_accuracy","validation_cross_entropy","train_cross_entropy")),"invalid/nonfinite history")
    return min(history,key=lambda r:(-r["validation_accuracy"],r["validation_cross_entropy"],r["epoch"]))

def fit_model(train_trials,val_trials,training,device):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if device.type=="cuda": torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    def tensors(trials):
        x=np.stack([decoder.spectral_features(t.eeg) for t in trials]); y=[decoder.FREQUENCIES.index(t.frequency_hz) for t in trials]
        return torch.tensor(x,dtype=torch.float32,device=device),torch.tensor(y,dtype=torch.long,device=device)
    x,y=tensors(train_trials); vx,vy=tensors(val_trials); model=decoder.HarmonicSpectrumMLP().to(device)
    with torch.no_grad(): model.feature_mean.copy_(x.mean(0)); model.feature_std.copy_(x.std(0,correction=0).clamp_min(1e-6))
    opt=torch.optim.AdamW(model.parameters(),lr=training["learning_rate"],weight_decay=training["weight_decay"]); loss_fn=nn.CrossEntropyLoss(); history=[]; best_state=None; best_key=None
    for epoch in range(1,training["epochs"]+1):
        model.train(); total=0.; order=torch.randperm(len(y),device=device)
        for ix in order.split(training["batch_size"]):
            opt.zero_grad(set_to_none=True); loss=loss_fn(model(x[ix]),y[ix]); require(bool(torch.isfinite(loss)),"nonfinite training loss"); loss.backward(); require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),"nonfinite gradient"); opt.step(); total+=float(loss.detach())*len(ix)
        model.eval()
        with torch.no_grad(): logits=model(vx); require(bool(torch.isfinite(logits).all()),"nonfinite validation logits"); row={"epoch":epoch,"train_cross_entropy":total/len(y),"validation_accuracy":float((logits.argmax(1)==vy).float().mean()),"validation_cross_entropy":float(loss_fn(logits,vy))}
        history.append(row); key=(-row["validation_accuracy"],row["validation_cross_entropy"],epoch)
        if best_key is None or key<best_key: best_key=key; best_state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    require(select_epoch(history)["epoch"]==best_key[2],"checkpoint selection mismatch"); model.load_state_dict(best_state,strict=True); return model,best_state,history

def summary(rows):
    require(rows,"empty evaluation"); confusion={str(f):{str(p):0 for p in decoder.FREQUENCIES} for f in decoder.FREQUENCIES}; commands={str(f):{**{c:0 for c in decoder.COMMANDS.values()},"UNKNOWN":0} for f in decoder.FREQUENCIES}
    for r in rows: confusion[str(int(r["frequency_hz"]))][str(int(r["predicted_frequency_hz"]))]+=1; commands[str(int(r["frequency_hz"]))][r["command"]]+=1
    accepted=[r for r in rows if r["accepted"]]; matches=sum(r["raw_label_match"] for r in rows)
    return {"trial_count":len(rows),"raw_labelmatch_count":matches,"raw_labelmatch":matches/len(rows),"raw_confusion":confusion,"command_confusion_including_UNKNOWN":commands,"accepted_count":len(accepted),"accepted_coverage":len(accepted)/len(rows),"accepted_accuracy":sum(r["raw_label_match"] for r in accepted)/len(accepted) if accepted else None,"UNKNOWN_count":len(rows)-len(accepted),"executed":False}

def evaluate_trials(model,trials,device):
    model.eval()
    with torch.no_grad(): probs=model(torch.tensor(np.stack([decoder.spectral_features(t.eeg) for t in trials]),dtype=torch.float32,device=device)).softmax(-1).cpu().numpy()
    rows=[]
    for t,p in zip(trials,probs):
        pred=decoder.command_prediction(p); rows.append({"trial_id":t.trial_id,"subject":t.subject,"block":t.block,"frequency_hz":t.frequency_hz,"group_id":t.group_id,**pred,"raw_label_match":pred["predicted_frequency_hz"]==t.frequency_hz})
    out=summary(rows); out["per_subject"]={str(s):summary([r for r in rows if r["subject"]==s]) for s in sorted({t.subject for t in trials})}; out["rows"]=rows; return out

def evaluate_baselines(trials,mlp_report):
    result={}
    for name,algorithm in (("CCA",baseline.CCA(decoder.FREQUENCIES)),("FBCCA",baseline.FBCCA(decoder.FREQUENCIES))):
        rows=[]
        for t in trials:
            p=algorithm.forward(t.eeg); accepted=p["best_score"]>=BASELINE_SPEC["score_thresholds"][name] and p["margin"]>=.05; f=p["predicted_frequency_hz"]; rows.append({"subject":t.subject,"frequency_hz":t.frequency_hz,"predicted_frequency_hz":f,"accepted":accepted,"raw_label_match":f==t.frequency_hz,"command":decoder.COMMANDS[f] if accepted else "UNKNOWN","executed":False})
        r=summary(rows); r["per_subject"]={str(s):summary([x for x in rows if x["subject"]==s]) for s in sorted({t.subject for t in trials})}; r["mlp_accuracy_minus_baseline"]=mlp_report["raw_labelmatch"]-r["raw_labelmatch"]; r["mlp_worse_than_baseline"]=mlp_report["raw_labelmatch"]<r["raw_labelmatch"]; result[name]=r
    return result
def train(mat_root,source_spec,output_dir,*,epochs=100,device="cuda",require_cuda=False):
    chosen=select_device(device,require_cuda); output_dir=Path(output_dir).resolve(); require(not output_dir.exists(),"run directory must be new; no overwrite/resume")
    prereg=make_preregistration(mat_root,source_spec,epochs=epochs,device=device,require_cuda=require_cuda); output_dir.mkdir(parents=False,exist_ok=False)
    write_json_new(output_dir/"preregistration.json",prereg); write_json_new(output_dir/"preregistration_lock.json",{"sha256":digest(prereg)})
    dataset=baseline.load_wang(Path(prereg["mat_root"]),Path(prereg["source_spec_path"]),subjects=SPLITS["train"]+SPLITS["validation"])
    parts=validate_dataset(dataset,prereg,("train","validation")); model,state,history=fit_model(parts["train"],parts["validation"],prereg["training"],chosen); selected=select_epoch(history)
    validation=evaluate_trials(model,parts["validation"],chosen); validation["statistical_baselines"]=evaluate_baselines(parts["validation"],validation); validation.update(decoder.FLAGS); validation.update({"scope":SCOPE,"split":"validation"})
    write_json_new(output_dir/"history.json",{"epochs":history}); write_json_new(output_dir/"validation_report.json",validation)
    with (output_dir/"best_weights.pt").open("xb") as handle: torch.save(state,handle)
    selection={**decoder.FLAGS,"scope":SCOPE,"schema_version":1,"selection_rule":SELECTION_RULE,"selected":selected,"preregistration_sha256":digest(prereg),"artifacts":{n:baseline.sha256_file(output_dir/n) for n in ("best_weights.pt","history.json","validation_report.json")},"checkpoint_file":"best_weights.pt","checkpoint_format":"state_dict_only","source_hashes":prereg["source_hashes"],"subjects":SPLITS,"final_test_evaluated":False,"device_used":str(chosen),"torch_version":str(torch.__version__),"cuda_version":torch.version.cuda}
    write_json_new(output_dir/"selection.json",selection); write_json_new(output_dir/"selection_lock.json",{"sha256":digest(selection)})
    return {**decoder.FLAGS,"run_dir":str(output_dir),"selected_epoch":selected["epoch"],"validation_raw_labelmatch":validation["raw_labelmatch"],"final_test_evaluated":False}

def load_selected_model(run_dir,device):
    run_dir=Path(run_dir).resolve(); names=("preregistration.json","preregistration_lock.json","selection.json","selection_lock.json","best_weights.pt","history.json","validation_report.json")
    require(all((run_dir/n).is_file() and not (run_dir/n).is_symlink() for n in names),"missing/unsafe selection artifact")
    prereg=baseline.read_json(run_dir/"preregistration.json"); selection=baseline.read_json(run_dir/"selection.json")
    require(baseline.read_json(run_dir/"preregistration_lock.json")=={"sha256":digest(prereg)} and baseline.read_json(run_dir/"selection_lock.json")=={"sha256":digest(selection)} and selection.get("preregistration_sha256")==digest(prereg),"manifest lock/hash mismatch")
    for manifest in (prereg,selection): require(all(manifest.get(k)==v for k,v in decoder.FLAGS.items()) and manifest.get("scope")==SCOPE and manifest.get("subjects")==SPLITS,"manifest flags/splits changed")
    require(prereg.get("model")==decoder.MODEL_SPEC and prereg.get("preprocessing")==decoder.PREPROCESSING and prereg.get("gate")==decoder.GATE and prereg.get("frequencies_hz")==list(decoder.FREQUENCIES) and prereg.get("commands")=={str(k):v for k,v in decoder.COMMANDS.items()} and prereg.get("statistical_baselines")==BASELINE_SPEC,"manifest contract changed")
    check_spec(prereg["source_spec"]); require(prereg.get("code_hashes")==code_hashes(),"training code/loader changed since preregistration")
    require(selection.get("selection_rule")==SELECTION_RULE and selection.get("checkpoint_file")=="best_weights.pt" and selection.get("checkpoint_format")=="state_dict_only" and selection.get("final_test_evaluated") is False,"invalid checkpoint selection")
    require(set(selection.get("artifacts",{}))=={"best_weights.pt","history.json","validation_report.json"},"selection artifacts changed")
    verify_hashes({str(run_dir/n):h for n,h in selection["artifacts"].items()}); history=baseline.read_json(run_dir/"history.json")["epochs"]; require(len(history)==prereg["training"]["epochs"] and select_epoch(history)==selection["selected"],"checkpoint is not best validation epoch")
    require(selection.get("source_hashes")==prereg["source_hashes"],"source bindings changed"); verify_hashes(prereg["source_hashes"])
    model=decoder.HarmonicSpectrumMLP(); state=torch.load(run_dir/"best_weights.pt",map_location="cpu",weights_only=True); expected=model.state_dict(); require(isinstance(state,dict) and set(state)==set(expected),"checkpoint state_dict keys mismatch")
    for n,v in state.items(): require(isinstance(v,torch.Tensor) and v.shape==expected[n].shape and v.dtype==expected[n].dtype and bool(torch.isfinite(v).all()),"invalid/nonfinite checkpoint tensor")
    require(bool((state["feature_std"]>=1e-6).all()),"invalid feature standardizer"); model.load_state_dict(state,strict=True); return model.to(device).eval(),prereg,selection

def evaluate_test(run_dir,*,device="cuda",require_cuda=False):
    chosen=select_device(device,require_cuda); run_dir=Path(run_dir).resolve(); require(not (run_dir/"final_test.json").exists() and not (run_dir/"test_attempt.json").exists(),"final test already attempted; no repeat/overwrite")
    model,prereg,selection=load_selected_model(run_dir,chosen)
    write_json_new(run_dir/"test_attempt.json",{**decoder.FLAGS,"started_utc":datetime.now(timezone.utc).isoformat(),"selection_sha256":digest(selection),"subjects":SPLITS["test"],"retry_authorized":False})
    dataset=baseline.load_wang(Path(prereg["mat_root"]),Path(prereg["source_spec_path"]),subjects=SPLITS["test"]); trials=validate_dataset(dataset,prereg,("test",))["test"]
    report=evaluate_trials(model,trials,chosen); report["statistical_baselines"]=evaluate_baselines(trials,report); report.update({**decoder.FLAGS,"scope":SCOPE,"split":"final_test","subjects":SPLITS["test"],"selection_sha256":digest(selection),"checkpoint_sha256":selection["artifacts"]["best_weights.pt"],"preregistration_sha256":digest(prereg),"gate":decoder.GATE,"device_used":str(chosen),"source_hashes":{str(Path(p).resolve()):h for p,h in dataset.input_hashes.items()}})
    write_json_new(run_dir/"final_test.json",report); return {**decoder.FLAGS,"run_dir":str(run_dir),"test_raw_labelmatch":report["raw_labelmatch"],"accepted_coverage":report["accepted_coverage"],"accepted_accuracy":report["accepted_accuracy"]}

def make_parser():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True); tr=sub.add_parser("train"); tr.add_argument("--mat-root",required=True,type=Path); tr.add_argument("--source-spec",required=True,type=Path); tr.add_argument("--output-dir",required=True,type=Path); tr.add_argument("--epochs",type=int,default=100); te=sub.add_parser("evaluate_test"); te.add_argument("--run-dir",required=True,type=Path)
    for c in (tr,te): c.add_argument("--device",choices=("cuda","cpu"),default="cuda"); c.add_argument("--require-cuda",action="store_true")
    return p

def main(argv=None):
    args=make_parser().parse_args(argv)
    try:
        result=train(args.mat_root,args.source_spec,args.output_dir,epochs=args.epochs,device=args.device,require_cuda=args.require_cuda) if args.command=="train" else evaluate_test(args.run_dir,device=args.device,require_cuda=args.require_cuda)
    except (ValueError,OSError,RuntimeError,KeyError,TypeError) as exc:
        print(json.dumps({**decoder.FLAGS,"status":"error","error":str(exc)},ensure_ascii=False,allow_nan=False)); return 2
    print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
