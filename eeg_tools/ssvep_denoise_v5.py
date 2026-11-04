"""v5 multichannel SSVEP reference-reconstruction denoiser.

Research-only: the target is a recorded reference after deterministic synthetic
corruption, not clean EEG ground truth. Training reads only frozen train subjects.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, math
from pathlib import Path
from typing import Any
import numpy as np
try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:
    torch = None; nn = None; F = None
from eeg_tools import ssvep_reference_v5 as reference_safety

WORKSPACE=Path(__file__).resolve().parents[1]
REFERENCE_RELATIVE=Path("datasets/processed/ssvep_reference_v5_20261007")
CONFIG_RELATIVE=Path("configs/learned_denoiser_v5_ssvep.json")
MODEL_VERSION="learned_denoise_v5_ssvep"
REFERENCE_MANIFEST_SHA256="607617315889697a5bfcea986bf055bf4ed31182e8a69776842c736f481604b6"
PARTITION_LOCK_SHA256="cb5e05b9d33fb3034f178392f65817185226756cf4c609dcfec53b4c2d4b8a8f"
EPSILON=1e-8

class SSVEPDenoiseV5Error(ValueError):
    """Fail-closed v5 training, checkpoint, or inference error."""

def sha256(path: Path) -> str:
    with Path(path).open("rb") as h: return hashlib.file_digest(h,"sha256").hexdigest()

def _plain(v: Any) -> bool:
    if v is None or type(v) in (bool,int,str): return True
    if type(v) is float: return bool(np.isfinite(v))
    if type(v) is list: return all(_plain(x) for x in v)
    if type(v) is dict: return all(type(k) is str and _plain(x) for k,x in v.items())
    return False

def _json(path: Path) -> dict:
    try: v=json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError,UnicodeError,json.JSONDecodeError) as e: raise SSVEPDenoiseV5Error(f"invalid JSON: {path}") from e
    if not isinstance(v,dict) or not _plain(v): raise SSVEPDenoiseV5Error(f"JSON must be finite object: {path}")
    return v

def load_config(path: str|Path=CONFIG_RELATIVE, *, root: Path=WORKSPACE) -> dict:
    p=(root/path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    c=_json(p); validate_config(c); return c

def validate_config(c: dict) -> None:
    keys={"schema_version","status","model_version","model_type","reference_version","candidate_signal_key","input_channels","window_samples","hop_samples","normalization","model","corruption","loss","training","runtime","safety"}
    if set(c)!=keys or not _plain(c): raise SSVEPDenoiseV5Error("v5 config schema mismatch")
    if c["schema_version"]!=1 or c["status"]!="formal" or c["model_version"]!=MODEL_VERSION or c["reference_version"]!="ssvep_reference_v5" or c["candidate_signal_key"]!="processed_v5": raise SSVEPDenoiseV5Error("v5 formal/reference mismatch")
    if (c["input_channels"],c["window_samples"],c["hop_samples"],c["normalization"])!=(8,256,128,"per_channel_window_mean_rms_with_eps"): raise SSVEPDenoiseV5Error("v5 input/window contract mismatch")
    if c["model"]!={"hidden_channels":64,"kernel_size":7,"dilations":[1,2,4,8,16,32],"dropout":0.0,"residual_limit_normalized":3.0}: raise SSVEPDenoiseV5Error("v5 architecture mismatch")
    if c["corruption"]["seed"]!=20261007 or c["corruption"]["identity_probability"]!=0.20 or c["corruption"]["combine_probability"]!=0.20 or c["corruption"]["families"]!=["transient","low_frequency_drift","high_frequency_burst","common_mode"]: raise SSVEPDenoiseV5Error("v5 corruption contract mismatch")
    if c["corruption"]["target_definition"]!="recorded_reference_reconstruction; reference is not clean ground truth": raise SSVEPDenoiseV5Error("v5 target must be recorded reference")
    if c["loss"]!={"time_smooth_l1_weight":1.0,"band_log_magnitude_weight":0.50,"identity_weight":2.0,"change_penalty_weight":0.02,"protected_band_hz":[8.0,40.0]}: raise SSVEPDenoiseV5Error("v5 loss contract mismatch")
    expected={"fit_subjects":[1,2,3,4],"internal_validation_subjects":[5,6],"outer_validation_subjects":[7,8],"holdout_subjects":[9,10],"epochs":12,"batch_size":64,"learning_rate":0.0005,"weight_decay":0.0001,"seed":20261007,"selection":"minimum_internal_validation_total_loss_then_earliest_epoch","amp":False,"workers":0,"require_cuda":True}
    if c["training"]!=expected: raise SSVEPDenoiseV5Error("v5 training schedule/partition mismatch")
    if c["runtime"]!={"enabled":False,"validation_status":"not_assessed","deployment_scope":"offline_candidate_evaluation_only","production_validated":False}: raise SSVEPDenoiseV5Error("v5 runtime must remain disabled")
    if c["safety"]!={"holdout_waveforms_loaded_before_candidate_freeze":False,"outer_validation_not_used_for_training_or_selection":True,"native_units_assumed":False,"cross_dataset_native_power_aggregation":False,"clinical_validity_claim":False}: raise SSVEPDenoiseV5Error("v5 safety contract mismatch")

def _range(section: dict, name: str) -> tuple[float,float]:
    a,b=section[name]
    if len(section[name])!=2 or not 0<float(a)<=float(b): raise SSVEPDenoiseV5Error("invalid corruption range")
    return float(a),float(b)

def _pick(rng, bounds): return float(rng.uniform(bounds[0],bounds[1]))
def _channels(rng, bounds, n):
    lo,hi=int(bounds[0]),min(int(bounds[1]),n)
    return np.asarray(rng.choice(n,size=int(rng.integers(lo,hi+1)),replace=False),dtype=np.int64)

def corrupt_window(signal: np.ndarray, sample_rate_hz: int, config: dict, seed: int) -> tuple[np.ndarray,bool,list[str]]:
    x=np.asarray(signal,dtype=np.float64)
    if x.ndim!=2 or x.shape[0]!=8 or not np.isfinite(x).all(): raise SSVEPDenoiseV5Error("corruption expects finite [8,time]")
    rng=np.random.default_rng(int(seed)); c=config["corruption"]
    if float(rng.random())<float(c["identity_probability"]): return x.copy(),True,["identity"]
    y=x.copy(); scale=np.sqrt(np.mean((x-x.mean(axis=1,keepdims=True))**2,axis=1)+EPSILON); t=np.arange(x.shape[1],dtype=np.float64)/float(sample_rate_hz)
    families=list(c["families"]); chosen=[families[int(rng.integers(0,len(families)))]]
    if float(rng.random())<float(c["combine_probability"]):
        second=families[int(rng.integers(0,len(families)))]
        if second not in chosen: chosen.append(second)
    for family in chosen:
        if family=="transient":
            amp=_pick(rng,_range(c["transient"],"amplitude_rms_range")); width=int(rng.integers(*map(int,c["transient"]["width_samples_range"]),endpoint=True)); center=int(rng.integers(0,x.shape[1])); pulse=np.exp(-0.5*((np.arange(x.shape[1])-center)/max(width/2.355,1.0))**2)
            for ch in _channels(rng,c["transient"]["channel_count_range"],8): y[ch]+=amp*scale[ch]*pulse*float(rng.choice([-1.0,1.0]))
        elif family=="low_frequency_drift":
            freq=_pick(rng,_range(c["low_frequency_drift"],"frequency_hz_range")); amp=_pick(rng,_range(c["low_frequency_drift"],"amplitude_rms_range")); wave=np.sin(2*math.pi*freq*t+_pick(rng,(0,2*math.pi)))
            for ch in _channels(rng,c["low_frequency_drift"]["channel_count_range"],8): y[ch]+=amp*scale[ch]*wave
        elif family=="high_frequency_burst":
            freq=min(_pick(rng,_range(c["high_frequency_burst"],"frequency_hz_range")),sample_rate_hz/2-1); amp=_pick(rng,_range(c["high_frequency_burst"],"amplitude_rms_range")); duration=int(rng.integers(*map(int,c["high_frequency_burst"]["duration_samples_range"]),endpoint=True)); start=int(rng.integers(0,max(1,x.shape[1]-duration+1))); n=min(duration,x.shape[1]-start); local=np.arange(n,dtype=np.float64); env=np.sin(math.pi*(local+1)/(n+1))**2; burst=np.zeros(x.shape[1]); burst[start:start+n]=env*np.sin(2*math.pi*freq*local/sample_rate_hz+_pick(rng,(0,2*math.pi)))
            for ch in _channels(rng,c["high_frequency_burst"]["channel_count_range"],8): y[ch]+=amp*scale[ch]*burst
        elif family=="common_mode":
            amp=_pick(rng,_range(c["common_mode"],"amplitude_rms_range")); width=int(rng.integers(*map(int,c["common_mode"]["width_samples_range"]),endpoint=True)); center=int(rng.integers(0,x.shape[1])); common=np.exp(-0.5*((np.arange(x.shape[1])-center)/max(width/2.355,1.0))**2); y+=(amp*float(scale.mean()))*common[None,:]
        else: raise SSVEPDenoiseV5Error(f"unknown corruption family: {family}")
    if not np.isfinite(y).all(): raise SSVEPDenoiseV5Error("corruption produced nonfinite values")
    return y,False,chosen

def normalize_window(signal: np.ndarray):
    x=np.asarray(signal,dtype=np.float64); mean=x.mean(axis=-1,keepdims=True); scale=np.sqrt(np.mean((x-mean)**2,axis=-1,keepdims=True)+EPSILON); return ((x-mean)/scale).astype(np.float32),mean.astype(np.float64),scale.astype(np.float64)

@dataclass
class ReferenceRecord:
    dataset: str
    subject_id: int
    sample_rate_hz: int
    signal: np.ndarray
    npz_sha256: str
    sidecar_sha256: str

def _inside(root: Path, path: Path) -> Path:
    try: return reference_safety._inside(root,path).resolve()
    except Exception as e: raise SSVEPDenoiseV5Error(f"unsafe path: {path}") from e

def load_reference_records(root: Path=WORKSPACE, subjects: list[int]|tuple[int,...]=(1,2,3,4,5,6)) -> list[ReferenceRecord]:
    root=Path(root).resolve(); rd=_inside(root,root/REFERENCE_RELATIVE); mp=rd/"manifest.json"
    if sha256(mp)!=REFERENCE_MANIFEST_SHA256: raise SSVEPDenoiseV5Error("frozen reference manifest SHA mismatch")
    m=_json(mp); requested={int(x) for x in subjects}
    if not requested or not requested.issubset(set(range(1,7))): raise SSVEPDenoiseV5Error("loader only permits train subjects 1..6")
    rows=[r for r in m["materialized_subjects"] if int(r["subject_id"]) in requested and r["split"]=="train"]
    if {int(r["subject_id"]) for r in rows}!=requested: raise SSVEPDenoiseV5Error("requested train subject set incomplete")
    result=[]
    for row in rows:
        path=_inside(root,rd/row["relative_path"]); sidepath=_inside(root,Path(str(path)+".manifest.json"))
        if sha256(path)!=row["npz_sha256"] or sha256(sidepath)!=row["manifest_sha256"]: raise SSVEPDenoiseV5Error(f"source hash mismatch: {path}")
        side=_json(sidepath)
        if side.get("raw_reference_is_clean_ground_truth") is not False: raise SSVEPDenoiseV5Error("reference clean-ground-truth flag unsafe")
        with np.load(path,allow_pickle=False) as loaded:
            allowed={"reference","trial_index","target_id","block_id","electrode_id","frequency_hz","phase_pi_multiple","manifest_json"}
            if set(loaded.files)-allowed: raise SSVEPDenoiseV5Error("unexpected NPZ member")
            signal=np.asarray(loaded["reference"])
        if signal.dtype!=np.float64 or signal.ndim!=3 or signal.shape[2]!=8 or not np.isfinite(signal).all() or tuple(signal.shape)!=tuple(side["reference_shape"]): raise SSVEPDenoiseV5Error(f"invalid reference array: {path}")
        result.append(ReferenceRecord(row["dataset"],int(row["subject_id"]),int(side["native_sample_rate_hz"]),signal,row["npz_sha256"],row["manifest_sha256"]))
    return sorted(result,key=lambda r:(r.dataset,r.subject_id))

class ReferenceWindowDataset:
    def __init__(self,records,subjects,config,*,epoch=0):
        allowed=set(subjects); self.records=[r for r in records if r.subject_id in allowed]; self.config=config; self.epoch=int(epoch); self.window=int(config["window_samples"]); self.hop=int(config["hop_samples"])
        if not self.records: raise SSVEPDenoiseV5Error("empty subject partition")
        self.entries=[]
        for ri,r in enumerate(self.records):
            for trial in range(r.signal.shape[0]):
                starts=list(range(0,max(1,r.signal.shape[1]-self.window+1),self.hop)); final=r.signal.shape[1]-self.window
                if final>=0 and (not starts or starts[-1]!=final): starts.append(final)
                self.entries.extend((ri,trial,int(s)) for s in starts)
        if not self.entries: raise SSVEPDenoiseV5Error("empty window table")
    def set_epoch(self,epoch): self.epoch=int(epoch)
    def __len__(self): return len(self.entries)
    def __getitem__(self,index):
        ri,trial,start=self.entries[index]; r=self.records[ri]; target=np.ascontiguousarray(r.signal[trial,start:start+self.window,:].T)
        seed=int(self.config["corruption"]["seed"]+self.epoch*1000003+index*7919+r.subject_id*101+trial)
        noisy,identity,families=corrupt_window(target,r.sample_rate_hz,self.config,seed); nnorm,mean,scale=normalize_window(noisy); tnorm=((target-mean)/scale).astype(np.float32)
        return {"noisy":nnorm,"target":tnorm,"identity":np.bool_(identity),"sample_rate_hz":np.int64(r.sample_rate_hz),"dataset":r.dataset,"subject_id":np.int64(r.subject_id),"trial":np.int64(trial),"start":np.int64(start),"families":",".join(families)}

if nn is not None:
    class ResidualBlock(nn.Module):
        def __init__(self,channels,kernel,dilation,dropout):
            super().__init__(); p=dilation*(kernel-1)//2
            self.net=nn.Sequential(nn.Conv1d(channels,channels,kernel,padding=p,dilation=dilation),nn.GroupNorm(8,channels),nn.GELU(),nn.Dropout(dropout),nn.Conv1d(channels,channels,kernel,padding=p,dilation=dilation),nn.GroupNorm(8,channels))
        def forward(self,x): return x+0.5*self.net(x)
    class SSVEPDenoiser(nn.Module):
        def __init__(self,config):
            super().__init__(); m=config["model"]; h=int(m["hidden_channels"]); k=int(m["kernel_size"]); self.limit=float(m["residual_limit_normalized"])
            self.input=nn.Conv1d(8,h,k,padding=k//2); self.blocks=nn.Sequential(*[ResidualBlock(h,k,int(d),float(m["dropout"])) for d in m["dilations"]]); self.output=nn.Sequential(nn.GELU(),nn.Conv1d(h,8,1))
        def forward(self,x):
            if x.ndim!=3 or x.shape[1]!=8: raise ValueError("SSVEPDenoiser expects [batch,8,time]")
            return x+torch.tanh(self.output(self.blocks(self.input(x))))*self.limit
else:
    SSVEPDenoiser=None

def build_model(config):
    validate_config(config)
    if torch is None: raise SSVEPDenoiseV5Error("PyTorch required")
    return SSVEPDenoiser(config)

def _band_loss(pred,target,fs,low=8.0,high=40.0):
    freq=torch.fft.rfftfreq(pred.shape[-1],d=1.0/float(fs),device=pred.device); mask=(freq>=low)&(freq<=high)
    a=torch.log1p(torch.abs(torch.fft.rfft(pred,dim=-1))[...,mask]); b=torch.log1p(torch.abs(torch.fft.rfft(target,dim=-1))[...,mask]); return F.smooth_l1_loss(a,b)

def compute_loss(pred,target,noisy,identity,fs,config):
    lc=config["loss"]; vals={"time":F.smooth_l1_loss(pred,target),"band":_band_loss(pred,target,fs),"change":torch.mean((pred-noisy)**2)}
    vals["identity"]=F.smooth_l1_loss(pred[identity],target[identity]) if bool(torch.any(identity)) else torch.zeros((),device=pred.device,dtype=pred.dtype)
    vals["total"]=lc["time_smooth_l1_weight"]*vals["time"]+lc["band_log_magnitude_weight"]*vals["band"]+lc["identity_weight"]*vals["identity"]+lc["change_penalty_weight"]*vals["change"]
    if not all(bool(torch.isfinite(v)) for v in vals.values()): raise SSVEPDenoiseV5Error("nonfinite v5 loss")
    return vals

def denoise_array(model,signal,sample_rate_hz,config,device="cpu"):
    """Apply the fixed native-rate model with symmetric-Hann residual OLA.

    The model predicts a normalized change. Changes are restored with the
    source window RMS, Hann-weighted across overlapping windows, and added to
    the original float64 signal. Samples with zero Hann support are unchanged.
    """
    if torch is None: raise SSVEPDenoiseV5Error("PyTorch required")
    arr=np.asarray(signal,dtype=np.float64)
    if arr.ndim!=3 or arr.shape[2]!=8 or not np.isfinite(arr).all(): raise SSVEPDenoiseV5Error("inference expects finite [trial,time,8]")
    model.eval(); out=np.empty_like(arr); window=int(config["window_samples"]); hop=int(config["hop_samples"])
    weights_kernel=np.hanning(window).astype(np.float64)
    with torch.no_grad():
        for trial in range(arr.shape[0]):
            length=arr.shape[1]
            if length<window: raise SSVEPDenoiseV5Error("inference signal shorter than fixed window")
            starts=list(range(0,max(1,length-window+1),hop)); final=length-window
            if final>=0 and (not starts or starts[-1]!=final): starts.append(final)
            accum=np.zeros((8,length),dtype=np.float64); weights=np.zeros(length,dtype=np.float64)
            for start in starts:
                end=min(length,start+window); chunk=arr[trial,start:end,:].T; actual=chunk.shape[1]
                if actual<window: chunk=np.pad(chunk,((0,0),(0,window-actual)),mode="reflect")
                normalized,mean,scale=normalize_window(chunk); x=torch.from_numpy(normalized[None]).to(device=device,dtype=torch.float32)
                pred=model(x).detach().cpu().numpy()[0].astype(np.float64)
                residual=(pred-normalized.astype(np.float64))*scale
                local_weights=weights_kernel[:actual]
                accum[:,start:end]+=residual[:,:actual]*local_weights[None,:]; weights[start:end]+=local_weights
            supported=weights>0
            out[trial]=arr[trial]
            out[trial,supported,:]+= (accum[:,supported]/weights[supported][None,:]).T
    if not np.isfinite(out).all(): raise SSVEPDenoiseV5Error("inference produced nonfinite values")
    return out

def checkpoint_metadata(config,training_report):
    validate_config(config)
    if training_report.get("reference_manifest_sha256")!=REFERENCE_MANIFEST_SHA256 or training_report.get("partition_lock_sha256")!=PARTITION_LOCK_SHA256: raise SSVEPDenoiseV5Error("checkpoint reference binding mismatch")
    if not _plain(training_report): raise SSVEPDenoiseV5Error("training report must be finite primitive metadata")
    return {"checkpoint_schema_version":1,"model_version":MODEL_VERSION,"config":config,"reference_manifest_sha256":REFERENCE_MANIFEST_SHA256,"partition_lock_sha256":PARTITION_LOCK_SHA256,"training_report":training_report,"runtime":config["runtime"],"clean_ground_truth_used":False,"holdout_waveforms_loaded":False,"outer_validation_used_for_selection":False}

def load_checkpoint(path,config=None,device="cpu"):
    if torch is None: raise SSVEPDenoiseV5Error("PyTorch required")
    cfg=config or load_config(); validate_config(cfg); ck=torch.load(Path(path),map_location=device,weights_only=True)
    if not isinstance(ck,dict) or ck.get("model_version")!=MODEL_VERSION or ck.get("reference_manifest_sha256")!=REFERENCE_MANIFEST_SHA256 or ck.get("partition_lock_sha256")!=PARTITION_LOCK_SHA256: raise SSVEPDenoiseV5Error("invalid v5 checkpoint binding")
    if ck.get("clean_ground_truth_used") is not False or ck.get("holdout_waveforms_loaded") is not False or ck.get("outer_validation_used_for_selection") is not False: raise SSVEPDenoiseV5Error("unsafe v5 provenance flags")
    if ck.get("config")!=cfg or set(ck.get("models",{}))!={"Nakanishi2015","Wearable"}: raise SSVEPDenoiseV5Error("v5 checkpoint config/domain mismatch")
    models={}
    for domain,state in ck["models"].items():
        model=build_model(cfg).to(device=device,dtype=torch.float32); model.load_state_dict(state,strict=True); model.eval(); models[domain]=model
    ck["loaded_models"]=models; return ck

