"""Frozen task-reference trials, NOT supervised signal ground truth.

Freeze consumes audit metadata and author evidence only (source lstat is allowed,
no source bytes/waveforms). Build selects train/validation variables only.
Native values/rates/time points are retained; no signal processing is performed.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import numpy as np
from scipy.io import loadmat, whosmat
from scripts import audit_local_ssvep_v5 as safety

WORKSPACE=Path(__file__).resolve().parents[1]
VERSION="ssvep_reference_v5"
MEMORY_MAX_MB=256
FREQUENCIES=[9.25,11.25,13.25,9.75,11.75,13.75,10.25,12.25,14.25,10.75,12.75,14.75]
PHASES=[0.,0.,0.,.5,.5,.5,1.,1.,1.,1.5,1.5,1.5]
PARTITIONS={"train":list(range(1,7)),"validation":[7,8],"holdout":[9,10]}
EVIDENCE_PATHS={
    "nakanishi_author":"datasets/external/v5_ssvep_local_audit/evidence/nak_author_readme.txt",
    "wearable_readme":"datasets/external/v5_ssvep_local_audit/evidence/Wearable_Readme.pdf.txt",
    "wearable_stimulation":"datasets/external/v5_ssvep_local_audit/evidence/Wearable_stimulation_information.pdf.txt"}
SPECS={
    "Nakanishi2015":{"variable":"eeg","shape":[12,8,1114,15],"relative_template":"Nakanishi2015/data/s{subject_id}.mat",
        "source_axes":["target","channel","time","block"],"sample_rate_hz":256,"onset_sample_zero_based":38,
        "channel_order":None,"units":None,"frequency_order_hz":FREQUENCIES,"phase_pi_multiples":None,
        "electrode_order":None,"trial_order":"target outer, block inner; both IDs zero-based","evidence_roles":["nakanishi_author"]},
    "Wearable":{"variable":"data","shape":[8,710,2,10,12],"relative_template":"Wearable/S{subject_id:03d}.mat",
        "source_axes":["channel","time","electrode","block","target"],"sample_rate_hz":250,"onset_sample_zero_based":125,
        "channel_order":["POz","PO3","PO4","PO5","PO6","Oz","O1","O2"],"units":None,
        "frequency_order_hz":FREQUENCIES,"phase_pi_multiples":PHASES,"electrode_order":["dry","wet"],
        "trial_order":"target outer, block middle, electrode inner; all IDs zero-based",
        "evidence_roles":["wearable_readme","wearable_stimulation"]}}

class SSVEPReferenceV5Error(ValueError):
    """Evidence, identity, scope, memory, array or output safety failure."""

def sha256(path):return safety.stream_sha256(Path(path),safety.ReadBudget(0))

def _plain(value):
    if value is None or type(value) in (str,int,bool):return True
    if type(value) is float:return math.isfinite(value)
    if type(value) is list:return all(_plain(v) for v in value)
    if type(value) is dict:return all(type(k) is str and _plain(v) for k,v in value.items())
    return False

def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode("utf-8")).hexdigest()

def _sha(value):
    if type(value) is not str or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):raise SSVEPReferenceV5Error("lowercase SHA256 required")
    return value

def _guard_chain(path):
    path=Path(os.path.abspath(path))
    for parent in reversed([path]+list(path.parents)):
        if parent.exists() or parent.is_symlink():
            if safety._is_link(parent):raise SSVEPReferenceV5Error(f"symlink/junction/reparse path refused: {parent}")
    return path

def _inside(workspace,path):
    workspace=_guard_chain(workspace).resolve()
    path=Path(path)
    if not path.is_absolute():path=workspace/path
    path=_guard_chain(path)
    if not path.resolve().is_relative_to(workspace) or path.resolve()==workspace:
        raise SSVEPReferenceV5Error("output/evidence metadata must remain within workspace")
    return path.resolve()

def _output(workspace,path,source_root=None):
    path=_inside(workspace,path)
    if source_root is not None and (path.is_relative_to(source_root) or source_root.is_relative_to(path)):
        raise SSVEPReferenceV5Error("output must not be under/enclose source root")
    if os.path.splitdrive(str(path))[0].casefold()=="d:":raise SSVEPReferenceV5Error("D drive is read-only input; output must be in workspace")
    if path.exists() or path.is_symlink():raise SSVEPReferenceV5Error("output exists; refuse overwrite")
    return path

def _json(path,maximum=32*1024*1024):
    path=_guard_chain(path)
    if not path.is_file() or path.stat().st_size>maximum:raise SSVEPReferenceV5Error("JSON missing or exceeds bounded metadata budget")
    value=json.loads(path.read_text(encoding="utf-8-sig"))
    if type(value) is not dict or not _plain(value):raise SSVEPReferenceV5Error("metadata must be finite plain JSON")
    return value

def _relative_source(root,relative):
    if type(relative) is not str or "\\" in relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise SSVEPReferenceV5Error("source identity requires safe POSIX relative path")
    path=_guard_chain(root/relative)
    if not path.resolve().is_relative_to(root) or path.resolve()==root:raise SSVEPReferenceV5Error("source path escape")
    return path.resolve()

def _subjects():
    return [(name,sid,split) for name in SPECS for split,ids in PARTITIONS.items() for sid in ids]

def _trial_count(name,spec):
    shape=spec["shape"]
    return int(shape[0]*shape[3]) if name=="Nakanishi2015" else int(shape[2]*shape[3]*shape[4])

def _evidence(workspace):
    records={};texts={}
    for role,relative in EVIDENCE_PATHS.items():
        path=_inside(workspace,workspace/relative)
        if not path.is_file() or path.stat().st_size>4*1024*1024:raise SSVEPReferenceV5Error("author evidence missing/over metadata budget")
        before=safety._stamp(path);digest=sha256(path);text=path.read_text(encoding="utf-8-sig")
        if safety._stamp(path)!=before or sha256(path)!=digest:raise SSVEPReferenceV5Error("author evidence changed during freeze")
        records[role]={"relative_path":relative,"sha256":digest};texts[role]=text
    # Check the cited texts actually support the frozen constants. Do not infer
    # additional units/electrodes/subjects from vague filenames or visual labels.
    import re
    naka=re.sub(r"\s+"," ",texts["nakanishi_author"])
    freq=re.search(r"\[\s*9\.25\s*,([^\]]+)\]",naka)
    sequence=[float(x.strip()) for x in ("9.25,"+freq.group(1)).split(",")] if freq else []
    if sequence!=FREQUENCIES or not re.search(r"Sampling rate.*?256",naka,re.I) or "39th sample" not in naka:
        raise SSVEPReferenceV5Error("Nakanishi author evidence does not support rates/order/onset")
    wearable=re.sub(r"\s+"," ",texts["wearable_readme"])
    if (not re.search(r"250\s*Hz",wearable,re.I) or not re.search(r"\[\s*8\s*,\s*710\s*,\s*2\s*,\s*10\s*,\s*12\s*\]",wearable)
            or not all(x in wearable for x in ("POz","PO3","PO4","PO5","PO6","Oz","O1","O2","0.5 s","1:dry","2:wet"))):
        raise SSVEPReferenceV5Error("Wearable author evidence does not support axes/channels/onset")
    rows=[]
    for line in texts["wearable_stimulation"].splitlines():
        fields=line.strip().split()
        if len(fields)==3:
            try:rows.append((fields[0],float(fields[1]),float(fields[2])))
            except ValueError:pass
    if ([r[0] for r in rows]!=list("1234567890*#") or [r[1] for r in rows]!=FREQUENCIES or [r[2] for r in rows]!=PHASES):
        raise SSVEPReferenceV5Error("Wearable stimulation evidence does not match frequency/phase order")
    return records

def _publish_json(path,value):
    if not _plain(value):raise SSVEPReferenceV5Error("nonprimitive metadata")
    path.parent.mkdir(parents=True,exist_ok=True);_guard_chain(path.parent)
    fd,name=tempfile.mkstemp(prefix=path.name+".",suffix=".pending",dir=path.parent);os.close(fd);temporary=Path(name)
    try:
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        os.link(temporary,path)  # Exclusive, never overwrite even on a race.
    finally:temporary.unlink(missing_ok=True)


def freeze_reference(audit_path,output_lock,workspace=None):
    workspace=Path(workspace or WORKSPACE).resolve()
    audit_path=_inside(workspace,audit_path)
    output_lock=_output(workspace,output_lock)
    audit_before=safety._stamp(audit_path);audit_digest=sha256(audit_path);audit=_json(audit_path)
    if (audit.get("schema_version")!=1 or audit.get("audit_type")!="read_only_local_ssvep_v5_point_in_time"
            or audit.get("inventory_complete") is not True or audit.get("options",{}).get("full_integrity") is not True):
        raise SSVEPReferenceV5Error("complete full-integrity local audit required")
    if type(audit.get("root")) is not str or not Path(audit["root"]).is_absolute():
        raise SSVEPReferenceV5Error("audit root must be explicitly absolute")
    root=_guard_chain(Path(audit["root"]))
    if not root.is_absolute() or not root.is_dir():raise SSVEPReferenceV5Error("audit source root must be an actual directory")
    root=root.resolve();_output(workspace,output_lock,root)
    evidence=_evidence(workspace)
    inventory=audit.get("inventory")
    if type(inventory) is not list:raise SSVEPReferenceV5Error("audit inventory missing")
    rows={}
    for row in inventory:
        if type(row) is not dict or type(row.get("relative_path")) is not str:raise SSVEPReferenceV5Error("malformed audit inventory")
        relative=row["relative_path"]
        if relative in rows:raise SSVEPReferenceV5Error("duplicate audit file identity")
        rows[relative]=row
    sources=[]
    changed=set(audit.get("changes",{}).get("changed_files",[]))|set(audit.get("changes",{}).get("new_files",[]))|set(audit.get("changes",{}).get("missing_files",[]))
    for name,sid,split in _subjects():
        spec=SPECS[name];relative=spec["relative_template"].format(subject_id=sid);row=rows.get(relative)
        if row is None:raise SSVEPReferenceV5Error(f"required audited subject absent: {relative}")
        path=_relative_source(root,relative)
        if Path(os.path.abspath(row.get("path","")))!=path or row.get("dataset")!=name:raise SSVEPReferenceV5Error("audit root/path/dataset namespace mismatch")
        stamps=[row.get(key) for key in ("stat_initial","stat_before","stat_after","stat_final")]
        if (relative in changed or row.get("changed") is not False or row.get("status") not in ("header_only","ready_stable_matrix")
                or row.get("errors")!=[] or any(type(x) is not dict for x in stamps) or any(x!=stamps[0] for x in stamps)
                or stamps[0].get("size",0)<=0 or not stat.S_ISREG(stamps[0].get("mode",0))):
            raise SSVEPReferenceV5Error("subject audit is changed/inprogress/error or lacks stable stat evidence")
        if safety._stamp(path)!=stamps[0]:raise SSVEPReferenceV5Error("source changed since audit; refreeze from fresh audit")
        if row.get("magic",{}).get("type")!="matlab_v5" or row.get("type")!="matlab_v5":raise SSVEPReferenceV5Error("MATLAB v5 true magic required")
        _sha(row.get("sha256"))
        mat=row.get("mat",{})
        if mat.get("header_reader")!="scipy.io.whosmat_metadata_only" or type(mat.get("variables")) is not list:raise SSVEPReferenceV5Error("complete MATLAB header required")
        target=[v for v in mat["variables"] if v.get("name")==spec["variable"]]
        if len(target)!=1 or target[0].get("shape")!=spec["shape"] or target[0].get("matlab_class") not in safety.NUMERIC_MAT_CLASSES:
            raise SSVEPReferenceV5Error("required subject variable shape/type mismatch")
        sources.append({"dataset":name,"subject_id":sid,"group_key":f"{name}:subject_{sid:03d}","split":split,
            "relative_path":relative,"path":str(path),"sha256":row["sha256"],"audit_stat":deepcopy(stamps[0]),
            "magic":"matlab_v5","variable":spec["variable"],"shape":list(spec["shape"]),
            "matlab_class":target[0]["matlab_class"],"expected_trials":_trial_count(name,spec)})
    lock={"schema_version":1,"reference_version":VERSION,"stage":"reference_identity_freeze","status":"formal",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),"workspace_root":str(workspace),"source_root":str(root),
        "audit":{"relative_path":audit_path.relative_to(workspace).as_posix(),"sha256":audit_digest},"evidence":evidence,
        "dataset_specs":deepcopy(SPECS),"partitions":deepcopy(PARTITIONS),"sources":sources,
        "split_expected_trials":{split:int(sum(r["expected_trials"] for r in sources if r["split"]==split)) for split in PARTITIONS},
        "total_expected_trials":int(sum(r["expected_trials"] for r in sources)),
        "within_dataset_subject_disjoint":True,"cross_dataset_participant_identity_verified":False,
        "subject_namespace":"dataset:subject_ID; same number across datasets is not participant identity proof",
        "raw_reference_is_clean_ground_truth":False,"dry_wet_is_clean_noisy_pair":False,
        "runtime":{"enabled":False,"production_validated":False},"model_metrics_computed":False,
        "train_eligibility":{"scope":"task_labels_native_reference_only","supervised_clean_ground_truth":False,"production_acceptance":False},
        "freeze_source_access":"lstat/path safety only; source SHA/header copied from bound audit, no source byte/waveform reads",
        "holdout_waveform_loaded":False}
    lock["payload_sha256"]=_digest(lock)
    validate_lock(lock,workspace)
    if safety._stamp(audit_path)!=audit_before or sha256(audit_path)!=audit_digest:raise SSVEPReferenceV5Error("audit changed during freeze")
    _publish_json(output_lock,lock)
    return lock

def validate_lock(lock,workspace=None):
    workspace=Path(workspace or WORKSPACE).resolve()
    if type(lock) is not dict or not _plain(lock):raise SSVEPReferenceV5Error("lock must be finite plain JSON")
    expected_keys={"schema_version","reference_version","stage","status","created_at_utc","workspace_root","source_root",
        "audit","evidence","dataset_specs","partitions","sources","split_expected_trials","total_expected_trials",
        "within_dataset_subject_disjoint","cross_dataset_participant_identity_verified","subject_namespace",
        "raw_reference_is_clean_ground_truth","dry_wet_is_clean_noisy_pair","runtime","model_metrics_computed",
        "train_eligibility","freeze_source_access","holdout_waveform_loaded","payload_sha256"}
    if set(lock)!=expected_keys:raise SSVEPReferenceV5Error("unexpected/incomplete lock schema")
    if (type(lock["schema_version"]) is not int or lock["schema_version"]!=1 or lock["reference_version"]!=VERSION
            or lock["stage"]!="reference_identity_freeze" or lock["status"]!="formal"):
        raise SSVEPReferenceV5Error("lock version/stage/formal status mismatch")
    _sha(lock["payload_sha256"])
    if _digest({k:v for k,v in lock.items() if k!="payload_sha256"})!=lock["payload_sha256"]:
        raise SSVEPReferenceV5Error("lock payload SHA256 mismatch")
    if lock["workspace_root"]!=str(workspace):raise SSVEPReferenceV5Error("lock workspace mismatch")
    root=Path(lock["source_root"])
    if not root.is_absolute() or str(root.resolve())!=str(root):raise SSVEPReferenceV5Error("source_root must be bound absolute canonical path")
    _guard_chain(root)
    if lock["partitions"]!=PARTITIONS or lock["dataset_specs"]!=SPECS:
        raise SSVEPReferenceV5Error("fixed preregistered subject groups/specs differ; overlap or metric-driven regrouping refused")
    if (lock["within_dataset_subject_disjoint"] is not True or lock["cross_dataset_participant_identity_verified"] is not False
            or lock["raw_reference_is_clean_ground_truth"] is not False or lock["dry_wet_is_clean_noisy_pair"] is not False
            or lock["runtime"]!={"enabled":False,"production_validated":False} or lock["model_metrics_computed"] is not False
            or lock["holdout_waveform_loaded"] is not False
            or lock["train_eligibility"]!={"scope":"task_labels_native_reference_only","supervised_clean_ground_truth":False,"production_acceptance":False}):
        raise SSVEPReferenceV5Error("reference safety scope/ground-truth/production claim mismatch")
    if set(lock["evidence"])!=set(EVIDENCE_PATHS):raise SSVEPReferenceV5Error("all author evidence bindings required")
    for role,entry in lock["evidence"].items():
        if set(entry)!={"relative_path","sha256"} or entry["relative_path"]!=EVIDENCE_PATHS[role]:raise SSVEPReferenceV5Error("evidence path mismatch")
        _sha(entry["sha256"])
    if set(lock["audit"])!={"relative_path","sha256"}:raise SSVEPReferenceV5Error("audit binding schema mismatch")
    _inside(workspace,workspace/lock["audit"]["relative_path"]);_sha(lock["audit"]["sha256"])
    expected=list(_subjects());sources=lock["sources"]
    if type(sources) is not list or len(sources)!=20:raise SSVEPReferenceV5Error("20 uniquely namespaced subjects required")
    triples=[]
    for row in sources:
        if set(row)!={"dataset","subject_id","group_key","split","relative_path","path","sha256","audit_stat","magic","variable","shape","matlab_class","expected_trials"}:
            raise SSVEPReferenceV5Error("source identity schema mismatch")
        name,sid,split=row["dataset"],row["subject_id"],row["split"]
        if name not in SPECS or type(sid) is not int or split not in PARTITIONS:raise SSVEPReferenceV5Error("invalid subject group")
        spec=SPECS[name];relative=spec["relative_template"].format(subject_id=sid)
        if (sid not in PARTITIONS[split] or row["relative_path"]!=relative or row["path"]!=str(root/relative)
                or row["group_key"]!=f"{name}:subject_{sid:03d}" or row["variable"]!=spec["variable"]
                or row["shape"]!=spec["shape"] or row["magic"]!="matlab_v5" or row["matlab_class"] not in safety.NUMERIC_MAT_CLASSES
                or type(row["expected_trials"]) is not int or row["expected_trials"]!=_trial_count(name,spec)):
            raise SSVEPReferenceV5Error("source labels/split/path/shape differ from fixed contract")
        _sha(row["sha256"])
        info=row["audit_stat"]
        if (set(info)!={"size","mtime_ns","ctime_ns","device","inode","mode","file_attributes"}
                or any(type(v) is not int for v in info.values()) or info["size"]<=0
                or not stat.S_ISREG(info["mode"]) or info["file_attributes"]&0x400):raise SSVEPReferenceV5Error("source stable stat schema mismatch")
        triples.append((name,sid,split))
    if triples!=expected or len({row["group_key"] for row in sources})!=20:raise SSVEPReferenceV5Error("subject ordering/overlap/coverage mismatch")
    counts={split:int(sum(r["expected_trials"] for r in sources if r["split"]==split)) for split in PARTITIONS}
    if lock["split_expected_trials"]!=counts or lock["total_expected_trials"]!=sum(counts.values()):raise SSVEPReferenceV5Error("expected trial counts disagree with shapes")
    return root


def _verify_metadata_bindings(lock,workspace):
    entries={"audit":lock["audit"],**lock["evidence"]}
    for role,entry in entries.items():
        path=_inside(workspace,workspace/entry["relative_path"])
        if not path.is_file() or sha256(path)!=entry["sha256"]:raise SSVEPReferenceV5Error(f"bound audit/evidence SHA256 changed: {role}")


def _stable_identity(root,row):
    path=_relative_source(root,row["relative_path"])
    before=safety._stamp(path)
    if before!=row["audit_stat"]:raise SSVEPReferenceV5Error("train/validation source stat changed since freeze")
    if sha256(path)!=row["sha256"]:raise SSVEPReferenceV5Error("train/validation source SHA256 mismatch")
    if safety._stamp(path)!=before:raise SSVEPReferenceV5Error("source changed/inprogress while hashing")
    return path,before


def subject_to_trials(name,array):
    """Native array -> contiguous float64 trial,time,channel + integer labels."""
    if name not in SPECS:raise SSVEPReferenceV5Error("unknown dataset")
    spec=SPECS[name]
    if (not isinstance(array,np.ndarray) or list(array.shape)!=spec["shape"] or array.dtype.kind not in "biuf"
            or not np.isfinite(array).all()):raise SSVEPReferenceV5Error("native source must be exact-shaped finite real numeric array, never object/pickle")
    if array.dtype.kind in "iu" and array.dtype.itemsize==8:
        if np.any(array>2**53) or (array.dtype.kind=="i" and np.any(array<-(2**53))):raise SSVEPReferenceV5Error("integer precision cannot be preserved by float64")
    if name=="Nakanishi2015":
        targets,channels,samples,blocks=array.shape
        reference=np.ascontiguousarray(array.transpose(0,3,2,1).reshape(targets*blocks,samples,channels),dtype=np.float64)
        target=np.repeat(np.arange(targets,dtype=np.int64),blocks)
        block=np.tile(np.arange(blocks,dtype=np.int64),targets)
        electrode=np.full(targets*blocks,-1,dtype=np.int64)
    else:
        channels,samples,electrodes,blocks,targets=array.shape
        reference=np.ascontiguousarray(array.transpose(4,3,2,1,0).reshape(targets*blocks*electrodes,samples,channels),dtype=np.float64)
        target=np.repeat(np.arange(targets,dtype=np.int64),blocks*electrodes)
        block=np.tile(np.repeat(np.arange(blocks,dtype=np.int64),electrodes),targets)
        electrode=np.tile(np.arange(electrodes,dtype=np.int64),targets*blocks)
    members={"reference":reference,"trial_index":np.arange(len(reference),dtype=np.int64),"target_id":target,
        "block_id":block,"electrode_id":electrode,"frequency_hz":np.asarray(spec["frequency_order_hz"],dtype=np.float64)[target]}
    if spec["phase_pi_multiples"] is not None:members["phase_pi_multiple"]=np.asarray(spec["phase_pi_multiples"],dtype=np.float64)[target]
    return members


def _array_sha(array):
    digest=hashlib.sha256()
    digest.update(json.dumps({"shape":list(array.shape),"dtype":array.dtype.str},sort_keys=True).encode())
    raw=memoryview(array).cast("B")
    for start in range(0,len(raw),safety.CHUNK):digest.update(raw[start:start+safety.CHUNK])
    return digest.hexdigest()


def _materialize_subject(root,row,memory_bytes):
    name=row["dataset"];spec=SPECS[name]
    estimated=math.prod(spec["shape"])*8*4+64*safety.CHUNK
    if estimated>memory_bytes:raise SSVEPReferenceV5Error("subject exceeds conservative <=256MiB working-data budget")
    path,before=_stable_identity(root,row)
    magic=safety.identify_magic(path,before["size"],safety.ReadBudget(0))
    if magic["type"]!="matlab_v5":raise SSVEPReferenceV5Error("current subject magic is not MATLAB v5")
    header=whosmat(path)
    targets=[v for v in header if v[0]==spec["variable"]]
    if len(targets)!=1 or list(targets[0][1])!=spec["shape"] or targets[0][2]!=row["matlab_class"]:
        raise SSVEPReferenceV5Error("current selected-variable header mismatch")
    # IMPORTANT: other Nakanishi variables (including large train/ans) and all
    # holdout files are never selected for numeric loading.
    loaded=loadmat(path,variable_names=[spec["variable"]],verify_compressed_data_integrity=True)
    if set(loaded)-{spec["variable"],"__header__","__version__","__globals__"}:
        raise SSVEPReferenceV5Error("reader unexpectedly selected non-reference variables")
    array=loaded.get(spec["variable"])
    members=subject_to_trials(name,array)
    del loaded,array
    if safety._stamp(path)!=before or sha256(path)!=row["sha256"] or safety._stamp(path)!=before:
        raise SSVEPReferenceV5Error("subject changed/inprogress while loading; output refused")
    if len(members["reference"])!=row["expected_trials"]:raise SSVEPReferenceV5Error("materialized subject trial count mismatch")
    return members


def _publish_subject(output,members,manifest):
    output.parent.mkdir(parents=True,exist_ok=True);_guard_chain(output.parent)
    fd,name=tempfile.mkstemp(prefix=output.name+".",suffix=".pending",dir=output.parent);os.close(fd);temporary=Path(name)
    try:
        encoded=np.asarray(json.dumps(manifest,ensure_ascii=False,sort_keys=True,allow_nan=False))
        with temporary.open("wb") as handle:np.savez(handle,**members,manifest_json=encoded)
        os.link(temporary,output)
    finally:temporary.unlink(missing_ok=True)
    sidecar={**manifest,"npz_sha256":sha256(output),"npz_relative_path":output.name}
    _publish_json(Path(str(output)+".manifest.json"),sidecar)
    return sidecar


def build_reference(lock_path,output_dir,workspace=None,memory_budget_mb=256):
    workspace=Path(workspace or WORKSPACE).resolve()
    if type(memory_budget_mb) is not int or not 1<=memory_budget_mb<=MEMORY_MAX_MB:raise SSVEPReferenceV5Error("working memory budget must be 1..256 MiB")
    lock_path=_inside(workspace,lock_path);lock_file_sha=sha256(lock_path);lock=_json(lock_path,4*1024*1024)
    root=validate_lock(lock,workspace);output=_output(workspace,output_dir,root)
    _verify_metadata_bindings(lock,workspace)
    selected=[row for row in lock["sources"] if row["split"] in ("train","validation")]
    # Metadata/hash preflight only on subjects that will be loaded. Holdout
    # waveforms/headers/current source hashes are deliberately NOT accessed.
    for row in selected:_stable_identity(root,row)
    if sha256(lock_path)!=lock_file_sha:raise SSVEPReferenceV5Error("lock changed during preflight")
    output.parent.mkdir(parents=True,exist_ok=True);_guard_chain(output.parent)
    output.mkdir(exist_ok=False)
    subjects=[]
    try:
        for row in selected:
            members=_materialize_subject(root,row,memory_budget_mb*safety.CHUNK)
            spec=SPECS[row["dataset"]]
            subject_manifest={"schema_version":1,"reference_version":VERSION,"status":"formal",
                "dataset":row["dataset"],"subject_id":row["subject_id"],"group_key":row["group_key"],"split":row["split"],
                "source":deepcopy(row),"source_lock_file_sha256":lock_file_sha,"source_lock_payload_sha256":lock["payload_sha256"],
                "reference_shape":list(members["reference"].shape),"reference_dtype":"float64","reference_sha256":_array_sha(members["reference"]),
                "native_sample_rate_hz":spec["sample_rate_hz"],"onset_sample_zero_based":spec["onset_sample_zero_based"],
                "channel_order":spec["channel_order"],"native_units":spec["units"],"source_axes":spec["source_axes"],
                "output_axes":["trial","time","channel"],"trial_order":spec["trial_order"],"target_id_base":0,"block_id_base":0,
                "electrode_id_base":0,"electrode_unknown_id":-1,"electrode_order":spec["electrode_order"],
                "frequency_order_hz":spec["frequency_order_hz"],"phase_pi_multiples":spec["phase_pi_multiples"],
                "materialized_trials":int(len(members["reference"])),"npz_members":sorted(list(members)+["manifest_json"]),
                "raw_reference_is_clean_ground_truth":False,"dry_wet_is_clean_noisy_pair":False,"runtime_enabled":False,
                "signal_processing":{"filtered":False,"resampled":False,"normalized":False,"interpolated":False,"cropped":False},
                "train_eligibility_scope":"task_labels_native_reference_only; not supervised ground truth or production acceptance"}
            destination=output/row["dataset"]/f"subject_{row['subject_id']:03d}.npz"
            artifact=_publish_subject(destination,members,subject_manifest)
            subjects.append({"dataset":row["dataset"],"subject_id":row["subject_id"],"split":row["split"],
                "relative_path":destination.relative_to(output).as_posix(),"npz_sha256":artifact["npz_sha256"],
                "manifest_sha256":sha256(Path(str(destination)+".manifest.json")),"trials":subject_manifest["materialized_trials"]})
            del members
        _verify_metadata_bindings(lock,workspace)
        if sha256(lock_path)!=lock_file_sha:raise SSVEPReferenceV5Error("lock changed during build")
        counts={split:int(sum(s["trials"] for s in subjects if s["split"]==split)) for split in ("train","validation")}
        if any(counts[s]!=lock["split_expected_trials"][s] for s in counts):raise SSVEPReferenceV5Error("actual build count differs from shape-derived count")
        manifest={"schema_version":1,"reference_version":VERSION,"status":"formal","build_completed":True,
            "created_at_utc":datetime.now(timezone.utc).isoformat(),"source_lock_file_sha256":lock_file_sha,
            "source_lock_payload_sha256":lock["payload_sha256"],"source_root":str(root),"evidence":deepcopy(lock["evidence"]),
            "selected_splits":["train","validation"],"materialized_subjects":subjects,"materialized_trial_counts":counts,
            "total_materialized_trials":sum(counts.values()),"shape_derived_all_trial_counts":lock["split_expected_trials"],
            "holdout_identities":[deepcopy(r) for r in lock["sources"] if r["split"]=="holdout"],
            "holdout_access":"identities/header copied from lock only; no current holdout source byte/hash/header/waveform reads",
            "holdout_waveform_loaded":False,"holdout_materialized":False,"model_metrics_computed":False,
            "within_dataset_subject_disjoint":True,"cross_dataset_participant_identity_verified":False,
            "raw_reference_is_clean_ground_truth":False,"dry_wet_is_clean_noisy_pair":False,
            "runtime":{"enabled":False,"production_validated":False},"train_eligibility":deepcopy(lock["train_eligibility"]),
            "memory_budget_mb":memory_budget_mb,"memory_scope":"conservative per-subject working-data allocation budget, not OS-enforced RSS cap"}
        _publish_json(output/"manifest.json",manifest)
        return manifest
    except Exception as error:
        # Unique reserved directory is never presented as a completed build.
        # Preserve partial artifacts for audit; no recursive delete/move and no
        # overwrite/retry into this directory is permitted.
        failure={"schema_version":1,"reference_version":VERSION,"build_completed":False,"status":"incomplete_refused",
            "error":str(error),"completed_subject_artifact_count":len(subjects),"holdout_waveform_loaded":False,"model_metrics_computed":False}
        try:_publish_json(output/"build_failure.json",failure)
        except OSError:pass
        raise


__all__=["SSVEPReferenceV5Error","freeze_reference","validate_lock","build_reference","subject_to_trials"]
