"""Fixed v7 internal diagnostic: fresh synthetic seeds on known subjects5-6 only.

No torch import or binary read at import/check-only. Not independent validation.
Training50 + train-lock + checkpoint + training-report = diagnostic53 bindings;
a separate pre-execution code lock binds this evaluator's three new files.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).absolute().parents[1]
CONFIG_RELATIVE = "configs/learned_denoiser_v7_ssvep.json"
CONFIG_SHA256 = "ac578c9f2343510aeb7f6eeb843497b6e7a8320028fc409d8afd5cdfc660e243"
PREREG_RELATIVE = "analysis_outputs/ssvep_v7_training_preregistration_20261007.json"
PREREG_SHA256 = "2e8c2b123a9d292b2d37c36b9aa21020dcafbe97a769a50ab8fafb2e06be23cd"
TRAIN_LOCK = "analysis_outputs/ssvep_v7_training_lock_20261007.json"
EVAL_LOCK = "analysis_outputs/ssvep_v7_internal_evaluation_lock_20261007.json"
CODE_LOCK = "analysis_outputs/ssvep_v7_internal_evaluation_code_lock_20261007.json"
CHECKPOINT = "models/denoiser_v7_ssvep.pt"
TRAIN_REPORT = "analysis_outputs/ssvep_v7_training_20261007.json"
PREFIX = "analysis_outputs/ssvep_v7_internal_diagnostic_20261007"
CODE_PATHS = ("eeg_tools/ssvep_v7_internal_eval.py", "scripts/evaluate_ssvep_denoiser_v7_internal.py", "tests/test_ssvep_v7_internal_eval.py")
DATASETS = ("Nakanishi2015", "Wearable")
CONDITIONS = ("identity", "transient", "low_frequency_drift", "high_frequency_burst", "common_mode")
TRIALS = {"Nakanishi2015": 180, "Wearable": 240}
RATES = {"Nakanishi2015": 256, "Wearable": 250}
STARTS = {"Nakanishi2015": 38, "Wearable": 125}
FALSE_FLAGS = ("runtime_enabled", "production_validated", "outer_validation_waveforms_loaded", "holdout_waveforms_loaded", "outer_validation_used_for_training_or_selection")
METRICS = {"input_normalized_mse", "output_normalized_mse", "relative_reduction", "identity_normalized_change", "outcome", "protected_complex_energy_ratio", "protected_weighted_abs_phase_radians", "undefined_phase_bin_count"}
PROVENANCE = {"dataset", "subject_id", "trial_index", "condition", "seed", "window_samples", "window_start_sample", "sample_rate_hz"}


class DiagnosticError(ValueError):
    pass


def require(ok, text):
    if not ok:
        raise DiagnosticError(text)


def _c():
    return importlib.import_module("eeg_tools.ssvep_v6_contract")


def finite_tree(x):
    if x is None or type(x) in (str, bool, int):
        return True
    if type(x) is float:
        return math.isfinite(x)
    if type(x) is list:
        return all(finite_tree(v) for v in x)
    if type(x) is dict:
        return all(type(k) is str and finite_tree(v) for k, v in x.items())
    return False


def case_seed(dataset, sid, trial, condition, cfg):
    require(dataset in DATASETS and type(sid) is int and sid in (5, 6), "internal subject allowlist")
    require(type(trial) is int and 0 <= trial < TRIALS[dataset] and condition in CONDITIONS, "trial/condition allowlist")
    return cfg["seeds"]["diagnostic"] + DATASETS.index(dataset)*10000000 + sid*100000 + trial*10 + CONDITIONS.index(condition)


def validate_cases(rows, cfg):
    require(type(rows) is list and len(rows) == 4200, "exact4200 cases required")
    seen, seeds = set(), set()
    undefined_cases = undefined_bins = identity_count = 0
    for r in rows:
        require(type(r) is dict and set(r) == METRICS | PROVENANCE and finite_tree(r), "case schema/finite mismatch")
        d, s, t, k = (r[f] for f in ("dataset", "subject_id", "trial_index", "condition"))
        seed = case_seed(d, s, t, k, cfg)
        key = d, s, t, k
        require(key not in seen and seed not in seeds, "duplicatecase/seed")
        seen.add(key); seeds.add(seed)
        for field, value in (("seed", seed), ("window_samples", 256), ("window_start_sample", STARTS[d]), ("sample_rate_hz", RATES[d])):
            require(type(r[field]) is int and r[field] == value, "case provenance mismatch")
        a, b = r["input_normalized_mse"], r["output_normalized_mse"]
        require(type(a) is float and type(b) is float and a >= 0 and b >= 0, "case MSE invalid")
        reduction = (a-b)/a if a > 0 and k != "identity" else None
        require(r["relative_reduction"] == reduction, "case ratio mismatch")
        require(r["outcome"] == ("win" if b < a else "loss" if b > a else "tie"), "case outcome mismatch")
        bins = r["undefined_phase_bin_count"]
        require(type(bins) is int and bins >= 0, "invalid phase bins")
        complex_value, phase = r["protected_complex_energy_ratio"], r["protected_weighted_abs_phase_radians"]
        if k == "identity":
            require(a == 0 and r["identity_normalized_change"] == b, "identity consistency")
            for v in (complex_value, phase):
                require(v is None or (type(v) is float and v >= 0), "invalid phase/complex")
            require(phase is None or phase <= math.pi, "phase exceeds pi")
            require(bins == 0 or phase is None, "undefined phase must remain null")
            identity_count += 1
            undefined_cases += int(bins > 0 or complex_value is None or phase is None)
            undefined_bins += bins
        else:
            require(r["identity_normalized_change"] is None and complex_value is None and phase is None and bins == 0, "nonidentity phase must be null")
    expected = {(d,s,t,k) for d in DATASETS for s in (5,6) for t in range(TRIALS[d]) for k in CONDITIONS}
    require(seen == expected, "incomplete internal cohort")
    return {"complete_cohort": True, "case_count": len(rows), "reference_trial_count": 840, "phase_case_count": identity_count, "undefined_phase_case_count": undefined_cases, "undefined_phase_bin_count": undefined_bins, "phase_verifiable": identity_count == 840 and undefined_cases == 0}


def summarize(rows, *, mixed=False):
    require(type(mixed) is bool and rows, "nonempty summary and bool mixed required")
    require(mixed or len({r["dataset"] for r in rows}) == 1, "cross-domain pooled gate forbidden")
    result = {}
    for k in CONDITIONS:
        group = [r for r in rows if r["condition"] == k]
        if not group:
            continue
        a = statistics.fmean(r["input_normalized_mse"] for r in group)
        b = statistics.fmean(r["output_normalized_mse"] for r in group)
        def complete_mean(field):
            vals = [r[field] for r in group]
            return None if any(v is None for v in vals) else statistics.fmean(vals)
        result[k] = {"case_count": len(group), "mean_input_normalized_mse": a, "mean_output_normalized_mse": b, "ratio_of_mean_reduction": (a-b)/a if a > 0 and k != "identity" else None, "mean_identity_normalized_change": complete_mean("identity_normalized_change"), "mean_protected_complex_energy_ratio": complete_mean("protected_complex_energy_ratio"), "mean_protected_weighted_abs_phase_radians": complete_mean("protected_weighted_abs_phase_radians"), "undefined_phase_case_count": sum(int(r["undefined_phase_bin_count"] > 0 or (k == "identity" and (r["protected_complex_energy_ratio"] is None or r["protected_weighted_abs_phase_radians"] is None))) for r in group)}
    require(finite_tree(result), "nonfinite summary")
    return result


def gates(summary, cfg):
    thresholds = cfg["internal_research_gates"]
    require(thresholds["high_frequency_burst_mean_error_reduction_min"] == .05 and thresholds["other_corrupted_condition_mean_error_reduction_min"] == 0 and thresholds["identity_mean_normalized_change_max"] == .0005 and thresholds["identity_protected_band_complex_energy_ratio_max"] == 1e-6 and thresholds["identity_protected_band_weighted_abs_phase_radians_max"] == .02, "frozen gates drift")
    result = {}
    def cmp(k, field, threshold, comparison):
        section = summary.get(k, {})
        v = section.get(field)
        valid = type(v) in (int,float) and math.isfinite(v) and type(section.get("case_count")) is int and section["case_count"] > 0
        passed = valid and (v >= threshold if comparison == ">=" else v <= threshold)
        result[k+"."+field] = {"passed": bool(passed), "value": v, "threshold": threshold, "comparison": comparison}
    for k in CONDITIONS[1:]:
        cmp(k, "ratio_of_mean_reduction", .05 if k == "high_frequency_burst" else 0.0, ">=")
    for field, limit in (("mean_identity_normalized_change",.0005),("mean_protected_complex_energy_ratio",1e-6),("mean_protected_weighted_abs_phase_radians",.02)):
        cmp("identity", field, limit, "<=")
    undefined = summary.get("identity", {}).get("undefined_phase_case_count")
    result["identity.phase_defined"] = {"passed": type(undefined) is int and undefined == 0, "value": undefined, "threshold": 0, "comparison": "=="}
    return {"all_passed": all(x["passed"] for x in result.values()), "gates": result}


def summarize_diagnostic(rows, cfg):
    counts = validate_cases(rows,cfg)
    buckets = {k:defaultdict(list) for k in ("dataset","condition","dataset_condition","subject","subject_condition")}
    for r in rows:
        d,s,k = r["dataset"],r["subject_id"],r["condition"]
        for group,key in (("dataset",d),("condition",k),("dataset_condition",f"{d}/{k}"),("subject",f"{d}/{s}"),("subject_condition",f"{d}/{s}/{k}")):
            buckets[group][key].append(r)
    aggregates = {"global": summarize(rows,mixed=True)}
    for name,groups in buckets.items():
        aggregates[name] = {key:summarize(bucket,mixed=name=="condition") for key,bucket in sorted(groups.items())}
    domain_gates = {d:gates(aggregates["dataset"][d],cfg) for d in DATASETS}
    return {**counts,"aggregates":aggregates,"dataset_gates":domain_gates,"all_internal_gates_passed":all(g["all_passed"] for g in domain_gates.values()) and counts["phase_verifiable"]}


@dataclass(frozen=True)
class State:
    root: Path
    config: dict
    files: dict
    lock_sha256: str
    code_sha256: str
    deferred: tuple
    verified: dict
    stamps: dict
    check_only: bool

    def report(self):
        return {"stage":"v7_internal_diagnostic","check_only":self.check_only,"verified_hashes":self.verified,"deferred":list(self.deferred),"lock_sha256":self.lock_sha256,"code_lock_sha256":self.code_sha256,"waveforms_loaded_this_call":False,"checkpoint_loaded_this_call":False,"cuda_initialized_this_call":False}


def preflight(root=ROOT, *, check_only=True):
    require(type(check_only) is bool, "check_only bool required")
    c = _c(); root = c.safe_path(root,".")
    cfg=c.read_json(root,CONFIG_RELATIVE); pr=c.read_json(root,PREREG_RELATIVE)
    require(c.file_sha256(root,CONFIG_RELATIVE)==CONFIG_SHA256 and c.file_sha256(root,PREREG_RELATIVE)==PREREG_SHA256, "config/prereg hash mismatch")
    train=c.read_json(root,TRAIN_LOCK); lock=c.read_json(root,EVAL_LOCK); code=c.read_json(root,CODE_LOCK)
    expected=dict(pr["bound_files"]); expected[PREREG_RELATIVE]=PREREG_SHA256
    for p in pr["new_source_files"]:
        expected[p]=train.get("files",{}).get(p)
    require(len(expected)==50 and train.get("files")==expected and train.get("stage")=="frozen_before_v7_training_waveform_read", "training50 freeze mismatch")
    expected.update({TRAIN_LOCK:c.file_sha256(root,TRAIN_LOCK),CHECKPOINT:lock.get("checkpoint_sha256"),TRAIN_REPORT:lock.get("files",{}).get(TRAIN_REPORT)})
    require(len(expected)==53 and lock.get("files")==expected and lock.get("stage")=="frozen_before_v7_internal_diagnostic" and lock.get("training_performed") is True, "diagnostic53 freeze mismatch")
    require(lock.get("training_lock_sha256")==expected[TRAIN_LOCK], "training lock SHA mismatch")
    for item in (train,lock,code):
        require(item.get("schema_version")==1 and item.get("status")=="formal", "formal lock required")
        require(all(item.get("scope",{}).get(k) is False for k in FALSE_FLAGS), "unsafe/missing lock flags")
    require(code.get("stage")=="frozen_before_v7_internal_diagnostic_code_execution" and set(code.get("files",{}))==set(CODE_PATHS), "diagnostic code freeze mismatch")
    digests={**expected,**code["files"]}
    require(all(type(h) is str and len(h)==64 and all(x in '0123456789abcdef' for x in h) for h in digests.values()), "bad digest")
    stamps={}; verified={}; deferred=[]
    for p,h in digests.items():
        stamps[p]=c._stamp(root,p)
        defer=(p==CHECKPOINT and check_only) or (p.endswith('.npz') and (check_only or Path(p).name not in ('subject_005.npz','subject_006.npz')))
        if defer:deferred.append(p)
        else:
            verified[p]=c.file_sha256(root,p); require(verified[p]==h,"frozen file changed: "+p)
    return State(root,cfg,digests,c.file_sha256(root,EVAL_LOCK),c.file_sha256(root,CODE_LOCK),tuple(deferred),verified,stamps,check_only)


def verify_frozen(state):
    c=_c()
    require(type(state) is State,"state required")
    require(c.file_sha256(state.root,EVAL_LOCK)==state.lock_sha256 and c.file_sha256(state.root,CODE_LOCK)==state.code_sha256,"diagnostic lock changed")
    for p,h in state.files.items():
        require(c._stamp(state.root,p)==state.stamps[p],"file stat changed: "+p)
        if p not in state.deferred:require(c.file_sha256(state.root,p)==h==state.verified[p],"file bytes changed: "+p)


def output_paths(root,prefix=PREFIX,*,check_only=False):
    c=_c(); path=c.safe_path(root,prefix)
    require(path.parent==c.safe_path(root,'analysis_outputs') and path.name.startswith('ssvep_v7_internal_diagnostic') and path.suffix=='',"output prefix refused")
    suffix='_check_only' if check_only else ''
    outputs=tuple(path.relative_to(c.safe_path(root,'.')).as_posix()+suffix+ext for ext in ('.json','.md'))
    for p in outputs:require(not c.safe_path(root,p).exists(),"existing output refused")
    return outputs


def _model_digest(models):
    h=hashlib.sha256()
    for d in sorted(models):
        for name,tensor in sorted(models[d].state_dict().items()):
            h.update((d+'/'+name).encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def infer_cases(state,records,models,torch):
    require(type(state) is State and state.check_only is False,"full diagnostic preflight required")
    c=_c(); rows=[]
    expected={(d,s) for d in DATASETS for s in (5,6)}
    require(len(records)==4 and {(r.dataset,r.subject_id) for r in records}==expected,"only four internal5-6 records allowed")
    for record in records:
        require(record.sample_rate_hz==RATES[record.dataset] and record.signal.shape==(TRIALS[record.dataset],1114 if record.dataset==DATASETS[0] else 710,8),"record shape/rate")
    import numpy as np
    legacy=importlib.import_module('eeg_tools.ssvep_denoise_v5')
    with torch.inference_mode():
        for d in DATASETS:
            model=models[d]
            model.eval()
            cases=[]
            for record in sorted((r for r in records if r.dataset==d),key=lambda x:x.subject_id):
                for trial in range(TRIALS[d]):
                    ref=np.ascontiguousarray(record.signal[trial,STARTS[d]:STARTS[d]+256,:].T,dtype=np.float64)
                    for condition in CONDITIONS:
                        seed=case_seed(d,record.subject_id,trial,condition,state.config)
                        override=dict(state.config['corruption']); override.update(identity_probability=0.,combine_probability=0.,families=[condition])
                        if condition=='identity':noisy=ref.copy()
                        else:
                            noisy,identity,chosen=legacy.corrupt_window(ref.copy(),RATES[d],{'corruption':override},seed)
                            require(identity is False and chosen==[condition],"single fixed pollution violated")
                        norm,_,rms=c.normalize_window(noisy)
                        cases.append((ref,noisy,norm,rms,record.subject_id,trial,condition,seed))
            for start in range(0,len(cases),64):
                batch=cases[start:start+64]; x=torch.from_numpy(np.stack([v[2] for v in batch])).to(device='cuda:0',dtype=torch.float32)
                pred=model(x)
                require(pred.shape==x.shape and pred.dtype==torch.float32 and bool(torch.isfinite(pred).all()),"invalid predicted batch")
                predictions=pred.detach().cpu().numpy()
                for i,(ref,noisy,norm,rms,sid,trial,condition,seed) in enumerate(batch):
                    out=c.restore_window(noisy,norm,predictions[i],rms)
                    metrics=c.case_metrics(ref,noisy,out,identity=condition=='identity',sample_rate_hz=RATES[d])
                    rows.append({**metrics,'dataset':d,'subject_id':sid,'trial_index':trial,'condition':condition,'seed':seed,'window_samples':256,'window_start_sample':STARTS[d],'sample_rate_hz':RATES[d]})
    validate_cases(rows,state.config)
    return rows


def run_diagnostic(state):
    require(type(state) is State and not state.check_only,"cannot execute check-only")
    verify_frozen(state)
    runtime=importlib.import_module('eeg_tools.ssvep_v7_experiment')
    runtime.cublas_environment()  # Set before torch import; schema-check on CPU before CUDA.
    core=importlib.import_module('eeg_tools.ssvep_denoise_v7'); torch=core.torch
    checkpoint=torch.load(_c().safe_path(state.root,CHECKPOINT),map_location='cpu',weights_only=True)
    # Runner validator is the authoritative v7 checkpoint schema boundary.
    models=runtime.validate_checkpoint(checkpoint,state.config,state.files[TRAIN_LOCK])
    runtime.configure_training(state.config)
    models={d:m.to(device='cuda:0',dtype=torch.float32).eval() for d,m in models.items()}
    before=_model_digest(models)
    records=importlib.import_module('eeg_tools.ssvep_denoise_v5').load_reference_records(state.root,subjects=(5,6))
    verify_frozen(state)
    rows=infer_cases(state,records,models,torch)
    after=_model_digest(models); require(before==after,"model state changed during inference")
    verify_frozen(state)
    gpu={'device':'cuda:0','device_name':torch.cuda.get_device_name(0),'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'amp':False,'allow_tf32':False,'deterministic_algorithms':True}
    return {'schema_version':1,'status':'formal','model_version':'learned_denoise_v7_ssvep','created_at_utc':datetime.now(timezone.utc).isoformat(),'stage':'v7_fixed_internal_diagnostic','check_only':False,'inference_performed':True,'gpu':gpu,'diagnostic_seed':state.config['seeds']['diagnostic'],'diagnostic_epoch':0,'preflight':state.report(),'frozen':{'configuration_sha256':CONFIG_SHA256,'preregistration_sha256':PREREG_SHA256,'training_lock_sha256':state.files[TRAIN_LOCK],'checkpoint_sha256':state.files[CHECKPOINT],'training_report_sha256':state.files[TRAIN_REPORT],'evaluation_lock_sha256':state.lock_sha256,'code_lock_sha256':state.code_sha256},'scope':{'training_performed_this_diagnostic':False,'checkpoint_prior_training':True,'epoch_selection_performed':False,'parameter_selection_performed':False,'clean_ground_truth_used':False,'independent_acceptance':False,**dict.fromkeys(FALSE_FLAGS,False)},'deployment':{'status':'blocked','runtime_enabled':False,'production_validated':False},'limitations':['same subjects5-6 as internal selection; fresh synthetic seeds are not independent subject validation','recorded reference is not clean ground truth','fixed256-point rectangularFFT window; no Hann OLA/continuous stream/task decoding/clinical acceptance proof','global/condition aggregates descriptive dimensionless only; gates applied per-domain'],'model_state_sha256_before':before,'model_state_sha256_after':after,**summarize_diagnostic(rows,state.config),'cases':rows}


def markdown(report):
    text=['# SSVEP v7 internal diagnostic','','**INTERNAL ONLY; deployment blocked even if engineering gates pass.**','',f"Cases: {report.get('case_count',0)}; all internal gates passed: {report.get('all_internal_gates_passed',False)}",'','Fresh synthetic seeds on known development subjects5-6, not independent validation.','Recorded reference is not clean ground truth.','No runtime/production activation.','','## Per-domain gates','','```json',json.dumps(report.get('dataset_gates',{}),ensure_ascii=False,indent=2,allow_nan=False),'```','']
    return '\n'.join(text).encode('utf-8')


def evaluate(root=ROOT,*,check_only=False,prefix=PREFIX):
    paths=output_paths(root,prefix,check_only=check_only)
    state=preflight(root,check_only=check_only)
    if check_only:
        report={'schema_version':1,'status':'formal','check_only':True,'case_count':0,'phase_case_count':0,'all_internal_gates_passed':False,'runtime_enabled':False,'production_validated':False,'waveforms_loaded':False,'checkpoint_loaded':False,'cuda_initialized':False,'preflight':state.report()}
    else:report=run_diagnostic(state)
    require(finite_tree(report),'finite JSON required')
    verify_frozen(state)
    runtime=importlib.import_module('eeg_tools.ssvep_v7_experiment')
    runtime.publish_bundle(state.root,{paths[0]:(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'),paths[1]:markdown(report)})
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args(argv)
    try:
        result=evaluate(check_only=args.check_only)
    except (ValueError,OSError) as exc:
        print('v7 diagnostic refused: '+str(exc),file=sys.stderr); return 2
    print(json.dumps({k:result[k] for k in ('check_only','case_count','phase_case_count','all_internal_gates_passed')},allow_nan=False))
    return 0
