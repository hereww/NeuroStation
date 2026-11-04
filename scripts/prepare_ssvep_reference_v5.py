"""Freeze subject identities from audit/evidence, then build native task references."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from eeg_tools.ssvep_reference_v5 import freeze_reference,build_reference,SSVEPReferenceV5Error


def make_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    freeze=commands.add_parser("freeze",help="audit/author metadata only, never load source waveforms")
    freeze.add_argument("--audit",type=Path,required=True)
    freeze.add_argument("--output-lock",type=Path,required=True)
    build=commands.add_parser("build",help="train/validation native trials only, holdout metadata only")
    build.add_argument("--lock",type=Path,required=True)
    build.add_argument("--output-dir",type=Path,required=True)
    build.add_argument("--memory-budget-mb",type=int,default=256)
    return parser


def main(argv=None):
    args=make_parser().parse_args(argv)
    try:
        if args.command=="freeze":
            lock=freeze_reference(args.audit,args.output_lock)
            result={"status":"formal_reference_identity_frozen","lock":str(args.output_lock.resolve()),
                "payload_sha256":lock["payload_sha256"],"subject_count":len(lock["sources"]),
                "expected_trial_counts":lock["split_expected_trials"],"source_waveforms_loaded":False}
        else:
            manifest=build_reference(args.lock,args.output_dir,memory_budget_mb=args.memory_budget_mb)
            result={"status":"formal_native_task_references_materialized","output_dir":str(args.output_dir.resolve()),
                "actual_materialized_trial_counts":manifest["materialized_trial_counts"],
                "total_materialized_trials":manifest["total_materialized_trials"],"holdout_waveform_loaded":False}
        result.update(raw_reference_is_clean_ground_truth=False,dry_wet_is_clean_noisy_pair=False,
            runtime_enabled=False,production_validated=False,model_metrics_computed=False,
            train_eligibility_scope="task_labels_reference_only")
        print(json.dumps(result,ensure_ascii=True,allow_nan=False,indent=2));return 0
    except (SSVEPReferenceV5Error,OSError,ValueError,KeyError,TypeError) as error:
        print(json.dumps({"status":"error","error":str(error)},ensure_ascii=True,allow_nan=False));return 2

if __name__=="__main__":raise SystemExit(main())
