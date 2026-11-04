"""Reproduce personal development outputs and known-reference synthetic checks.

All results are offline diagnostics. No labels enter the decoder forward call.
This does not train, accept acquisition quality, or validate physiological recovery.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def synthetic_check(device):
    import numpy as np
    from eeg_tools import ssvep_test_runtime as rt
    from eeg_tools.workstation.v9_offline import _traditional
    from eeg_tools.personal_ssvep_decoder import decode_window

    rt.preflight()
    models, gpu = rt.load_models(device=device)
    rng = np.random.default_rng(20261013)
    t = np.arange(4096) / 250.0
    targets = [(str(freq), float(freq)) for freq in (10, 12, 15, 20)]
    rows = []
    for frequency in (10, 12, 15, 20):
        clean = np.column_stack([np.sin(2*np.pi*frequency*t+ch/8)
                + 0.25*np.sin(2*np.pi*2*frequency*t+ch/7)
                + 0.2*np.sin(2*np.pi*7*t+ch/5) for ch in range(8)])
        noise = {
            "identity": np.zeros_like(clean),
            "line_50_hz": np.column_stack([2*np.sin(2*np.pi*50*t+ch/6) for ch in range(8)]),
            "drift_0_3_hz": np.column_stack([2*np.sin(2*np.pi*0.3*t+ch/6) for ch in range(8)]),
            "gaussian": rng.standard_normal(clean.shape)*0.5,
        }
        noise["combined"] = noise["line_50_hz"]+noise["drift_0_3_hz"]+noise["gaussian"]
        for condition, corruption in noise.items():
            original = clean + corruption + np.arange(8)[None, :]*100 + 500
            denoised, mask, receipt = rt.process_signal(original, np.ones(len(t), dtype=bool),
                models['Wearable'], sample_rate_hz=250, batch_size=64)
            tracks = {"raw": original, "traditional": _traditional(original), "v9": denoised}
            metrics = {}
            for name, signal in tracks.items():
                metrics[name] = {}
                for region, segment in (("full", slice(None)), ("interior", slice(250, -250))):
                    actual, reference = signal[segment], clean[segment]
                    actual = actual - actual.mean(axis=0)
                    reference = reference - reference.mean(axis=0)
                    mse = float(np.mean((actual-reference)**2))
                    energy = float(np.mean(reference**2))
                    metrics[name][region+"_mean_centered_mse"] = mse
                    metrics[name][region+"_normalized_mse"] = mse / energy
                score = decode_window(signal[250:-250].T, 250, targets)
                metrics[name]["cca_candidate"] = score['target_id']
                metrics[name]["cca_score_margin"] = score['confidence']
                metrics[name]["known_synthetic_frequency_match"] = score['target_id'] == str(frequency)
            rows.append({"frequency_hz": frequency, "condition": condition, "samples": len(t),
                "processed_samples": int(mask.sum()), "preserved_samples": int((~mask).sum()),
                "metrics": metrics, "max_added_boundary_jump_input_units": receipt['max_added_boundary_jump_input_units']})
    return {"status": "formal", "seed": 20261013, "gpu": gpu, "cases": rows,
        "definition": "known numerical sinusoid mixture with artificial corruption; arbitrary units; mean centered error",
        "identity_case": "zero artificial noise; measures waveform distortion",
        "training_performed": False, "personal_clean_eeg_truth_available": False,
        "biological_recovery_validated": False, "independent_validation": False,
        "parameter_search_performed_this_run": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', action='append', required=True, type=Path)
    parser.add_argument('--report-directory', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    args = parser.parse_args(argv)
    from eeg_tools.workstation import v9_offline as backend
    report_dir = backend._path(args.report_directory)
    backend._require(report_dir.is_relative_to(ROOT / 'analysis_outputs'), 'report directory outside analysis_outputs')
    backend._require(not (report_dir / 'summary.json').exists(), 'report already completed; choose a fresh report directory')
    sources = [backend._path(source) for source in args.session]
    backend._require(len({str(source).casefold() for source in sources}) == len(sources), 'duplicate source session')
    backend._require(all(not report_dir.is_relative_to(source) for source in sources), 'report overlaps source')
    report_dir.mkdir(parents=True, exist_ok=True)
    before = {str(source): backend._source_hashes(source) for source in sources}
    summaries = []
    for source in sources:
        output = backend.new_output_dir(source)
        backend.run_job(source, output, device=args.device, acknowledge_out_of_domain=True)
        result = backend.read_result(output, session_dir=source)
        decoder = result['personal_decoder']
        summaries.append({"session_name": source.name, "source_session": str(source),
            "output_directory": str(output), "result_sha256": backend._sha(output/'result.json'),
            "samples": sum(result['processing'][field] for field in ('processed_samples','preserved_samples')),
            "processing": {key: result['processing'][key] for key in ('processed_samples','preserved_samples')},
            "fft": {name: {key: result['comparison']['tracks'][name].get(key)
                    for key in ('trial_count','presented_match_count')} for name in ('raw','traditional','v9')},
            "cca": decoder['tracks'], "controls": decoder['controls'], "gpu": result['gpu'],
            "config_sha256": decoder['config_sha256']})
    print(json.dumps({'stage':'known_reference','message':'Checking known synthetic waveforms and artificial noise'}),flush=True)
    synthetic = synthetic_check(args.device)
    _write(report_dir/'synthetic_known_reference.json', synthetic)
    after = {str(source): backend._source_hashes(source) for source in sources}
    backend._require(before == after, 'source data changed')
    backend._preflight()
    verification = {"source_sha256_before": before, "source_sha256_after": after,
        "source_hashes_match": True, "source_file_count": sum(len(value) for value in before.values()),
        "raw_sources_modified": False, "frozen_preflight_passed": True,
        "bundle_receipts_verified": True, "synthetic_report_sha256": backend._sha(report_dir/'synthetic_known_reference.json')}
    _write(report_dir/'verification.json', verification)
    summary = {"schema_version":1, "status":"formal", "completion_status":"complete",
        "created_at_utc":datetime.now(timezone.utc).isoformat(), "sessions":summaries,
        "source_domain":"personal_Cyton", "production_validated":False,
        "development_data":True, "both_sessions_previously_explored":True,
        "independent_gaze_truth_available":False, "independent_subjects":False,
        "training_performed":False, "raw_sources_modified":False,
        "limitations":["Presented target matches are development proxy counts, not gaze accuracy.",
            "CCA is a fixed transparent decoder, not a trained denoising model.",
            "Forced candidates exist on noise; no-target rejection is uncalibrated.",
            "Synthetic numerical references cannot prove physiological EEG recovery.",
            "V9 remains a conservative unvalidated transfer branch; all original gates remain unchanged."],
        "verification_sha256":backend._sha(report_dir/'verification.json')}
    _write(report_dir/'summary.json', summary)
    print(json.dumps({'status':'complete','summary':str(report_dir/'summary.json')}),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
