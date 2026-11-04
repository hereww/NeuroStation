"""Explicit offline research processing of a native NeuroStation session with v12.

Requires acknowledgement of unvalidated personal-domain transfer. No training,
hardware access or default workstation/runtime enablement is performed. Source
files are read-only. report.json is published last as the completion marker;
partial arrays without that report do not constitute a completed analysis.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_FILES = ("session.json", "channel_config.json", "raw_brainflow.tsv")
VALIDATION_FLAGS = {
    "runtime_enabled": False,
    "production_validated": False,
    "source_domain_validated": False,
    "denoising_efficacy_validated": False,
    "presented_matches_are_gaze_accuracy": False,
}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def build_parser():
    """Argument parsing is stdlib-only; it never loads torch or probes CUDA."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--session", required=True, type=Path,
                        help="native NeuroStation Cyton session directory (read-only)")
    parser.add_argument("--output", required=True, type=Path,
                        help="NEW directory outside the source session; parent must exist")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--acknowledge-out-of-domain", action="store_true", required=True,
                        help="acknowledge explicit offline research in an unvalidated source domain")
    return parser


def run_analysis(session_dir, output_dir, *, device="cpu", acknowledge_out_of_domain=False):
    """Process one session; only an exclusively created output directory is writable."""
    started = time.perf_counter()
    _require(acknowledge_out_of_domain is True,
             "--acknowledge-out-of-domain is required for unvalidated personal_Cyton transfer")
    _require(device in ("cpu", "cuda:0"), "device must be cpu or cuda:0")

    # These path and hash helpers are stdlib-only. Reject unsafe/existing outputs
    # before importing scientific dependencies or attempting to load a model.
    from eeg_tools.workstation import v9_offline as b

    output_dir = b._path(output_dir, root=Path.cwd())
    _require(not os.path.lexists(output_dir), "output directory already exists; refusing to overwrite")
    _require(output_dir.parent.is_dir(), "output parent directory must already exist")
    session_dir = b._path(session_dir, root=Path.cwd())
    _require(session_dir.is_dir(), "--session must be a native NeuroStation directory")
    _require(not output_dir.is_relative_to(session_dir)
             and not session_dir.is_relative_to(output_dir), "source/output directories must not overlap")
    paths = {name: b._path(session_dir / name) for name in SOURCE_FILES}
    _require(all(path.is_file() for path in paths.values()),
             "native session requires session.json, channel_config.json and raw_brainflow.tsv")
    # Include every source-session file, not only the three files consumed below.
    before = b._source_hashes(session_dir)
    payloads = {name: path.read_bytes() for name, path in paths.items()}
    _require(all(hashlib.sha256(data).hexdigest() == before[name]
                 for name, data in payloads.items()), "source SHA256 changed while reading")

    import numpy as np
    from scripts import diagnose_neurostation_v9_output as adapter
    from eeg_tools import ssvep_v12_runtime as runtime

    session = json.loads(payloads["session.json"].decode("utf-8-sig"))
    channels = json.loads(payloads["channel_config.json"].decode("utf-8-sig"))
    adapter.validate_metadata(session, channels)
    raw, header = adapter.parse_raw(payloads["raw_brainflow.tsv"])
    count = session.get("recorded_samples_per_channel")
    if count is not None:
        _require(type(count) is int and count == len(raw),
                 "session.json: recorded_samples_per_channel disagrees with raw rows")
    _require(raw.dtype.kind == "f" and raw.ndim == 2 and raw.shape[1] == 8
             and len(raw) >= 512 and np.isfinite(raw).all(),
             "v12 requires finite floating native 250 Hz [T>=512,8]; no sample repair or resampling")
    _require(np.all(np.std(raw, axis=0) > 1e-8), "flat channel; no channel replacement")

    # Traditional filtering is an independent, fixed reference. The notch array
    # below comes from v12's own processing API, preserving its exact contract.
    traditional = b._traditional(raw)
    model, model_metadata = runtime.load_model(ROOT, device=device)
    v12, notch, processing = runtime.process_recording(raw, model)
    arrays = {"raw.npy": raw, "notch.npy": notch,
              "traditional.npy": traditional, "v12.npy": v12}
    for name, value in arrays.items():
        _require(isinstance(value, np.ndarray) and value.dtype.kind == "f"
                 and value.shape == raw.shape and np.isfinite(value).all(),
                 "invalid output array: " + name)
    _require(b._source_hashes(session_dir) == before, "source SHA256 changed during processing")

    report = {
        "schema_version": 1, "status": "formal", "completion_status": "complete",
        "usage_scope": "explicit_offline_research_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_session": str(session_dir), "output_directory": str(output_dir),
        "source_domain": "personal_Cyton", "out_of_domain_acknowledged": True,
        "sampling_rate_hz": 250, "source_channel_labels": list(adapter.CHANNEL_LABELS),
        "configured_channel_order": channels["channel_order"],
        "extracted_eeg_columns": list(adapter.EEG_COLUMNS), "source_raw_header": header,
        "input_shape": list(raw.shape), "input_dtype": str(raw.dtype),
        "requested_device": device, "model": model_metadata, "processing": processing,
        "tracks": {
            "raw": {"file": "raw.npy", "method": "named EEG columns without numeric unit conversion"},
            "notch": {"file": "notch.npy", "method": "fixed 50 Hz Q30 offline notch returned by v12 runtime",
                      "learned": False},
            "traditional": {"file": "traditional.npy", "method": "v9_offline._traditional: fixed 50 Hz Q30 notch and fourth-order 1-45 Hz Butterworth, zero phase",
                            "learned": False},
            "v12": {"file": "v12.npy", "method": "v12 runtime fixed notch followed by learned residual and overlap-add",
                    "neural_comparison_reference": "notch.npy"},
        },
        "fixed_filter_benefits_attributed_to_network": False,
        "training_performed": False, "checkpoint_selection_performed": False,
        "inference_executed": True, "classifier_executed": False, "hardware_opened": False,
        "raw_sources_modified": False, "unit_conversion_performed": False,
        "resampling_performed": False, "default_workstation_modified": False,
        "historical_validation_status_rewritten": False, "acquisition_quality_accepted": False,
        "clean_ground_truth_available": False, "physical_electrode_positions_verified": False,
        "hardware_gain_and_unit_calibration_verified": False,
        **VALIDATION_FLAGS,
        "limitations": [
            "This formal artifact is for explicit offline research; completion does not validate signal quality or efficacy.",
            "Personal Cyton is an unvalidated source domain; matching the native 250 Hz rate does not validate transfer.",
            "Clean paired EEG ground truth, independently verified electrode positions and calibration are unavailable.",
            "Fixed notch/traditional filter effects must not be attributed to the neural network; compare v12 with notch separately.",
            "No gaze classification or improvement is established here; presented-target matches are not gaze accuracy.",
            "Offline noncausal filtering is not validated for real-time control or clinical use.",
        ],
    }

    output_dir.mkdir()  # Exclusive: never reuse or overwrite an existing bundle.
    artifacts = {}
    for name, value in arrays.items():
        path = output_dir / name
        with path.open("xb") as stream:
            np.save(stream, value, allow_pickle=False)
        artifacts[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                           "sha256": b._sha(path)}
    after = b._source_hashes(session_dir)
    _require(after == before, "source SHA256 changed while writing; output remains incomplete")
    report.update(artifacts=artifacts, source_sha256_before=before, source_sha256_after=after,
                  source_hashes_match=True, elapsed_seconds=time.perf_counter() - started)
    # Serialize first, then publish a complete JSON file atomically. Failures may
    # leave partial arrays, but must never leave a report claiming completion.
    data = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    pending = output_dir / ".report.json.tmp"
    try:
        with pending.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        pending.rename(output_dir / "report.json")
    except BaseException:
        pending.unlink(missing_ok=True)
        raise
    return report


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        report = run_analysis(args.session, args.output, device=args.device,
                              acknowledge_out_of_domain=args.acknowledge_out_of_domain)
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps({"status": "error", "complete": False,
                          "error": str(exc) or type(exc).__name__, **VALIDATION_FLAGS},
                         ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"status": "complete", "usage_scope": report["usage_scope"],
                      "output": report["output_directory"],
                      "report": str(Path(report["output_directory"]) / "report.json"),
                      "source_hashes_match": report["source_hashes_match"], **VALIDATION_FLAGS},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
