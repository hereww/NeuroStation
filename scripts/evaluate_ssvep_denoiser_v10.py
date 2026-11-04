"""Evaluate an already selected v10 checkpoint; never train or select a model.

Public subjects 7/8 test the neural stage against normalized NPZ surrogates.
Long synthetic and personal recordings separately expose the fixed notch stage.
A new analysis_outputs directory is mandatory; summary.json is written last.
Import and CLI validation do not initialize Torch, CUDA or read EEG waveforms.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PUBLIC_CONDITIONS = ("identity", "drift", "high_burst", "gaussian", "combined")
SYNTHETIC_CONDITIONS = ("identity", "line50", "drift.3", "gaussian", "combined")
FREQUENCIES = (10, 12, 15, 20)
BATCH_SIZE = 32
SEED = 20261013


def require(ok, message):
    if not ok:
        raise ValueError("v10 evaluation: " + message)


def validate_paths(output_directory, personal_sessions=(), *, root=ROOT):
    """Validate without creating anything; reject existing and aliased outputs."""
    from eeg_tools.workstation import v9_offline as b
    root = b._path(root)
    analysis = b._path(root / "analysis_outputs")
    output = b._path(output_directory, root)
    require(output != analysis and output.is_relative_to(analysis),
            "--output-directory must be a new directory beneath analysis_outputs")
    require(not output.exists(), "output already exists; refusing to overwrite or resume")
    sources = []
    for value in personal_sessions:
        source = b._path(value, root)
        require(source.is_dir(), "personal source directory does not exist")
        require(not output.is_relative_to(source) and not source.is_relative_to(output),
                "output and personal source must not overlap")
        require(all(not source.is_relative_to(old) and not old.is_relative_to(source)
                    for old in sources), "duplicate/overlapping personal source directories")
        require(all(b._path(source / name).is_file() for name in b.NATIVE_FILES),
                "personal source requires session.json, channel_config.json, raw_brainflow.tsv")
        sources.append(source)
    return output, sources


def _write_json(path, value):
    # Exclusive create also rejects a race with another run. Strict finite JSON.
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_signal(signal):
    import numpy as np
    require(isinstance(signal, np.ndarray) and signal.ndim == 2
            and signal.shape[1] == 8 and len(signal) >= 512
            and signal.dtype.kind == "f", "expected floating native-250Hz signal [T>=512,8]")
    require(bool(np.isfinite(signal).all()), "nonfinite signal; no interpolation")
    require(bool((np.std(signal, axis=0) > 1e-8).all()), "flat channel; no replacement")
    return signal


def validate_processed(raw, output, notch, receipt):
    import numpy as np
    validate_signal(raw)
    for name, array in (("v10", output), ("notch", notch)):
        require(isinstance(array, np.ndarray) and array.shape == raw.shape
                and array.dtype == np.float64 and bool(np.isfinite(array).all()),
                name + " must be finite float64 [T,8]")
    require(isinstance(receipt, dict), "missing processing receipt")
    for key, expected in (("window_samples", 512), ("hop_samples", 128),
                          ("processed_samples", len(raw))):
        require(type(receipt.get(key)) is int and receipt[key] == expected,
                "invalid receipt " + key)
    expected_windows = (len(raw) - 512) // 128 + 1 + int((len(raw) - 512) % 128 != 0)
    require(type(receipt.get("window_count")) is int
            and receipt["window_count"] == expected_windows, "invalid receipt window_count")
    ratio = receipt.get("max_window_protected_complex_energy_ratio")
    require(type(ratio) in (int, float) and math.isfinite(ratio) and ratio >= 0,
            "invalid receipt max_window_protected_complex_energy_ratio")
    # Keep the runtime receipt verbatim. Its maximum alone cannot establish how
    # many windows had defined protected energy, or whole-record preservation.
    return receipt


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def window_metrics(reference, corrupted, prediction, protected_ratio, protected_defined):
    """Per-window normalized-space errors; undefined protected ratios are null."""
    import numpy as np
    arrays = [np.asarray(x) for x in (reference, corrupted, prediction)]
    require(arrays[0].ndim == 3 and arrays[0].shape[0] > 0
            and arrays[0].shape[1:] == (8, 512), "reference must be [N,8,512]")
    require(all(a.shape == arrays[0].shape and np.isfinite(a).all() for a in arrays),
            "neural stage output/input shape or finite mismatch")
    ref, noisy, pred = (a.astype(np.float64) for a in arrays)
    ratios, defined = np.asarray(protected_ratio), np.asarray(protected_defined)
    require(ratios.shape == defined.shape == (len(ref),) and defined.dtype == np.bool_,
            "protected auxiliary values must be [N] ratios and boolean defined mask")
    require(bool(np.isfinite(ratios[defined]).all() and (ratios[defined] >= 0).all()),
            "invalid defined protected ratios")
    incoming = np.mean((noisy - ref) ** 2, axis=(1, 2))
    outgoing = np.mean((pred - ref) ** 2, axis=(1, 2))
    change = np.mean((pred - noisy) ** 2, axis=(1, 2))
    rows = [{"input_mse": float(a), "output_mse": float(z),
             "output_to_input_mse_ratio": _ratio(z, a),
             "normalized_change_from_input": float(c),
             "protected_energy_defined": bool(d),
             "protected_complex_energy_ratio": float(r) if d else None}
            for a, z, c, r, d in zip(incoming, outgoing, change, ratios, defined)]
    defined_values = ratios[defined]
    return {
        "window_count": len(rows), "input_mse": float(incoming.mean()),
        "output_mse": float(outgoing.mean()),
        "output_to_input_mse_ratio": _ratio(outgoing.mean(), incoming.mean()),
        "mean_normalized_change_from_input": float(change.mean()),
        "max_normalized_change_from_input": float(change.max()),
        "protected_defined_window_count": int(defined.sum()),
        "protected_undefined_window_count": int((~defined).sum()),
        "max_defined_protected_complex_energy_ratio": (
            float(defined_values.max()) if len(defined_values) else None),
        "windows": rows,
    }


def evaluate_public(model, cfg, device, *, root=ROOT):
    """One evaluation of selected weights; no outer checkpoint/epoch search."""
    import numpy as np
    import torch
    from eeg_tools import ssvep_v10_data as data
    require(cfg["data"]["evaluation_subjects"] == [7, 8], "evaluation subjects must be 7/8")
    require(cfg["runtime"]["enabled"] is False
            and cfg["runtime"]["production_validated"] is False, "runtime must remain disabled")
    reference, identities = data.load_reference_windows([7, 8], root=root)
    require(isinstance(reference, np.ndarray) and reference.dtype == np.float32
            and reference.ndim == 3 and reference.shape[1:] == (8, 512)
            and len(reference) > 0 and bool(np.isfinite(reference).all()),
            "expected normalized float32 public reference [N,8,512]")
    require(len(identities) == len(reference)
            and {row["subject_id"] for row in identities} == {7, 8}, "public identities mismatch")
    model.eval()
    conditions = {}
    for index, condition in enumerate(PUBLIC_CONDITIONS):
        seed = SEED + 100 + index
        noisy = data.corrupt(reference.copy(), condition, seed, cfg)
        require(noisy.dtype == np.float32 and noisy.shape == reference.shape
                and np.isfinite(noisy).all(), "invalid public corruption output")
        if condition == "identity":
            require(np.array_equal(noisy, reference), "identity corruption changed reference")
        outputs, ratios, defined = [], [], []
        for start in range(0, len(noisy), BATCH_SIZE):
            tensor = torch.as_tensor(noisy[start:start + BATCH_SIZE], device=device,
                                     dtype=torch.float32)
            with torch.inference_mode():
                prediction, aux = model.forward_with_aux(tensor)
            require(prediction.shape == tensor.shape and prediction.dtype == torch.float32
                    and prediction.device == tensor.device, "invalid neural output contract")
            outputs.append(prediction.detach().cpu().numpy())
            ratios.append(aux["protected_complex_energy_ratio"].detach().cpu().numpy())
            defined.append(aux["protected_energy_defined"].detach().cpu().numpy())
        metrics = window_metrics(reference, noisy, np.concatenate(outputs),
                                 np.concatenate(ratios), np.concatenate(defined))
        for row, identity in zip(metrics["windows"], identities):
            row["identity"] = dict(identity)
        conditions[condition] = {"corruption_seed": seed, **metrics}
    return {
        "stage": "neural_only_direct_forward_no_runtime_notch_or_OLA",
        "subjects": [7, 8], "reference_shape": list(reference.shape),
        "reference_dtype": str(reference.dtype), "conditions": conditions,
        "mse_definition": "mean squared error in NPZ normalized space across 8x512 values",
        "identity_mse_ratio_definition": "null when input MSE is zero; use output MSE/change",
        "protection_definition": "input-output complex energy ratio only where auxiliary defined=true",
        "reference_is_clean_truth": False,
        "evaluation_is_pristine_project_holdout": False,
        "all_public_subjects_previously_available_in_project": True,
        "public_reference_cca_performed": False,
        "checkpoint_selection_performed": False,
        "parameter_search_performed": False,
    }


def centered_errors(clean, tracks):
    import numpy as np
    validate_signal(clean)
    result = {}
    for name, signal in tracks.items():
        require(signal.shape == clean.shape and np.isfinite(signal).all(), "invalid comparison track")
        regions = {}
        for region, section in (("full", slice(None)), ("interior", slice(512, -512))):
            truth, value = clean[section], signal[section]
            require(len(truth) > 0, "empty interior metric region")
            truth_centered = truth - truth.mean(axis=0, keepdims=True)
            value_centered = value - value.mean(axis=0, keepdims=True)
            mse = float(np.mean((value_centered - truth_centered) ** 2))
            energy = float(np.mean(truth_centered ** 2))
            regions[region] = {"samples": len(truth), "clean_centered_mse": mse,
                               "clean_centered_normalized_mse": _ratio(mse, energy),
                               "mean_offset_mse": float(np.mean((value.mean(axis=0)
                                                                 - truth.mean(axis=0)) ** 2))}
        result[name] = regions
    return result


def synthetic_case(frequency, condition, seed):
    """Known 4096-sample waveform; fixed harmonics, background and DC offsets."""
    import numpy as np
    require(frequency in FREQUENCIES and condition in SYNTHETIC_CONDITIONS,
            "unknown synthetic frequency/condition")
    time = np.arange(4096, dtype=np.float64)[:, None] / 250.0
    phases = np.arange(8, dtype=np.float64)[None, :] * 0.27
    clean = (np.sin(2 * np.pi * frequency * time + phases)
             + 0.3 * np.sin(2 * np.pi * 2 * frequency * time + phases / 2)
             + 0.2 * np.sin(2 * np.pi * 7 * time + phases)
             + 100 + 5 * np.arange(8)[None, :])
    rng = np.random.default_rng(seed)
    noises = {
        "line50": 2.0 * np.sin(2 * np.pi * 50 * time + phases),
        "drift.3": 1.0 * np.sin(2 * np.pi * 0.3 * time + phases),
        "gaussian": 0.5 * rng.standard_normal(clean.shape),
    }
    noise = np.zeros_like(clean) if condition == "identity" else (
        sum(noises.values()) if condition == "combined" else noises[condition])
    return clean, np.asarray(clean + noise, dtype=np.float64)


def _decode(signal, targets):
    from eeg_tools.personal_ssvep_decoder import decode_window
    try:
        return decode_window(signal.T, 250.0, targets)
    except (ValueError, ArithmeticError) as exc:
        return {"available": False, "reason": str(exc), "target_id": None,
                "production_validated": False}


def evaluate_synthetic(model):
    import numpy as np
    from eeg_tools import ssvep_v10_runtime as runtime
    from eeg_tools.workstation import v9_offline as b
    targets = [(str(hz), float(hz)) for hz in FREQUENCIES]
    cases = []
    for index, (hz, condition) in enumerate((hz, condition) for hz in FREQUENCIES
                                           for condition in SYNTHETIC_CONDITIONS):
        seed = SEED + 200 + index
        clean, raw = synthetic_case(hz, condition, seed)
        before = raw.copy()
        output, notch, receipt = runtime.process_recording(raw.copy(), model, batch_size=BATCH_SIZE)
        validate_processed(raw, output, notch, receipt)
        require(np.array_equal(raw, before), "synthetic input mutated")
        tracks = {"raw": raw, "notch": notch, "traditional": b._traditional(raw), "v10": output}
        # Fixed interior region avoids making known-frequency truth an input to decoding.
        predictions = {name: _decode(value[512:-512], targets) for name, value in tracks.items()}
        for prediction in predictions.values():
            prediction["matches_known_frequency"] = bool(prediction.get("available")
                                                          and prediction["target_id"] == str(hz))
        errors = centered_errors(clean, tracks)
        contributions = {}
        for region in ("full", "interior"):
            mse = {name: metric[region]["clean_centered_mse"] for name, metric in errors.items()}
            contributions[region] = {
                "fixed_notch_mse_reduction_from_raw": mse["raw"] - mse["notch"],
                "neural_plus_OLA_mse_reduction_from_notch": mse["notch"] - mse["v10"],
                "v10_to_notch_mse_ratio": _ratio(mse["v10"], mse["notch"]),
            }
        cases.append({"frequency_hz": hz, "condition": condition, "seed": seed,
                      "errors": errors, "contributions": contributions,
                      "cca": predictions, "receipt": receipt})
    return {"sample_rate_hz": 250, "channel_count": 8, "samples_per_case": 4096,
            "case_count": len(cases), "cases": cases,
            "reference_definition": "known fundamental + second harmonic + 7Hz background + per-channel DC",
            "noise_definition": "line50 amplitude2; drift0.3Hz amplitude1; Gaussian std0.5; combined sum",
            "centering_definition": "each track and clean centered separately per channel in each region; DC error separate",
            "cca_candidates_hz": list(FREQUENCIES), "cca_interval": [512, 3584],
            "cca_preprocessing": "shared decoder independently applies its own notch and bandpass to every track",
            "biological_clean_truth_claim": False,
            "receipt_protection_limitation": "runtime maximum alone does not verify defined-energy coverage or full-record protection"}


def evaluate_personal(source, output_directory, model):
    import numpy as np
    from eeg_tools import ssvep_v10_runtime as runtime
    from eeg_tools.workstation import v9_offline as b
    from scripts.diagnose_neurostation_v9_output import parse_raw, validate_metadata
    before = b._source_hashes(source)
    session = json.loads((source / "session.json").read_text(encoding="utf-8-sig"))
    channels = json.loads((source / "channel_config.json").read_text(encoding="utf-8-sig"))
    validate_metadata(session, channels)
    raw_bytes = (source / "raw_brainflow.tsv").read_bytes()
    require(hashlib.sha256(raw_bytes).hexdigest() == before["raw_brainflow.tsv"],
            "raw source changed before parsing")
    raw, header = parse_raw(raw_bytes)
    validate_signal(raw)
    count = session.get("recorded_samples_per_channel")
    require(count is None or (type(count) is int and count == len(raw)), "recorded sample count mismatch")
    denoised, notch, receipt = runtime.process_recording(raw.copy(), model, batch_size=BATCH_SIZE)
    validate_processed(raw, denoised, notch, receipt)
    tracks = {"raw": raw, "notch": notch, "v10": denoised}
    targets, intervals, metadata_reason = b._classification_inputs(source, raw)
    rows = []
    for interval in intervals:
        predictions = {}
        for name, signal in tracks.items():
            reason = metadata_reason or interval.get("unavailable_reason")
            predictions[name] = ({"available": False, "reason": reason, "target_id": None}
                                 if reason else _decode(signal[interval["window_start_sample_index"]:
                                                                 interval["offset_sample_index"]], targets))
        # Labels enter only after all three predictions have been computed.
        presented = interval.get("presented_target_id", "")
        for prediction in predictions.values():
            prediction["matches_presented_target"] = (
                bool(prediction.get("available") and prediction.get("target_id") == presented)
                if presented else None)
        rows.append({**interval, "predictions": predictions})
    summaries = {}
    for name in tracks:
        valid = [row["predictions"][name] for row in rows if row["predictions"][name].get("available")]
        comparable = [p for p in valid if p["matches_presented_target"] is not None]
        summaries[name] = {"trial_count": len(valid), "recorded_trial_count": len(rows),
                           "presented_comparable_count": len(comparable),
                           "presented_match_count": sum(p["matches_presented_target"] for p in comparable)}
    after = b._source_hashes(source)
    require(before == after, "personal source SHA changed during analysis")
    output_directory.mkdir(exist_ok=False)
    artifacts = {}
    for name, signal in (("original.npy", raw), ("notch.npy", notch), ("v10.npy", denoised)):
        with (output_directory / name).open("xb") as stream:
            np.save(stream, signal, allow_pickle=False)
        artifacts[name] = {"sha256": _sha(output_directory / name),
                           "shape": list(signal.shape), "dtype": str(signal.dtype)}
    final_hashes = b._source_hashes(source)
    require(before == final_hashes, "personal source SHA changed before completion")
    report = {"status": "formal", "completion_status": "complete", "source_session": str(source),
              "source_sha256_before": before, "source_sha256_after": final_hashes,
              "source_hashes_match": True, "source_header": header, "receipt": receipt,
              "targets": targets, "tracks": summaries, "trials": rows,
              "classification_unavailable_reason": metadata_reason, "artifacts": artifacts,
              "source_domain": "personal_Cyton", "sample_rate_hz": 250,
              "independent_personal_validation": False, "clean_ground_truth_available": False,
              "physical_electrode_positions_verified": False, "production_validated": False,
              "runtime_enabled": False, "task_labels_enter_forward": False,
              "cca_available_is_detection": False, "cca_confidence_is_probability": False,
              "cca_preprocessing": "shared decoder independently applies its own notch and bandpass to every track",
              "presented_matches_are_gaze_accuracy": False,
              "denoising_efficacy_validated": False,
              "receipt_protection_limitation": "runtime maximum alone does not verify defined-energy coverage or full-record protection"}
    _write_json(output_directory / "report.json", report)
    return report


def research_gate(public, synthetic, cfg):
    """Fixed research rules only; neither checkpoint selection nor deployment."""
    limits = cfg["engineering_gates"]
    conditions = public["conditions"]
    require(set(conditions) == set(PUBLIC_CONDITIONS), "incomplete public conditions")
    identity = conditions["identity"]["mean_normalized_change_from_input"]
    corrupted = [conditions[name] for name in PUBLIC_CONDITIONS if name != "identity"]
    n = sum(item["window_count"] for item in corrupted)
    incoming = sum(item["input_mse"] * item["window_count"] for item in corrupted) / n
    outgoing = sum(item["output_mse"] * item["window_count"] for item in corrupted) / n
    reduction = 1 - outgoing / incoming if incoming > 0 else None
    defined = sum(item["protected_defined_window_count"] for item in conditions.values())
    undefined = sum(item["protected_undefined_window_count"] for item in conditions.values())
    values = [item["max_defined_protected_complex_energy_ratio"] for item in conditions.values()
              if item["max_defined_protected_complex_energy_ratio"] is not None]
    max_ratio = max(values) if values else None
    nonworse = all(item["output_mse"] <= item["input_mse"] for item in corrupted)
    checks = {
        "identity_change": identity <= limits["identity_mean_normalized_change_max"],
        "corrupted_error_reduction": reduction is not None and reduction >= limits["corrupted_mean_error_reduction_min"],
        "defined_window_protection": defined > 0 and max_ratio is not None
                                    and max_ratio <= limits["protected_window_complex_energy_ratio_max"],
        "all_corruption_conditions_nonworse": nonworse or not limits["all_conditions_nonworse"],
        "synthetic_frequency_preservation": len(synthetic["cases"]) == 20
            and {(case["frequency_hz"], case["condition"]) for case in synthetic["cases"]}
                == {(hz, condition) for hz in FREQUENCIES for condition in SYNTHETIC_CONDITIONS}
            and all(case["cca"]["v10"].get("matches_known_frequency") is True for case in synthetic["cases"]),
    }
    return {"passed": all(checks.values()), "checks": checks, "limits": dict(limits),
            "identity_mean_normalized_change": identity,
            "corrupted_pooled_error_reduction": reduction,
            "max_defined_protected_complex_energy_ratio": max_ratio,
            "protected_defined_window_count": defined,
            "protected_undefined_window_count": undefined,
            "undefined_windows_protection_verified": False,
            "decision_scope": "research-only numerical gates plus known-synthetic four-frequency CCA preservation",
            "checkpoint_selection_performed": False, "runtime_enabled": False,
            "independent_personal_validation": False, "clean_ground_truth_available": False,
            "production_validated": False}


def evaluate(output_directory, personal_sessions=(), *, device="cuda:0", root=ROOT):
    require(device in ("cuda:0", "cpu"), "device must be cuda:0 or cpu")
    output, sources = validate_paths(output_directory, personal_sessions, root=root)
    from eeg_tools import ssvep_v10_data as data
    from eeg_tools import ssvep_v10_runtime as runtime
    from eeg_tools.workstation import v9_offline as b
    cfg = data.load_config(root=root)
    # Runtime verifies the frozen manifest before returning the one selected model.
    # Do not inspect evaluation waveforms or choose epochs before this completes.
    model, metadata = runtime.load_model(root=root, device=device)
    manifest_sha256 = _sha(Path(root) / runtime.MANIFEST)
    require(_sha(Path(root) / runtime.MODEL) == metadata.get("checkpoint_sha256"),
            "selected checkpoint changed after manifest validation")
    require(metadata.get("device") == device and type(metadata.get("selected_epoch")) is int
            and metadata["selected_epoch"] >= 1 and metadata.get("checkpoint_sha256")
            and metadata.get("runtime_enabled") is False
            and metadata.get("production_validated") is False, "invalid selected checkpoint metadata")
    output.mkdir(parents=True, exist_ok=False)
    (output / "public").mkdir()
    public = evaluate_public(model, cfg, device, root=root)
    _write_json(output / "public" / "report.json", public)
    (output / "synthetic").mkdir()
    synthetic = evaluate_synthetic(model)
    _write_json(output / "synthetic" / "report.json", synthetic)
    personal = []
    for index, source in enumerate(sources):
        folder = output / ("personal_%02d" % index)
        report = evaluate_personal(source, folder, model)
        personal.append({"source_session": str(source), "report": folder.name + "/report.json",
                         "tracks": report["tracks"], "source_sha256": report["source_sha256_before"]})
    for item, source in zip(personal, sources):
        require(b._source_hashes(source) == item["source_sha256"], "personal source changed before summary")
    require(_sha(Path(root) / runtime.MODEL) == metadata["checkpoint_sha256"]
            and _sha(Path(root) / runtime.MANIFEST) == manifest_sha256
            and data.canonical_sha(data.load_config(root=root)) == data.canonical_sha(cfg),
            "selected checkpoint, manifest or config changed during evaluation")
    report_paths = ["public/report.json", "synthetic/report.json"] + [p["report"] for p in personal]
    summary = {
        "schema_version": 1, "status": "formal", "completion_status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": metadata, "checkpoint_manifest_sha256": manifest_sha256,
        "config": cfg, "config_sha256": data.canonical_sha(cfg),
        "trained_model_evaluated": True, "training_performed": False,
        "checkpoint_selection_performed": False, "outer_selection_performed": False,
        "output_directory": str(output), "public_evaluation_subjects": [7, 8],
        "public": {"report": "public/report.json", "stage": public["stage"]},
        "synthetic": {"report": "synthetic/report.json", "case_count": synthetic["case_count"]},
        "personal": personal, "research_gate": research_gate(public, synthetic, cfg),
        "artifacts": {name: {"sha256": _sha(output / name)} for name in report_paths},
        "all_public_subjects_previously_available_in_project": True,
        "evaluation_is_pristine_project_holdout": False,
        "independent_personal_validation": False, "clean_ground_truth_available": False,
        "production_validated": False, "runtime_enabled": False,
        "limitations": [
            "Trained neural model and fixed 50Hz Q30 notch have separately reported contributions.",
            "Public neural-stage MSE compares line-conditioned recorded surrogates, not biological clean truth.",
            "Full-record synthesis includes notch, normalization, neural shrinkage and Hann overlap-add; neural-only guarantees do not extend automatically.",
            "Undefined protected-energy rows are excluded; a numeric zero is not proof of protection.",
            "Known synthetic references provide artificial truth only, not personal clean EEG truth.",
            "All public subjects were previously available in this project; 7/8 are evaluation data for this fixed experiment, not pristine project holdout.",
            "Personal presented-target matches are development proxies, not independent gaze accuracy.",
            "CCA always uses the shared decoder preprocessing and forced argmax; margin is not probability.",
            "Passing research gates never enables production runtime or validates a command.",
        ],
    }
    _write_json(output / "summary.json", summary)  # Overall completion marker last.
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--personal-session", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        # Reject paths before loading Torch, checkpoint, public data or personal EEG.
        validate_paths(args.output_directory, args.personal_session)
        result = evaluate(args.output_directory, args.personal_session, device=args.device)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"summary": str(Path(result["output_directory"]) / "summary.json"),
                      "research_gate_passed": result["research_gate"]["passed"],
                      "runtime_enabled": False, "production_validated": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
