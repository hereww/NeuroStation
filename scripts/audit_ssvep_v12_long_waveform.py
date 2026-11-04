"""Audit frozen v12 on artificial long waveforms without training or EEG access.

Only summary/public/synthetic JSON and frozen model bindings are read. Numerical
reproduction is an integrity check, never a deployment gate. Scientific imports
are deferred until paths, evaluation hashes and model bindings have passed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_EVALUATION = "analysis_outputs/v12_evaluation_20261013"
FREQUENCIES = (10, 12, 15, 20)
CONDITIONS = ("identity", "line50", "drift.3", "gaussian", "combined")
TRACKS = ("raw", "notch", "traditional", "v12")
REGIONS = {"full": (0, 4096), "interior": (512, 3584)}
# Fixed numerical reproduction tolerances, not scientific acceptance thresholds.
MSE_RTOL = 1e-5
MSE_ATOL = 1e-9
PHASE_AMPLITUDE_FLOOR = 1e-12
MODEL = "models/denoiser_v12_personal.pt"
MANIFEST = "models/denoiser_v12_personal.manifest.json"
CONFIG = "configs/learned_denoiser_v12_personal.json"
LOCK = "analysis_outputs/ssvep_v12_training_lock_20261013.json"
TRAINING_REPORT = "analysis_outputs/ssvep_v12_training_20261013.json"
RUNTIME = "eeg_tools/ssvep_v12_runtime.py"


def require(condition, message):
    if not condition:
        raise ValueError("v12 long waveform audit: " + message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("v12 long waveform audit: nonfinite JSON constant: " + value)


def _read_json(path):
    data = Path(path).read_bytes()
    value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=_unique,
                       parse_constant=_invalid_constant)
    require(isinstance(value, dict), "JSON object required: " + str(path))
    return value, hashlib.sha256(data).hexdigest()


def _canonical_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    """Serialize before exclusive creation; never replace an existing artifact."""
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def validate_paths(evaluation_directory, output, *, root=ROOT):
    """Read-only stdlib preflight, including Windows alias/reparse protection."""
    from eeg_tools.workstation.v9_offline import _path
    root = _path(root)
    evaluation = _path(evaluation_directory, root)
    destination = _path(output, root)
    analysis = _path(root / "analysis_outputs")
    require(evaluation.is_dir(), "evaluation directory missing")
    require(destination != analysis and destination.is_relative_to(analysis),
            "output must be a new directory under analysis_outputs")
    require(not destination.exists(), "output already exists")
    require(not destination.is_relative_to(evaluation)
            and not evaluation.is_relative_to(destination), "evaluation/output overlap")
    for name in ("summary.json", "public.json", "synthetic.json"):
        require(_path(evaluation / name).is_file(), "evaluation file missing: " + name)
    return evaluation, destination


def load_evaluation(evaluation_directory):
    """Verify exact bytes against summary before interpreting public/synthetic."""
    directory = Path(evaluation_directory)
    summary, summary_sha = _read_json(directory / "summary.json")
    require(summary.get("status") == "formal"
            and summary.get("completion_status") == "complete", "incomplete evaluation")
    documents = {"summary": summary}
    hashes = {str(directory / "summary.json"): summary_sha}
    for name in ("public", "synthetic"):
        filename = name + ".json"
        expected = summary.get("artifacts", {}).get(filename, {}).get("sha256")
        require(isinstance(expected, str) and len(expected) == 64,
                "missing summary SHA256: " + filename)
        require(summary.get(name, {}).get("report") == filename,
                "unexpected evaluation report path: " + name)
        value, actual = _read_json(directory / filename)
        require(actual == expected, "SHA256 mismatch: " + filename)
        documents[name] = value
        hashes[str(directory / filename)] = actual
    return documents, hashes


def validate_cases(documents):
    summary, public, synthetic = (documents[name] for name in ("summary", "public", "synthetic"))
    require(summary.get("runtime_enabled") is False
            and summary.get("production_validated") is False
            and summary.get("independent_personal_validation") is False,
            "evaluation must retain research-only flags")
    require(public.get("reference_is_clean_truth") is False,
            "public recorded surrogate must not claim clean EEG truth")
    require(synthetic.get("known_clean_is_artificial_only") is True,
            "synthetic truth must be explicitly artificial")
    require(synthetic.get("case_count") == summary.get("synthetic", {}).get("case_count") == 20,
            "exactly 20 stored cases required")
    require(synthetic.get("frequencies") == list(FREQUENCIES)
            and synthetic.get("conditions") == list(CONDITIONS), "synthetic case grid changed")
    rows = synthetic.get("cases")
    require(isinstance(rows, list) and len(rows) == 20, "incomplete synthetic cases")
    indexed = {}
    for row in rows:
        frequency, condition = row.get("frequency_hz"), row.get("condition")
        require(type(frequency) in (int, float) and frequency in FREQUENCIES
                and condition in CONDITIONS, "unknown synthetic case")
        key = (frequency, condition)
        require(key not in indexed, "duplicate synthetic case")
        require(type(row.get("seed")) is int and row["seed"] >= 0, "invalid exact case seed")
        require(set(row.get("errors", {})) == set(TRACKS), "missing stored comparison track")
        for name in TRACKS:
            saved = row["errors"][name].get("interior", {})
            value = saved.get("clean_centered_mse")
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    "invalid stored interior centered MSE")
            require(saved.get("samples") == 3072, "stored interior must contain 3072 samples")
        indexed[key] = row
    require(set(indexed) == {(f, c) for f in FREQUENCIES for c in CONDITIONS},
            "missing synthetic case")
    return [indexed[(f, c)] for f in FREQUENCIES for c in CONDITIONS]


def verify_frozen_bindings(summary, *, root=ROOT):
    """Bind the evaluated checkpoint/epoch/config without loading Torch."""
    from eeg_tools.workstation.v9_offline import _path
    root = Path(root)
    manifest_path = _path(root / MANIFEST)
    manifest, manifest_sha = _read_json(manifest_path)
    config, config_file_sha = _read_json(_path(root / CONFIG))
    require(manifest.get("status") == "formal"
            and manifest.get("model_version") == "learned_denoise_v12_personal",
            "invalid frozen manifest")
    require(manifest.get("runtime_enabled") is False
            and manifest.get("production_validated") is False, "model must remain disabled")
    require(config.get("runtime", {}).get("enabled") is False
            and config.get("runtime", {}).get("production_validated") is False,
            "config must remain disabled")
    config_sha = _canonical_sha(config)
    require(config_sha == manifest.get("config_sha256") == summary.get("config_sha256"),
            "evaluated/frozen config SHA256 mismatch")
    evaluated_model = summary.get("model", {})
    require(evaluated_model.get("checkpoint_sha256") == manifest.get("checkpoint_sha256")
            and evaluated_model.get("selected_epoch") == manifest.get("selected_epoch")
            and evaluated_model.get("model_version") == manifest.get("model_version"),
            "evaluated checkpoint/epoch does not match frozen manifest")
    hashes = {str(manifest_path): manifest_sha, str(_path(root / CONFIG)): config_file_sha}
    fields = {MODEL: "checkpoint_sha256", LOCK: "training_lock_sha256",
              TRAINING_REPORT: "training_report_sha256"}
    for name in (MODEL, LOCK, TRAINING_REPORT, RUNTIME):
        path = _path(root / name)
        actual = sha256(path)
        require(actual == manifest.get("files", {}).get(name), "frozen file SHA256 mismatch: " + name)
        if name in fields:
            require(actual == manifest.get(fields[name]), "manifest SHA256 mismatch: " + name)
        hashes[str(path)] = actual
    require(manifest.get("files", {}).get(CONFIG) == config_file_sha,
            "config file fingerprint mismatch")
    # Bind the actual existing generator/filter implementations used by this audit.
    for name in ("scripts/evaluate_ssvep_denoiser_v10.py",
                 "eeg_tools/workstation/v9_offline.py", "eeg_tools/ssvep_v10_data.py",
                 "eeg_tools/ssvep_denoise_v10.py"):
        path = _path(root / name)
        hashes[str(path)] = sha256(path)
    return manifest, hashes


def preflight(evaluation_directory, output, device="cpu", *, root=ROOT):
    require(device in ("cpu", "cuda:0"), "choose explicit cpu or cuda:0")
    evaluation, destination = validate_paths(evaluation_directory, output, root=root)
    documents, hashes = load_evaluation(evaluation)
    rows = validate_cases(documents)
    manifest, model_hashes = verify_frozen_bindings(documents["summary"], root=root)
    return {"evaluation": evaluation, "output": destination, "documents": documents,
            "rows": rows, "manifest": manifest, "source_hashes": {**hashes, **model_hashes}}


def wrap_phase(angle):
    """Principal signed phase in [-pi, pi), including crossing the +/-pi cut."""
    import numpy as np
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def joint_sine_fit(signal, frequencies, *, sample_rate_hz=250.0, start_sample=0):
    """Joint multi-channel LS: intercept, centered time trend and all sin/cos pairs.

    Convention: A*sin(2*pi*f*t + phase). Time uses absolute sample indices even
    in the interior region. Fitting jointly prevents leakage among components.
    """
    import numpy as np
    signal = np.asarray(signal, dtype=np.float64)
    frequencies = tuple(float(f) for f in frequencies)
    require(signal.ndim == 2 and signal.shape[0] > 2 + 2 * len(frequencies)
            and signal.shape[1] > 0 and np.isfinite(signal).all(), "invalid LS signal")
    require(math.isfinite(sample_rate_hz) and sample_rate_hz > 0
            and len(set(frequencies)) == len(frequencies) and len(frequencies) > 0
            and all(0 < f < sample_rate_hz / 2 for f in frequencies), "invalid LS frequencies")
    time = (np.arange(signal.shape[0], dtype=np.float64) + start_sample) / sample_rate_hz
    origin = float(time.mean())
    columns = [np.ones_like(time), time - origin]
    for frequency in frequencies:
        columns.extend((np.sin(2 * np.pi * frequency * time),
                        np.cos(2 * np.pi * frequency * time)))
    design = np.column_stack(columns)
    coefficients, _, rank, singular = np.linalg.lstsq(design, signal, rcond=None)
    require(rank == design.shape[1], "rank-deficient LS design")
    components = []
    for index, frequency in enumerate(frequencies):
        sine, cosine = coefficients[2 + index * 2:4 + index * 2]
        amplitude = np.hypot(sine, cosine)
        phase = wrap_phase(np.arctan2(cosine, sine))
        components.append({"frequency_hz": frequency, "sine_coefficients": sine.tolist(),
                           "cosine_coefficients": cosine.tolist(), "amplitude": amplitude.tolist(),
                           "phase_radians": [float(p) if a > PHASE_AMPLITUDE_FLOOR else None
                                             for a, p in zip(amplitude, phase)]})
    return {"samples": len(signal), "start_sample": start_sample, "design_rank": int(rank),
            "condition_number": float(singular[0] / singular[-1]),
            "trend_time_origin_seconds": origin,
            "intercept_at_time_origin": coefficients[0].tolist(),
            "linear_trend_units_per_second": coefficients[1].tolist(),
            "residual_mse_per_channel": np.mean((signal - design @ coefficients) ** 2, axis=0).tolist(),
            "phase_convention": "A*sin(2*pi*f*t+phase); absolute sample time; wrap [-pi,pi)",
            "components": components}


def compare_fits(reference_fit, measured_fit):
    result = []
    for reference, measured in zip(reference_fit["components"], measured_fit["components"]):
        require(reference["frequency_hz"] == measured["frequency_hz"], "LS frequency mismatch")
        ratios, errors = [], []
        for a, b, p, q in zip(reference["amplitude"], measured["amplitude"],
                              reference["phase_radians"], measured["phase_radians"]):
            ratios.append(b / a if a > PHASE_AMPLITUDE_FLOOR else None)
            errors.append(float(wrap_phase(q - p)) if p is not None and q is not None else None)
        result.append({"frequency_hz": reference["frequency_hz"], "amplitude_ratio_per_channel": ratios,
                       "wrapped_phase_error_radians_per_channel": errors,
                       "wrapped_phase_error_degrees_per_channel": [math.degrees(x) if x is not None
                                                                  else None for x in errors]})
    return result


def waveform_errors(clean, signal):
    """Center independently per channel, retaining phase and mean-offset errors."""
    import numpy as np
    truth, value = (np.asarray(x, dtype=np.float64) for x in (clean, signal))
    require(truth.ndim == 2 and truth.shape == value.shape and truth.size > 0
            and np.isfinite(truth).all() and np.isfinite(value).all(), "invalid error arrays")
    tc = truth - truth.mean(axis=0, keepdims=True)
    vc = value - value.mean(axis=0, keepdims=True)
    channel_energy = np.mean(tc ** 2, axis=0)
    require(np.all(channel_energy > 0), "undefined NMSE: flat clean channel")
    channel_error = np.mean((vc - tc) ** 2, axis=0)
    offsets = value.mean(axis=0) - truth.mean(axis=0)
    mse, energy = float(channel_error.mean()), float(channel_energy.mean())
    return {"samples": len(truth), "scalar_count": int(truth.size),
            "clean_centered_mse": mse, "clean_centered_normalized_mse": mse / energy,
            "clean_centered_energy": energy, "mean_offset_mse": float(np.mean(offsets ** 2)),
            "uncentered_mse": float(np.mean((value - truth) ** 2)),
            "centered_mse_per_channel": channel_error.tolist(),
            "centered_nmse_per_channel": (channel_error / channel_energy).tolist(),
            "mean_offset_per_channel": offsets.tolist()}


def contribution(before, after):
    """Positive means less artificial-reference error; negative means worse."""
    initial, final = before["clean_centered_mse"], after["clean_centered_mse"]
    return {"centered_mse_reduction": initial - final,
            "centered_nmse_reduction": before["clean_centered_normalized_mse"]
                                       - after["clean_centered_normalized_mse"],
            "fraction_of_stage_input_error_removed": (initial - final) / initial if initial > 0 else None,
            "stage_input_error_is_zero": initial == 0}


def compare_stored_mse(stored_row, regions):
    checks = []
    for name in TRACKS:
        saved = stored_row["errors"][name]["interior"]["clean_centered_mse"]
        actual = regions["interior"]["tracks"][name]["errors"]["clean_centered_mse"]
        checks.append({"track": name, "stored_interior_centered_mse": saved,
                       "recomputed_interior_centered_mse": actual,
                       "absolute_difference": abs(saved - actual),
                       "allowed_absolute_difference": max(MSE_ATOL, MSE_RTOL * max(abs(saved), abs(actual))),
                       "matches": math.isclose(saved, actual, rel_tol=MSE_RTOL, abs_tol=MSE_ATOL)})
    return checks


def measure_case(stored_row, clean, tracks):
    import numpy as np
    require(np.asarray(clean).shape == (4096, 8) and set(tracks) == set(TRACKS),
            "exact 4096x8 four-track case required")
    require(all(np.asarray(v).shape == (4096, 8) for v in tracks.values()), "track shape mismatch")
    frequency = stored_row["frequency_hz"]
    regions = {}
    for region, (start, stop) in REGIONS.items():
        reference = clean[start:stop]
        reference_fit = joint_sine_fit(reference, (frequency, 2 * frequency, 7), start_sample=start)
        measurements = {}
        for name in TRACKS:
            signal = tracks[name][start:stop]
            fitted = joint_sine_fit(signal, (frequency, 2 * frequency, 7), start_sample=start)
            measurements[name] = {"errors": waveform_errors(reference, signal), "joint_fit": fitted,
                                  "comparison_to_artificial_clean": compare_fits(reference_fit, fitted)}
        errors = {name: measurements[name]["errors"] for name in TRACKS}
        regions[region] = {"start_sample": start, "stop_sample_exclusive": stop,
                           "artificial_clean_joint_fit": reference_fit, "tracks": measurements,
                           "stage_contributions": {
                               "fixed_notch_raw_to_notch": contribution(errors["raw"], errors["notch"]),
                               "neural_notch_to_v12": contribution(errors["notch"], errors["v12"]),
                               "traditional_notch_to_traditional": contribution(errors["notch"], errors["traditional"])}}
    return {"frequency_hz": frequency, "condition": stored_row["condition"], "seed": stored_row["seed"],
            "regions": regions, "stored_mse_checks": compare_stored_mse(stored_row, regions)}


def _aggregate_group(cases, region):
    count = sum(case["regions"][region]["tracks"]["raw"]["errors"]["scalar_count"] for case in cases)
    tracks = {}
    for name in TRACKS:
        rows = [case["regions"][region]["tracks"][name]["errors"] for case in cases]
        weighted = lambda key: math.fsum(row[key] * row["scalar_count"] for row in rows) / count
        mse, energy = weighted("clean_centered_mse"), weighted("clean_centered_energy")
        tracks[name] = {"clean_centered_mse": mse, "clean_centered_energy": energy,
                        "clean_centered_normalized_mse": mse / energy,
                        "uncentered_mse": weighted("uncentered_mse"),
                        "mean_offset_mse": weighted("mean_offset_mse"),
                        "mean_per_case_centered_nmse": math.fsum(row["clean_centered_normalized_mse"]
                                                                for row in rows) / len(rows)}
    differences = [case["regions"][region]["tracks"]["v12"]["errors"]["clean_centered_mse"]
                   - case["regions"][region]["tracks"]["traditional"]["errors"]["clean_centered_mse"]
                   for case in cases]
    neural, traditional = tracks["v12"], tracks["traditional"]
    return {"case_count": len(cases), "scalar_count": count, "tracks": tracks,
            "stage_contributions": {
                "fixed_notch_raw_to_notch": contribution(tracks["raw"], tracks["notch"]),
                "neural_notch_to_v12": contribution(tracks["notch"], tracks["v12"]),
                "traditional_notch_to_traditional": contribution(tracks["notch"], tracks["traditional"]),
                "combined_raw_to_v12": contribution(tracks["raw"], tracks["v12"])},
            "traditional_comparison": {
                "v12_minus_traditional_centered_nmse": neural["clean_centered_normalized_mse"]
                                                        - traditional["clean_centered_normalized_mse"],
                "v12_to_traditional_centered_mse_ratio": neural["clean_centered_mse"] / traditional["clean_centered_mse"]
                                                        if traditional["clean_centered_mse"] > 0 else None,
                "v12_lower_error_case_count": sum(x < 0 for x in differences),
                "traditional_lower_error_case_count": sum(x > 0 for x in differences),
                "exact_equal_error_case_count": sum(x == 0 for x in differences),
                "comparison_is_descriptive_only": True}}


def aggregate_measurements(cases):
    return {region: {"conditions": {condition: _aggregate_group(
                [case for case in cases if case["condition"] == condition], region) for condition in CONDITIONS},
                     "pooled_all_cases": _aggregate_group(cases, region),
                     "pooled_corrupted_cases": _aggregate_group(
                         [case for case in cases if case["condition"] != "identity"], region)}
            for region in REGIONS}


def plot_nmse(metrics, path):
    """One standard scientific PNG; linear axis keeps identity/zero values visible."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(11, 5.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    x = np.arange(len(CONDITIONS))
    for index, name in enumerate(TRACKS):
        values = [metrics["interior"]["conditions"][c]["tracks"][name]["clean_centered_normalized_mse"]
                  for c in CONDITIONS]
        axis.bar(x + (index - 1.5) * 0.2, values, 0.2, label=name)
    axis.set_xticks(x, CONDITIONS)
    axis.set_ylabel("Interior centered NMSE (pooled over four frequencies)")
    axis.set_title("Frozen v12: artificial signals only; not real EEG validation")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend()
    with Path(path).open("xb") as stream:
        figure.savefig(stream, format="png", dpi=160)


def audit(evaluation_directory=DEFAULT_EVALUATION, *, output, device="cpu", plot=False, root=ROOT):
    state = preflight(evaluation_directory, output, device, root=root)
    if plot:
        import importlib.util
        require(importlib.util.find_spec("matplotlib") is not None, "--plot requires installed matplotlib")
    # No Torch, NumPy or CUDA usage above this boundary.
    import numpy as np
    from eeg_tools import ssvep_v12_runtime as runtime
    from scripts.evaluate_ssvep_denoiser_v10 import synthetic_case
    from eeg_tools.workstation.v9_offline import _traditional
    model, metadata = runtime.load_model(root, device)
    require(metadata["checkpoint_sha256"] == state["manifest"]["checkpoint_sha256"]
            and metadata["selected_epoch"] == state["manifest"]["selected_epoch"],
            "loaded checkpoint differs from preflight")
    destination = state["output"]
    destination.mkdir(parents=True, exist_ok=False)
    measurements = []
    for stored in state["rows"]:
        clean, raw = synthetic_case(stored["frequency_hz"], stored["condition"], stored["seed"])
        before = raw.copy()
        prediction, notch, receipt = runtime.process_recording(raw.copy(), model)
        require(np.array_equal(raw, before), "synthetic raw input was mutated")
        tracks = {"raw": raw, "notch": notch, "traditional": _traditional(raw.copy()), "v12": prediction}
        case = measure_case(stored, clean, tracks)
        case["runtime_receipt"] = receipt
        measurements.append(case)
    metrics = aggregate_measurements(measurements)
    checks = [check for case in measurements for check in case["stored_mse_checks"]]
    numerical_match = all(check["matches"] for check in checks)
    flags = {"known_clean_is_artificial_only": True, "real_eeg_clean_ground_truth_available": False,
             "clean_ground_truth_available": False, "production_validated": False,
             "runtime_enabled": False, "independent_personal_validation": False}
    detail = {"schema_version": 1, "status": "formal", "validation_domain": "artificial_signal_only_not_real_EEG",
              "channels": [f"CH{i}" for i in range(1, 9)], "sample_rate_hz": 250,
              "samples_per_case": 4096, "case_count": 20, "cases": measurements, **flags}
    write_json(destination / "case_measurements.json", detail)
    write_json(destination / "metrics.json", {"schema_version": 1, **flags,
               "aggregation": "pooled centered error energy / pooled centered artificial-clean energy; all channels and cases",
               "regions": metrics})
    artifact_names = ["case_measurements.json", "metrics.json"]
    if plot:
        plot_nmse(metrics, destination / "interior_nmse.png")
        artifact_names.append("interior_nmse.png")
    for path, expected in state["source_hashes"].items():
        require(sha256(path) == expected, "source changed during audit: " + path)
    public = state["documents"]["public"]
    public_metrics = {name: {key: row.get(key) for key in
                      ("window_count", "input_mse", "output_mse", "output_to_input_mse_ratio")}
                      for name, row in public["conditions"].items()}
    report = {"schema_version": 1, "status": "formal", "completion_status": "complete",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "validation_domain": "artificial_signal_only_not_real_EEG", **flags,
              "training_performed": False, "checkpoint_selection_performed": False,
              "parameter_search_performed": False, "acceptance_thresholds_modified": False,
              "deployment_gate_evaluated": False, "personal_raw_data_accessed": False,
              "evaluation_directory": str(state["evaluation"]), "model": metadata,
              "source_sha256_before": state["source_hashes"],
              "source_hashes_unchanged": True, "evaluation_public_synthetic_sha_verified": True,
              "case_count": len(measurements), "sample_rate_hz": 250, "samples_per_case": 4096,
              "regions": {name: {"start_sample": bounds[0], "stop_sample_exclusive": bounds[1]}
                          for name, bounds in REGIONS.items()},
              "reproduction": {"verified": numerical_match, "comparison_count": len(checks),
                               "matched_count": sum(check["matches"] for check in checks),
                               "rtol": MSE_RTOL, "atol": MSE_ATOL,
                               "max_absolute_mse_difference": max(check["absolute_difference"] for check in checks),
                               "scope": "stored four-track interior centered MSE; numerical integrity only"},
              "amplitude_phase_method": {
                  "fit": "joint least squares: intercept + centered linear time + sin/cos(f, 2*f, 7Hz)",
                  "time": "absolute sample index / 250; same for artificial clean and each track",
                  "phase_difference": "wrap(measured_phase - artificial_clean_phase) in [-pi,pi)",
                  "undefined_phase_amplitude_floor": PHASE_AMPLITUDE_FLOOR,
                  "floor_is_numerical_only": True, "scientific_acceptance_thresholds": None},
              "nmse_by_region_and_condition": metrics,
              "stage_contribution_interpretation": "raw->notch fixed contribution; notch->v12 neural contribution; positive signed error reduction improves artificial reference error, negative worsens; not biological noise removal",
              "existing_research_gate_passed": state["documents"]["summary"].get("research_gate", {}).get("passed"),
              "existing_research_gate_is_deployment_acceptance": False,
              "public_saved_window_metrics": public_metrics,
              "public_metrics_scope": "existing shared-coordinate window MSE on recorded surrogates; no full/interior four-track public NMSE can be reconstructed from these JSON; no public EEG reloaded",
              "limitations": [
                  "Known clean is constructed artificial signal only, not a real clean EEG recording.",
                  "A numerical reproduction pass and an existing research gate do not establish deployment acceptance.",
                  "No independent personal validation, new epoch selection or threshold tuning is performed.",
                  "Centering removes DC offset only; phase differences remain in centered MSE and joint LS.",
                  "Public/synthetic SHA256 values are checked against summary, which is not a cryptographic signature."],
              "artifacts": {name: {"sha256": sha256(destination / name)} for name in artifact_names}}
    # Completion marker must be the final file written, including when mismatch is reported.
    write_json(destination / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-directory", default=DEFAULT_EVALUATION)
    parser.add_argument("--output", required=True, help="New directory under analysis_outputs")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--plot", action="store_true", help="Save a standard matplotlib static PNG")
    args = parser.parse_args(argv)
    try:
        result = audit(args.evaluation_directory, output=args.output, device=args.device, plot=args.plot)
    except (ValueError, OSError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"report": str(Path(args.output) / "report.json"),
                      "reproduction_verified": result["reproduction"]["verified"],
                      "production_validated": False, "runtime_enabled": False}, ensure_ascii=False))
    return 0 if result["reproduction"]["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
