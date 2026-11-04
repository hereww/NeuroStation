"""Independent audit of saved v5 validation JSON; standard library only.
No evaluator imports, waveform access, inference, or internal reference traversal.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import stat
import statistics
import tempfile
from fractions import Fraction

PREREGISTRATION = "analysis_outputs/learned_denoiser_v5_ssvep_task_preservation_preregistration.json"
BASELINE = "analysis_outputs/ssvep_task_v5_validation_baseline_20261007.json"
CANDIDATE = "analysis_outputs/ssvep_task_v5_candidate_validation_hann3_20261007.json"
FINAL_VERIFICATION = "analysis_outputs/ssvep_task_v5_final_verification_20261007.json"
EVALUATION_LOCK = "analysis_outputs/learned_denoiser_v5_ssvep_candidate_evaluation_lock_20261007.json"
PREREGISTRATION_SHA256 = "6f28f1b344e654d66d1e4132d34c0128a8a27185bc21916e31caed57dc914990"
CANDIDATE_SHA256 = "721fc58d69800cee03a348820e5b63b96beccb6738899486cd1aed953886d3ce"
# Read from both existing final verification and candidate evaluation lock.
BASELINE_SHA256 = "a5d1e532ae776263508a3f9a326494c994d908fabdb64783f35d44a9c8fdcf9c"
FIXED_INPUTS = (PREREGISTRATION, BASELINE, CANDIDATE, FINAL_VERIFICATION, EVALUATION_LOCK)
CANDIDATE_ID = "datasets/processed/ssvep_denoiser_v5_candidate_20261007_hann3"
FREQUENCIES = (9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
               10.25, 12.25, 14.25, 10.75, 12.75, 14.75)
EPSILON = 1e-12
GATE_KEYS = {
    "validation_macro_accuracy_drop_max_absolute": 0.02,
    "validation_micro_accuracy_drop_max_absolute": 0.02,
    "validation_prediction_disagreement_max_fraction": 0.02,
    "validation_target_score_ratio_min": 0.95,
}
REPORT_FIELDS = {
    "candidate_dirs", "candidate_signal_key", "candidate_task_metrics_computed", "check_only",
    "clean_ground_truth", "created_at_utc", "dataset_summary", "decoder", "domain_summary",
    "evaluation_version", "global_accuracy_summary", "holdout_loaded", "learned_model_loaded",
    "limitations", "macro_accuracy", "micro_accuracy", "model_metrics_computed", "model_metrics_scope",
    "native_units_assumed", "npz_content_hashes_verified", "npz_verification_scope", "production_validated",
    "reference_dir", "reference_manifest_sha256", "reference_version", "runtime_enabled", "schema_version",
    "selected_subjects", "split", "status", "subjects", "task_metrics_computed", "threshold_selection_performed",
    "training_performed", "trial_alignment_values_verified", "validation_use", "waveforms_loaded",
}
SUBJECT_FIELDS = {
    "baseline", "candidates", "dataset", "domains", "fundamental_bins", "native_sample_rate_hz", "native_units",
    "native_units_assumed", "second_harmonic_bins", "source_stimulation_window", "split", "subject_id", "trials",
}
CANDIDATE_FIELDS = {
    "accuracy_delta", "candidate_id", "metrics", "prediction_agreement", "predictions",
    "spectral_score_ratio", "target_frequency_power",
}
TRIAL_COLUMNS = {"baseline_prediction", "baseline_target_frequency_power", "block_id",
                 "electrode_id", "frequency_hz", "target_id", "trial_index"}


class AuditError(ValueError):
    """Fail-closed input or publication contract violation."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def integer(value, where, minimum=0, maximum=None):
    require(type(value) is int, f"{where}: expected integer, not bool/float")
    require(value >= minimum and (maximum is None or value <= maximum),
            f"{where}: integer out of range")
    return value


def number(value, where, minimum=None):
    require(type(value) in (int, float), f"{where}: expected number, not bool")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite, f"{where}: non-finite number")
    require(minimum is None or value >= minimum, f"{where}: negative number")
    return value


def _finite_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)
    elif type(value) in (int, float):
        number(value, "JSON")


def strict_json_loads(data):
    """Reject duplicate keys at every depth and non-standard/non-finite numbers."""
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise AuditError(f"invalid JSON numeric constant: {value}")

    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise AuditError(f"invalid JSON: {exc}") from exc
    require(type(value) is dict, "JSON root must be an object")
    _finite_tree(value)
    return value


def _object(value, where):
    require(type(value) is dict, f"{where}: expected object")
    return value


def _list(value, where, length=None):
    require(type(value) is list, f"{where}: expected array")
    require(length is None or len(value) == length, f"{where}: array length mismatch")
    return value


def _header(report, where):
    _object(report, where)
    integer(report.get("schema_version"), where + ".schema_version", 1, 1)
    require(report.get("status") == "formal", where + ": expected formal status")


def validate_preregistration(prereg):
    _finite_tree(prereg)
    _header(prereg, "preregistration")
    require(prereg.get("experiment") == "learned_denoise_v5_ssvep_task_preservation", "wrong experiment")
    scope = _object(prereg.get("scope"), "scope")
    require(scope.get("datasets") == ["Nakanishi2015", "Wearable"], "wrong datasets")
    require(scope.get("split_for_initial_report") == "validation", "wrong prereg split")
    for key in ("runtime_enabled", "production_validated"):
        require(scope.get(key) is False, f"preregistration {key} must be false")
    contract = _object(prereg.get("fixed_reference_contract"), "reference contract")
    require(contract.get("reference_version") == "ssvep_reference_v5", "wrong reference")
    partition = contract.get("subject_partition")
    require(partition == {"train": [1, 2, 3, 4, 5, 6], "validation": [7, 8], "holdout": [9, 10]},
            "wrong partition")
    for ids in partition.values():
        for sid in ids:
            integer(sid, "partition subject", 1, 10)
    decoder = _object(prereg.get("decoder"), "preregistered decoder")
    require(decoder.get("type") == "deterministic_fft_power_baseline", "wrong decoder")
    windows = _object(decoder.get("windows"), "decoder windows")
    for ds in scope["datasets"]:
        window = _object(windows.get(ds), "dataset window")
        require(window.get("frequency_order_hz") == list(FREQUENCIES), "wrong target order")
        for value in window["frequency_order_hz"]:
            number(value, "frequency", 0)
    gates = _object(prereg.get("predeclared_research_gates"), "preregistered gates")
    for key, expected in GATE_KEYS.items():
        require(number(gates.get(key), key) == expected, f"gate changed: {key}")
    require(gates.get("no_threshold_tuning_after_candidate_metrics") is True, "threshold tuning forbidden")


def _parse_report(report, candidate):
    _finite_tree(report)
    _header(report, "report")
    require(set(report) <= REPORT_FIELDS, "unknown report schema field")
    for key in ("clean_ground_truth", "learned_model_loaded", "model_metrics_computed", "native_units_assumed",
                "npz_content_hashes_verified", "trial_alignment_values_verified", "waveforms_loaded"):
        if key in report:
            require(type(report[key]) is bool, f"report {key}: expected bool")
    require(report.get("evaluation_version") == "ssvep_task_eval_v5", "wrong evaluator schema")
    require(report.get("reference_version") == "ssvep_reference_v5", "wrong reference version")
    require(report.get("split") == "validation", "only validation split is allowed")
    for key in ("holdout_loaded", "runtime_enabled", "production_validated", "check_only",
                "threshold_selection_performed", "training_performed"):
        require(report.get(key) is False, f"report {key} must be false")
    require(report.get("task_metrics_computed") is True, "missing saved task metrics")
    require(report.get("candidate_task_metrics_computed") is candidate, "wrong candidate metric flag")
    require(report.get("candidate_dirs") == ([CANDIDATE_ID] if candidate else []), "wrong candidate identity")
    decoder = _object(report.get("decoder"), "saved decoder")
    require(number(decoder.get("epsilon"), "decoder epsilon", 0) == EPSILON, "wrong score epsilon")
    require(decoder.get("predictions_use_labels") is False, "label-based decoder rejected")
    groups = _list(report.get("subjects"), "subjects")
    require(bool(groups), "no validation subjects")
    rows, seen_groups = {}, set()
    for group in groups:
        _object(group, "subject")
        require(set(group) <= SUBJECT_FIELDS, "unknown subject schema field")
        for key in ("fundamental_bins", "second_harmonic_bins"):
            if key in group:
                for value in _list(group[key], key, len(FREQUENCIES)):
                    integer(value, key)
        if "native_sample_rate_hz" in group:
            integer(group["native_sample_rate_hz"], "sample rate", 1)
        if "source_stimulation_window" in group:
            window = _object(group["source_stimulation_window"], "source window")
            require(set(window) == {"start_inclusive", "end_exclusive"}, "source window schema mismatch")
            start = integer(window["start_inclusive"], "window start")
            end = integer(window["end_exclusive"], "window end")
            require(end > start, "empty source window")
        if "native_units_assumed" in group:
            require(group["native_units_assumed"] is False, "assumed native units rejected")
        ds = group.get("dataset")
        require(ds in ("Nakanishi2015", "Wearable"), "unknown dataset")
        sid = integer(group.get("subject_id"), "subject_id", 7, 8)
        require(group.get("split") == "validation", "wrong subject split/holdout")
        group_id = (ds, sid)
        require(group_id not in seen_groups, "duplicate subject identity")
        seen_groups.add(group_id)
        trials = _object(group.get("trials"), "trials")
        require(set(trials) <= TRIAL_COLUMNS, "unknown trial schema field")
        require(TRIAL_COLUMNS - {"baseline_target_frequency_power"} <= set(trials), "missing trial field")
        labels = _list(trials["target_id"], "target_id")
        n = len(labels)
        require(n > 0, "empty trial group")
        for key, values in trials.items():
            _list(values, key, n)
        candidates = _list(group.get("candidates"), "subject candidates", int(candidate))
        c = _object(candidates[0], "candidate") if candidate else {}
        if candidate:
            require(set(c) <= CANDIDATE_FIELDS, "unknown candidate schema field")
            require(c.get("candidate_id") == CANDIDATE_ID, "wrong candidate id")
            _list(c.get("predictions"), "candidate predictions", n)
            if "target_frequency_power" in c:
                _list(c["target_frequency_power"], "candidate scores", n)
        for i, label in enumerate(labels):
            y = integer(label, "target_id", 0, 11)
            index = integer(trials["trial_index"][i], "trial_index")
            block = integer(trials["block_id"][i], "block_id")
            electrode = integer(trials["electrode_id"][i], "electrode_id", -1, 1)
            require(electrode in ((-1,) if ds == "Nakanishi2015" else (0, 1)), "invalid electrode/domain")
            freq = number(trials["frequency_hz"][i], "frequency_hz", 0)
            require(freq == FREQUENCIES[y], "target frequency/label mismatch")
            pred = integer(trials["baseline_prediction"][i], "baseline_prediction", 0, 11)
            cpred = integer(c["predictions"][i], "candidate prediction", 0, 11) if candidate else None
            bp = (number(trials["baseline_target_frequency_power"][i], "baseline score", 0)
                  if "baseline_target_frequency_power" in trials else None)
            cp = (number(c["target_frequency_power"][i], "candidate score", 0)
                  if "target_frequency_power" in c else None)
            ratio = cp / bp if bp is not None and cp is not None and bp > EPSILON else None
            if ratio is not None:
                number(ratio, "recomputed ratio", 0)
            identity = (ds, sid, index)
            require(identity not in rows, "duplicate trial identity")
            # Fixed saved v5 axis: Nakanishi -1 unknown; Wearable 0 dry, 1 wet.
            domain = "electrode_unknown" if electrode == -1 else ("dry" if electrode == 0 else "wet")
            rows[identity] = dict(dataset=ds, subject=sid, index=index, block=block,
                                  electrode=electrode, frequency=freq, target=y, baseline=pred,
                                  candidate=cpred, baseline_score=bp, candidate_score=cp,
                                  ratio=ratio, domain=ds + "/" + domain)
    integer(report.get("selected_subjects"), "selected_subjects", 1)
    require(report["selected_subjects"] == len(seen_groups), "subject count mismatch")
    for key in ("global_accuracy_summary", "dataset_summary", "domain_summary"):
        _object(report.get(key), key)
    return rows


def _aggregate(rows):
    """Macro averages subjects *within the selected rows/domain*."""
    subjects = {}
    for row in rows:
        subjects.setdefault((row["dataset"], row["subject"]), []).append(row)
    n = len(rows)
    result = {"trials": n, "subjects": len(subjects)}
    for role in ("baseline", "candidate"):
        correct = sum(r[role] == r["target"] for r in rows)
        micro = Fraction(correct, n)
        macro = sum((Fraction(sum(r[role] == r["target"] for r in group), len(group))
                     for group in subjects.values()), Fraction()) / len(subjects)
        result[role] = {"correct_trials": correct, "micro_accuracy": float(micro), "macro_accuracy": float(macro)}
        result["_" + role + "_micro"] = micro
        result["_" + role + "_macro"] = macro
    disagree = sum(r["baseline"] != r["candidate"] for r in rows)
    result.update(disagreements=disagree, prediction_disagreement=float(Fraction(disagree, n)),
                  prediction_agreement=float(Fraction(n - disagree, n)))
    for metric in ("micro", "macro"):
        delta = result["_candidate_" + metric] - result["_baseline_" + metric]
        result[metric + "_accuracy_delta"] = float(delta)
        result[metric + "_accuracy_drop"] = float(-delta)
    ratios = [r["ratio"] for r in rows if r["ratio"] is not None]
    missing = sum(r["baseline_score"] is None or r["candidate_score"] is None for r in rows)
    # Preserve the exact mean of binary64 saved-score ratios for strict gating.
    # A single nextafter-below-threshold trial must not round the mean up to a pass.
    result["_ratio_mean_exact"] = (sum((Fraction.from_float(float(v)) for v in ratios), Fraction()) /
                                    len(ratios) if ratios else None)
    result["target_score_ratio"] = {
        "verifiable": missing == 0, "gate_verifiable": missing == 0 and len(ratios) == n,
        "defined_trials": len(ratios), "undefined_trials": n - len(ratios) - missing,
        "missing_score_trials": missing,
        "mean": float(result["_ratio_mean_exact"]) if ratios else None,
        "median": (statistics.median(ratios) if len(ratios) % 2 else
                   float((Fraction.from_float(float(sorted(ratios)[len(ratios) // 2 - 1])) +
                          Fraction.from_float(float(sorted(ratios)[len(ratios) // 2]))) / 2)) if ratios else None,
        "minimum": min(ratios) if ratios else None, "maximum": max(ratios) if ratios else None,
        "reason": ("recomputed from saved labeled target scores" if missing == 0 else
                   "unverifiable: saved labeled target scores missing; summaries not substituted"),
    }
    return result


def research_gates(aggregate):
    """Exact rational count boundaries; no tolerance or abs(delta)."""
    result = {}
    for metric in ("micro", "macro"):
        drop = aggregate["_baseline_" + metric] - aggregate["_candidate_" + metric]
        result["validation_" + metric + "_accuracy_drop"] = {
            "observed_drop": float(drop), "operator": "<=", "threshold": 0.02,
            "passed": drop <= Fraction(1, 50),
        }
    result["validation_prediction_disagreement"] = {
        "observed": aggregate["prediction_disagreement"], "operator": "<=", "threshold": 0.02,
        "passed": Fraction(aggregate["disagreements"], aggregate["trials"]) <= Fraction(1, 50),
    }
    ratio = aggregate["target_score_ratio"]
    result["validation_target_score_ratio"] = {
        "observed": ratio["mean"], "operator": ">=", "threshold": 0.95,
        "verifiable": ratio["gate_verifiable"],
        "passed": ratio["gate_verifiable"] and aggregate["_ratio_mean_exact"] >= Fraction.from_float(0.95),
        "comparison": "exact mean of recomputed binary64 ratios >= binary64 0.95; no tolerance",
    }
    return result


def _public(value):
    if isinstance(value, dict):
        return {k: _public(v) for k, v in value.items() if not k.startswith("_")}
    if isinstance(value, list):
        return [_public(v) for v in value]
    return value


def _check_claims(baseline, candidate, rows, aggregates):
    """Only independently recoverable declarations; no metric is consumed as truth."""
    mismatches, unverifiable = [], []

    def compare(where, observed, expected):
        if expected is None:
            if observed is not None:
                number(observed, where)
            unverifiable.append(where)
            return
        if type(expected) is int:
            integer(observed, where)
            match = observed == expected
        elif isinstance(expected, dict):
            _object(observed, where)
            require(set(observed) == set(expected), where + ": key schema mismatch")
            for key, value in expected.items():
                compare(where + "." + key, observed[key], value)
            return
        else:
            number(observed, where)
            # Serialization/aggregation rounding only; gates never use tolerance.
            match = math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)
        if not match:
            mismatches.append({"field": where, "saved": observed, "recomputed": expected})

    def metrics(saved, subset, role, aggregate, where, subject=False):
        _object(saved, where)
        allowed = {"accuracy", "trials", "mean_confidence", "mean_relative_score", "mean_target_frequency_power",
                   "correct_trials"} | ({"macro_target_accuracy", "per_target_accuracy"} if subject else
                                        {"subjects", "micro_accuracy", "macro_accuracy"})
        require(set(saved) <= allowed, where + ": unknown metric schema field")
        for key in ("accuracy", "macro_target_accuracy", "micro_accuracy", "macro_accuracy",
                    "mean_confidence", "mean_relative_score"):
            if key in saved:
                require(number(saved[key], where + "." + key, 0) <= 1, where + ": metric outside [0,1]")
        if "mean_target_frequency_power" in saved:
            number(saved["mean_target_frequency_power"], where + ".mean_target_frequency_power", 0)
        accuracy = aggregate[role]["micro_accuracy"]
        expected = dict(trials=len(subset), accuracy=accuracy)
        if subject:
            targets = {}
            for row in subset:
                targets.setdefault(row["target"], []).append(row)
            per_target = {str(k): sum(r[role] == k for r in group) / len(group)
                          for k, group in targets.items()}
            expected.update(correct_trials=aggregate[role]["correct_trials"], per_target_accuracy=per_target,
                            macro_target_accuracy=math.fsum(per_target.values()) / len(per_target))
        else:
            expected.update(subjects=aggregate["subjects"], micro_accuracy=accuracy,
                            macro_accuracy=aggregate[role]["macro_accuracy"])
        if "correct_trials" in saved:
            expected["correct_trials"] = aggregate[role]["correct_trials"]
        scores = [r[role + "_score"] for r in subset]
        if "mean_target_frequency_power" in saved:
            expected["mean_target_frequency_power"] = (float(sum((Fraction.from_float(float(v)) for v in scores), Fraction()) / len(scores))
                                                        if all(s is not None for s in scores) else None)
        for key, value in expected.items():
            require(key in saved, where + ": missing " + key)
            compare(where + "." + key, saved[key], value)
        for key in ("mean_confidence", "mean_relative_score"):
            if key in saved:
                number(saved[key], where + "." + key)
                unverifiable.append(where + "." + key)

    def ratio_claim(saved, subset, where):
        _object(saved, where)
        require(set(saved) == {"mean", "median", "minimum", "maximum", "defined_trials", "undefined_trials", "per_trial"},
                where + ": ratio schema mismatch")
        for key in ("defined_trials", "undefined_trials"):
            integer(saved[key], where + "." + key)
        for key in ("mean", "median", "minimum", "maximum"):
            if saved[key] is not None:
                number(saved[key], where + "." + key, 0)
        aggregate = _aggregate(subset)["target_score_ratio"]
        for key in ("mean", "median", "minimum", "maximum", "defined_trials", "undefined_trials"):
            require(key in saved, where + ": missing " + key)
            compare(where + "." + key, saved[key], aggregate[key] if aggregate["verifiable"] else None)
        per_trial = _list(saved.get("per_trial"), where + ".per_trial", len(subset))
        for i, (value, row) in enumerate(zip(per_trial, subset)):
            if row["baseline_score"] is None or row["candidate_score"] is None:
                if value is not None:
                    number(value, where + ".per_trial", 0)
                unverifiable.append(where + f".per_trial[{i}]")
            elif row["ratio"] is None:
                require(value is None, "undefined saved ratio must be null")
            else:
                compare(where + f".per_trial[{i}]", value, row["ratio"])

    def container(saved, subset, where, is_candidate, subject=False, global_scope=False):
        _object(saved, where)
        if not subject or ".domains." in where:
            require(set(saved) == {"baseline", "candidates"}, where + ": container schema mismatch")
        aggregate = _aggregate(subset)
        metrics(saved.get("baseline"), subset, "baseline", aggregate, where + ".baseline", subject)
        cs = _list(saved.get("candidates"), where + ".candidates", int(is_candidate))
        if is_candidate:
            c = _object(cs[0], where + ".candidate")
            fields = ({"candidate_id", "metrics", "micro_accuracy_delta", "macro_accuracy_delta"} if not subject else
                      {"candidate_id", "metrics", "accuracy_delta"})
            if not global_scope:
                fields |= ({"prediction_agreement", "spectral_score_ratio"} if subject else
                           {"prediction_agreement", "mean_spectral_score_ratio", "defined_ratio_trials"})
            if subject and ".domains." not in where:
                fields |= {"predictions", "target_frequency_power"}
            require(set(c) <= fields, where + ": unknown candidate summary field")
            require(c.get("candidate_id") == CANDIDATE_ID, "summary candidate id mismatch")
            metrics(c.get("metrics"), subset, "candidate", aggregate, where + ".candidate.metrics", subject)
            keys = ("accuracy_delta",) if subject else ("micro_accuracy_delta", "macro_accuracy_delta")
            for key in keys:
                expected = aggregate["micro_accuracy_delta" if key == "accuracy_delta" else key]
                compare(where + ".candidate." + key, c.get(key), expected)
            if not global_scope:
                compare(where + ".candidate.prediction_agreement", c.get("prediction_agreement"),
                        aggregate["prediction_agreement"])
                if subject:
                    ratio_claim(c.get("spectral_score_ratio"), subset, where + ".candidate.ratio")
                else:
                    ratio = aggregate["target_score_ratio"]
                    compare(where + ".candidate.mean_spectral_score_ratio", c.get("mean_spectral_score_ratio"),
                            ratio["mean"] if ratio["verifiable"] else None)
                    compare(where + ".candidate.defined_ratio_trials", c.get("defined_ratio_trials"),
                            ratio["defined_trials"] if ratio["verifiable"] else None)

    row_map = {(r["dataset"], r["subject"], r["index"]): r for r in rows}
    for name, report, is_candidate in (("baseline_report", baseline, False), ("candidate_report", candidate, True)):
        for metric in ("micro", "macro"):
            compare(name + "." + metric + "_accuracy", report.get(metric + "_accuracy"),
                    aggregates["global"]["baseline"][metric + "_accuracy"])
        container(report["global_accuracy_summary"], rows, name + ".global", is_candidate, global_scope=True)
        for field, dimension in (("dataset_summary", "dataset"), ("domain_summary", "domain")):
            saved = report[field]
            keys = {r[dimension] for r in rows}
            require(set(saved) == keys, field + ": group schema mismatch")
            for key in sorted(keys):
                container(saved[key], [r for r in rows if r[dimension] == key], name + "." + field + "." + key, is_candidate)
        for group in report["subjects"]:
            ds, sid = group["dataset"], group["subject_id"]
            subset = [row_map[(ds, sid, i)] for i in group["trials"]["trial_index"]]
            where = name + f".subjects.{ds}:{sid}"
            container(group, subset, where, is_candidate, subject=True)
            domains = _object(group.get("domains"), where + ".domains")
            domain_keys = {r["domain"].split("/", 1)[1] for r in subset}
            require(set(domains) == domain_keys, "subject domain schema mismatch")
            for domain in sorted(domain_keys):
                part = [r for r in subset if r["domain"] == ds + "/" + domain]
                container(domains[domain], part, where + ".domains." + domain, is_candidate, subject=True)
    return mismatches, sorted(set(unverifiable))


def audit_saved_reports(baseline, candidate, preregistration):
    """Pure in-memory audit. Production hash binding is in audit_workspace."""
    validate_preregistration(preregistration)
    b = _parse_report(baseline, False)
    c = _parse_report(candidate, True)
    require(b.keys() == c.keys(), "trial identity sets differ")
    for identity in b:
        for key in ("block", "electrode", "frequency", "target", "baseline", "baseline_score"):
            require(b[identity][key] == c[identity][key], f"reports differ at {identity}: {key}")
    rows = list(c.values())
    aggregates = {"global": _aggregate(rows)}
    for output, dimension in (("dataset", "dataset"), ("domain", "domain")):
        aggregates[output] = {key: _aggregate([r for r in rows if r[dimension] == key])
                              for key in sorted({r[dimension] for r in rows})}
    aggregates["subject"] = {f"{ds}:{sid}": _aggregate([r for r in rows if (r["dataset"], r["subject"]) == (ds, sid)])
                              for ds, sid in sorted({(r["dataset"], r["subject"]) for r in rows})}
    aggregates["subject_domain"] = {f"{domain}/subject_{sid}": _aggregate([
        r for r in rows if (r["domain"], r["subject"]) == (domain, sid)])
        for domain, sid in sorted({(r["domain"], r["subject"]) for r in rows})}
    mismatches, unverifiable = _check_claims(baseline, candidate, rows, aggregates)
    gates = research_gates(aggregates["global"])
    gates_pass = all(g["passed"] for g in gates.values())
    consistent = not mismatches
    return {
        "schema_version": 1, "status": "formal", "audit_version": "ssvep_saved_audit_v5",
        "input_scope": "saved validation JSON only; no internal references followed",
        "split": "validation", "audit_consistent": consistent,
        "research_gates_passed": gates_pass, "research_acceptance": consistent and gates_pass,
        "deployment": "blocked" if not (consistent and gates_pass) else "not_authorized_by_saved_audit",
        "runtime_enabled": False, "production_validated": False,
        "waveforms_loaded": False, "holdout_loaded": False, "d_drive_accessed": False,
        "metrics": _public(aggregates), "research_gates": gates,
        "summary_mismatches": mismatches, "unverifiable_summary_fields": unverifiable,
        "limitations": [
            "Saved predictions are audited, not independently inferred from waveforms.",
            "Only labeled target scores are saved; argmax, confidence and relative score cannot be verified.",
            "Domain macro averages subjects within that domain, not whole-subject accuracies.",
            "No clean ground truth, efficacy, hardware, runtime, production or clinical acceptance is established.",
            "Gates use signed drops; summary float comparison tolerance is never used for gates.",
        ],
    }


def workspace_path(workspace, path, *, output=False):
    """Lexical containment plus lstat of every ancestor; never resolve links."""
    root = Path(os.path.abspath(os.fspath(workspace)))
    raw = os.fspath(path)
    require(isinstance(raw, str) and bool(raw), "empty path")
    require(".." not in Path(raw).parts and ".." not in PureWindowsPath(raw).parts, "parent traversal rejected")
    require(not PureWindowsPath(raw).drive or Path(raw).is_absolute(), "foreign drive/UNC path")
    target = Path(os.path.abspath(root / raw))
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise AuditError("path outside workspace") from exc
    require(relative.parts, "workspace root is not a file path")
    for part in relative.parts:
        require(not any(ch in part for ch in ':<>"|?*') and not part.endswith((".", " ")),
                "unsafe path component/alternate data stream")
        require(not PureWindowsPath(part).is_reserved(), "reserved path component")
    chain = list(reversed(target.parents)) + [target]
    for component in chain:
        try:
            info = component.lstat()
        except FileNotFoundError:
            require(output and component == target, "path or parent does not exist")
            continue
        require(not stat.S_ISLNK(info.st_mode) and not (getattr(info, "st_file_attributes", 0) & 0x400),
                "symlink/reparse path rejected")
        if component != target:
            require(stat.S_ISDIR(info.st_mode), "parent is not a directory")
        else:
            require(not output, "output exists; no clobber")
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "input must be a non-linked regular file")
    return target


def _read_fixed_json(workspace, relative):
    require(relative in FIXED_INPUTS, "only fixed JSON inputs may be read")
    path = workspace_path(workspace, relative)
    before = path.lstat()
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        require((before.st_dev, before.st_ino) == (opened.st_dev, opened.st_ino), "input changed during open")
        data = handle.read()
    after = path.lstat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "input changed during read")
    workspace_path(workspace, relative)
    return strict_json_loads(data), hashlib.sha256(data).hexdigest()


def audit_workspace(workspace):
    """Fixed allowlist: no CLI input paths and no paths inside JSON are opened."""
    loaded = {name: _read_fixed_json(workspace, name) for name in FIXED_INPUTS}
    for name, expected in ((PREREGISTRATION, PREREGISTRATION_SHA256),
                           (BASELINE, BASELINE_SHA256), (CANDIDATE, CANDIDATE_SHA256)):
        require(loaded[name][1] == expected, f"SHA-256 mismatch: {name}")
    final, lock = loaded[FINAL_VERIFICATION][0], loaded[EVALUATION_LOCK][0]
    _header(final, "final verification")
    _header(lock, "evaluation lock")
    require(final.get("verification_passed") is True, "final verification not passed")
    require(_object(final.get("artifact_sha256"), "final hashes").get(BASELINE) == BASELINE_SHA256,
            "final verification baseline binding differs")
    require(_object(lock.get("files"), "lock hashes").get(BASELINE) == BASELINE_SHA256,
            "evaluation lock baseline binding differs")
    require(final["artifact_sha256"].get(PREREGISTRATION) == PREREGISTRATION_SHA256,
            "final verification preregistration binding differs")
    require(lock.get("split") == "validation" and lock.get("candidate_subjects") == [7, 8], "lock split differs")
    for sid in lock["candidate_subjects"]:
        integer(sid, "lock subject", 7, 8)
    for document in (final, lock):
        for key in ("holdout_waveforms_loaded", "runtime_enabled", "production_validated"):
            require(document.get(key) is False, f"lock/final {key} must be false")
    result = audit_saved_reports(loaded[BASELINE][0], loaded[CANDIDATE][0], loaded[PREREGISTRATION][0])
    require(set(result["metrics"]["subject"]) == {"Nakanishi2015:7", "Nakanishi2015:8", "Wearable:7", "Wearable:8"},
            "fixed workspace subject coverage differs")
    observed = result["metrics"]["global"]
    require((observed["trials"], observed["baseline"]["correct_trials"],
             observed["candidate"]["correct_trials"], observed["disagreements"]) == (840, 443, 442, 19),
            "fixed workspace independent counts differ")
    result["input_sha256"] = {name: value[1] for name, value in loaded.items()}
    result["hash_binding_verified"] = True
    result["baseline_hash_binding_sources"] = [FINAL_VERIFICATION, EVALUATION_LOCK]
    return result


def render_markdown(report):
    lines = ["# v5 saved-JSON independent audit", "",
             f"- Audit consistent: `{report['audit_consistent']}`",
             f"- Research gates passed: `{report['research_gates_passed']}`",
             f"- Research acceptance: `{report['research_acceptance']}`",
             f"- Deployment: `{report['deployment']}`",
             "- Runtime enabled: `false`; production validated: `false`", "",
             "Metrics recomputed from saved trial labels/predictions/scores.",
             "Macro = unweighted mean of subjects **within the listed scope/domain**.", "",
             "| Scope | Trials | Baseline correct | Candidate correct | Baseline micro / macro | Candidate micro / macro | Disagreements | Score ratio |",
             "|---|---:|---:|---:|---|---|---:|---|"]
    scopes = [("global", report["metrics"]["global"])]
    for dimension in ("dataset", "domain", "subject", "subject_domain"):
        scopes.extend((dimension + ":" + key, value) for key, value in report["metrics"][dimension].items())
    for scope, metrics in scopes:
        b, c, ratio = metrics["baseline"], metrics["candidate"], metrics["target_score_ratio"]
        ratio_text = str(ratio["mean"]) if ratio["gate_verifiable"] else "UNVERIFIABLE / undefined"
        lines.append(f"| {scope} | {metrics['trials']} | {b['correct_trials']} | {c['correct_trials']} | "
                     f"{b['micro_accuracy']:.12g} / {b['macro_accuracy']:.12g} | "
                     f"{c['micro_accuracy']:.12g} / {c['macro_accuracy']:.12g} | {metrics['disagreements']} | {ratio_text} |")
    lines.extend(["", "## Research gates", "", "```json", json.dumps(report["research_gates"], indent=2), "```",
                  "", "## Summary verification", "", f"Mismatches: {len(report['summary_mismatches'])}",
                  "```json", json.dumps(report["summary_mismatches"], indent=2), "```",
                  "", "Unverifiable summary fields (not independently audited metrics):"])
    lines.extend("- " + field for field in report["unverifiable_summary_fields"])
    lines.extend(["", "## Limitations", ""] + ["- " + text for text in report["limitations"]])
    if "input_sha256" in report:
        lines.extend(["", "## Fixed JSON hashes", "", "```json", json.dumps(report["input_sha256"], indent=2), "```"])
    return "\n".join(lines) + "\n"


def validate_output_paths(workspace, json_output, markdown_output):
    paths = [workspace_path(workspace, p, output=True) for p in (json_output, markdown_output)]
    require(paths[0] != paths[1], "output paths must differ")
    require(paths[0].suffix.lower() == ".json" and paths[1].suffix.lower() == ".md", "wrong output extensions")
    return paths


def publish_reports(workspace, report, json_output, markdown_output):
    """Stage/fsync both files, then atomic hard-link creates (never replace).

    Each publication is atomic/no-clobber. No two-file transaction is available:
    on second-file failure roll back only our own first inode, not old outputs.
    """
    paths = validate_output_paths(workspace, json_output, markdown_output)
    texts = [json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", render_markdown(report)]
    staged, published = [], []
    try:
        for path, text in zip(paths, texts):
            workspace_path(workspace, path, output=True)
            fd, temporary = tempfile.mkstemp(prefix=".ssvep_saved_audit_v5-", suffix=".tmp", dir=path.parent)
            staged.append(Path(temporary))
            with os.fdopen(fd, "wb") as handle:
                handle.write(text.encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
        for temporary, path in zip(staged, paths):
            workspace_path(workspace, path, output=True)
            # Unlike os.replace/Unix rename, link fails atomically if dst exists.
            os.link(temporary, path, follow_symlinks=False)
            published.append((path, temporary.stat()))
    except Exception as exc:
        rollback_errors = []
        for path, own in reversed(published):
            try:
                current = path.lstat()
                if (current.st_dev, current.st_ino) == (own.st_dev, own.st_ino):
                    path.unlink()
                else:
                    rollback_errors.append("output inode changed: " + str(path))
            except OSError as error:
                rollback_errors.append(str(error))
        raise AuditError(f"publication failed without overwriting old outputs: {exc}; rollback={rollback_errors}") from exc
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)
    return paths
