"""Synthetic-only saved JSON audit tests; never open real reports/waveforms."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest import mock

from eeg_tools import ssvep_saved_audit_v5 as audit


def preregistration():
    return {
        "schema_version": 1, "status": "formal",
        "experiment": "learned_denoise_v5_ssvep_task_preservation",
        "scope": {"datasets": ["Nakanishi2015", "Wearable"], "split_for_initial_report": "validation",
                  "runtime_enabled": False, "production_validated": False},
        "fixed_reference_contract": {"reference_version": "ssvep_reference_v5",
                                     "subject_partition": {"train": [1, 2, 3, 4, 5, 6],
                                                           "validation": [7, 8], "holdout": [9, 10]}},
        "decoder": {"type": "deterministic_fft_power_baseline", "windows": {
            ds: {"frequency_order_hz": list(audit.FREQUENCIES)} for ds in ("Nakanishi2015", "Wearable")}},
        "predeclared_research_gates": dict(audit.GATE_KEYS, no_threshold_tuning_after_candidate_metrics=True),
    }


def synthetic_rows(groups=None, baseline_correct=None, switches=(), ratio=1.0):
    """Procedural fake labels/predictions, not copied from real saved trials."""
    groups = groups or [("Nakanishi2015", 7, 50)]
    rows = []
    total = sum(g[2] for g in groups)
    baseline_correct = total if baseline_correct is None else baseline_correct
    for ds, sid, n in groups:
        for i in range(n):
            label = (i * 5 + sid) % 12
            offset = len(rows)
            pred = label if offset < baseline_correct else (label + 1) % 12
            cpred = ((label + 1) % 12 if pred == label else label) if offset in switches else pred
            electrode = -1 if ds == "Nakanishi2015" else i % 2
            domain = "electrode_unknown" if electrode == -1 else ("dry" if electrode == 0 else "wet")
            rows.append(dict(dataset=ds, subject=sid, index=i, target=label, baseline=pred, candidate=cpred,
                             baseline_score=1.0, candidate_score=ratio, electrode=electrode, block=i // 12,
                             domain=ds + "/" + domain))
    return rows


def fixture_metrics(rows, role, subject):
    n = len(rows)
    correct = sum(r[role] == r["target"] for r in rows)
    result = {"trials": n, "accuracy": correct / n,
              "mean_target_frequency_power": sum(r[role + "_score"] for r in rows) / n}
    if subject:
        targets = sorted({r["target"] for r in rows})
        per_target = {str(y): sum(r[role] == y for r in rows if r["target"] == y) /
                            sum(r["target"] == y for r in rows) for y in targets}
        result.update(correct_trials=correct, per_target_accuracy=per_target,
                      macro_target_accuracy=sum(per_target.values()) / len(per_target))
    else:
        groups = {(r["dataset"], r["subject"]) for r in rows}
        macro = sum(sum(r[role] == r["target"] for r in rows if (r["dataset"], r["subject"]) == g) /
                    sum((r["dataset"], r["subject"]) == g for r in rows) for g in groups) / len(groups)
        result.update(subjects=len(groups), micro_accuracy=correct / n, macro_accuracy=macro)
    return result


def fixture_container(rows, candidate, subject=False, global_scope=False):
    result = {"baseline": fixture_metrics(rows, "baseline", subject), "candidates": []}
    if not candidate:
        return result
    metrics = fixture_metrics(rows, "candidate", subject)
    c = {"candidate_id": audit.CANDIDATE_ID, "metrics": metrics}
    if subject:
        c["accuracy_delta"] = metrics["accuracy"] - result["baseline"]["accuracy"]
    else:
        for metric in ("micro_accuracy", "macro_accuracy"):
            c[metric + "_delta"] = metrics[metric] - result["baseline"][metric]
    if not global_scope:
        ratios = [r["candidate_score"] / r["baseline_score"] if r["baseline_score"] > audit.EPSILON else None
                  for r in rows]
        defined = [v for v in ratios if v is not None]
        c["prediction_agreement"] = sum(r["baseline"] == r["candidate"] for r in rows) / len(rows)
        if subject:
            c["spectral_score_ratio"] = {
                "per_trial": ratios, "defined_trials": len(defined), "undefined_trials": len(rows) - len(defined),
                "mean": sum(defined) / len(defined) if defined else None,
                "median": statistics.median(defined) if defined else None,
                "minimum": min(defined) if defined else None, "maximum": max(defined) if defined else None,
            }
        else:
            c.update(defined_ratio_trials=len(defined),
                     mean_spectral_score_ratio=sum(defined) / len(defined) if defined else None)
    result["candidates"] = [c]
    return result


def reports(rows=None):
    rows = synthetic_rows() if rows is None else rows
    result = []
    for candidate in (False, True):
        global_claim = fixture_container(rows, candidate, global_scope=True)
        # Actual global schema intentionally has no pooled native power metric.
        global_claim["baseline"].pop("mean_target_frequency_power")
        for c in global_claim["candidates"]:
            c["metrics"].pop("mean_target_frequency_power")
        report = {
            "schema_version": 1, "status": "formal", "evaluation_version": "ssvep_task_eval_v5",
            "reference_version": "ssvep_reference_v5", "split": "validation",
            "holdout_loaded": False, "runtime_enabled": False, "production_validated": False,
            "check_only": False, "threshold_selection_performed": False, "training_performed": False,
            "task_metrics_computed": True, "candidate_task_metrics_computed": candidate,
            "candidate_dirs": [audit.CANDIDATE_ID] if candidate else [],
            "decoder": {"epsilon": 1e-12, "predictions_use_labels": False},
            "selected_subjects": len({(r["dataset"], r["subject"]) for r in rows}),
            "subjects": [], "global_accuracy_summary": global_claim,
            "micro_accuracy": global_claim["baseline"]["micro_accuracy"],
            "macro_accuracy": global_claim["baseline"]["macro_accuracy"],
            "dataset_summary": {}, "domain_summary": {},
            # Deliberately hostile metadata: an auditor must NEVER follow these.
            "reference_dir": "D:/forbidden/raw_waveforms",
        }
        for ds, sid in sorted({(r["dataset"], r["subject"]) for r in rows}):
            subset = [r for r in rows if (r["dataset"], r["subject"]) == (ds, sid)]
            group = fixture_container(subset, candidate, subject=True)
            group.update(dataset=ds, subject_id=sid, split="validation", domains={}, trials={
                "target_id": [r["target"] for r in subset],
                "baseline_prediction": [r["baseline"] for r in subset],
                "baseline_target_frequency_power": [r["baseline_score"] for r in subset],
                "trial_index": [r["index"] for r in subset], "block_id": [r["block"] for r in subset],
                "electrode_id": [r["electrode"] for r in subset],
                "frequency_hz": [audit.FREQUENCIES[r["target"]] for r in subset],
            })
            if candidate:
                group["candidates"][0].update(predictions=[r["candidate"] for r in subset],
                                               target_frequency_power=[r["candidate_score"] for r in subset])
            for domain in sorted({r["domain"] for r in subset}):
                group["domains"][domain.split("/", 1)[1]] = fixture_container(
                    [r for r in subset if r["domain"] == domain], candidate, subject=True)
            report["subjects"].append(group)
        for field, dim in (("dataset_summary", "dataset"), ("domain_summary", "domain")):
            for key in sorted({r[dim] for r in rows}):
                report[field][key] = fixture_container([r for r in rows if r[dim] == key], candidate)
        result.append(report)
    return result


def execute(rows=None):
    b, c = reports(rows)
    return audit.audit_saved_reports(b, c, preregistration())


class SavedAuditTests(unittest.TestCase):
    def test_independent_counts_and_research_refusal_are_distinct(self):
        # Same totals as fixed contract, but entirely different procedural fake trials.
        rows = synthetic_rows([("Nakanishi2015", 7, 180), ("Nakanishi2015", 8, 180),
                               ("Wearable", 7, 240), ("Wearable", 8, 240)],
                              baseline_correct=443, switches=list(range(10)) + list(range(443, 452)))
        result = execute(rows)
        global_metrics = result["metrics"]["global"]
        self.assertEqual((global_metrics["trials"], global_metrics["baseline"]["correct_trials"],
                          global_metrics["candidate"]["correct_trials"], global_metrics["disagreements"]),
                         (840, 443, 442, 19))
        self.assertTrue(result["audit_consistent"])
        self.assertFalse(result["research_acceptance"])
        self.assertEqual(result["deployment"], "blocked")
        self.assertFalse(result["runtime_enabled"])
        self.assertFalse(result["production_validated"])

    def test_micro_macro_and_domain_subject_macro(self):
        rows = synthetic_rows([("Wearable", 7, 4), ("Wearable", 8, 8)], baseline_correct=4)
        # Subject 7 dry=1, wet=0; subject 8 dry=0, wet=0. Whole-subject macro is .25, NOT dry .5.
        for r in rows:
            correct = r["subject"] == 7 and r["electrode"] == 0
            r["baseline"] = r["candidate"] = r["target"] if correct else (r["target"] + 1) % 12
        result = execute(rows)
        dry = result["metrics"]["domain"]["Wearable/dry"]
        wet = result["metrics"]["domain"]["Wearable/wet"]
        self.assertAlmostEqual(dry["baseline"]["micro_accuracy"], 1 / 3)
        self.assertEqual(dry["baseline"]["macro_accuracy"], 0.5)
        self.assertEqual(wet["baseline"]["micro_accuracy"], 0)
        self.assertEqual(wet["baseline"]["macro_accuracy"], 0)
        self.assertEqual(result["metrics"]["global"]["baseline"]["macro_accuracy"], 0.25)
        unequal = execute(synthetic_rows([("Nakanishi2015", 7, 2), ("Nakanishi2015", 8, 8)], baseline_correct=2))
        self.assertEqual(unequal["metrics"]["global"]["baseline"]["micro_accuracy"], 0.2)
        self.assertEqual(unequal["metrics"]["global"]["baseline"]["macro_accuracy"], 0.5)

    def test_ratio_mean_of_ratios_not_ratio_of_means(self):
        rows = synthetic_rows([("Nakanishi2015", 7, 2)])
        rows[0].update(baseline_score=1, candidate_score=2)
        rows[1].update(baseline_score=10, candidate_score=5)
        result = execute(rows)
        self.assertEqual(result["metrics"]["global"]["target_score_ratio"]["mean"], 1.25)
        self.assertTrue(result["audit_consistent"])

    def test_summary_tampering_does_not_supply_audited_metrics(self):
        for mutation in ("global", "subject", "dataset", "domain", "ratio", "per_trial_ratio"):
            with self.subTest(mutation=mutation):
                b, c = reports()
                if mutation == "global":
                    c["global_accuracy_summary"]["candidates"][0]["metrics"]["micro_accuracy"] = 0.123
                elif mutation == "subject":
                    b["subjects"][0]["baseline"]["correct_trials"] = 0
                elif mutation == "dataset":
                    b["dataset_summary"]["Nakanishi2015"]["baseline"]["macro_accuracy"] = 0.25
                elif mutation == "domain":
                    c["domain_summary"]["Nakanishi2015/electrode_unknown"]["candidates"][0]["metrics"]["macro_accuracy"] = 0.4
                elif mutation == "ratio":
                    c["subjects"][0]["candidates"][0]["spectral_score_ratio"]["mean"] = 0.1
                else:
                    c["subjects"][0]["candidates"][0]["spectral_score_ratio"]["per_trial"][0] = 0.3
                result = audit.audit_saved_reports(b, c, preregistration())
                self.assertFalse(result["audit_consistent"])
                self.assertFalse(result["research_acceptance"])
                self.assertTrue(result["research_gates_passed"])
                self.assertEqual(result["metrics"]["global"]["candidate"]["micro_accuracy"], 1)
                self.assertEqual(result["metrics"]["global"]["target_score_ratio"]["mean"], 1)
                self.assertTrue(result["summary_mismatches"])

    def test_no_saved_scores_is_explicitly_unverifiable(self):
        b, c = reports()
        c["subjects"][0]["candidates"][0].pop("target_frequency_power")
        result = audit.audit_saved_reports(b, c, preregistration())
        ratio = result["metrics"]["global"]["target_score_ratio"]
        self.assertFalse(ratio["verifiable"])
        self.assertIsNone(ratio["mean"])
        self.assertIn("summaries not substituted", ratio["reason"])
        self.assertFalse(result["research_acceptance"])
        self.assertIn("UNVERIFIABLE", audit.render_markdown(result))
        for r in (b, c):
            r["subjects"][0]["trials"].pop("baseline_target_frequency_power")
        result = audit.audit_saved_reports(b, c, preregistration())
        self.assertFalse(result["metrics"]["global"]["target_score_ratio"]["gate_verifiable"])

    def test_zero_and_epsilon_scores_are_undefined_not_one(self):
        for denominator in (0.0, audit.EPSILON):
            with self.subTest(denominator=denominator):
                rows = synthetic_rows()
                rows[0]["baseline_score"] = denominator
                result = execute(rows)
                ratio = result["metrics"]["global"]["target_score_ratio"]
                self.assertEqual(ratio["undefined_trials"], 1)
                self.assertFalse(ratio["gate_verifiable"])
                self.assertFalse(result["research_acceptance"])

    def test_exact_gate_boundaries(self):
        equal = execute(synthetic_rows(switches=[0], ratio=0.95))
        self.assertTrue(equal["research_acceptance"])
        above = execute(synthetic_rows(switches=[0, 1], ratio=0.95))
        self.assertFalse(above["research_acceptance"])
        for key in ("validation_micro_accuracy_drop", "validation_macro_accuracy_drop", "validation_prediction_disagreement"):
            self.assertTrue(equal["research_gates"][key]["passed"])
            self.assertFalse(above["research_gates"][key]["passed"])
        below = execute(synthetic_rows(ratio=math.nextafter(0.95, 0.0)))
        self.assertFalse(below["research_gates"]["validation_target_score_ratio"]["passed"])

    def test_single_below_ratio_boundary_is_not_rounded_up(self):
        rows = synthetic_rows(ratio=0.95)
        rows[0]["candidate_score"] = math.nextafter(0.95, 0.0)
        result = execute(rows)
        # The rounded displayed mean can be .95, but the exact gate must fail.
        self.assertFalse(result["research_gates"]["validation_target_score_ratio"]["passed"])

    def test_pure_audit_never_opens_paths(self):
        b, c = reports()
        with mock.patch("builtins.open", side_effect=AssertionError("forbidden input read")), \
                mock.patch.object(Path, "open", side_effect=AssertionError("forbidden input read")):
            result = audit.audit_saved_reports(b, c, preregistration())
        self.assertTrue(result["audit_consistent"])

    def test_improvement_is_not_absolute_drop(self):
        result = execute(synthetic_rows(baseline_correct=0, switches=range(50)))
        for key in ("validation_micro_accuracy_drop", "validation_macro_accuracy_drop"):
            self.assertTrue(result["research_gates"][key]["passed"])
            self.assertEqual(result["research_gates"][key]["observed_drop"], -1)
        self.assertFalse(result["research_gates"]["validation_prediction_disagreement"]["passed"])

    def test_order_changes_are_joined_by_trial_identity(self):
        b, c = reports()
        for values in b["subjects"][0]["trials"].values():
            values.reverse()
        self.assertTrue(audit.audit_saved_reports(b, c, preregistration())["audit_consistent"])

    def test_unverifiable_confidence_is_not_independent_evidence(self):
        b, c = reports()
        c["subjects"][0]["candidates"][0]["metrics"]["mean_confidence"] = 0.9
        result = audit.audit_saved_reports(b, c, preregistration())
        self.assertTrue(any("mean_confidence" in s for s in result["unverifiable_summary_fields"]))


class RejectionTests(unittest.TestCase):
    def test_duplicate_identities(self):
        for variant in ("trial", "subject"):
            b, c = reports()
            if variant == "trial":
                c["subjects"][0]["trials"]["trial_index"][1] = 0
            else:
                c["subjects"].append(copy.deepcopy(c["subjects"][0]))
                c["selected_subjects"] += 1
            with self.subTest(variant=variant), self.assertRaisesRegex(audit.AuditError, "duplicate"):
                audit.audit_saved_reports(b, c, preregistration())

    def test_trial_mismatch(self):
        for field, value in (("trial_index", 900), ("target_id", 1), ("block_id", 9),
                             ("baseline_prediction", 1), ("baseline_target_frequency_power", 2.0)):
            b, c = reports()
            c["subjects"][0]["trials"][field][0] = value
            if field == "target_id":
                c["subjects"][0]["trials"]["frequency_hz"][0] = audit.FREQUENCIES[value]
            with self.subTest(field=field), self.assertRaises(audit.AuditError):
                audit.audit_saved_reports(b, c, preregistration())
        b, c = reports(synthetic_rows([("Wearable", 7, 50)]))
        c["subjects"][0]["trials"]["electrode_id"][0] = 1
        with self.assertRaisesRegex(audit.AuditError, "reports differ"):
            audit.audit_saved_reports(b, c, preregistration())

    def test_label_prediction_and_integer_types_and_ranges(self):
        for field in ("target_id", "baseline_prediction", "trial_index", "block_id", "electrode_id"):
            for value in (True, False, 1.0, "1", None, -2):
                with self.subTest(field=field, value=value):
                    b, c = reports()
                    c["subjects"][0]["trials"][field][0] = value
                    with self.assertRaises(audit.AuditError):
                        audit.audit_saved_reports(b, c, preregistration())
        for field in ("target_id", "baseline_prediction"):
            b, c = reports()
            c["subjects"][0]["trials"][field][0] = 12
            with self.assertRaises(audit.AuditError):
                audit.audit_saved_reports(b, c, preregistration())
        for value in (True, False, 12, -1, 2.0, "2", None):
            with self.subTest(candidate_prediction=value):
                b, c = reports()
                c["subjects"][0]["candidates"][0]["predictions"][0] = value
                with self.assertRaises(audit.AuditError):
                    audit.audit_saved_reports(b, c, preregistration())

    def test_invalid_numeric_values_everywhere(self):
        for field in ("baseline_target_frequency_power", "frequency_hz"):
            for value in (True, float("nan"), float("inf"), -float("inf"), -1.0):
                b, c = reports()
                c["subjects"][0]["trials"][field][0] = value
                with self.subTest(field=field, value=value), self.assertRaises(audit.AuditError):
                    audit.audit_saved_reports(b, c, preregistration())
        for value in (True, float("nan"), float("inf"), -1.0):
            b, c = reports()
            c["subjects"][0]["candidates"][0]["target_frequency_power"][0] = value
            with self.assertRaises(audit.AuditError):
                audit.audit_saved_reports(b, c, preregistration())
        b, c = reports()
        c["micro_accuracy"] = float("nan")
        with self.assertRaises(audit.AuditError):
            audit.audit_saved_reports(b, c, preregistration())

    def test_wrong_split_and_holdout_and_flags(self):
        for split in ("train", "holdout", "test", None):
            for level in ("root", "subject"):
                b, c = reports()
                (c if level == "root" else c["subjects"][0])["split"] = split
                with self.subTest(split=split, level=level), self.assertRaises(audit.AuditError):
                    audit.audit_saved_reports(b, c, preregistration())
        for sid in (1, 6, 9, 10, True, 7.0):
            b, c = reports()
            c["subjects"][0]["subject_id"] = sid
            with self.subTest(sid=sid), self.assertRaises(audit.AuditError):
                audit.audit_saved_reports(b, c, preregistration())
        for flag in ("holdout_loaded", "runtime_enabled", "production_validated", "training_performed"):
            for value in (True, 0, None):
                b, c = reports()
                c[flag] = value
                with self.subTest(flag=flag, value=value), self.assertRaises(audit.AuditError):
                    audit.audit_saved_reports(b, c, preregistration())

    def test_schema_failures(self):
        mutations = [lambda c: c.update(schema_version=True), lambda c: c.update(schema_version=2),
                     lambda c: c.update(status="draft"), lambda c: c.update(evaluation_version="other"),
                     lambda c: c.update(subjects={}), lambda c: c.update(selected_subjects=True),
                     lambda c: c.update(selected_subjects=5), lambda c: c.update(candidate_dirs=["D:/x.npz"]),
                     lambda c: c["subjects"][0]["trials"].pop("trial_index"),
                     lambda c: c["subjects"][0]["trials"].update(waveforms=[]),
                     lambda c: c["subjects"][0]["trials"]["target_id"].pop(),
                     lambda c: c["subjects"][0]["candidates"][0]["predictions"].pop(),
                     lambda c: c["subjects"][0]["candidates"].clear(),
                     lambda c: c["subjects"][0]["candidates"][0].update(candidate_id="foreign"),
                     lambda c: c["decoder"].update(epsilon=True),
                     lambda c: c["domain_summary"].clear(),
                     lambda c: c.update(saved_waveforms=[]),
                     lambda c: c["subjects"][0].update(unrecognized_field=1),
                     lambda c: c["subjects"][0]["candidates"][0].update(unrecognized_field=1),
                     lambda c: c["subjects"][0]["baseline"].update(unrecognized_metric=1),
                     lambda c: c["subjects"][0]["candidates"][0]["spectral_score_ratio"].update(defined_trials=True),
                     lambda c: c["subjects"][0].update(source_stimulation_window={"start_inclusive": True, "end_exclusive": 4})]
        for i, mutation in enumerate(mutations):
            b, c = reports()
            mutation(c)
            with self.subTest(i=i), self.assertRaises(audit.AuditError):
                audit.audit_saved_reports(b, c, preregistration())

    def test_preregistration_gate_changes_and_boolean_numbers(self):
        for key in audit.GATE_KEYS:
            for value in (True, 0.03):
                p = preregistration()
                p["predeclared_research_gates"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(audit.AuditError):
                    audit.validate_preregistration(p)
        p = preregistration()
        p["fixed_reference_contract"]["subject_partition"]["validation"] = [9, 10]
        with self.assertRaises(audit.AuditError):
            audit.validate_preregistration(p)

    def test_duplicate_json_keys_nonfinite_and_schema_roots(self):
        for data in ('{"x":1,"x":2}', '{"nested":{"x":1,"x":2}}', '{"v":NaN}',
                     '{"v":Infinity}', '{"v":-Infinity}', '{"v":1e999}', '[]', 'null', '{bad'):
            with self.subTest(data=data), self.assertRaises(audit.AuditError):
                audit.strict_json_loads(data)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-v5-audit-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result = execute()

    def test_json_and_markdown_atomic_new_outputs(self):
        paths = audit.publish_reports(self.root, self.result, "new.json", "new.md")
        self.assertEqual(json.loads(paths[0].read_text(encoding="utf-8")), self.result)
        self.assertIn("within the listed scope/domain", paths[1].read_text(encoding="utf-8"))
        self.assertEqual(paths[0].stat().st_nlink, 1)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["new.json", "new.md"])

    def test_existing_either_output_is_never_overwritten(self):
        for existing in ("old.json", "old.md"):
            with self.subTest(existing=existing):
                path = self.root / existing
                path.write_bytes(b"original")
                with self.assertRaises(audit.AuditError):
                    audit.publish_reports(self.root, self.result, "old.json", "old.md")
                self.assertEqual(path.read_bytes(), b"original")
                self.assertEqual(list(self.root.iterdir()), [path])
                path.unlink()

    def test_second_publication_failure_rolls_back_first(self):
        real_link = os.link
        def fail_second(src, dst, **kwargs):
            if Path(dst).suffix == ".md":
                raise OSError("synthetic disk failure")
            return real_link(src, dst, **kwargs)
        with mock.patch.object(audit.os, "link", side_effect=fail_second):
            with self.assertRaisesRegex(audit.AuditError, "publication failed"):
                audit.publish_reports(self.root, self.result, "new.json", "new.md")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_racing_existing_output_is_preserved(self):
        real_link = os.link
        def race(src, dst, **kwargs):
            if Path(dst).suffix == ".md":
                Path(dst).write_bytes(b"another writer")
            return real_link(src, dst, **kwargs)
        with mock.patch.object(audit.os, "link", side_effect=race):
            with self.assertRaises(audit.AuditError):
                audit.publish_reports(self.root, self.result, "new.json", "new.md")
        self.assertEqual((self.root / "new.md").read_bytes(), b"another writer")
        self.assertFalse((self.root / "new.json").exists())
        self.assertFalse(list(self.root.glob(".ssvep_saved_audit_v5-*.tmp")))

    def test_outside_traversal_ads_reserved_missing_parents_and_extensions(self):
        for path in (self.root.parent / "escape.json", "../escape.json", "sub/../escape.json",
                     "D:/waveforms.json", "NUL.json", "a.json:stream", "a. /b.json", "missing/a.json"):
            with self.subTest(path=path), self.assertRaises(audit.AuditError):
                audit.workspace_path(self.root, path, output=True)
        with self.assertRaises(audit.AuditError):
            audit.publish_reports(self.root, self.result, "a.npz", "a.md")
        with self.assertRaises(audit.AuditError):
            audit.publish_reports(self.root, self.result, "same.json", "same.json")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_symlink_parent_rejected(self):
        (self.root / "actual").mkdir()
        try:
            (self.root / "linked").symlink_to(self.root / "actual", target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"OS does not allow symlink creation: {exc}")
        with self.assertRaisesRegex(audit.AuditError, "symlink/reparse"):
            audit.publish_reports(self.root, self.result, "linked/a.json", "linked/a.md")

    def test_symlink_mode_parent_rejected_without_os_privilege(self):
        parent = self.root / "mock-symlink"
        parent.mkdir()
        real_lstat = Path.lstat
        def fake_lstat(path, *args, **kwargs):
            original = real_lstat(path, *args, **kwargs)
            if path == parent:
                info = mock.Mock()
                info.st_mode = 0o120777  # S_IFLNK; simulated independently of reparse attributes.
                info.st_file_attributes = 0
                return info
            return original
        with mock.patch.object(Path, "lstat", fake_lstat):
            with self.assertRaisesRegex(audit.AuditError, "symlink/reparse"):
                audit.workspace_path(self.root, "mock-symlink/a.json", output=True)

    def test_reparse_parent_rejected_even_without_symlink_mode(self):
        parent = self.root / "reparse"
        parent.mkdir()
        real_lstat = Path.lstat
        def fake_lstat(path, *args, **kwargs):
            original = real_lstat(path, *args, **kwargs)
            if path == parent:
                info = mock.Mock()
                info.st_mode = original.st_mode
                info.st_file_attributes = 0x400
                return info
            return original
        with mock.patch.object(Path, "lstat", fake_lstat):
            with self.assertRaisesRegex(audit.AuditError, "symlink/reparse"):
                audit.workspace_path(self.root, "reparse/a.json", output=True)

    def test_linked_input_rejected(self):
        (self.root / "a.json").write_text("{}", encoding="utf-8")
        os.link(self.root / "a.json", self.root / "hard.json")
        with self.assertRaisesRegex(audit.AuditError, "non-linked"):
            audit.workspace_path(self.root, "a.json")

    def test_invalid_payload_publishes_nothing(self):
        bad = copy.deepcopy(self.result)
        bad["bad"] = float("nan")
        with self.assertRaises(ValueError):
            audit.publish_reports(self.root, bad, "new.json", "new.md")
        self.assertEqual(list(self.root.iterdir()), [])


class JSONOnlyProductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-v5-json-only-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "analysis_outputs").mkdir()
        rows = synthetic_rows([("Nakanishi2015", 7, 180), ("Nakanishi2015", 8, 180),
                               ("Wearable", 7, 240), ("Wearable", 8, 240)],
                              baseline_correct=443, switches=list(range(10)) + list(range(443, 452)))
        baseline, candidate = reports(rows)
        self.documents = {audit.PREREGISTRATION: preregistration(), audit.BASELINE: baseline,
                          audit.CANDIDATE: candidate}
        self.hashes = {k: hashlib.sha256(json.dumps(v).encode()).hexdigest() for k, v in self.documents.items()}
        self.documents[audit.FINAL_VERIFICATION] = {
            "schema_version": 1, "status": "formal", "verification_passed": True,
            "artifact_sha256": {audit.BASELINE: self.hashes[audit.BASELINE],
                                audit.PREREGISTRATION: self.hashes[audit.PREREGISTRATION]},
            "holdout_waveforms_loaded": False, "runtime_enabled": False, "production_validated": False,
            "protected_baseline": {"path": "D:/must-not-read/lock.json"}}
        self.documents[audit.EVALUATION_LOCK] = {
            "schema_version": 1, "status": "formal", "split": "validation", "candidate_subjects": [7, 8],
            "files": {audit.BASELINE: self.hashes[audit.BASELINE], "D:/must-not-read/model.pt": "unused"},
            "holdout_waveforms_loaded": False, "runtime_enabled": False, "production_validated": False,
            "candidate_dir": "D:/must-not-read/holdout"}
        self.write_documents()
        self.pin_patches = mock.patch.multiple(
            audit, PREREGISTRATION_SHA256=self.hashes[audit.PREREGISTRATION],
            BASELINE_SHA256=self.hashes[audit.BASELINE], CANDIDATE_SHA256=self.hashes[audit.CANDIDATE])
        self.pin_patches.start()
        self.addCleanup(self.pin_patches.stop)

    def write_documents(self):
        for key, value in self.documents.items():
            (self.root / key).write_text(json.dumps(value), encoding="utf-8")

    def test_fixed_json_allowlist_and_no_internal_reference_access(self):
        real_open = Path.open
        reads = []
        def checked_open(path, *args, **kwargs):
            relative = path.relative_to(self.root).as_posix()
            self.assertIn(relative, audit.FIXED_INPUTS)
            self.assertEqual(path.suffix, ".json")
            reads.append(relative)
            return real_open(path, *args, **kwargs)
        with mock.patch.object(Path, "open", checked_open):
            result = audit.audit_workspace(self.root)
        self.assertEqual(reads, list(audit.FIXED_INPUTS))
        self.assertTrue(result["hash_binding_verified"])
        self.assertTrue(result["audit_consistent"])
        self.assertFalse(result["research_acceptance"])
        self.assertFalse(result["waveforms_loaded"])
        self.assertFalse(result["holdout_loaded"])
        self.assertFalse(result["d_drive_accessed"])

    def test_foreign_input_path_rejected_before_open(self):
        for path in ("D:/raw.npz", "analysis_outputs/unapproved.json", "../x.json", "holdout.npz"):
            with mock.patch.object(Path, "open", side_effect=AssertionError("must not open")):
                with self.subTest(path=path), self.assertRaises(audit.AuditError):
                    audit._read_fixed_json(self.root, path)

    def test_all_three_hash_pins(self):
        for key in (audit.PREREGISTRATION, audit.BASELINE, audit.CANDIDATE):
            self.write_documents()
            path = self.root / key
            path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.subTest(key=key), self.assertRaisesRegex(audit.AuditError, "SHA-256 mismatch"):
                audit.audit_workspace(self.root)

    def test_both_baseline_hash_binding_sources_must_agree(self):
        for key, field in ((audit.FINAL_VERIFICATION, "artifact_sha256"), (audit.EVALUATION_LOCK, "files")):
            original = self.documents[key][field][audit.BASELINE]
            self.documents[key][field][audit.BASELINE] = "0" * 64
            self.write_documents()
            with self.subTest(key=key), self.assertRaisesRegex(audit.AuditError, "binding differs"):
                audit.audit_workspace(self.root)
            self.documents[key][field][audit.BASELINE] = original

    def test_cli_is_fixed_workspace_and_cannot_accept_input_override(self):
        script = Path(__file__).absolute().parents[1] / "scripts" / "audit_ssvep_saved_predictions_v5.py"
        spec = importlib.util.spec_from_file_location("saved_audit_cli_synthetic_test", script)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        self.assertEqual(cli.WORKSPACE, Path(__file__).absolute().parents[1])
        with mock.patch.object(cli, "WORKSPACE", self.root), mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                cli.main(["--workspace", "D:/", "--output-json", "a.json", "--output-markdown", "a.md"])
            with self.assertRaises(SystemExit):
                cli.main(["--baseline", "D:/raw.npz", "--output-json", "a.json", "--output-markdown", "a.md"])
        with mock.patch.object(cli, "WORKSPACE", self.root), mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(cli.main(["--output-json", "new.json", "--output-markdown", "new.md"]), 0)
        # Running against SYNTHETIC temp JSON only; no real audit output is created.
        self.assertEqual(json.loads((self.root / "new.json").read_text(encoding="utf-8"))["deployment"], "blocked")
        with mock.patch.object(cli, "WORKSPACE", self.root), mock.patch("sys.stderr", new_callable=io.StringIO):
            with mock.patch.object(cli, "audit_workspace", side_effect=AssertionError("precheck first")):
                self.assertEqual(cli.main(["--output-json", "new.json", "--output-markdown", "new.md"]), 2)

    def test_loader_duplicate_keys_and_schema_rejected_before_hash_acceptance(self):
        path = self.root / audit.BASELINE
        path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
        with self.assertRaisesRegex(audit.AuditError, "duplicate JSON key"):
            audit.audit_workspace(self.root)

    def test_stdlib_only_no_old_evaluator_or_waveform_imports(self):
        source = Path(audit.__file__).read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module)
        self.assertEqual(imports, {"__future__", "hashlib", "json", "math", "os", "pathlib",
                                   "stat", "statistics", "tempfile", "fractions"})
        self.assertNotIn("np.load", source)
        self.assertNotIn("ssvep_task_eval_v5 import", source)


if __name__ == "__main__":
    unittest.main()
