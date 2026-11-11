"""Post-hoc validation diagnostic for two-window SSVEP move confirmation.

This reads only the saved S25-S29 validation report for the isolated 4 s
candidate. It does not fit a model, tune thresholds, read final-test subjects,
read personal sessions, or execute game commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import benchmark_matched_public_ssvep as benchmark

DEFAULT_INPUT = (Path(__file__).resolve().parents[1] / "analysis_outputs"
                 / "wang2016_spectral_mlp_live_4s_20261015" / "validation_report.json")
DEFAULT_OUTPUT = (Path(__file__).resolve().parents[1] / "analysis_outputs"
                  / "wang2016_spectral_mlp_live_4s_20261015"
                  / "temporal_confirmation_diagnostic.json")
OFFSETS = (0, 250)
MOVE_COMMANDS = frozenset(("MOVE_LEFT", "MOVE_RIGHT"))
EXPECTED_ACTION = {10: "MOVE_LEFT", 12: "MOVE_RIGHT", 15: "HOLD"}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def row_action(row: dict) -> str:
    raw = row.get("raw")
    require(isinstance(raw, dict), "validation row lacks raw model output")
    require(type(row.get("ood_passed")) is bool, "validation row lacks fixed OOD result")
    if not row["ood_passed"] or raw.get("accepted") is not True:
        return "HOLD"
    command = raw.get("command")
    return command if command in MOVE_COMMANDS else "HOLD"


def summarize(trials: list[dict]) -> dict:
    total = len(trials)
    require(total > 0, "no complete source trials")
    correct = sum(row["action"] == EXPECTED_ACTION[row["frequency_hz"]] for row in trials)
    false_move = sum(row["action"] in MOVE_COMMANDS
                     and row["action"] != EXPECTED_ACTION[row["frequency_hz"]]
                     for row in trials)
    predicted = defaultdict(int)
    confusion = {str(frequency): {action: 0 for action in ("MOVE_LEFT", "MOVE_RIGHT", "HOLD")}
                 for frequency in EXPECTED_ACTION}
    for row in trials:
        action = row["action"]
        predicted[action] += 1
        confusion[str(row["frequency_hz"])][action] += 1
    return {
        "source_trial_count": total,
        "action_accuracy_against_presentation_proxy": correct / total,
        "false_move_count": false_move,
        "false_move_rate_against_presentation_proxy": false_move / total,
        "accepted_move_count": sum(row["action"] in MOVE_COMMANDS for row in trials),
        "hold_count": predicted["HOLD"],
        "confusion_by_presented_frequency_hz": confusion,
        "executed": False,
    }


def analyze(report: dict) -> dict:
    require(report.get("scope", "").startswith("Wang2016 cross-subject"),
            "input is not the expected public Wang validation report")
    require(report.get("split") == "validation" and report.get("subjects") == [25, 26, 27, 28, 29],
            "only the locked S25-S29 validation report is accepted")
    require(report.get("window_samples") == 1000 and report.get("window_offsets_samples") == list(OFFSETS),
            "unexpected live-window contract")
    require(report.get("personal_validation") is False and report.get("final_test_evaluated") is False,
            "report flags do not match the bounded validation experiment")
    rows = report.get("rows")
    require(isinstance(rows, list) and rows, "validation rows are missing")
    groups = defaultdict(list)
    for row in rows:
        require(isinstance(row, dict), "invalid validation row")
        key = (row.get("trial_id"), row.get("subject"), row.get("block"), row.get("frequency_hz"))
        require(isinstance(key[0], str) and type(key[1]) is int and type(key[2]) is int
                and key[3] in EXPECTED_ACTION, "invalid source-trial identity or target")
        require(type(row.get("window_start_offset_samples")) is int
                and row["window_start_offset_samples"] in OFFSETS, "unexpected window offset")
        groups[key].append(row)
    trial_rows = []
    for (trial_id, subject, block, frequency), windows in sorted(groups.items()):
        by_offset = {row["window_start_offset_samples"]: row for row in windows}
        require(len(windows) == 2 and set(by_offset) == set(OFFSETS),
                "each source trial must have exactly one row for each locked window")
        first_action, second_action = (row_action(by_offset[offset]) for offset in OFFSETS)
        confirmed = first_action if first_action in MOVE_COMMANDS and first_action == second_action else "HOLD"
        trial_rows.append({
            "trial_id": trial_id, "subject": subject, "block": block,
            "presented_frequency_hz": frequency,
            "first_window_action": first_action,
            "second_window_action": second_action,
            "confirmed_action": confirmed,
            "executed": False,
        })
    methods = {}
    for name, field in (("first_window_only", "first_window_action"),
                        ("second_window_only", "second_window_action"),
                        ("two_consecutive_same_direction", "confirmed_action")):
        selected = [{"frequency_hz": row["presented_frequency_hz"], "action": row[field]}
                    for row in trial_rows]
        methods[name] = summarize(selected)
        methods[name]["latency_note"] = ("one additional rolling-window result; windows overlap by 750/1000 samples"
                                        if name == "two_consecutive_same_direction" else "single window")
    subjects = {str(subject): {
        "source_trial_count": sum(row["subject"] == subject for row in trial_rows),
        "two_window_confirmation": summarize([
            {"frequency_hz": row["presented_frequency_hz"], "action": row["confirmed_action"]}
            for row in trial_rows if row["subject"] == subject
        ]),
    } for subject in sorted({row["subject"] for row in trial_rows})}
    return {
        "status": "formal",
        "exploratory_only": True,
        "analysis_kind": "posthoc_validation_diagnostic_not_model_selection",
        "source_report": None,
        "source_report_sha256": None,
        "source_binding_status": "unbound_in_memory",
        "source_split": "validation_S25-S29",
        "source_trials_not_window_rows": len(trial_rows),
        "rule": "move only when both adjacent 1000-sample windows independently pass the existing fixed gate and predict the same direction; otherwise HOLD",
        "window_offsets_samples": list(OFFSETS),
        "window_overlap_samples": 750,
        "target_semantics": "recorded stimulus frequency is only a presentation-label proxy, not gaze or user-intent truth",
        "methods": methods,
        "per_subject": subjects,
        "trial_rows": trial_rows,
        "personal_validation": False,
        "final_test_evaluated": False,
        "commands_executed": False,
        "runtime_enabled": False,
        "production_validated": False,
        "default_deployment_authorized": False,
        "selection_use": False,
    }


def run(input_path: Path, output_path: Path) -> dict:
    input_path, output_path = Path(input_path).absolute(), Path(output_path).absolute()
    for path in (input_path, output_path):
        require(not any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
                        for p in (path, *path.parents)), "unsafe symlink/junction report path")
    input_path, output_path = input_path.resolve(), output_path.resolve()
    require(input_path.is_file() and not input_path.is_symlink(), "missing or unsafe validation report")
    require(not output_path.exists(), "diagnostic output already exists; refusing overwrite")
    source_bytes = input_path.read_bytes()
    report = json.loads(source_bytes)
    result = analyze(report)
    result["source_report"] = str(input_path)
    result["source_report_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    result["source_binding_status"] = "bound_consumed_file_bytes"
    require(input_path.read_bytes() == source_bytes, "input changed during analysis")
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = run(args.input, args.output)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc), "executed": False},
                         ensure_ascii=False, allow_nan=False))
        return 2
    summary = {"status": result["status"], "output": str(args.output.resolve()),
               "methods": {name: {key: value for key, value in metrics.items()
                                   if key in ("source_trial_count", "action_accuracy_against_presentation_proxy",
                                              "false_move_count", "false_move_rate_against_presentation_proxy",
                                              "accepted_move_count", "hold_count")}
                           for name, metrics in result["methods"].items()},
               "runtime_enabled": result["runtime_enabled"], "commands_executed": False}
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
