"""Contract checks use synthetic temporary sessions, never personal/test EEG."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eeg_tools import personal_ssvep_session_contract as contract
from scripts import validate_personal_ssvep_session_contract as cli


class PersonalSSVEPSessionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.session = self.root / "session_001"
        self.session.mkdir()
        self.sample_count = 12100
        self.write_json("session.json", {
            "session_id": "session_001", "status": "completed",
            "validation_mode": "formal_candidate", "simulated": False,
            "board": "cyton", "sampling_rate_hz": 250, "channel_count": 8,
            "recorded_samples_per_channel": self.sample_count,
        })
        self.write_json("channel_config.json", {
            "channel_order": [f"CH{i}" for i in range(1, 9)],
            "channels": [{"name": f"CH{i}"} for i in range(1, 9)],
        })
        self.write_json("ssvep_config.json", {
            "status": "formal", "targets": [{"id": "A", "frequency_hz": 10}],
        })
        self.write_json("quality.json", {
            "status": "ok", "recording_status": "completed",
            "timestamp_gap_count": 0, "frame_timing": {"cadence_status": "pass"},
        })
        self.write_json("preflight.json", {"checks": [{"name": "handshake", "status": "passed"}]})
        header = "\t".join(["sample_index"] + [f"eeg_ch{i}" for i in range(1, 9)])
        with (self.session / "raw_brainflow.tsv").open("w", encoding="utf-8", newline="") as handle:
            handle.write(header + "\n")
            for i in range(self.sample_count):
                handle.write(str(i) + "\t" + "\t".join(str((i % 37) + j * 0.1) for j in range(8)) + "\n")
        self.events = []
        for i in range(8):
            start = 100 + i * 1500
            for event, sample in (("stimulus_onset", start), ("stimulus_offset", start + 1250)):
                self.events.append({"event_name": event, "trial_index": str(i), "target_id": "A",
                                    "presented_target_id": "A", "gaze_target_id": "", "source": "software_protocol",
                                    "label_source": "", "sample_index": str(sample), "frequency_hz": "10"})
        self.write_events()

    def write_json(self, name, data):
        (self.session / name).write_text(json.dumps(data), encoding="utf-8")

    def update_json(self, name, **changes):
        data = json.loads((self.session / name).read_text(encoding="utf-8"))
        data.update(changes)
        self.write_json(name, data)

    def write_events(self):
        import csv
        fields = sorted(contract.CORE_EVENT_FIELDS)
        with (self.session / "events.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(self.events)

    def write_gaze(self):
        evidence = []
        for role in sorted(contract.GAZE_EVIDENCE_ROLES):
            path = self.session / (role + ".txt")
            # Synthetic traceability fixture, not actual valid gaze evidence.
            path.write_text("synthetic fixture: " + role, encoding="utf-8")
            evidence.append({"role": role, "path": path.name,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        self.write_json("gaze_truth.json", {
            "schema_version": 1, "session_id": "session_001", "source_type": "eye_tracker",
            "independent": True, "evidence_files": evidence,
            "trial_labels": [{"trial_index": i, "target_id": "A", "start_sample_index": 100 + i * 1500,
                              "end_sample_index": 1350 + i * 1500} for i in range(8)],
        })

    def test_complete_proxy_input_never_implies_gaze_or_runtime_acceptance(self):
        before = {path.name: path.read_bytes() for path in self.session.iterdir()}
        report = contract.audit_session(self.session)
        self.assertTrue(report["eligibility"]["proxy_calibration_candidate"])
        self.assertFalse(report["eligibility"]["independent_personal_validation_candidate"])
        self.assertFalse(report["eligibility"]["runtime_enabled"])
        self.assertFalse(report["eligibility"]["production_validated"])
        self.assertFalse(report["eligibility"]["default_deployment_authorized"])
        self.assertFalse(report["raw"]["signal_quality_accepted"])
        self.assertEqual(report["validation_mode_recorded"], "formal_candidate")
        self.assertEqual(report["label_evidence"]["presented_target_count"], 8)
        self.assertTrue(report["label_evidence"]["presented_target_is_proxy"])
        self.assertTrue(report["source_files_unchanged"])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.session.iterdir()})

    def test_keyboard_marks_are_counted_as_separate_events_without_promoting_truth(self):
        for index in (0, 1):
            self.events.append({**self.events[2 * index], "event_name": "gaze_target_mark",
                                "presented_target_id": "", "gaze_target_id": "A",
                                "source": "manual_gaze_annotation", "label_source": "keyboard_self_report_or_operator"})
        self.write_events()
        report = contract.audit_session(self.session)
        self.assertEqual(report["label_evidence"]["manual_self_report_count"], 2)
        self.assertFalse(report["label_evidence"]["independent_gaze"]["independent"])

    def test_missing_protocol_class_and_insufficient_repetitions_are_blockers(self):
        self.update_json("ssvep_config.json", targets=[{"id": "A", "frequency_hz": 10}, {"id": "B", "frequency_hz": 12}])
        report = contract.audit_session(self.session)
        self.assertEqual(report["events"]["target_counts"], {"A": 8, "B": 0})
        self.assertIn("minimum_repetitions_each_protocol_target", report["blocking_checks"])

    def test_real_cadence_pass_is_accepted_and_review_is_not(self):
        self.update_json("quality.json", frame_timing={"cadence_status": "review"}, timestamp_gap_count=2)
        report = contract.audit_session(self.session)
        self.assertIn("display_cadence_pass", report["blocking_checks"])
        self.assertIn("timestamp_gaps_zero", report["blocking_checks"])

    def test_bad_intervals_and_duplicates_block_proxy_candidate(self):
        for mode in ("duplicate", "reversed", "outside", "short", "frequency"):
            with self.subTest(mode=mode):
                original = json.loads(json.dumps(self.events))
                if mode == "duplicate":
                    self.events.append(dict(self.events[0]))
                elif mode == "reversed":
                    self.events[1]["sample_index"] = "99"
                elif mode == "outside":
                    self.events[1]["sample_index"] = str(self.sample_count)
                elif mode == "short":
                    self.events[1]["sample_index"] = "1100"
                else:
                    self.events[0]["frequency_hz"] = "12"
                self.write_events()
                report = contract.audit_session(self.session)
                self.assertIn("stimulus_pairs_valid", report["blocking_checks"])
                self.events = original

    def test_wrong_channel_contract_and_metadata_count_are_blockers(self):
        self.update_json("channel_config.json", channel_order=["CH1"] * 8)
        self.update_json("session.json", recorded_samples_per_channel=1)
        report = contract.audit_session(self.session)
        self.assertIn("channel_order_ch1_to_ch8", report["blocking_checks"])
        self.assertIn("recorded_sample_count_matches", report["blocking_checks"])

    def test_incomplete_raw_and_missing_events_header_are_rejected(self):
        raw_path = self.session / "raw_brainflow.tsv"
        original = raw_path.read_bytes()
        raw_path.write_text("sample_index\teeg_ch1\n0\t0\n", encoding="utf-8")
        with self.assertRaisesRegex(contract.PersonalSSVEPSessionContractError, "exactly eeg_ch1..8"):
            contract.audit_session(self.session)
        raw_path.write_bytes(original)
        (self.session / "events.tsv").write_text("event_name\ttrial_index\n", encoding="utf-8")
        with self.assertRaisesRegex(contract.PersonalSSVEPSessionContractError, "missing required label/event fields"):
            contract.audit_session(self.session)

    def test_json_rejects_duplicate_keys_and_nonfinite_values(self):
        for payload in ('{"status":"completed","status":"completed"}', '{"sampling_rate_hz":NaN}'):
            with self.subTest(payload=payload):
                (self.session / "session.json").write_text(payload, encoding="utf-8")
                with self.assertRaises(contract.PersonalSSVEPSessionContractError):
                    contract.audit_session(self.session)

    def test_eye_tracker_declaration_alone_does_not_satisfy_evidence_contract(self):
        self.write_json("gaze_truth.json", {"schema_version": 1, "session_id": "session_001",
                                           "source_type": "eye_tracker", "independent": True,
                                           "trial_labels": [{"trial_index": 0, "target_id": "A"}]})
        gaze = contract.audit_session(self.session)["label_evidence"]["independent_gaze"]
        self.assertTrue(gaze["independent_declared"])
        self.assertFalse(gaze["evidence_contract_passed"])
        self.assertFalse(gaze["independent"])

    def test_traceable_gaze_files_still_require_independent_review(self):
        self.write_gaze()
        report = contract.audit_session(self.session)
        gaze = report["label_evidence"]["independent_gaze"]
        self.assertTrue(gaze["evidence_contract_passed"])
        self.assertEqual(gaze["trial_label_count"], 8)
        self.assertFalse(gaze["independent"])
        self.assertFalse(report["eligibility"]["independent_personal_validation_candidate"])
        self.assertTrue(all(role + ".txt" in report["source_sha256"] for role in contract.GAZE_EVIDENCE_ROLES))

    def test_gaze_duplicate_unknown_targets_and_invalid_time_ranges_do_not_pass(self):
        self.write_gaze()
        original = json.loads((self.session / "gaze_truth.json").read_text(encoding="utf-8"))
        for mode in ("duplicate", "target", "interval", "session", "hash"):
            with self.subTest(mode=mode):
                data = json.loads(json.dumps(original))
                if mode == "duplicate":
                    data["trial_labels"][1] = dict(data["trial_labels"][0])
                elif mode == "target":
                    data["trial_labels"][0]["target_id"] = "unknown"
                elif mode == "interval":
                    data["trial_labels"][0]["end_sample_index"] = 9999
                elif mode == "session":
                    data["session_id"] = "other"
                else:
                    data["evidence_files"][0]["sha256"] = "f" * 64
                self.write_json("gaze_truth.json", data)
                gaze = contract.audit_session(self.session)["label_evidence"]["independent_gaze"]
                self.assertFalse(gaze["evidence_contract_passed"])

    def test_evidence_paths_cannot_escape_session(self):
        self.write_gaze()
        data = json.loads((self.session / "gaze_truth.json").read_text(encoding="utf-8"))
        data["evidence_files"][0]["path"] = "../outside.txt"
        self.write_json("gaze_truth.json", data)
        with self.assertRaisesRegex(contract.PersonalSSVEPSessionContractError, "inside the session"):
            contract.audit_session(self.session)

    def test_optional_gaze_sidecar_mutation_is_detected(self):
        self.write_gaze()
        original = contract._gaze_evidence
        def changed(*args):
            result = original(*args)
            self.update_json("gaze_truth.json", session_id="changed-during-audit")
            return result
        with patch.object(contract, "_gaze_evidence", side_effect=changed):
            with self.assertRaisesRegex(contract.PersonalSSVEPSessionContractError, "source session changed"):
                contract.audit_session(self.session)

    def test_cli_rejects_in_session_output_and_does_not_overwrite_reports(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["--session", str(self.session), "--output", str(self.session / "audit.json")]), 1)
        self.assertFalse((self.session / "audit.json").exists())
        output = self.root / "report.json"
        args = ["--session", str(self.session), "--output", str(output)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(args), 0)
        original = output.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(args), 1)
        self.assertEqual(original, output.read_bytes())


if __name__ == "__main__":
    unittest.main()
