"""Read-only dependency identity tests; no private file or historical lock needed."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eeg_tools.evidence_bindings import EvidenceBindings, EvidenceBindingError
from scripts.audit_historical_test_evidence import main


class EvidenceBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="quality-evidence-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root / "old"
        self.new = self.root / "relocated 空格"
        self.new.mkdir()
        self.file = self.new / "report.json"
        self.file.write_bytes(b'{"synthetic":true}')
        self.hash = hashlib.sha256(self.file.read_bytes()).hexdigest()
        self.manifest = {"schema_version": 1,
                         "root_mappings": [{"from": str(self.old), "to": str(self.new)}],
                         "files": [{"id": "report", "recorded_path": str(self.old / self.file.name),
                                    "expected_sha256": self.hash, "identity_authority": "local_inventory"}]}

    def audit(self):
        return EvidenceBindings(self.manifest).audit()

    def test_explicit_relocation_verifies_bytes_but_not_historical_conclusion(self):
        before = self.file.read_bytes()
        result = self.audit()
        self.assertTrue(result["all_declared_bytes_verified"])
        self.assertEqual(result["files"][0]["observed_sha256"], self.hash)
        self.assertEqual(result["files"][0]["identity_authority"], "local_inventory")
        self.assertFalse(result["historical_conclusions_verified"])
        self.assertEqual(self.file.read_bytes(), before)
        self.assertFalse(self.old.exists())

    def test_missing_mapping_does_not_search_for_a_matching_basename(self):
        self.manifest["root_mappings"] = []
        result = self.audit()
        self.assertFalse(result["all_declared_bytes_verified"])
        self.assertEqual(result["files"][0]["expected_sha256"], self.hash)
        self.assertIn("missing bound file", result["files"][0]["reason"])

    def test_tamper_retains_original_expected_identity(self):
        self.file.write_bytes(b'{"tampered":true}')
        result = self.audit()
        self.assertEqual(result["files"][0]["reason"], "content_sha256_mismatch")
        self.assertEqual(self.manifest["files"][0]["expected_sha256"], self.hash)

    def test_unknown_identity_is_not_autopinned_or_hashed_as_success(self):
        self.manifest["files"][0].update(expected_sha256=None, identity_authority="unknown")
        with patch.object(Path, "open", side_effect=AssertionError("unknown file must not be read")):
            result = self.audit()
        self.assertEqual(result["blocked_files"], 1)
        self.assertIsNone(result["files"][0]["observed_sha256"])
        self.assertIn("unknown", result["files"][0]["reason"])

    def test_missing_file_retains_path_and_hash(self):
        self.file.unlink()
        row = self.audit()["files"][0]
        self.assertEqual(row["expected_sha256"], self.hash)
        self.assertIn("report.json", row["reason"])

    def test_overlapping_roots_are_rejected(self):
        self.manifest["root_mappings"].append({"from": str(self.old / "child"), "to": str(self.new)})
        with self.assertRaisesRegex(EvidenceBindingError, "overlapping"):
            EvidenceBindings(self.manifest)

    def test_duplicate_ids_and_recorded_paths_are_rejected(self):
        for field in ("id", "recorded_path"):
            with self.subTest(field=field):
                value = copy.deepcopy(self.manifest)
                second = dict(value["files"][0])
                second.update(id="other", recorded_path=str(self.old / "other"))
                second[field] = value["files"][0][field]
                value["files"].append(second)
                with self.assertRaises(EvidenceBindingError):
                    EvidenceBindings(value)

    def test_aliased_destination_is_not_verified_twice(self):
        alternate = self.root / "alternate"
        self.manifest["root_mappings"].append({"from": str(alternate), "to": str(self.new)})
        self.manifest["files"].append({**self.manifest["files"][0], "id": "alias",
                                      "recorded_path": str(alternate / self.file.name)})
        result = self.audit()
        self.assertEqual((result["verified_files"], result["blocked_files"]), (1, 1))
        self.assertIn("same file", result["files"][1]["reason"])

    def test_invalid_schema_digest_authority_and_traversal_fail_closed(self):
        for mutate in (lambda m: m.update(schema_version=True),
                       lambda m: m["files"][0].update(expected_sha256="A"*64),
                       lambda m: m["files"][0].pop("expected_sha256"),
                       lambda m: m["files"][0].update(identity_authority="unknown"),
                       lambda m: m["files"][0].update(recorded_path="relative.json"),
                       lambda m: m["files"][0].update(recorded_path=str(self.old / ".." / "report.json"))):
            value = copy.deepcopy(self.manifest); mutate(value)
            with self.subTest(value=value), self.assertRaises(EvidenceBindingError):
                EvidenceBindings(value)

    def test_duplicate_json_keys_are_not_silently_overwritten(self):
        path = self.root / "binding.json"
        path.write_text('{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(EvidenceBindingError, "duplicate JSON key"):
            EvidenceBindings.load(path)

    def test_junction_is_rejected_before_file_read(self):
        with patch.object(Path, "is_junction", lambda p: p == self.new, create=True):
            self.assertIn("junction", self.audit()["files"][0]["reason"])

    def test_cli_records_manifest_hash_and_refuses_overwrite(self):
        manifest = self.root / "binding.json"; output = self.root / "audit.json"
        manifest.write_text(json.dumps(self.manifest), encoding="utf-8")
        before = manifest.read_bytes(), self.file.read_bytes()
        self.assertEqual(main(["--manifest", str(manifest), "--output", str(output)]), 0)
        result = json.loads(output.read_text("utf-8"))
        self.assertEqual(result["manifest_sha256"], hashlib.sha256(before[0]).hexdigest())
        with self.assertRaises(SystemExit):
            main(["--manifest", str(manifest), "--output", str(output)])
        self.assertEqual(before, (manifest.read_bytes(), self.file.read_bytes()))

    def test_cli_missing_evidence_is_nonzero_with_blocked_report(self):
        manifest = self.root / "binding.json"; output = self.root / "audit.json"
        manifest.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.file.unlink()
        self.assertEqual(main(["--manifest", str(manifest), "--output", str(output)]), 1)
        self.assertFalse(json.loads(output.read_text("utf-8"))["all_declared_bytes_verified"])

    def test_changed_while_hashing_is_not_verified(self):
        real_sha256 = hashlib.sha256
        target = self.file
        class MutatingHash:
            def __init__(self): self.h = real_sha256()
            def update(self, chunk):
                self.h.update(chunk)
                target.write_bytes(b"changed after the original read")
            def hexdigest(self): return self.h.hexdigest()
        with patch("eeg_tools.evidence_bindings.hashlib.sha256", MutatingHash):
            row = self.audit()["files"][0]
        self.assertEqual(row["status"], "blocked")
        self.assertEqual(row["reason"], "evidence changed during read")

    def test_malformed_or_traversing_root_mapping_rejected(self):
        for value in (None, 3, {}, {"from": 5, "to": str(self.new)},
                      {"from": str(self.old), "to": str(self.new / ".." / "alias")}):
            m = copy.deepcopy(self.manifest); m["root_mappings"] = [value]
            with self.subTest(mapping=value), self.assertRaises(EvidenceBindingError):
                EvidenceBindings(m)

    def test_output_cannot_write_into_mapped_evidence(self):
        manifest = self.root / "binding.json"
        manifest.write_text(json.dumps(self.manifest))
        output = self.new / "new-audit.json"
        with self.assertRaisesRegex(EvidenceBindingError, "outside mapped evidence"):
            main(["--manifest", str(manifest), "--output", str(output)])
        self.assertFalse(output.exists())

    def test_missing_declared_input_cannot_be_filled_by_audit_output(self):
        self.manifest["root_mappings"] = []
        bindings = EvidenceBindings(self.manifest)
        with self.assertRaisesRegex(EvidenceBindingError, "declared input"):
            bindings.validate_output(self.old / self.file.name)
        self.assertFalse(self.old.exists())

    def test_manifest_nonfinite_values_are_rejected(self):
        path = self.root / "binding.json"
        path.write_text('{"schema_version": NaN}')
        with self.assertRaisesRegex(EvidenceBindingError, "nonfinite"):
            EvidenceBindings.load(path)

    def test_cli_alias_dependency_is_reported_as_blocked_not_suppressed(self):
        manifest = self.root / "binding.json"; output = self.root / "audit.json"
        manifest.write_text(json.dumps(self.manifest))
        with patch.object(Path, "is_junction", lambda p: p == self.new, create=True):
            self.assertEqual(main(["--manifest", str(manifest), "--output", str(output)]), 1)
        self.assertIn("junction", json.loads(output.read_text("utf-8"))["files"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
