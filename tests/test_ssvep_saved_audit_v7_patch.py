"""Synthetic tests for the v7 patched JSON-only audit."""
import unittest

from eeg_tools import ssvep_saved_audit_v7_patch as audit


class PatchAuditTests(unittest.TestCase):
    def test_report_schema_constants_are_separate(self):
        self.assertIn("deviation", audit.PATCH_FIELDS)
        self.assertIn("patch", audit.PATCH_FIELDS)
        self.assertNotIn("deviation", audit.BASE_FIELDS)

    def test_source_scope_is_exactly_three_files(self):
        self.assertEqual(len(audit.AUDIT_SOURCE_PATHS), 3)
        self.assertTrue(all(p.startswith(("eeg_tools/", "scripts/", "tests/")) for p in audit.AUDIT_SOURCE_PATHS))

    def test_check_only_does_not_read_report(self):
        result = audit.audit_saved_report(
            report_relative="analysis_outputs/ssvep_v7_internal_diagnostic_patch_missing.json",
            output_prefix="analysis_outputs/ssvep_v7_saved_internal_audit_patch_test",
            check_only=True,
        )
        self.assertFalse(result["audit_performed"])
        self.assertEqual(result["report_read_deferred"], "analysis_outputs/ssvep_v7_internal_diagnostic_patch_missing.json")


if __name__ == "__main__":
    unittest.main()
