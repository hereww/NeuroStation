from __future__ import annotations

import unittest

from workstation import build_diagnostics


class WorkstationDiagnosticsTests(unittest.TestCase):
    def test_preflight_report_distinguishes_configuration_from_hardware(self) -> None:
        report = build_diagnostics()

        self.assertIn("platform", report)
        self.assertTrue(report["resources"]["protocol_ready"])
        self.assertTrue(report["resources"]["locales_ready"])
        self.assertIn(report["configuration"]["status"], {"ready", "draft", "invalid"})
        self.assertIsInstance(report["configuration"]["warnings"], list)
        self.assertFalse(report["hardware"]["cyton_connected"])
        self.assertIn("does not open the serial port", report["hardware"]["note"])

    def test_check_cyton_cli_is_explicit_when_port_is_unavailable(self) -> None:
        # This uses a deliberately invalid port so it never touches a real
        # device. The command must return JSON rather than a native traceback.
        import json
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "check_cyton_live.py", "--port", "NEUROSTATION-NOT-A-PORT", "--seconds", "0.01"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self.assertEqual(3, result.returncode)
        payload = json.loads(result.stdout)
        self.assertEqual("error", payload["status"])
        self.assertEqual("NEUROSTATION-NOT-A-PORT", payload["port"])
        self.assertTrue(payload["hint"])


if __name__ == "__main__":
    unittest.main()
