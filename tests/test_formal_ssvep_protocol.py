"""Formal SSVEP calibration protocol must collect enough repeated trials."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class FormalSSVEPProtocolTests(unittest.TestCase):
    def test_formal_protocol_has_calibration_repetitions_and_rest(self):
        path = ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(value["status"], "formal")
        self.assertGreaterEqual(value["trial"]["repetitions"], 10)
        self.assertGreaterEqual(value["trial"]["pre_session_rest_s"], 10)
        self.assertGreaterEqual(value["trial"]["pre_trial_rest_s"], 2)
        self.assertEqual([target["frequency_hz"] for target in value["targets"]], [10, 12, 15, 20])


if __name__ == "__main__":
    unittest.main()
