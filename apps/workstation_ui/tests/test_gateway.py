import unittest

from apps.workstation_ui.gateway import CaptureConfig, CaptureMode, format_duration


class GatewayContractTests(unittest.TestCase):
    def test_default_production_timing_and_counts(self):
        config = CaptureConfig()
        self.assertEqual(CaptureMode.CYTON, config.mode)
        self.assertEqual(config.trials, 12)
        self.assertEqual(config.recording_seconds, 93)
        self.assertEqual(config.total_seconds, 98)
        self.assertEqual(config.expected_events, 27)
        self.assertEqual(format_duration(98), "00:01:38")

    def test_real_hardware_config_requires_identity_and_risk_acknowledgement(self):
        with self.assertRaisesRegex(ValueError, "validation.identity"):
            CaptureConfig().validate()
        config = CaptureConfig(
            participant="U0001",
            name="Cyton session",
            user_id="U0001",
            user_name="Test user",
            acknowledge_flicker_risk=True,
        )
        config.validate()

    def test_legacy_modes_are_rejected(self):
        for mode in (CaptureMode.DEMO, CaptureMode.VISUAL_PREVIEW, CaptureMode.SYNTHETIC):
            with self.subTest(mode=mode):
                config = CaptureConfig(
                    participant="P001",
                    name="legacy",
                    mode=mode,
                    acknowledge_flicker_risk=True,
                )
                with self.assertRaisesRegex(ValueError, "validation.real_hardware_only"):
                    config.validate()

if __name__ == "__main__":
    unittest.main()
